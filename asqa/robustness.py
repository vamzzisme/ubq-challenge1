#!/usr/bin/env python3
"""Robustness: accuracy as the input signal is deliberately degraded.

The brief asks for a curve rather than a single clean number, on the argument
that *"a curve that stays flat as conditions worsen is stronger evidence of a
usable system than any single clean number"*.

Three degradations are applied to the raw 25 Hz windows, before feature
extraction, so the whole pipeline experiences them the way it would in the field:

    noise     Gaussian noise on the accelerometer, scaled by measured gravity so
              the level means the same thing for the users reporting in g and
              the users reporting in m/s^2
    dropout   a fraction of samples lost and linearly bridged, as happens when a
              phone throttles its sensors
    rate      the stream decimated below 25 Hz and interpolated back, which is
              the brief's "sampling rate below 25 Hz" case
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from asqa import config, context, features as feat
from asqa.decode import decode_labels

NOISE_LEVELS = (0.0, 0.01, 0.025, 0.05, 0.1, 0.2)
DROPOUT_RATES = (0.0, 0.05, 0.10, 0.20, 0.35, 0.50)
SAMPLE_RATES = (25.0, 20.0, 15.0, 12.5, 10.0, 5.0)


def add_noise(windows: np.ndarray, sigma_g: float, rng: np.random.Generator) -> np.ndarray:
    """Gaussian noise on the accelerometer, in units of g."""
    if sigma_g <= 0:
        return windows
    out = windows.copy()
    acc = out[:, :, config.ACC_SLICE]
    # Scale by each window's gravity magnitude so a level means the same thing
    # regardless of the unit the device reported in.
    scale = np.linalg.norm(acc.mean(axis=1), axis=1, keepdims=True)[:, None, :]
    scale = np.where(scale > 1e-6, scale, 1.0)
    out[:, :, config.ACC_SLICE] = acc + rng.normal(0.0, sigma_g, acc.shape) * scale
    return out


def drop_samples(windows: np.ndarray, rate: float, rng: np.random.Generator) -> np.ndarray:
    """Lose a fraction of samples and bridge the holes by interpolation."""
    if rate <= 0:
        return windows
    out = windows.copy()
    n_samples = windows.shape[1]
    grid = np.arange(n_samples)
    for index in range(len(out)):
        keep = rng.random(n_samples) > rate
        keep[0] = keep[-1] = True  # keep the ends so interpolation is bounded
        for channel in range(out.shape[2]):
            out[index, :, channel] = np.interp(grid, grid[keep], out[index, keep, channel])
    return out


def resample_lower(windows: np.ndarray, rate_hz: float) -> np.ndarray:
    """Decimate to `rate_hz` and interpolate back onto the 25 Hz grid."""
    if rate_hz >= config.TARGET_RATE_HZ:
        return windows
    n_samples = windows.shape[1]
    duration = n_samples / config.TARGET_RATE_HZ
    coarse_n = max(2, int(round(duration * rate_hz)))
    coarse_positions = np.linspace(0, n_samples - 1, coarse_n)
    grid = np.arange(n_samples)
    out = np.empty_like(windows)
    for index in range(len(windows)):
        for channel in range(windows.shape[2]):
            coarse = np.interp(coarse_positions, grid, windows[index, :, channel])
            out[index, :, channel] = np.interp(grid, coarse_positions, coarse)
    return out


def score(pipeline, windows: np.ndarray, epoch_ts: np.ndarray, truth: np.ndarray) -> float:
    X = feat.extract_batch(windows)
    Xc = context.augment(X, epoch_ts)
    probabilities = pipeline.context_aware.predict_proba(Xc)
    predicted = decode_labels(probabilities, epoch_ts, pipeline.transitions(), "viterbi")
    return float((predicted == truth).mean())


def main() -> int:
    from asqa.pipeline import Pipeline
    from asqa.preprocess import load_cached
    from asqa.recognise import build_context_features
    from asqa.splits import load_folds

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, default=3)
    parser.add_argument("--max-windows", type=int, default=3000, help="Cap per user, for runtime.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    folds = load_folds()
    pipeline = Pipeline(fold=args.fold)
    rng = np.random.default_rng(args.seed)

    users = folds["splits"][str(args.fold)]["test"][:2]
    windows_all, ts_all, truth_all = [], [], []
    for user_id in users:
        cache = load_cached(user_id)
        Xc, coarse, epoch_ts = build_context_features(user_id)
        truth, _ = pipeline.context_aware.standing_split.apply(Xc, coarse)
        limit = min(args.max_windows, len(cache.windows))
        windows_all.append(cache.windows[:limit])
        ts_all.append(epoch_ts[:limit])
        truth_all.append(truth[:limit])

    windows = np.concatenate(windows_all)
    epoch_ts = np.concatenate(ts_all)
    truth = np.concatenate(truth_all)
    print(f"Robustness on {len(windows)} windows from {len(users)} held-out users\n")

    results: dict[str, list[dict]] = {}

    print(f"{'noise sigma (g)':<18}{'accuracy':>10}")
    results["noise"] = []
    for sigma in NOISE_LEVELS:
        accuracy = score(pipeline, add_noise(windows, sigma, rng), epoch_ts, truth)
        results["noise"].append({"level": sigma, "accuracy": accuracy})
        print(f"{sigma:<18.3f}{accuracy:>10.3f}")

    print(f"\n{'dropout rate':<18}{'accuracy':>10}")
    results["dropout"] = []
    for rate in DROPOUT_RATES:
        accuracy = score(pipeline, drop_samples(windows, rate, rng), epoch_ts, truth)
        results["dropout"].append({"level": rate, "accuracy": accuracy})
        print(f"{rate:<18.2f}{accuracy:>10.3f}")

    print(f"\n{'sample rate (Hz)':<18}{'accuracy':>10}")
    results["sample_rate"] = []
    for rate_hz in SAMPLE_RATES:
        accuracy = score(pipeline, resample_lower(windows, rate_hz), epoch_ts, truth)
        results["sample_rate"].append({"level": rate_hz, "accuracy": accuracy})
        print(f"{rate_hz:<18.1f}{accuracy:>10.3f}")

    payload = {"fold": args.fold, "n_windows": int(len(windows)), "users": users, "curves": results}
    config.EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    output = config.EVALUATION_DIR / "robustness.json"
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {output}")
    return 0


if __name__ == "__main__":
    from asqa.robustness import main as _main

    raise SystemExit(_main())
