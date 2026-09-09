#!/usr/bin/env python3
"""L1 -- preprocessing: clock-true resampling of raw ExtraSensory captures to 25 Hz.

Each ExtraSensory minute contributes one capture per modality, and each capture
holds 800 rows of ``uptime_seconds x y z``.  The two modalities do *not* share a
sampling regime:

    m_raw_acc    800 rows over ~23.2 s   (~34.5 Hz, dt 0.024-0.077 s, uneven)
    m_proc_gyro  800 rows over ~20.0 s   (40.0 Hz, uniform)

They do, however, share one device uptime clock, which is what makes honest
fusion possible.  This module therefore resamples *by timestamp*, not by sample
index: it builds a uniform 25 Hz grid across the interval both modalities
actually observed and interpolates all six channels onto that single grid.

Resampling by index instead -- the previous implementation's approach -- silently
compressed every accelerometer window by ~16% and left acc and gyro describing
different stretches of time.  That corrupts cadence and dominant-frequency
features, which are precisely the features that separate walking from running
from bicycling.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from asqa import config


# ── Loading raw captures ─────────────────────────────────────────────────────


def load_capture(path: Path) -> np.ndarray:
    """Load one ``.dat`` capture as an (N, 4) array of ``uptime, x, y, z``."""
    values = np.loadtxt(path, dtype=np.float64)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.ndim != 2 or values.shape[1] != 4:
        raise ValueError(f"{path}: expected 4 columns (uptime, x, y, z), got {values.shape}")
    return values


@dataclass
class WindowRejection:
    """Why one recording could not be reconstructed at 25 Hz."""

    reason: str
    detail: str = ""


def resample_to_grid(
    acc: np.ndarray,
    gyro: np.ndarray,
    n_samples: int = config.WINDOW_SAMPLES,
    rate_hz: float = config.TARGET_RATE_HZ,
    max_gap_s: float = config.MAX_SAMPLE_GAP_S,
    min_overlap_s: float = config.MIN_OVERLAP_S,
) -> tuple[np.ndarray, dict[str, float]] | WindowRejection:
    """Interpolate acc and gyro onto one shared uniform grid.

    Returns ``(windows[n_samples, 6], diagnostics)`` or a ``WindowRejection``.
    The grid spans the interval observed by *both* modalities, so the returned
    channels are sample-for-sample comparable across the two sensors.
    """
    if len(acc) < 2 or len(gyro) < 2:
        return WindowRejection("too_few_samples", f"acc={len(acc)} gyro={len(gyro)}")

    acc_t, gyro_t = acc[:, 0], gyro[:, 0]

    # The device clock must advance monotonically for interpolation to mean
    # anything.  Sort defensively rather than assuming file order.
    acc_order, gyro_order = np.argsort(acc_t), np.argsort(gyro_t)
    acc, gyro = acc[acc_order], gyro[gyro_order]
    acc_t, gyro_t = acc[:, 0], gyro[:, 0]

    if not (np.isfinite(acc).all() and np.isfinite(gyro).all()):
        return WindowRejection("non_finite")

    # Largest hole in either stream.  A window with a half-second hole cannot
    # honestly claim a 25 Hz reconstruction across that hole.
    acc_gap = float(np.max(np.diff(acc_t))) if len(acc_t) > 1 else np.inf
    gyro_gap = float(np.max(np.diff(gyro_t))) if len(gyro_t) > 1 else np.inf
    max_gap = max(acc_gap, gyro_gap)
    if max_gap > max_gap_s:
        return WindowRejection("gap_too_large", f"{max_gap:.3f}s")

    # The overlap is the only stretch where fusion is real rather than
    # extrapolated.
    start = max(acc_t[0], gyro_t[0])
    end = min(acc_t[-1], gyro_t[-1])
    overlap_s = end - start
    if overlap_s < min_overlap_s:
        return WindowRejection("insufficient_overlap", f"{overlap_s:.2f}s")

    # A fixed-length grid at exactly the target period, anchored at the start of
    # the overlap.  The window span is checked against the overlap above, so the
    # grid lies wholly inside the interval both sensors observed and every
    # returned sample is interpolated between two real measurements.  Clamping
    # the grid to the overlap instead would hold the final sample constant for
    # the remainder of the window, fabricating a motionless tail that biases the
    # window toward "static" and corrupts its spectrum.
    period = 1.0 / rate_hz
    grid = start + np.arange(n_samples, dtype=np.float64) * period
    if grid[-1] > end + 1e-9:
        return WindowRejection("insufficient_overlap", f"{overlap_s:.2f}s")

    channels = [np.interp(grid, acc_t, acc[:, axis]) for axis in (1, 2, 3)]
    channels += [np.interp(grid, gyro_t, gyro[:, axis]) for axis in (1, 2, 3)]
    window = np.column_stack(channels).astype(np.float32)

    diagnostics = {
        "overlap_s": overlap_s,
        "window_span_s": float(grid[-1] - grid[0]),
        "acc_span_s": float(acc_t[-1] - acc_t[0]),
        "gyro_span_s": float(gyro_t[-1] - gyro_t[0]),
        "acc_effective_hz": float(len(acc_t) / max(acc_t[-1] - acc_t[0], 1e-9)),
        "gyro_effective_hz": float(len(gyro_t) / max(gyro_t[-1] - gyro_t[0], 1e-9)),
        "max_gap_s": max_gap,
        "grid_start_uptime_s": float(start),
    }
    return window, diagnostics


# ── Labels ───────────────────────────────────────────────────────────────────


def load_labels(user_id: str) -> dict[int, str]:
    """Map ``epoch timestamp -> coarse activity`` for one user.

    ``standing`` is returned as the coarse ExtraSensory label; it is split into
    ``standing_in_place`` / ``standing_and_moving`` later, once the signal has
    been read (see :func:`split_standing`).
    """
    path = config.DATA_DIR / f"{user_id}{config.LABEL_SUFFIX}"
    labels: dict[int, str] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = set(config.LABEL_COLUMNS.values()) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} missing label columns: {sorted(missing)}")
        for row in reader:
            active = [name for name, column in config.LABEL_COLUMNS.items() if row[column] == "1"]
            # Verified across all 15 users: these six labels never co-occur.
            if len(active) == 1:
                labels[int(row["timestamp"])] = active[0]
    return labels


# ── Per-user preprocessing ───────────────────────────────────────────────────


@dataclass
class UserCache:
    user_id: str
    windows: np.ndarray  # (N, 500, 6) float32
    epoch_ts: np.ndarray  # (N,) int64 -- ExtraSensory minute identifier
    uptime_start: np.ndarray  # (N,) float64 -- device clock at grid sample 0
    labels: list[str]  # (N,) coarse activity
    rejections: Counter = field(default_factory=Counter)
    diagnostics: dict[str, float] = field(default_factory=dict)


def preprocess_user(user_id: str, limit: int | None = None, verbose: bool = True) -> UserCache:
    """Reconstruct every labelled window for one user at a true 25 Hz."""
    acc_dir = config.ACC_DIR / user_id
    gyro_dir = config.GYRO_DIR / user_id
    if not acc_dir.is_dir() or not gyro_dir.is_dir():
        raise FileNotFoundError(f"Missing raw directories for {user_id}")

    labels = load_labels(user_id)

    # File counts differ between modalities for several users, so intersect on
    # the timestamp rather than assuming a pairing.
    acc_ts = {int(p.name.split(".", 1)[0]) for p in acc_dir.glob(f"*{config.ACC_SUFFIX}")}
    gyro_ts = {int(p.name.split(".", 1)[0]) for p in gyro_dir.glob(f"*{config.GYRO_SUFFIX}")}
    usable = sorted(acc_ts & gyro_ts & labels.keys())

    rejections: Counter = Counter()
    rejections["no_gyro_pair"] = len(acc_ts & labels.keys()) - len(usable)
    rejections["unlabelled"] = len(acc_ts & gyro_ts) - len(usable)

    if limit is not None:
        usable = usable[:limit]

    windows: list[np.ndarray] = []
    kept_ts: list[int] = []
    kept_uptime: list[float] = []
    kept_labels: list[str] = []
    diag_accumulator: Counter = Counter()

    for index, timestamp in enumerate(usable, start=1):
        acc_path = acc_dir / f"{timestamp}{config.ACC_SUFFIX}"
        gyro_path = gyro_dir / f"{timestamp}{config.GYRO_SUFFIX}"
        try:
            acc = load_capture(acc_path)
            gyro = load_capture(gyro_path)
        except (ValueError, OSError) as exc:
            rejections[f"unreadable"] += 1
            if verbose and rejections["unreadable"] <= 3:
                print(f"  unreadable {timestamp}: {exc}", file=sys.stderr)
            continue

        result = resample_to_grid(acc, gyro)
        if isinstance(result, WindowRejection):
            rejections[result.reason] += 1
            continue

        window, diagnostics = result
        windows.append(window)
        kept_ts.append(timestamp)
        kept_uptime.append(diagnostics["grid_start_uptime_s"])
        kept_labels.append(labels[timestamp])
        for key in ("overlap_s", "acc_span_s", "gyro_span_s", "acc_effective_hz", "gyro_effective_hz"):
            diag_accumulator[key] += diagnostics[key]

        if verbose and (index % 2000 == 0 or index == len(usable)):
            print(f"  {user_id[:8]} {index}/{len(usable)} windows", flush=True)

    if not windows:
        raise ValueError(f"No usable windows for {user_id}")

    kept = len(windows)
    diagnostics = {key: value / kept for key, value in diag_accumulator.items()}
    diagnostics["kept"] = kept
    diagnostics["candidates"] = len(usable)

    return UserCache(
        user_id=user_id,
        windows=np.stack(windows),
        epoch_ts=np.asarray(kept_ts, dtype=np.int64),
        uptime_start=np.asarray(kept_uptime, dtype=np.float64),
        labels=kept_labels,
        rejections=rejections,
        diagnostics=diagnostics,
    )


def cache_path(user_id: str) -> Path:
    return config.CACHE_DIR / f"{user_id}.npz"


def save_cache(cache: UserCache) -> Path:
    path = cache_path(cache.user_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        windows=cache.windows,
        epoch_ts=cache.epoch_ts,
        uptime_start=cache.uptime_start,
        labels=np.asarray(cache.labels),
        rejections=json.dumps(dict(cache.rejections)),
        diagnostics=json.dumps(cache.diagnostics),
    )
    return path


def load_cached(user_id: str) -> UserCache:
    with np.load(cache_path(user_id), allow_pickle=False) as data:
        return UserCache(
            user_id=user_id,
            windows=data["windows"],
            epoch_ts=data["epoch_ts"],
            uptime_start=data["uptime_start"],
            labels=[str(x) for x in data["labels"]],
            rejections=Counter(json.loads(str(data["rejections"]))),
            diagnostics=json.loads(str(data["diagnostics"])),
        )


def available_users() -> list[str]:
    """Users with raw acc, raw gyro, and a label file all present."""
    acc = {p.name for p in config.ACC_DIR.iterdir() if p.is_dir()}
    gyro = {p.name for p in config.GYRO_DIR.iterdir() if p.is_dir()}
    labelled = {p.name.replace(config.LABEL_SUFFIX, "") for p in config.DATA_DIR.glob(f"*{config.LABEL_SUFFIX}")}
    return sorted(acc & gyro & labelled)


def cached_users() -> list[str]:
    return sorted(p.stem for p in config.CACHE_DIR.glob("*.npz"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user", action="append", help="Preprocess only this user (repeatable).")
    parser.add_argument("--limit", type=int, help="Cap windows per user (quick validation runs).")
    parser.add_argument("--overwrite", action="store_true", help="Rebuild caches that already exist.")
    args = parser.parse_args()

    users = args.user or available_users()
    print(f"Preprocessing {len(users)} user(s) to a true {config.TARGET_RATE_HZ:g} Hz grid\n")

    summary: dict[str, dict] = {}
    for user_id in users:
        if cache_path(user_id).exists() and not args.overwrite:
            print(f"{user_id[:8]}  cached, skipping (use --overwrite to rebuild)")
            continue
        print(f"{user_id[:8]}  reconstructing...")
        cache = preprocess_user(user_id, limit=args.limit)
        path = save_cache(cache)
        diagnostics = cache.diagnostics
        dropped = sum(v for k, v in cache.rejections.items() if k not in {"no_gyro_pair", "unlabelled"})
        print(
            f"{user_id[:8]}  kept {diagnostics['kept']}/{diagnostics['candidates']} "
            f"(dropped {dropped}) | acc {diagnostics['acc_effective_hz']:.1f} Hz over "
            f"{diagnostics['acc_span_s']:.1f}s, gyro {diagnostics['gyro_effective_hz']:.1f} Hz over "
            f"{diagnostics['gyro_span_s']:.1f}s -> overlap {diagnostics['overlap_s']:.1f}s "
            f"| {path.stat().st_size / 1e6:.0f} MB"
        )
        if dropped:
            print(f"          rejections: {dict(cache.rejections)}")
        summary[user_id] = {"diagnostics": diagnostics, "rejections": dict(cache.rejections)}

    if summary:
        report = config.OUTPUT_DIR / "preprocess_report.json"
        report.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"\nWrote {report}")
    return 0


if __name__ == "__main__":
    # Import through the package rather than calling the local `main`, so any
    # object pickled here records its class as `asqa.preprocess.X` and not
    # `__main__.X` -- the latter cannot be unpickled by any other entry point.
    from asqa.preprocess import main as _main

    raise SystemExit(_main())
