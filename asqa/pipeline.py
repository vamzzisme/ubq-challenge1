"""End-to-end pipeline: a recording in, a grounded timeline out."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np

from asqa import config, context, features as feat, timeline as tl
from asqa.decode import TimeAwareTransitions, decode_labels, estimate_transitions
from asqa.preprocess import cache_path, load_capture, load_cached, resample_to_grid, WindowRejection

MIN_WINDOWS_FOR_CONTEXT = 5

TIMESTAMP_RE = re.compile(r"^(?P<ts>\d+)\.m_(?P<modality>[a-z_]+)\.dat$")


@dataclass
class LoadedRecording:
    windows: np.ndarray
    epoch_ts: np.ndarray
    rejected: int = 0
    source: str = ""


def _pair_raw_directory(acc_dir: Path, gyro_dir: Path) -> LoadedRecording:
    acc = {int(m.group("ts")): p for p in acc_dir.glob("*.dat") if (m := TIMESTAMP_RE.match(p.name))}
    gyro = {int(m.group("ts")): p for p in gyro_dir.glob("*.dat") if (m := TIMESTAMP_RE.match(p.name))}
    shared = sorted(set(acc) & set(gyro))

    kept: list[np.ndarray] = []
    stamps: list[int] = []
    rejected = 0
    for timestamp in shared:
        result = resample_to_grid(load_capture(acc[timestamp]), load_capture(gyro[timestamp]))
        if isinstance(result, WindowRejection):
            rejected += 1
            continue
        window, _ = result
        kept.append(window)
        stamps.append(timestamp)

    if not kept:
        raise ValueError(f"No usable windows in {acc_dir}")
    return LoadedRecording(np.stack(kept), np.asarray(stamps, dtype=np.int64), rejected, str(acc_dir))


def load_recording(recording: str | Path) -> LoadedRecording:
    """Load a recording from a cached user id, a raw directory, or a raw file."""
    target = Path(recording)
    name = str(recording)

    if "/" not in name and not target.suffix and cache_path(name).exists():
        data = load_cached(name)
        return LoadedRecording(data.windows, data.epoch_ts, source=name)

    if not target.exists():
        cached = cache_path(name)
        if cached.exists():
            data = load_cached(name)
            return LoadedRecording(data.windows, data.epoch_ts, source=name)
        raise FileNotFoundError(f"No recording found for {recording!r}")

    if target.suffix == ".npz":
        data = load_cached(target.stem)
        return LoadedRecording(data.windows, data.epoch_ts, source=target.stem)

    if target.is_dir():
        gyro_dir = Path(str(target).replace(f"{config.ACC_DIR.name}/", f"{config.GYRO_DIR.name}/"))
        if not gyro_dir.is_dir():
            candidate = config.GYRO_DIR / target.name
            gyro_dir = candidate if candidate.is_dir() else target
        return _pair_raw_directory(target, gyro_dir)

    match = TIMESTAMP_RE.match(target.name)
    if match is None:
        raise ValueError(f"Cannot read a recording timestamp from {target.name}")
    timestamp = int(match.group("ts"))
    gyro_path = Path(str(target).replace("raw_acc", "proc_gyro").replace("/acc/", "/gyro/"))
    if not gyro_path.exists():
        raise FileNotFoundError(f"No gyroscope partner for {target} (looked for {gyro_path})")

    result = resample_to_grid(load_capture(target), load_capture(gyro_path))
    if isinstance(result, WindowRejection):
        raise ValueError(f"{target.name} cannot be reconstructed at 25 Hz: {result.reason} {result.detail}")
    window, _ = result
    return LoadedRecording(window[None, ...], np.asarray([timestamp], dtype=np.int64), source=str(target))


class Pipeline:
    """Recording -> features -> activities -> decoded path -> timeline."""

    def __init__(self, fold: int = 0, model_dir: Path | None = None):
        directory = model_dir or config.MODEL_DIR
        path = directory / f"recogniser_fold{fold}.joblib"
        if not path.exists():
            raise FileNotFoundError(
                f"No trained model at {path}. Run `python -m asqa.recognise --cv` first."
            )
        saved = joblib.load(path)
        self.context_free = saved["context_free"]
        self.context_aware = saved["context_aware"]
        self.fold = fold
        self._transitions: TimeAwareTransitions | None = None

    def transitions(self) -> TimeAwareTransitions:
        if self._transitions is None:
            path = config.MODEL_DIR / f"transitions_fold{self.fold}.npy"
            if path.exists():
                self._transitions = TimeAwareTransitions(np.load(path))
            else:
                from asqa.recognise import build_context_features
                from asqa.splits import load_folds

                split = load_folds()["splits"][str(self.fold)]
                sequences = []
                for user_id in split["train"]:
                    X, coarse, epoch_ts = build_context_features(user_id)
                    labels, _ = self.context_aware.standing_split.apply(X, coarse)
                    sequences.append((epoch_ts, labels))
                matrix = estimate_transitions(sequences)
                path.parent.mkdir(parents=True, exist_ok=True)
                np.save(path, matrix)
                self._transitions = TimeAwareTransitions(matrix)
        return self._transitions

    def run(self, recording: str | Path, decode: bool = True) -> tl.Timeline:
        loaded = load_recording(recording)
        return self.run_loaded(loaded, decode=decode)

    def run_loaded(self, loaded: LoadedRecording, decode: bool = True) -> tl.Timeline:
        base = feat.extract_batch(loaded.windows)
        use_context = len(base) >= MIN_WINDOWS_FOR_CONTEXT

        if use_context:
            X = context.augment(base, loaded.epoch_ts)
            model = self.context_aware
        else:
            X = base
            model = self.context_free

        probabilities = model.predict_proba(X)
        if decode and len(X) > 1:
            activities = decode_labels(probabilities, loaded.epoch_ts, self.transitions(), "viterbi")
        else:
            activities = np.asarray(config.ACTIVITIES)[np.argmax(probabilities, axis=1)]

        confidences = probabilities[np.arange(len(probabilities)), np.argmax(probabilities, axis=1)]
        return tl.build_timeline(
            activities=activities,
            epoch_ts=loaded.epoch_ts,
            X=base,
            confidences=confidences,
            used_context_model=use_context,
        )
