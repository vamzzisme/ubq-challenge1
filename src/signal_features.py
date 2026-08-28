"""Small, interpretable feature set for one tri-axial sensor recording."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


def load_axes(csv_path: str | Path) -> np.ndarray:
    """Load X/Y/Z from a CSV produced by prepare_sensor_data.py."""
    with Path(csv_path).open(newline="", encoding="utf-8") as handle:
        rows = csv.DictReader(handle)
        axes = [[float(row["x"]), float(row["y"]), float(row["z"])] for row in rows]
    values = np.asarray(axes, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or len(values) < 2:
        raise ValueError(f"{csv_path}: expected at least two X/Y/Z rows")
    return values


def _dominant_frequency(values: np.ndarray, sample_rate_hz: float) -> float:
    spectrum = np.abs(np.fft.rfft(values - values.mean())) ** 2
    if len(spectrum) <= 1 or spectrum[1:].sum() == 0:
        return 0.0
    frequencies = np.fft.rfftfreq(len(values), d=1 / sample_rate_hz)
    return float(frequencies[1 + np.argmax(spectrum[1:])])


def _spectral_entropy(values: np.ndarray) -> float:
    power = np.abs(np.fft.rfft(values - values.mean())) ** 2
    power = power[1:]
    total = power.sum()
    if total <= 0:
        return 0.0
    probabilities = power / total
    return float(-np.sum(probabilities * np.log2(probabilities + 1e-12)) / np.log2(len(probabilities)))


def feature_names() -> list[str]:
    channels = ["x", "y", "z", "magnitude"]
    statistics = [
        "mean",
        "std",
        "min",
        "max",
        "median",
        "iqr",
        "rms",
        "mean_abs_diff",
        "dominant_frequency_hz",
        "spectral_entropy",
    ]
    return [f"{channel}_{statistic}" for channel in channels for statistic in statistics] + [
        "correlation_xy",
        "correlation_xz",
        "correlation_yz",
    ]


def extract_features(axes: np.ndarray, sample_rate_hz: float = 25.0) -> np.ndarray:
    """Return interpretable time- and frequency-domain features for X/Y/Z."""
    magnitude = np.linalg.norm(axes, axis=1)
    channels = [axes[:, 0], axes[:, 1], axes[:, 2], magnitude]
    features: list[float] = []
    for values in channels:
        features.extend(
            [
                float(values.mean()),
                float(values.std()),
                float(values.min()),
                float(values.max()),
                float(np.median(values)),
                float(np.percentile(values, 75) - np.percentile(values, 25)),
                float(np.sqrt(np.mean(values**2))),
                float(np.mean(np.abs(np.diff(values)))),
                _dominant_frequency(values, sample_rate_hz),
                _spectral_entropy(values),
            ]
        )
    correlations = np.corrcoef(axes, rowvar=False)
    features.extend(float(np.nan_to_num(correlations[i, j])) for i, j in [(0, 1), (0, 2), (1, 2)])
    return np.asarray(features, dtype=np.float64)
