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
    channels = ["acc_x", "acc_y", "acc_z", "acc_magnitude", "gyro_x", "gyro_y", "gyro_z", "gyro_magnitude"]
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
        "acc_correlation_xy",
        "acc_correlation_xz",
        "acc_correlation_yz",
        "gyro_correlation_xy",
        "gyro_correlation_xz",
        "gyro_correlation_yz",
    ]


def extract_features(axes: np.ndarray, sample_rate_hz: float = 25.0) -> np.ndarray:
    """Return interpretable time- and frequency-domain features for 6 channels (Acc + Gyro)."""
    acc_axes = axes[:, :3]
    gyro_axes = axes[:, 3:]
    
    acc_magnitude = np.linalg.norm(acc_axes, axis=1)
    gyro_magnitude = np.linalg.norm(gyro_axes, axis=1)
    
    channels = [
        acc_axes[:, 0], acc_axes[:, 1], acc_axes[:, 2], acc_magnitude,
        gyro_axes[:, 0], gyro_axes[:, 1], gyro_axes[:, 2], gyro_magnitude
    ]
    
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
        
    acc_correlations = np.corrcoef(acc_axes, rowvar=False)
    features.extend(float(np.nan_to_num(acc_correlations[i, j])) for i, j in [(0, 1), (0, 2), (1, 2)])
    
    gyro_correlations = np.corrcoef(gyro_axes, rowvar=False)
    features.extend(float(np.nan_to_num(gyro_correlations[i, j])) for i, j in [(0, 1), (0, 2), (1, 2)])
    
    return np.asarray(features, dtype=np.float64)
