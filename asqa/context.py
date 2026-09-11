"""L2b temporal context features over a recording's ordered windows."""

from __future__ import annotations

import numpy as np

from asqa import config, features as feat

RADII: tuple[int, ...] = (2, 5, 15, 30)

MAX_CONTEXT_S = 2400.0

CONTEXT_CHANNELS: tuple[str, ...] = (
    "body_acc_rms",
    "gravity_tilt_deg",
    "cadence_hz",
    "gyro_rms",
    "spectral_entropy",
)


def feature_names() -> list[str]:
    """Names of the context columns, in the order `augment` emits them."""
    names: list[str] = []
    for channel in CONTEXT_CHANNELS:
        for radius in RADII:
            names.append(f"ctx_{channel}_mean_{radius}")
            names.append(f"ctx_{channel}_std_{radius}")
        names.extend(
            [
                f"ctx_{channel}_prev",
                f"ctx_{channel}_next",
                f"ctx_{channel}_delta_prev",
                f"ctx_{channel}_delta_next",
            ]
        )
    names.extend(["ctx_gap_prev_s", "ctx_gap_next_s", "ctx_neighbours_within_30min"])
    return names


N_CONTEXT_FEATURES = len(feature_names())


def _rolling(values: np.ndarray, times: np.ndarray, radius: int) -> tuple[np.ndarray, np.ndarray]:
    """Time-bounded rolling mean and std over +/- `radius` windows."""
    n = len(values)
    means = np.empty(n)
    stds = np.empty(n)
    for index in range(n):
        low = max(0, index - radius)
        high = min(n, index + radius + 1)
        window_times = times[low:high]
        near = np.abs(window_times - times[index]) <= MAX_CONTEXT_S
        selected = values[low:high][near]
        if len(selected) == 0:
            means[index] = values[index]
            stds[index] = 0.0
        else:
            means[index] = selected.mean()
            stds[index] = selected.std()
    return means, stds


def augment(X: np.ndarray, epoch_ts: np.ndarray) -> np.ndarray:
    """Append temporal-context columns to one recording's feature matrix."""
    if len(X) != len(epoch_ts):
        raise ValueError(f"X has {len(X)} rows but epoch_ts has {len(epoch_ts)}")
    if len(X) == 0:
        return X.reshape(0, X.shape[1] + N_CONTEXT_FEATURES)

    order = np.argsort(epoch_ts)
    ordered = X[order]
    times = np.asarray(epoch_ts, dtype=np.float64)[order]
    n = len(ordered)

    columns: list[np.ndarray] = []
    for channel in CONTEXT_CHANNELS:
        values = ordered[:, feat.FEATURE_NAMES.index(channel)]
        for radius in RADII:
            means, stds = _rolling(values, times, radius)
            columns.append(means)
            columns.append(stds)

        previous = np.r_[values[:1], values[:-1]]
        following = np.r_[values[1:], values[-1:]]
        gap_before = np.r_[[0.0], np.diff(times)]
        gap_after = np.r_[np.diff(times), [0.0]]
        previous = np.where(gap_before <= MAX_CONTEXT_S, previous, values)
        following = np.where(gap_after <= MAX_CONTEXT_S, following, values)
        columns.extend([previous, following, np.abs(values - previous), np.abs(values - following)])

    gap_before = np.r_[[0.0], np.diff(times)]
    gap_after = np.r_[np.diff(times), [0.0]]
    within = np.array(
        [int(np.sum(np.abs(times - times[i]) <= 1800.0) - 1) for i in range(n)], dtype=np.float64
    )
    columns.extend([gap_before, gap_after, within])

    augmented = np.column_stack([ordered] + columns)
    result = np.empty_like(augmented)
    result[order] = augmented
    return result


def augment_isolated(X: np.ndarray) -> np.ndarray:
    """Context columns for windows with no neighbours (the Task 1 case)."""
    return np.vstack([augment(X[i : i + 1], np.zeros(1)) for i in range(len(X))])


FEATURE_NAMES_FULL = tuple(feat.FEATURE_NAMES) + tuple(feature_names())
