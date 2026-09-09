#!/usr/bin/env python3
"""L2b -- temporal context features over a recording's ordered windows.

`features.py` describes one window in isolation.  That is not enough to tell
lying down from sitting, and the gap is not marginal: pooled over five folds,
`lying_down -> sitting` alone accounted for 9,167 errors, 35% of every error the
system made, and per-window `lying_down` recall was 0.104.

The reason is physical rather than statistical.  A single 15-second window of
stillness looks the same whether the person is lying, sitting, or has left the
phone on a table -- the accelerometer sees one gravity direction and no motion in
all three cases.  What separates them is what surrounds the window: lying down
comes in long, uninterrupted stretches of a stable orientation, while sitting is
punctuated by shifts, reaches, and short walks.  These features expose that
surrounding structure to the classifier.

Measured on the 15-user corpus, adding them lifted 5-fold accuracy from 0.578 to
0.668 and `lying_down` recall from 0.104 to 0.842, improving every fold.

Two properties matter for correctness:

* **No cross-recording bleed.**  Context is computed per recording.  Windows
  belonging to different people, or to different recordings, never contribute to
  one another.
* **Real time, not index distance.**  ExtraSensory drops out for minutes to
  hours.  Averaging over "the previous 30 windows" can silently reach across a
  six-hour gap and call it context.  Every neighbourhood here is bounded by
  elapsed seconds as well as by count.
"""

from __future__ import annotations

import numpy as np

from asqa import config, features as feat

# Neighbourhood radii, in windows.  ExtraSensory samples about once a minute, so
# these read roughly as +/- 2, 5, 15 and 30 minutes.
RADII: tuple[int, ...] = (2, 5, 15, 30)

# A neighbour may only contribute if it lies within this many seconds.  Set to
# comfortably exceed the widest radius at the nominal one-minute cadence
# (30 windows ~ 1800 s) while still refusing to bridge a real dropout.
MAX_CONTEXT_S = 2400.0

# Per-window channels that context is computed over.  Energy and tilt are the
# pair validated in the experiment; cadence, rotation and spectral shape are
# included so Phase C can measure whether the extra channels earn their place.
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
    """Time-bounded rolling mean and std over +/- `radius` windows.

    A neighbour counts only if it is both within `radius` positions and within
    ``MAX_CONTEXT_S`` seconds, so a dropout truncates the neighbourhood instead of
    being averaged across.
    """
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
    """Append temporal-context columns to one recording's feature matrix.

    ``X`` is (N, F) from `features.extract_batch` and ``epoch_ts`` the matching
    window timestamps.  Rows may arrive in any order; the result is returned in
    the caller's original order.

    Call this once per recording.  Passing two users' rows in one array would let
    one person's stillness describe another's, so callers concatenate *after*
    augmenting, never before.
    """
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

        # Immediate neighbours.  At a boundary a window is its own neighbour, so
        # the delta is 0 rather than an invented jump.
        previous = np.r_[values[:1], values[:-1]]
        following = np.r_[values[1:], values[-1:]]
        # A neighbour separated by a dropout is not a neighbour.
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
    """Context columns for windows with no neighbours (the Task 1 case).

    Every rolling statistic collapses to the window's own value, every spread to
    zero, and every neighbour to itself.  This keeps the column count consistent
    so a context-aware model can be *evaluated* on isolated windows -- though it
    should not be *used* that way: measured, the context model scores 0.431 on
    isolated windows against the context-free model's 0.444, which is why
    `recognise.py` trains and routes to both.
    """
    return np.vstack([augment(X[i : i + 1], np.zeros(1)) for i in range(len(X))])


FEATURE_NAMES_FULL = tuple(feat.FEATURE_NAMES) + tuple(feature_names())
