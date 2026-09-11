"""Guards for temporal-context features."""

from __future__ import annotations

import numpy as np

from asqa import context, features as feat

F = len(feat.FEATURE_NAMES)
RMS = feat.FEATURE_NAMES.index("body_acc_rms")


def _matrix(values: list[float]) -> np.ndarray:
    """Feature matrix whose body_acc_rms column carries `values`."""
    X = np.zeros((len(values), F))
    X[:, RMS] = values
    return X


def _minutes(count: int, start: int = 1_400_000_000) -> np.ndarray:
    return np.array([start + 60 * i for i in range(count)], dtype=np.int64)


def test_column_count_matches_declared_names() -> None:
    X = _matrix([0.1] * 5)
    out = context.augment(X, _minutes(5))
    assert out.shape == (5, F + context.N_CONTEXT_FEATURES)
    assert len(context.feature_names()) == context.N_CONTEXT_FEATURES
    assert len(context.FEATURE_NAMES_FULL) == out.shape[1]


def test_original_features_are_preserved() -> None:
    values = [0.1, 0.5, 0.9, 0.2, 0.7]
    X = _matrix(values)
    out = context.augment(X, _minutes(5))
    assert np.allclose(out[:, :F], X), "augment must not disturb the per-window features"


def test_row_order_is_restored() -> None:
    """Rows may arrive unsorted; results must come back in the caller's order."""
    values = [0.1, 0.2, 0.3, 0.4]
    times = _minutes(4)
    shuffle = np.array([2, 0, 3, 1])

    straight = context.augment(_matrix(values), times)
    shuffled = context.augment(_matrix([values[i] for i in shuffle]), times[shuffle])
    assert np.allclose(shuffled, straight[shuffle]), "augment mis-restores the input order"


def test_no_context_across_a_dropout() -> None:
    """A window beside a six-hour gap must draw no context across it."""
    times = np.array([0, 60, 120, 120 + 6 * 3600, 120 + 6 * 3600 + 60], dtype=np.int64)
    X = _matrix([0.0, 0.0, 0.0, 1.0, 1.0])
    out = context.augment(X, times)

    mean_col = F + context.feature_names().index("ctx_body_acc_rms_mean_2")
    assert np.allclose(out[:3, mean_col], 0.0), f"context leaked across the gap: {out[:3, mean_col]}"
    assert np.allclose(out[3:, mean_col], 1.0), f"context leaked across the gap: {out[3:, mean_col]}"

    prev_col = F + context.feature_names().index("ctx_body_acc_rms_prev")
    assert out[3, prev_col] == 1.0, "a window across a dropout was treated as an adjacent neighbour"


def test_recordings_do_not_bleed_into_each_other() -> None:
    """Augmenting per recording must be unaffected by other recordings."""
    times = _minutes(6)
    alone = context.augment(_matrix([0.0] * 6), times)

    other = context.augment(_matrix([5.0] * 6), times)
    again = context.augment(_matrix([0.0] * 6), times)

    assert np.allclose(alone, again), "augment is not deterministic per recording"
    assert not np.allclose(alone, other), "different recordings produced identical context"


def test_isolated_windows_collapse_to_themselves() -> None:
    """The Task 1 case: one window, no neighbours, no invented context."""
    X = _matrix([0.4, 0.8])
    out = context.augment_isolated(X)
    assert out.shape == (2, F + context.N_CONTEXT_FEATURES)

    names = context.feature_names()
    mean_col = F + names.index("ctx_body_acc_rms_mean_5")
    std_col = F + names.index("ctx_body_acc_rms_std_5")
    prev_col = F + names.index("ctx_body_acc_rms_prev")
    delta_col = F + names.index("ctx_body_acc_rms_delta_prev")

    assert np.allclose(out[:, mean_col], [0.4, 0.8]), "an isolated window's mean must be itself"
    assert np.allclose(out[:, std_col], 0.0), "an isolated window has no spread"
    assert np.allclose(out[:, prev_col], [0.4, 0.8]), "an isolated window is its own neighbour"
    assert np.allclose(out[:, delta_col], 0.0), "an isolated window has no neighbour delta"


def test_context_is_finite_everywhere() -> None:
    rng = np.random.default_rng(0)
    X = np.zeros((40, F))
    X[:, RMS] = rng.random(40)
    times = np.cumsum(rng.integers(50, 90, 40)).astype(np.int64)
    out = context.augment(X, times)
    assert np.isfinite(out).all(), "context produced non-finite values"


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS  {name}")
            except AssertionError as exc:
                failures += 1
                print(f"FAIL  {name}: {exc}")
    raise SystemExit(1 if failures else 0)
