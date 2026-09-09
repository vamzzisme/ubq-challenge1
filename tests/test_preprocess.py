#!/usr/bin/env python3
"""Regression tests for L1 clock-true resampling.

The central claim of the preprocessing layer is that a signal of a known
frequency keeps that frequency after resampling.  The previous implementation
resampled by sample index, which compressed a 23.2 s accelerometer capture into
a nominal 20 s window and shifted every frequency by ~16%.  test_frequency_is
_preserved_despite_uneven_clock is the test that fails under that approach and
passes under this one.
"""

from __future__ import annotations

import numpy as np

from asqa import config
from asqa.preprocess import WindowRejection, resample_to_grid


def _capture(freq_hz: float, n: int, span_s: float, jitter: float = 0.0, seed: int = 0) -> np.ndarray:
    """Build an (n, 4) capture of a sine at `freq_hz` spread over `span_s`."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, span_s, n)
    if jitter:
        # Perturb the clock the way a real phone does, keeping it monotonic.
        t = np.sort(t + rng.uniform(-jitter, jitter, n))
        t -= t[0]
    signal = np.sin(2 * np.pi * freq_hz * t)
    return np.column_stack([t, signal, signal * 0.5, signal * 0.25])


def _dominant_frequency(values: np.ndarray, rate_hz: float) -> float:
    spectrum = np.abs(np.fft.rfft(values - values.mean())) ** 2
    freqs = np.fft.rfftfreq(len(values), d=1.0 / rate_hz)
    return float(freqs[1 + int(np.argmax(spectrum[1:]))])


def test_grid_is_exactly_25hz() -> None:
    acc = _capture(2.0, 800, 23.18)
    gyro = _capture(2.0, 800, 19.99)
    result = resample_to_grid(acc, gyro)
    assert not isinstance(result, WindowRejection), result
    window, diagnostics = result

    assert window.shape == (config.WINDOW_SAMPLES, 6)
    assert window.dtype == np.float32
    # The window must span exactly (N-1)/25 s of real, interpolated time.
    assert abs(diagnostics["window_span_s"] - config.WINDOW_SPAN_S) < 1e-6
    # The grid period is exactly 1/25 s by construction.
    assert abs(config.TARGET_PERIOD_S - 0.04) < 1e-12


def test_frequency_is_preserved_despite_uneven_clock() -> None:
    """A 2 Hz sine on a 23.18 s uneven acc clock must read back as 2 Hz.

    Index-based resampling would report 2 * (23.18/20) ~= 2.32 Hz.
    """
    true_hz = 2.0
    acc = _capture(true_hz, 800, 23.18, jitter=0.012, seed=1)
    gyro = _capture(true_hz, 800, 19.99)

    result = resample_to_grid(acc, gyro)
    assert not isinstance(result, WindowRejection), result
    window, _ = result

    measured = _dominant_frequency(window[:, 0], config.TARGET_RATE_HZ)
    assert abs(measured - true_hz) < 0.1, f"expected ~{true_hz} Hz, measured {measured} Hz"

    # And the error the old approach would have made is large enough that this
    # test genuinely discriminates between the two implementations.
    index_based_error = true_hz * (23.18 / 20.0) - true_hz
    assert index_based_error > 0.3


def test_acc_and_gyro_land_on_one_time_base() -> None:
    """Both modalities must describe the same instants after resampling."""
    # Same 1.5 Hz content, but presented on the two different real clocks.
    acc = _capture(1.5, 800, 23.18, jitter=0.012, seed=2)
    gyro = _capture(1.5, 800, 19.99)

    result = resample_to_grid(acc, gyro)
    assert not isinstance(result, WindowRejection), result
    window, _ = result

    acc_hz = _dominant_frequency(window[:, 0], config.TARGET_RATE_HZ)
    gyro_hz = _dominant_frequency(window[:, 3], config.TARGET_RATE_HZ)
    assert abs(acc_hz - gyro_hz) < 0.1, f"acc {acc_hz} Hz vs gyro {gyro_hz} Hz"

    # Correlated inputs must stay correlated once they share a grid.
    correlation = np.corrcoef(window[:, 0], window[:, 3])[0, 1]
    assert correlation > 0.9, f"acc/gyro correlation collapsed to {correlation}"


def test_rejects_large_gaps() -> None:
    acc = _capture(2.0, 800, 23.18)
    acc[400:, 0] += 2.0  # a two-second hole
    gyro = _capture(2.0, 800, 19.99)
    result = resample_to_grid(acc, gyro)
    assert isinstance(result, WindowRejection)
    assert result.reason == "gap_too_large"


def test_rejects_insufficient_overlap() -> None:
    acc = _capture(2.0, 800, 23.18)
    gyro = _capture(2.0, 800, 19.99)
    gyro[:, 0] += 15.0  # push gyro almost entirely past the acc capture
    result = resample_to_grid(acc, gyro)
    assert isinstance(result, WindowRejection)
    assert result.reason == "insufficient_overlap"


def test_rejects_non_finite() -> None:
    acc = _capture(2.0, 800, 23.18)
    acc[10, 1] = np.nan
    gyro = _capture(2.0, 800, 19.99)
    result = resample_to_grid(acc, gyro)
    assert isinstance(result, WindowRejection)
    assert result.reason == "non_finite"


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
