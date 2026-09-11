"""L2a interpretable features built around body mechanics, not generic statistics."""

from __future__ import annotations

import numpy as np
from scipy import signal as scipy_signal # for filtering

from asqa import config


# gravity is the part of acceleration that barely changes: a 0.3 Hz lowpass
# separates posture (which way is down) from motion (what the body is doing).
_GRAVITY_CUTOFF_HZ = 0.3
_GRAVITY_SOS = scipy_signal.butter(
    4, _GRAVITY_CUTOFF_HZ, btype="lowpass", fs=config.TARGET_RATE_HZ, output="sos"
)


def separate_gravity(acc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split accelerometer samples into (gravity, body) components."""
    # sosfiltfilt needs more samples than the filter's padding length, so very
    # short inputs fall back to the mean, which is gravity for a still window.
    if len(acc) <= 12:
        gravity = np.repeat(acc.mean(axis=0, keepdims=True), len(acc), axis=0)
    else:
        gravity = scipy_signal.sosfiltfilt(_GRAVITY_SOS, acc, axis=0)
    return gravity, acc - gravity


def _dominant_frequency(values: np.ndarray, rate_hz: float = config.TARGET_RATE_HZ) -> float:
    """Frequency of the strongest non-DC spectral component."""
    # Bin 0 is the DC term, already removed by centring; skipping it stops a
    # constant offset from being reported as the dominant frequency.
    spectrum = np.abs(np.fft.rfft(values - values.mean())) ** 2
    if len(spectrum) <= 1 or spectrum[1:].sum() == 0:
        return 0.0
    freqs = np.fft.rfftfreq(len(values), d=1.0 / rate_hz)
    return float(freqs[1 + int(np.argmax(spectrum[1:]))])


def _spectral_entropy(values: np.ndarray) -> float:
    """Normalised spectral flatness: 0 = pure tone, 1 = white noise."""
    power = np.abs(np.fft.rfft(values - values.mean())) ** 2
    power = power[1:]
    total = power.sum()
    if total <= 0:
        return 0.0
    p = power / total
    return float(-np.sum(p * np.log2(p + 1e-12)) / np.log2(len(p)))


def _band_power(values: np.ndarray, low: float, high: float, rate_hz: float = config.TARGET_RATE_HZ) -> float:
    """Fraction of AC power falling in [low, high) Hz."""
    power = np.abs(np.fft.rfft(values - values.mean())) ** 2
    freqs = np.fft.rfftfreq(len(values), d=1.0 / rate_hz)
    total = power[1:].sum()
    if total <= 0:
        return 0.0
    mask = (freqs >= low) & (freqs < high)
    return float(power[mask].sum() / total)


def cadence_hz(values: np.ndarray, rate_hz: float = config.TARGET_RATE_HZ) -> tuple[float, float]:
    """step rate via autocorrelation, returned as (hz, strength).

    strength is the height of the autocorrelation peak, 0 to 1. It is the
    more trustworthy than freq: measured medians are 0.13 lying and 0.17
    sitting against 0.44 walking and 0.53 running, so it reads as "is this
    rhythmic" even where the frequency itself is unreliable
    """
    centred = values - values.mean()
    if len(centred) < 4 or np.allclose(centred, 0):
        return 0.0, 0.0
    correlation = np.correlate(centred, centred, mode="full")[len(centred) - 1 :]
    if correlation[0] <= 0:
        return 0.0, 0.0
    correlation = correlation / correlation[0]

    # Lags are whole samples, so only rate_hz/lag is reachable: 4.17, 3.57,
    # 3.12, 2.78, 2.50 ... The spacing is coarse above ~2.5 Hz, which is exactly
    # the running range, and there is no interpolation between peaks.
    min_lag = max(1, int(rate_hz / 4.0))
    max_lag = min(len(correlation) - 1, int(rate_hz / 0.5))
    if max_lag <= min_lag:
        return 0.0, 0.0

    # argmax takes the largest peak in range, which may be the stride (one
    # cycle) rather than the step (two per stride), and on a still window it is
    # just noise. Always read this alongside the strength it returns.
    window = correlation[min_lag : max_lag + 1]
    peak = int(np.argmax(window))
    lag = min_lag + peak
    return float(rate_hz / lag), float(window[peak])


# 40 features in five groups: posture from gravity, intensity from body
# acceleration, rhythm from the frequency domain, rotation from the gyroscope,
# and cross-axis correlations. Order shd match the `values` list in extract().
FEATURE_NAMES: tuple[str, ...] = (
    "gravity_magnitude",
    "acc_unit_scale",
    "gravity_tilt_deg",
    "gravity_x", "gravity_y", "gravity_z",
    "gravity_std",
    "orientation_change_deg",
    "body_acc_rms",
    "body_acc_std",
    "body_acc_max",
    "body_acc_iqr",
    "body_acc_energy_x", "body_acc_energy_y", "body_acc_energy_z",
    "cadence_hz",
    "cadence_strength",
    "dominant_freq_hz",
    "spectral_entropy",
    "band_power_0_5_1_5",
    "band_power_1_5_2_5",
    "band_power_2_5_4_0",
    "band_power_4_0_12_5",
    "jerk_rms",
    "jerk_max",
    "crest_factor",
    "zero_crossing_rate",
    "gyro_rms",
    "gyro_std",
    "gyro_max",
    "gyro_dominant_freq_hz",
    "gyro_spectral_entropy",
    "gyro_energy_x", "gyro_energy_y", "gyro_energy_z",
    "gyro_axis_dominance",
    "acc_gyro_correlation",
    "acc_corr_xy", "acc_corr_xz", "acc_corr_yz",
)

N_FEATURES = len(FEATURE_NAMES)


def extract(window: np.ndarray, rate_hz: float = config.TARGET_RATE_HZ) -> np.ndarray:
    """Return the feature vector for one (T, 6) acc+gyro window."""

    acc = window[:, config.ACC_SLICE].astype(np.float64)
    gyro = window[:, config.GYRO_SLICE].astype(np.float64)

    gravity, body = separate_gravity(acc)
    gravity_mean = gravity.mean(axis=0)
    raw_gravity_magnitude = float(np.linalg.norm(gravity_mean))

    # dividing by the measured gravity magnitude puts every recording in g, 
    # so a threshold learnt on one user transfers to another.

    scale = raw_gravity_magnitude if raw_gravity_magnitude > 1e-6 else 1.0
    acc = acc / scale
    gravity = gravity / scale
    body = body / scale
    gravity_mean = gravity_mean / scale
    gravity_magnitude = float(np.linalg.norm(gravity_mean))

    # angle between gravity and the device z axis: near 0 when lying flat,
    # near 90 when upright. abs() folds face-up and face-down together.
    unit = gravity_mean / (gravity_magnitude + 1e-12)
    tilt_deg = float(np.degrees(np.arccos(np.clip(abs(unit[2]), 0.0, 1.0))))

    # total rotation of the gravity vector across the window: to diiferentiate a
    # posture that is held from one that is changing.
    gravity_unit = gravity / (np.linalg.norm(gravity, axis=1, keepdims=True) + 1e-12)
    cosines = np.clip(np.sum(gravity_unit[1:] * gravity_unit[:-1], axis=1), -1.0, 1.0)
    orientation_change = float(np.degrees(np.arccos(cosines)).sum())

    body_magnitude = np.linalg.norm(body, axis=1)
    body_rms = float(np.sqrt(np.mean(body_magnitude**2)))

    # Jerk is the derivative of acceleration. 
    jerk = np.diff(body, axis=0) * rate_hz
    jerk_magnitude = np.linalg.norm(jerk, axis=1) if len(jerk) else np.zeros(1)

    centred_body = body_magnitude - body_magnitude.mean()
    zero_crossings = float(np.mean(np.abs(np.diff(np.sign(centred_body))) > 0)) if len(centred_body) > 1 else 0.0

    cadence, cadence_power = cadence_hz(body_magnitude, rate_hz)

    gyro_magnitude = np.linalg.norm(gyro, axis=1)

    # Cycling rotates mostly about one axis, so energy concentrates; walking
    # spreads it. axis_dominance is that concentration as a fraction.
    gyro_energy = np.array([float(np.mean(gyro[:, axis] ** 2)) for axis in range(3)])
    gyro_total = gyro_energy.sum()
    axis_dominance = float(gyro_energy.max() / gyro_total) if gyro_total > 0 else 0.0

    def _corr(a: np.ndarray, b: np.ndarray) -> float:
        if np.std(a) < 1e-12 or np.std(b) < 1e-12:
            return 0.0
        return float(np.nan_to_num(np.corrcoef(a, b)[0, 1]))

    values = [
        gravity_magnitude,
        raw_gravity_magnitude,
        tilt_deg,
        float(gravity_mean[0]), float(gravity_mean[1]), float(gravity_mean[2]),
        float(np.linalg.norm(gravity.std(axis=0))),
        orientation_change,
        body_rms,
        float(body_magnitude.std()),
        float(body_magnitude.max()),
        float(np.percentile(body_magnitude, 75) - np.percentile(body_magnitude, 25)),
        float(np.mean(body[:, 0] ** 2)), float(np.mean(body[:, 1] ** 2)), float(np.mean(body[:, 2] ** 2)),
        cadence,
        cadence_power,
        _dominant_frequency(body_magnitude, rate_hz),
        _spectral_entropy(body_magnitude),
        _band_power(body_magnitude, 0.5, 1.5, rate_hz),
        _band_power(body_magnitude, 1.5, 2.5, rate_hz),
        _band_power(body_magnitude, 2.5, 4.0, rate_hz),
        _band_power(body_magnitude, 4.0, 12.5, rate_hz),
        float(np.sqrt(np.mean(jerk_magnitude**2))),
        float(jerk_magnitude.max()),
        float(body_magnitude.max() / (body_rms + 1e-12)),
        zero_crossings,
        float(np.sqrt(np.mean(gyro_magnitude**2))),
        float(gyro_magnitude.std()),
        float(gyro_magnitude.max()),
        _dominant_frequency(gyro_magnitude, rate_hz),
        _spectral_entropy(gyro_magnitude),
        float(gyro_energy[0]), float(gyro_energy[1]), float(gyro_energy[2]),
        axis_dominance,
        _corr(body_magnitude, gyro_magnitude),
        _corr(acc[:, 0], acc[:, 1]), _corr(acc[:, 0], acc[:, 2]), _corr(acc[:, 1], acc[:, 2]),
    ]

    # safety check to avoid adding new features in values without having matching name
    result = np.asarray(values, dtype=np.float64)
    if len(result) != N_FEATURES:
        raise AssertionError(f"produced {len(result)} features, FEATURE_NAMES declares {N_FEATURES}")
    return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)


def extract_batch(windows: np.ndarray, rate_hz: float = config.TARGET_RATE_HZ) -> np.ndarray:
    """Extract features for a stack of (N, T, 6) windows."""
    return np.vstack([extract(window, rate_hz) for window in windows])


BODY_RMS_INDEX = FEATURE_NAMES.index("body_acc_rms")
TILT_INDEX = FEATURE_NAMES.index("gravity_tilt_deg")
CADENCE_INDEX = FEATURE_NAMES.index("cadence_hz")
