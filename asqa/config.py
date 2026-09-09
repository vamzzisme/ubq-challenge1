"""Shared constants for the Ask-the-Sensors QA pipeline.

Every layer imports its constants from here so that the sampling rate, the class
vocabulary, and the sensor-channel vocabulary cannot drift between the training
path, the inference path, and the evaluation path.  Channel drift between those
paths is exactly what made evidence grounding unscoreable in the previous
implementation, so the channel names in particular are defined once and only
once.
"""

from __future__ import annotations

import os
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "Original Data"
ACC_DIR = RAW_DIR / "acc"
GYRO_DIR = RAW_DIR / "gyro"

OUTPUT_DIR = REPO_ROOT / "outputs"

# Window length is a design parameter with a real accuracy trade-off (see
# WINDOW_SAMPLES below), so caches, features and models are namespaced by a tag
# and two configurations can coexist while being compared.
#     ASQA_TAG=w8 ASQA_WINDOW_SAMPLES=200 python -m asqa.preprocess
CACHE_TAG = os.environ.get("ASQA_TAG", "w15")
CACHE_DIR = OUTPUT_DIR / "cache" / CACHE_TAG
FOLDS_PATH = OUTPUT_DIR / f"folds_{CACHE_TAG}.json"
MODEL_DIR = OUTPUT_DIR / "models"
TIMELINE_DIR = OUTPUT_DIR / "timelines"
FIGURE_DIR = OUTPUT_DIR / "figures"
EVALUATION_DIR = OUTPUT_DIR / "evaluation"

# ── Sampling ─────────────────────────────────────────────────────────────────

TARGET_RATE_HZ = 25.0
TARGET_PERIOD_S = 1.0 / TARGET_RATE_HZ  # 0.04 s exactly

# The analysis window is 15 s, not the 20 s a "20-second capture" would suggest.
# Devices in ExtraSensory disagree about what 800 samples means: accelerometers
# run at ~32-50 Hz and gyroscopes at ~22-90 Hz, so one capture covers anywhere
# from ~9 s to ~25 s, and the two modalities overlap for less time still.
# Measured over a 2,250-window sample across all 15 users, the fraction of
# windows whose acc/gyro overlap supports a given analysis length is:
#
#     20 s -> 33.6%      16 s -> 69.2%      15 s -> 91.3%      10 s -> 92.0%
#
# 15 s sits on the knee: it keeps 91.3% of the data (13/15 users above 90%)
# while leaving ample frequency resolution (0.067 Hz) to separate a 1.8 Hz
# walking cadence from a 2.8 Hz running cadence.
# The shorter the window, the more of the corpus survives -- and the rare
# classes live disproportionately in the users with short captures.  Measured
# on the labelled running windows specifically:
#
#     window   running kept   bicycling kept
#       15 s      122/169        1179/1181
#        8 s      166/169        1181/1181
#
# The whole difference is one user (1DBB0F6F) whose gyroscope captures span only
# ~8.3 s.  Running is the scarcest class in the corpus, so this trade is
# evaluated rather than assumed; see outputs/evaluation/window_comparison.json.
WINDOW_SAMPLES = int(os.environ.get("ASQA_WINDOW_SAMPLES", "375"))
WINDOW_DURATION_S = WINDOW_SAMPLES / TARGET_RATE_HZ

# Quality gates applied per window during preprocessing.  ExtraSensory was
# recorded in the wild: the accelerometer clock is genuinely uneven (measured
# dt spans 0.024-0.077 s) and whole stretches are missing.  A window that
# cannot support an honest 25 Hz reconstruction is dropped and counted, never
# silently stretched or padded.
MAX_SAMPLE_GAP_S = 0.5

# The grid must fit entirely inside the interval both sensors actually observed:
# every returned sample is interpolated between two real measurements, never
# extrapolated or held.
WINDOW_SPAN_S = (WINDOW_SAMPLES - 1) * TARGET_PERIOD_S  # 14.96 s
MIN_OVERLAP_S = WINDOW_SPAN_S

# ── Activity vocabulary ──────────────────────────────────────────────────────

# The seven challenge classes, in a fixed order used for every confusion
# matrix, probability vector, and transition matrix in the project.
ACTIVITIES: tuple[str, ...] = (
    "lying_down",
    "sitting",
    "standing_in_place",
    "standing_and_moving",
    "walking",
    "running",
    "bicycling",
)
ACTIVITY_INDEX = {name: index for index, name in enumerate(ACTIVITIES)}

# Human-readable forms used when writing answers back to the user.
DISPLAY_NAMES = {
    "lying_down": "lying down",
    "sitting": "sitting",
    "standing_in_place": "standing in place",
    "standing_and_moving": "standing and moving",
    "walking": "walking",
    "running": "running",
    "bicycling": "bicycling",
}

# The static/dynamic partition used by the hierarchical recogniser.
STATIC_ACTIVITIES = ("lying_down", "sitting", "standing_in_place")
DYNAMIC_ACTIVITIES = ("standing_and_moving", "walking", "running", "bicycling")

# ExtraSensory self-report columns.  Verified across all 15 users: these six
# labels are mutually exclusive (0 rows carry more than one), so no
# disambiguation is required.  `OR_standing` is later split into the two
# standing classes by measured body-acceleration energy -- see preprocess.py.
LABEL_COLUMNS = {
    "lying_down": "label:LYING_DOWN",
    "sitting": "label:SITTING",
    "standing": "label:OR_standing",
    "walking": "label:FIX_walking",
    "running": "label:FIX_running",
    "bicycling": "label:BICYCLING",
}

# Provenance markers so the report can score the derived class separately from
# the annotated ones.
PROVENANCE_ANNOTATED = "annotated"
PROVENANCE_DERIVED = "derived"

# ── Sensor channel vocabulary ────────────────────────────────────────────────

# Column order of every cached window array: acc X/Y/Z then gyro X/Y/Z.
CHANNEL_NAMES: tuple[str, ...] = (
    "Acc X",
    "Acc Y",
    "Acc Z",
    "Gyro X",
    "Gyro Y",
    "Gyro Z",
)
ACC_SLICE = slice(0, 3)
GYRO_SLICE = slice(3, 6)

# The single source of truth for the two evidence fields the brief requires.
# Both the predicted timeline and the ground-truth timeline call these, which
# is what keeps `Sensor Modality` and `Sensor Channel(s)` comparable at scoring
# time.
SENSOR_MODALITY = "Accelerometer, Gyroscope"
SENSOR_CHANNELS = "All"


def evidence_source() -> dict[str, str]:
    """Return the modality/channel evidence fields for a fused acc+gyro window."""
    return {"modality": SENSOR_MODALITY, "channels": SENSOR_CHANNELS}


# ── Time base ────────────────────────────────────────────────────────────────

# The brief requires one consistent convention, stated explicitly.
TIME_BASE = "seconds from the start of the recording"

# ── Raw file naming ──────────────────────────────────────────────────────────

ACC_SUFFIX = ".m_raw_acc.dat"
GYRO_SUFFIX = ".m_proc_gyro.dat"
LABEL_SUFFIX = ".features_labels.csv"
