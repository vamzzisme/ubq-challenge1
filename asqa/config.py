"""Shared constants for the Ask-the-Sensors QA pipeline."""

from __future__ import annotations

import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "Original Data"
ACC_DIR = RAW_DIR / "acc"
GYRO_DIR = RAW_DIR / "gyro"

OUTPUT_DIR = REPO_ROOT / "outputs"

CACHE_TAG = os.environ.get("ASQA_TAG", "w15")
CACHE_DIR = OUTPUT_DIR / "cache" / CACHE_TAG
FOLDS_PATH = OUTPUT_DIR / f"folds_{CACHE_TAG}.json"
MODEL_DIR = OUTPUT_DIR / "models"
TIMELINE_DIR = OUTPUT_DIR / "timelines"
FIGURE_DIR = OUTPUT_DIR / "figures"
EVALUATION_DIR = OUTPUT_DIR / "evaluation"


TARGET_RATE_HZ = 25.0
TARGET_PERIOD_S = 1.0 / TARGET_RATE_HZ

WINDOW_SAMPLES = int(os.environ.get("ASQA_WINDOW_SAMPLES", "375"))
WINDOW_DURATION_S = WINDOW_SAMPLES / TARGET_RATE_HZ

MAX_SAMPLE_GAP_S = 0.5

WINDOW_SPAN_S = (WINDOW_SAMPLES - 1) * TARGET_PERIOD_S
MIN_OVERLAP_S = WINDOW_SPAN_S


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

DISPLAY_NAMES = {
    "lying_down": "lying down",
    "sitting": "sitting",
    "standing_in_place": "standing in place",
    "standing_and_moving": "standing and moving",
    "walking": "walking",
    "running": "running",
    "bicycling": "bicycling",
}

STATIC_ACTIVITIES = ("lying_down", "sitting", "standing_in_place")
DYNAMIC_ACTIVITIES = ("standing_and_moving", "walking", "running", "bicycling")

LABEL_COLUMNS = {
    "lying_down": "label:LYING_DOWN",
    "sitting": "label:SITTING",
    "standing": "label:OR_standing",
    "walking": "label:FIX_walking",
    "running": "label:FIX_running",
    "bicycling": "label:BICYCLING",
}

PROVENANCE_ANNOTATED = "annotated"
PROVENANCE_DERIVED = "derived"


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

SENSOR_MODALITY = "Accelerometer, Gyroscope"
SENSOR_CHANNELS = "All"


def evidence_source() -> dict[str, str]:
    """Return the modality/channel evidence fields for a fused acc+gyro window."""
    return {"modality": SENSOR_MODALITY, "channels": SENSOR_CHANNELS}


TIME_BASE = "seconds from the start of the recording"


ACC_SUFFIX = ".m_raw_acc.dat"
GYRO_SUFFIX = ".m_proc_gyro.dat"
LABEL_SUFFIX = ".features_labels.csv"
