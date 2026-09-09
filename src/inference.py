"""Create grounded, machine-readable predictions from one sensor recording."""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from signal_features import extract_features, load_axes


TIMESTAMP_PATTERN = re.compile(r"^(?P<timestamp>\d+)\.m_(?P<modality>.+)\.csv$")
MODALITY_DETAILS = {
    "raw_acc": ("accelerometer", ["Acc X", "Acc Y", "Acc Z"]),
    "proc_gyro": ("gyroscope", ["Gyro X", "Gyro Y", "Gyro Z"]),
}


def _recording_identity(csv_path: Path) -> tuple[int, str]:
    """Read the recording timestamp and modality from its stable filename."""
    match = TIMESTAMP_PATTERN.match(csv_path.name)
    if match is None:
        raise ValueError(f"Cannot infer timestamp/modality from {csv_path.name}")
    return int(match.group("timestamp")), match.group("modality")


def load_model(model_path: Path) -> dict[str, Any]:
    saved = joblib.load(model_path)
    required = {"model", "feature_names", "sample_rate_hz"}
    if not required.issubset(saved):
        raise ValueError(f"{model_path} is not a compatible saved baseline model")
    return saved


def predict_recording(csv_path: Path, saved_model: dict[str, Any]) -> dict[str, Any]:
    """Return prediction, probabilities, and exact sensor evidence for one file."""
    start_time_s, modality_key = _recording_identity(csv_path)
    acc_axes = load_axes(csv_path)
    
    gyro_path = Path(str(csv_path).replace("raw_acc", "proc_gyro"))
    if gyro_path.exists():
        gyro_axes = load_axes(gyro_path)
        modality = "accelerometer, gyroscope"
        channels = ["Acc X", "Acc Y", "Acc Z", "Gyro X", "Gyro Y", "Gyro Z"]
    else:
        gyro_axes = np.zeros_like(acc_axes)
        modality = "accelerometer"
        channels = ["Acc X", "Acc Y", "Acc Z"]
        
    combined_axes = np.hstack((acc_axes, gyro_axes))
    features = extract_features(combined_axes, sample_rate_hz=float(saved_model["sample_rate_hz"]))
    
    model = saved_model["model"]
    probabilities = model.predict_proba(features.reshape(1, -1))[0]
    predicted_index = int(np.argmax(probabilities))
    duration_s = len(acc_axes) / float(saved_model["sample_rate_hz"])

    feature_names = list(saved_model["feature_names"])
    importances = getattr(model, "feature_importances_", np.zeros(len(feature_names)))
    top_indices = np.argsort(importances)[-5:][::-1]
    evidence_features = [
        {
            "name": feature_names[int(index)],
            "value": round(float(features[int(index)]), 6),
            "global_model_importance": round(float(importances[int(index)]), 6),
        }
        for index in top_indices
    ]

    return {
        "schema_version": "1.0",
        "recording": {
            "source_csv": str(csv_path),
            "start_time_s": start_time_s,
            "end_time_s": start_time_s + duration_s,
            "sample_count": int(len(acc_axes)),
            "sample_rate_hz": float(saved_model["sample_rate_hz"]),
        },
        "prediction": {
            "activity": str(model.classes_[predicted_index]),
            "confidence": round(float(probabilities[predicted_index]), 6),
            "probabilities": {
                str(activity): round(float(probability), 6)
                for activity, probability in zip(model.classes_, probabilities)
            },
        },
        "evidence": {
            "sensor_modality": modality,
            "sensor_channels": channels,
            "feature_values": evidence_features,
        },
    }
