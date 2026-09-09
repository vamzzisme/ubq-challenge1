#!/usr/bin/env python3
"""Inference bridge for the CNN model — same interface as predict_recording()."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from signal_features import load_axes
from train_cnn import AccelCNN


TIMESTAMP_PATTERN = re.compile(r"^(?P<timestamp>\d+)\.m_(?P<modality>.+)\.csv$")
MODALITY_DETAILS = {
    "raw_acc": ("accelerometer", ["Acc X", "Acc Y", "Acc Z"]),
    "proc_gyro": ("gyroscope", ["Gyro X", "Gyro Y", "Gyro Z"]),
}


def load_cnn_model(checkpoint_path: Path) -> dict[str, Any]:
    """Load a CNN checkpoint and return a dict compatible with the RF interface."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = AccelCNN(num_classes=checkpoint["num_classes"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return {
        "model": model,
        "class_names": checkpoint["class_names"],
        "class_to_idx": checkpoint["class_to_idx"],
        "sample_rate_hz": checkpoint["sample_rate_hz"],
        "model_type": "cnn",
    }


def predict_recording_cnn(csv_path: Path, saved: dict[str, Any]) -> dict[str, Any]:
    """Return prediction JSON in the same schema as inference.predict_recording()."""
    match = TIMESTAMP_PATTERN.match(csv_path.name)
    if match is None:
        raise ValueError(f"Cannot infer timestamp/modality from {csv_path.name}")
    start_time_s = int(match.group("timestamp"))
    modality_key = match.group("modality")
    modality, channels = MODALITY_DETAILS.get(modality_key, (modality_key, ["X", "Y", "Z"]))

    acc_axes = load_axes(csv_path)
    sample_rate = float(saved["sample_rate_hz"])
    duration_s = len(acc_axes) / sample_rate

    # Normalise acc
    acc_mean = acc_axes.mean(axis=0, keepdims=True)
    acc_std = acc_axes.std(axis=0, keepdims=True) + 1e-8
    acc_norm = (acc_axes - acc_mean) / acc_std

    # Try to load gyro
    gyro_path = Path(str(csv_path).replace("raw_acc", "proc_gyro"))
    if gyro_path.exists():
        gyro_axes = load_axes(gyro_path)
        gyro_mean = gyro_axes.mean(axis=0, keepdims=True)
        gyro_std = gyro_axes.std(axis=0, keepdims=True) + 1e-8
        gyro_norm = (gyro_axes - gyro_mean) / gyro_std
    else:
        gyro_norm = np.zeros_like(acc_norm)

    combined = np.hstack((acc_norm, gyro_norm)) # (500, 6)
    tensor = torch.from_numpy(combined.T).float().unsqueeze(0)  # (1, 6, 500)

    model = saved["model"]
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0].numpy()

    class_names = saved["class_names"]
    predicted_idx = int(np.argmax(probs))

    return {
        "schema_version": "1.0",
        "recording": {
            "source_csv": str(csv_path),
            "start_time_s": start_time_s,
            "end_time_s": start_time_s + duration_s,
            "sample_count": int(len(acc_axes)),
            "sample_rate_hz": sample_rate,
        },
        "prediction": {
            "activity": class_names[predicted_idx],
            "confidence": round(float(probs[predicted_idx]), 6),
            "probabilities": {
                name: round(float(p), 6)
                for name, p in zip(class_names, probs)
            },
        },
        "evidence": {
            "sensor_modality": modality,
            "sensor_channels": channels,
            "feature_values": [],  # CNN uses raw signals, no hand-crafted features
        },
    }
