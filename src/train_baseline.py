#!/usr/bin/env python3
"""Train an interpretable Random Forest activity-recognition baseline.

The model uses one 500-sample accelerometer recording at a time. It does not
mix windows from a participant between train, validation, and test splits.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import numpy.typing as npt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from signal_features import extract_features, feature_names, load_axes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=Path("data/processed/raw_acc_training_index.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/baseline"))
    parser.add_argument("--validation-user", required=True, help="User ID held out for validation.")
    parser.add_argument("--test-user", required=True, help="User ID held out for final testing.")
    parser.add_argument("--trees", type=int, default=300, help="Number of trees (default: 300).")
    parser.add_argument("--max-depth", type=int, help="Maximum tree depth; omit for unrestricted depth.")
    parser.add_argument("--min-samples-leaf", type=int, default=2, help="Minimum recordings in a leaf (default: 2).")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def read_index(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"user_id", "recording_timestamp_s", "activity", "sensor_csv_path"}
    if not rows or set(rows[0]) != required:
        raise ValueError(f"{path} must contain exactly {sorted(required)}")
    return rows


def make_features(rows: list[dict[str, str]]) -> npt.NDArray[Any]:
    return np.vstack([extract_features(load_axes(row["sensor_csv_path"])) for row in rows])


def write_confusion_matrix(path: Path, matrix: npt.NDArray[Any], labels: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true_activity", *labels])
        for label, values in zip(labels, matrix):
            writer.writerow([label, *values])


def evaluate(model: RandomForestClassifier, features: npt.NDArray[Any], labels: npt.NDArray[Any], class_order: list[str]) -> dict[str, object]:
    predictions = model.predict(features)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "macro_f1": f1_score(labels, predictions, labels=class_order, average="macro", zero_division=0),  # type: ignore
        "report": classification_report(labels, predictions, labels=class_order, output_dict=True, zero_division=0),  # type: ignore
        "predictions": predictions,
    }


def main() -> int:
    args = parse_args()
    if args.validation_user == args.test_user:
        raise ValueError("Validation and test users must be different.")
    rows = read_index(args.index)
    train_rows = [row for row in rows if row["user_id"] not in {args.validation_user, args.test_user}]
    validation_rows = [row for row in rows if row["user_id"] == args.validation_user]
    test_rows = [row for row in rows if row["user_id"] == args.test_user]
    if not train_rows or not validation_rows or not test_rows:
        raise ValueError("Each split must contain at least one recording. Check the supplied user IDs.")

    print(f"Extracting {len(train_rows)} train, {len(validation_rows)} validation, and {len(test_rows)} test feature vectors...")
    train_x, validation_x, test_x = (make_features(split) for split in [train_rows, validation_rows, test_rows])
    train_y = np.asarray([row["activity"] for row in train_rows])
    validation_y = np.asarray([row["activity"] for row in validation_rows])
    test_y = np.asarray([row["activity"] for row in test_rows])
    class_order = sorted(set(train_y) | set(validation_y) | set(test_y))

    model = RandomForestClassifier(
        n_estimators=args.trees,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        max_features="sqrt",
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=args.seed,
    )
    print(f"Training {args.trees}-tree class-weighted Random Forest on {len(train_y)} recordings...")
    model.fit(train_x, train_y)
    validation = evaluate(model, validation_x, validation_y, class_order)
    test = evaluate(model, test_x, test_y, class_order)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_config = {
        "model_type": "RandomForestClassifier",
        "trees": args.trees,
        "max_depth": args.max_depth,
        "min_samples_leaf": args.min_samples_leaf,
        "max_features": "sqrt",
        "class_weight": "balanced_subsample",
        "seed": args.seed,
    }
    joblib.dump(
        {"model": model, "feature_names": feature_names(), "sample_rate_hz": 25.0, "training_config": model_config},
        args.output_dir / "random_forest.joblib",
    )
    importances = sorted(zip(feature_names(), model.feature_importances_), key=lambda item: item[1], reverse=True)
    with (args.output_dir / "feature_importance.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["feature", "importance"])
        writer.writerows(importances)
    write_confusion_matrix(args.output_dir / "test_confusion_matrix.csv", confusion_matrix(test_y, test["predictions"], labels=class_order), class_order)  # type: ignore[arg-type]
    summary = {
        "split": {"train_users": sorted({r['user_id'] for r in train_rows}), "validation_user": args.validation_user, "test_user": args.test_user},
        "recordings": {"train": len(train_rows), "validation": len(validation_rows), "test": len(test_rows)},
        "train_class_counts": dict(Counter(train_y)),
        "model_config": model_config,
        "validation": {key: value for key, value in validation.items() if key != "predictions"},
        "test": {key: value for key, value in test.items() if key != "predictions"},
    }
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"Validation accuracy={validation['accuracy']:.3f}, macro-F1={validation['macro_f1']:.3f}")
    print(f"Test accuracy={test['accuracy']:.3f}, macro-F1={test['macro_f1']:.3f}")
    print(f"Wrote model and metrics to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
