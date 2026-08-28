#!/usr/bin/env python3
"""Match resampled sensor recordings to one unambiguous activity label.

The ExtraSensory feature-label CSV contains one row per recording timestamp.
This script keeps rows for the supported activity labels only, verifies that
exactly one target label is active, and emits a compact training index that
points to the corresponding resampled sensor CSV.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


CLASS_COLUMNS = {
    "lying_down": "label:LYING_DOWN",
    "sitting": "label:SITTING",
    "standing": "label:OR_standing",
    "walking": "label:FIX_walking",
    "running": "label:FIX_running",
    "bicycling": "label:BICYCLING",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--resampled-dir",
        type=Path,
        default=Path("data/processed/raw_acc_25hz"),
        help="Directory created by prepare_sensor_data.py for one modality.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/raw_acc_training_index.csv"),
    )
    return parser.parse_args()


def build_recording_lookup(resampled_dir: Path) -> dict[tuple[str, str], Path]:
    """Index the resampled file tree once, avoiding a directory scan per label."""
    lookup: dict[tuple[str, str], Path] = {}
    for path in resampled_dir.rglob("*.csv"):
        timestamp = path.name.split(".", maxsplit=1)[0]
        key = (path.parent.name, timestamp)
        if key in lookup:
            raise ValueError(f"Duplicate resampled recordings for user/timestamp {key}")
        lookup[key] = path
    return lookup


def main() -> int:
    args = parse_args()
    rows: list[dict[str, str]] = []
    class_counts: Counter[str] = Counter()
    skipped_unlabelled = 0
    ambiguous = 0
    missing_recordings = 0

    label_files = sorted(args.data_dir.glob("*.features_labels.csv"))
    if not label_files:
        raise FileNotFoundError(f"No *.features_labels.csv files found under {args.data_dir}")
    recording_lookup = build_recording_lookup(args.resampled_dir)
    if not recording_lookup:
        raise FileNotFoundError(f"No resampled CSV files found under {args.resampled_dir}")

    for label_file in label_files:
        user_id = label_file.name.removesuffix(".features_labels.csv")
        with label_file.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            missing_columns = set(CLASS_COLUMNS.values()) - set(reader.fieldnames or [])
            if missing_columns:
                raise ValueError(f"{label_file} is missing label columns: {sorted(missing_columns)}")
            for source_row in reader:
                active_classes = [
                    class_name
                    for class_name, column_name in CLASS_COLUMNS.items()
                    if source_row[column_name] == "1"
                ]
                if not active_classes:
                    skipped_unlabelled += 1
                    continue
                if len(active_classes) > 1:
                    ambiguous += 1
                    continue
                timestamp = source_row["timestamp"]
                csv_path = recording_lookup.get((user_id, timestamp))
                if csv_path is None:
                    missing_recordings += 1
                    continue
                activity = active_classes[0]
                rows.append(
                    {
                        "user_id": user_id,
                        "recording_timestamp_s": timestamp,
                        "activity": activity,
                        "sensor_csv_path": str(csv_path),
                    }
                )
                class_counts[activity] += 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["user_id", "recording_timestamp_s", "activity", "sensor_csv_path"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} labelled recordings to {args.output}")
    print("Class counts:")
    for activity in CLASS_COLUMNS:
        print(f"  {activity}: {class_counts[activity]}")
    print(f"Skipped: {skipped_unlabelled} without a target label, {ambiguous} ambiguous, {missing_recordings} missing sensor CSV")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
