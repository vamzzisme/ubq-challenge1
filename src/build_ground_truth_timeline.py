#!/usr/bin/env python3
"""Build a timeline from self-labels for QA evaluation only."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from build_timeline import add_relative_times, build_segments


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=Path("data/processed/raw_acc_training_index.csv"))
    parser.add_argument("--user-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with args.index.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["user_id"] == args.user_id]
    if not rows:
        raise ValueError(f"No labelled recordings for user {args.user_id}")
    records: list[dict[str, Any]] = []
    for row in rows:
        start_time_s = float(row["recording_timestamp_s"])
        records.append(
            {
                "recording": {"source_csv": row["sensor_csv_path"], "start_time_s": start_time_s, "end_time_s": start_time_s + 20},
                "prediction": {"activity": row["activity"], "confidence": 1.0},
                "evidence": {
                    "sensor_modality": "accelerometer",
                    "sensor_channels": ["Acc X", "Acc Y", "Acc Z"],
                    "feature_values": [],
                },
            }
        )
    records.sort(key=lambda record: record["recording"]["start_time_s"])
    segments = build_segments(records, max_gap_s=60.0)
    recording_start = records[0]["recording"]["start_time_s"]
    add_relative_times(segments, recording_start)
    timeline = {
        "schema_version": "1.0",
        "source": "self-labelled ground truth",
        "time_base": "seconds from the first observed recording window",
        "recording_start_time_s": recording_start,
        "evidence_policy": "Each labelled row represents one directly observed 20-second accelerometer window.",
        "segment_count": len(segments),
        "segments": segments,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(timeline, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(segments)} ground-truth segments to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
