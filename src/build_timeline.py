#!/usr/bin/env python3
"""Aggregate prediction JSON files into an evidence-preserving activity timeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--max-unobserved-gap-seconds",
        type=float,
        default=60.0,
        help="Merge equal-activity windows only when their unsampled gap is no larger than this (default: 60).",
    )
    return parser.parse_args()


def load_predictions(predictions_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(predictions_dir.rglob("*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            recording = record["recording"]
            prediction = record["prediction"]
            evidence = record["evidence"]
            _ = recording["start_time_s"], recording["end_time_s"], prediction["activity"], evidence["sensor_modality"]
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise ValueError(f"Invalid prediction record: {path}") from exc
        records.append(record)
    return sorted(records, key=lambda item: item["recording"]["start_time_s"])


def make_segment(record: dict[str, Any]) -> dict[str, Any]:
    recording, prediction, evidence = record["recording"], record["prediction"], record["evidence"]
    window = {
        "start_time_s": recording["start_time_s"],
        "end_time_s": recording["end_time_s"],
        "confidence": prediction["confidence"],
        "source_csv": recording["source_csv"],
    }
    return {
        "activity": prediction["activity"],
        "sensor_modality": evidence["sensor_modality"],
        "sensor_channels": evidence["sensor_channels"],
        "observed_windows": [window],
        "observed_duration_s": recording["end_time_s"] - recording["start_time_s"],
        "mean_confidence": prediction["confidence"],
        "evidence_features": evidence["feature_values"],
    }


def append_to_segment(segment: dict[str, Any], record: dict[str, Any]) -> None:
    recording, prediction = record["recording"], record["prediction"]
    segment["observed_windows"].append(
        {
            "start_time_s": recording["start_time_s"],
            "end_time_s": recording["end_time_s"],
            "confidence": prediction["confidence"],
            "source_csv": recording["source_csv"],
        }
    )
    window_count = len(segment["observed_windows"])
    segment["observed_duration_s"] += recording["end_time_s"] - recording["start_time_s"]
    segment["mean_confidence"] = round(
        ((segment["mean_confidence"] * (window_count - 1)) + prediction["confidence"]) / window_count,
        6,
    )


def build_segments(records: list[dict[str, Any]], max_gap_s: float) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    for record in records:
        if not segments:
            segments.append(make_segment(record))
            continue
        current = segments[-1]
        current_end = current["observed_windows"][-1]["end_time_s"]
        next_start = record["recording"]["start_time_s"]
        same_evidence_source = (
            current["activity"] == record["prediction"]["activity"]
            and current["sensor_modality"] == record["evidence"]["sensor_modality"]
            and current["sensor_channels"] == record["evidence"]["sensor_channels"]
        )
        if same_evidence_source and next_start - current_end <= max_gap_s:
            append_to_segment(current, record)
        else:
            segments.append(make_segment(record))

    for segment in segments:
        windows = segment["observed_windows"]
        segment["start_time_s"] = windows[0]["start_time_s"]
        segment["end_time_s"] = windows[-1]["end_time_s"]
        segment["sampled_span_s"] = segment["end_time_s"] - segment["start_time_s"]
        segment["unobserved_gap_s"] = segment["sampled_span_s"] - segment["observed_duration_s"]
    return segments


def add_relative_times(segments: list[dict[str, Any]], recording_start_time_s: float) -> None:
    """Expose the challenge-required seconds-from-start time base alongside raw Unix times."""
    for segment in segments:
        segment["relative_start_s"] = segment["start_time_s"] - recording_start_time_s
        segment["relative_end_s"] = segment["end_time_s"] - recording_start_time_s
        for window in segment["observed_windows"]:
            window["relative_start_s"] = window["start_time_s"] - recording_start_time_s
            window["relative_end_s"] = window["end_time_s"] - recording_start_time_s


def main() -> int:
    args = parse_args()
    records = load_predictions(args.predictions_dir)
    if not records:
        raise FileNotFoundError(f"No prediction JSON files found under {args.predictions_dir}")
    segments = build_segments(records, args.max_unobserved_gap_seconds)
    recording_start_time_s = records[0]["recording"]["start_time_s"]
    add_relative_times(segments, recording_start_time_s)
    timeline = {
        "schema_version": "1.0",
        "time_base": "seconds from the first observed recording window",
        "recording_start_time_s": recording_start_time_s,
        "evidence_policy": (
            "observed_windows are the only directly observed sensor intervals. "
            "sampled_span_s includes unobserved gaps and must not be reported as observed activity duration."
        ),
        "segment_count": 0,
        "segments": segments,
    }
    timeline["segment_count"] = len(timeline["segments"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(timeline, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {timeline['segment_count']} evidence-preserving segments to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
