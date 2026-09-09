#!/usr/bin/env python3
"""Generate reproducible labelled QA cases from a ground-truth timeline."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

from qa import ACTIVITY_PATTERNS, execute


def display_name(activity: str) -> str:
    return "lying down" if activity == "lying_down" else activity


def add_case(cases: list[dict[str, Any]], question_type: str, question: str, timeline: dict[str, Any]) -> None:
    cases.append(
        {
            "id": f"{question_type}_{len(cases) + 1:04d}",
            "question_type": question_type,
            "question": question,
            "ground_truth": execute(question, timeline),
        }
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth-timeline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--identification-per-activity", type=int, default=2)
    args = parser.parse_args()
    timeline = json.loads(args.ground_truth_timeline.read_text(encoding="utf-8"))
    by_activity: dict[str, list[dict[str, Any]]] = {}
    for segment in timeline["segments"]:
        by_activity.setdefault(segment["activity"], []).append(segment)
    present = sorted(by_activity)
    cases: list[dict[str, Any]] = []

    for activity in present:
        for segment in by_activity[activity][: args.identification_per_activity]:
            time_s = segment["observed_windows"][0]["relative_start_s"] + 1
            add_case(cases, "identification", f"What activity was the user doing at {time_s:.0f} seconds?", timeline)
        name = display_name(activity)
        add_case(cases, "verification", f"Did the user {name}?", timeline)
        add_case(cases, "duration", f"How long was the user {name}?", timeline)
        add_case(cases, "count", f"How many times did the user {name}?", timeline)
        add_case(cases, "grounding", f"When did the user begin {name}?", timeline)

    for activity in sorted(ACTIVITY_PATTERNS):
        if activity not in present:
            add_case(cases, "verification", f"Did the user {display_name(activity)}?", timeline)
    for first, second in list(itertools.combinations(present, 2))[:6]:
        add_case(cases, "comparison", f"Did the user spend more time {display_name(first)} or {display_name(second)}?", timeline)

    output = {"schema_version": "1.0", "case_count": len(cases), "cases": cases}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(cases)} labelled QA cases to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
