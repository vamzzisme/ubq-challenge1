#!/usr/bin/env python3
"""Answer core activity questions from an evidence timeline without an LLM."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


ACTIVITY_PATTERNS = {
    "lying_down": ["lying down", "lie down", "lying"],
    "sitting": ["sitting", "sit"],
    "standing": ["standing", "stand"],
    "walking": ["walking", "walk"],
    "running": ["running", "run"],
    "bicycling": ["bicycling", "cycling", "bicycle", "bike"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeline", type=Path, required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--json", action="store_true", help="Print the structured result as JSON instead of the required text format.")
    return parser.parse_args()


def find_activities(question: str) -> list[str]:
    lower = question.lower()
    found: list[tuple[int, str]] = []
    for activity, phrases in ACTIVITY_PATTERNS.items():
        positions = [lower.find(phrase) for phrase in phrases if lower.find(phrase) >= 0]
        if positions:
            found.append((min(positions), activity))
    return [activity for _, activity in sorted(found)]


def find_time(question: str) -> float | None:
    match = re.search(r"(?:at|around|from)\s+(\d+(?:\.\d+)?)\s*(?:seconds?|s)?", question.lower())
    return float(match.group(1)) if match else None


def observed_windows(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [window for segment in segments for window in segment["observed_windows"]]


def segments_covering_time(segments: list[dict[str, Any]], time_s: float) -> list[dict[str, Any]]:
    """Return evidence limited to the observed windows that directly cover a time."""
    matches: list[dict[str, Any]] = []
    for segment in segments:
        windows = [window for window in segment["observed_windows"] if window["relative_start_s"] <= time_s <= window["relative_end_s"]]
        if windows:
            matches.append({**segment, "observed_windows": windows, "observed_duration_s": sum(w["end_time_s"] - w["start_time_s"] for w in windows)})
    return matches


def format_ranges(segments: list[dict[str, Any]]) -> str:
    windows = observed_windows(segments)
    if not windows:
        return "N/A"
    return ", ".join(f"{window['relative_start_s']:.0f} to {window['relative_end_s']:.0f}" for window in windows) + " seconds from start"


def evidence(segments: list[dict[str, Any]]) -> dict[str, str]:
    if not segments:
        return {"timestamps": "N/A", "modality": "N/A", "channels": "N/A", "intervals": []}
    modalities = sorted({segment["sensor_modality"] for segment in segments})
    channels = sorted({channel for segment in segments for channel in segment["sensor_channels"]})
    intervals = [
        {"start_s": window["relative_start_s"], "end_s": window["relative_end_s"]}
        for window in observed_windows(segments)
    ]
    return {"timestamps": format_ranges(segments), "modality": ", ".join(modalities), "channels": ", ".join(channels), "intervals": intervals}


def result(answer: str, event: str, segments: list[dict[str, Any]], explanation: str) -> dict[str, Any]:
    return {
        "time_base": "seconds from the first observed recording window",
        "answer": answer,
        "activity_event": event,
        "evidence": evidence(segments),
        "explanation": explanation,
    }


def execute(question: str, timeline: dict[str, Any]) -> dict[str, Any]:
    segments = timeline["segments"]
    activities = find_activities(question)
    lower = question.lower()
    time_s = find_time(question)
    by_activity: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for segment in segments:
        by_activity[segment["activity"]].append(segment)

    if any(phrase in lower for phrase in ["how long", "duration", "total time"]):
        if len(activities) == 1:
            selected = by_activity[activities[0]]
            duration = sum(segment["observed_duration_s"] for segment in selected)
            return result(
                f"{duration:.0f} observed seconds",
                activities[0],
                selected,
                f"This sums only the {len(observed_windows(selected))} directly observed 20-second sensor windows predicted as {activities[0]}; unobserved gaps are not counted.",
            )

    if any(phrase in lower for phrase in ["how many times", "how often", "number of times", "count"]):
        if len(activities) == 1:
            selected = by_activity[activities[0]]
            return result(
                str(len(selected)),
                f"{activities[0]} bouts",
                selected,
                "A bout is one group of equal-activity observed windows with no more than the configured unobserved gap between them.",
            )

    if "more time" in lower or "longer" in lower or "compare" in lower:
        if len(activities) == 2:
            first, second = activities
            first_duration = sum(segment["observed_duration_s"] for segment in by_activity[first])
            second_duration = sum(segment["observed_duration_s"] for segment in by_activity[second])
            winner = first if first_duration > second_duration else second if second_duration > first_duration else "Equal"
            selected = by_activity[first] + by_activity[second]
            return result(
                winner,
                f"{first}, {second}",
                selected,
                f"Observed {first_duration:.0f} seconds for {first} and {second_duration:.0f} seconds for {second}; unobserved gaps are excluded.",
            )

    if any(phrase in lower for phrase in ["begin", "start", "when did", "when was"]):
        if len(activities) == 1:
            selected = by_activity[activities[0]]
            if not selected:
                return result("No", activities[0], [], f"No observed window was predicted as {activities[0]}.")
            first = selected[0]
            return result(
                f"{first['relative_start_s']:.0f} seconds from start",
                f"onset of {activities[0]}",
                [first],
                f"The first observed window predicted as {activities[0]} begins at the cited time.",
            )

    if time_s is not None and ("what" in lower or "activity" in lower or "doing" in lower):
        matching = segments_covering_time(segments, time_s)
        if not matching:
            return result("N/A", "N/A", [], f"No sensor window directly covers {time_s:.0f} seconds from start.")
        selected = matching[0]
        return result(
            selected["activity"],
            selected["activity"],
            [selected],
            f"The cited sensor window covering {time_s:.0f} seconds was predicted as {selected['activity']} with mean confidence {selected['mean_confidence']:.2f}.",
        )

    if time_s is None and ("what" in lower or "activity" in lower or "doing" in lower):
        if not segments:
            return result("N/A", "N/A", [], "The timeline contains no observed sensor windows.")
        durations: dict[str, float] = defaultdict(float)
        for segment in segments:
            durations[segment["activity"]] += segment["observed_duration_s"]
        activity = max(durations, key=durations.get)
        selected = by_activity[activity]
        return result(
            activity,
            activity,
            selected,
            f"{activity} has the greatest directly observed duration in this timeline ({durations[activity]:.0f} seconds).",
        )

    if any(lower.startswith(prefix) for prefix in ["is ", "was ", "did ", "has "]) or "whether" in lower:
        if len(activities) == 1:
            selected = by_activity[activities[0]]
            if time_s is not None:
                selected = segments_covering_time(selected, time_s)
            answer = "Yes" if selected else "No"
            location = f" at {time_s:.0f} seconds from start" if time_s is not None else ""
            return result(answer, activities[0], selected, f"{len(observed_windows(selected))} observed sensor windows were predicted as {activities[0]}{location}.")

    # Fallback to SLM for Task 4: Open-World Activity Reasoning
    try:
        from slm_engine import slm_open_world_reasoning
        return slm_open_world_reasoning(question, timeline)
    except Exception as e:
        return result("N/A", "N/A", [], f"SLM open-world reasoning failed or not installed: {e}")


def render_text(answer: dict[str, Any]) -> str:
    evidence_block = answer["evidence"]
    return "\n".join(
        [
            f"Time base: {answer['time_base']}",
            f"Answer: {answer['answer']}",
            f"Activity/Event: {answer['activity_event']}",
            "Evidence:",
            f"  Timestamp(s): {evidence_block['timestamps']}",
            f"  Sensor Modality: {evidence_block['modality']}",
            f"  Sensor Channel(s): {evidence_block['channels']}",
            f"Explanation: {answer['explanation']}",
        ]
    )


def main() -> int:
    args = parse_args()
    timeline = json.loads(args.timeline.read_text(encoding="utf-8"))
    answer = execute(args.question, timeline)
    print(json.dumps(answer, indent=2) if args.json else render_text(answer))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
