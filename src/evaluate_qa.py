#!/usr/bin/env python3
"""Score deterministic QA answers and their cited evidence against labelled cases."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from qa import execute


def numeric_value(answer: str) -> float | None:
    match = re.search(r"-?\d+(?:\.\d+)?", answer)
    return float(match.group()) if match else None


def merge_intervals(intervals: list[dict[str, float]]) -> list[tuple[float, float]]:
    merged: list[tuple[float, float]] = []
    for start, end in sorted((float(item["start_s"]), float(item["end_s"])) for item in intervals):
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def interval_iou(predicted: list[dict[str, float]], expected: list[dict[str, float]]) -> float:
    pred, truth = merge_intervals(predicted), merge_intervals(expected)
    if not pred and not truth:
        return 1.0
    if not pred or not truth:
        return 0.0
    intersection = sum(max(0.0, min(a_end, b_end) - max(a_start, b_start)) for a_start, a_end in pred for b_start, b_end in truth)
    pred_length = sum(end - start for start, end in pred)
    truth_length = sum(end - start for start, end in truth)
    return intersection / (pred_length + truth_length - intersection)


def answer_correct(question_type: str, predicted: dict[str, Any], expected: dict[str, Any], numeric_tolerance_s: float) -> bool:
    if question_type in {"duration", "grounding"}:
        predicted_number, expected_number = numeric_value(predicted["answer"]), numeric_value(expected["answer"])
        return predicted_number is not None and expected_number is not None and abs(predicted_number - expected_number) <= numeric_tolerance_s
    return predicted["answer"].strip().lower() == expected["answer"].strip().lower()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predicted-timeline", type=Path, required=True)
    parser.add_argument("--qa-cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--numeric-tolerance-seconds", type=float, default=20.0)
    parser.add_argument("--evidence-iou-threshold", type=float, default=0.5)
    args = parser.parse_args()
    predicted_timeline = json.loads(args.predicted_timeline.read_text(encoding="utf-8"))
    cases = json.loads(args.qa_cases.read_text(encoding="utf-8"))["cases"]
    results: list[dict[str, Any]] = []
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for case in cases:
        predicted = execute(case["question"], predicted_timeline)
        expected = case["ground_truth"]
        correct = answer_correct(case["question_type"], predicted, expected, args.numeric_tolerance_seconds)
        iou = interval_iou(predicted["evidence"]["intervals"], expected["evidence"]["intervals"])
        same_source = (
            predicted["evidence"]["modality"] == expected["evidence"]["modality"]
            and predicted["evidence"]["channels"] == expected["evidence"]["channels"]
        )
        grounded = correct and iou >= args.evidence_iou_threshold and same_source
        result = {"id": case["id"], "question_type": case["question_type"], "question": case["question"], "answer_correct": correct, "evidence_iou": round(iou, 6), "grounded_correct": grounded, "predicted": predicted, "expected": expected}
        results.append(result)
        by_type[case["question_type"]].append(result)

    summary_by_type = {
        question_type: {
            "count": len(items),
            "answer_accuracy": sum(item["answer_correct"] for item in items) / len(items),
            "grounded_accuracy": sum(item["grounded_correct"] for item in items) / len(items),
            "mean_evidence_iou": sum(item["evidence_iou"] for item in items) / len(items),
        }
        for question_type, items in sorted(by_type.items())
    }
    output = {"case_count": len(results), "numeric_tolerance_seconds": args.numeric_tolerance_seconds, "evidence_iou_threshold": args.evidence_iou_threshold, "by_question_type": summary_by_type, "results": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"Evaluated {len(results)} QA cases")
    for question_type, summary in summary_by_type.items():
        print(f"{question_type}: answer={summary['answer_accuracy']:.3f}, grounded={summary['grounded_accuracy']:.3f}, IoU={summary['mean_evidence_iou']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
