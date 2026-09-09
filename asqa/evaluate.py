#!/usr/bin/env python3
"""QA evaluation: score answers by question type, under the rule each type deserves.

The brief is specific about scoring: a categorical answer is judged by exact
match, a numeric one by closeness, a temporal one by overlap, and an answer only
counts as *grounded* when the answer is right **and** the cited interval, modality
and channels hold up.  Those rules are implemented separately here rather than
collapsed into one accuracy number.

**On circularity.**  The obvious way to build ground truth is to run the same
question handlers over a label-derived timeline.  That measures the recogniser
but hides every bug in the interface layer, because the same code produces both
sides of the comparison.  So the reference answers here are computed *directly
from the label sequence* by `_truth_*` functions that never import the answer
router.  Where the two implementations disagree, the disagreement is real.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from asqa import config
from asqa.answer import answer_question
from asqa.timeline import DEFAULT_MERGE_GAP_S, Interval, Timeline, build_timeline

NUMERIC_TOLERANCE_FRACTION = 0.10  # "within about ten percent for a duration"
NUMERIC_TOLERANCE_FLOOR_S = 30.0
COUNT_TOLERANCE = 1  # "plus or minus one for a count"
IOU_THRESHOLD = 0.5


# ── Ground truth, computed independently of the answer router ────────────────


def truth_intervals(labels: np.ndarray, epoch_ts: np.ndarray, activity: str) -> list[tuple[float, float]]:
    """Merged true intervals for one activity, in seconds from the recording start."""
    order = np.argsort(epoch_ts)
    labels, epoch_ts = np.asarray(labels)[order], np.asarray(epoch_ts)[order]
    start_epoch = float(epoch_ts[0])

    spans: list[tuple[float, float]] = []
    current: tuple[float, float] | None = None
    previous_end = None
    for label, timestamp in zip(labels, epoch_ts):
        start = float(timestamp) - start_epoch
        end = start + config.WINDOW_DURATION_S
        if label != activity:
            continue
        if current is None:
            current = (start, end)
        elif previous_end is not None and (start - previous_end) <= DEFAULT_MERGE_GAP_S:
            current = (current[0], end)
        else:
            spans.append(current)
            current = (start, end)
        previous_end = end
    if current is not None:
        spans.append(current)
    return spans


def _truth_duration(spans: list[tuple[float, float]]) -> float:
    return sum(end - start for start, end in spans)


def _truth_count(spans: list[tuple[float, float]]) -> int:
    return len(spans)


def _truth_onset(spans: list[tuple[float, float]]) -> float | None:
    return spans[0][0] if spans else None


# ── Scoring rules ────────────────────────────────────────────────────────────


def categorical_correct(predicted: str, expected: str) -> bool:
    return predicted.strip().lower() == expected.strip().lower()


def numeric_value(text: str) -> float | None:
    import re

    match = re.search(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return float(match.group()) if match else None


def numeric_correct(predicted: str, expected: float, relative: float, floor: float) -> bool:
    value = numeric_value(predicted)
    if value is None:
        return False
    tolerance = max(floor, abs(expected) * relative)
    return abs(value - expected) <= tolerance


def count_correct(predicted: str, expected: int, tolerance: int = COUNT_TOLERANCE) -> bool:
    value = numeric_value(predicted)
    return value is not None and abs(value - expected) <= tolerance


def merge(spans: list[tuple[float, float]]) -> list[tuple[float, float]]:
    merged: list[tuple[float, float]] = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def interval_iou(predicted: list[tuple[float, float]], expected: list[tuple[float, float]]) -> float:
    """Intersection over union between two sets of intervals."""
    a, b = merge(predicted), merge(expected)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    intersection = sum(
        max(0.0, min(a_end, b_end) - max(a_start, b_start))
        for a_start, a_end in a
        for b_start, b_end in b
    )
    union = sum(e - s for s, e in a) + sum(e - s for s, e in b) - intersection
    return intersection / union if union > 0 else 0.0


# ── Case generation ──────────────────────────────────────────────────────────


@dataclass
class Case:
    question: str
    kind: str
    check: Callable[[Any], bool]
    truth_spans: list[tuple[float, float]]
    description: str


def build_cases(labels: np.ndarray, epoch_ts: np.ndarray) -> list[Case]:
    """Questions of every type, with reference answers from the labels."""
    present = [a for a in config.ACTIVITIES if a in set(labels)]
    absent = [a for a in config.ACTIVITIES if a not in set(labels)]
    cases: list[Case] = []

    for activity in present:
        name = config.DISPLAY_NAMES[activity]
        spans = truth_intervals(labels, epoch_ts, activity)

        cases.append(Case(f"Did the user {name}?", "verification",
                          lambda a: categorical_correct(a.answer, "Yes"), spans, "expect Yes"))
        duration = _truth_duration(spans)
        cases.append(Case(f"How long was the user {name}?", "duration",
                          lambda a, d=duration: numeric_correct(a.answer, d, NUMERIC_TOLERANCE_FRACTION, NUMERIC_TOLERANCE_FLOOR_S),
                          spans, f"expect ~{duration:.0f} s"))
        count = _truth_count(spans)
        cases.append(Case(f"How many times did the user {name}?", "count",
                          lambda a, c=count: count_correct(a.answer, c), spans, f"expect {count}"))
        onset = _truth_onset(spans)
        if onset is not None:
            cases.append(Case(f"When did the user begin {name}?", "grounding",
                              lambda a, o=onset: numeric_correct(a.answer, o, NUMERIC_TOLERANCE_FRACTION, NUMERIC_TOLERANCE_FLOOR_S),
                              spans[:1], f"expect onset ~{onset:.0f} s"))
        # Identification pinned to a moment inside a real bout.
        if spans:
            middle = (spans[0][0] + spans[0][1]) / 2
            cases.append(Case(f"What activity was the user doing at {middle:.0f} seconds?", "identification",
                              lambda a, n=name: categorical_correct(a.answer, n), spans[:1], f"expect {name}"))

    for activity in absent:
        name = config.DISPLAY_NAMES[activity]
        cases.append(Case(f"Did the user {name}?", "verification",
                          lambda a: categorical_correct(a.answer, "No"), [], "expect No"))

    # Comparison, over the two most common activities.
    totals = {a: _truth_duration(truth_intervals(labels, epoch_ts, a)) for a in present}
    ranked = sorted(totals, key=lambda a: totals[a], reverse=True)
    if len(ranked) >= 2:
        first, second = ranked[0], ranked[1]
        winner = config.DISPLAY_NAMES[first if totals[first] >= totals[second] else second]
        cases.append(Case(
            f"Did the user spend more time {config.DISPLAY_NAMES[first]} or {config.DISPLAY_NAMES[second]}?",
            "comparison",
            lambda a, w=winner: categorical_correct(a.answer, w),
            truth_intervals(labels, epoch_ts, first),
            f"expect {winner}",
        ))
        cases.append(Case("What activity is the user performing?", "identification",
                          lambda a, w=config.DISPLAY_NAMES[ranked[0]]: categorical_correct(a.answer, w),
                          truth_intervals(labels, epoch_ts, ranked[0]), f"expect {config.DISPLAY_NAMES[ranked[0]]}"))

    # Open-world: behaviour language rather than class names.
    resting = merge(truth_intervals(labels, epoch_ts, "lying_down") + truth_intervals(labels, epoch_ts, "sitting"))
    if resting:
        cases.append(Case("Was the user resting for a prolonged period?", "open_world",
                          lambda a: categorical_correct(a.answer, "Yes"), resting, "expect Yes"))
    active = merge(
        truth_intervals(labels, epoch_ts, "walking")
        + truth_intervals(labels, epoch_ts, "running")
        + truth_intervals(labels, epoch_ts, "bicycling")
    )
    cases.append(Case("Was the user doing anything strenuous?", "open_world",
                      lambda a, e=("Yes" if active else "No"): categorical_correct(a.answer, e),
                      active, f"expect {'Yes' if active else 'No'}"))
    return cases


# ── Running an evaluation ────────────────────────────────────────────────────


def predicted_spans(answer) -> list[tuple[float, float]]:
    return [(i["start_s"], i["end_s"]) for i in (answer.intervals or [])]


def evaluate_recording(
    timeline: Timeline,
    labels: np.ndarray,
    epoch_ts: np.ndarray,
    use_slm: bool = False,
    iou_threshold: float = IOU_THRESHOLD,
) -> list[dict[str, Any]]:
    """Score every generated case against one predicted timeline."""
    results = []
    for case in build_cases(labels, epoch_ts):
        answer = answer_question(case.question, timeline, use_slm=use_slm)
        correct = bool(case.check(answer))
        spans = predicted_spans(answer)
        iou = interval_iou(spans, case.truth_spans)

        source = config.evidence_source()
        cites = answer.timestamps != "N/A"
        source_ok = (not cites) or (
            answer.modality == source["modality"] and answer.channels == source["channels"]
        )
        # A negative answer legitimately cites nothing; it is grounded if correct.
        grounded = correct and source_ok and (iou >= iou_threshold if case.truth_spans else not cites)

        results.append({
            "question": case.question,
            "question_type": case.kind,
            "expected": case.description,
            "answer": answer.answer,
            "answer_correct": correct,
            "evidence_iou": round(iou, 4),
            "grounded_correct": bool(grounded),
            "cited_intervals": len(spans),
        })
    return results


def summarise(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_type: dict[str, list[dict]] = defaultdict(list)
    for row in results:
        by_type[row["question_type"]].append(row)

    per_type = {
        kind: {
            "count": len(rows),
            "answer_accuracy": float(np.mean([r["answer_correct"] for r in rows])),
            "grounded_accuracy": float(np.mean([r["grounded_correct"] for r in rows])),
            "mean_evidence_iou": float(np.mean([r["evidence_iou"] for r in rows])),
        }
        for kind, rows in sorted(by_type.items())
    }
    return {
        "case_count": len(results),
        # Macro-averaged across question types, as the brief specifies, so the
        # abundant easy sedentary cases do not dominate.
        "overall_qa_accuracy_macro": float(np.mean([v["answer_accuracy"] for v in per_type.values()])),
        "overall_grounded_accuracy_macro": float(np.mean([v["grounded_accuracy"] for v in per_type.values()])),
        "overall_qa_accuracy_micro": float(np.mean([r["answer_correct"] for r in results])),
        "by_question_type": per_type,
    }


def strictness_sweep(
    timeline: Timeline, labels: np.ndarray, epoch_ts: np.ndarray, use_slm: bool = False
) -> dict[str, Any]:
    """Accuracy as the IoU threshold and the numeric tolerance are tightened."""
    cases = build_cases(labels, epoch_ts)
    answers = [(case, answer_question(case.question, timeline, use_slm=use_slm)) for case in cases]

    iou_points = []
    for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        accepted = 0
        total = 0
        for case, answer in answers:
            if not case.truth_spans:
                continue
            total += 1
            if interval_iou(predicted_spans(answer), case.truth_spans) >= threshold:
                accepted += 1
        iou_points.append({"threshold": threshold, "fraction": accepted / total if total else 0.0})

    tolerance_points = []
    for fraction in [0.02, 0.05, 0.10, 0.20, 0.35, 0.50]:
        accepted = 0
        total = 0
        for case, answer in answers:
            if case.kind not in {"duration", "grounding"}:
                continue
            total += 1
            expected = numeric_value(case.description)
            if expected is not None and numeric_correct(answer.answer, expected, fraction, NUMERIC_TOLERANCE_FLOOR_S):
                accepted += 1
        tolerance_points.append({"tolerance_fraction": fraction, "fraction": accepted / total if total else 0.0})

    return {"iou": iou_points, "numeric_tolerance": tolerance_points}


def main() -> int:
    from asqa.pipeline import Pipeline
    from asqa.recognise import build_context_features
    from asqa.splits import load_folds

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, help="Evaluate one fold's test users.")
    parser.add_argument("--cv", action="store_true", help="Evaluate every fold.")
    parser.add_argument("--use-slm", action="store_true", help="Enable the language model (slow).")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    folds = load_folds()
    indices = range(folds["n_folds"]) if args.cv else [args.fold if args.fold is not None else 0]

    all_results: list[dict[str, Any]] = []
    per_fold: list[dict[str, Any]] = []
    sweeps: list[dict[str, Any]] = []

    for fold_index in indices:
        pipeline = Pipeline(fold=fold_index)
        for user_id in folds["splits"][str(fold_index)]["test"]:
            X, coarse, epoch_ts = build_context_features(user_id)
            truth, _ = pipeline.context_aware.standing_split.apply(X, coarse)
            timeline = pipeline.run(user_id)

            results = evaluate_recording(timeline, truth, epoch_ts, use_slm=args.use_slm)
            for row in results:
                row["user"] = user_id
                row["fold"] = fold_index
            all_results.extend(results)
            sweeps.append(strictness_sweep(timeline, truth, epoch_ts, use_slm=args.use_slm))

        fold_rows = [r for r in all_results if r["fold"] == fold_index]
        summary = summarise(fold_rows)
        summary["fold"] = fold_index
        per_fold.append(summary)
        print(
            f"fold {fold_index}: QA accuracy (macro) {summary['overall_qa_accuracy_macro']:.3f}, "
            f"grounded {summary['overall_grounded_accuracy_macro']:.3f}, {summary['case_count']} cases"
        )

    overall = summarise(all_results)
    print(f"\n{'=' * 72}")
    print(f"{'question type':<18}{'n':>6}{'answer acc':>13}{'grounded acc':>15}{'mean IoU':>11}")
    for kind, stats in overall["by_question_type"].items():
        print(
            f"{kind:<18}{stats['count']:>6}{stats['answer_accuracy']:>13.3f}"
            f"{stats['grounded_accuracy']:>15.3f}{stats['mean_evidence_iou']:>11.3f}"
        )
    print(f"\noverall QA accuracy (macro over types): {overall['overall_qa_accuracy_macro']:.3f}")
    print(f"overall grounded accuracy (macro)     : {overall['overall_grounded_accuracy_macro']:.3f}")
    print(f"overall QA accuracy (micro)           : {overall['overall_qa_accuracy_micro']:.3f}")

    # Average the sweeps across recordings for the strictness figure.
    def average(points: list[list[dict]], key: str) -> list[dict]:
        merged: dict[float, list[float]] = defaultdict(list)
        for series in points:
            for point in series:
                merged[point[key]].append(point["fraction"])
        return [{key: k, "fraction": float(np.mean(v))} for k, v in sorted(merged.items())]

    payload = {
        "overall": overall,
        "per_fold": per_fold,
        "strictness": {
            "iou": average([s["iou"] for s in sweeps], "threshold"),
            "numeric_tolerance": average([s["numeric_tolerance"] for s in sweeps], "tolerance_fraction"),
        },
        "scoring_rules": {
            "identification/verification/comparison/open_world": "exact match on the categorical answer",
            "duration/grounding": f"within max({NUMERIC_TOLERANCE_FLOOR_S:.0f} s, {NUMERIC_TOLERANCE_FRACTION:.0%}) of the reference",
            "count": f"within +/-{COUNT_TOLERANCE} bouts",
            "grounded": f"answer correct AND evidence IoU >= {IOU_THRESHOLD} AND modality/channels match",
        },
        "results": all_results,
    }
    config.EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    output = config.EVALUATION_DIR / (args.output or ("qa_cv.json" if args.cv else f"qa_fold{indices[0]}.json"))
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {output}")
    return 0


if __name__ == "__main__":
    from asqa.evaluate import main as _main

    raise SystemExit(_main())
