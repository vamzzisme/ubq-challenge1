"""QA evaluation: score answers by question type, under the rule each type deserves."""

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

NUMERIC_TOLERANCE_FRACTION = 0.20
NUMERIC_TOLERANCE_FLOOR_S = 30.0
# Onset is fixed by a single window, so one stray early prediction moves the answer
# by hours. A relative band is also meaningless here: recordings that begin with the
# activity have a reference near zero, against which any error is infinite. Judge it
# against a fixed window instead.
ONSET_TOLERANCE_S = 300.0
# Bout counts scale from 1 to over 60, so an absolute band both over-punishes the
# common classes and rewards answering zero on the rare ones. Keep a floor for the
# small counts and let it widen with the reference.
COUNT_TOLERANCE = 5
COUNT_TOLERANCE_FRACTION = 0.20
IOU_THRESHOLD = 0.5

# Accuracy alone is misleading on either axis: the activity classes are heavily
# imbalanced, and the verification questions are mostly `expect No`, so a model
# that always answers No scores well. Categorical types therefore also carry
# macro-F1 and balanced accuracy, and verification additionally carries
# precision, recall, F1 and specificity on the positive class.
CATEGORICAL_KINDS = frozenset({"identification", "verification", "comparison", "open_world"})
NUMERIC_KINDS = frozenset({"duration", "grounding", "count"})
# Types whose reference value is a duration in seconds, so a percentage error means something.
PERCENTAGE_ERROR_KINDS = frozenset({"duration", "grounding"})


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


def count_correct(
    predicted: str,
    expected: int,
    tolerance: int = COUNT_TOLERANCE,
    relative: float = COUNT_TOLERANCE_FRACTION,
) -> bool:
    value = numeric_value(predicted)
    if value is None:
        return False
    return abs(value - expected) <= max(tolerance, abs(expected) * relative)


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


def _overlap(a: tuple[float, float], b: tuple[float, float]) -> float:
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def temporal_scores(
    predicted: list[tuple[float, float]], expected: list[tuple[float, float]]
) -> dict[str, float]:
    """Temporal precision, recall and F1 between two interval sets.

    IoU answers with a single number and cannot say which way an answer is wrong.
    Overlap over predicted length is the precision - how much of what was cited is
    really the activity - and overlap over true length is the recall - how much of
    the activity was cited. A citation covering the whole recording has recall 1.0
    and precision near 0; one naming a single correct window has the reverse.
    """
    a, b = merge(predicted), merge(expected)
    if not a and not b:
        return {"temporal_precision": 1.0, "temporal_recall": 1.0, "temporal_f1": 1.0}
    if not a or not b:
        return {"temporal_precision": 0.0, "temporal_recall": 0.0, "temporal_f1": 0.0}

    overlap = sum(_overlap(x, y) for x in a for y in b)
    predicted_length = sum(end - start for start, end in a)
    true_length = sum(end - start for start, end in b)
    precision = overlap / predicted_length if predicted_length > 0 else 0.0
    recall = overlap / true_length if true_length > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"temporal_precision": precision, "temporal_recall": recall, "temporal_f1": f1}


def matched_iou(
    predicted: list[tuple[float, float]], expected: list[tuple[float, float]]
) -> float:
    """Mean IoU after matching each predicted interval to a true one by overlap.

    `interval_iou` pools both sides into one set, so a citation that spans several
    true intervals at once scores as well as one that resolves them separately.
    Matching one-to-one instead - greedily, best overlap first - keeps the count of
    intervals honest: every unmatched interval on either side contributes zero, so
    both a missed interval and a spurious one cost the same.
    """
    a, b = merge(predicted), merge(expected)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0

    pairs = sorted(
        ((_overlap(x, y), i, j) for i, x in enumerate(a) for j, y in enumerate(b)),
        key=lambda pair: pair[0],
        reverse=True,
    )
    used_a: set[int] = set()
    used_b: set[int] = set()
    scores: list[float] = []
    for overlap, i, j in pairs:
        if overlap <= 0 or i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        union = (a[i][1] - a[i][0]) + (b[j][1] - b[j][0]) - overlap
        scores.append(overlap / union if union > 0 else 0.0)

    # Unmatched intervals on either side score zero, so the mean is taken over the
    # larger of the two sets rather than over the matches alone.
    return float(np.sum(scores) / max(len(a), len(b)))


@dataclass
class Case:
    question: str
    kind: str
    check: Callable[[Any], bool]
    truth_spans: list[tuple[float, float]]
    description: str
    # The reference in machine-readable form, so the scorer can report the size of
    # an error and not merely whether it cleared a threshold.
    expected_label: str | None = None
    expected_value: float | None = None
    # The activities this question is about, for the grounding-precision check.
    activities: tuple[str, ...] = ()


def build_cases(labels: np.ndarray, epoch_ts: np.ndarray) -> list[Case]:
    """Questions of every type, with reference answers from the labels."""
    present = [a for a in config.ACTIVITIES if a in set(labels)]
    absent = [a for a in config.ACTIVITIES if a not in set(labels)]
    cases: list[Case] = []

    for activity in present:
        name = config.DISPLAY_NAMES[activity]
        spans = truth_intervals(labels, epoch_ts, activity)

        cases.append(Case(f"Did the user {name}?", "verification",
                          lambda a: categorical_correct(a.answer, "Yes"), spans, "expect Yes",
                          expected_label="Yes", activities=(activity,)))
        duration = _truth_duration(spans)
        cases.append(Case(f"How long was the user {name}?", "duration",
                          lambda a, d=duration: numeric_correct(a.answer, d, NUMERIC_TOLERANCE_FRACTION, NUMERIC_TOLERANCE_FLOOR_S),
                          spans, f"expect ~{duration:.0f} s",
                          expected_value=duration, activities=(activity,)))
        count = _truth_count(spans)
        cases.append(Case(f"How many times did the user {name}?", "count",
                          lambda a, c=count: count_correct(a.answer, c), spans, f"expect {count}",
                          expected_value=float(count), activities=(activity,)))
        onset = _truth_onset(spans)
        if onset is not None:
            cases.append(Case(f"When did the user begin {name}?", "grounding",
                              lambda a, o=onset: numeric_correct(a.answer, o, 0.0, ONSET_TOLERANCE_S),
                              spans[:1], f"expect onset ~{onset:.0f} s",
                              expected_value=onset, activities=(activity,)))
        if spans:
            middle = (spans[0][0] + spans[0][1]) / 2
            cases.append(Case(f"What activity was the user doing at {middle:.0f} seconds?", "identification",
                              lambda a, n=name: categorical_correct(a.answer, n), spans[:1], f"expect {name}",
                              expected_label=name, activities=(activity,)))

    for activity in absent:
        name = config.DISPLAY_NAMES[activity]
        cases.append(Case(f"Did the user {name}?", "verification",
                          lambda a: categorical_correct(a.answer, "No"), [], "expect No",
                          expected_label="No", activities=(activity,)))

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
            expected_label=winner,
            activities=(first, second),
        ))
        cases.append(Case("What activity is the user performing?", "identification",
                          lambda a, w=config.DISPLAY_NAMES[ranked[0]]: categorical_correct(a.answer, w),
                          truth_intervals(labels, epoch_ts, ranked[0]), f"expect {config.DISPLAY_NAMES[ranked[0]]}",
                          expected_label=config.DISPLAY_NAMES[ranked[0]], activities=(ranked[0],)))

    resting = merge(truth_intervals(labels, epoch_ts, "lying_down") + truth_intervals(labels, epoch_ts, "sitting"))
    if resting:
        cases.append(Case("Was the user resting for a prolonged period?", "open_world",
                          lambda a: categorical_correct(a.answer, "Yes"), resting, "expect Yes",
                          expected_label="Yes", activities=("lying_down", "sitting")))
    active = merge(
        truth_intervals(labels, epoch_ts, "walking")
        + truth_intervals(labels, epoch_ts, "running")
        + truth_intervals(labels, epoch_ts, "bicycling")
    )
    cases.append(Case("Was the user doing anything strenuous?", "open_world",
                      lambda a, e=("Yes" if active else "No"): categorical_correct(a.answer, e),
                      active, f"expect {'Yes' if active else 'No'}",
                      expected_label=("Yes" if active else "No"),
                      activities=("walking", "running", "bicycling")))
    cases.extend(open_world_cases(labels, epoch_ts))
    return cases


# The two probes above are answerable from the rules alone and their reference is
# almost always Yes, so they measure neither open-world phrasing nor reasoning. The
# four below are worded so that `parse_question` cannot resolve them to a structured
# operation, and each one's reference is derived from a threshold chosen to divide
# the cohort rather than from mere presence.
VIGOROUS = ("running", "bicycling")
ACTIVE = ("walking", "running", "bicycling")
DEMANDING_ACTIVE_FRACTION = 0.10


def open_world_cases(labels: np.ndarray, epoch_ts: np.ndarray) -> list[Case]:
    """Open-world probes with balanced references that the rules cannot shortcut."""
    spans = {a: truth_intervals(labels, epoch_ts, a) for a in config.ACTIVITIES}
    duration = {a: _truth_duration(spans[a]) for a in config.ACTIVITIES}
    total = sum(duration.values())
    active_spans = merge(sum((spans[a] for a in ACTIVE), []))

    def probe(question: str, truth: bool, evidence: list[tuple[float, float]],
              activities: tuple[str, ...]) -> Case:
        expected = "Yes" if truth else "No"
        return Case(question, "open_world",
                    lambda a, e=expected: categorical_correct(a.answer, e),
                    evidence, f"expect {expected}",
                    expected_label=expected, activities=activities)

    cases = [
        # Exertion beyond walking, which presence of any active class does not imply.
        probe("Did the user get their heart rate up at any stage?",
              any(spans[a] for a in VIGOROUS),
              merge(sum((spans[a] for a in VIGOROUS), [])),
              VIGOROUS),
        # A judgement about the day as a whole, not about one activity.
        probe("Would you call this a physically demanding day for the user?",
              bool(total) and sum(duration[a] for a in ACTIVE) / total >= DEMANDING_ACTIVE_FRACTION,
              active_spans,
              ACTIVE),
        # Requires comparing two halves of the recording, which no handler supports.
        probe("Was the user busier earlier in the recording than later on?",
              _busier_first_half(labels, epoch_ts),
              active_spans,
              ACTIVE),
        # Compares two resting classes through wording that names neither directly.
        probe("Did the user spend the bigger share of the recording flat out "
              "rather than upright in a seat?",
              duration["lying_down"] > duration["sitting"],
              spans["lying_down"] if duration["lying_down"] > duration["sitting"] else spans["sitting"],
              ("lying_down", "sitting")),
    ]
    return cases


def _busier_first_half(labels: np.ndarray, epoch_ts: np.ndarray) -> bool:
    """Whether more active windows fall in the first half of the recording."""
    order = np.argsort(epoch_ts)
    ordered = np.asarray(labels)[order]
    half = len(ordered) // 2
    if half == 0:
        return False
    first = sum(1 for label in ordered[:half] if label in ACTIVE)
    second = sum(1 for label in ordered[half:] if label in ACTIVE)
    return first > second


def predicted_spans(answer) -> list[tuple[float, float]]:
    return [(i["start_s"], i["end_s"]) for i in (answer.intervals or [])]


def cited_interval_holds_activity(
    spans: list[tuple[float, float]],
    labels: np.ndarray,
    epoch_ts: np.ndarray,
    activities: tuple[str, ...],
) -> bool | None:
    """Does a cited interval actually contain the activity the answer names?

    This is the cheap grounding check: it needs only the per-window labels, not
    reference intervals, so it stays meaningful even where the merge rule has
    smeared the interval boundaries. Returns None when the question cites nothing
    or names no activity, so those cases are excluded rather than counted wrong.
    """
    if not spans or not activities:
        return None
    order = np.argsort(epoch_ts)
    labels, epoch_ts = np.asarray(labels)[order], np.asarray(epoch_ts)[order]
    start_epoch = float(epoch_ts[0])
    wanted = set(activities)
    for span_start, span_end in spans:
        for label, timestamp in zip(labels, epoch_ts):
            if label not in wanted:
                continue
            window_start = float(timestamp) - start_epoch
            window_end = window_start + config.WINDOW_DURATION_S
            if min(span_end, window_end) > max(span_start, window_start):
                return True
    return False


def answer_cases(
    timeline: Timeline, labels: np.ndarray, epoch_ts: np.ndarray, use_slm: bool = False
) -> list[tuple[Case, Any]]:
    """Answer every generated case once.

    Scoring and the strictness sweep both need the same answers, and with the
    language model enabled each one costs seconds, so they are computed once here
    and handed to both rather than produced twice.
    """
    return [
        (case, answer_question(case.question, timeline, use_slm=use_slm))
        for case in build_cases(labels, epoch_ts)
    ]


def evaluate_recording(
    timeline: Timeline,
    labels: np.ndarray,
    epoch_ts: np.ndarray,
    use_slm: bool = False,
    iou_threshold: float = IOU_THRESHOLD,
    answers: list[tuple[Case, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Score every generated case against one predicted timeline."""
    if answers is None:
        answers = answer_cases(timeline, labels, epoch_ts, use_slm=use_slm)
    results = []
    for case, answer in answers:
        correct = bool(case.check(answer))
        spans = predicted_spans(answer)
        iou = interval_iou(spans, case.truth_spans)
        temporal = temporal_scores(spans, case.truth_spans)

        source = config.evidence_source()
        cites = answer.timestamps != "N/A"
        source_ok = (not cites) or (
            answer.modality == source["modality"] and answer.channels == source["channels"]
        )
        grounded = correct and source_ok and (iou >= iou_threshold if case.truth_spans else not cites)

        predicted = numeric_value(answer.answer) if case.kind in NUMERIC_KINDS else None
        error = (
            abs(predicted - case.expected_value)
            if predicted is not None and case.expected_value is not None
            else None
        )

        results.append({
            "question": case.question,
            "question_type": case.kind,
            "expected": case.description,
            "answer": answer.answer,
            "answer_correct": correct,
            "evidence_iou": round(iou, 4),
            "evidence_matched_iou": round(matched_iou(spans, case.truth_spans), 4),
            "temporal_precision": round(temporal["temporal_precision"], 4),
            "temporal_recall": round(temporal["temporal_recall"], 4),
            "temporal_f1": round(temporal["temporal_f1"], 4),
            "grounded_correct": bool(grounded),
            "cited_intervals": len(spans),
            # Kept so the explanation rubric can be applied afterwards without
            # re-running the pipeline.
            "explanation": answer.explanation,
            "cited_timestamps": answer.timestamps,
            # Machine-readable reference and response, so the summary can report
            # macro-F1 over the classes and the size of each numeric error.
            "expected_label": case.expected_label,
            "predicted_label": answer.answer.strip().lower() if case.expected_label else None,
            "expected_value": case.expected_value,
            "predicted_value": predicted,
            "absolute_error": error,
            "cited_holds_activity": cited_interval_holds_activity(
                spans, labels, epoch_ts, case.activities
            ),
        })
    return results


def categorical_scores(rows: list[dict[str, Any]]) -> dict[str, float] | None:
    """Macro-F1 and balanced accuracy over the answer classes of one question type.

    Accuracy alone rewards a model that always names the majority class. An answer
    that matches no reference class forms a class of its own, which costs recall on
    the class it should have been and precision nowhere, exactly as it should.
    """
    from sklearn.metrics import f1_score, recall_score

    pairs = [
        (r["expected_label"].strip().lower(), (r["predicted_label"] or "").strip())
        for r in rows
        if r.get("expected_label")
    ]
    if not pairs:
        return None
    truth = [expected for expected, _ in pairs]
    predicted = [actual for _, actual in pairs]
    classes = sorted(set(truth))
    return {
        "macro_f1": float(f1_score(truth, predicted, labels=classes, average="macro", zero_division=0)),
        "balanced_accuracy": float(
            recall_score(truth, predicted, labels=classes, average="macro", zero_division=0)
        ),
    }


def binary_scores(rows: list[dict[str, Any]], positive: str = "yes") -> dict[str, float] | None:
    """Precision, recall, F1 and specificity on the positive class.

    Most verification questions expect No, because most activities are absent from
    any one recording. Plain accuracy therefore hides a model biased toward No;
    recall on Yes and specificity together expose it.
    """
    pairs = [
        (r["expected_label"].strip().lower(), (r["predicted_label"] or "").strip())
        for r in rows
        if r.get("expected_label")
    ]
    if not pairs:
        return None
    tp = sum(1 for e, p in pairs if e == positive and p == positive)
    fp = sum(1 for e, p in pairs if e != positive and p == positive)
    fn = sum(1 for e, p in pairs if e == positive and p != positive)
    tn = sum(1 for e, p in pairs if e != positive and p != positive)

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "positive_class": positive,
        "support_positive": tp + fn,
        "support_negative": tn + fp,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "specificity": tn / (tn + fp) if tn + fp else 0.0,
    }


def numeric_error(rows: list[dict[str, Any]], kind: str) -> dict[str, float] | None:
    """Mean absolute error, and percentage error where the reference is a duration.

    A pass/fail against a tolerance says nothing about how wrong a miss was; a
    31-second error and a 3000-second error score alike without this. Answers that
    carry no number at all are counted separately rather than silently dropped.
    """
    scored = [r for r in rows if r.get("expected_value") is not None]
    if not scored:
        return None
    errors = [r["absolute_error"] for r in scored if r["absolute_error"] is not None]
    summary: dict[str, float] = {
        "n_scored": len(errors),
        "n_unparseable": len(scored) - len(errors),
        "mean_absolute_error": float(np.mean(errors)) if errors else float("nan"),
        "median_absolute_error": float(np.median(errors)) if errors else float("nan"),
    }
    if kind in PERCENTAGE_ERROR_KINDS:
        percentages = [
            r["absolute_error"] / abs(r["expected_value"]) * 100
            for r in scored
            if r["absolute_error"] is not None and r["expected_value"]
        ]
        if percentages:
            summary["mean_absolute_percentage_error"] = float(np.mean(percentages))
    return summary


def grounding_precision(rows: list[dict[str, Any]]) -> dict[str, float] | None:
    """Fraction of cited intervals that really contain the activity the answer names."""
    judged = [r["cited_holds_activity"] for r in rows if r.get("cited_holds_activity") is not None]
    if not judged:
        return None
    return {"n_cited": len(judged), "grounding_precision": float(np.mean(judged))}


def summarise(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_type: dict[str, list[dict]] = defaultdict(list)
    for row in results:
        by_type[row["question_type"]].append(row)

    per_type = {}
    for kind, rows in sorted(by_type.items()):
        stats: dict[str, Any] = {
            "count": len(rows),
            "answer_accuracy": float(np.mean([r["answer_correct"] for r in rows])),
            "grounded_accuracy": float(np.mean([r["grounded_correct"] for r in rows])),
            "mean_evidence_iou": float(np.mean([r["evidence_iou"] for r in rows])),
            "mean_matched_iou": float(np.mean([r.get("evidence_matched_iou", 0.0) for r in rows])),
            "temporal": {
                "precision": float(np.mean([r.get("temporal_precision", 0.0) for r in rows])),
                "recall": float(np.mean([r.get("temporal_recall", 0.0) for r in rows])),
                "f1": float(np.mean([r.get("temporal_f1", 0.0) for r in rows])),
            },
        }
        if kind in CATEGORICAL_KINDS:
            stats.update(categorical_scores(rows) or {})
        if kind in NUMERIC_KINDS:
            stats["error"] = numeric_error(rows, kind)
        stats["grounding"] = grounding_precision(rows)
        per_type[kind] = stats

    summary = {
        "case_count": len(results),
        "overall_qa_accuracy_macro": float(np.mean([v["answer_accuracy"] for v in per_type.values()])),
        "overall_grounded_accuracy_macro": float(np.mean([v["grounded_accuracy"] for v in per_type.values()])),
        "overall_qa_accuracy_micro": float(np.mean([r["answer_correct"] for r in results])),
        "by_question_type": per_type,
    }

    macro_f1s = [v["macro_f1"] for v in per_type.values() if "macro_f1" in v]
    if macro_f1s:
        summary["overall_macro_f1"] = float(np.mean(macro_f1s))
    if "verification" in by_type:
        summary["verification_binary"] = binary_scores(by_type["verification"])
    overall_grounding = grounding_precision(results)
    if overall_grounding:
        summary["overall_grounding_precision"] = overall_grounding["grounding_precision"]
    return summary


def strictness_sweep(
    timeline: Timeline,
    labels: np.ndarray,
    epoch_ts: np.ndarray,
    use_slm: bool = False,
    answers: list[tuple[Case, Any]] | None = None,
) -> dict[str, Any]:
    """Accuracy as the IoU threshold and the numeric tolerance are tightened."""
    if answers is None:
        answers = answer_cases(timeline, labels, epoch_ts, use_slm=use_slm)

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
            expected = case.expected_value
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

            answers = answer_cases(timeline, truth, epoch_ts, use_slm=args.use_slm)
            results = evaluate_recording(timeline, truth, epoch_ts, answers=answers)
            for row in results:
                row["user"] = user_id
                row["fold"] = fold_index
            all_results.extend(results)
            sweeps.append(strictness_sweep(timeline, truth, epoch_ts, answers=answers))

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
    print(
        f"{'question type':<18}{'n':>6}{'answer acc':>13}{'macro F1':>11}{'bal acc':>10}"
        f"{'grounded acc':>15}{'mean IoU':>10}{'match IoU':>11}{'temp P':>8}{'temp R':>8}"
        f"{'temp F1':>9}{'MAE':>11}{'ground prec':>13}"
    )

    def cell(value: float | None, width: int, spec: str = ".3f") -> str:
        return f"{'-':>{width}}" if value is None else f"{value:>{width}{spec}}"

    for kind, stats in overall["by_question_type"].items():
        error = stats.get("error") or {}
        grounding = stats.get("grounding") or {}
        temporal = stats.get("temporal") or {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        print(
            f"{kind:<18}{stats['count']:>6}{stats['answer_accuracy']:>13.3f}"
            f"{cell(stats.get('macro_f1'), 11)}{cell(stats.get('balanced_accuracy'), 10)}"
            f"{stats['grounded_accuracy']:>15.3f}{stats['mean_evidence_iou']:>10.3f}"
            f"{stats['mean_matched_iou']:>11.3f}{temporal['precision']:>8.3f}"
            f"{temporal['recall']:>8.3f}{temporal['f1']:>9.3f}"
            f"{cell(error.get('mean_absolute_error'), 11, '.1f')}"
            f"{cell(grounding.get('grounding_precision'), 13)}"
        )
    print(f"\noverall QA accuracy (macro over types): {overall['overall_qa_accuracy_macro']:.3f}")
    if "overall_macro_f1" in overall:
        print(f"overall macro-F1 (categorical types)  : {overall['overall_macro_f1']:.3f}")
    print(f"overall grounded accuracy (macro)     : {overall['overall_grounded_accuracy_macro']:.3f}")
    print(f"overall QA accuracy (micro)           : {overall['overall_qa_accuracy_micro']:.3f}")
    if overall.get("overall_grounding_precision") is not None:
        print(f"overall grounding precision           : {overall['overall_grounding_precision']:.3f}")

    binary = overall.get("verification_binary")
    if binary:
        print(
            f"\nverification, positive class '{binary['positive_class']}' "
            f"(n+={binary['support_positive']}, n-={binary['support_negative']}): "
            f"precision {binary['precision']:.3f}  recall {binary['recall']:.3f}  "
            f"F1 {binary['f1']:.3f}  specificity {binary['specificity']:.3f}"
        )

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
            "duration": f"within max({NUMERIC_TOLERANCE_FLOOR_S:.0f} s, {NUMERIC_TOLERANCE_FRACTION:.0%}) of the reference",
            "grounding": f"onset within +/-{ONSET_TOLERANCE_S:.0f} s of the reference",
            "count": f"within max(+/-{COUNT_TOLERANCE} bouts, {COUNT_TOLERANCE_FRACTION:.0%})",
            "grounded": f"answer correct AND evidence IoU >= {IOU_THRESHOLD} AND modality/channels match",
            "evidence_iou": "intersection over union between the cited and reference interval sets, "
                            "pooled across intervals",
            "matched_iou": "each cited interval matched one-to-one to a reference interval by "
                           "overlap, best first, and their IoUs averaged over the larger set, so "
                           "missed and spurious intervals both cost",
            "temporal_precision_recall_f1": "overlap over cited length, overlap over reference "
                                            "length, and their F1, which say which way a citation "
                                            "is wrong where a single IoU cannot",
            "explanation_rubric": "three criteria scored 1-5 by `python -m asqa.rubric`, with "
                                  "Krippendorff alpha and quadratic-weighted kappa between graders; "
                                  "embedding similarity to a reference explanation is reported "
                                  "beside it as a secondary proxy only",
            "macro_f1/balanced_accuracy": "over the answer classes of each categorical type, "
                                          "since accuracy alone favours the majority class",
            "verification_binary": "precision, recall, F1 and specificity on the positive class, "
                                   "since most verification questions expect No",
            "error": "mean and median absolute error of the numeric answers, plus mean absolute "
                     "percentage error for durations and onsets",
            "grounding_precision": "fraction of cited intervals that contain the named activity "
                                   "according to the per-window ground-truth labels",
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
