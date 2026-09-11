"""Explanation quality: a fixed 1-5 rubric, grader agreement, and a similarity proxy.

Correctness scoring says whether an answer is right, not whether the reasoning given
for it holds up. An explanation can reach the right answer while citing features that
were never measured, and a pass/fail check cannot tell the two apart. Every
explanation is therefore graded on three criteria, each 1-5 and each worded once in
`slm_worker.RUBRIC_CRITERIA` so that a model grader and a human grader read the same
sheet.

Grading may be done by the language model or by people. Where several graders are
used the spread between them is reported alongside the mean, since a mean rubric
score with no agreement figure says nothing about how repeatable the grade is.
Embedding similarity to a reference explanation is also computed, but stays secondary
to the rubric: it rewards surface overlap with the reference wording and would favour
an explanation that echoes the template without reasoning at all.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from asqa import config

CRITERIA = ("cites_features", "features_support", "plausible")

SCALE_MIN, SCALE_MAX = 1, 5

# One worker launch loads the model once, so explanations are graded in blocks
# rather than one process per explanation.
BATCH_SIZE = 16

WORKER_TIMEOUT_S = 3600


def _call_worker(request: dict, timeout: int = WORKER_TIMEOUT_S) -> dict:
    completed = subprocess.run(
        [sys.executable, "-m", "asqa.slm_worker"],
        input=json.dumps(request),
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=str(config.REPO_ROOT),
        env={**os.environ, "PYTHONPATH": str(config.REPO_ROOT)},
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"worker exited {completed.returncode}: {completed.stderr.strip()[-200:]}"
        )
    response = json.loads(completed.stdout)
    if not response.get("ok"):
        raise RuntimeError(response.get("error", "unknown worker failure"))
    return response["payload"]


def gradable(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rows carrying an explanation worth grading.

    Rows whose explanation is the fixed unavailable-or-out-of-scope message carry no
    reasoning to judge, so grading them would report on a template rather than on the
    system, and they are left out rather than scored low.
    """
    keep = []
    for row in rows:
        explanation = (row.get("explanation") or "").strip()
        if not explanation or explanation in {"N/A", ""}:
            continue
        if explanation.startswith("This question falls outside"):
            continue
        keep.append(row)
    return keep


def grade_with_model(
    rows: list[dict[str, Any]], passes: int = 1, temperature: float = 0.0
) -> list[dict[str, list[int]]]:
    """Grade each explanation `passes` times, returning one score list per criterion.

    A second pass at temperature 0 would repeat the first exactly, so repeated passes
    sample instead; the spread between them is the model grader's own consistency,
    which is the closest analogue to agreement between two people.
    """
    items = [
        {
            "question": row["question"],
            "evidence": row.get("cited_timestamps", "N/A"),
            "answer": row["answer"],
            "explanation": row["explanation"],
        }
        for row in rows
    ]

    per_pass: list[list[dict[str, int]]] = []
    for index in range(passes):
        graded: list[dict[str, int]] = []
        for start in range(0, len(items), BATCH_SIZE):
            block = items[start : start + BATCH_SIZE]
            payload = _call_worker({
                "mode": "judge",
                "items": block,
                "temperature": 0.0 if index == 0 else temperature,
            })
            graded.extend(payload["grades"])
            print(f"  pass {index + 1}: graded {len(graded)}/{len(items)}", flush=True)
        per_pass.append(graded)

    return [
        {criterion: [pass_[i][criterion] for pass_ in per_pass] for criterion in CRITERIA}
        for i in range(len(items))
    ]


def load_human_grades(path: Path) -> dict[str, dict[str, list[int]]]:
    """Read human grades from a CSV of question,grader,<criterion>... rows."""
    grades: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    with Path(path).open(encoding="utf-8") as handle:
        for record in csv.DictReader(handle):
            key = record["question"].strip()
            for criterion in CRITERIA:
                if record.get(criterion):
                    grades[key][criterion].append(int(record[criterion]))
    return {k: dict(v) for k, v in grades.items()}


def quadratic_weighted_kappa(a: list[int], b: list[int]) -> float | None:
    """Cohen's kappa with quadratic weights, for two graders on an ordinal scale.

    Quadratic weighting is what makes this appropriate here: on a 1-5 scale a 4
    against a 5 is near-agreement and a 1 against a 5 is not, and unweighted kappa
    would treat both as a plain mismatch.
    """
    if len(a) != len(b) or not a:
        return None
    size = SCALE_MAX - SCALE_MIN + 1
    observed = np.zeros((size, size))
    for x, y in zip(a, b):
        observed[x - SCALE_MIN, y - SCALE_MIN] += 1
    observed /= observed.sum()

    rows = observed.sum(axis=1)
    columns = observed.sum(axis=0)
    expected = np.outer(rows, columns)

    indices = np.arange(size)
    weights = (indices[:, None] - indices[None, :]) ** 2 / (size - 1) ** 2

    denominator = float((weights * expected).sum())
    if denominator == 0:
        # Every grader used one category throughout, so chance agreement is total
        # and kappa is undefined rather than perfect.
        return None
    return float(1 - (weights * observed).sum() / denominator)


def krippendorff_alpha(matrix: list[list[int]]) -> float | None:
    """Krippendorff's alpha with an interval difference function.

    Kappa handles two graders; alpha generalises to any number and is what the
    agreement figure uses when three or more graders are present.
    """
    # A unit is one explanation and its grades, so the disagreement within a unit is
    # measured across that row of graders, not down a grader's column.
    units = [[v for v in row if v is not None] for row in matrix]
    units = [u for u in units if len(u) >= 2]
    if not units:
        return None

    observed_pairs = []
    for unit in units:
        weight = len(unit) - 1
        for i, x in enumerate(unit):
            for j, y in enumerate(unit):
                if i != j:
                    observed_pairs.append(((x - y) ** 2) / weight)
    values = [v for unit in units for v in unit]
    if len(values) < 2:
        return None

    observed = float(np.sum(observed_pairs)) / len(values)
    expected_pairs = [
        (x - y) ** 2 for i, x in enumerate(values) for j, y in enumerate(values) if i != j
    ]
    expected = float(np.sum(expected_pairs)) / (len(values) * (len(values) - 1))
    if expected == 0:
        return None
    return float(1 - observed / expected)


def agreement(graded: list[dict[str, list[int]]]) -> dict[str, Any]:
    """Agreement between graders, per criterion and pooled."""
    n_graders = max((len(v) for row in graded for v in row.values()), default=0)
    if n_graders < 2:
        return {"n_graders": n_graders, "note": "a single grader, so no agreement to report"}

    report: dict[str, Any] = {"n_graders": n_graders}
    for criterion in CRITERIA:
        matrix = [row[criterion] for row in graded if len(row.get(criterion, [])) == n_graders]
        if not matrix:
            continue
        entry: dict[str, Any] = {"krippendorff_alpha": krippendorff_alpha(matrix)}
        if n_graders == 2:
            entry["quadratic_weighted_kappa"] = quadratic_weighted_kappa(
                [row[0] for row in matrix], [row[1] for row in matrix]
            )
        report[criterion] = entry
    return report


def reference_explanation(row: dict[str, Any]) -> str:
    """What a faithful explanation of this answer would have to say.

    Built from the answer and the evidence the pipeline actually returned, so
    similarity measures how closely the explanation tracks its own evidence.
    """
    return (
        f"The answer is {row['answer']}, supported by the sensor evidence at "
        f"{row.get('cited_timestamps', 'N/A')}, measured from the body acceleration "
        f"and gyroscope statistics of those intervals."
    )


def similarity_proxy(rows: list[dict[str, Any]]) -> list[float]:
    """Cosine similarity between each explanation and its reference."""
    pairs = [[row["explanation"], reference_explanation(row)] for row in rows]
    similarities: list[float] = []
    for start in range(0, len(pairs), BATCH_SIZE):
        payload = _call_worker({"mode": "embed", "pairs": pairs[start : start + BATCH_SIZE]})
        similarities.extend(payload["similarities"])
    return similarities


def summarise(
    rows: list[dict[str, Any]],
    graded: list[dict[str, list[int]]],
    similarities: list[float] | None = None,
) -> dict[str, Any]:
    """Mean rubric score overall, per criterion and per question type."""
    means = [
        {criterion: float(np.mean(scores)) for criterion, scores in row.items()}
        for row in graded
    ]

    per_criterion = {
        criterion: float(np.mean([m[criterion] for m in means if criterion in m]))
        for criterion in CRITERIA
    }
    overall = float(np.mean([np.mean(list(m.values())) for m in means])) if means else 0.0

    by_type: dict[str, list[float]] = defaultdict(list)
    for row, mean in zip(rows, means):
        by_type[row["question_type"]].append(float(np.mean(list(mean.values()))))

    report: dict[str, Any] = {
        "n_explanations": len(rows),
        "scale": f"{SCALE_MIN}-{SCALE_MAX}",
        "criteria": _worker_criteria(),
        "mean_rubric_score": overall,
        "mean_by_criterion": per_criterion,
        "mean_by_question_type": {k: float(np.mean(v)) for k, v in sorted(by_type.items())},
        "agreement": agreement(graded),
    }
    if similarities:
        report["embedding_similarity_proxy"] = {
            "mean": float(np.mean(similarities)),
            "median": float(np.median(similarities)),
            "note": "secondary to the rubric; rewards surface overlap with the reference",
        }
    return report


def _worker_criteria() -> dict[str, str]:
    from asqa.slm_worker import RUBRIC_CRITERIA

    return RUBRIC_CRITERIA


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="qa_cv.json",
                        help="Evaluation JSON in outputs/evaluation/ to grade.")
    parser.add_argument("--passes", type=int, default=2,
                        help="Model grading passes; two or more yield an agreement figure.")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature for passes after the first.")
    parser.add_argument("--question-type", default=None,
                        help="Grade only one question type, e.g. open_world.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--human-grades", type=Path, default=None,
                        help="CSV of question,grader,<criterion>... to use instead of the model.")
    parser.add_argument("--no-similarity", action="store_true")
    parser.add_argument("--output", default="explanation_rubric.json")
    args = parser.parse_args()

    source = config.EVALUATION_DIR / args.results
    data = json.loads(source.read_text(encoding="utf-8"))
    rows = gradable(data["results"])
    if args.question_type:
        rows = [r for r in rows if r["question_type"] == args.question_type]
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        raise SystemExit(
            "No gradable explanations found. Re-run the evaluation with --use-slm, "
            "since without it the open-world questions carry no reasoning to judge."
        )

    print(f"Grading {len(rows)} explanations from {source.name}")
    if args.human_grades:
        human = load_human_grades(args.human_grades)
        graded = [human.get(r["question"].strip(), {}) for r in rows]
        keep = [i for i, g in enumerate(graded) if g]
        rows, graded = [rows[i] for i in keep], [graded[i] for i in keep]
        print(f"  matched human grades for {len(rows)} explanations")
    else:
        graded = grade_with_model(rows, passes=args.passes, temperature=args.temperature)

    similarities = None if args.no_similarity else similarity_proxy(rows)
    report = summarise(rows, graded, similarities)

    print(f"\nmean rubric score (1-5): {report['mean_rubric_score']:.2f}")
    for criterion, value in report["mean_by_criterion"].items():
        print(f"  {criterion:<18} {value:.2f}")
    print("\nby question type:")
    for kind, value in report["mean_by_question_type"].items():
        print(f"  {kind:<18} {value:.2f}")
    agreement_report = report["agreement"]
    if agreement_report.get("n_graders", 0) >= 2:
        print(f"\nagreement over {agreement_report['n_graders']} graders:")
        for criterion in CRITERIA:
            entry = agreement_report.get(criterion)
            if not entry:
                continue
            alpha = entry.get("krippendorff_alpha")
            kappa = entry.get("quadratic_weighted_kappa")
            parts = [f"alpha {alpha:.3f}" if alpha is not None else "alpha n/a"]
            if kappa is not None:
                parts.append(f"quadratic-weighted kappa {kappa:.3f}")
            print(f"  {criterion:<18} {'  '.join(parts)}")
    else:
        print(f"\n{agreement_report.get('note', '')}")
    if similarities:
        print(f"\nembedding similarity to reference (secondary): "
              f"mean {report['embedding_similarity_proxy']['mean']:.3f}")

    report["graded"] = [
        {"question": r["question"], "question_type": r["question_type"],
         "explanation": r["explanation"], "scores": g}
        for r, g in zip(rows, graded)
    ]
    config.EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    destination = config.EVALUATION_DIR / args.output
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
