#!/usr/bin/env python3
"""Generate the five required evaluation figures from existing artifacts.

Figures:
  1. Accuracy by question type (bar chart)
  2. Confusion matrix heatmap with precision/recall/F1
  3. Accuracy vs strictness (sweep tolerance and IoU)
  4. Accuracy vs overhead (model size / latency trade-off)
  5. Robustness under noise and dropout
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# ── Style ────────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=1.1)
PALETTE = sns.color_palette("Set2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/figures"))
    parser.add_argument("--qa-results", type=Path, default=Path("artifacts/evaluation/qa_results_0A986.json"))
    parser.add_argument("--qa-cases", type=Path, default=Path("artifacts/evaluation/qa_cases_0A986.json"))
    parser.add_argument("--predicted-timeline", type=Path, default=Path("artifacts/timelines/0A986513-7828-4D53-AA1F-E02D6DF9561B.json"))
    parser.add_argument("--only", nargs="*", help="Generate only these figure numbers (1-5). Default: all.")
    return parser.parse_args()


# ── Figure 1: Accuracy by question type ──────────────────────────────────────

def fig1_accuracy_by_question_type(qa_results: dict[str, Any], output_dir: Path) -> None:
    by_type = qa_results["by_question_type"]
    types = list(by_type.keys())
    answer_acc = [by_type[t]["answer_accuracy"] for t in types]
    grounded_acc = [by_type[t]["grounded_accuracy"] for t in types]

    x = np.arange(len(types))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - width / 2, answer_acc, width, label="Answer Accuracy", color=PALETTE[0])
    bars2 = ax.bar(x + width / 2, grounded_acc, width, label="Grounded Accuracy", color=PALETTE[1])

    ax.set_xlabel("Question Type")
    ax.set_ylabel("Accuracy")
    ax.set_title("QA Accuracy by Question Type (RF-Full, Test User)")
    ax.set_xticks(x)
    ax.set_xticklabels(types, rotation=25, ha="right")
    ax.set_ylim(0, 1.1)
    ax.legend()

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{bar.get_height():.0%}", ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{bar.get_height():.0%}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_dir / "fig1_accuracy_by_question_type.png", dpi=200)
    plt.close(fig)
    print(f"  → fig1_accuracy_by_question_type.png")


# ── Figure 2: Confusion matrix heatmap ───────────────────────────────────────

def _load_confusion_matrices() -> dict[str, tuple[np.ndarray, list[str]]]:
    """Load confusion matrices from all available model artifact dirs."""
    matrices = {}
    for tag, dir_path in [("RF-Full", Path("artifacts/baseline")),
                          ("RF-Medium", Path("artifacts/rf_medium")),
                          ("RF-Tiny", Path("artifacts/rf_tiny")),
                          ("CNN", Path("artifacts/cnn"))]:
        cm_path = dir_path / "test_confusion_matrix.csv"
        if cm_path.exists():
            with cm_path.open(newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                header = next(reader)
                labels = header[1:]  # first col is true_activity
                rows = []
                for row in reader:
                    rows.append([int(v) for v in row[1:]])
            matrices[tag] = (np.array(rows), labels)
    return matrices


def fig2_confusion_matrix(output_dir: Path) -> None:
    matrices = _load_confusion_matrices()
    if not matrices:
        print("  ⚠ No confusion matrices found, skipping fig2")
        return

    # Use the best available model (RF-Full typically)
    for preferred in ["CNN", "RF-Full", "RF-Medium", "RF-Tiny"]:
        if preferred in matrices:
            tag = preferred
            break
    else:
        tag = next(iter(matrices))

    cm, labels = matrices[tag]
    # Compute per-class metrics
    per_class_precision = np.zeros(len(labels))
    per_class_recall = np.zeros(len(labels))
    per_class_f1 = np.zeros(len(labels))
    for i in range(len(labels)):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        per_class_precision[i] = tp / (tp + fp) if (tp + fp) > 0 else 0
        per_class_recall[i] = tp / (tp + fn) if (tp + fn) > 0 else 0
        if per_class_precision[i] + per_class_recall[i] > 0:
            per_class_f1[i] = 2 * per_class_precision[i] * per_class_recall[i] / (per_class_precision[i] + per_class_recall[i])

    # Normalise for display
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm.astype(float), row_sums, where=row_sums > 0, out=np.zeros_like(cm, dtype=float))

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="YlOrRd",
                xticklabels=labels, yticklabels=labels, ax=ax,
                cbar_kws={"label": "Row-normalised proportion"})
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix — {tag} (Test Set)")

    # Add per-class metrics as side text
    metric_text = "\n".join(
        f"{l[:8]:>8s}  P={p:.2f}  R={r:.2f}  F1={f:.2f}"
        for l, p, r, f in zip(labels, per_class_precision, per_class_recall, per_class_f1)
    )
    fig.text(0.98, 0.5, metric_text, fontsize=8, family="monospace",
             va="center", ha="left", transform=fig.transFigure,
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    fig.tight_layout(rect=[0, 0, 0.78, 1])
    fig.savefig(output_dir / "fig2_confusion_matrix.png", dpi=200)
    plt.close(fig)
    print(f"  → fig2_confusion_matrix.png")


# ── Figure 3: Accuracy vs strictness ────────────────────────────────────────

def fig3_accuracy_vs_strictness(qa_cases_path: Path, predicted_timeline_path: Path, output_dir: Path) -> None:
    from qa import execute

    if not qa_cases_path.exists() or not predicted_timeline_path.exists():
        print("  ⚠ QA cases or predicted timeline not found, skipping fig3")
        return

    cases = json.loads(qa_cases_path.read_text(encoding="utf-8"))["cases"]
    timeline = json.loads(predicted_timeline_path.read_text(encoding="utf-8"))

    tolerances = [5, 10, 20, 40, 80]
    iou_thresholds = [0.1, 0.25, 0.5, 0.75]

    import re
    def numeric_value(answer: str) -> float | None:
        match = re.search(r"-?\d+(?:\.\d+)?", answer)
        return float(match.group()) if match else None

    def merge_intervals(intervals):
        merged = []
        for start, end in sorted((float(i["start_s"]), float(i["end_s"])) for i in intervals):
            if not merged or start > merged[-1][1]:
                merged.append((start, end))
            else:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        return merged

    def interval_iou(pred, truth):
        pred_m, truth_m = merge_intervals(pred), merge_intervals(truth)
        if not pred_m and not truth_m:
            return 1.0
        if not pred_m or not truth_m:
            return 0.0
        intersection = sum(max(0, min(a1, b1) - max(a0, b0)) for a0, a1 in pred_m for b0, b1 in truth_m)
        pred_len = sum(e - s for s, e in pred_m)
        truth_len = sum(e - s for s, e in truth_m)
        return intersection / (pred_len + truth_len - intersection)

    # Pre-compute predictions
    predictions = [execute(case["question"], timeline) for case in cases]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Panel A: Grounded accuracy vs IoU threshold (fixed tolerance=20s)
    grounded_by_iou = []
    for iou_thresh in iou_thresholds:
        correct = 0
        for case, pred in zip(cases, predictions):
            expected = case["ground_truth"]
            qt = case["question_type"]
            if qt in {"duration", "grounding"}:
                pv, ev = numeric_value(pred["answer"]), numeric_value(expected["answer"])
                ans_ok = pv is not None and ev is not None and abs(pv - ev) <= 20
            else:
                ans_ok = pred["answer"].strip().lower() == expected["answer"].strip().lower()
            iou = interval_iou(pred["evidence"]["intervals"], expected["evidence"]["intervals"])
            same_src = (pred["evidence"]["modality"] == expected["evidence"]["modality"]
                        and pred["evidence"]["channels"] == expected["evidence"]["channels"])
            if ans_ok and iou >= iou_thresh and same_src:
                correct += 1
        grounded_by_iou.append(correct / len(cases))
    ax1.plot(iou_thresholds, grounded_by_iou, "o-", color=PALETTE[2], linewidth=2)
    ax1.set_xlabel("Evidence IoU Threshold")
    ax1.set_ylabel("Grounded Accuracy")
    ax1.set_title("Grounded Accuracy vs IoU Threshold\n(tolerance=20s)")
    ax1.set_ylim(0, max(grounded_by_iou) * 1.5 + 0.05)

    # Panel B: Answer accuracy vs numeric tolerance (for duration/grounding Qs)
    answer_by_tol = []
    for tol in tolerances:
        correct = 0
        total = 0
        for case, pred in zip(cases, predictions):
            expected = case["ground_truth"]
            qt = case["question_type"]
            if qt in {"duration", "grounding"}:
                pv, ev = numeric_value(pred["answer"]), numeric_value(expected["answer"])
                if pv is not None and ev is not None and abs(pv - ev) <= tol:
                    correct += 1
                total += 1
        answer_by_tol.append(correct / total if total > 0 else 0)
    ax2.plot(tolerances, answer_by_tol, "s-", color=PALETTE[3], linewidth=2)
    ax2.set_xlabel("Numeric Tolerance (seconds)")
    ax2.set_ylabel("Answer Accuracy (duration/grounding Qs)")
    ax2.set_title("Answer Accuracy vs Numeric Tolerance")
    ax2.set_ylim(0, max(answer_by_tol) * 1.5 + 0.05)

    fig.suptitle("Accuracy vs Evaluation Strictness", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "fig3_accuracy_vs_strictness.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  → fig3_accuracy_vs_strictness.png")


# ── Figure 4: Accuracy vs overhead ──────────────────────────────────────────

def fig4_accuracy_vs_overhead(output_dir: Path) -> None:
    benchmarks_dir = Path("artifacts/benchmarks")
    model_dirs = {
        "RF-Full": (Path("artifacts/baseline/metrics.json"), benchmarks_dir / "rf_full.json"),
        "RF-Medium": (Path("artifacts/rf_medium/metrics.json"), benchmarks_dir / "rf_medium.json"),
        "RF-Tiny": (Path("artifacts/rf_tiny/metrics.json"), benchmarks_dir / "rf_tiny.json"),
        "CNN": (Path("artifacts/cnn/metrics.json"), benchmarks_dir / "cnn.json"),
    }

    points: list[dict[str, Any]] = []
    for tag, (metrics_path, bench_path) in model_dirs.items():
        if not metrics_path.exists() or not bench_path.exists():
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        bench = json.loads(bench_path.read_text(encoding="utf-8"))
        test_acc = metrics["test"]["accuracy"]
        test_f1 = metrics["test"]["macro_f1"]
        size_mb = bench["model_size_mb"]
        latency_ms = bench["latency_ms"]["median"]
        points.append({"tag": tag, "accuracy": test_acc, "f1": test_f1,
                        "size_mb": size_mb, "latency_ms": latency_ms})

    if not points:
        print("  ⚠ No benchmark + metrics pairs found, skipping fig4")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    tags = [p["tag"] for p in points]
    accs = [p["accuracy"] for p in points]
    sizes = [p["size_mb"] for p in points]
    lats = [p["latency_ms"] for p in points]
    f1s = [p["f1"] for p in points]

    # Panel A: Accuracy vs Model Size
    scatter1 = ax1.scatter(sizes, accs, s=120, c=[PALETTE[i] for i in range(len(points))], zorder=5, edgecolors="black")
    for i, tag in enumerate(tags):
        ax1.annotate(tag, (sizes[i], accs[i]), textcoords="offset points",
                     xytext=(8, 5), fontsize=9, fontweight="bold")
    ax1.set_xlabel("Model Size (MB)")
    ax1.set_ylabel("Test Accuracy")
    ax1.set_title("Accuracy vs Model Size")

    # Panel B: Accuracy vs Latency
    scatter2 = ax2.scatter(lats, accs, s=120, c=[PALETTE[i] for i in range(len(points))], zorder=5, edgecolors="black")
    for i, tag in enumerate(tags):
        ax2.annotate(tag, (lats[i], accs[i]), textcoords="offset points",
                     xytext=(8, 5), fontsize=9, fontweight="bold")
    ax2.set_xlabel("Median Inference Latency (ms)")
    ax2.set_ylabel("Test Accuracy")
    ax2.set_title("Accuracy vs Latency")

    fig.suptitle("Accuracy–Efficiency Trade-off", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "fig4_accuracy_vs_overhead.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Also save a table
    with (output_dir / "efficiency_table.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "test_accuracy", "test_macro_f1", "size_mb", "latency_ms"])
        for p in points:
            writer.writerow([p["tag"], f"{p['accuracy']:.4f}", f"{p['f1']:.4f}", p["size_mb"], p["latency_ms"]])
    print(f"  → fig4_accuracy_vs_overhead.png + efficiency_table.csv")


# ── Figure 5: Robustness under noise / dropout ──────────────────────────────

def fig5_robustness(output_dir: Path) -> None:
    """Inject noise and dropout into test features and measure accuracy degradation."""
    from signal_features import extract_features, load_axes

    model_path = Path("artifacts/baseline/random_forest.joblib")
    index_path = Path("data/processed/raw_acc_training_index.csv")
    if not model_path.exists() or not index_path.exists():
        print("  ⚠ Model or index not found, skipping fig5")
        return

    import joblib
    saved = joblib.load(model_path)
    model = saved["model"]
    sample_rate = float(saved["sample_rate_hz"])

    with index_path.open(newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["user_id"] == "0A986513-7828-4D53-AA1F-E02D6DF9561B"]

    if not rows:
        print("  ⚠ No test user rows, skipping fig5")
        return

    # Load raw axes for test user
    print("  Loading test user sensor data for robustness analysis...")
    all_axes = []
    labels = []
    for row in rows:
        try:
            axes = load_axes(row["sensor_csv_path"])
            all_axes.append(axes)
            labels.append(row["activity"])
        except Exception:
            continue
    labels = np.array(labels)

    # Clean accuracy
    clean_features = np.vstack([extract_features(ax, sample_rate) for ax in all_axes])
    clean_acc = float((model.predict(clean_features) == labels).mean())

    # Noise sweep
    noise_levels = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0]
    noise_accs = []
    rng = np.random.default_rng(42)
    for sigma in noise_levels:
        if sigma == 0.0:
            noise_accs.append(clean_acc)
            continue
        noisy_features = np.vstack([
            extract_features(ax + rng.normal(0, sigma, ax.shape), sample_rate)
            for ax in all_axes
        ])
        noise_accs.append(float((model.predict(noisy_features) == labels).mean()))

    # Dropout sweep (zero out random samples)
    dropout_rates = [0.0, 0.05, 0.10, 0.20, 0.30, 0.50]
    dropout_accs = []
    for rate in dropout_rates:
        if rate == 0.0:
            dropout_accs.append(clean_acc)
            continue
        dropped_features = []
        for ax in all_axes:
            mask = rng.random(len(ax)) > rate
            ax_dropped = ax.copy()
            ax_dropped[~mask] = 0.0
            dropped_features.append(extract_features(ax_dropped, sample_rate))
        dropped_features = np.vstack(dropped_features)
        dropout_accs.append(float((model.predict(dropped_features) == labels).mean()))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.plot(noise_levels, noise_accs, "o-", color=PALETTE[4], linewidth=2, markersize=8)
    ax1.axhline(y=clean_acc, color="gray", linestyle="--", alpha=0.5, label=f"Clean: {clean_acc:.1%}")
    ax1.set_xlabel("Gaussian Noise σ (added to accelerometer)")
    ax1.set_ylabel("Test Accuracy")
    ax1.set_title("Robustness to Additive Noise")
    ax1.legend()

    ax2.plot([r * 100 for r in dropout_rates], dropout_accs, "s-", color=PALETTE[5], linewidth=2, markersize=8)
    ax2.axhline(y=clean_acc, color="gray", linestyle="--", alpha=0.5, label=f"Clean: {clean_acc:.1%}")
    ax2.set_xlabel("Sample Dropout Rate (%)")
    ax2.set_ylabel("Test Accuracy")
    ax2.set_title("Robustness to Dropped Samples")
    ax2.legend()

    fig.suptitle("Model Robustness (RF-Full)", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "fig5_robustness.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  → fig5_robustness.png")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figures_to_run = set(int(x) for x in args.only) if args.only else {1, 2, 3, 4, 5}

    qa_results = None
    if args.qa_results.exists():
        qa_results = json.loads(args.qa_results.read_text(encoding="utf-8"))

    print(f"Generating figures → {args.output_dir}/")

    if 1 in figures_to_run:
        if qa_results:
            fig1_accuracy_by_question_type(qa_results, args.output_dir)
        else:
            print("  ⚠ QA results not found, skipping fig1")

    if 2 in figures_to_run:
        fig2_confusion_matrix(args.output_dir)

    if 3 in figures_to_run:
        fig3_accuracy_vs_strictness(args.qa_cases, args.predicted_timeline, args.output_dir)

    if 4 in figures_to_run:
        fig4_accuracy_vs_overhead(args.output_dir)

    if 5 in figures_to_run:
        fig5_robustness(args.output_dir)

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
