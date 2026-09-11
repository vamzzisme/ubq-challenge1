"""The figures the brief requires, drawn from committed evaluation artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from asqa import config

SLOT = {
    "blue": "#2a78d6",
    "orange": "#eb6834",
    "aqua": "#1baf7a",
    "yellow": "#eda100",
    "magenta": "#e87ba4",
}
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#dedcd6"

BLUE_RAMP = LinearSegmentedColormap.from_list(
    "asqa_blue", ["#fcfcfb", "#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]
)


def _style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK_SECONDARY,
        "axes.titlecolor": INK,
        "text.color": INK,
        "xtick.color": INK_SECONDARY,
        "ytick.color": INK_SECONDARY,
        "grid.color": GRID,
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 200,
    })


def _load(name: str) -> dict | None:
    path = config.EVALUATION_DIR / name
    if not path.exists():
        print(f"  ! missing {path.name}; skipping")
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def figure_accuracy_by_question_type(output_dir: Path) -> None:
    data = _load("qa_cv.json")
    if data is None:
        return
    overall = data["overall"]
    by_type = overall["by_question_type"]
    kinds = list(by_type)
    answer = [by_type[k]["answer_accuracy"] for k in kinds]
    grounded = [by_type[k]["grounded_accuracy"] for k in kinds]
    # Defined only for the categorical types; numeric types leave a gap rather than a zero.
    macro_f1 = [by_type[k].get("macro_f1") for k in kinds]

    labels = kinds + ["OVERALL\n(macro)"]
    answer.append(overall["overall_qa_accuracy_macro"])
    grounded.append(overall["overall_grounded_accuracy_macro"])
    macro_f1.append(overall.get("overall_macro_f1"))

    x = np.arange(len(labels))
    width = 0.27
    fig, ax = plt.subplots(figsize=(12, 5.2))
    bars_a = ax.bar(x - width, answer, width, label="Answer correct (accuracy)", color=SLOT["blue"], zorder=3)
    bars_f = ax.bar(
        [xi for xi, v in zip(x, macro_f1) if v is not None],
        [v for v in macro_f1 if v is not None],
        width, label="Macro-F1 over answer classes", color=SLOT["aqua"], zorder=3,
    )
    bars_g = ax.bar(x + width, grounded, width, label="Answer correct AND evidence valid",
                    color=SLOT["orange"], zorder=3)

    for group in (bars_a, bars_f, bars_g):
        for bar in group:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
                    f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=7.5, color=INK_SECONDARY)

    ax.axvline(len(kinds) - 0.5, color=GRID, linewidth=1)
    ax.set_ylabel("Fraction of questions")
    ax.set_ylim(0, 1.12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_title("QA accuracy by question type, five-fold user-disjoint cross-validation")
    ax.legend(loc="upper left", frameon=False)
    ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)

    counts = ", ".join(f"{k} n={by_type[k]['count']}" for k in kinds)
    rules = data["scoring_rules"]
    fig.text(0.5, -0.14,
             "Correctness rule differs by type: identification, verification, comparison and open-world are scored by "
             f"exact match; duration by {rules['duration']}; grounding by {rules['grounding']}; "
             f"count by {rules['count']}. "
             f"Grounded = {rules['grounded']}. Macro-F1 is defined for the categorical types only, and is "
             "shown because the classes are imbalanced and accuracy alone favours the majority answer."
             f"\n{counts}. The overall accuracy and grounded bars are macro-averaged over all seven types; "
             "the overall macro-F1 bar averages the four categorical types only, so it sits higher for that "
             "reason and is not directly comparable with the bar beside it.",
             ha="center", fontsize=7.5, color=INK_SECONDARY, wrap=True)

    fig.tight_layout()
    fig.savefig(output_dir / "fig1_accuracy_by_question_type.png", bbox_inches="tight")
    plt.close(fig)
    print("  fig1_accuracy_by_question_type.png")


def figure_confusion_matrix(output_dir: Path) -> None:
    data = _load("recognition_cv.json")
    if data is None:
        return
    labels = list(config.ACTIVITIES)
    matrix = np.zeros((len(labels), len(labels)))
    for fold in data:
        matrix += np.asarray(fold["context_aware"]["confusion_matrix"], dtype=float)

    row_sums = matrix.sum(axis=1, keepdims=True)
    normalised = np.divide(matrix, row_sums, where=row_sums > 0, out=np.zeros_like(matrix))

    precision, recall, f1 = [], [], []
    for i in range(len(labels)):
        tp = matrix[i, i]
        p = tp / matrix[:, i].sum() if matrix[:, i].sum() else 0.0
        r = tp / matrix[i, :].sum() if matrix[i, :].sum() else 0.0
        precision.append(p)
        recall.append(r)
        f1.append(2 * p * r / (p + r) if (p + r) else 0.0)

    fig, (ax, ax_table) = plt.subplots(
        1, 2, figsize=(13.5, 6.4), gridspec_kw={"width_ratios": [1.35, 1]}
    )
    pretty = [config.DISPLAY_NAMES[a] for a in labels]
    image = ax.imshow(normalised, cmap=BLUE_RAMP, vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), pretty, rotation=40, ha="right")
    ax.set_yticks(range(len(labels)), pretty)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Activity confusion matrix (row-normalised, pooled over 5 folds)")
    for i in range(len(labels)):
        for j in range(len(labels)):
            value = normalised[i, j]
            if value >= 0.005:
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8,
                        color="#ffffff" if value > 0.55 else INK)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Share of true class")

    ax_table.axis("off")
    rows = [[pretty[i], f"{precision[i]:.2f}", f"{recall[i]:.2f}", f"{f1[i]:.2f}", f"{int(matrix[i].sum()):,}"]
            for i in range(len(labels))]
    rows.append(["macro average", f"{np.mean(precision):.2f}", f"{np.mean(recall):.2f}",
                 f"{np.mean(f1):.2f}", f"{int(matrix.sum()):,}"])
    table = ax_table.table(cellText=rows, colLabels=["class", "precision", "recall", "F1", "n"],
                           loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.55)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor(GRID)
        if row == 0:
            cell.set_text_props(weight="bold", color=INK)
        if row == len(rows):
            cell.set_text_props(weight="bold")
    ax_table.set_title("Per-class precision, recall and F1", pad=18)

    fig.text(0.5, -0.03,
             "Recognition backbone: hierarchical XGBoost with temporal context, before HMM decoding. "
             "The dominant residual confusion is lying down against sitting, which a still window cannot "
             "separate on device orientation alone.",
             ha="center", fontsize=7.5, color=INK_SECONDARY)
    fig.tight_layout()
    fig.savefig(output_dir / "fig2_confusion_matrix.png", bbox_inches="tight")
    plt.close(fig)
    print("  fig2_confusion_matrix.png")


def _count_tolerance_sweep(results: list[dict]) -> list[dict]:
    """Fraction of bout counts accepted as the absolute tolerance widens."""
    rows = [
        r for r in results
        if r.get("question_type") == "count" and r.get("expected_value") is not None
    ]
    points = []
    for tolerance in (0, 1, 2, 3, 5, 8):
        accepted = [
            r["predicted_value"] is not None
            and abs(r["predicted_value"] - r["expected_value"]) <= tolerance
            for r in rows
        ]
        points.append({
            "tolerance": tolerance,
            "fraction": float(np.mean(accepted)) if accepted else 0.0,
        })
    return points


def figure_accuracy_vs_strictness(output_dir: Path) -> None:
    data = _load("qa_cv.json")
    if data is None:
        return
    strictness = data["strictness"]
    iou = strictness["iou"]
    tolerance = strictness["numeric_tolerance"]
    # Counts are numeric answers too, so the brief's tolerance axis applies to them.
    # The per-case values are stored, so the sweep is recovered from the rows rather
    # than re-answering 1,300 questions.
    counts = _count_tolerance_sweep(data["results"])

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16.5, 4.8))

    xs = [p["threshold"] for p in iou]
    ys = [p["fraction"] for p in iou]
    ax1.plot(xs, ys, "-o", color=SLOT["blue"], linewidth=2, markersize=7, zorder=3)
    for x, y in zip(xs, ys):
        ax1.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, 9),
                     ha="center", fontsize=7.5, color=INK_SECONDARY)
    ax1.set_xlabel("Evidence IoU threshold")
    ax1.set_ylabel("Fraction of cited intervals accepted")
    ax1.set_title("Evidence grounding versus IoU strictness")
    ax1.set_ylim(0, max(ys) * 1.25 + 0.05)
    ax1.yaxis.grid(True, zorder=0)
    ax1.set_axisbelow(True)

    xs2 = [p["tolerance_fraction"] * 100 for p in tolerance]
    ys2 = [p["fraction"] for p in tolerance]
    ax2.plot(xs2, ys2, "-s", color=SLOT["orange"], linewidth=2, markersize=7, zorder=3)
    for x, y in zip(xs2, ys2):
        ax2.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, 9),
                     ha="center", fontsize=7.5, color=INK_SECONDARY)
    ax2.set_xlabel("Numeric tolerance (% of the reference value)")
    ax2.set_ylabel("Fraction of numeric answers accepted")
    ax2.set_title("Duration and onset accuracy versus tolerance")
    ax2.set_ylim(0, max(ys2) * 1.25 + 0.05)
    ax2.yaxis.grid(True, zorder=0)
    ax2.set_axisbelow(True)

    xs3 = [p["tolerance"] for p in counts]
    ys3 = [p["fraction"] for p in counts]
    ax3.plot(xs3, ys3, "-^", color=SLOT["aqua"], linewidth=2, markersize=7, zorder=3)
    for x, y in zip(xs3, ys3):
        ax3.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, 9),
                     ha="center", fontsize=7.5, color=INK_SECONDARY)
    ax3.set_xlabel("Count tolerance (bouts, absolute)")
    ax3.set_ylabel("Fraction of count answers accepted")
    ax3.set_title("Bout-count accuracy versus tolerance")
    ax3.set_xticks(xs3)
    ax3.set_ylim(0, max(ys3) * 1.25 + 0.05)
    ax3.yaxis.grid(True, zorder=0)
    ax3.set_axisbelow(True)

    fig.suptitle("Accuracy versus evaluation strictness", fontsize=13, y=1.02)
    fig.text(0.5, -0.08,
             "Left: how often a cited evidence interval overlaps the true interval, as the required overlap tightens. "
             "Middle: how often a duration or onset lands within a given tolerance. Right: the same for bout counts, "
             "where the reported headline uses max(+/-5, 20%). A curve that falls slowly indicates near misses; a cliff "
             "indicates wild ones. The count curve is the shallowest of the three, so the headline count accuracy is "
             "sensitive to where the band is drawn and the band is stated wherever that number appears.",
             ha="center", fontsize=7.5, color=INK_SECONDARY)
    fig.tight_layout()
    fig.savefig(output_dir / "fig3_accuracy_vs_strictness.png", bbox_inches="tight")
    plt.close(fig)
    print("  fig3_accuracy_vs_strictness.png")


def figure_accuracy_vs_overhead(output_dir: Path) -> None:
    data = _load("benchmark.json")
    if data is None:
        return
    points = data["operating_points"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.2))

    def draw(ax, xs, ys, xlabel, title) -> None:
        xs = np.asarray(xs, dtype=float)
        ys = np.asarray(ys, dtype=float)

        order = np.argsort(xs)
        frontier = []
        best = -np.inf
        for index in order:
            if ys[index] > best:
                best = ys[index]
                frontier.append(index)
        ax.plot(xs[frontier], ys[frontier], "--", color=SLOT["orange"], linewidth=1.8,
                zorder=2, label="Pareto frontier")

        ax.scatter(xs, ys, s=130, color=SLOT["blue"], edgecolors=SURFACE, linewidth=2, zorder=4)

        span_x = (xs.max() - xs.min()) or 1.0
        span_y = (ys.max() - ys.min()) or 1.0
        placed: list[tuple[float, float]] = []
        for x, y, point in zip(xs, ys, points):
            offset = (9, -4)
            for previous_x, previous_y in placed:
                if abs(x - previous_x) / span_x < 0.18 and abs(y - previous_y) / span_y < 0.10:
                    offset = (9, 12)
                    break
            placed.append((x, y))
            ax.annotate(f"{point['configuration']} {y:.3f}", (x, y), textcoords="offset points",
                        xytext=offset, fontsize=8.5, color=INK)
        ax.margins(x=0.18, y=0.16)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Overall QA accuracy (macro over question types)")
        ax.set_title(title)
        ax.legend(frameon=False, loc="lower right")
        ax.yaxis.grid(True, zorder=0)
        ax.set_axisbelow(True)

    # The brief asks for overall QA accuracy here, not per-window recognition
    # accuracy; older benchmark files carry only the latter, so fall back to it
    # rather than fail, and say which one was plotted in the caption.
    is_qa = all("qa_accuracy_macro" in p for p in points)
    accuracies = [p["qa_accuracy_macro"] if is_qa else p.get("window_accuracy", p.get("accuracy")) for p in points]
    draw(ax1, [p["size_mb"] for p in points], accuracies, "Model size on disk (MB)",
         "QA accuracy versus model size")
    draw(ax2, [p["median_ms"] for p in points], accuracies,
         "Median single-window latency (ms)", "QA accuracy versus inference latency")

    target = data["target"]
    slm = data.get("language_model", {})
    note = (
        f"Measured on {target['machine']} / {target['system']}, single process. "
        f"Feature extraction adds {data.get('feature_extraction_ms', float('nan')):.2f} ms per window. "
    )
    if is_qa:
        window_accuracies = ", ".join(
            f"{p['configuration']} {p['window_accuracy']:.3f}" for p in points if "window_accuracy" in p
        )
        note += (
            "The y-axis is end-to-end QA accuracy, macro-averaged over the seven question types, "
            f"so each point is carried through decoding and aggregation rather than stopping at the "
            f"classifier. Per-window recognition accuracy for the same points: {window_accuracies}. "
        )
    else:
        note += "The y-axis is per-window recognition accuracy; re-run `python -m asqa.benchmark` for QA accuracy. "
    if slm and "peak_rss_mb" in slm:
        note += (
            f"The open-world language model is reported separately and dominates: "
            f"{slm['parameters'] / 1e6:.0f}M parameters, {slm['peak_rss_mb']:.0f} MB peak RSS, "
            f"{slm['latency_s_including_cold_start']:.1f} s per query including weight loading   "
            f"roughly {slm['latency_s_including_cold_start'] * 1000 / max(points[0]['median_ms'], 1e-6):,.0f}x "
            f"the cost of classifying a window."
        )
    fig.suptitle("Accuracy versus resource cost", fontsize=13, y=1.02)
    fig.text(0.5, -0.08, note, ha="center", fontsize=7.5, color=INK_SECONDARY, wrap=True)
    fig.tight_layout()
    fig.savefig(output_dir / "fig4_accuracy_vs_overhead.png", bbox_inches="tight")
    plt.close(fig)
    print("  fig4_accuracy_vs_overhead.png")


def figure_robustness(output_dir: Path) -> None:
    data = _load("robustness.json")
    if data is None:
        return
    curves = data["curves"]
    # The brief puts accuracy on a fixed question set on this y-axis. Older
    # robustness files carry only per-window accuracy, so fall back to it.
    metric = "qa_accuracy" if all(
        "qa_accuracy" in point for series in curves.values() for point in series
    ) else "accuracy"
    panels = [
        ("noise", "Accelerometer noise sigma (g)", "Additive sensor noise", SLOT["blue"], False),
        ("dropout", "Samples lost (%)", "Dropped samples", SLOT["orange"], True),
        ("sample_rate", "Sampling rate (Hz)", "Reduced sampling rate", SLOT["aqua"], False),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    clean = curves["noise"][0][metric]

    for ax, (key, xlabel, title, colour, as_percent) in zip(axes, panels):
        xs = [p["level"] * (100 if as_percent else 1) for p in curves[key]]
        ys = [p[metric] for p in curves[key]]
        ax.plot(xs, ys, "-o", color=colour, linewidth=2, markersize=7, zorder=3)
        for x, y in zip(xs, ys):
            ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, 9),
                        ha="center", fontsize=7.5, color=INK_SECONDARY)
        ax.axhline(clean, color=INK_SECONDARY, linestyle=":", linewidth=1.2,
                   label=f"Undegraded: {clean:.2f}")
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        ax.set_ylim(0, max(max(ys), clean) * 1.22)
        ax.legend(frameon=False, loc="lower left", fontsize=8)
        ax.yaxis.grid(True, zorder=0)
        ax.set_axisbelow(True)
        if key == "sample_rate":
            ax.invert_xaxis()
    axes[0].set_ylabel(
        "QA accuracy on a fixed question set" if metric == "qa_accuracy"
        else "Per-window accuracy (decoded)"
    )

    fig.suptitle("Robustness to degraded input", fontsize=13, y=1.03)
    fig.text(0.5, -0.07,
             f"Degradations applied to the raw 25 Hz windows before feature extraction, on "
             f"{data['n_windows']:,} windows from {len(data['users'])} held-out users. "
             "Noise is scaled by each window's measured gravity so a level means the same thing across devices "
             "reporting in g and in m/s^2. "
             + ("The y-axis is question-answering accuracy, macro-averaged over question types, on the same "
                "question set at every level: the reference answers depend only on the labels, which the "
                "degradation does not touch, so only the signal the system reads changes."
                if metric == "qa_accuracy" else
                "The y-axis is per-window recognition accuracy; re-run `python -m asqa.robustness` for QA accuracy."),
             ha="center", fontsize=7.5, color=INK_SECONDARY)
    fig.tight_layout()
    fig.savefig(output_dir / "fig5_robustness.png", bbox_inches="tight")
    plt.close(fig)
    print("  fig5_robustness.png")


def figure_balanced_and_error(output_dir: Path) -> None:
    """Required QA metrics: precision/recall/F1, error magnitude, and evidence grounding."""
    data = _load("qa_cv.json")
    if data is None:
        return
    overall = data["overall"]
    by_type = overall["by_question_type"]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4.6))

    binary = overall.get("verification_binary")
    if binary:
        names = ["Precision", "Recall", "F1", "Specificity"]
        values = [binary["precision"], binary["recall"], binary["f1"], binary["specificity"]]
        colours = [SLOT["blue"], SLOT["orange"], SLOT["aqua"], SLOT["yellow"]]
        bars = ax1.bar(names, values, color=colours, width=0.62, zorder=3)
        for bar in bars:
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                     f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=8, color=INK_SECONDARY)
        ax1.set_ylim(0, 1.15)
        ax1.set_ylabel("Score on the positive class (Yes)")
        # Recall covers the Yes class and specificity the No class; reporting both
        # states performance on each answer separately, as the brief requires.
        ax1.set_title(
            f"Precision, recall and F1 on verification\n"
            f"{binary['support_positive']} Yes vs {binary['support_negative']} No",
            fontsize=10,
        )
        ax1.tick_params(axis="x", labelrotation=15)
    else:
        ax1.axis("off")

    # Absolute error, in the units each type is actually answered in.
    error_kinds = [k for k in by_type if (by_type[k].get("error") or {}).get("n_scored")]
    if error_kinds:
        maes = [by_type[k]["error"]["mean_absolute_error"] for k in error_kinds]
        medians = [by_type[k]["error"]["median_absolute_error"] for k in error_kinds]
        x = np.arange(len(error_kinds))
        ax2.bar(x - 0.19, maes, 0.38, label="Mean", color=SLOT["blue"], zorder=3)
        ax2.bar(x + 0.19, medians, 0.38, label="Median", color=SLOT["magenta"], zorder=3)
        for xi, kind, mae in zip(x, error_kinds, maes):
            mape = by_type[kind]["error"].get("mean_absolute_percentage_error")
            if mape is not None:
                ax2.annotate(f"{mape:.0f}% MAPE", (xi, mae), textcoords="offset points",
                             xytext=(0, 12), ha="center", fontsize=7.5, color=INK_SECONDARY)
        ax2.set_xticks(x)
        ax2.set_xticklabels(error_kinds, rotation=15, ha="right")
        # Counts are bouts and the others are seconds, four orders of magnitude
        # apart. A linear axis would flatten the count bars to nothing.
        ax2.set_yscale("log")
        ax2.set_ylabel("Absolute error, log scale (seconds; bouts for count)")
        ax2.set_title("Error magnitude on the numeric\nanswers (duration and count)", fontsize=10)
        ax2.legend(frameon=False, fontsize=8, loc="upper left")
    else:
        ax2.axis("off")

    grounding_kinds = [k for k in by_type if (by_type[k].get("grounding") or {}).get("n_cited")]
    if grounding_kinds:
        precisions = [by_type[k]["grounding"]["grounding_precision"] for k in grounding_kinds]
        bars = ax3.barh(grounding_kinds, precisions, color=SLOT["aqua"], height=0.6, zorder=3)
        for bar in bars:
            ax3.text(bar.get_width() + 0.015, bar.get_y() + bar.get_height() / 2,
                     f"{bar.get_width():.2f}", va="center", fontsize=8, color=INK_SECONDARY)
        ax3.set_xlim(0, 1.15)
        ax3.set_xlabel("Fraction of cited intervals containing the named activity")
        ax3.set_title("Evidence grounding, judged against\nthe per-window labels", fontsize=10)
        ax3.xaxis.grid(True, zorder=0)
    else:
        ax3.axis("off")

    for ax in (ax1, ax2):
        ax.yaxis.grid(True, zorder=0)
    for ax in (ax1, ax2, ax3):
        ax.set_axisbelow(True)

    fig.suptitle("Question-answering performance and error analysis", fontsize=13, y=1.03)
    fig.text(0.5, -0.1,
             "Left: the classification metrics the brief asks for, on the yes/no questions - precision, recall "
             "and F1 for the Yes class, with specificity giving the same view of the No class. Centre: error "
             "analysis for the questions answered with a number, reported as mean and median absolute error "
             "with percentage error annotated; a mean well above the median indicates a few large misses rather "
             "than uniform drift. Right: whether the evidence returned with each answer is correct, measured as "
             "the fraction of cited intervals whose ground-truth window labels contain the named activity.",
             ha="center", fontsize=7.5, color=INK_SECONDARY, wrap=True)
    fig.tight_layout()
    fig.savefig(output_dir / "fig6_balanced_and_error.png", bbox_inches="tight")
    plt.close(fig)
    print("  fig6_balanced_and_error.png")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=config.FIGURE_DIR)
    parser.add_argument("--only", nargs="*", type=int, help="Figure numbers to draw (default: all).")
    args = parser.parse_args()

    _style()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    wanted = set(args.only) if args.only else {1, 2, 3, 4, 5, 6}
    print(f"Writing figures to {args.output_dir}/")

    if 1 in wanted:
        figure_accuracy_by_question_type(args.output_dir)
    if 2 in wanted:
        figure_confusion_matrix(args.output_dir)
    if 3 in wanted:
        figure_accuracy_vs_strictness(args.output_dir)
    if 4 in wanted:
        figure_accuracy_vs_overhead(args.output_dir)
    if 5 in wanted:
        figure_robustness(args.output_dir)
    if 6 in wanted:
        figure_balanced_and_error(args.output_dir)
    print("Done.")
    return 0


if __name__ == "__main__":
    from asqa.figures import main as _main

    raise SystemExit(_main())
