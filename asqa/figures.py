"""Recognition figures, drawn from committed evaluation artifacts."""

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
             "Recognition backbone: hierarchical XGBoost with temporal context and time of day, before HMM "
             "decoding. The largest confusions are standing read as sitting (both still and upright) and "
             "running read as walking (similar rhythm, and some running labels come from an off-body phone).",
             ha="center", fontsize=7.5, color=INK_SECONDARY)
    fig.tight_layout()
    fig.savefig(output_dir / "fig2_confusion_matrix.png", bbox_inches="tight")
    plt.close(fig)
    print("  fig2_confusion_matrix.png")


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
        ax.set_ylabel("Per-window recognition accuracy")
        ax.set_title(title)
        ax.legend(frameon=False, loc="lower right")
        ax.yaxis.grid(True, zorder=0)
        ax.set_axisbelow(True)

    accuracies = [p.get("window_accuracy", p.get("accuracy")) for p in points]
    draw(ax1, [p["size_mb"] for p in points], accuracies, "Model size on disk (MB)",
         "Accuracy versus model size")
    draw(ax2, [p["median_ms"] for p in points], accuracies,
         "Median single-window latency (ms)", "Accuracy versus inference latency")

    target = data["target"]
    slm = data.get("language_model", {})
    note = (
        f"Measured on {target['machine']} / {target['system']}, single process. "
        f"Feature extraction adds {data.get('feature_extraction_ms', float('nan')):.2f} ms per window. "
    )
    note += "The y-axis is per-window recognition accuracy on one fold's test users. "
    if slm and "peak_rss_mb" in slm:
        note += (
            f"The open-world language model is reported separately and dominates: "
            f"{slm['parameters'] / 1e6:.0f}M parameters, {slm['peak_rss_mb']:.0f} MB peak RSS, "
            f"{slm['latency_s_including_cold_start']:.1f} s per query including weight loading, "
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
    metric = "accuracy"
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
    axes[0].set_ylabel("Per-window accuracy (decoded)")

    fig.suptitle("Robustness to degraded input", fontsize=13, y=1.03)
    fig.text(0.5, -0.07,
             f"Degradations applied to the raw 25 Hz windows before feature extraction, on "
             f"{data['n_windows']:,} windows from {len(data['users'])} held-out users. "
             "Noise is scaled by each window's measured gravity so a level means the same thing across devices "
             "reporting in g and in m/s^2. "
             + "The y-axis is per-window recognition accuracy after HMM decoding. Two users only, so "
               "differences of a few points between levels are within noise.",
             ha="center", fontsize=7.5, color=INK_SECONDARY)
    fig.tight_layout()
    fig.savefig(output_dir / "fig5_robustness.png", bbox_inches="tight")
    plt.close(fig)
    print("  fig5_robustness.png")


def _pooled(matrices: list) -> np.ndarray:
    return sum(np.asarray(m, dtype=float) for m in matrices)


def _row_normalise(matrix: np.ndarray) -> np.ndarray:
    sums = matrix.sum(axis=1, keepdims=True)
    return np.divide(matrix, sums, where=sums > 0, out=np.zeros_like(matrix))


def _comparison_systems() -> list[tuple[str, np.ndarray]] | None:
    """Pooled confusion matrices: the paper's method (both sensor sets) and ours."""
    paper = _load("paper_baseline_cv.json")
    ours = _load("recognition_cv.json")
    if paper is None or ours is None:
        return None
    return [
        ("Vaizman et al., phone Acc+Gyro (LFA)",
         _pooled([f["results"]["LFA_phone_acc_gyro"]["confusion_matrix"] for f in paper])),
        ("Vaizman et al., all 6 sensors (LFA)",
         _pooled([f["results"]["LFA_all_six"]["confusion_matrix"] for f in paper])),
        ("Ours, phone Acc+Gyro",
         _pooled([f["context_aware"]["confusion_matrix"] for f in ours])),
    ]


def figure_paper_vs_ours_confusion(output_dir: Path) -> None:
    systems = _comparison_systems()
    if systems is None:
        return
    pretty = [config.DISPLAY_NAMES[a] for a in config.ACTIVITIES]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.6))
    for ax, (title, matrix) in zip(axes, systems):
        normalised = _row_normalise(matrix)
        image = ax.imshow(normalised, cmap=BLUE_RAMP, vmin=0, vmax=1)
        n = len(pretty)
        for i in range(n):
            for j in range(n):
                value = normalised[i, j]
                if value >= 0.005:
                    ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7.5,
                            color="#ffffff" if value > 0.55 else INK,
                            weight="bold" if i == j else "normal")
        ax.set_xticks(range(n), pretty, rotation=40, ha="right")
        ax.set_yticks(range(n), pretty if ax is axes[0] else [""] * n)
        ax.set_xlabel("Predicted")
        macro_recall = float(np.mean(np.diag(normalised)))
        ax.set_title(f"{title}\nmean per-class recall {macro_recall:.2f}", fontsize=10)
    axes[0].set_ylabel("True")
    fig.colorbar(image, ax=list(axes), fraction=0.015, pad=0.02, label="Share of true class")
    fig.text(0.45, -0.06,
             "Row-normalised, pooled over the same 5 user-disjoint folds and the same 165,787 windows. "
             "The paper's method is reproduced on its published features; a window's class is the label with the "
             "highest probability. Ours: hierarchical XGBoost with temporal context and time of day, before HMM decoding.",
             ha="center", fontsize=7.5, color=INK_SECONDARY)
    fig.savefig(output_dir / "fig6_paper_vs_ours_confusion.png", bbox_inches="tight")
    plt.close(fig)
    print("  fig6_paper_vs_ours_confusion.png")


def figure_per_class_recall(output_dir: Path) -> None:
    systems = _comparison_systems()
    if systems is None:
        return
    labels = list(config.ACTIVITIES)
    pretty = [config.DISPLAY_NAMES[a] for a in labels] + ["mean"]
    colours = [SLOT["blue"], SLOT["orange"], SLOT["aqua"]]
    x = np.arange(len(pretty))
    width = 0.26
    fig, ax = plt.subplots(figsize=(12, 4.6))
    for k, ((name, matrix), colour) in enumerate(zip(systems, colours)):
        recall = np.diag(_row_normalise(matrix))
        values = np.r_[recall, recall.mean()]
        bars = ax.bar(x + (k - 1) * (width + 0.01), values, width, color=colour, label=name, zorder=3)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012, f"{bar.get_height():.2f}",
                    ha="center", va="bottom", fontsize=7, color=INK_SECONDARY)
    ax.axvline(len(labels) - 0.5, color=GRID, linewidth=1)
    ax.set_xticks(x, pretty)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Recall (share of true class found)")
    ax.set_title("Per-class recall on identical test windows")
    ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    fig.tight_layout()
    fig.savefig(output_dir / "fig7_per_class_recall.png", bbox_inches="tight")
    plt.close(fig)
    print("  fig7_per_class_recall.png")


def figure_ablation(output_dir: Path) -> None:
    data = _load("ablation_cv.json")
    if data is None:
        return
    summary = data["summary"]
    f1_index = data["metrics"].index("macro_f1")
    deployed = "deployed: hierarchical XGBoost"
    names = sorted(summary, key=lambda n: summary[n]["mean"][f1_index])
    values = [summary[n]["mean"][f1_index] for n in names]
    errors = [summary[n]["std"][f1_index] for n in names]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    colours = [SLOT["blue"] if n == deployed else "#9ec5f4" for n in names]
    bars = ax.barh(range(len(names)), values, xerr=errors, color=colours, height=0.62, zorder=3,
                   error_kw={"ecolor": INK_SECONDARY, "elinewidth": 1, "capsize": 2})
    for bar, value in zip(bars, values):
        ax.text(0.012, bar.get_y() + bar.get_height() / 2, f"{value:.3f}", va="center",
                fontsize=8, color=INK, zorder=4)
    ax.axvline(summary[deployed]["mean"][f1_index], color=INK_SECONDARY, linestyle=":", linewidth=1.2, zorder=4)
    ax.set_yticks(range(len(names)), ["deployed model" if n == deployed else n for n in names])
    ax.set_xlim(0, 0.7)
    ax.set_xlabel("Macro-F1 (mean over 5 folds, bars show one std)")
    ax.set_title("Ablation: one change at a time from the deployed per-window model")
    ax.xaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(output_dir / "fig8_ablation.png", bbox_inches="tight")
    plt.close(fig)
    print("  fig8_ablation.png")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=config.FIGURE_DIR)
    parser.add_argument("--only", nargs="*", type=int, help="Figure numbers to draw (default: all).")
    args = parser.parse_args()

    _style()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    wanted = set(args.only) if args.only else {2, 4, 5, 6, 7, 8}
    print(f"Writing figures to {args.output_dir}/")

    if 2 in wanted:
        figure_confusion_matrix(args.output_dir)
    if 4 in wanted:
        figure_accuracy_vs_overhead(args.output_dir)
    if 5 in wanted:
        figure_robustness(args.output_dir)
    if 6 in wanted:
        figure_paper_vs_ours_confusion(args.output_dir)
    if 7 in wanted:
        figure_per_class_recall(args.output_dir)
    if 8 in wanted:
        figure_ablation(args.output_dir)
    print("Done.")
    return 0


if __name__ == "__main__":
    from asqa.figures import main as _main

    raise SystemExit(_main())
