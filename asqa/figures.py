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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=config.FIGURE_DIR)
    parser.add_argument("--only", nargs="*", type=int, help="Figure numbers to draw (default: all).")
    args = parser.parse_args()

    _style()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    wanted = set(args.only) if args.only else {2, 4, 5}
    print(f"Writing figures to {args.output_dir}/")

    if 2 in wanted:
        figure_confusion_matrix(args.output_dir)
    if 4 in wanted:
        figure_accuracy_vs_overhead(args.output_dir)
    if 5 in wanted:
        figure_robustness(args.output_dir)
    print("Done.")
    return 0


if __name__ == "__main__":
    from asqa.figures import main as _main

    raise SystemExit(_main())
