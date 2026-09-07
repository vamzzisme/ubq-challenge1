#!/usr/bin/env python3
"""Measure one saved model's on-disk size and single-recording inference cost."""

from __future__ import annotations

import argparse
import json
import resource
import statistics
import sys
import time
from pathlib import Path
from typing import Any

from inference import load_model, predict_recording


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory containing resampled CSV recordings.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=30, help="Number of recordings to measure (default: 30).")
    parser.add_argument("--warmup", type=int, default=3, help="Unmeasured recordings run before timing (default: 3).")
    return parser.parse_args()


def peak_rss_mb() -> float:
    """Return process peak resident memory; macOS reports bytes, Linux KiB."""
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return usage / (1024 * 1024) if sys.platform == "darwin" else usage / 1024


def main() -> int:
    args = parse_args()
    if args.samples <= 0 or args.warmup < 0:
        raise ValueError("--samples must be positive and --warmup must be non-negative")
    csv_paths = sorted(args.input_dir.rglob("*.csv"))
    needed = args.warmup + args.samples
    if len(csv_paths) < needed:
        raise ValueError(f"Need at least {needed} CSV files under {args.input_dir}; found {len(csv_paths)}")

    saved_model = load_model(args.model)
    for path in csv_paths[: args.warmup]:
        predict_recording(path, saved_model)

    timings_ms: list[float] = []
    for path in csv_paths[args.warmup : needed]:
        started = time.perf_counter()
        predict_recording(path, saved_model)
        timings_ms.append((time.perf_counter() - started) * 1000)

    benchmark: dict[str, Any] = {
        "model_path": str(args.model),
        "model_size_bytes": args.model.stat().st_size,
        "model_size_mb": round(args.model.stat().st_size / (1024 * 1024), 4),
        "recordings_measured": len(timings_ms),
        "latency_ms": {
            "mean": round(statistics.mean(timings_ms), 4),
            "median": round(statistics.median(timings_ms), 4),
            "p95": round(sorted(timings_ms)[max(0, int(len(timings_ms) * 0.95) - 1)], 4),
        },
        "process_peak_rss_mb": round(peak_rss_mb(), 4),
        "measurement_scope": "Model loading, feature extraction, and one-recording prediction in the current Python process.",
        "model_config": saved_model.get("training_config", "Not stored in this legacy model artifact."),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(benchmark, indent=2) + "\n", encoding="utf-8")
    print(f"Model size: {benchmark['model_size_mb']} MB")
    print(f"Median single-recording latency: {benchmark['latency_ms']['median']} ms")
    print(f"Wrote benchmark to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
