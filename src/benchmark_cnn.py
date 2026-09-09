#!/usr/bin/env python3
"""Benchmark CNN model inference: size and per-recording latency."""

from __future__ import annotations

import argparse
import json
import resource
import statistics
import sys
import time
from pathlib import Path

from cnn_inference import load_cnn_model, predict_recording_cnn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=3)
    return parser.parse_args()


def peak_rss_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return usage / (1024 * 1024) if sys.platform == "darwin" else usage / 1024


def main() -> int:
    args = parse_args()
    csv_paths = sorted(args.input_dir.rglob("*.csv"))
    needed = args.warmup + args.samples
    if len(csv_paths) < needed:
        raise ValueError(f"Need {needed} CSVs, found {len(csv_paths)}")

    saved = load_cnn_model(args.checkpoint)

    for p in csv_paths[:args.warmup]:
        predict_recording_cnn(p, saved)

    timings_ms = []
    for p in csv_paths[args.warmup:needed]:
        t0 = time.perf_counter()
        predict_recording_cnn(p, saved)
        timings_ms.append((time.perf_counter() - t0) * 1000)

    benchmark = {
        "model_path": str(args.checkpoint),
        "model_size_bytes": args.checkpoint.stat().st_size,
        "model_size_mb": round(args.checkpoint.stat().st_size / (1024 * 1024), 4),
        "recordings_measured": len(timings_ms),
        "latency_ms": {
            "mean": round(statistics.mean(timings_ms), 4),
            "median": round(statistics.median(timings_ms), 4),
            "p95": round(sorted(timings_ms)[max(0, int(len(timings_ms) * 0.95) - 1)], 4),
        },
        "process_peak_rss_mb": round(peak_rss_mb(), 4),
        "model_config": "AccelCNN (4 conv blocks, 94K params)",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(benchmark, indent=2) + "\n", encoding="utf-8")
    print(f"Model size: {benchmark['model_size_mb']} MB")
    print(f"Median latency: {benchmark['latency_ms']['median']} ms")
    print(f"Wrote benchmark to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
