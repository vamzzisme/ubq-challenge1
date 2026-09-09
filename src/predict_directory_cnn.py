#!/usr/bin/env python3
"""Create evidence JSON records for every resampled sensor CSV using the CNN model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cnn_inference import load_cnn_model, predict_recording_cnn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, help="Process at most this many recordings for a smoke test.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    csv_paths = sorted(args.input_dir.rglob("*.csv"))
    if args.limit is not None:
        csv_paths = csv_paths[: args.limit]
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found under {args.input_dir}")
    
    saved_model = load_cnn_model(args.model)
    written = skipped = 0
    for index, csv_path in enumerate(csv_paths, start=1):
        relative_path = csv_path.relative_to(args.input_dir)
        output_path = args.output_dir / relative_path.with_suffix(".prediction.json")
        if output_path.exists() and not args.overwrite:
            skipped += 1
            continue
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(predict_recording_cnn(csv_path, saved_model), indent=2) + "\n", encoding="utf-8")
        written += 1
        if index % 500 == 0 or index == len(csv_paths):
            print(f"Processed {index}/{len(csv_paths)} recordings")
    print(f"Wrote {written} prediction records; skipped {skipped} existing records")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
