#!/usr/bin/env python3
"""Predict one recording and write an auditable JSON evidence record."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from inference import load_model, predict_recording


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path, help="Resampled sensor CSV to classify.")
    parser.add_argument("--model", type=Path, default=Path("artifacts/baseline/random_forest.joblib"))
    parser.add_argument("--output", type=Path, help="Write JSON here; otherwise print it to stdout.")
    args = parser.parse_args()
    if not args.csv_path.is_file():
        raise FileNotFoundError(f"Sensor CSV not found: {args.csv_path}")
    if not args.model.is_file():
        raise FileNotFoundError(f"Model not found: {args.model}")

    result = predict_recording(args.csv_path, load_model(args.model))
    rendered = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print(f"Wrote evidence record to {args.output}")
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
