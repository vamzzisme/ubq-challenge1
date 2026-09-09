#!/usr/bin/env python3
"""Predict one recording and write an auditable JSON evidence record using the CNN."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cnn_inference import load_cnn_model, predict_recording_cnn


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path, help="Resampled sensor CSV to classify.")
    parser.add_argument("--model", type=Path, required=True, help="Path to CNN model checkpoint (.pt)")
    parser.add_argument("--output", type=Path, help="Write JSON here; otherwise print it to stdout.")
    args = parser.parse_args()
    
    if not args.csv_path.is_file():
        raise FileNotFoundError(f"Sensor CSV not found: {args.csv_path}")
    if not args.model.is_file():
        raise FileNotFoundError(f"Model not found: {args.model}")

    result = predict_recording_cnn(args.csv_path, load_cnn_model(args.model))
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
