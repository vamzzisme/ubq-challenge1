#!/usr/bin/env python3
import argparse
from pathlib import Path
import joblib
from signal_features import extract_features, load_axes

def main():
    parser = argparse.ArgumentParser(description="Run inference on a single accelerometer CSV.")
    parser.add_argument("csv_path", type=Path, help="Path to the preprocessed accelerometer CSV file")
    parser.add_argument("--model", type=Path, default=Path("artifacts/baseline/random_forest.joblib"))
    args = parser.parse_args()

    if not args.csv_path.exists():
        print(f"Error: {args.csv_path} does not exist.")
        return 1

    if not args.model.exists():
        print(f"Error: Model file {args.model} not found.")
        return 1

    print(f"Loading model from {args.model}...")
    saved_data = joblib.load(args.model)
    model = saved_data["model"]
    
    print(f"Loading data and extracting features from {args.csv_path}...")
    axes = load_axes(args.csv_path)
    features = extract_features(axes).reshape(1, -1)
    
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    
    print("\n--- Prediction Results ---")
    print(f"Predicted Activity: {prediction}")
    print("Probabilities:")
    for class_name, prob in zip(model.classes_, probabilities):
        print(f"  {class_name}: {prob:.3f}")

if __name__ == "__main__":
    main()
