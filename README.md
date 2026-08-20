# Sensor-Grounded Activity Question Answering

CS60055 Ubiquitous Computing Hackathon Challenge 1.

This project will answer natural-language questions about wearable accelerometer and gyroscope recordings, while citing the timestamped sensor evidence supporting each answer.

## Planned pipeline

1. Convert space-separated raw accelerometer and gyroscope `.dat` captures to CSV.
2. Resample each 20-second, 40 Hz capture to 25 Hz (500 samples).
3. Align recordings with the self-labelled metadata and map them to the seven challenge activities.
4. Train a lightweight six-axis activity-recognition model.
5. Aggregate predictions into activity intervals and answer questions with grounded evidence.

## Repository layout

```text
data/           Local data only; never committed
src/            Preprocessing, modelling, and QA code
notebooks/      Exploratory analysis
reports/        Report sources and generated figures
```

## Data

Place the raw `.dat` recordings and self-label CSV under `data/raw/`. The raw dataset is intentionally excluded from version control. Once the files are available, the preprocessing scripts will document the expected naming and timestamp conventions.

## Academic integrity

All external datasets, models, libraries, and AI assistance used during development will be cited and disclosed in the final report, as required by the challenge brief.
