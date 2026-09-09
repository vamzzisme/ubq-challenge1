# Sensor-Grounded Activity Question Answering

CS60055 Ubiquitous Computing Hackathon Challenge 1.

This project answers natural-language questions about wearable accelerometer and gyroscope recordings, while citing the timestamped sensor evidence supporting each answer.

## Completed Pipeline

1. **Preprocessing**: Convert space-separated raw accelerometer and gyroscope `.dat` captures to CSV, resampled to 25 Hz.
2. **Alignment**: Align recordings with self-labelled metadata and map them to six challenge activities.
3. **Modelling**: Train models including a lightweight Random Forest baseline and a 6-axis 1D-CNN (Gyroscope + Accelerometer fusion).
4. **Timeline**: Aggregate predictions into continuous activity intervals (timeline).
5. **Question Answering**: Query the timeline with grounded evidence using a deterministic QA evaluation script.

## Repository Layout

```text
data/           Local data only; never committed (raw, processed)
src/            Preprocessing, modelling, inference, and QA scripts
notebooks/      Exploratory analysis
reports/        Report sources and generated figures
artifacts/      Model checkpoints, timelines, and evaluation results
```

## Setup & Preprocessing

```bash
# Create & activate virtualenv (one-time)
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Convert and Resample Sensors

The supplied files live under `data/Original Data`. Convert the raw accelerometer and gyroscope captures to readable CSV and resample to 25 Hz:

```bash
# Accelerometer
python3 src/prepare_sensor_data.py --input-dir "data/Original Data" --output-dir data/processed --modality raw_acc

# Gyroscope
python3 src/prepare_sensor_data.py --input-dir "data/Original Data" --output-dir data/processed --modality proc_gyro
```

### Build the Labelled Training Index

```bash
python3 src/build_labeled_index.py \
  --data-dir data \
  --resampled-dir data/processed/raw_acc_25hz \
  --output data/processed/raw_acc_training_index.csv
```

## Modelling

You can train either the Random Forest Baseline (accelerometer only) or the 1D-CNN (6-channel fused accelerometer + gyroscope).

### Random Forest Baseline (Acc-only)
```bash
python3 src/train_baseline.py \
  --validation-user 2C32C23E-E30C-498A-8DD2-0EFB9150A02E \
  --test-user 0A986513-7828-4D53-AA1F-E02D6DF9561B
```

### 1D-CNN (Acc + Gyro Fusion)
```bash
python3 src/train_cnn.py --epochs 60 --batch-size 128
```

## Inference & Timeline Generation

Predict a single recording to see the evidence JSON schema:
```bash
# Using Random Forest
python3 src/predict.py data/processed/raw_acc_25hz/<USER_ID>/<TIMESTAMP>.m_raw_acc.csv

# Using CNN
python3 src/predict_cnn.py data/processed/raw_acc_25hz/<USER_ID>/<TIMESTAMP>.m_raw_acc.csv --model artifacts/cnn/cnn_checkpoint.pt
```

Batch-predict a whole user's directory and build a timeline:
```bash
# Predict directory with CNN
python3 src/predict_directory_cnn.py \
  --input-dir data/processed/raw_acc_25hz/<USER_ID> \
  --output-dir artifacts/predictions_cnn/<USER_ID> \
  --model artifacts/cnn/cnn_checkpoint.pt

# Build the timeline
python3 src/build_timeline.py \
  --predictions-dir artifacts/predictions_cnn/<USER_ID> \
  --output artifacts/timelines/<USER_ID>_cnn.json
```

## Grounded Question Answering

Once you have a timeline, you can ask direct duration, verification, or counting questions:
```bash
python3 src/qa.py \
  --timeline artifacts/timelines/<USER_ID>_cnn.json \
  --question "How long was the user walking?"
```

## Evaluation

To evaluate the pipeline against ground truth:
```bash
# 1. Build Ground Truth Timeline
python3 src/build_ground_truth_timeline.py --user-id 0A986513-7828-4D53-AA1F-E02D6DF9561B --output artifacts/evaluation/ground_truth.json

# 2. Generate QA Cases
python3 src/generate_qa_cases.py --ground-truth-timeline artifacts/evaluation/ground_truth.json --output artifacts/evaluation/qa_cases.json

# 3. Evaluate your predicted timeline against the QA cases
python3 src/evaluate_qa.py --predicted-timeline artifacts/timelines/0A986513-7828-4D53-AA1F-E02D6DF9561B_cnn.json --qa-cases artifacts/evaluation/qa_cases.json --output artifacts/evaluation/qa_results.json
```

## Academic integrity

All external datasets, models, libraries, and AI assistance used during development will be cited and disclosed in the final report, as required by the challenge brief.
