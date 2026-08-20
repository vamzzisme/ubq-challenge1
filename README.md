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

Place each user's raw `.dat` recordings under `data/<USER_ID>/` and its self-label file at `data/<USER_ID>.features_labels.csv`. The raw dataset is intentionally excluded from version control.

### Convert and resample a modality

The supplied files currently live directly under `data/`. Convert the raw
accelerometer captures to readable CSV and create their 25 Hz versions with:

```bash
python3 src/prepare_sensor_data.py --input-dir data --output-dir data/processed --modality raw_acc
```

This writes an 800-row native CSV and a 500-row resampled CSV for every
recording, plus `data/processed/raw_acc_manifest.csv`. When gyroscope files
arrive, run the same command with the gyro filename token, for example
`--modality proc_gyro` if its filenames end in `.m_proc_gyro.dat`.

## Academic integrity

All external datasets, models, libraries, and AI assistance used during development will be cited and disclosed in the final report, as required by the challenge brief.
