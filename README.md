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

### Build the labelled training index

After resampling, create the activity-to-recording index with:

```bash
python3 src/build_labeled_index.py \
  --data-dir data \
  --resampled-dir data/processed/raw_acc_25hz \
  --output data/processed/raw_acc_training_index.csv
```

It retains only recordings with one of the six unambiguous labels available in
the supplied files: lying down, sitting, standing, walking, running, and
bicycling. It prints the class balance and never silently chooses a class for
an ambiguous row.

### Train the accelerometer baseline

The first baseline extracts 43 understandable time- and frequency-domain
features per recording, then fits a class-weighted Random Forest. It keeps
participants separate across splits. Install its small dependency set and run:

```bash
python3 -m pip install -r requirements.txt
python3 src/train_baseline.py \
  --validation-user 2C32C23E-E30C-498A-8DD2-0EFB9150A02E \
  --test-user 0A986513-7828-4D53-AA1F-E02D6DF9561B
```

The held-out users are intentional: all bicycling examples except one belong
to another participant, so this split retains enough rare examples for model
training. Metrics, the confusion matrix, feature importances, and the saved
model are written beneath `artifacts/baseline/`.

### Compare model size and inference cost

Train smaller Random Forest variants into distinct output directories. For
example, a medium operating point is:

```bash
.venv/bin/python src/train_baseline.py \
  --validation-user 2C32C23E-E30C-498A-8DD2-0EFB9150A02E \
  --test-user 0A986513-7828-4D53-AA1F-E02D6DF9561B \
  --trees 100 --max-depth 15 --min-samples-leaf 3 \
  --output-dir artifacts/rf_medium
```

Benchmark any trained variant on the same input recordings:

```bash
.venv/bin/python src/benchmark_inference.py \
  --model artifacts/rf_medium/random_forest.joblib \
  --input-dir data/processed/raw_acc_25hz/0A986513-7828-4D53-AA1F-E02D6DF9561B \
  --output artifacts/benchmarks/rf_medium.json
```

Use one benchmark JSON and one held-out QA-accuracy result for each operating
point in the later accuracy-versus-overhead plot.

### Create evidence records and a timeline

Predictions are stored as JSON rather than only printed, so later QA answers
can cite their exact source windows:

```bash
.venv/bin/python src/predict.py \
  data/processed/raw_acc_25hz/<USER_ID>/<TIMESTAMP>.m_raw_acc.csv \
  --output artifacts/predictions/<USER_ID>/<TIMESTAMP>.prediction.json
```

For a whole user, load the model once and batch-generate these records:

```bash
.venv/bin/python src/predict_directory.py \
  --input-dir data/processed/raw_acc_25hz/<USER_ID> \
  --output-dir artifacts/predictions/<USER_ID>
```

After producing prediction JSON files for a recording set, aggregate them:

```bash
.venv/bin/python src/build_timeline.py \
  --predictions-dir artifacts/predictions/<USER_ID> \
  --output artifacts/timelines/<USER_ID>.json
```

The timeline keeps every observed 20-second window. It may group nearby equal
predictions for navigation, but it separately records unobserved gaps; QA must
sum `observed_duration_s`, never the grouped span, when reporting evidence.

### Ask core activity questions

The first QA layer is deterministic and uses the timeline as its only source
of facts. It supports verification, duration, count, comparison, onset, and
identification at an observed time:

```bash
.venv/bin/python src/qa.py \
  --timeline artifacts/timelines/<USER_ID>.json \
  --question "How long was the user walking?"
```

It always prints the required answer fields and reports durations as directly
observed sensor seconds. An SLM can be added later to translate flexible
language into these same bounded operations; it will not be allowed to invent
activity labels or timestamps.

### Evaluate QA answers and grounding

Build a self-label timeline and deterministic labelled QA cases for a held-out
user. These are evaluation-only artifacts, never training data:

```bash
.venv/bin/python src/build_ground_truth_timeline.py \
  --user-id 0A986513-7828-4D53-AA1F-E02D6DF9561B \
  --output artifacts/evaluation/ground_truth_0A986.json

.venv/bin/python src/generate_qa_cases.py \
  --ground-truth-timeline artifacts/evaluation/ground_truth_0A986.json \
  --output artifacts/evaluation/qa_cases_0A986.json
```

After batch-predicting the same user and building its predicted timeline,
score answer correctness, evidence IoU, and fully grounded accuracy:

```bash
.venv/bin/python src/evaluate_qa.py \
  --predicted-timeline artifacts/timelines/0A986513-7828-4D53-AA1F-E02D6DF9561B.json \
  --qa-cases artifacts/evaluation/qa_cases_0A986.json \
  --output artifacts/evaluation/qa_results_0A986.json
```

The output has per-question-type answer accuracy, grounded accuracy, and mean
evidence IoU. It is the source data for the required accuracy-by-question-type
and accuracy-versus-strictness figures.

## Academic integrity

All external datasets, models, libraries, and AI assistance used during development will be cited and disclosed in the final report, as required by the challenge brief.
