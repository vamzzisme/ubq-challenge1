# Pipeline: Raw `.dat` → Processed CSV → Training → Prediction

Full command reference for processing ExtraSensory data end-to-end.

---

## Prerequisites

```bash
# Create & activate virtualenv (one-time)
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Step 0 — Place raw data

Put each user's raw `.dat` recordings and label files in the right locations:

```
data/Original Data/<USER_ID>/          ← raw .dat files (e.g. 1444079161.m_raw_acc.dat)
data/<USER_ID>.features_labels.csv     ← self-labelled metadata CSV
```

**User IDs currently available:**
- `00EABED2-271D-49D8-B599-1D4A09240601`
- `0A986513-7828-4D53-AA1F-E02D6DF9561B`
- `0BFC35E2-4817-4865-BFA7-764742302A2D`
- `0E6184E1-90C0-48EE-B25A-F1ECB7B9714E`
- `1DBB0F6F-1F81-4A50-9DF4-CD62ACFA4842`
- `2C32C23E-E30C-498A-8DD2-0EFB9150A02E`

---

## Step 1 — Convert & resample raw accelerometer `.dat` → CSV (25 Hz)

**Status: ✅ DONE**

```bash
.venv/bin/python src/prepare_sensor_data.py \
  --input-dir "data/Original Data" \
  --output-dir data/processed \
  --modality raw_acc
```

**What it does:**
- Reads 4-column `.dat` files (timestamp, x, y, z)
- Writes native-rate CSV to `data/processed/raw_acc_csv/<USER_ID>/`
- Resamples 800 samples @ 40 Hz → 500 samples @ 25 Hz into `data/processed/raw_acc_25hz/<USER_ID>/`
- Creates manifest: `data/processed/raw_acc_manifest.csv`

---

## Step 2 — Build labelled training index (accelerometer)

**Status: ✅ DONE**

```bash
.venv/bin/python src/build_labeled_index.py \
  --data-dir data \
  --resampled-dir data/processed/raw_acc_25hz \
  --output data/processed/raw_acc_training_index.csv
```

**What it does:**
- Matches each resampled recording to its activity label from `<USER_ID>.features_labels.csv`
- Keeps only unambiguous labels: lying_down, sitting, standing, walking, running, bicycling
- Writes: `data/processed/raw_acc_training_index.csv`

---

## Step 3 — Train baseline model (accelerometer only)

**Status: ✅ DONE**

```bash
.venv/bin/python src/train_baseline.py \
  --validation-user 2C32C23E-E30C-498A-8DD2-0EFB9150A02E \
  --test-user 0A986513-7828-4D53-AA1F-E02D6DF9561B
```

**What it does:**
- Extracts 43 time/frequency features per recording
- Trains a 300-tree class-weighted Random Forest
- Leave-one-user-out split (4 train, 1 val, 1 test)
- Outputs to `artifacts/baseline/`:
  - `random_forest.joblib` — saved model
  - `metrics.json` — accuracy, F1, per-class report
  - `test_confusion_matrix.csv`
  - `feature_importance.csv`

**Current results:**
| Split      | Accuracy | Macro F1 |
|------------|----------|----------|
| Validation | 0.469    | 0.213    |
| Test       | 0.527    | 0.286    |

---

## Step 4 — Run prediction on a single recording

```bash
.venv/bin/python src/predict.py <path_to_resampled_csv>

# Example:
.venv/bin/python src/predict.py \
  data/processed/raw_acc_25hz/0A986513-7828-4D53-AA1F-E02D6DF9561B/1449601855.m_raw_acc.csv
```

---

## 🔜 TODO: Gyroscope data processing

Repeat Steps 1–3 for gyroscope data when `.dat` files with `proc_gyro` modality are available.

### Step 1g — Convert & resample gyro `.dat` → CSV

```bash
.venv/bin/python src/prepare_sensor_data.py \
  --input-dir "data/Original Data" \
  --output-dir data/processed \
  --modality proc_gyro
```

> **Note:** Check the actual filename token in the gyro `.dat` files.
> If files are named `1444079161.m_proc_gyro.dat`, use `--modality proc_gyro`.
> Adjust `--source-rate-hz` and `--expected-source-samples` if the gyro
> sampling rate differs from 40 Hz / 800 samples.

**Expected output:**
- `data/processed/proc_gyro_csv/<USER_ID>/` — native-rate CSVs
- `data/processed/proc_gyro_25hz/<USER_ID>/` — resampled 25 Hz CSVs
- `data/processed/proc_gyro_manifest.csv`

### Step 2g — Build labelled training index (gyro)

```bash
.venv/bin/python src/build_labeled_index.py \
  --data-dir data \
  --resampled-dir data/processed/proc_gyro_25hz \
  --output data/processed/proc_gyro_training_index.csv
```

### Step 3g — Train with 6-axis features (acc + gyro)

> **TODO:** Modify `signal_features.py` and `train_baseline.py` to:
> 1. Load both acc and gyro CSVs per recording
> 2. Extract features from all 6 axes (acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z)
> 3. Concatenate into a wider feature vector
> 4. Retrain the Random Forest (or try a new model)

---

## Quick-reference: full pipeline from scratch

```bash
# 1. Setup
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Process accelerometer
.venv/bin/python src/prepare_sensor_data.py \
  --input-dir "data/Original Data" --output-dir data/processed --modality raw_acc

# 3. Build acc training index
.venv/bin/python src/build_labeled_index.py \
  --data-dir data --resampled-dir data/processed/raw_acc_25hz \
  --output data/processed/raw_acc_training_index.csv

# 4. Train
.venv/bin/python src/train_baseline.py \
  --validation-user 2C32C23E-E30C-498A-8DD2-0EFB9150A02E \
  --test-user 0A986513-7828-4D53-AA1F-E02D6DF9561B

# 5. Predict
.venv/bin/python src/predict.py data/processed/raw_acc_25hz/<USER_ID>/<TIMESTAMP>.m_raw_acc.csv
```
