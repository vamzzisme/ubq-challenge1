# BTP Plan: ExtraSensory activity recognition vs Vaizman et al. (2017)

Reference paper: Y. Vaizman, K. Ellis, G. Lanckriet, *Recognizing Detailed Human Context In-the-Wild from Smartphones and Smartwatches*, IEEE Pervasive Computing 16(4), 2017.

## Goal

Compare our recogniser with the paper's method on the **same users, folds and classes**, then show where ours improves on it. The challenge Task 1–4 QA scores are no longer reported. The dashboard, language model (Qwen 2.5 3B) and QA pipeline stay as a working demo.

## Constraints

- No new data. We use only:
  - the 35 cached training users in `outputs/cache/w15/`;
  - their label files `data/*.features_labels.csv`, which contain all ExtraSensory labels and the paper's precomputed features;
  - the 5 held-out users in `data/5 new users/`.
- All splits are user-disjoint (the 5-fold assignment in `asqa/splits.py`).
- Report only legitimate metrics: no cherry-picked folds and no test-set tuning.

## Where we start (current system, 7 classes)

| Stage | Accuracy | Macro-F1 |
|---|---|---|
| Per-window, no context | 0.457 ± 0.034 | 0.407 ± 0.052 |
| + temporal context | 0.660 ± 0.035 | 0.529 ± 0.060 |
| + HMM decoding (deployed) | 0.703 ± 0.032 | 0.525 ± 0.061 |

Ablation already run (context-aware features, same folds):

| Model | Accuracy | Macro-F1 |
|---|---|---|
| EF, hierarchical XGBoost (ours) | 0.660 | 0.529 |
| EF, flat XGBoost | 0.657 | 0.512 |
| EF, logistic regression | 0.492 | 0.397 |
| Late fusion, average (LFA) | 0.643 | 0.509 |
| Late fusion, learned (LFL) | 0.538 | 0.392 |

Mean per-class recall over the paper's 5 classes (lying, sitting, walking, running, bicycling):

| Paper, phone Acc+Gyro | Paper, + watch | Ours, no context | Ours, with context |
|---|---|---|---|
| 0.49 | 0.59 | 0.42 | 0.56 |

## Phases

### 1. Drop the derived class (≈1.5 h) — DONE

6-class results (5-fold, user-disjoint):

| Stage | Accuracy | Balanced acc. | Macro-F1 |
|---|---|---|---|
| Per-window, no context | 0.525 ± 0.034 | 0.457 ± 0.060 | 0.419 ± 0.065 |
| + temporal context | 0.665 ± 0.038 | 0.557 ± 0.075 | 0.551 ± 0.074 |
| + HMM (Viterbi, deployed) | 0.693 ± 0.037 | not computed | 0.570 ± 0.077 |

Per-class F1 with context: lying 0.775, sitting 0.712, bicycling 0.621, walking 0.496, running 0.395, standing 0.307.

- Merge `standing_in_place` and `standing_and_moving` back into `standing`, giving 6 annotated classes.
- Remove `StandingSplit`, `fit_standing_split` and `PROVENANCE_DERIVED`.
- Update `config.py`, `recognise.py`, `answer.py`, `dashboard.py`, `slm_worker.py` and `tools/parse_strategies.py`.
- Retrain and record the new 6-class baseline.

### 2. Time-of-day features (≈1 h)
- Compute from the cached `epoch_ts`; no download needed.
- Use the paper's encoding: San Diego local hour, 8 half-overlapping bins (00–06, 03–09, 06–12, 09–15, 12–18, 15–21, 18–24, 21–03).
- Target: the lying vs sitting confusion.

### 3. Label cleaning (≈2.5 h)
- Paper rules, using the other labels in the label files:
  - walking and running are false when on a bus, in a car, driving, motorbike, skateboarding or at the pool;
  - running is also false during baseball or frisbee.
- Our rule: drop walking and running windows whose body-acceleration energy is at lying level (phone not on the person).
- Clean training data only. Report test results on both raw and cleaned labels.

### 4. Reproduce the paper on our data (≈3.5 h)
- Paper pipeline:
  - standardise features;
  - logistic regression with balanced class weights;
  - choose C from {0.001 … 100} by grid search, scored on F1 over a held-out third of the training data.
- Use the paper's precomputed Acc and Gyro features from the label files.
- Fusion: EF, LFA, LFL.
- Same 35 users, 5 folds and 6 classes as ours.
- Optional: add their watch-accelerometer, audio and phone-state features to show what extra sensors give.

### 5. Evaluation (≈2.5 h)
- Metrics: accuracy, balanced accuracy, mean per-class recall, macro-F1 and per-class F1, as mean ± std over the 5 folds.
- Score the paper's method and ours on identical test windows.
- Final check on the 5 held-out users, which are never used for training or tuning.
- Ablation: no context → + context → + hierarchy → + HMM → + time-of-day → + cleaning; also logistic regression vs XGBoost, and EF vs LFA vs LFL.
- Show our confusion matrices next to the paper's Fig. 4B/C.

### 6. Demo upkeep (≈1 h)
- Drop the Task 1–4 QA scores from reported results.
- Keep `evaluate.py` and `rubric.py` in the code, but don't use them for headline numbers.
- Make the dashboard, QA pipeline and language model work with 6 classes, then smoke-test end to end.

### 7. Write-up (≈2.5 h)
- Rewrite README and ARCHITECTURE as a paper-vs-ours comparison: tables, figures and an explicit list of differences:
  - we predict one label from 6 classes; the paper runs a separate yes/no model per label;
  - the paper's 87% accuracy is an average over yes/no labels, so it is not comparable to our 6-class accuracy;
  - we use phone sensors only; the paper also uses a watch.

## Timeline

| Phase | Estimate |
|---|---|
| 1. Drop derived class | 1.5 h |
| 2. Time-of-day | 1 h |
| 3. Label cleaning | 2.5 h |
| 4. Paper reproduction | 3.5 h |
| 5. Evaluation | 2.5 h |
| 6. Demo upkeep | 1 h |
| 7. Write-up | 2.5 h |
| **Total** | **≈14 h (2 focused days, about 3 with retraining runs and review)** |

Phases 1–3 come first, because every later number depends on them.
