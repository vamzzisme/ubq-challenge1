# Results log

Phase-by-phase results, for book-keeping.

Setup for every row unless stated otherwise:
- 35 ExtraSensory users, 5-fold cross-validation, no user in both train and test (`outputs/folds_w15.json`).
- Values are mean ± std over the 5 folds.
- Balanced accuracy is the mean per-class recall.

## Baseline: original challenge system (7 classes, with derived standing split)

| Stage | Accuracy | Macro-F1 |
|---|---|---|
| Per window, no context | 0.457 ± 0.034 | 0.407 ± 0.052 |
| + temporal context | 0.660 ± 0.035 | 0.529 ± 0.060 |
| + HMM decoding | 0.703 ± 0.032 | 0.525 ± 0.061 |

Per-class F1 with context: lying 0.776, sitting 0.702, bicycling 0.644, walking 0.508, standing and moving 0.449, running 0.398, standing in place 0.227.

## Phase 1: drop the derived class (7 → 6 annotated classes)

| Stage | Accuracy | Balanced acc. | Macro-F1 |
|---|---|---|---|
| Per window, no context | 0.525 ± 0.034 | 0.457 ± 0.060 | 0.419 ± 0.065 |
| + temporal context | 0.665 ± 0.038 | 0.557 ± 0.075 | 0.551 ± 0.074 |
| + HMM decoding | 0.693 ± 0.037 | not computed | 0.570 ± 0.077 |

Per-class F1 with context: lying 0.775, sitting 0.712, bicycling 0.621, walking 0.496, running 0.395, standing 0.307.

Note: the 7-class and 6-class numbers are not directly comparable, because the class sets differ.

## Phase 2: time-of-day features (paper's 8 overlapping 6-hour bins, San Diego time)

| Stage | Accuracy | Balanced acc. | Macro-F1 |
|---|---|---|---|
| Per window, no context | 0.525 ± 0.034 | 0.457 ± 0.060 | 0.419 ± 0.065 |
| + temporal context + time of day | **0.695 ± 0.038** | **0.564 ± 0.067** | **0.561 ± 0.070** |
| + HMM decoding | **0.715 ± 0.038** | not computed | 0.558 ± 0.074 |

- Per-class F1 with context: lying 0.831, sitting 0.734, bicycling 0.609, walking 0.500, running 0.360 (±0.33 across folds), standing 0.334.
- A single window with no neighbours: 0.483 → 0.562 accuracy.
- Gain over Phase 1 (with context, per window): accuracy +0.030, balanced accuracy +0.007, macro-F1 +0.010.

## Phase 3: label cleaning (training and validation only)

- **The paper's rules are already in the data.** ExtraSensory's `FIX_walking` and `FIX_running` columns are its corrected labels. An audit found zero walking or running windows that also carry a vehicle label.
- **Our rule:** walking, running and bicycling windows with body-acceleration energy at or below the 90th percentile of training lying-down windows (≈0.008 g, fitted per fold) are dropped. In these windows the phone was not moving with the person.
- **How much this drops:** about 17% of walking, 25% of running and 11% of bicycling windows. That's 1,549–2,213 training windows per fold.
- **Headline numbers use raw test labels.**

| Stage | Accuracy | Balanced acc. | Macro-F1 |
|---|---|---|---|
| Per window, no context | 0.528 ± 0.038 | 0.461 ± 0.058 | 0.433 ± 0.058 |
| + temporal context + time of day | 0.696 ± 0.037 | 0.561 ± 0.071 | 0.566 ± 0.076 |
| + HMM, Viterbi (deployed) | **0.718 ± 0.037** | not computed | 0.564 ± 0.078 |
| + HMM, posterior | 0.716 ± 0.037 | not computed | **0.583 ± 0.078** |
| With context, on cleaned test labels (secondary) | 0.707 | 0.595 | 0.583 |

- Per-class F1 with context: lying 0.832, sitting 0.735, bicycling 0.629, walking 0.499, running 0.369 (±0.32), standing 0.334.
- Gain over Phase 2 (with context, per window, raw test): accuracy +0.001, balanced accuracy −0.003, macro-F1 +0.005. On raw labels the gain is within noise. Cleaning mainly helps on windows whose labels are trustworthy.
- Running is unchanged at 0.18 recall (pooled), because the raw test labels still contain off-body running.
- The no-cleaning run for the Phase 5 ablation is `python -m asqa.recognise --cv --no-clean`. It writes `recognition_cv_noclean.json` and saves no models.

## Phase 4: the paper's method, reproduced on our data

`python -m asqa.paper_baseline` writes `outputs/evaluation/paper_baseline_cv.json`. It takes about 4 minutes.

The paper's pipeline is reproduced exactly:
- its published per-sensor features, which match the paper's counts: Acc 26, Gyro 26, WAcc 46, Loc 17, Aud 26, PS 34;
- per-sensor standardisation;
- one binary logistic regression per label with balanced class weights, C chosen by F1 on an internal third of the training data;
- EF, LFA and LFL fusion.

It runs on the same 35 users, 5 folds, windows and 6 raw labels as ours. A window's multiclass prediction is the label with the highest probability.

**Check that the reproduction is faithful:** per-label binary balanced accuracy against the paper's Table S1 (ours / paper):

| System | Lying | Sitting | Standing | Walking | Running | Bicycling |
|---|---|---|---|---|---|---|
| Acc | 0.73 / 0.72 | 0.62 / 0.63 | 0.60 / 0.60 | 0.76 / 0.77 | 0.71 / 0.69 | 0.81 / 0.81 |
| LFL, 6 sensors | 0.88 / 0.88 | 0.77 / 0.75 | 0.66 / 0.68 | 0.80 / 0.80 | 0.71 / 0.71 | 0.84 / 0.87 |

Nearly every cell is within 0.02 (bicycling up to 0.04), even though we use 35 of their 60 users.

### Multiclass comparison (same test windows, raw labels, mean ± std over folds)

| System | Sensors | Accuracy | Balanced acc. | Macro-F1 |
|---|---|---|---|---|
| Paper EF | phone Acc+Gyro | 0.479 ± 0.043 | 0.437 ± 0.051 | 0.348 ± 0.018 |
| Paper LFA | phone Acc+Gyro | 0.526 ± 0.012 | 0.401 ± 0.021 | 0.366 ± 0.020 |
| Paper LFL | phone Acc+Gyro | 0.505 ± 0.015 | 0.441 ± 0.066 | 0.330 ± 0.023 |
| Paper EF | all 6 (incl. watch, audio, location, phone state) | 0.662 ± 0.040 | 0.527 ± 0.056 | 0.467 ± 0.023 |
| Paper LFA | all 6 | 0.674 ± 0.034 | 0.502 ± 0.039 | 0.478 ± 0.030 |
| Paper LFL | all 6 | 0.673 ± 0.046 | 0.523 ± 0.059 | 0.468 ± 0.023 |
| **Ours, no context** | phone Acc+Gyro | 0.528 ± 0.038 | 0.461 ± 0.058 | 0.433 ± 0.058 |
| **Ours, context + time of day** | phone Acc+Gyro | **0.696 ± 0.037** | **0.561 ± 0.071** | **0.566 ± 0.076** |
| **Ours + HMM (deployed)** | phone Acc+Gyro | **0.718 ± 0.037** | not computed | 0.564 ± 0.078 |

- With the same two phone sensors, ours beats every fusion variant of the paper's method on all three metrics. That holds even without temporal context (macro-F1 0.433 vs 0.366 at best).
- With context, ours also beats the paper's full 6-sensor system (macro-F1 0.566 vs 0.478) while using only the phone's motion sensors.

### On the paper's own metric (per-label binary BA, mean over folds)

| System | Lying | Sitting | Standing | Walking | Running | Bicycling | Mean |
|---|---|---|---|---|---|---|---|
| Paper LFA, Acc+Gyro | 0.73 | 0.63 | 0.60 | 0.78 | 0.76 | 0.83 | 0.720 |
| Paper LFA, all 6 | 0.87 | 0.76 | 0.65 | 0.79 | 0.75 | 0.83 | **0.776** |
| Ours, no context | 0.70 | 0.65 | 0.52 | 0.77 | 0.62 | 0.78 | 0.674 |
| Ours, context + time of day | 0.87 | 0.76 | 0.62 | 0.78 | 0.65 | 0.78 | 0.744 |

- On this metric ours beats the paper's phone-only method (0.744 vs 0.720) but trails its 6-sensor system (0.744 vs 0.776). The gap comes from running, bicycling and standing.
- The reason is that the paper trains a separate yes/no detector for each label, each with its own 0.5 threshold. That design is tuned for this metric, and a rare label can fire even when another label is more likely.
- Our model picks one winning class per window, so a rare class only scores when it beats every other class. That is the right behaviour for a 6-way decision, and it's why ours wins on the multiclass metrics.

## Comparison with Vaizman et al. (2017)

Mean per-class recall over the paper's 5 classes. Each class's recall is pooled over the 5 test folds; the paper's values come from its Fig. 4B/C.

| Class | Paper: phone Acc+Gyro | Paper: + watch | Ours, Phase 2, no context | Ours, Phase 2, with context | Ours, Phase 3, with context |
|---|---|---|---|---|---|
| Lying down | 0.79 | 0.82 | 0.51 | 0.81 | 0.82 |
| Sitting | 0.44 | 0.67 | 0.62 | 0.73 | 0.73 |
| Walking | 0.51 | 0.56 | 0.67 | 0.65 | 0.64 |
| Running | 0.19 | 0.23 | 0.17 | 0.18 | 0.18 |
| Bicycling | 0.50 | 0.66 | 0.57 | 0.58 | 0.58 |
| **Mean** | **0.49** | **0.59** | **0.51** | **0.59** | **0.59** |

Caveats:
- Our recall is measured before HMM decoding.
- The paper's matrices come from a separate model trained only on its 5 classes. Our model chooses among 6, so it can also err by predicting standing.

## Classifier and fusion ablation (7-class system, context features, same folds)

| Model | Accuracy | Macro-F1 |
|---|---|---|
| Early fusion, hierarchical XGBoost (ours) | 0.660 | 0.529 |
| Early fusion, flat XGBoost | 0.657 | 0.512 |
| Early fusion, logistic regression (paper's classifier) | 0.492 | 0.397 |
| Accelerometer only, XGBoost | 0.645 | 0.503 |
| Gyroscope only, XGBoost | 0.551 | 0.436 |
| Late fusion, average (LFA) | 0.643 | 0.509 |
| Late fusion, learned weights (LFL) | 0.538 | 0.392 |

To be re-run on 6 classes in Phase 5.
