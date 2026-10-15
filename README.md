# Phone-only activity recognition vs Vaizman et al. (2017)

B.Tech Project. A phone-only activity recogniser for the ExtraSensory dataset,
compared like for like with the dataset's reference method:

> Y. Vaizman, K. Ellis, G. Lanckriet. *Recognizing Detailed Human Context
> In-the-Wild from Smartphones and Smartwatches.* IEEE Pervasive Computing,
> 16(4), 2017.

We re-implement the paper's method and run it on **the same users, folds,
windows and labels** as ours. The task is six annotated body states: lying down,
sitting, standing, walking, running and bicycling.

The report is [`report/btp_report.tex`](report/btp_report.tex). The phase-by-phase
log is [`RESULTS.md`](RESULTS.md), and the plan is [`PLAN.md`](PLAN.md).

## Headline results

Five-fold cross-validation, user-disjoint, 35 users, 165,787 windows, raw test
labels, mean ± std over folds:

| System | Sensors | Accuracy | Balanced acc. | Macro-F1 |
|---|---|---|---|---|
| Paper, best fusion (LFA) | phone Acc+Gyro | 0.526 ± 0.012 | 0.401 ± 0.021 | 0.366 ± 0.020 |
| Paper, best fusion (LFA) | all 6 (incl. watch, audio, location) | 0.674 ± 0.034 | 0.502 ± 0.039 | 0.478 ± 0.030 |
| **Ours, per window** | phone Acc+Gyro | **0.696 ± 0.037** | **0.561 ± 0.071** | **0.566 ± 0.076** |
| **Ours + HMM (Viterbi, deployed)** | phone Acc+Gyro | **0.718 ± 0.037** | 0.555 ± 0.079 | 0.564 ± 0.078 |

On **5 held-out users** that no fold ever trained or tuned on, ours + HMM scores
0.741 accuracy and 0.496 macro-F1. The paper's method scores 0.650 / 0.409 with all
6 sensors and 0.571 / 0.349 with phone sensors.

**Is the reproduction faithful?** Its per-label balanced accuracy matches the
paper's Table S1 to within about 0.02 (bicycling under fusion within 0.04).

**On the paper's own metric** (per-label yes/no balanced accuracy):

| System | Score |
|---|---|
| Paper, phone only | 0.720 |
| Ours | 0.744 |
| Paper, all 6 sensors | 0.776 |

The paper's per-label detectors are built for that metric, and a rare label can
fire on its own threshold. Our model has to name one activity per window. See
§6.3 of the report.

**What drives the gain** (ablation, deployed model's macro-F1 margin, paired by
fold):

| Component | Gain | Folds |
|---|---|---|
| Temporal context | +0.133 | 5 of 5 |
| Early fusion over learned late fusion | +0.107 | 5 of 5 |
| XGBoost over logistic regression | +0.093 | 5 of 5 |
| Time of day | +0.020 | 5 of 5 |

The hierarchy, label cleaning and adding the gyroscope to the accelerometer are
within fold-to-fold noise.

Figures are in [`outputs/figures/`](outputs/figures/) (fig6–fig8 are the
comparison), and the JSON behind every number is in
[`outputs/evaluation/`](outputs/evaluation/).

## Run it

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The 35 preprocessed recordings, the trained models and the HMM transition
matrices are committed, so the demo runs straight away:

```bash
python run.py --recording 098A72A5-E3E5-4F54-A152-BBDA0DF7B694 --fold 4 \
              --question "How long was the user walking?"
```

```
Query: "How long was the user walking?"
Answer: 26344 seconds
Activity/Event: walking
Evidence:
    Timestamp(s): 19611 to 21666, 77502 to 80697, 94242 to 95877, 112601 to 116696,
                  167459 to 170234, 1236590 to 1239785 (+18 shorter intervals totalling 9394 s)
                  (seconds from start)
    Sensor Modality: Accelerometer, Gyroscope
    Sensor Channel(s): All
Explanation: Walking was detected in 24 intervals spanning 26344 s in total (439 min).
             The recording samples about 3% of wall-clock time, so this span of 26344 s
             rests on 6825 s of directly observed sensor data.
```

- **Pick the right fold.** Pass the fold that held the user out (`--fold`; see
  `outputs/folds_w15.json` or `tools/which_fold.py`). The default, fold 0, was
  trained on most users, so its answers for them are not a fair test.
- **Other useful flags:**
  - `--questions FILE` reads one question per line;
  - `--json` gives machine-readable output;
  - `--no-slm` skips the language model;
  - `--save-timeline FILE` keeps the intermediate activity timeline.

### Dashboard

```bash
python -m asqa.dashboard --open        # http://127.0.0.1:8000
```

The dashboard serves the five held-out users. Ask a question, then click an answer
to highlight the intervals it cited. It calls the same `answer_question()` as the
CLI, and `tests/test_dashboard.py` asserts the two agree.

### Things worth knowing

- **Dashboard ids and CLI ids are different sets.** The CLI reads the 35 cached
  users (`ls outputs/cache/w15/`). The dashboard reads the 5 held-out users.
- **First analysis of a dashboard recording takes 20–45 s.** After that it is cached.
- **The language model is optional.** The first question that needs it downloads
  about 6 GB (`Qwen/Qwen2.5-3B-Instruct`). Peak memory was 3.2 GB on Apple MPS;
  CPU-only machines need more. Keyword rules answer most questions in about
  0.16 ms without it. Disable it with `--no-slm`.
- **A raw accelerometer directory with no gyroscope beside it is accepted
  silently.** Accelerometer data then fills the gyroscope channels. Pass both
  sensors.

## Reproduce the results

```bash
python -m asqa.recognise --cv      # our model, 5 folds           -> recognition_cv.json
python -m asqa.decode --cv         # HMM decoding                 -> decode_cv.json
python -m asqa.paper_baseline      # the paper's method (~4 min)  -> paper_baseline_cv.json
python -m asqa.holdout             # 5 held-out users             -> holdout.json
python -m asqa.ablation            # ablation table               -> ablation_cv.json
python -m asqa.robustness          # noise / dropout / rate       -> robustness.json
python -m asqa.benchmark           # size and latency             -> benchmark.json
python -m asqa.figures             # every figure
python -m asqa.recognise --cv --no-clean   # optional: no label cleaning, saves no models
```

- **The paper baseline and the held-out evaluation need the ExtraSensory label
  files** (`data/<UUID>.features_labels.csv`), which hold the paper's
  precomputed features. They are not committed; see below.
- **The fold assignment is committed** in `outputs/folds_w15.json`.

### Rebuilding from raw data

Only needed to re-preprocess. Download from <http://extrasensory.ucsd.edu/>:

```
data/
  <UUID>.features_labels.csv          ExtraSensory.per_uuid_features_labels.zip
  Original Data/
    acc/<UUID>/<timestamp>.m_raw_acc.dat      ExtraSensory.raw_measurements.raw_acc.zip
    gyro/<UUID>/<timestamp>.m_proc_gyro.dat   ExtraSensory.raw_measurements.proc_gyro.zip
```

```bash
python -m asqa.preprocess              # raw .dat -> 25 Hz windows (~15 min, ~1.2 GB)
python -m asqa.splits                  # user-disjoint, rare-class-aware folds
python -m asqa.recognise --build-features
```

## Design

```
raw acc + gyro → clock-true resampling → 40 features + 63 context + 8 time of day
              → hierarchical XGBoost (+ calibration) → time-aware HMM → timeline → Q&A demo
```

| Module | Role |
|---|---|
| [`preprocess.py`](asqa/preprocess.py) | Resamples both sensors onto one 25 Hz grid by *timestamp*, not by sample index. The two clocks are skewed (dt 0.024–0.077 s). A window with a gap over 0.5 s is rejected. |
| [`features.py`](asqa/features.py) | 40 interpretable features: gravity tilt, motion energy, cadence, spectral shape, gyroscope rotation. |
| [`context.py`](asqa/context.py) | Rolling statistics over ±2…±30 neighbouring windows, never across gaps longer than 40 min. Also the paper's 8 time-of-day bins (San Diego local time). |
| [`recognise.py`](asqa/recognise.py) | Off-body label cleaning (training only); static vs dynamic, then one model per branch; temperature calibration. |
| [`decode.py`](asqa/decode.py) | HMM whose per-minute transitions are raised to the power Δt/60, so long gaps relax toward the stationary distribution. |
| [`paper_baseline.py`](asqa/paper_baseline.py) | The paper's method: its features, per-label logistic regression with its C search, EF / LFA / LFL fusion. |
| [`holdout.py`](asqa/holdout.py), [`ablation.py`](asqa/ablation.py) | The held-out evaluation and the ablation. |
| [`timeline.py`](asqa/timeline.py), [`answer.py`](asqa/answer.py), [`intent.py`](asqa/intent.py), [`slm.py`](asqa/slm.py) | The question-answering demo. Answers are computed from decoded intervals only. The language model only maps unfamiliar wording onto the class vocabulary. |

More detail is in [`ARCHITECTURE.md`](ARCHITECTURE.md).

## Tests

```bash
python -m tests.test_preprocess   # clock-true resampling (catches a 2.0 -> 2.3 Hz error)
python -m tests.test_context      # no context across gaps or recordings; time-of-day bins
python -m tests.test_answer       # question parsing and the output contract
python -m tests.test_intent       # time windows; validating model proposals
python -m tests.test_dashboard    # dashboard API; matches the CLI
```

All 56 pass with the committed data. No language model is needed.

## Known limitations

- **We use 35 of the paper's 60 users** (plus 5 held out). Per-fold std is about
  0.07, so differences under about 0.02 are noise.
- **Running stays hard** (0.18 recall, 0.34% of windows). Many running labels
  describe a phone that was not on the body.
- **Standing is still and upright like sitting** (0.32 recall).
- **Temporal context needs neighbours.** A lone 15 s window scores 0.564 accuracy.
- **We address six body states only.** The paper also models dozens of context
  labels that motion cannot capture.
- **PyTorch and XGBoost cannot share a process on this platform.** The language
  model therefore runs in its own worker ([`slm_worker.py`](asqa/slm_worker.py)).

## History

This repository began as *Ask the Sensors*, a 7-class grounded question-answering
system for the CS60055 Ubiquitous Computing hackathon, built with Devank Jain
and Aditya Srivastava. That system split ExtraSensory's single standing label
into two derived classes. The BTP made four changes:
- dropped the derived classes (6 annotated classes);
- added time of day and label cleaning;
- re-implemented the reference method;
- reframed the evaluation as this comparison.

The challenge's question-answering scores are no longer reported.

## Citations

- **Dataset and reference method:** Vaizman, Ellis and Lanckriet, IEEE Pervasive
  Computing 2017. <http://extrasensory.ucsd.edu/>
- **Language model:** Qwen2.5-3B-Instruct (Apache 2.0). `ASQA_SLM_MODEL`
  selects another.
- **Libraries:** NumPy, SciPy, scikit-learn, XGBoost, pandas, PyTorch, Hugging
  Face Transformers, Matplotlib.
