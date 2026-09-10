# Ask the Sensors

Grounded, explainable activity question answering from wearable accelerometer and
gyroscope signals. CS60055 Ubiquitous Computing, Hackathon Challenge 1.

Given a sensor recording and a natural-language question, the system returns a
structured answer that cites the specific stretch of signal supporting it.

```
Query: "How long was the user walking?"
Answer: 11559 seconds
Activity/Event: walking
Evidence:
    Timestamp(s): 25485 to 26160, 28545 to 30300, 163563 to 165558, 183483 to 184278,
                  202223 to 202958, 255083 to 256247 (+11 shorter intervals totalling 4440 s)
                  (seconds from start)
    Sensor Modality: Accelerometer, Gyroscope
    Sensor Channel(s): All
Explanation: Walking was detected in 17 intervals spanning 11559 s in total (193 min).
             The recording samples about 23% of wall-clock time, so this span rests on
             3075 s of directly observed sensor data.
```

**All timestamps are seconds from the start of the recording.**

## Results

Five-fold, user-disjoint cross-validation over 35 ExtraSensory users
(165,787 windows, 7 activity classes). No user appears in both the training and
evaluation side of any fold.

### Activity recognition

| stage | accuracy | macro-F1 |
|---|---|---|
| per-window, no temporal context | 0.457 ± 0.034 | 0.407 ± 0.052 |
| per-window, with temporal context | 0.660 ± 0.035 | 0.529 ± 0.060 |
| **+ HMM decoding (deployed)** | **0.703 ± 0.032** | 0.525 ± 0.061 |

Per-class F1 (context-aware): lying down 0.776, sitting 0.702, bicycling 0.644,
walking 0.508, standing and moving 0.449, running 0.398, standing in place 0.227.

### Question answering

1,159 generated cases, scored under the rule each answer type deserves.

| question type | n | answer accuracy | grounded accuracy |
|---|---|---|---|
| verification | 245 | 0.902 | 0.445 |
| open-world | 70 | 0.986 | 0.686 |
| comparison | 34 | 0.882 | 0.500 |
| identification | 228 | 0.474 | 0.360 |
| grounding | 194 | 0.330 | 0.227 |
| duration | 194 | 0.253 | 0.144 |
| count | 194 | 0.144 | 0.072 |
| **overall (macro across types)** | **1159** | **0.567** | **0.348** |

Correctness rules: categorical answers by exact match; durations and onsets
within `max(30 s, 10%)`; counts within ±1; *grounded* requires the answer to be
correct **and** the cited interval to reach IoU ≥ 0.5 **and** the modality and
channels to match.

### Cost

Measured on an arm64 laptop CPU, single process.

| configuration | accuracy | size on disk | median latency | tree nodes |
|---|---|---|---|---|
| full (deployed) | 0.690 | 9.10 MB | 0.37 ms | 214,421 |
| pruned | 0.689 | 8.05 MB | 0.48 ms | 191,702 |
| **shallow (edge)** | 0.547 | **0.95 MB** | 0.36 ms | 11,322 |
| context-free | 0.442 | 4.69 MB | 0.32 ms | 110,998 |

Feature extraction adds 0.33 ms per window. `shallow` is the edge operating
point: a 9.6× size reduction for 14 accuracy points.

The open-world language model (Qwen2.5-0.5B-Instruct) is reported separately
because it dominates: 494M parameters, 1,544 MB peak RSS, 8.9 s per query
including weight loading — roughly 24,000× the cost of classifying a window.
Only Task 4 questions that do not map onto the seven classes invoke it.

Figures are in [`outputs/figures/`](outputs/figures/); the JSON behind every
number is in [`outputs/evaluation/`](outputs/evaluation/).

## Design

The system is four layers, as the brief suggests, with a decoder between the
second and third.

```
L1 preprocess  →  L2 recognise  →  HMM decode  →  L3 aggregate  →  L4 interface
 clock-true       P(activity|w)    Viterbi       intervals,       question → operation
 25 Hz windows    XGBoost          time-aware    durations        → required format
```

**L1 — [`asqa/preprocess.py`](asqa/preprocess.py).** Resamples by *sensor
timestamp*, not sample index. The accelerometer clock is genuinely uneven
(measured dt 0.024–0.077 s) and 800 samples spans anywhere from ~9 s to ~25 s
depending on the device, so index-based resampling compresses the window and
misaligns the two modalities. Acc and gyro are interpolated onto one shared grid
covering only the interval both actually observed.

**L2 — [`asqa/features.py`](asqa/features.py), [`asqa/recognise.py`](asqa/recognise.py).**
Physics-based features (gravity separation, tilt, autocorrelation cadence, jerk,
spectral band powers) feeding a three-node hierarchy: static vs dynamic, then
posture, then gait. Each node's decision variable is a nameable physical
quantity, which is what lets the interface layer explain an answer rather than
assert it.

**Temporal context — [`asqa/context.py`](asqa/context.py).** Time-bounded rolling
statistics over neighbouring windows. This is the single largest accuracy lever
(+9 points): a 15-second window of stillness cannot distinguish lying from
sitting, but a stretch of them can.

**Decoder — [`asqa/decode.py`](asqa/decode.py).** A time-aware HMM. The
per-minute transition matrix is raised to the power `Δt/60`, so a multi-hour
dropout relaxes toward the stationary distribution instead of asserting that the
user kept doing the same thing.

**L3 — [`asqa/timeline.py`](asqa/timeline.py).** Merges windows into intervals
and separates *observed* time from *spanned* time. ExtraSensory observes ~15 s in
every 60, so a bout spanning 300 s rests on ~75 s of signal.

**L4 — [`asqa/answer.py`](asqa/answer.py), [`asqa/slm.py`](asqa/slm.py).** Routes
a question to one of six operations. Every timestamp, modality and channel it
prints is copied from an interval the earlier layers produced. For open-world
questions the language model picks from a numbered menu of real intervals; an id
outside that menu is discarded rather than rendered.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Data

The ExtraSensory dataset (Vaizman et al., 2017) is **not committed**. Download
from <http://extrasensory.ucsd.edu/> and arrange as:

```
data/
  <UUID>.features_labels.csv          from ExtraSensory.per_uuid_features_labels.zip (215 MB)
  Original Data/
    acc/<UUID>/<timestamp>.m_raw_acc.dat      from ExtraSensory.raw_measurements.raw_acc.zip
    gyro/<UUID>/<timestamp>.m_proc_gyro.dat   from ExtraSensory.raw_measurements.proc_gyro.zip
```

The reported results use 35 users. The pipeline works with any subset present.

## Reproducing the results

```bash
python -m asqa.preprocess              # raw .dat -> 25 Hz windows, cached per user
python -m asqa.splits                  # rare-class-aware, user-disjoint folds
python -m asqa.recognise --build-features
python -m asqa.recognise --cv          # trains context-free + context-aware per fold
python -m asqa.decode --cv             # HMM decoding, both variants, both methods
python -m asqa.evaluate --cv           # QA scoring by question type
python -m asqa.benchmark --fold 3      # size, latency, memory, operating points
python -m asqa.robustness --fold 3     # noise / dropout / sampling-rate sweeps
python -m asqa.figures                 # the five required figures
```

Preprocessing takes roughly 15 minutes for 35 users and produces ~1.2 GB of
cache. Everything downstream reads that cache.

Fold assignment is committed (`outputs/folds_w15.json`) so the split behind the
reported numbers is exactly reproducible.

## Explore it interactively

```bash
python -m asqa.dashboard --open        # http://127.0.0.1:8000
```

By default the picker offers only the **held-out recordings** in
`data/5 new users/` — five users that appear in no model's training set, so what
you see is how the system behaves on a genuinely new person. Pick one and ask
questions in plain language. Clicking any answer highlights, on a timeline of the whole recording,
exactly the intervals that answer cited — hovering a segment shows its measured
energy, cadence, tilt and rotation.

Two details worth knowing:

- **The model is chosen for you.** Never-seen users are answered by fold 3 (all
  five folds are equally valid for them; fold 3 scored highest on this corpus).
  Training users, if served with `--training-users`, are answered by the fold
  that held them out. Either way a recording is never questioned with a model
  that trained on it.
- **First analysis of a recording takes 20-45 s** (a week of raw signal at
  25 Hz); it is cached to disk after that.
- **Deep links work.** `?user=0A986513&q=How long was the user walking?` opens
  straight into that view, which is convenient for a demo.

The dashboard adds no dependency (stdlib `http.server`) and contains no
answering logic of its own: it calls the same `answer_question()` the CLI does,
and `tests/test_dashboard.py` asserts the two agree exactly.

## Answering questions about a recording

```bash
python run.py --recording <path|user-id> --questions tests/questions_all_tiers.txt
python run.py --recording <path|user-id> --question "How long was the user walking?"
```

`--recording` accepts a directory of raw `.dat` files, a single `.m_raw_acc.dat`
file (the Task 1 single-window case), or a preprocessed user id. Useful flags:
`--json` for machine-readable output, `--output FILE` to write to disk,
`--save-timeline FILE` to keep the intermediate activity timeline, and
`--no-slm` to disable the language model.

## Tests

```bash
python -m tests.test_preprocess   # clock-true resampling; catches the 2.000 -> 2.300 Hz error
python -m tests.test_context      # no cross-recording bleed, no context across dropouts
python -m tests.test_answer       # question parsing and the required output contract
python -m tests.test_dashboard    # dashboard API contract; asserts it matches the CLI
```

## Repository layout

```
asqa/          the system, one module per layer
run.py         the runnable entry point
tools/         which_fold.py — which model is safe for a given user
tests/         regression tests and an all-tier question set
outputs/       models, timelines, evaluation JSON, figures (regenerable)
data/          local data only; never committed
src/           LEGACY first implementation, kept for the before/after comparison
```

## Known limitations

Stated plainly, because several are properties of the data rather than of the
system.

- **`standing_in_place` is weak (0.227 F1).** It is a *derived* class: ExtraSensory
  annotates one `OR_standing` label, and the brief asks for two standing classes,
  so the split is made on measured body-acceleration energy with a threshold
  fitted on training users only. Rows carry a provenance marker so this class can
  be scored separately from the five annotated ones.
- **Counts and durations score lowest.** Both depend on bout boundaries, so a
  fragmented prediction hurts them more than it hurts a yes/no answer.
- **Running remains hard (0.398 F1, high variance).** It is 0.34% of labelled
  windows. Worse, some running-labelled windows contain no motion at all — one
  user's 68 running windows have the same body-acceleration energy as lying down
  (0.003 g) while their walking reads 0.333 g, i.e. the phone was not on them.
- **Lying vs sitting is the dominant residual confusion.** When the phone is
  off-body and still, the two are not separable from accelerometer geometry
  alone. Temporal context substantially mitigates this but does not remove it.
- **The 0.5B language model is a weak reasoner.** Asked to judge "a wheeled or
  pedal-based mode of movement" it answered "Pedal-based mode" while citing a
  *sitting* interval. Behaviour phrases that map onto the seven classes are
  therefore resolved deterministically; the model is used for prose, not verdicts.
- **PyTorch and XGBoost cannot share an interpreter here** — importing torch into
  a process that has run XGBoost segfaults on this platform, on both MPS and CPU.
  The language model runs in an isolated worker process
  ([`asqa/slm_worker.py`](asqa/slm_worker.py)).

## Citations

- **Dataset.** Y. Vaizman, K. Ellis, G. Lanckriet. "Recognizing Detailed Human
  Context In-the-Wild from Smartphones and Smartwatches." *IEEE Pervasive
  Computing*, 2017. <http://extrasensory.ucsd.edu/>
- **Language model.** Qwen2.5-0.5B-Instruct, Alibaba Cloud (Apache 2.0), used for
  Task 4 open-world reasoning.
- **Libraries.** NumPy, SciPy, scikit-learn, XGBoost, PyTorch, Hugging Face
  Transformers, Matplotlib.

## Academic integrity

Per-member contributions and the AI-use disclosure are in the technical report.
