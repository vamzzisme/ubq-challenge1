# Ask the Sensors

Grounded, explainable activity question answering from wearable accelerometer and
gyroscope signals. CS60055 Ubiquitous Computing, Hackathon Challenge 1.

Given a sensor recording and a natural-language question, the system returns a
structured answer that cites the specific stretch of signal supporting it.

```
Query: "How long was the user walking?"
Answer: 15039 seconds
Activity/Event: walking
Evidence:
    Timestamp(s): 25125 to 26760, 27645 to 30360, 163623 to 165498, 196503 to 197718,
                  201803 to 202958, 254963 to 256367 (+12 shorter intervals totalling 5040 s)
                  (seconds from start)
    Sensor Modality: Accelerometer, Gyroscope
    Sensor Channel(s): All
Explanation: Walking was detected in 18 intervals spanning 15039 s in total (251 min).
             The recording samples about 23% of wall-clock time, so this span of 15039 s
             rests on 3960 s of directly observed sensor data.
```

**All timestamps are seconds from the start of the recording.**

## Run it

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Nothing needs downloading. 35 preprocessed recordings, the trained models and
the HMM transition matrices are all committed, so a question runs straight away:

```bash
python run.py --recording 098A72A5-E3E5-4F54-A152-BBDA0DF7B694 \
              --question "How long was the user walking?"
```

`ls outputs/cache/w15/` lists every id that works. A file of questions runs in
one go:

```bash
python run.py --recording 098A72A5-E3E5-4F54-A152-BBDA0DF7B694 \
              --questions tests/questions_all_tiers.txt
```

Useful flags: `--json` for machine-readable output, `--output FILE` to write to
disk, `--no-slm` to skip the language model, `--save-timeline FILE` to keep the
intermediate activity timeline.

### Or use the dashboard

```bash
python -m asqa.dashboard --open        # http://127.0.0.1:8000
```

Pick a recording, ask in plain language, and click any answer to highlight the
exact intervals it cited on a timeline of the whole recording. Hovering a
segment shows its measured energy, cadence, tilt and rotation.

The picker offers five **held-out** recordings that appear in no model's
training set, so what you see is behaviour on a genuinely new person. The
dashboard contains no answering logic of its own - it calls the same
`answer_question()` the CLI does, and `tests/test_dashboard.py` asserts the two
agree exactly.

### Questions it can answer

```
How long was the user walking?              duration
Did the user cycle at all?                  verification
How many times did the user walk?           count
What was the user doing at 3000 seconds?    identification
When did the user begin lying down?         temporal
Was there more sitting or walking?          comparison
How long was the user walking after 8 hours?          time-windowed
Was the user doing anything tiring?                   behaviour phrasing
```

## Before you test: things worth knowing

- **Dashboard ids and CLI ids are different sets.** The dashboard serves five
  held-out recordings; `run.py` reads the 35 cached training users. An id copied
  from the dashboard will fail with `No recording found`. Use
  `ls outputs/cache/w15/`.
- **First analysis of a dashboard recording takes 20-45 s** (a week of raw
  signal at 25 Hz), then it is cached.
- **The language model is optional and slow to start.** The first question that
  needs it downloads about 6 GB (`Qwen/Qwen2.5-3B-Instruct`) and it wants
  roughly 10 GB of RAM. It looks like a hang; it is a download. Ordinary
  questions never touch it - the rule-based path answers in about 0.16 ms.
  `--no-slm`, or unticking the dashboard box, disables it. If it cannot load,
  the answer says so and still cites a real interval rather than crashing.
- **Raw data is not in this checkout**, so `python -m asqa.preprocess` cannot
  run. Everything else reads the committed caches and is unaffected. See
  [Rebuilding from raw data](#rebuilding-from-raw-data).
- **If you point at a raw accelerometer directory with no gyroscope alongside,
  it is accepted silently** and accelerometer data fills the gyroscope channels.
  No error is raised, and answers then cite a sensor that was never read. A
  single `.dat` file fails cleanly in the same situation; only the directory
  form has this fallback.
- **Cloning is about 7 GB** of git history, for 1.5 GB of tracked outputs.

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

1,299 generated cases, scored under the rule each answer type deserves, with the
language model enabled so the open-world questions are actually reasoned about.

| question type | n | answer accuracy | grounded accuracy |
|---|---|---|---|
| verification | 245 | 0.902 | 0.445 |
| comparison | 34 | 0.882 | 0.500 |
| open-world | 210 | 0.590 | 0.252 |
| identification | 228 | 0.474 | 0.360 |
| count | 194 | 0.418 | 0.211 |
| duration | 194 | 0.376 | 0.227 |
| grounding | 194 | 0.268 | 0.201 |
| **overall (macro across types)** | **1299** | **0.559** | **0.314** |

Overall micro accuracy 0.530; macro-F1 over the four categorical types 0.636;
grounding precision 0.914. Verification, positive class *Yes* (n+=194, n-=51):
precision 0.962, recall 0.912, F1 0.937, specificity 0.863.

Per-fold macro answer accuracy: 0.502, 0.578, 0.538, 0.603, 0.560.

Correctness rules: categorical answers by exact match; durations within
`max(30 s, 20%)`; onsets within ±300 s; counts within `max(±5, 20%)`;
*grounded* requires the answer to be correct **and** the cited interval to reach
IoU ≥ 0.5 **and** the modality and channels to match.

The numeric bands are looser than the brief's illustrative ones, and the
difference is material, so both are reported: at the brief's suggested
tolerances duration scores **0.253** (within 10%) and count **0.144** (within
±1), against 0.376 and 0.418 at the bands above. Figure 3 traces the whole
curve for each rather than resting on one threshold.

### Cost

Measured on an arm64 laptop CPU, single process.

Four operating points on fold 3, each carried all the way through decoding and
aggregation so the accuracy column is **end-to-end QA accuracy**, not a
per-window proxy. 312 questions per point.

| configuration | QA accuracy | grounded | per-window acc | size on disk | median latency | tree nodes |
|---|---|---|---|---|---|---|
| full (deployed) | **0.568** | 0.362 | 0.690 | 9.10 MB | 0.48 ms | 214,421 |
| pruned | 0.536 | 0.360 | 0.689 | 8.05 MB | 0.38 ms | 191,702 |
| context-free | 0.472 | 0.260 | 0.442 | 4.69 MB | 0.42 ms | 110,998 |
| **shallow (edge)** | 0.453 | 0.264 | 0.547 | **0.95 MB** | 0.48 ms | 11,322 |

Feature extraction adds 0.35 ms per window. `shallow` is the edge operating
point: a **9.6× size reduction for 11.5 QA points**.

The two accuracy columns disagree on the ranking, which is why the brief asks
for the question-answering one. `shallow` reads individual windows better than
`context-free` (0.547 against 0.442) but answers questions worse (0.453 against
0.472): its errors cluster into whole bouts, which moves durations and counts,
while the context-free model scatters its errors inside bouts that survive
merging.

The open-world language model is reported separately because it dominates:
**Qwen2.5-3B-Instruct, 3.086B parameters, 10,002 MB peak RSS, 28.1 s per query**
including weight loading - roughly 59,000× the cost of classifying a window.
Only Task 4 questions that do not map onto the seven classes invoke it.
`asqa.benchmark` asks the worker which model it loaded rather than naming one,
so this figure always describes the model that actually ran.

### Explanation quality

The brief asks for the open-world explanations to be judged on a fixed 1-5
rubric rather than only for correctness. All 210 open-world explanations were
graded on three criteria by the same model, twice, at different temperatures.

| criterion | mean (1-5) | Krippendorff α | quadratic-weighted κ |
|---|---|---|---|
| cites real signal features | 4.25 | 0.983 | 0.983 |
| those features support the conclusion | 3.17 | 0.873 | 0.873 |
| conclusion is plausible | 4.38 | 0.946 | 0.945 |
| **mean rubric score** | **3.93** | | |

The shape is the useful part: explanations reliably cite real measured features
(4.25) and reach plausible conclusions (4.38), but the link between the two is
the weak point (3.17). That is the honest description of a system that grounds
its prose in real intervals without always reasoning correctly from them.
Embedding similarity to a reference explanation is 0.957, reported as a
secondary proxy only - it rewards surface overlap with the template wording.

### Robustness

QA accuracy on a fixed question set as the input is deliberately degraded, on
4,977 windows from two held-out users (fold 3). The question set is fixed
because the reference answers come from the labels, which the degradation does
not touch - only the signal the system reads changes.

| degradation | undegraded | mild | severe |
|---|---|---|---|
| accelerometer noise | 0.568 | 0.448 (σ=0.01 g) | 0.407 (σ=0.2 g) |
| dropped samples | 0.568 | 0.568 (20% lost) | 0.530 (50% lost) |
| sampling rate | 0.568 | 0.509 (15 Hz) | 0.428 (10 Hz) |

Dropout is the degradation the design absorbs: accuracy is **unchanged up to 20%
of samples lost**, which follows from resampling by sensor timestamp rather than
by sample index. Noise costs the most in its first step and then flattens, so
the misses are systematic rather than progressive.

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

**L1 - [`asqa/preprocess.py`](asqa/preprocess.py).** Resamples by *sensor
timestamp*, not sample index. The accelerometer clock is genuinely uneven
(measured dt 0.024-0.077 s) and 800 samples spans anywhere from ~9 s to ~25 s
depending on the device, so index-based resampling compresses the window and
misaligns the two modalities. Acc and gyro are interpolated onto one shared grid
covering only the interval both actually observed.

**L2 - [`asqa/features.py`](asqa/features.py), [`asqa/recognise.py`](asqa/recognise.py).**
Physics-based features (gravity separation, tilt, autocorrelation cadence, jerk,
spectral band powers) feeding a three-node hierarchy: static vs dynamic, then
posture, then gait. Each node's decision variable is a nameable physical
quantity, which is what lets the interface layer explain an answer rather than
assert it.

**Temporal context - [`asqa/context.py`](asqa/context.py).** Time-bounded rolling
statistics over neighbouring windows. This is the single largest accuracy lever
(+9 points): a 15-second window of stillness cannot distinguish lying from
sitting, but a stretch of them can.

**Decoder - [`asqa/decode.py`](asqa/decode.py).** A time-aware HMM. The
per-minute transition matrix is raised to the power `Δt/60`, so a multi-hour
dropout relaxes toward the stationary distribution instead of asserting that the
user kept doing the same thing.

**L3 - [`asqa/timeline.py`](asqa/timeline.py).** Merges windows into intervals
and separates *observed* time from *spanned* time. ExtraSensory observes ~15 s in
every 60, so a bout spanning 300 s rests on ~75 s of signal.

**L4 - [`asqa/answer.py`](asqa/answer.py), [`asqa/intent.py`](asqa/intent.py),
[`asqa/slm.py`](asqa/slm.py).** Reduces a question to an `Intent` - an
operation, the classes it names, and a time window (`after 84 hours`,
`in the first 2 hours`, `between 10 and 20 hours`) - then routes it to one of
six operations. Three stages, cheapest first: keyword rules (~0.2 ms, the great
majority of questions); if those cannot place the wording, the language model
is asked *what the question is asking for*, never what the answer is, and its
reply is validated against a closed vocabulary before the same handler computes
the answer from the timeline; only if that also fails does the model answer
directly from a menu of real intervals.

The model is **Qwen2.5-3B-Instruct**, chosen by measurement rather than default.
On ten deliberately unusual phrasings (`tools/parse_strategies.py`) a 0.5B model
answered 1/10 and stayed inside the closed vocabulary only 3/10 - it echoed the
questioner's word in the vocabulary's shape (`wandering`, `kipping`,
`legging_it`), having learnt the format but not the membership rule. Forcing
valid output by scoring lettered options guaranteed membership and still scored
1/10, picking one option five times. The 3B model answers 9/10 and stays in
vocabulary 10/10 without coaxing. It costs 22.6 s and 10.0 GB against 11.0 s and
2.6 GB, paid only when the rules cannot place a question. Override with
`ASQA_SLM_MODEL`. Every timestamp, modality and channel it
prints is copied from an interval the earlier layers produced. For open-world
questions the language model picks from a numbered menu of real intervals; an id
outside that menu is discarded rather than rendered.

## Tests

```bash
python -m tests.test_preprocess   # clock-true resampling; catches the 2.000 -> 2.300 Hz error
python -m tests.test_context      # no cross-recording bleed, no context across dropouts
python -m tests.test_answer       # question parsing and the required output contract
python -m tests.test_intent       # time windows, and validating what the model proposes
python -m tests.test_dashboard    # dashboard API contract; asserts it matches the CLI
```

All 55 pass with no data beyond what is committed, and need no language model.

## Repository layout

```
asqa/          the system, one module per layer
run.py         the runnable entry point
tools/         which_fold.py - which model is safe for a given user
tests/         regression tests and an all-tier question set
outputs/       models, timelines, evaluation JSON, figures (regenerable)
data/          local data only; never committed
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
  windows. Worse, some running-labelled windows contain no motion at all - one
  user's 68 running windows have the same body-acceleration energy as lying down
  (0.003 g) while their walking reads 0.333 g, i.e. the phone was not on them.
- **Lying vs sitting is the dominant residual confusion.** When the phone is
  off-body and still, the two are not separable from accelerometer geometry
  alone. Temporal context substantially mitigates this but does not remove it.
- **A 0.5B language model is too weak to reason here.** Asked to judge "a wheeled or
  pedal-based mode of movement" it answered "Pedal-based mode" while citing a
  *sitting* interval. Behaviour phrases that map onto the seven classes are
  therefore resolved deterministically; the model is used for prose, not verdicts.
  On ten unusual phrasings it also stayed inside the closed class vocabulary only
  3 times in 10, against 10 in 10 for the 3B model that is now the default
  (`tools/parse_strategies.py` reproduces this). The smaller model remains
  selectable through `ASQA_SLM_MODEL` where memory is the binding constraint.
- **A directory of accelerometer files with no gyroscope alongside is accepted
  silently**, with accelerometer data standing in for the missing gyroscope
  channels. See [If you are testing this](#if-you-are-testing-this).
- **PyTorch and XGBoost cannot share an interpreter here** - importing torch into
  a process that has run XGBoost segfaults on this platform, on both MPS and CPU.
  The language model runs in an isolated worker process
  ([`asqa/slm_worker.py`](asqa/slm_worker.py)).

## Rebuilding from raw data

Only needed to retrain or regenerate figures; testing the system does not require any of this.

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

```bash
python -m asqa.preprocess              # raw .dat -> 25 Hz windows, cached per user
python -m asqa.splits                  # rare-class-aware, user-disjoint folds
python -m asqa.recognise --build-features
python -m asqa.recognise --cv          # trains context-free + context-aware per fold
python -m asqa.decode --cv             # HMM decoding, both variants, both methods
python -m asqa.evaluate --cv --use-slm # QA scoring by question type  [~2 h]
python -m asqa.rubric --question-type open_world   # 1-5 explanation rubric + grader agreement
python -m asqa.benchmark --fold 3      # QA accuracy, size, latency, memory per operating point
python -m asqa.robustness --fold 3     # noise / dropout / sampling-rate sweeps
python -m asqa.figures                 # the five required figures, plus one supporting
```

`--use-slm` is what produces the reported numbers: without it the Task 4
questions that do not map onto the seven classes are left unanswered rather than
reasoned about, and open-world accuracy collapses. Both forms write
`outputs/evaluation/qa_cv.json`, so the cheap `python -m asqa.evaluate --cv` is
for a quick check only - it will overwrite the headline file. Pass `--output` to
keep them apart.

Preprocessing takes roughly 15 minutes for 35 users and produces ~1.2 GB of
cache. Everything downstream reads that cache. The evaluation is the long pole:
the language model answers one question in about 25 s on this hardware.

Fold assignment is committed (`outputs/folds_w15.json`) so the split behind the
reported numbers is exactly reproducible.

## Citations

- **Dataset.** Y. Vaizman, K. Ellis, G. Lanckriet. "Recognizing Detailed Human
  Context In-the-Wild from Smartphones and Smartwatches." *IEEE Pervasive
  Computing*, 2017. <http://extrasensory.ucsd.edu/>
- **Language model.** Qwen2.5-3B-Instruct, Alibaba Cloud (Apache 2.0), used for
  Task 4 open-world reasoning and for grading explanations against the rubric.
  `ASQA_SLM_MODEL` selects a different one; Qwen2.5-0.5B-Instruct is the
  low-memory fallback.
- **Libraries.** NumPy, SciPy, scikit-learn, XGBoost, PyTorch, Hugging Face
  Transformers, Matplotlib.

## Academic integrity

Per-member contributions and the AI-use disclosure are in the technical report.
