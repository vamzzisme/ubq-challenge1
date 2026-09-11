# Ask the Sensors - Architecture

How a raw wearable recording becomes a natural-language answer that cites the
signal it came from.

The system is four layers. Each one takes the layer below as its only input and
adds exactly one kind of structure. Nothing skips a layer, and nothing invents a
fact that the layer below did not measure.

```
  raw .dat files
        │
   L1   │  preprocess.py     clock-true resampling to a 25 Hz grid
        │                    → 15 s windows of 6 channels
        ▼
   L2a  │  features.py       40 interpretable features per window
   L2b  │  context.py        + 63 temporal-context features = 103
   L2   │  recognise.py      hierarchical XGBoost → P(activity | window)
   L2c  │  decode.py         time-aware HMM → one label per window
        ▼
   L3   │  timeline.py       windows → bouts → Timeline (the evidence store)
        ▼
   L4   │  intent.py         question → Intent (operation, classes, time window)
        │  answer.py         Intent + Timeline → Answer
   L4b  │  slm.py            unfamiliar wording only: Qwen2.5-3B as a parser
        ▼
  Answer + cited intervals + sensor channels + explanation
```

The invariant that shapes everything: **the language layer selects evidence, it
never produces it.** Every number, timestamp and class name in an answer is read
out of an `Interval` object built by L3 from measured signal.

---

## 1. Entry points

### Primary - what a user runs

| Entry point | Invocation | Purpose |
|---|---|---|
| `run.py` | `python run.py --recording <path\|user-id> --question "..."` | The deliverable CLI. One structured answer per question. |
| `asqa/dashboard.py` | `python -m asqa.dashboard --port 8000` | Local web UI: pick a recording, ask, see cited evidence. |

`run.py` accepts three forms of `--recording`: a directory of raw
`<timestamp>.m_raw_acc.dat` files (the gyroscope directory is located
automatically), a single `.dat` file, or an ExtraSensory user id already in the
preprocessing cache. Flags: `--questions <file>`, `--json`, `--fold N`,
`--no-slm`, `--save-timeline`, `--quiet`.

### Pipeline stages - run once, in order, to reproduce from scratch

| Entry point | Produces |
|---|---|
| `python -m asqa.preprocess` | `outputs/cache/w15/<user>.npz` - resampled windows |
| `python -m asqa.splits` | `outputs/folds_w15.json` - subject-disjoint 5-fold split |
| `python -m asqa.recognise` | `outputs/models/recogniser_fold{0..4}.joblib` |
| `python -m asqa.decode` | `outputs/models/transitions_fold{0..4}.npy` |
| `python -m asqa.evaluate` | `outputs/evaluation/*.json` |
| `python -m asqa.benchmark` | latency, memory, model size |
| `python -m asqa.robustness` | noise and dropout degradation curves |
| `python -m asqa.figures` | `outputs/figures/*.png` |

### Subprocess - never invoked directly by a user

| Entry point | Protocol |
|---|---|
| `python -m asqa.slm_worker` | one JSON request on **stdin** → one JSON response on **stdout** |

### Tools and tests

| Entry point | Purpose |
|---|---|
| `tools/parse_strategies.py` | Reproduces the 0.5B-vs-3B parsing comparison |
| `tools/which_fold.py` | Reports which fold holds a given user out |
| `python -m tests.test_*` | 55 tests across 5 files, no pytest dependency |

---

## 2. Exit points

Everything the system emits, and where it goes.

| Exit | Format | Written by |
|---|---|---|
| stdout | The brief's 6-field answer block | `Answer.render()` in `answer.py` |
| `--output <file>` | Same, to a file | `run.py` |
| `--json` | `Answer.to_dict()` - includes raw `intervals` | `answer.py` |
| `--save-timeline` | Full `Timeline` as JSON | `timeline.py` |
| HTTP `/api/ask` | JSON answer + evidence | `dashboard.py` |
| `outputs/models/*.joblib` | Trained classifier per fold (58 MB total) | `recognise.py` |
| `outputs/models/*.npy` | HMM transition matrices | `decode.py` |
| `outputs/evaluation/*.json` | All reported metrics | `evaluate.py` |

The answer block is fixed by the brief:

```
Answer:            <the direct answer>
Activity/Event:    <the behaviour in plain words>
Evidence:
    Timestamp(s):     <seconds from recording start>
    Sensor Modality:  Accelerometer, Gyroscope
    Sensor Channel(s):<which channels carried the evidence>
Explanation:       <why the measurements support the answer>
```

---

## 3. Layer by layer

### L1 - `preprocess.py`: clock-true resampling

**In:** raw `.dat` captures. **Out:** `(N, 375, 6)` windows on an exact 25 Hz grid.

ExtraSensory's accelerometer and gyroscope are sampled on **separate, drifting
clocks**. Naively zipping them by index misaligns the two sensors by a growing
offset, which silently corrupts every cross-sensor feature.

So resampling is *clock-true*: both streams are interpolated onto one absolute
time grid at exactly `0.04 s` per sample. A window is `375` samples = **15 s**
(`WINDOW_SPAN_S = 14.96 s` between first and last sample). Any window containing
a sensor gap wider than `MAX_SAMPLE_GAP_S = 0.5 s` is **rejected**, not
interpolated across - a `WindowRejection` records why.

This is why timestamps are trustworthy enough to cite.

**Key calls:** `load_capture()` → `resample_to_grid()` → `preprocess_user()` → `save_cache()`

### L2a - `features.py`: 40 interpretable features

**In:** one `(375, 6)` window. **Out:** a 40-vector.

Features are chosen so a human can read the explanation, not for benchmark
scores. `separate_gravity()` splits acceleration into a gravity component (which
gives **device tilt**, separating lying from upright) and body acceleration
(which gives **motion energy**). `cadence_hz()` recovers step frequency  - 
1.5-2.2 Hz is walking, 2.5-3.5 Hz is running. Plus spectral entropy, band power,
and gyroscope rotation magnitude.

This is what lets an answer say *"a 2.5 Hz cadence and a gravity vector 85° from
the device axis"* instead of *"the classifier said so."*

**Key calls:** `extract()`, `extract_batch()`

### L2b - `context.py`: temporal context (+63 features)

**In:** the ordered feature matrix for a recording. **Out:** 103 features.

A single 15-second window is genuinely ambiguous - sitting still and lying still
look alike. `augment()` adds rolling statistics over neighbouring windows and
time-of-day encodings.

**This is the single largest accuracy gain in the system: 0.457 → 0.660** window
accuracy. `augment_isolated()` exists for the honest single-window case, where
no neighbours are available; it scores 0.463, and the gap between those two
numbers *is* the value of context.

### L2 - `recognise.py`: hierarchical XGBoost

**In:** 103 features. **Out:** calibrated `P(activity | window)` over 7 classes.

Not one 7-way classifier but **three**, in a hierarchy:

```
stage1   binary: static vs dynamic          (the easy, high-accuracy split)
  ├─ stage2a  lying / sitting / standing_in_place
  └─ stage2b  standing_and_moving / walking / running / bicycling
```

`predict_proba()` composes them as `P(leaf) = P(branch) × P(leaf | branch)`,
renormalises, then applies a fitted **temperature** (≈1.14) so the probabilities
are calibrated - which matters because L2c consumes them as likelihoods, not as
argmax decisions.

The static/dynamic boundary is the most reliable thing in accelerometry, so it
is decided first and never revisited. `StandingSplit` handles the
`standing_in_place` / `standing_and_moving` boundary with a fitted energy
threshold rather than asking the tree to learn it.

**Classes:** `lying_down`, `sitting`, `standing_in_place`,
`standing_and_moving`, `walking`, `running`, `bicycling`

**Key calls:** `train()` → `Recogniser.predict_proba()` → `run_fold()`

### L2c - `decode.py`: time-aware HMM

**In:** per-window probabilities. **Out:** one label per window.

Per-window predictions flicker - a single misread window inside a ten-minute
walk produces a spurious one-window "bout", which would then be cited as
evidence. The decoder imposes temporal coherence.

Transitions are **time-aware**: because windows are not perfectly contiguous
(rejected windows leave gaps), the transition matrix is exponentiated by the
actual elapsed time via a generator matrix, rather than assuming a fixed step.

`viterbi()` gives the most likely *sequence*; `forward_backward()` gives
per-window posteriors. Viterbi ships, because a coherent sequence is what L3
needs to form bouts.

**Gain: 0.660 → 0.703** accuracy.

### L3 - `timeline.py`: the evidence store

**In:** decoded labels + features. **Out:** a `Timeline`.

Three dataclasses:

- **`Window`** - one 15 s observation: `start_s`, `end_s`, `activity`, `confidence`, signal summary
- **`Interval`** - a *bout*: consecutive same-activity windows, merged across gaps of ≤180 s. Carries `duration_s`, `observed_s`, `n_windows`
- **`Timeline`** - the whole recording, with `by_activity()`, `present_activities()`, `duration_s`

The distinction between **`duration_s`** (wall-clock span of the bout) and
**`observed_s`** (seconds of sensor data actually captured within it) is
deliberate and load-bearing. ExtraSensory samples roughly 25% of wall-clock
time. Reporting a 20-minute bout while hiding that it rests on 5 minutes of
signal would be a quiet lie, so `_sampling_note()` states the ratio in the
explanation.

**This layer is the evidence boundary.** Above it, nothing touches raw signal.

**Key calls:** `build_timeline()` → `cite()`, `evidence_block()`

### L4 - `intent.py` + `answer.py`: question to answer

**`intent.py`** reduces a question to an `Intent`:

```python
Intent(operation, activities, window: TimeWindow|None, at_time_s, group, source)
```

`TimeWindow` handles `after 84h`, `before 12 hours`, `between 10 and 20 hours`,
`the first 2 hours`, `the last 3 hours`. A `from_end` window can only be
resolved once the recording length is known, so `resolve(duration_s)` is
deferred to answer time.

`clip()` applies a window to evidence by **trimming, not filtering**: a walk from
80 h to 90 h contributes only its post-84 h portion to *"how long after 84
hours"*, and `observed_s` scales by the kept fraction. Citing the whole bout
would overstate what the window contains.

**`answer.py`** routes to six handlers:

| Operation | Question form | Handler |
|---|---|---|
| identification | "what is the user doing" | `answer_identification()` |
| verification | "is / was / did the user…" | `answer_verification()` |
| duration | "how long" | `answer_duration()` |
| count | "how many times" | `answer_count()` |
| comparison | "more time X or Y" | `answer_comparison()` |
| temporal | "when did X begin" | `answer_temporal()` |

Each takes `(question, timeline, intent)` and returns an `Answer` built by
`_from()`, which attaches the citing intervals. `GROUP_PHRASES` maps behavioural
language ("resting", "tiring", "wheeled") onto class groups **in the rules**,
deliberately not in the model - see §5.

### L4b - `slm.py` + `slm_worker.py`: the language model

Reached only when the rules cannot place a question.

**Model: `Qwen2.5-3B-Instruct`** (override with `ASQA_SLM_MODEL`).

Three worker modes:

| Mode | Job | Status |
|---|---|---|
| `parse` | question → operation + activities, from a closed vocabulary | **ships** |
| `choose` | score lettered options; cannot emit an invalid word | measured, inferior |
| `answer` | answer directly from a menu of real intervals | last resort |

`parse_intent()` keeps the **rules'** operation and time window and uses only
the model's activity slot - because the rules read question *form* reliably
("did…" is verification) and lack only *vocabulary*. The model's operation is
used solely when a typo defeats the keyword outright (`"how mcuh time…"`).

---

## 4. Call graph

**Asking a question:**

```
run.py:main()
 └─ Pipeline.run(recording)                          [pipeline.py]
     ├─ load_recording()                             → LoadedRecording
     ├─ preprocess.resample_to_grid()                → (N, 375, 6)
     ├─ features.extract_batch()                     → (N, 40)
     ├─ context.augment()                            → (N, 103)
     ├─ Recogniser.predict_proba()                   → (N, 7)
     ├─ decode.viterbi()                             → labels
     └─ timeline.build_timeline()                    → Timeline
 └─ answer_question(question, timeline, use_slm)     [answer.py]
     ├─ parse_question()                             → Intent      (~0.2 ms)
     ├─ HANDLERS[intent.operation](...)              → Answer
     │   └─ _select() → intent.clip() → _from() → evidence_block()
     └─ if unplaceable and use_slm:
         ├─ slm.parse_intent() ──subprocess──▶ slm_worker.parse()
         │   └─ resolve_activity_word() / resolve_group_word()
         │   └─ HANDLERS[...] again, with the model's activities
         └─ else slm.answer_open_world()
             └─ retrieve() → render_menu() → generate() → resolve_citations()
```

**Dashboard:** `Handler` → `Backend.ask()` → the same `answer_question()`. The
UI has no answering logic of its own, so CLI and web cannot diverge.

---

## 5. Design decisions

### Evidence flows up; language only selects

Every field of every answer is read from an `Interval`. The language model
receives a *menu of real intervals* and may only cite by number
(`resolve_citations()` discards any citation not on the menu). This is what
makes the system auditable: an answer can be wrong, but it cannot be
*unfounded* - you can always check the interval it names.

### Three stages, cheapest first

Rules (~0.2 ms) handle the great majority. The model is consulted only for
wording the rules cannot place, and answers directly only when parsing also
fails. Ordinary questions never load a 10 GB model.

### The model parses; it does not judge

A 0.5B model asked to judge *"wheeled movement"* was measured answering
*"Pedal-based mode"* while citing a **sitting** interval. Behavioural synonyms
therefore resolve in `GROUP_PHRASES` (rules), and the model's job is narrowed to
mapping wording onto a closed vocabulary. A misparse yields the wrong
*operation* - visible and checkable - rather than a fabricated interval.

### Model size was measured, not assumed

Ten deliberately unusual phrasings (`tools/parse_strategies.py`):

| Strategy | 0.5B | 3B |
|---|---|---|
| generate | 1/10 | **9/10** |
| stayed in vocabulary | 3/10 | **10/10** |
| choose (letter scoring) | 1/10 | 8/10 |
| generate + resolve *(ships)* | 5/10 | **9/10** |

The 0.5B model did not follow the closed-vocabulary instruction - it echoed the
questioner's word in the vocabulary's *shape* (`wandering`, `kipping`,
`legging_it`), having learnt the format but not the membership rule.

Constraining the decoding so an invalid word is impossible **guaranteed validity
and bought no accuracy**: 1/10, picking one option five times, including 0.912
confidence that *"pedalling"* meant `standing_in_place`. It was choosing a
letter, not reading the question.

The 3B model simply obeys. The lesson worth carrying: **validity and correctness
are separate problems**, and below some capability threshold no prompt or
decoding engineering substitutes for parameters. Cost: 22.6 s and 10.0 GB peak
RSS versus 11.0 s and 2.6 GB - paid only when the rules fail.

### The language model runs in its own process

Loading PyTorch into an interpreter that has already run XGBoost **segfaults**
on this platform (exit 139), on both MPS and CPU; `KMP_DUPLICATE_LIB_OK` does
not help. The two runtimes bring incompatible native threading libraries into
one address space.

The isolation pays for itself twice: the brief asks for per-component resource
cost, and a separate process makes the model's memory and latency **directly
measurable** instead of tangled with the classifier's.

### Subject-disjoint folds

`splits.py` partitions by **user**, never by window. Windows from one person are
highly autocorrelated; a random split would leak a subject across train and test
and inflate accuracy substantially. Every reported number is
leave-users-out.

### Rejection over interpolation

A window spanning a sensor gap > 0.5 s is dropped. Interpolating across it would
manufacture signal, and that signal would then be cited as evidence.

---

## 6. Measured results

**Recognition** (5-fold subject-disjoint CV, mean):

| Configuration | Window acc | After HMM | Macro F1 |
|---|---|---|---|
| context-free | 0.457 | 0.576 | 0.475 |
| **context-aware** | **0.660** | **0.703** | **0.525** |
| single isolated window | 0.463 | - | - |

Context is worth **+0.20**; HMM decoding a further **+0.04**.

**Cost** (fold 3, 36,087 test windows, laptop CPU, no GPU):

| Metric | Value |
|---|---|
| Feature extraction | 0.333 ms/window |
| Inference | 0.387 ms mean, 0.478 ms p95 |
| Model size | 9.10 MB (214,421 tree nodes) |
| Peak RSS | 855 MB |
| Answer routing (rules) | ~0.16 ms |

**Robustness** (fold 3): accuracy 0.778 clean → 0.625 at 1% additive noise →
0.520 at 20%. Largely **insensitive to sample dropout** (0.775 at 10%), which
follows from clock-true resampling - dropout is exactly what it was built to
absorb.

**Question answering** - ⚠️ **these numbers are stale.** They were generated at
01:02 today, *before* the time-window fix, the SLM parser, and the 3B upgrade.
`duration` and `count` in particular were measured while range qualifiers were
still being silently dropped, so they understate current behaviour. **Re-run
`python -m asqa.evaluate` before quoting these.**

| Type | n | Answer acc | Grounded acc |
|---|---|---|---|
| verification | 245 | 0.90 | 0.44 |
| comparison | 34 | 0.88 | 0.50 |
| identification | 228 | 0.47 | 0.36 |
| grounding | 194 | 0.33 | 0.23 |
| duration | 194 | 0.25 † | 0.14 |
| count | 194 | 0.14 † | 0.07 |
| open_world | 70 | 0.99 | 0.69 |

† measured before time windows were applied.

*Grounded accuracy* requires the answer to be right **and** to cite the correct
intervals - always lower than answer accuracy, and the stricter number to trust.

---

## 7. Known limitations

- **QA metrics predate today's fixes.** Re-run `asqa.evaluate` before reporting.
- **`count` and `duration` are the weakest operations.** Bout segmentation is
  sensitive to the 180 s merge gap; the reported count depends on that constant.
- **The model stage is best-effort, not reliable.** 9/10 on obscure wording, and
  its failures are visible in the explanation rather than silent.
- **`running` is rarely detected** in some recordings - the confusion matrix
  shows near-zero recall for it on fold 0, reflecting genuine class scarcity.
- **10 GB peak RSS** when the 3B model loads. Set
  `ASQA_SLM_MODEL=Qwen/Qwen2.5-0.5B-Instruct` where memory is binding, accepting
  the accuracy drop documented above.
