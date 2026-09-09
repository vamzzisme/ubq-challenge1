# Ask the Sensors — build status

Layered rebuild in `asqa/`. Legacy `src/` is kept read-only as a parts bin and
as the recorded "before" baseline; it is retired in Phase 9.

## Done

- [x] **Phase 0** Scaffold (`asqa/`, `tests/`, `outputs/`), 5.9 GB reclaimed.
- [x] **Phase 1** L1 clock-true resampling to a true 25 Hz. Regression-tested.
- [x] **Phase 2** Rare-class-aware, user-disjoint 5-fold split (`outputs/folds_w15.json`).
- [x] **Phase 3** L2 hierarchical XGBoost recogniser + physics features.
- [x] **Phase 4** L2c time-aware HMM/Viterbi decoder.

## Next

- [ ] **Phase 5** L3 aggregation: decoded path -> intervals, capped evidence.
- [ ] **Phase 6** L4 interface: `answer.py`, constrained SLM, `run.py`.
- [ ] **Phase 7** Evaluation harness + the five required figures.
- [ ] **Phase 8** Motion-primitive codebook (explanation quality).
- [ ] **Phase 9** Retire `src/`, write the report.

## Measured so far (5-fold, user-disjoint, 7 classes)

| stage | accuracy | macro-F1 |
|---|---|---|
| per-window classifier | 0.444 +/- 0.020 | 0.327 +/- 0.056 |
| **after HMM decoding** | **0.578** | **0.374** |

Legacy `src/` for comparison: 0.532 accuracy / 0.257 macro-F1, but on a single
held-out user and only 6 classes.

## Data findings that shaped the design

1. **Mixed accelerometer units.** 5 of 15 users report m/s^2 (|gravity| ~ 9.7),
   10 report g (~1.0) — a 9.8x cross-user scale difference. Normalising by
   measured gravity magnitude lifted macro-F1 from 0.267 to 0.396 on fold 0.
2. **Captures are not 20 s.** 800 samples means ~9 s to ~25 s depending on the
   device; acc/gyro overlap supports 20 s for only 33.6% of windows but 15 s for
   91.3%. Hence a 15 s analysis window.
3. **Off-body windows.** 39.2% of windows show no motion at all. For lying down
   this is correct; for `running` it is not — user 9DC38D04's 68 running windows
   have the same body-acceleration energy as lying down (0.003 g), i.e. the phone
   was not on them. That is 56% of all running in the corpus.
4. **Per-class isotonic calibration is harmful here** — it drove bicycling F1
   from 0.66 to 0.00. Replaced with single-parameter temperature scaling.
