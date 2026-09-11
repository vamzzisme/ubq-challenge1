# Ask the Sensors - status

Layered rebuild in `asqa/`. Legacy `src/` retained as the recorded "before"
baseline for the report's comparison.

## Done

- [x] **L1** clock-true 25 Hz resampling, regression-tested
- [x] **Splits** rare-class-aware, user-disjoint 5-fold (`outputs/folds_w15.json`)
- [x] **L2** hierarchical XGBoost + physics features
- [x] **Context** time-bounded rolling features (+9 accuracy points)
- [x] **Decoder** time-aware HMM, Viterbi and forward-backward
- [x] **L3** interval aggregation, observed vs spanned time, bounded citation
- [x] **L4** question router, constrained SLM, `run.py`
- [x] **Evaluation** per-type QA scoring, benchmark, robustness, five figures
- [x] **README** setup, reproduction, limitations, citations

## Remaining

- [ ] **Technical report** (10-12 pages) - draft
- [ ] **Per-member contribution statement** - yours to write
- [ ] **AI-use disclosure** - yours to confirm
- [ ] Optional: motion-primitive codebook for richer explanations
- [ ] Optional: retire `src/` once the report's before/after is written

## Untested accuracy levers (deliberately stopped here)

Hyperparameter search, context breadth beyond ±30 windows, class-weight scheme
comparison, an explicit "insufficient evidence" state, per-session orientation
reference. Each is expected to be worth a small amount; the decision was to stop
and build the system instead.
