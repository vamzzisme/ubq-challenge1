# Project Completion TODO List

Here is the final roadmap for the remaining deliverables in the project:

## 1. Random Forest Gyroscope Fusion (Done)
- [x] Update `src/signal_features.py` to extract 86 features across all 6 axes (acc + gyro).
- [x] Update `src/train_baseline.py` to load both modalities and train the updated RF.
- [x] Update `src/inference.py` to zero-pad missing gyro data and maintain pipeline stability.

## 2. SLM Query Engine & Open-World Reasoning (Done)
- [x] Integrate Qwen2.5-0.5B-Instruct via HuggingFace `transformers`.
- [x] Convert timeline segments into a compact prompt.
- [x] Update `src/qa.py` to seamlessly route untrained/open-world queries to the SLM.

## 3. Robustness Evaluation
- [ ] Build a script to test the model against degraded inputs (dropped samples, added noise).
- [ ] Generate the required Robustness Curve (Figure 5).

## 4. Final Figures & Deliverables
- [ ] Re-generate the final versions of Figures 1-4 with the new 6-axis RF and SLM query engine.
- [ ] Draft the 10-12 page Technical Report outlining our design and accuracy tradeoffs.

## 5. Interactive QA Dashboard (UX Enhancement)
- [ ] Build a modern, interactive web dashboard where a user can upload a recording/timeline and ask the SLM natural language questions.
- [ ] Incorporate rich design elements to provide a premium user experience.
