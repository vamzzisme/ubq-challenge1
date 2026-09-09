# Project Completion TODO List

Here is the final roadmap for the remaining deliverables in the project:

## 1. Random Forest Gyroscope Fusion (Done)
- [x] Update `src/signal_features.py` to extract 86 features across all 6 axes (acc + gyro).
- [x] Update `src/train_baseline.py` to load both modalities and train the updated RF.
- [x] Update `src/inference.py` to zero-pad missing gyro data and maintain pipeline stability.

## 2. SLM Query Engine & Open-World Reasoning
- [ ] Replace the deterministic regex logic in `src/qa.py` with an SLM (Small Language Model).
- [ ] Ensure the SLM strictly grounds its answers using the timeline and doesn't invent timestamps.
- [ ] Implement Task 4 (Open-World Activity Reasoning) by prompting the SLM to explain untrained behaviors (e.g., "strenuous activity", "wheeled movement") based on signal characteristics.

## 3. Robustness Evaluation
- [ ] Build a script to test the model against degraded inputs (dropped samples, added noise).
- [ ] Generate the required Robustness Curve (Figure 5).

## 4. Final Figures & Deliverables
- [ ] Re-generate the final versions of Figures 1-4 with the new 6-axis RF and SLM query engine.
- [ ] Draft the 10-12 page Technical Report outlining our design and accuracy tradeoffs.

## 5. Interactive QA Dashboard (UX Enhancement)
- [ ] Build a modern, interactive web dashboard where a user can upload a recording/timeline and ask the SLM natural language questions.
- [ ] Incorporate rich design elements to provide a premium user experience.
