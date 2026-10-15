## Passed a labels CSV as --recording
- When: `python3 run.py --recording "data/5 new users/<uuid>.features_labels.csv" ...`
- Did: used the per-user labels CSV as the recording
- Symptom: `ValueError: Cannot read a recording timestamp from <uuid>.features_labels.csv` (asqa/pipeline.py:80)
- Cause: --recording takes a raw dir (`data/5 new users/acc/<uuid>`), a single `*.m_raw_acc.dat`, or a cached user id (training users in `data/*.features_labels.csv` names)
- Rule: use one of those three forms; never the .csv
