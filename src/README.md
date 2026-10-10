# Source code

- `extract_features.py`: extracts per-region max-minus-min range features from time-course `.txt` files and joins labels from a demographic CSV.
- `train_baseline.py`: evaluates Logistic Regression and RBF-SVM using stratified cross-validation, then saves metrics and fitted models.
- `original_reference/`: unmodified legacy research scripts; reference only, not expected to run in the current environment.

Confirm all input column names, participant-ID conventions, eligibility criteria, and feature definitions against the authorized dataset documentation before running.
