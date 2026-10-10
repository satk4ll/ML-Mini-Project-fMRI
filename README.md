# Classifying Adolescent Excessive Alcohol Drinkers from fMRI Data

Machine-learning mini-project informed by the original research scripts. This repository contains a modern baseline pipeline and preserves the original scripts for reference.

## Current status
- Environment and repository setup completed.
- Feature-extraction and baseline training code prepared.
- **Authorized real dataset is not yet available in this repository.**
- Therefore, no scientific performance results are claimed.

## Workflow
1. Obtain approved access to the fMRI time-course data and demographic labels.
2. Verify the dataset schema, participant identifiers, label definitions, eligibility criteria, and any repeated visits.
3. Extract per-region max-minus-min features.
4. Compare Logistic Regression and RBF-SVM using cross-validation.
5. Review participant-level independence and possible leakage; report metrics only from approved real data.

## Layout
- `src/extract_features.py` — fMRI time-course feature extraction.
- `src/train_baseline.py` — Logistic Regression/SVM evaluation and model saving.
- `src/original_reference/` — six original legacy scripts, preserved unchanged.
- `tests/` — unit tests using artificial arrays only.
- `report/report_template.md` — report template.
- `presentation/slide_outline.md` — presentation plan.
- `data/README.md` — data access and expected schema.

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Software tests
```bash
python -m unittest discover -s tests -v
```
Tests use tiny artificial arrays only to test software behavior; they are not training data and do not produce scientific results.

## Extract features
After receiving authorized data, and after mapping the source CSV to the expected columns:
```bash
python src/extract_features.py \
  --time-course-dir /path/to/ICA_time_course \
  --demographics /path/to/demographic.csv \
  --output data/features.csv
```
By default, only brain-region range features are predictors. Add `--include-demographics` only if including age/sex/scanner is justified and clearly documented.

## Train and evaluate
```bash
python src/train_baseline.py --data data/features.csv --label label
```
The input must contain one row per independent participant, a binary `label` column encoded 0/1, numeric feature columns, and optionally a unique `participant_id` column. If there are repeated visits, this script's ordinary stratified CV is not suitable until the evaluation is changed to group by participant.

## Data privacy
Do not commit restricted participant-level data, demographic files, time courses, model artifacts, or other sensitive files to GitHub. Follow the dataset's approval and data-use conditions.
