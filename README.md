# Classifying Adolescent Excessive Alcohol Drinkers

## Project overview

This machine-learning mini-project is inspired by the original research
topic, "Classifying Adolescent Excessive Alcohol Drinkers from fMRI Data."

Because the original preprocessed fMRI dataset was not available for this
implementation, an alternative dataset was used with faculty approval:
the UCI Student Performance dataset.

**Important:** This implementation uses student survey data, not fMRI
measurements. It predicts a project-defined alcohol-consumption category
and does not diagnose alcohol use disorder or reproduce the original fMRI
experiment.

## Dataset

Source: UCI Machine Learning Repository — Student Performance
https://archive.ics.uci.edu/dataset/320/student%2Bperformance

Main file: `data/student-por.csv`

- 649 student records
- 33 original columns
- Includes demographic, social, school, and alcohol-consumption variables
- `Dalc`: reported workday alcohol consumption, scale 1–5
- `Walc`: reported weekend alcohol consumption, scale 1–5

The dataset is subject to the UCI dataset's terms and citation requirements.

## Target definition

The preparation script creates `high_consumption`:

- `1`: Dalc >= 4 OR Walc >= 4
- `0`: both Dalc and Walc are below 4

This is a threshold defined for this project, not a clinical definition.

The resulting dataset contains 509 records labelled 0 and 140 labelled 1.

`Dalc` and `Walc` are excluded from model inputs because they are used
to define the target. The dataset is imbalanced, so accuracy alone is
not sufficient to evaluate model performance.

## Models and evaluation

Models:
- Logistic Regression
- Support Vector Machine with an RBF kernel

The pipeline uses:
- Stratified 80/20 train/test split with random state 42
- Five-fold stratified cross-validation on training data
- Median imputation and standardization for numerical features
- Most-frequent imputation and one-hot encoding for categorical features
- Class weighting to account for class imbalance

Preprocessing is fitted within each training fold through the scikit-learn
pipeline to reduce data leakage.

Metrics include accuracy, balanced accuracy, precision, recall, F1-score,
ROC-AUC, cross-validation F1, and confusion matrices.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Place `student-por.csv` in `data/`, then run:

```bash
python src/prepare_dataset.py
python src/train_baseline.py
python src/visualize_results.py
```

Generated outputs are saved under `results/`.

## Current test-set results

Results from the current stratified test split:

| Metric | Logistic Regression | RBF SVM |
|---|---:|---:|
| Accuracy | 0.746 | 0.792 |
| Balanced accuracy | 0.735 | 0.738 |
| Precision | 0.444 | 0.514 |
| Recall | 0.714 | 0.643 |
| F1-score | 0.548 | 0.571 |
| ROC-AUC | 0.826 | 0.854 |
| Five-fold CV F1 mean | 0.550 | 0.527 |
| Five-fold CV F1 standard deviation | 0.030 | 0.041 |

These are results from this implementation and dataset, not results copied
from the original research paper. They are based on one train/test split;
further validation is needed before making general claims.

## Repository layout

- `data/`: dataset and data instructions
- `src/prepare_dataset.py`: creates the binary target
- `src/train_baseline.py`: preprocessing, training, and evaluation
- `src/visualize_results.py`: generates result plots
- `src/original_reference/`: original reference scripts
- `tests/`: software tests
- `results/`: metrics and generated figures
- `report/`: report materials
- `presentation/`: presentation outline

## Limitations and responsible interpretation

The dataset is observational and survey-based. Its consumption ratings
are self-reported. The chosen threshold does not establish a clinical
diagnosis. The sample is limited to the supplied dataset and may not
generalize to other populations. Class imbalance affects precision and
recall. Predictive associations do not establish causation.

This alternative-data experiment does not use brain imaging and must not
be described as an fMRI classifier.

## References

1. UCI Machine Learning Repository. Student Performance.
   https://archive.ics.uci.edu/dataset/320/student%2Bperformance
2. Original project reference: Noh et al., "Classifying Adolescent
   Excessive Alcohol Drinkers from fMRI Data," CS229 project materials.
   https://github.com/jnoh4/CS229ToShare

The original scripts are retained for reference; the reported results in
this repository come from the alternative UCI survey dataset.
