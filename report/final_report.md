
# Classifying Adolescent Excessive Alcohol Drinkers
## A Survey-Data Classification Adaptation

**Course:** UE24CS352A — Machine Learning  
**Institution:** PES University  
**Dataset:** UCI Student Performance — Portuguese-language records

## Abstract

This project investigates binary classification of student-reported alcohol-consumption levels using supervised machine learning. The original project topic concerns excessive alcohol consumption classification from functional magnetic resonance imaging (fMRI). Because the original preprocessed fMRI dataset was unavailable for this implementation, an alternative dataset was used with faculty approval. The UCI Student Performance dataset contains 649 Portuguese-language student records. A project-defined binary target was created from reported workday and weekend alcohol-consumption ratings. Logistic Regression and an RBF-kernel Support Vector Machine (SVM) were evaluated using a stratified train/test split and five-fold cross-validation. On the held-out test set, SVM achieved 79.2% accuracy and an F1-score of 0.571 for the high-consumption class. These results demonstrate a working survey-data classification pipeline, not an fMRI-based classifier or clinical diagnostic tool.

## 1. Problem statement

The aim of this implementation is to investigate whether student survey and demographic characteristics can distinguish records in a project-defined high alcohol-consumption category from records in a lower-consumption category.

The original research topic focuses on adolescent excessive drinking and fMRI data. The present implementation is an adaptation using an alternative dataset. It does not measure brain activity and does not reproduce the original experiment.

## 2. Dataset description

The dataset was obtained from the UCI Machine Learning Repository's Student Performance dataset.

Source: https://archive.ics.uci.edu/dataset/320/student%2Bperformance

The Portuguese-language file contains 649 records and 33 original columns. Variables cover student demographics, social circumstances, school-related information, and self-reported alcohol consumption.

The two relevant variables are:

- **Dalc:** workday alcohol consumption rating, on a scale from 1 to 5.
- **Walc:** weekend alcohol consumption rating, on a scale from 1 to 5.

These ordinal survey ratings are not direct measurements of alcohol quantity and are not a clinical assessment.

### 2.1 Target construction

A binary variable named `high_consumption` was created:

- Class 1: `Dalc >= 4` or `Walc >= 4`.
- Class 0: both `Dalc < 4` and `Walc < 4`.

This threshold is an operational definition chosen for the project. It should not be interpreted as a validated clinical definition of excessive drinking.

The resulting class distribution was:

| Class | Records | Percentage |
|---|---:|---:|
| Lower consumption (0) | 509 | 78.4% |
| High consumption (1) | 140 | 21.6% |
| Total | 649 | 100% |

The `Dalc` and `Walc` columns were removed from model inputs because they define the target. This prevents direct target leakage.

## 3. Methodology

### 3.1 Preprocessing

The input variables were separated into numerical and categorical features. Numerical missing values, if present, were handled using median imputation, followed by standardization. Categorical missing values were handled using most-frequent imputation, and categorical variables were one-hot encoded.

Preprocessing was implemented within a scikit-learn pipeline. Consequently, imputation, scaling, and encoding were fitted on training data within each cross-validation fold rather than on the full dataset.

### 3.2 Train/test split and validation

The dataset was divided into training and test sets using an 80/20 stratified split with random state 42. The held-out test set contained 130 records, including 102 lower-consumption and 28 high-consumption records.

Five-fold StratifiedKFold cross-validation, with shuffling and random state 42, was applied to the training set. F1-score was used as the cross-validation scoring metric.

Each row represents a student survey record in the supplied dataset. No repeated-visit structure was used in this implementation; participant-grouped validation was therefore not performed.

### 3.3 Models

**Logistic Regression:** A linear classification baseline with a maximum of 2,000 iterations and balanced class weights.

**RBF-kernel SVM:** A Support Vector Machine using the radial basis function kernel and balanced class weights. Probability estimates were enabled to calculate ROC-AUC.

Balanced class weights were used to reduce the influence of the unequal class frequencies during model fitting.

### 3.4 Evaluation metrics

- **Accuracy:** proportion of all test records classified correctly.
- **Balanced accuracy:** average recall across the two classes.
- **Precision:** proportion of predicted high-consumption records that were labelled high consumption.
- **Recall:** proportion of actual high-consumption records identified by the model.
- **F1-score:** harmonic mean of precision and recall.
- **ROC-AUC:** measures ranking discrimination across classification thresholds.
- **Confusion matrix:** counts correct and incorrect predictions for each class.

## 4. Results

The following values were generated by the current implementation.

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

### 4.1 Confusion matrices

Rows indicate actual classes and columns indicate predicted classes, in the order lower consumption and high consumption.

**Logistic Regression**

| | Predicted lower | Predicted high |
|---|---:|---:|
| Actual lower | 77 | 25 |
| Actual high | 8 | 20 |

**RBF SVM**

| | Predicted lower | Predicted high |
|---|---:|---:|
| Actual lower | 85 | 17 |
| Actual high | 10 | 18 |

The generated figures are stored in `results/model_comparison.png`, `results/confusion_matrices.png`, and `results/class_distribution.png`.

### 4.2 Interpretation

RBF SVM achieved higher test accuracy, precision, F1-score, and ROC-AUC in this split. Logistic Regression achieved higher recall for the high-consumption class and a slightly higher mean cross-validation F1-score.

Logistic Regression identified 20 of 28 high-consumption test records, whereas SVM identified 18. SVM made fewer false-positive predictions among lower-consumption records, but it missed more high-consumption records than Logistic Regression.

The cross-validation results do not establish that one model is consistently superior. The test-set difference is modest, and further validation would be needed.

## 5. Discussion and limitations

### Dataset and target limitations

The UCI dataset is a student survey dataset, not an fMRI dataset. The target is constructed from self-reported consumption ratings, and the selected threshold is project-defined rather than clinically validated. Results must not be described as identifying alcohol use disorder or diagnosing individual students.

### Class imbalance

Only 21.6% of records belong to the high-consumption class. As a result, accuracy alone can give an incomplete view of model performance. Precision, recall, F1-score, balanced accuracy, ROC-AUC, and the confusion matrices provide additional context.

### Generalization

The results come from one stratified train/test split and five-fold cross-validation on the training set. Performance may vary with other splits or populations. The dataset's limited size and context restrict generalization.

### Confounding and causality

Student characteristics may be associated with reported consumption, but predictive associations do not establish causation. Observational survey data cannot establish that a feature causes alcohol consumption.

### Overfitting and model selection

The experiment compares two baseline models. It does not constitute exhaustive hyperparameter tuning or external validation. The small difference in scores should not be interpreted as definitive evidence of model superiority.

### Difference from the original fMRI approach

The original project concerns fMRI-derived features and excessive drinking classification. This adaptation uses survey data and does not extract brain-region time-series features, perform fMRI parcellation, or compare neural-network architectures. These differences are central limitations, not details that can be ignored.

## 6. Conclusion

A reproducible baseline classification pipeline was implemented using the UCI Student Performance dataset. Logistic Regression and RBF SVM were evaluated with preprocessing inside the training pipeline, stratified splitting, cross-validation, and multiple classification metrics.

RBF SVM achieved 79.2% test accuracy, an F1-score of 0.571, and ROC-AUC of 0.854. Logistic Regression achieved higher recall for the high-consumption class. These results are preliminary and specific to the chosen survey-based target.

Future work could examine alternative target definitions, threshold sensitivity, calibration, additional validation, and other authorized datasets. A genuine extension of the original research would require appropriate fMRI data and a valid participant-level target.

## 7. Reproducibility

From the repository root, activate the environment and run:

```bash
source .venv/bin/activate
python src/prepare_dataset.py
python src/train_baseline.py
python src/visualize_results.py
```

The dataset must be obtained from the UCI source and placed at `data/student-por.csv`. The scripts use a fixed random state of 42 for the train/test split and cross-validation.

Generated metrics are stored in `results/baseline_metrics.json`; plots are saved in the `results/` directory.

## References

1. Cortez, P., & Silva, A. (2008). Using data mining to predict secondary school student performance. Proceedings of the 5th Future Business Technology Conference (FUBUTEC).

2. UCI Machine Learning Repository. Student Performance dataset. https://archive.ics.uci.edu/dataset/320/student%2Bperformance

3. Noh, J. et al. Original CS229 project materials, *Classifying Adolescent Excessive Alcohol Drinkers from fMRI Data*. https://github.com/jnoh4/CS229ToShare

The original project materials are cited for motivation and context only. The reported numerical results in this report were produced by the alternative UCI survey-data implementation.

