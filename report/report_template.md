# Classifying Adolescent Excessive Alcohol Drinkers from fMRI Data

**Course:** UE24CS352A Machine Learning  
**Student:** Satyanarayana Kallakuri  
**Institution:** PES University, Bengaluru

## 1. Problem statement
Describe the binary classification task and intended research question. Avoid implying clinical diagnosis or causal inference.

## 2. Dataset and ethics
- Dataset/access approval:
- Inclusion/exclusion criteria:
- Number of participants after quality checks:
- Class distribution:
- Parcellation and number of regions:
- Time-course format and number of time points:
- Data-use/privacy restrictions:

Complete this section only from the authorized dataset and its documentation.

## 3. Preprocessing and features
Describe time-course normalization as performed by the source dataset, time-point selection, participant matching, handling of invalid files, and the per-region range feature (maximum minus minimum). State whether age, sex, or scanner are predictors. Explain why identifiers are never predictors.

## 4. Models and validation
Describe Logistic Regression and RBF-SVM. State the validation scheme, random seed, fold count, and any class weighting. Confirm each row is an independent participant; if repeated visits occur, use participant-grouped splitting. Fit imputation/scaling only within each training fold.

## 5. Results
Insert only results produced from authorized real data. Report accuracy, balanced accuracy, precision, recall, F1, ROC-AUC, and confusion matrices. Include fold variability where available. Do not copy results from the paper as if they were produced by this implementation.

## 6. Discussion and limitations
Discuss sample size, class imbalance, possible confounding (including age/scanner), overfitting, dataset access, generalizability, and why classification does not establish causality or diagnose individuals.

## 7. Reproducibility
List Python version, package versions, commands, repository commit, and data preprocessing decisions.

## References
Cite the original paper, source code repository, and dataset documentation/approval as applicable.
