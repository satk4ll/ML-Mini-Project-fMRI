# Presentation outline (replace placeholders after experiments)

1. **Title and team** — project title, course, roles.
2. **Motivation and task** — research question; emphasize research classification, not clinical diagnosis.
3. **Dataset** — NCANDA context, authorized sample size, inclusion criteria, labels; fill only from verified data.
4. **fMRI representation** — BOLD time courses, time points, brain-region parcellation.
5. **Feature engineering** — per-region range (max − min), optional demographic covariates and justification.
6. **Models** — Logistic Regression baseline and RBF-SVM comparison.
7. **Validation** — stratified cross-validation; participant-level independence; leakage controls.
8. **Results** — real metrics and confusion matrices only; compare with paper carefully.
9. **Limitations and ethics** — small sample, confounds, generalizability, access/privacy constraints.
10. **Conclusion and next steps** — supported findings, what remains uncertain, reproducibility.

Demo plan: show repository structure, feature extraction command, cross-validation command, saved metrics and plots if available. If dataset access is still pending, demonstrate the software tests only and explicitly state that no model performance results are available.
