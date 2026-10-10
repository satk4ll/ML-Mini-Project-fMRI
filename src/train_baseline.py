"""Train/evaluate Logistic Regression and RBF-SVM on one-row-per-participant CSV."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"


def build_models():
    common = [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    return {
        "logistic_regression": Pipeline(common + [
            ("classifier", LogisticRegression(
                max_iter=3000, class_weight="balanced", random_state=42
            ))
        ]),
        "svm_rbf": Pipeline(common + [
            ("classifier", SVC(
                kernel="rbf", probability=True, class_weight="balanced", random_state=42
            ))
        ]),
    }


def calculate_metrics(y_true, pred, prob):
    result = {
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "confusion_matrix_tn_fp_fn_tp": confusion_matrix(
            y_true, pred, labels=[0, 1]
        ).ravel().tolist(),
    }
    result["roc_auc"] = (
        float(roc_auc_score(y_true, prob)) if len(np.unique(y_true)) == 2 else None
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path, help="Prepared feature CSV.")
    parser.add_argument("--label", default="label")
    parser.add_argument("--id-column", default="participant_id")
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    if args.label not in df.columns:
        raise ValueError(f"Label column {args.label!r} not found.")
    if args.id_column in df.columns:
        ids = df[args.id_column].astype(str)
        if ids.duplicated().any():
            raise ValueError(
                "Repeated participant IDs detected. Aggregate/resolve repeated visits "
                "or use participant-grouped evaluation before proceeding."
            )
        df = df.drop(columns=[args.id_column])

    y = pd.to_numeric(df.pop(args.label), errors="raise")
    if y.isna().any() or set(y.unique()) != {0, 1}:
        raise ValueError("Label must be complete and binary, encoded 0/1.")
    X = df.select_dtypes(include="number")
    if X.empty:
        raise ValueError("No numeric predictor columns found.")
    if len(X) < 10:
        raise ValueError("At least 10 rows are required by this script's basic guard.")
    min_class = int(y.value_counts().min())
    n_splits = min(args.folds, min_class)
    if n_splits < 2:
        raise ValueError("Need at least two participants in each class.")
    if X.columns.duplicated().any():
        raise ValueError("Duplicate feature names detected.")

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summaries = []
    for name, model in build_models().items():
        print(f"Cross-validating {name} ({n_splits} folds)...")
        # All transformations are inside the pipeline to prevent fold leakage.
        pred = cross_val_predict(model, X, y, cv=cv, method="predict")
        prob = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
        metrics = calculate_metrics(y, pred, prob)
        summaries.append({"model": name, **metrics})
        (RESULTS_DIR / f"{name}_metrics.json").write_text(
            json.dumps(metrics, indent=2) + "\n"
        )
        model.fit(X, y)
        joblib.dump(
            {"model": model, "feature_columns": list(X.columns), "label_column": args.label},
            RESULTS_DIR / f"{name}.joblib",
        )

    pd.DataFrame(summaries).to_csv(RESULTS_DIR / "baseline_metrics.csv", index=False)
    print(f"Saved metrics and fitted models under {RESULTS_DIR}")
    print(pd.DataFrame(summaries).to_string(index=False))
    print("These metrics are valid only if rows are independent participants and "
          "the feature/label definitions match the approved study protocol.")


if __name__ == "__main__":
    main()
