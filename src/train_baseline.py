import json
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

DATA_PATH = Path("data/student-por-labelled.csv")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def main():
    df = pd.read_csv(DATA_PATH)
    target = "high_consumption"

    # Dalc and Walc define the target, so exclude them.
    X = df.drop(columns=[target, "Dalc", "Walc"])
    y = df[target]

    categorical = X.select_dtypes(
        include=["object", "category", "bool", "str"]
    ).columns.tolist()
    numerical = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocess = ColumnTransformer([
        ("numeric", numeric_pipeline, numerical),
        ("categorical", categorical_pipeline, categorical),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=42
        ),
        "RBF SVM": SVC(
            kernel="rbf", class_weight="balanced",
            probability=True, random_state=42
        ),
    }

    results = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in models.items():
        pipeline = Pipeline([
            ("preprocess", preprocess),
            ("model", model),
        ])

        cv_scores = cross_val_score(
            pipeline, X_train, y_train, cv=cv, scoring="f1"
        )
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        probabilities = pipeline.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "balanced_accuracy": float(
                balanced_accuracy_score(y_test, predictions)
            ),
            "precision": float(
                precision_score(y_test, predictions, zero_division=0)
            ),
            "recall": float(
                recall_score(y_test, predictions, zero_division=0)
            ),
            "f1": float(f1_score(y_test, predictions, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, probabilities)),
            "cv_f1_mean": float(cv_scores.mean()),
            "cv_f1_std": float(cv_scores.std()),
            "confusion_matrix": confusion_matrix(
                y_test, predictions, labels=[0, 1]
            ).tolist(),
        }
        results[name] = metrics

        print(f"\n{'=' * 50}\n{name}\n{'=' * 50}")
        for key, value in metrics.items():
            if key != "confusion_matrix":
                print(f"{key}: {value:.3f}")
        print("Confusion matrix:", metrics["confusion_matrix"])
        print(classification_report(
            y_test, predictions,
            labels=[0, 1],
            target_names=["Lower consumption", "High consumption"],
            zero_division=0,
        ))

    output = RESULTS_DIR / "baseline_metrics.json"
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved results to {output}")


if __name__ == "__main__":
    main()
