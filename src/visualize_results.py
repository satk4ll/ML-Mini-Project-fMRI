
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay

RESULTS = Path("results")
with open(RESULTS / "baseline_metrics.json", encoding="utf-8") as f:
    results = json.load(f)

models = list(results.keys())

# Graph 1: test-set metrics
metrics = [
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
]

x = np.arange(len(metrics))
width = 0.35

fig, ax = plt.subplots(figsize=(11, 5))

for i, model in enumerate(models):
    values = [results[model][metric] for metric in metrics]
    bars = ax.bar(
        x + (i - 0.5) * width,
        values,
        width,
        label=model,
    )
    ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=8)

ax.set_xticks(x)
ax.set_xticklabels(
    ["Accuracy", "Balanced accuracy", "Precision",
     "Recall", "F1-score", "ROC-AUC"],
    rotation=15,
)
ax.set_ylim(0, 1.15)
ax.set_ylabel("Score")
ax.set_title("Baseline Model Performance on Test Set")
ax.legend()
fig.tight_layout()
fig.savefig(RESULTS / "model_comparison.png", dpi=200)
plt.close(fig)

# Graph 2: confusion matrices
fig, axes = plt.subplots(1, len(models), figsize=(10, 4))

for ax, model in zip(axes, models):
    display = ConfusionMatrixDisplay(
        confusion_matrix=np.array(
            results[model]["confusion_matrix"]
        ),
        display_labels=["Lower", "High"],
    )
    display.plot(ax=ax, colorbar=False, values_format="d")
    ax.set_title(model)

fig.suptitle("Confusion Matrices — Test Set")
fig.tight_layout()
fig.savefig(RESULTS / "confusion_matrices.png", dpi=200)
plt.close(fig)

# Graph 3: target distribution
df = pd.read_csv("data/student-por-labelled.csv")
counts = df["high_consumption"].value_counts().sort_index()

fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.bar(
    ["Lower consumption", "High consumption"],
    [counts.get(0, 0), counts.get(1, 0)],
)
ax.bar_label(bars, padding=3)
ax.set_ylabel("Number of records")
ax.set_title("Target Class Distribution")
fig.tight_layout()
fig.savefig(RESULTS / "class_distribution.png", dpi=200)
plt.close(fig)

print("Updated all three graphs in results/")

