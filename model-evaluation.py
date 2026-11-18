
"""
STEP 6: FINAL MODEL EVALUATION

Evaluate the selected model on the test set.

This program calculates:
- Accuracy
- Balanced accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Average precision
- Brier score
- Confusion matrix
"""

import json
import joblib
import pandas as pd

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    brier_score_loss
)

from ml_utils import (
    MODELS,
    OUTPUTS
)


# Load datasets
(
    X_train,
    X_val,
    X_test,
    y_train,
    y_val,
    y_test
) = joblib.load(
    MODELS / "split_data.joblib"
)


# Load fitted model
model = joblib.load(
    MODELS / "churn_pipeline.joblib"
)


# Load chosen model settings
settings = joblib.load(
    MODELS / "settings.joblib"
)


# Predict probability of churn
prob = model.predict_proba(X_test)[:, 1]


# Convert probabilities into 0 or 1
pred = (
    prob >= settings["threshold"]
).astype(int)


# Evaluate the model
metrics = {

    "selected_model": settings["model"],

    "threshold": settings["threshold"],

    "accuracy": accuracy_score(
        y_test, pred
    ),

    "balanced_accuracy": balanced_accuracy_score(
        y_test, pred
    ),

    "precision": precision_score(
        y_test, pred,
        zero_division=0
    ),

    "recall": recall_score(
        y_test, pred,
        zero_division=0
    ),

    "f1": f1_score(
        y_test, pred,
        zero_division=0
    ),

    "roc_auc": roc_auc_score(
        y_test, prob
    ),

    "average_precision": average_precision_score(
        y_test, prob
    ),

    "brier_score": brier_score_loss(
        y_test, prob
    )
}


# Save the evaluation results
with open(
    OUTPUTS / "final_metrics.json",
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        metrics,
        file,
        indent=2
    )


# Save individual predictions
predictions = pd.DataFrame({

    "actual_churn": y_test.to_numpy(),

    "predicted_churn": pred,

    "churn_probability": prob
})

predictions.to_csv(
    OUTPUTS / "test_predictions.csv",
    index=False
)


# Create confusion matrix
cm = confusion_matrix(
    y_test,
    pred,
    labels=[0, 1]
)


# Display confusion matrix
ConfusionMatrixDisplay(
    cm,
    display_labels=["Retained", "Churned"]
).plot(
    cmap="Blues"
)

plt.title("Final Test Confusion Matrix")
plt.tight_layout()

plt.savefig(
    OUTPUTS / "confusion_matrix.png"
)

plt.close()


# Print results
print("\nFINAL TEST RESULTS")

for name, value in metrics.items():
    print(name, ":", value)


print("\nCONFUSION MATRIX")
print(cm)

print("\nEvaluation completed.")