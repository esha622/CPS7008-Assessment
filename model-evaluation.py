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

model = joblib.load(
    MODELS / "churn_pipeline.joblib"
)

settings = joblib.load(
    MODELS / "settings.joblib"
)

prob = model.predict_proba(X_test)[:, 1]

pred = (
    prob >= settings["threshold"]
).astype(int)

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

predictions = pd.DataFrame({

    "actual_churn": y_test.to_numpy(),

    "predicted_churn": pred,

    "churn_probability": prob
})

predictions.to_csv(
    OUTPUTS / "test_predictions.csv",
    index=False
)

cm = confusion_matrix(
    y_test,
    pred,
    labels=[0, 1]
)

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

print("\nFINAL TEST RESULTS")

for name, value in metrics.items():
    print(name, ":", value)


print("\nCONFUSION MATRIX")
print(cm)

print("\nEvaluation completed.")