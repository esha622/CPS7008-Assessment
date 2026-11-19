import joblib
import pandas as pd
from sklearn.inspection import permutation_importance

from sklearn.metrics import (
    recall_score,
    precision_score
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

importance = permutation_importance(
    model,
    X_test,
    y_test,
    scoring="average_precision",
    n_repeats=3,
    random_state=42,
    n_jobs=1
)

importance_table = pd.DataFrame({

    "feature": X_test.columns,

    "importance_mean": importance.importances_mean,

    "importance_std": importance.importances_std
})

importance_table = importance_table.sort_values(
    "importance_mean",
    ascending=False
)

importance_table.to_csv(
    OUTPUTS / "feature_importance.csv",
    index=False
)

prob = model.predict_proba(X_test)[:, 1]
pred = (
    prob >= settings["threshold"]
).astype(int)

rows = []

for category in X_test["has_gas"].dropna().unique():

    mask = (
        X_test["has_gas"] == category
    ).to_numpy()

    truth = y_test.to_numpy()[mask]

    predictions = pred[mask]

    rows.append({

        "has_gas": category,

        "count": int(mask.sum()),

        "actual_churn_rate": float(truth.mean()),

        "recall": recall_score(
            truth,
            predictions,
            zero_division=0
        ),

        "precision": precision_score(
            truth,
            predictions,
            zero_division=0
        )
    })

segment_table = pd.DataFrame(rows)

segment_table.to_csv(
    OUTPUTS / "segment_metrics.csv",
    index=False
)


print("Interpretability analysis completed.")

print("\nMost important features:")
print(importance_table.head(10))

print("\nCustomer segment results:")
print(segment_table)