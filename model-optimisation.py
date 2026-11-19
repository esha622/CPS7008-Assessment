import argparse
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import (
    StratifiedKFold,
    GridSearchCV
)

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score
)

from ml_utils import (
    MODELS,
    OUTPUTS,
    SEED,
    make_pipeline
)

parser = argparse.ArgumentParser()

parser.add_argument(
    "--quick",
    action="store_true"
)

args = parser.parse_args()
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

comparison = pd.read_csv(
    OUTPUTS / "model_comparison.csv"
)

best_name = comparison.loc[
    comparison["model"] != "Dummy",
    "model"
].iloc[0]

print("Model selected for tuning:", best_name)
params = {

    "Logistic Regression": {
        "model__C": [0.1, 1.0, 10.0]
    },

    "Decision Tree": {
        "model__max_depth": [4, 7, 12],
        "model__min_samples_leaf": [10, 30]
    },

    "Random Forest": {
        "model__max_depth": [6, 12],
        "model__min_samples_leaf": [5, 15]
    },

    "KNN": {
        "model__n_neighbors": [7, 15, 31]
    },

    "SVM": {
        "model__C": [0.3, 1.0, 3.0]
    }
}

cv = StratifiedKFold(
    n_splits=3 if args.quick else 5,
    shuffle=True,
    random_state=SEED
)

search = GridSearchCV(
    estimator=make_pipeline(best_name),
    param_grid=params[best_name],
    scoring="average_precision",
    cv=cv,
    n_jobs=1
)

search.fit(X_train, y_train)
model = search.best_estimator_
prob = model.predict_proba(X_val)[:, 1]

threshold_rows = []
for threshold in np.arange(0.10, 0.71, 0.05):
    prediction = (
        prob >= threshold
    ).astype(int)

    threshold_rows.append({

        "threshold": round(
            float(threshold), 2
        ),

        "precision": precision_score(
            y_val,
            prediction,
            zero_division=0
        ),

        "recall": recall_score(
            y_val,
            prediction,
            zero_division=0
        ),

        "f1": f1_score(
            y_val,
            prediction,
            zero_division=0
        )
    })

table = pd.DataFrame(threshold_rows)
chosen_threshold = float(
    table.sort_values(
        "f1",
        ascending=False
    ).iloc[0]["threshold"]
)

table.to_csv(
    OUTPUTS / "validation_thresholds.csv",
    index=False
)
joblib.dump(
    model,
    MODELS / "churn_pipeline.joblib"
)

joblib.dump(
    {
        "model": best_name,
        "threshold": chosen_threshold,
        "best_parameters": search.best_params_,
        "cross_validation_average_precision":
            search.best_score_
    },
    MODELS / "settings.joblib"
)

print("\nBest parameters:")
print(search.best_params_)

print("\nBest cross-validation average precision:")
print(search.best_score_)

print("\nSelected validation threshold:")
print(chosen_threshold)

print("\nModel saved successfully.")