
"""
STEP 5: MODEL OPTIMISATION

This program:
1. Reads the model comparison.
2. Selects the highest-ranked non-Dummy model.
3. Tunes its hyperparameters.
4. Chooses a classification threshold.
5. Saves the fitted model.
"""

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


# Quick mode for initial testing
parser = argparse.ArgumentParser()

parser.add_argument(
    "--quick",
    action="store_true"
)

args = parser.parse_args()


# Load saved datasets
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


# Read results from the previous stage
comparison = pd.read_csv(
    OUTPUTS / "model_comparison.csv"
)


# Dummy is only a reference model.
# Select the first ranked actual classifier.
best_name = comparison.loc[
    comparison["model"] != "Dummy",
    "model"
].iloc[0]

print("Model selected for tuning:", best_name)


# Define hyperparameter search spaces.
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


# Cross-validation for tuning.
cv = StratifiedKFold(
    n_splits=3 if args.quick else 5,
    shuffle=True,
    random_state=SEED
)


# Grid search tests different settings.
search = GridSearchCV(
    estimator=make_pipeline(best_name),
    param_grid=params[best_name],
    scoring="average_precision",
    cv=cv,
    n_jobs=1
)


# Fit only on the training dataset.
search.fit(X_train, y_train)


# Extract the best trained model.
model = search.best_estimator_


# Generate probabilities on validation customers.
prob = model.predict_proba(X_val)[:, 1]


# Test several possible classification thresholds.
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


# Create threshold comparison table.
table = pd.DataFrame(threshold_rows)


# Select the threshold with highest F1.
chosen_threshold = float(
    table.sort_values(
        "f1",
        ascending=False
    ).iloc[0]["threshold"]
)


# Save threshold experiment results.
table.to_csv(
    OUTPUTS / "validation_thresholds.csv",
    index=False
)


# Save the final fitted pipeline.
joblib.dump(
    model,
    MODELS / "churn_pipeline.joblib"
)


# Save model configuration.
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