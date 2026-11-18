
"""
STEP 4: TRAIN AND COMPARE CLASSIFICATION MODELS

Algorithms:
1. Dummy Classifier
2. Logistic Regression
3. Decision Tree
4. Random Forest
5. K-Nearest Neighbours
6. Support Vector Machine

All models use the same training data.
"""

import argparse
import joblib
import pandas as pd

from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate
)

from ml_utils import (
    make_pipeline,
    MODELS,
    OUTPUTS,
    SEED
)


# Allow a shorter run while testing code.
parser = argparse.ArgumentParser()

parser.add_argument(
    "--quick",
    action="store_true"
)

args = parser.parse_args()


# Load the saved data splits.
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


# List the machine learning algorithms.
names = [
    "Dummy",
    "Logistic Regression",
    "Decision Tree",
    "Random Forest",
    "KNN",
    "SVM"
]


# Quick mode uses fewer models and folds.
if args.quick:
    names = [
        "Dummy",
        "Logistic Regression",
        "Decision Tree"
    ]


# Store model results in this list.
results = []


# Use stratified cross-validation.
folds = StratifiedKFold(
    n_splits=3 if args.quick else 5,
    shuffle=True,
    random_state=SEED
)


# Train and evaluate every algorithm.
for name in names:

    print("\nTraining:", name)

    # Create preprocessing + classifier pipeline.
    pipeline = make_pipeline(name)

    # Cross-validation evaluates the model
    # across multiple training/validation folds.
    scores = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=folds,
        scoring={
            "roc_auc": "roc_auc",
            "average_precision": "average_precision",
            "f1": "f1",
            "recall": "recall",
            "precision": "precision"
        },
        n_jobs=1
    )

    # Save average scores for this model.
    row = {
        "model": name
    }

    for metric in [
        "roc_auc",
        "average_precision",
        "f1",
        "recall",
        "precision"
    ]:

        row[metric] = scores[
            f"test_{metric}"
        ].mean()

        # Standard deviation shows variation
        # between the different folds.
        row[metric + "_std"] = scores[
            f"test_{metric}"
        ].std()

    results.append(row)


# Convert results into a table.
comparison = pd.DataFrame(results)

# Rank models by average precision.
comparison = comparison.sort_values(
    "average_precision",
    ascending=False
)

# Save the model comparison.
comparison.to_csv(
    OUTPUTS / "model_comparison.csv",
    index=False
)

print("\nMODEL COMPARISON")
print(comparison.to_string(index=False))

print("\nResults saved successfully.")