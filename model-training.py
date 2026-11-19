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
names = [
    "Dummy",
    "Logistic Regression",
    "Decision Tree",
    "Random Forest",
    "KNN",
    "SVM"
]

if args.quick:
    names = [
        "Dummy",
        "Logistic Regression",
        "Decision Tree"
    ]
results = []

folds = StratifiedKFold(
    n_splits=3 if args.quick else 5,
    shuffle=True,
    random_state=SEED
)
for name in names:

    print("\nTraining:", name)
    pipeline = make_pipeline(name)
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
        row[metric + "_std"] = scores[
            f"test_{metric}"
        ].std()

    results.append(row)
comparison = pd.DataFrame(results)
comparison = comparison.sort_values(
    "average_precision",
    ascending=False
)

comparison.to_csv(
    OUTPUTS / "model_comparison.csv",
    index=False
)

print("\nMODEL COMPARISON")
print(comparison.to_string(index=False))

print("\nResults saved successfully.")