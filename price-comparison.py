
"""
OPTIONAL PRICING EXPERIMENT

Compare:
1. Customer-only features
2. Customer + historical pricing features

IMPORTANT:
This is a retrospective experiment.
The dataset does not establish a safe
prediction-time cutoff for every customer.
"""

import pandas as pd

from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score
)

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from ml_utils import (
    DATA,
    OUTPUTS,
    NUMERIC,
    CATEGORICAL,
    SEED,
    load_customers,
    make_price_features
)


# Load datasets
customers = load_customers()

prices = pd.read_csv(
    DATA / "price_data.csv"
)


# Aggregate historical prices
prices_aggregated = make_price_features(prices)


# Merge one pricing row per customer
joined = customers.merge(
    prices_aggregated,
    on="id",
    how="left",
    validate="one_to_one"
)


# Ensure the merge did not duplicate customers
assert len(joined) == len(customers)


# Find newly created price features
price_features = [
    column
    for column in prices_aggregated.columns
    if column != "id"
]


# Cross-validation configuration
cv = StratifiedKFold(
    n_splits=3,
    shuffle=True,
    random_state=SEED
)


# Store experimental results
rows = []


# Compare two feature sets
experiments = [
    ("Customer only", NUMERIC),

    (
        "Customer + historical pricing",
        NUMERIC + price_features
    )
]


for label, numerical in experiments:

    print("\nRunning experiment:", label)

    X = joined[numerical + CATEGORICAL]

    y = joined["churn"]


    # Preprocess the features
    prepare = ColumnTransformer([

        (
            "numbers",
            Pipeline([
                (
                    "fill",
                    SimpleImputer(
                        strategy="median"
                    )
                ),
                (
                    "scale",
                    StandardScaler()
                )
            ]),
            numerical
        ),

        (
            "categories",
            Pipeline([
                (
                    "fill",
                    SimpleImputer(
                        strategy="most_frequent"
                    )
                ),
                (
                    "encode",
                    OneHotEncoder(
                        handle_unknown="ignore"
                    )
                )
            ]),
            CATEGORICAL
        )
    ])


    # Use the same classifier for both experiments
    pipeline = Pipeline([

        ("prepare", prepare),

        (
            "model",
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=SEED
            )
        )
    ])


    # Cross-validation with average precision
    scores = cross_val_score(
        pipeline,
        X,
        y,
        scoring="average_precision",
        cv=cv
    )


    rows.append({

        "feature_set": label,

        "average_precision_mean": scores.mean(),

        "average_precision_std": scores.std()
    })


# Compare results
result = pd.DataFrame(rows)

result.to_csv(
    OUTPUTS / "price_comparison_retrospective.csv",
    index=False
)


print("\nPRICING EXPERIMENT RESULTS")
print(result.to_string(index=False))

print(
    "\nWarning: this comparison is retrospective, "
    "not a verified forward-looking prediction."
)