
"""
STEP 3: DATA PREPROCESSING

We divide the customer dataset into:
- Training data: 60%
- Validation data: 20%
- Testing data: 20%

The actual scaling and encoding are performed
inside the model pipeline in ml_utils.py.
"""

import joblib

from sklearn.model_selection import train_test_split

from ml_utils import (
    load_customers,
    get_xy,
    MODELS,
    SEED
)


# Load the dataset
customers = load_customers()

# Separate features (X) and target (y)
X, y = get_xy(customers)


# First split:
# 80% development data and 20% final test data
X_work, X_test, y_work, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=SEED
)


# Second split:
# 75% of development data for training
# 25% of development data for validation
X_train, X_val, y_train, y_val = train_test_split(
    X_work,
    y_work,
    test_size=0.25,
    stratify=y_work,
    random_state=SEED
)


# Save the splits for later stages.
# We save them rather than repeatedly changing them.
joblib.dump(
    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test
    ),
    MODELS / "split_data.joblib"
)


print("Training records:", len(X_train))
print("Validation records:", len(X_val))
print("Testing records:", len(X_test))

print("\nData splitting completed.")