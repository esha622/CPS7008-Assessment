import joblib

from sklearn.model_selection import train_test_split

from ml_utils import (
    load_customers,
    get_xy,
    MODELS,
    SEED
)

customers = load_customers()
X, y = get_xy(customers)


X_work, X_test, y_work, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=SEED
)

X_train, X_val, y_train, y_val = train_test_split(
    X_work,
    y_work,
    test_size=0.25,
    stratify=y_work,
    random_state=SEED
)

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