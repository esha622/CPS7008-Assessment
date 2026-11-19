from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUTPUTS = ROOT / "outputs"
MODELS = ROOT / "models"

SEED = 42

OUTPUTS.mkdir(exist_ok=True)
MODELS.mkdir(exist_ok=True)
NUMERIC = [
    "cons_12m",
    "cons_gas_12m",
    "cons_last_month",
    "forecast_cons_12m",
    "forecast_cons_year",
    "forecast_discount_energy",
    "forecast_meter_rent_12m",
    "forecast_price_energy_off_peak",
    "forecast_price_energy_peak",
    "forecast_price_pow_off_peak",
    "imp_cons",
    "margin_gross_pow_ele",
    "margin_net_pow_ele",
    "nb_prod_act",
    "net_margin",
    "num_years_antig",
    "pow_max"
]

CATEGORICAL = [
    "channel_sales",
    "has_gas",
    "origin_up"
]

FEATURES = NUMERIC + CATEGORICAL

PRICE_COLS = [
    "price_off_peak_var",
    "price_peak_var",
    "price_mid_peak_var",
    "price_off_peak_fix",
    "price_peak_fix",
    "price_mid_peak_fix"
]

def load_customers():

    customers = pd.read_csv(
        DATA / "client_data.csv"
    )

    check_customers(customers)

    return customers

def check_customers(df):

    if df["id"].isna().any():
        raise ValueError("Some customer IDs are missing")

    if not df["id"].is_unique:
        raise ValueError("Duplicate customer IDs found")

    if not df["churn"].isin([0, 1]).all():
        raise ValueError("Churn must contain 0 or 1")

    missing = [
        col for col in FEATURES
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

def get_xy(df):
    X = df[FEATURES].copy()
    y = df["churn"].astype(int).copy()

    return X, y

def make_pipeline(name):
    numeric_steps = Pipeline([
        (
            "fill",
            SimpleImputer(strategy="median")
        ),
        (
            "scale",
            StandardScaler()
        )
    ])
    categorical_steps = Pipeline([
        (
            "fill",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "onehot",
            OneHotEncoder(handle_unknown="ignore")
        )
    ])
    preparation = ColumnTransformer([
        ("numbers", numeric_steps, NUMERIC),
        ("categories", categorical_steps, CATEGORICAL)
    ])
    algorithms = {

        "Dummy": DummyClassifier(
            strategy="prior"
        ),

        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=SEED
        ),

        "Decision Tree": DecisionTreeClassifier(
            max_depth=7,
            min_samples_leaf=10,
            class_weight="balanced",
            random_state=SEED
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=120,
            max_depth=12,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            random_state=SEED,
            n_jobs=-1
        ),

        "KNN": KNeighborsClassifier(
            n_neighbors=15
        ),

        "SVM": SVC(
            C=1,
            kernel="rbf",
            class_weight="balanced",
            probability=True,
            random_state=SEED
        )
    }

    if name not in algorithms:
        raise ValueError(
            "Unknown model: " + str(name)
        )

    return Pipeline([
        ("prepare", preparation),
        ("model", algorithms[name])
    ])

def make_price_features(prices):

    prices = prices.copy()
    prices["price_date"] = pd.to_datetime(
        prices["price_date"],
        errors="coerce"
    )

    prices = prices.dropna(
        subset=["id", "price_date"]
    )

    prices = prices.sort_values([
        "id",
        "price_date"
    ])

    result = prices.groupby("id")[PRICE_COLS].agg([
        "mean",
        "std",
        "min",
        "max"
    ])

    result.columns = [
        f"{column}_{statistic}"
        for column, statistic in result.columns
    ]

    return result.reset_index()
