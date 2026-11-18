
"""
Shared functions for the PowerCo churn project.

This file contains:
1. Folder paths
2. Feature names
3. Data loading
4. Data validation
5. Model preprocessing
6. Classification algorithms
7. Pricing feature engineering
"""

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


# PROJECT FOLDERS

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUTPUTS = ROOT / "outputs"
MODELS = ROOT / "models"

SEED = 42

# Create output folders automatically
OUTPUTS.mkdir(exist_ok=True)
MODELS.mkdir(exist_ok=True)


# CUSTOMER FEATURES

# Numerical features contain quantities or measurements.
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

# Categorical features contain labels or groups.
CATEGORICAL = [
    "channel_sales",
    "has_gas",
    "origin_up"
]

FEATURES = NUMERIC + CATEGORICAL


# HISTORICAL PRICING COLUMNS

PRICE_COLS = [
    "price_off_peak_var",
    "price_peak_var",
    "price_mid_peak_var",
    "price_off_peak_fix",
    "price_peak_fix",
    "price_mid_peak_fix"
]


# LOAD CUSTOMER DATA

def load_customers():

    customers = pd.read_csv(
        DATA / "client_data.csv"
    )

    check_customers(customers)

    return customers


# CHECK CUSTOMER DATA

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


# SELECT INPUTS AND TARGET

def get_xy(df):

    # X contains the features used for prediction.
    X = df[FEATURES].copy()

    # y contains the actual churn outcomes.
    y = df["churn"].astype(int).copy()

    return X, y


# CREATE MACHINE LEARNING PIPELINE

def make_pipeline(name):

    # Prepare numerical features.
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

    # Prepare categorical features.
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

    # Apply the correct preparation to each group.
    preparation = ColumnTransformer([
        ("numbers", numeric_steps, NUMERIC),
        ("categories", categorical_steps, CATEGORICAL)
    ])

    # Define our classification algorithms.
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

    # Return preparation and model together.
    return Pipeline([
        ("prepare", preparation),
        ("model", algorithms[name])
    ])


# AGGREGATE HISTORICAL PRICING RECORDS

def make_price_features(prices):

    prices = prices.copy()

    # Convert date text into datetime values.
    prices["price_date"] = pd.to_datetime(
        prices["price_date"],
        errors="coerce"
    )

    # Remove records with invalid identifiers/dates.
    prices = prices.dropna(
        subset=["id", "price_date"]
    )

    # Sort each customer's history by date.
    prices = prices.sort_values([
        "id",
        "price_date"
    ])

    # Produce one row per customer.
    result = prices.groupby("id")[PRICE_COLS].agg([
        "mean",
        "std",
        "min",
        "max"
    ])

    # Convert multi-level column names into simple names.
    result.columns = [
        f"{column}_{statistic}"
        for column, statistic in result.columns
    ]

    return result.reset_index()
