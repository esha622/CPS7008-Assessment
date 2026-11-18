
"""
STEP 1: UNDERSTANDING THE DATA

This program checks:
- Dataset size
- Churn distribution
- Missing values
- Customer IDs
- Descriptive statistics
"""

import pandas as pd

from ml_utils import (
    DATA,
    OUTPUTS,
    load_customers
)


# Load the customer dataset
customers = load_customers()

# Load historical pricing data
prices = pd.read_csv(
    DATA / "price_data.csv"
)


# Display dataset dimensions
print("Customer dataset:", customers.shape)
print("Pricing dataset:", prices.shape)


# Display the first records
print("\nCUSTOMER DATA")
print(customers.head())

print("\nPRICING DATA")
print(prices.head())


# Display churn distribution
print("\nCHURN COUNTS")
print(customers["churn"].value_counts())

print("\nCHURN PERCENTAGES")
print(
    customers["churn"].value_counts(
        normalize=True
    ) * 100
)


# Check missing values
print("\nMissing customer values:")
print(customers.isna().sum().sum())

print("\nMissing pricing values:")
print(prices.isna().sum().sum())


# Check customer IDs
print("\nUnique customer IDs:")
print(customers["id"].nunique())

print("\nCustomer IDs in pricing dataset:")
print(prices["id"].nunique())

print("\nDuplicate customer IDs:")
print(customers["id"].duplicated().sum())


# Check customer coverage in pricing data
missing_prices = (
    ~customers["id"].isin(prices["id"])
).sum()

print("\nCustomers without pricing records:")
print(missing_prices)


# Save descriptive statistics
customers.describe(
    include="all"
).T.to_csv(
    OUTPUTS / "customer_summary.csv"
)

prices.describe(
    include="all"
).T.to_csv(
    OUTPUTS / "price_summary.csv"
)

print("\nData investigation completed.")