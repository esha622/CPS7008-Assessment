import pandas as pd

from ml_utils import (
    DATA,
    OUTPUTS,
    load_customers
)


customers = load_customers()

prices = pd.read_csv(
    DATA / "price_data.csv"
)

print("Customer dataset:", customers.shape)
print("Pricing dataset:", prices.shape)


print("\nCUSTOMER DATA")
print(customers.head())

print("\nPRICING DATA")
print(prices.head())


print("\nCHURN COUNTS")
print(customers["churn"].value_counts())

print("\nCHURN PERCENTAGES")
print(
    customers["churn"].value_counts(
        normalize=True
    ) * 100
)

print("\nMissing customer values:")
print(customers.isna().sum().sum())

print("\nMissing pricing values:")
print(prices.isna().sum().sum())


print("\nUnique customer IDs:")
print(customers["id"].nunique())

print("\nCustomer IDs in pricing dataset:")
print(prices["id"].nunique())

print("\nDuplicate customer IDs:")
print(customers["id"].duplicated().sum())


missing_prices = (
    ~customers["id"].isin(prices["id"])
).sum()

print("\nCustomers without pricing records:")
print(missing_prices)

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