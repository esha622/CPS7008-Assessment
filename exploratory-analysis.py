
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ml_utils import (
    load_customers,
    OUTPUTS
)

customers = load_customers()

figures = OUTPUTS / "figures"
figures.mkdir(exist_ok=True)

customers["churn"].value_counts().sort_index().plot.bar(
    color=["steelblue", "tomato"]
)

plt.title("Customer Churn Distribution")
plt.xlabel("Churn: 0 = Retained, 1 = Churned")
plt.ylabel("Number of Customers")
plt.tight_layout()

plt.savefig(
    figures / "01_churn_counts.png"
)

plt.close()

customers["tenure_group"] = pd.cut(
    customers["num_years_antig"],
    bins=[-1, 2, 5, 10, 100],
    labels=["0-2", "3-5", "6-10", "11+"]
)

summary = customers.groupby(
    "tenure_group",
    observed=True
).agg(
    customers=("churn", "size"),
    churn_rate=("churn", "mean")
)

summary.to_csv(
    OUTPUTS / "tenure_churn_summary.csv"
)

(summary["churn_rate"] * 100).plot.bar(
    color="steelblue"
)

plt.title("Churn Rate by Customer Tenure")
plt.xlabel("Customer Tenure (Years)")
plt.ylabel("Churn Rate (%)")
plt.tight_layout()

plt.savefig(
    figures / "02_tenure_churn.png"
)

plt.close()

for churn, group in customers.groupby("churn"):
    consumption = np.log1p(
        group["cons_12m"].clip(lower=0)
    )

    plt.hist(
        consumption,
        bins=35,
        alpha=0.5,
        density=True,
        label=f"Churn {churn}"
    )

plt.title("Electricity Consumption by Churn")
plt.xlabel("log(1 + Annual Consumption)")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()

plt.savefig(
    figures / "03_consumption.png"
)

plt.close()


print("EDA completed.")
print("Charts saved in outputs/figures/")