"""Generate a messy, realistic sample customer dataset (data/customers.csv).

It deliberately contains missing values, outliers, skewed columns, duplicate rows,
a constant column and a near-duplicate numeric column, so every step of the
platform has something to do.
"""

from pathlib import Path

import numpy as np
import pandas as pd


def make_dataframe(n=500, seed=42):
    rng = np.random.default_rng(seed)

    signup = pd.Timestamp("2022-01-01") + pd.to_timedelta(rng.integers(0, 1000, n), unit="D")
    last_purchase = signup + pd.to_timedelta(rng.integers(0, 400, n), unit="D")
    last_purchase = last_purchase.where(last_purchase <= pd.Timestamp("2026-11-01"),
                                        pd.Timestamp("2026-11-01"))

    age = rng.normal(38, 11, n).clip(18, 80).round()
    income = rng.lognormal(10.8, 0.45, n).round(2)
    income[rng.choice(n, 6, replace=False)] *= 8                    # extreme outliers
    num_orders = rng.poisson(8, n) + 1
    total_spend = (num_orders * rng.gamma(4, 30, n)).round(2)
    support_tickets = rng.poisson(1.5, n)
    plan = rng.choice(["free", "basic", "premium"], n, p=[0.4, 0.4, 0.2])
    country = rng.choice(["IN", "US", "GB", "DE", "FR", "BR"], n, p=[0.3, 0.25, 0.15, 0.1, 0.1, 0.1])

    logit = -1.0 + 0.5 * support_tickets - 0.08 * num_orders + 0.8 * (plan == "free")
    churned = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)

    df = pd.DataFrame({
        "customer_id": [f"CUST{i:04d}" for i in range(1, n + 1)],
        "signup_date": signup.strftime("%Y-%m-%d"),
        "last_purchase_date": last_purchase.strftime("%Y-%m-%d"),
        "age": age,
        "income": income,
        "num_orders": num_orders,
        "total_spend": total_spend,
        "total_spend_cents": (total_spend * 100 + rng.normal(0, 50, n)).round(),  # ~ duplicate of total_spend
        "support_tickets": support_tickets,
        "country": country,
        "plan": plan,
        "source_system": "web",                                      # constant column
        "churned": churned,
    })

    # Inject missing values
    for col, share in [("age", 0.08), ("income", 0.10), ("country", 0.05), ("last_purchase_date", 0.04)]:
        df.loc[rng.random(n) < share, col] = np.nan

    # Inject duplicate rows and shuffle
    df = pd.concat([df, df.sample(10, random_state=seed)], ignore_index=True)
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


if __name__ == "__main__":
    out = Path(__file__).resolve().parent / "data" / "customers.csv"
    out.parent.mkdir(exist_ok=True)
    make_dataframe().to_csv(out, index=False)
    print(f"Wrote {out}")
