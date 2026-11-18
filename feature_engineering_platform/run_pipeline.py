"""
Feature Engineering and Data Preparation Platform: command line runner.

Reads a CSV, profiles it, cleans it, splits it into train/test, fits the
feature pipeline on the training split only, transforms both splits and saves
the results together with the fitted pipeline.

Usage:
    python generate_sample_data.py
    python run_pipeline.py
    python run_pipeline.py --input data/customers.csv --target churned --out output
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from data_profiler import profile, quality_warnings
from pipeline import default_pipeline

DEFAULT_RATIOS = ["total_spend/num_orders:avg_order_value",
                  "support_tickets/num_orders:tickets_per_order"]


def parse_ratio(text):
    """'a/b:name' -> ('a', 'b', 'name')"""
    expr, name = text.split(":")
    num, den = expr.split("/")
    return num.strip(), den.strip(), name.strip()


def train_test_split(df, test_size=0.2, seed=42):
    idx = np.random.default_rng(seed).permutation(len(df))
    n_test = int(len(df) * test_size)
    test = df.iloc[idx[:n_test]].reset_index(drop=True)
    train = df.iloc[idx[n_test:]].reset_index(drop=True)
    return train, test


def run(input_csv, out_dir, target="churned", id_cols=("customer_id",),
        date_cols=("signup_date", "last_purchase_date"), ratios=tuple(DEFAULT_RATIOS),
        bin_cols=("age",), test_size=0.2, seed=42, scaler="standard"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(input_csv)
    prof = profile(raw)
    warnings = quality_warnings(prof, len(raw))
    prof.to_csv(out_dir / "profile.csv", index=False)

    # Row-level cleaning that must happen before the split
    df = raw.drop_duplicates().dropna(subset=[target]).reset_index(drop=True)

    train, test = train_test_split(df, test_size, seed)
    drop = [target, *[c for c in id_cols if c in df.columns]]
    X_train, y_train = train.drop(columns=drop), train[target]
    X_test, y_test = test.drop(columns=drop), test[target]

    pipe = default_pipeline(date_columns=date_cols,
                            ratio_specs=[parse_ratio(r) for r in ratios],
                            bin_columns=bin_cols, scaler=scaler)
    F_train = pipe.fit_transform(X_train, y_train)   # learn on training data only
    F_test = pipe.transform(X_test)                  # re-use what was learned

    F_train.assign(**{target: y_train.to_numpy()}).to_csv(out_dir / "train_features.csv", index=False)
    F_test.assign(**{target: y_test.to_numpy()}).to_csv(out_dir / "test_features.csv", index=False)
    pipe.report().to_csv(out_dir / "pipeline_report.csv", index=False)
    pipe.save(out_dir / "pipeline.pkl")

    return {
        "raw_rows": len(raw), "duplicates_removed": len(raw) - len(raw.drop_duplicates()),
        "train_rows": len(train), "test_rows": len(test),
        "raw_columns": raw.shape[1], "feature_columns": F_train.shape[1],
        "warnings": warnings, "report": pipe.report(), "features": pipe.feature_names_,
    }


def main():
    p = argparse.ArgumentParser(description="Feature engineering and data preparation")
    p.add_argument("--input", default="data/customers.csv")
    p.add_argument("--out", default="output")
    p.add_argument("--target", default="churned")
    p.add_argument("--id-cols", nargs="*", default=["customer_id"])
    p.add_argument("--date-cols", nargs="*", default=["signup_date", "last_purchase_date"])
    p.add_argument("--bin-cols", nargs="*", default=["age"])
    p.add_argument("--ratios", nargs="*", default=DEFAULT_RATIOS,
                   help="ratio features as numerator/denominator:new_name")
    p.add_argument("--scaler", choices=["standard", "minmax"], default="standard")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()

    result = run(a.input, a.out, a.target, a.id_cols, a.date_cols, a.ratios, a.bin_cols,
                 a.test_size, a.seed, a.scaler)

    print(f"Input: {result['raw_rows']} rows x {result['raw_columns']} columns "
          f"({result['duplicates_removed']} duplicate rows removed)")
    print(f"Split: {result['train_rows']} train / {result['test_rows']} test")

    print("\nData quality warnings:")
    for w in result["warnings"] or ["none"]:
        print(f"  - {w}")

    print("\nPipeline steps:")
    print(result["report"][["step", "columns_out", "added", "removed"]].to_string(index=False, max_colwidth=60))

    print(f"\nFinal feature set: {result['feature_columns']} columns")
    print("  " + ", ".join(result["features"]))
    print(f"\nSaved to {a.out}/: train_features.csv, test_features.csv, pipeline.pkl, "
          "profile.csv, pipeline_report.csv")


if __name__ == "__main__":
    main()
