"""Data profiling: a one-row-per-column summary plus simple data quality warnings."""

import pandas as pd
from pandas.api.types import is_numeric_dtype


def profile(df):
    """Return a DataFrame describing every column of df."""
    rows = []
    for col in df.columns:
        s = df[col]
        row = {
            "column": col,
            "dtype": str(s.dtype),
            "missing": int(s.isna().sum()),
            "missing_pct": round(100 * s.isna().mean(), 1),
            "unique": int(s.nunique(dropna=True)),
            "min": None, "max": None, "mean": None, "skew": None, "top_value": None,
        }
        if is_numeric_dtype(s) and s.notna().any():
            row["min"] = round(float(s.min()), 3)
            row["max"] = round(float(s.max()), 3)
            row["mean"] = round(float(s.mean()), 3)
            if s.nunique() > 2 and s.notna().sum() > 2:
                row["skew"] = round(float(s.skew()), 3)
        elif s.notna().any():
            row["top_value"] = str(s.mode(dropna=True).iloc[0])
        rows.append(row)
    return pd.DataFrame(rows)


def quality_warnings(prof, n_rows, missing_threshold=20.0, skew_threshold=1.0):
    """Turn a profile into human-readable warnings."""
    warnings = []
    for _, r in prof.iterrows():
        col = r["column"]
        if r["missing_pct"] >= missing_threshold:
            warnings.append(f"{col}: {r['missing_pct']}% missing values")
        if r["unique"] <= 1:
            warnings.append(f"{col}: constant column (no information)")
        if pd.notna(r["top_value"]) and n_rows and r["unique"] / n_rows > 0.9:
            warnings.append(f"{col}: looks like an identifier (almost all values unique)")
        if pd.notna(r["skew"]) and abs(r["skew"]) > skew_threshold:
            warnings.append(f"{col}: skewed distribution (skew {r['skew']})")
    return warnings
