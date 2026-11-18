"""
Reusable feature engineering / data preparation steps.

Every step follows the same small contract (like scikit-learn transformers):

    step.fit(X, y=None)  -> learns parameters from TRAINING data only
    step.transform(X)    -> applies them to any data, returns a new DataFrame

Learning parameters only from the training split and re-using them on the test
split prevents data leakage.
"""

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype


def numeric_columns(X, skip_binary=False):
    cols = [c for c in X.columns if is_numeric_dtype(X[c])]
    if skip_binary:
        cols = [c for c in cols if X[c].nunique(dropna=True) > 2]
    return cols


def categorical_columns(X):
    return [c for c in X.columns if not is_numeric_dtype(X[c])]


class Step:
    """Base class for all steps."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        raise NotImplementedError

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------
class OutlierClipper(Step):
    """Clip numeric values to [Q1 - k*IQR, Q3 + k*IQR] learned on the training data."""

    def __init__(self, factor=1.5, columns=None):
        self.factor = factor
        self.columns = columns

    def fit(self, X, y=None):
        cols = self.columns if self.columns is not None else numeric_columns(X, skip_binary=True)
        self.bounds_ = {}
        for c in cols:
            q1, q3 = X[c].quantile([0.25, 0.75])
            iqr = q3 - q1
            if pd.isna(iqr) or iqr == 0:
                continue
            self.bounds_[c] = (float(q1 - self.factor * iqr), float(q3 + self.factor * iqr))
        return self

    def transform(self, X):
        X = X.copy()
        for c, (lo, hi) in self.bounds_.items():
            if c in X:
                X[c] = X[c].clip(lo, hi)
        return X


class Imputer(Step):
    """Fill missing values: median (or mean) for numbers, a constant label for text.
    Optionally adds a '<col>_was_missing' flag for columns that had gaps in training."""

    def __init__(self, numeric_strategy="median", categorical_fill="Unknown", add_indicator=True):
        if numeric_strategy not in ("median", "mean"):
            raise ValueError("numeric_strategy must be 'median' or 'mean'")
        self.numeric_strategy = numeric_strategy
        self.categorical_fill = categorical_fill
        self.add_indicator = add_indicator

    def fit(self, X, y=None):
        self.fill_values_, self.indicator_cols_ = {}, []
        for c in X.columns:
            if is_numeric_dtype(X[c]):
                value = X[c].median() if self.numeric_strategy == "median" else X[c].mean()
                self.fill_values_[c] = 0.0 if pd.isna(value) else float(value)
            else:
                self.fill_values_[c] = self.categorical_fill
            if self.add_indicator and X[c].isna().any():
                self.indicator_cols_.append(c)
        return self

    def transform(self, X):
        X = X.copy()
        for c in self.indicator_cols_:
            if c in X:
                X[f"{c}_was_missing"] = X[c].isna().astype(int)
        for c, value in self.fill_values_.items():
            if c in X:
                X[c] = X[c].fillna(value)
        return X


# ---------------------------------------------------------------------------
# Feature creation
# ---------------------------------------------------------------------------
class DateFeatures(Step):
    """Replace date columns with year, month, day-of-week and 'days ago' features.
    'days ago' is measured from the latest date seen in the training data."""

    def __init__(self, columns):
        self.columns = list(columns)

    def _parse(self, X, c):
        return pd.to_datetime(X[c], errors="coerce")

    def fit(self, X, y=None):
        latest = [self._parse(X, c).max() for c in self.columns if c in X]
        self.reference_ = max(latest) if latest else pd.Timestamp("2000-01-01")
        return self

    def transform(self, X):
        X = X.copy()
        for c in self.columns:
            if c not in X:
                continue
            dt = self._parse(X, c)
            X[f"{c}_year"] = dt.dt.year
            X[f"{c}_month"] = dt.dt.month
            X[f"{c}_dayofweek"] = dt.dt.dayofweek
            X[f"{c}_days_ago"] = (self.reference_ - dt).dt.days
            X = X.drop(columns=[c])
        return X


class RatioFeatures(Step):
    """Create ratio features. specs = [(numerator, denominator, new_name), ...].
    Division by zero gives 0."""

    def __init__(self, specs):
        self.specs = list(specs)

    def transform(self, X):
        X = X.copy()
        for num, den, name in self.specs:
            X[name] = (X[num] / X[den].replace(0, np.nan)).fillna(0.0)
        return X


class LogTransform(Step):
    """Apply log1p to skewed, non-negative columns to reduce skew.
    If columns is None, the columns are chosen automatically on the training data."""

    def __init__(self, columns=None, skew_threshold=1.0):
        self.columns = columns
        self.skew_threshold = skew_threshold

    def fit(self, X, y=None):
        if self.columns is not None:
            self.columns_ = list(self.columns)
        else:
            self.columns_ = [
                c for c in numeric_columns(X, skip_binary=True)
                if X[c].min() >= 0 and abs(X[c].skew()) > self.skew_threshold
            ]
        return self

    def transform(self, X):
        X = X.copy()
        for c in self.columns_:
            if c in X:
                X[c] = np.log1p(X[c].clip(lower=0))
        return X


class Binner(Step):
    """Add '<col>_bin': the quantile bucket (0..n_bins-1) of a numeric column.
    Bucket edges are learned on the training data."""

    def __init__(self, columns, n_bins=4):
        self.columns = list(columns)
        self.n_bins = n_bins

    def fit(self, X, y=None):
        qs = np.linspace(0, 1, self.n_bins + 1)[1:-1]
        self.edges_ = {c: np.unique(X[c].quantile(qs).to_numpy()) for c in self.columns if c in X}
        return self

    def transform(self, X):
        X = X.copy()
        for c, edges in self.edges_.items():
            values = X[c].to_numpy(dtype=float)
            X[f"{c}_bin"] = np.where(np.isnan(values), -1, np.searchsorted(edges, values, side="right"))
        return X


# ---------------------------------------------------------------------------
# Encoding and scaling
# ---------------------------------------------------------------------------
class OneHotEncoder(Step):
    """One-hot encode text columns. Keeps the most frequent `max_categories`
    values seen in training; unseen or rare values become all zeros."""

    def __init__(self, columns=None, max_categories=10):
        self.columns = columns
        self.max_categories = max_categories

    def fit(self, X, y=None):
        cols = self.columns if self.columns is not None else categorical_columns(X)
        self.categories_ = {}
        for c in cols:
            top = X[c].value_counts().head(self.max_categories).index.tolist()
            self.categories_[c] = sorted(top, key=str)
        return self

    def transform(self, X):
        X = X.copy()
        new_cols = {}
        for c, cats in self.categories_.items():
            if c not in X:
                continue
            for cat in cats:
                new_cols[f"{c}_{cat}"] = (X[c] == cat).astype(int)
            X = X.drop(columns=[c])
        if new_cols:
            X = pd.concat([X, pd.DataFrame(new_cols, index=X.index)], axis=1)
        return X


class Scaler(Step):
    """Scale numeric (non-binary) columns: 'standard' (z-score) or 'minmax' (0..1)."""

    def __init__(self, method="standard", columns=None):
        if method not in ("standard", "minmax"):
            raise ValueError("method must be 'standard' or 'minmax'")
        self.method = method
        self.columns = columns

    def fit(self, X, y=None):
        cols = self.columns if self.columns is not None else numeric_columns(X, skip_binary=True)
        self.stats_ = {}
        for c in cols:
            if self.method == "standard":
                self.stats_[c] = (float(X[c].mean()), float(X[c].std(ddof=0)) or 1.0)
            else:
                lo, hi = float(X[c].min()), float(X[c].max())
                self.stats_[c] = (lo, (hi - lo) or 1.0)
        return self

    def transform(self, X):
        X = X.copy()
        for c, (center, scale) in self.stats_.items():
            if c in X:
                X[c] = (X[c] - center) / scale
        return X


# ---------------------------------------------------------------------------
# Feature selection
# ---------------------------------------------------------------------------
class VarianceFilter(Step):
    """Drop numeric columns whose variance is <= threshold (e.g. constant columns)."""

    def __init__(self, threshold=0.0):
        self.threshold = threshold

    def fit(self, X, y=None):
        self.drop_cols_ = [c for c in numeric_columns(X)
                           if not X[c].astype(float).var(ddof=0) > self.threshold]
        return self

    def transform(self, X):
        return X.drop(columns=[c for c in self.drop_cols_ if c in X])


class CorrelationFilter(Step):
    """Drop one column from every pair whose absolute correlation exceeds threshold."""

    def __init__(self, threshold=0.95):
        self.threshold = threshold

    def fit(self, X, y=None):
        corr = X[numeric_columns(X)].astype(float).corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
        self.drop_cols_ = [c for c in upper.columns if (upper[c] > self.threshold).any()]
        return self

    def transform(self, X):
        return X.drop(columns=[c for c in self.drop_cols_ if c in X])
