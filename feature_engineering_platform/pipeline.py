"""FeaturePipeline: chains steps, remembers what it learned, and can be saved/loaded."""

import pickle

import pandas as pd

from steps import (Binner, CorrelationFilter, DateFeatures, Imputer, LogTransform,
                   OneHotEncoder, OutlierClipper, RatioFeatures, Scaler, VarianceFilter)


class FeaturePipeline:
    def __init__(self, steps):
        """steps: list of (name, step) tuples."""
        self.steps = list(steps)

    def fit_transform(self, X, y=None):
        """Fit every step in order on X (training data) and return the transformed X."""
        self.report_ = []
        for name, step in self.steps:
            before = set(X.columns)
            X = step.fit_transform(X, y)
            after = set(X.columns)
            self.report_.append({
                "step": name,
                "rows": X.shape[0],
                "columns_out": X.shape[1],
                "added": ", ".join(sorted(after - before)),
                "removed": ", ".join(sorted(before - after)),
            })
        self.feature_names_ = list(X.columns)
        return X

    def fit(self, X, y=None):
        self.fit_transform(X, y)
        return self

    def transform(self, X):
        """Apply the fitted steps. The output always has exactly the training columns."""
        for _, step in self.steps:
            X = step.transform(X)
        return X.reindex(columns=self.feature_names_, fill_value=0)

    def report(self):
        return pd.DataFrame(self.report_)

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            return pickle.load(f)


def default_pipeline(date_columns=(), ratio_specs=(), bin_columns=(), scaler="standard",
                     corr_threshold=0.95, max_categories=10):
    """A sensible general-purpose pipeline:

    dates -> outlier clipping -> imputation -> ratios -> log transform -> binning
          -> one-hot encoding -> drop constant -> drop correlated -> scaling
    """
    steps = []
    if date_columns:
        steps.append(("date_features", DateFeatures(date_columns)))
    steps.append(("outlier_clipper", OutlierClipper(factor=1.5)))
    steps.append(("imputer", Imputer()))
    if ratio_specs:
        steps.append(("ratio_features", RatioFeatures(ratio_specs)))
    steps.append(("log_transform", LogTransform()))
    if bin_columns:
        steps.append(("binner", Binner(bin_columns, n_bins=4)))
    steps.append(("one_hot", OneHotEncoder(max_categories=max_categories)))
    steps.append(("variance_filter", VarianceFilter()))
    steps.append(("correlation_filter", CorrelationFilter(corr_threshold)))
    steps.append(("scaler", Scaler(scaler)))
    return FeaturePipeline(steps)
