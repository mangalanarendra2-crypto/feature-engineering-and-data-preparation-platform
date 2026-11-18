import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import run_pipeline
from data_profiler import profile, quality_warnings
from generate_sample_data import make_dataframe
from pipeline import FeaturePipeline, default_pipeline
from steps import (Binner, CorrelationFilter, DateFeatures, Imputer, LogTransform,
                   OneHotEncoder, OutlierClipper, RatioFeatures, Scaler, VarianceFilter)


class StepTests(unittest.TestCase):
    def test_imputer_uses_training_median_and_adds_indicator(self):
        train = pd.DataFrame({"a": [1.0, 2.0, np.nan, 100.0], "c": ["x", None, "y", "x"]})
        imp = Imputer().fit(train)
        out = imp.transform(pd.DataFrame({"a": [np.nan], "c": [None]}))
        self.assertEqual(out.loc[0, "a"], 2.0)               # median of training data
        self.assertEqual(out.loc[0, "c"], "Unknown")
        self.assertEqual(out.loc[0, "a_was_missing"], 1)

    def test_outlier_clipper(self):
        train = pd.DataFrame({"a": list(range(1, 21)) + [1000]})
        clip = OutlierClipper().fit(train)
        out = clip.transform(train)
        self.assertLess(out["a"].max(), 1000)
        self.assertEqual(out["a"].iloc[0], 1)

    def test_date_features(self):
        X = pd.DataFrame({"d": ["2026-01-01", "2026-01-11", None]})
        out = DateFeatures(["d"]).fit_transform(X)
        self.assertNotIn("d", out.columns)
        self.assertEqual(out["d_days_ago"].iloc[0], 10)
        self.assertEqual(out["d_month"].iloc[1], 1)
        self.assertTrue(np.isnan(out["d_year"].iloc[2]))

    def test_ratio_handles_zero_denominator(self):
        X = pd.DataFrame({"a": [10.0, 5.0], "b": [2.0, 0.0]})
        out = RatioFeatures([("a", "b", "r")]).transform(X)
        self.assertEqual(list(out["r"]), [5.0, 0.0])

    def test_log_transform_picks_skewed_columns_only(self):
        rng = np.random.default_rng(0)
        X = pd.DataFrame({"skewed": rng.lognormal(0, 1.5, 300), "normal": rng.normal(50, 5, 300)})
        step = LogTransform().fit(X)
        self.assertEqual(step.columns_, ["skewed"])

    def test_binner_uses_training_edges(self):
        train = pd.DataFrame({"x": np.arange(100.0)})
        b = Binner(["x"], n_bins=4).fit(train)
        out = b.transform(pd.DataFrame({"x": [0.0, 30.0, 60.0, 99.0, 1e9, np.nan]}))
        self.assertEqual(list(out["x_bin"]), [0, 1, 2, 3, 3, -1])

    def test_one_hot_handles_unseen_categories(self):
        enc = OneHotEncoder().fit(pd.DataFrame({"c": ["a", "b", "a"]}))
        out = enc.transform(pd.DataFrame({"c": ["a", "zzz"]}))
        self.assertEqual(list(out.columns), ["c_a", "c_b"])
        self.assertEqual(out.loc[1].sum(), 0)                # unseen -> all zeros

    def test_scaler_standard_and_minmax(self):
        X = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
        z = Scaler("standard").fit_transform(X)["x"]
        self.assertAlmostEqual(z.mean(), 0)
        self.assertAlmostEqual(z.std(ddof=0), 1)
        m = Scaler("minmax").fit_transform(X)["x"]
        self.assertEqual((m.min(), m.max()), (0.0, 1.0))

    def test_scaler_applies_training_statistics_to_new_data(self):
        s = Scaler("minmax").fit(pd.DataFrame({"x": [0.0, 10.0, 5.0]}))
        self.assertEqual(s.transform(pd.DataFrame({"x": [20.0]}))["x"].iloc[0], 2.0)

    def test_variance_and_correlation_filters(self):
        rng = np.random.default_rng(1)
        a = rng.normal(size=100)
        X = pd.DataFrame({"a": a, "a_copy": a * 3 + 1, "const": 7.0, "other": rng.normal(size=100)})
        X = VarianceFilter().fit_transform(X)
        self.assertNotIn("const", X.columns)
        X = CorrelationFilter(0.95).fit_transform(X)
        self.assertEqual(sorted(X.columns), ["a", "other"])


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = make_dataframe().drop_duplicates().reset_index(drop=True)
        train, test = run_pipeline.train_test_split(cls.df, 0.2, 42)
        drop = ["customer_id", "churned"]
        cls.X_train, cls.X_test = train.drop(columns=drop), test.drop(columns=drop)
        cls.pipe = default_pipeline(
            date_columns=["signup_date", "last_purchase_date"],
            ratio_specs=[("total_spend", "num_orders", "avg_order_value")],
            bin_columns=["age"])
        cls.F_train = cls.pipe.fit_transform(cls.X_train)
        cls.F_test = cls.pipe.transform(cls.X_test)

    def test_output_is_numeric_and_complete(self):
        for F in (self.F_train, self.F_test):
            self.assertEqual(int(F.isna().sum().sum()), 0)
            self.assertTrue(all(pd.api.types.is_numeric_dtype(F[c]) for c in F.columns))

    def test_train_and_test_have_identical_columns(self):
        self.assertEqual(list(self.F_train.columns), list(self.F_test.columns))

    def test_useless_features_are_removed(self):
        self.assertNotIn("source_system_web", self.F_train.columns)    # constant
        self.assertNotIn("total_spend_cents", self.F_train.columns)    # duplicate of total_spend
        self.assertNotIn("signup_date", self.F_train.columns)          # replaced by date features

    def test_engineered_features_exist(self):
        for col in ("avg_order_value", "age_bin", "signup_date_days_ago", "age_was_missing"):
            self.assertIn(col, self.F_train.columns)

    def test_no_leakage_scaler_uses_training_statistics(self):
        # Training columns are exactly standardised; the test split is only approximately so.
        col = "num_orders"
        self.assertAlmostEqual(self.F_train[col].mean(), 0, places=6)
        self.assertNotAlmostEqual(self.F_test[col].mean(), 0, places=6)

    def test_save_and_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "p.pkl"
            self.pipe.save(path)
            loaded = FeaturePipeline.load(path)
            pd.testing.assert_frame_equal(loaded.transform(self.X_test), self.F_test)

    def test_report_tracks_every_step(self):
        self.assertEqual(len(self.pipe.report()), len(self.pipe.steps))


class ProfilerTests(unittest.TestCase):
    def test_profile_and_warnings(self):
        df = pd.DataFrame({"id": [f"u{i}" for i in range(10)],
                           "const": 1,
                           "gap": [1.0] * 5 + [np.nan] * 5})
        prof = profile(df)
        self.assertEqual(prof.set_index("column").loc["gap", "missing"], 5)
        text = " ".join(quality_warnings(prof, len(df)))
        self.assertIn("id: looks like an identifier", text)
        self.assertIn("const: constant column", text)
        self.assertIn("gap: 50.0% missing", text)


class EndToEndTests(unittest.TestCase):
    def test_run_writes_all_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "customers.csv"
            make_dataframe().to_csv(csv_path, index=False)
            out = Path(tmp) / "out"
            result = run_pipeline.run(csv_path, out)
            for name in ("train_features.csv", "test_features.csv", "pipeline.pkl",
                         "profile.csv", "pipeline_report.csv"):
                self.assertTrue((out / name).exists(), name)
            self.assertEqual(result["duplicates_removed"], 10)
            self.assertEqual(result["train_rows"] + result["test_rows"], 500)
            train = pd.read_csv(out / "train_features.csv")
            test = pd.read_csv(out / "test_features.csv")
            self.assertEqual(list(train.columns), list(test.columns))
            self.assertIn("churned", train.columns)
            self.assertEqual(int(train.isna().sum().sum()), 0)


if __name__ == "__main__":
    unittest.main()
