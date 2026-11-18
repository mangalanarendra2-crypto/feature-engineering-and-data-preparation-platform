# Feature Engineering and Data Preparation Platform

A small, reusable platform for turning raw tabular data into model-ready
features. It profiles the data, cleans it, engineers new features, encodes and
scales them, drops useless ones, and saves a fitted pipeline you can re-apply to
new data. Every step learns only from the **training** split, so there is no
data leakage.

Python 3.9+ with `pandas` and `numpy`:
```bash
pip install -r requirements.txt
```

## Quick start
```bash
python generate_sample_data.py       # writes data/customers.csv (messy sample data)
python run_pipeline.py               # runs the whole platform
python -m unittest test_platform.py -v
```
Outputs go to `output/`: `train_features.csv`, `test_features.csv`,
`pipeline.pkl`, `profile.csv`, `pipeline_report.csv`.

## Files
| File | Purpose |
|---|---|
| `data_profiler.py` | Per-column profile and data quality warnings |
| `steps.py` | The reusable fit/transform steps |
| `pipeline.py` | `FeaturePipeline` (chain, report, save/load) and `default_pipeline()` |
| `run_pipeline.py` | CLI: profile, clean, split, fit, transform, save |
| `generate_sample_data.py` | Creates a messy sample dataset |
| `test_platform.py` | Unit and end-to-end tests |

## Steps available
| Step | What it does |
|---|---|
| `DateFeatures` | Date column becomes year, month, day-of-week and days-ago |
| `OutlierClipper` | Clips values outside Q1 - 1.5*IQR / Q3 + 1.5*IQR |
| `Imputer` | Median (or mean) for numbers, "Unknown" for text, plus `_was_missing` flags |
| `RatioFeatures` | Safe ratio of two columns (divide by zero gives 0) |
| `LogTransform` | log1p on skewed non-negative columns, chosen automatically |
| `Binner` | Quantile buckets, e.g. `age_bin` |
| `OneHotEncoder` | One-hot for text columns, top N categories, unseen gives zeros |
| `VarianceFilter` | Drops constant columns |
| `CorrelationFilter` | Drops one of each pair of highly correlated columns |
| `Scaler` | Standard (z-score) or min-max scaling |

## Use it on your own data
```bash
python run_pipeline.py --input mydata.csv --target label --id-cols id \
    --date-cols created_at --bin-cols age --ratios "revenue/visits:revenue_per_visit"
```
Or from Python:
```python
from pipeline import default_pipeline
pipe = default_pipeline(date_columns=["created_at"], bin_columns=["age"])
X_train_f = pipe.fit_transform(X_train)
X_test_f = pipe.transform(X_test)      # same columns as training, always
pipe.save("pipeline.pkl")              # later: FeaturePipeline.load("pipeline.pkl")
```
Add your own step by subclassing `Step` and implementing `fit` and `transform`.

## Ideas to extend
- Target encoding and weighted-frequency encoding
- Rolling or group-by aggregate features (per-customer history)
- Feature selection by importance or mutual information
- Pipeline versioning and a small feature store
- Plug the output into scikit-learn models
