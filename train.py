"""
train.py
========
End-to-end training pipeline for the house-price prediction model.

Pipeline overview
-----------------
1.  Load raw CSV (train.csv).
2.  Clean obvious outliers (square-footage and price-per-sqft).
3.  Engineer features (log_sqft, sqft_per_bhk, city, is_metro).
4.  Build a scikit-learn ColumnTransformer that:
        - one-hot encodes low-cardinality categoricals
        - target-encodes the high-cardinality CITY column
        - standard-scales numeric features
5.  Train Linear Regression (Ridge) + Random Forest.
6.  5-fold cross-validation on each model (negative RMSE).
7.  Pick the better model on the held-out test set, then run a
    RandomizedSearchCV hyperparameter sweep on it.
8.  Persist the final fitted pipeline to `model.joblib` and the list
    of feature columns to `feature_columns.json` for the API layer.

Run
---
    python train.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from preprocessing import TargetEncoder, engineer_features  # noqa: F401

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
DATA_PATH = HERE / "train.csv"
MODEL_PATH = HERE / "model.joblib"
META_PATH = HERE / "feature_columns.json"

RANDOM_STATE = 42
TARGET_COL = "TARGET(PRICE_IN_LACS)"


# ---------------------------------------------------------------------------
# 1. Load + clean
# ---------------------------------------------------------------------------
def load_and_clean(path: Path) -> pd.DataFrame:
    """Load the CSV and drop rows with absurd SQUARE_FT / price-per-sqft.

    The dataset has a few entries where SQUARE_FT is in the hundreds of
    millions, or where price/sqft is below ₹500 or above ₹100k — these
    are clearly data-entry errors and bias every regression model.
    """
    df = pd.read_csv(path)

    # Strip whitespace on string columns
    for c in df.select_dtypes(include="object").columns:
        df[c] = df[c].astype(str).str.strip()

    # Drop impossible square footage (< 100 sqft or > 20k sqft for a flat)
    df = df[(df["SQUARE_FT"] >= 100) & (df["SQUARE_FT"] <= 20000)]

    # Drop price-per-sqft outliers
    pps = df[TARGET_COL] * 100_000 / df["SQUARE_FT"]
    df = df[(pps >= 500) & (pps <= 100_000)]

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. Feature engineering   (TargetEncoder + engineer_features live in
#    preprocessing.py — see import above. They are kept in their own module
#    so joblib can correctly resolve the custom transformer at serve time.)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 3. Build preprocessing + model pipeline
# ---------------------------------------------------------------------------
NUMERIC_FEATURES = [
    "UNDER_CONSTRUCTION", "RERA", "BHK_NO.", "SQUARE_FT", "READY_TO_MOVE",
    "RESALE", "LONGITUDE", "LATITUDE", "log_sqft", "sqft_per_bhk", "is_metro",
]
LOW_CARD_CAT = ["POSTED_BY", "BHK_OR_RK"]
HIGH_CARD_CAT = ["city"]

FEATURE_COLUMNS = NUMERIC_FEATURES + LOW_CARD_CAT + HIGH_CARD_CAT


def build_preprocessor() -> ColumnTransformer:
    """ColumnTransformer that scales numerics, one-hot small cats, target-encodes city."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("low_cat", OneHotEncoder(handle_unknown="ignore"), LOW_CARD_CAT),
            ("city", TargetEncoder(smoothing=10.0), HIGH_CARD_CAT),
        ],
        remainder="drop",
    )


# ---------------------------------------------------------------------------
# 4. Train + evaluate
# ---------------------------------------------------------------------------
def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute RMSE / MAE / R²."""
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE":  float(mean_absolute_error(y_true, y_pred)),
        "R2":   float(r2_score(y_true, y_pred)),
    }


def cross_val_rmse(model: Pipeline, X, y, n_splits: int = 3) -> Tuple[float, float]:
    """K-fold CV on log-target, returns (mean RMSE, std) in log scale."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    # Use n_jobs=1 here: the inner estimator already uses n_jobs=-1 for trees,
    # nested parallelism with loky backend slows things down on small CPUs.
    scores = cross_val_score(
        model, X, y,
        scoring="neg_root_mean_squared_error",
        cv=kf, n_jobs=1,
    )
    rmse = -scores
    return float(rmse.mean()), float(rmse.std())


def main() -> None:
    print(f"Loading data from {DATA_PATH} ...")
    df = load_and_clean(DATA_PATH)
    df = engineer_features(df)
    print(f"Clean dataset: {len(df):,} rows")

    X = df[FEATURE_COLUMNS]
    # Train on log(price) — target is log-normal-ish, this stabilises errors
    y_log = np.log1p(df[TARGET_COL])

    X_train, X_test, y_train_log, y_test_log = train_test_split(
        X, y_log, test_size=0.2, random_state=RANDOM_STATE
    )
    y_test = np.expm1(y_test_log)

    # ---------------------- Linear Regression (Ridge) -----------------------
    ridge_pipe = Pipeline([
        ("pre", build_preprocessor()),
        ("model", Ridge(alpha=1.0, random_state=RANDOM_STATE)),
    ])
    rmse_cv, rmse_std = cross_val_rmse(ridge_pipe, X_train, y_train_log)
    ridge_pipe.fit(X_train, y_train_log)
    ridge_pred = np.expm1(ridge_pipe.predict(X_test))
    ridge_metrics = evaluate(y_test, ridge_pred)
    print(f"\n[Ridge]   CV log-RMSE: {rmse_cv:.4f} ± {rmse_std:.4f}")
    print(f"          Test  RMSE={ridge_metrics['RMSE']:.2f}  "
          f"MAE={ridge_metrics['MAE']:.2f}  R²={ridge_metrics['R2']:.4f}")

    # ---------------------- Random Forest -----------------------------------
    rf_pipe = Pipeline([
        ("pre", build_preprocessor()),
        ("model", RandomForestRegressor(
            n_estimators=120, max_depth=None,
            n_jobs=-1, random_state=RANDOM_STATE,
        )),
    ])
    rmse_cv, rmse_std = cross_val_rmse(rf_pipe, X_train, y_train_log)
    rf_pipe.fit(X_train, y_train_log)
    rf_pred = np.expm1(rf_pipe.predict(X_test))
    rf_metrics = evaluate(y_test, rf_pred)
    print(f"\n[RF]      CV log-RMSE: {rmse_cv:.4f} ± {rmse_std:.4f}")
    print(f"          Test  RMSE={rf_metrics['RMSE']:.2f}  "
          f"MAE={rf_metrics['MAE']:.2f}  R²={rf_metrics['R2']:.4f}")

    # ---------------------- Pick winner + tune ------------------------------
    if rf_metrics["R2"] >= ridge_metrics["R2"]:
        print("\nRandom Forest wins — running RandomizedSearchCV ...")
        param_dist = {
            "model__n_estimators":      [120, 200, 300],
            "model__max_depth":         [None, 16, 24],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf":  [1, 2, 4],
            "model__max_features":      ["sqrt", 0.5],
        }
        search = RandomizedSearchCV(
            rf_pipe, param_dist, n_iter=4, cv=3,
            scoring="neg_root_mean_squared_error",
            n_jobs=1, random_state=RANDOM_STATE, verbose=0,
        )
        search.fit(X_train, y_train_log)
        best = search.best_estimator_
        print(f"Best params: {search.best_params_}")
    else:
        print("\nRidge wins — sweeping alpha ...")
        param_dist = {"model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]}
        search = RandomizedSearchCV(
            ridge_pipe, param_dist, n_iter=5, cv=5,
            scoring="neg_root_mean_squared_error",
            n_jobs=1, random_state=RANDOM_STATE,
        )
        search.fit(X_train, y_train_log)
        best = search.best_estimator_
        print(f"Best params: {search.best_params_}")

    final_pred = np.expm1(best.predict(X_test))
    final_metrics = evaluate(y_test, final_pred)
    print(f"\n[FINAL]   Test  RMSE={final_metrics['RMSE']:.2f}  "
          f"MAE={final_metrics['MAE']:.2f}  R²={final_metrics['R2']:.4f}")

    # ---------------------- Persist artifacts -------------------------------
    joblib.dump(best, MODEL_PATH)
    print(f"\nSaved fitted pipeline → {MODEL_PATH}")

    meta = {
        "feature_columns": FEATURE_COLUMNS,
        "target_log_transformed": True,
        "metrics": {
            "ridge": ridge_metrics,
            "random_forest": rf_metrics,
            "final": final_metrics,
        },
    }
    META_PATH.write_text(json.dumps(meta, indent=2))
    print(f"Saved metadata        → {META_PATH}")


if __name__ == "__main__":
    main()
