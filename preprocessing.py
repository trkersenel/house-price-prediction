"""
preprocessing.py
================
Reusable feature-engineering helpers + the custom TargetEncoder.

Lives in its own module so that:
  * train.py (run as __main__) and app.py (Flask) both import the
    TargetEncoder class from the *same* module path, which is required
    for joblib to round-trip the saved pipeline correctly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# The 13 cities Indian buyers commonly call "metros" — used as a coarse
# binary feature so even simple models pick up the metro premium.
METRO_CITIES = {
    "Bangalore", "Mumbai", "Delhi", "New Delhi", "Chennai", "Kolkata",
    "Hyderabad", "Pune", "Ahmedabad", "Gurgaon", "Noida", "Faridabad",
    "Ghaziabad",
}


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived columns that help the model.

    log_sqft        — log-scale SQUARE_FT (target is heavy-tailed, log helps
                      both linear and tree models converge).
    sqft_per_bhk    — average area per bedroom; captures "spaciousness"
                      independent of total area or BHK count.
    city            — last segment of ADDRESS ("Locality, City"); a strong
                      proxy for property prices across India.
    is_metro        — 1 if city is one of the major metros (binary shortcut
                      so even simple models get a metro premium).
    """
    df = df.copy()
    df["log_sqft"] = np.log1p(df["SQUARE_FT"])
    df["sqft_per_bhk"] = df["SQUARE_FT"] / df["BHK_NO."].clip(lower=1)
    df["city"] = df["ADDRESS"].astype(str).str.split(",").str[-1].str.strip()
    df["is_metro"] = df["city"].isin(METRO_CITIES).astype(int)
    return df


class TargetEncoder:
    """Smoothed-mean target encoder for the high-cardinality `city` column.

    Implemented as a sklearn-compatible transformer so it slots into the
    ColumnTransformer cleanly. Rare cities are pulled towards the global
    mean to avoid overfitting to noisy per-city averages.

    Parameters
    ----------
    smoothing : float
        Pseudo-count added to every city group. Higher values pull rare
        cities harder towards the overall mean.
    """

    def __init__(self, smoothing: float = 10.0):
        self.smoothing = smoothing
        self.global_mean_: float = 0.0
        self.mapping_: dict = {}

    def fit(self, X, y):
        s = pd.Series(np.asarray(X).ravel())
        y = pd.Series(np.asarray(y))
        self.global_mean_ = float(y.mean())
        agg = y.groupby(s).agg(["mean", "count"])
        smoothed = (
            agg["mean"] * agg["count"] + self.global_mean_ * self.smoothing
        ) / (agg["count"] + self.smoothing)
        self.mapping_ = smoothed.to_dict()
        return self

    def transform(self, X):
        s = pd.Series(np.asarray(X).ravel())
        out = s.map(self.mapping_).fillna(self.global_mean_)
        return out.to_numpy().reshape(-1, 1)

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)

    # sklearn estimator API
    def get_params(self, deep=True):
        return {"smoothing": self.smoothing}

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self
