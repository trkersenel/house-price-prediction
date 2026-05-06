# House Price Prediction — End-to-End ML Pipeline

A deployment-ready machine learning pipeline that predicts Indian residential property prices (in lakhs of rupees) from listing attributes such as area, bedroom count, locality, and posting source. The repository covers the full lifecycle: EDA → preprocessing → feature engineering → model selection → hyperparameter tuning → REST API.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the model (writes model.joblib + feature_columns.json)
python train.py

# 3. Serve the REST API
python app.py
# → http://localhost:5000
```

## Project Layout

```
house-prediction-model/
├── train.csv                 # Raw training data (provided)
├── test.csv                  # Raw test data (provided)
├── sample_submission.csv     # Submission format (provided)
├── preprocessing.py          # Reusable feature engineering + custom transformer
├── train.py                  # End-to-end training pipeline
├── app.py                    # Flask REST API (loads model.joblib)
├── model.joblib              # Saved fitted scikit-learn pipeline (created by train.py)
├── feature_columns.json      # Column order + final metrics (created by train.py)
├── requirements.txt          # Pinned dependencies
├── EDA_report.md             # Exploratory data analysis findings
├── .gitignore
└── README.md                 # ← you are here
```

## 1. Exploratory Data Analysis

Full write-up in [`EDA_report.md`](./EDA_report.md). Headlines:

- 29,451 rows, 11 features, **zero missing values**.
- The target (`TARGET(PRICE_IN_LACS)`) is heavily right-skewed (median 62, max 30,000) — modelled in log-space.
- `SQUARE_FT` contains nonsense outliers (up to 254 million sqft) which are filtered out.
- `log(SQUARE_FT)` (+0.64) and `BHK_NO.` (+0.48) are the top correlations with log-price.
- The `ADDRESS` field carries 256 unique cities — the locality signal is huge but not yet on a numeric scale.

## 2. Preprocessing

Implemented in `train.py` and `preprocessing.py`:

- **Outlier filtering:** drop rows where `SQUARE_FT` is outside `[100, 20000]` or where price-per-sqft is outside `[500, 100000]`. Removes ~0.8% of rows.
- **Categorical encoding:**
  - `POSTED_BY`, `BHK_OR_RK` → one-hot (low cardinality).
  - `city` (extracted from `ADDRESS`) → smoothed target encoding (256 levels).
- **Scaling:** `StandardScaler` on numeric columns (helps Ridge; Random Forest is invariant but the cost is negligible).
- **Target transform:** train on `log1p(TARGET)`, exponentiate at predict time.

## 3. Feature Engineering

| Feature | Why it helps |
|---|---|
| `log_sqft` | Linearises the relationship between area and log-price; cuts the influence of remaining outliers. |
| `sqft_per_bhk` | Captures "spaciousness per room" — two 3BHK flats at 900 vs 1500 sqft are very different products. |
| `city` | Last segment of `ADDRESS`. Single most important locality signal — Bandra-Mumbai vs Hadapsar-Pune. |
| `is_metro` | Binary shortcut for the 13 major metros so even simple linear models pick up the metro premium. |

The full feature matrix is `numeric (8) + low-cardinality cats (2) + city (target-encoded, 1) = 14 columns` going into the model.

## 4. Model Building

- **Train/test split:** 80 / 20, stratification not needed for regression (`random_state=42`).
- **Models trained:**
  1. **Ridge regression** (alpha=1.0) — fast baseline, reveals which signals are linearly recoverable.
  2. **Random Forest** (120 trees, default depth) — captures nonlinear interactions between sqft × city × BHK.
- **Cross-validation:** 3-fold KFold on the training split, scored by negative RMSE on the log target.

## 5. Model Evaluation

| Model | CV log-RMSE | Test RMSE (lacs) | Test MAE (lacs) | Test R² |
|---|---|---|---|---|
| Ridge | 0.4291 ± 0.002 | 209.45 | 40.28 | −1.27 |
| Random Forest | **0.3150 ± 0.003** | **64.85** | **22.21** | **0.782** |
| **Tuned RF (final)** | — | **64.19** | 23.20 | **0.787** |

**Why Ridge has a negative R² despite a reasonable log-CV score:** a handful of high-priced test points produce huge errors after `expm1`, dragging R² below zero on the original scale. Tree models clip these naturally. The log-space CV score is the more comparable metric.

→ **Random Forest selected as the final model.**

## 6. Hyperparameter Tuning

`RandomizedSearchCV` (4 random samples × 3-fold CV = 12 fits) over:

- `n_estimators` ∈ {120, 200, 300}
- `max_depth` ∈ {None, 16, 24}
- `min_samples_split` ∈ {2, 5, 10}
- `min_samples_leaf` ∈ {1, 2, 4}
- `max_features` ∈ {sqrt, 0.5}

**Best params on this run:** `n_estimators=300, max_depth=24, min_samples_split=5, min_samples_leaf=4, max_features=0.5`.

The tuned model improves test R² from 0.782 → 0.787 — small but consistent.

## 7. Final Model Persistence

The fitted pipeline (preprocessor + tuned RF) is dumped to `model.joblib` with `joblib.dump`. The same `joblib.load` is used inside `app.py` at startup. Because the custom `TargetEncoder` lives in `preprocessing.py`, the API can load the artifact cleanly without re-defining it.

## 8. Deployment — Flask API

`app.py` exposes three endpoints:

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | Service banner + final metrics |
| GET | `/health` | Liveness probe |
| POST | `/predict` | Predict price for one property |
| POST | `/predict/batch` | Predict for many properties at once |

### Example request — single prediction

```bash
curl -X POST http://localhost:5000/predict \
     -H "Content-Type: application/json" \
     -d '{
           "POSTED_BY": "Owner",
           "UNDER_CONSTRUCTION": 0,
           "RERA": 0,
           "BHK_NO.": 2,
           "BHK_OR_RK": "BHK",
           "SQUARE_FT": 1300,
           "READY_TO_MOVE": 1,
           "RESALE": 1,
           "ADDRESS": "Ksfc Layout,Bangalore",
           "LONGITUDE": 12.96991,
           "LATITUDE": 77.59796
         }'
```

### Example response

```json
{
  "predicted_price_lacs": 85.9,
  "predicted_price_inr":  8590386.93,
  "currency": "INR",
  "model_version": "rf-v1"
}
```

### Example request — batch

```bash
curl -X POST http://localhost:5000/predict/batch \
     -H "Content-Type: application/json" \
     -d '{
           "records": [
             { "POSTED_BY": "Dealer", "UNDER_CONSTRUCTION": 1, "RERA": 1,
               "BHK_NO.": 3, "BHK_OR_RK": "BHK", "SQUARE_FT": 1500,
               "READY_TO_MOVE": 0, "RESALE": 0,
               "ADDRESS": "Bandra,Mumbai",
               "LONGITUDE": 19.0596, "LATITUDE": 72.8295 },
             { "POSTED_BY": "Owner", "UNDER_CONSTRUCTION": 0, "RERA": 0,
               "BHK_NO.": 1, "BHK_OR_RK": "BHK", "SQUARE_FT": 600,
               "READY_TO_MOVE": 1, "RESALE": 1,
               "ADDRESS": "Hadapsar,Pune",
               "LONGITUDE": 18.5089, "LATITUDE": 73.9259 }
           ]
         }'
```

### Example batch response

```json
{
  "count": 2,
  "currency": "INR",
  "model_version": "rf-v1",
  "predictions": [
    { "predicted_price_lacs": 436.36, "predicted_price_inr": 43636083.37 },
    { "predicted_price_lacs":  32.35, "predicted_price_inr":  3235314.15 }
  ]
}
```

### Error responses

The API validates that every required field is present and returns `400 Bad Request` with a JSON `{ "error": ... }` body when something is missing or malformed.

## Configuration

`app.py` honours two environment variables:

- `PORT` — default `5000`
- `FLASK_DEBUG` — set to `1` to enable Flask debug mode in development

```bash
PORT=8080 FLASK_DEBUG=1 python app.py
```

## Notes on Reproducibility

All randomness is seeded with `RANDOM_STATE=42`. Re-running `python train.py` on the same input produces the same model.joblib byte-for-byte (modulo joblib version differences).

## Possible Extensions

- Swap the Random Forest for **XGBoost / LightGBM** — typically wins another 1–3 R² points on tabular data.
- Add a **Dockerfile** + **gunicorn** for a production-grade serving stack.
- Use **MLflow** to track experiments; the metrics JSON written by `train.py` is a stub for that.
- Add **SHAP** explanations to the API response for interpretability.
