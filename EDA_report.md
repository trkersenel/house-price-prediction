# Exploratory Data Analysis Report

## Dataset Overview

- **Rows:** 29,451 (training) / 68,720 (test)
- **Columns:** 11 features + 1 target
- **Target:** `TARGET(PRICE_IN_LACS)` — house price in Indian lakhs (1 lakh = 100,000 ₹)
- **Missing values:** **None** in any column. The dataset is already clean on that axis.

## Feature Inventory

| Column | Type | Description |
|---|---|---|
| `POSTED_BY` | categorical | Listing agent: Owner / Dealer / Builder |
| `UNDER_CONSTRUCTION` | binary | 1 = under construction, 0 = not |
| `RERA` | binary | 1 = registered with the Real Estate Regulatory Authority |
| `BHK_NO.` | int | Number of bedrooms |
| `BHK_OR_RK` | categorical | "BHK" (Bedroom-Hall-Kitchen) vs "RK" (Room-Kitchen) — 99.9% BHK |
| `SQUARE_FT` | float | Built-up area in square feet |
| `READY_TO_MOVE` | binary | 1 = move-in ready |
| `RESALE` | binary | 1 = resale property (vs new) |
| `ADDRESS` | text | "Locality, City" |
| `LONGITUDE`, `LATITUDE` | float | Geo coordinates |

## Summary Statistics (key numerics)

| Stat | SQUARE_FT | BHK_NO. | TARGET (lacs) |
|---|---|---|---|
| count | 29,451 | 29,451 | 29,451 |
| mean | 19,802 | 2.4 | 142.9 |
| median | 1,175 | 2 | 62.0 |
| 95th pct | 2,619 | 4 | 300.0 |
| 99th pct | 5,463 | 5 | 1,045.0 |
| max | **254,545,500** | 20 | **30,000** |

## Outliers

The mean of `SQUARE_FT` (~20k) is two orders of magnitude above the median (~1,175). Extreme values are clearly data-entry errors — a max of **254 million sqft** is impossible for a residential property. Similar pattern on the target (max ₹3,000 cr / 30,000 lacs).

**Rule applied in `train.py`:**

- Drop rows where `SQUARE_FT < 100` or `> 20,000`.
- Drop rows where `price_per_sqft` falls outside ₹500–₹100,000.

This removes ~230 rows (0.8%) and dramatically stabilises every model that follows.

## Key Drivers of Price

Pearson correlation with **log(price)** (more meaningful than raw price because of the long tail):

| Feature | Correlation with log(price) |
|---|---|
| **log(SQUARE_FT)** | **+0.64** |
| **BHK_NO.** | **+0.48** |
| RERA | +0.14 |
| LONGITUDE | −0.13 |
| RESALE | −0.10 |
| LATITUDE | −0.06 |
| UNDER_CONSTRUCTION | +0.05 |

The two dominant signals are **size** and **bedroom count**, which is exactly what intuition would predict. The geographic coordinates plus the city extracted from `ADDRESS` carry the rest of the variance — a property in Mumbai-Bandra commands a price unrelated to its sqft. That's why we add a target-encoded `city` feature in `train.py`.

## Cardinality Note

`ADDRESS` parses out into ~256 unique cities. The top cities (Bangalore, Mumbai, Pune, Noida, Kolkata) account for ~70% of all listings. This high cardinality is why we use **smoothed target encoding** rather than one-hot.

## Takeaways That Shaped the Pipeline

1. **No missing values → no imputation step needed.** Saved complexity.
2. **Long-tailed target** → train on `log1p(TARGET)`, exponentiate predictions.
3. **Outliers** in `SQUARE_FT` → hard filter (above) before any model sees the data.
4. **Big dispersion across cities** → engineer a `city` column and target-encode it.
5. **Strong nonlinear interactions** (sqft × city × BHK) → tree-based models (Random Forest) outperform Ridge by a wide margin.
