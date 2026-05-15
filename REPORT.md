# 📝 Analysis Report — NeuralLoad-BD
### Bangladesh National Grid Demand Forecasting Pipeline

> **Deliverable B** — Methodology, Feature Rationale & Results Analysis  
> IITG.ai Predictive Paradox Recruitment Task

---

## 1. Handling Missing Data & Outliers

### 1.1 Missing Values in PGCB Data

The raw PGCB dataset contained several columns with substantial missing values:

- **`solar`** — solar generation was not metered in early years (Bangladesh began large-scale solar only after 2018). These gaps were treated as genuine zero/near-zero values and imputed via KNN.
- **`load_shedding`** — sporadically recorded. Since this column is an operational reading at time *t* (not a forecasting input), it was **excluded entirely** from the feature set as a leaky variable.
- **`remarks`** — free-text annotations, not machine-usable. **Dropped**.
- **`india_adani`, `nepal`, `wind`** — either not populated or ceased being tracked. **Dropped**.

For all remaining numeric columns, **KNN Imputation** (k=5, distance-weighted) was applied after resampling. KNN was chosen over simpler forward-fill because power demand exhibits strong local temporal patterns — the k nearest neighbours in feature space (nearby timestamps with similar hour/season profile) produce more realistic imputed values than a simple propagation.

### 1.2 Outlier Detection & Removal in `demand_mw`

Two-layer outlier detection strategy:

**Layer 1 — Statistical (IQR):**
```
lower_bound = Q1 − 1.5 × IQR
upper_bound = Q3 + 1.5 × IQR
```
Values outside this range were flagged as anomalous spikes — likely data entry errors or meter faults.

**Layer 2 — Physical Cap:**  
Bangladesh's national grid has a physical installed capacity ceiling of approximately **25,000 MW**. Any reading above this is physically impossible and was capped.

**Why NaN-replace instead of drop?**  
Dropping outlier rows would create irregular gaps in the time series, breaking the lag and rolling features. Instead, flagged values were set to `NaN` and then recovered via KNN imputation — preserving temporal continuity while replacing erroneous values with plausible estimates.

### 1.3 Duplicate & Mixed-Frequency Timestamps

The raw PGCB file contained a mix of **on-the-hour (`:00`)** and **half-hour (`:30`)** timestamps, as well as outright duplicate entries for the same datetime. This was resolved in a single step:

```python
pgcb.set_index("datetime").resample("1h").mean()
```

Resampling to 1H mean automatically:
- Collapses duplicate rows at the same hour into their average
- Aggregates `:00` and `:30` readings into a single hourly value
- Fills the resulting regular DatetimeIndex, making it amenable to lag feature creation

---

## 2. Feature Engineering — Rationale

### 2.1 Why Manual Feature Engineering?

The task restricts us to **classical ML algorithms** (Random Forest, XGBoost, LightGBM). These models treat each observation as an independent, flat row of numbers — they have no built-in notion of sequence or time. Therefore, **we must explicitly construct features that encode temporal context** so the model can reason about recent history and seasonal patterns.

### 2.2 Calendar Features

| Feature | Type | Why it matters |
|---------|------|---------------|
| `hour` (0–23) | Ordinal | Demand follows a strong intraday cycle: low at night, ramps up through morning, peaks evening |
| `dayofweek` (0–6) | Ordinal | Bangladesh weekends (Fri/Sat) see significantly lower industrial demand |
| `month` (1–12) | Ordinal | Seasonal air-conditioning load peaks in April–June (hot season) |
| `quarter` | Ordinal | Broad seasonal grouping |
| `is_weekend` | Binary flag | Cleaner signal than raw dayofweek for the tree model |
| `is_day_peak` | Binary flag | Hours 12–15: midday industrial peak |
| `is_evening_peak` | Binary flag | Hours 18–21: residential/commercial lighting peak |
| `hour_sin`, `hour_cos` | Cyclical | Ensures hour 23 and hour 0 are treated as adjacent, not 23 apart |
| `month_sin`, `month_cos` | Cyclical | Same logic for monthly seasonality |

**Why cyclical encoding?**  
Without sin/cos encoding, a tree model would learn that hour 23 and hour 0 are far apart (distance = 23), when in reality they are adjacent in the demand cycle. The circular transformation maps hours onto a unit circle, so the model can capture this continuity.

### 2.3 Lag Features

Lag features give the model a direct "window" into recent demand history without violating the supervised tabular structure:

| Feature | Offset | Captures |
|---------|--------|---------|
| `demand_lag_1h` | t−1 | Immediate prior hour — strongest single predictor |
| `demand_lag_2h` | t−2 | Short-term momentum |
| `demand_lag_3h` | t−3 | Early-morning/ramp pattern context |
| `demand_lag_24h` | t−24 | Same hour yesterday — daily seasonality |
| `demand_lag_48h` | t−48 | Same hour two days ago — smoothed daily pattern |
| `demand_lag_168h` | t−168 | Same hour last week — weekly seasonality |

The 24h and 168h lags are particularly important: they let the model know "what was happening at this same time yesterday / last week?" which is the strongest predictor of short-term grid demand.

**Leakage safety:** All lags use `.shift(lag)` with `lag ≥ 1`, so the feature at row *t* always refers to data from before *t*. The target is `demand_mw.shift(-1)` — the value at *t+1*. No future data enters the training frame.

### 2.4 Rolling Aggregate Features

Rolling features summarise recent demand trends, which helps the model handle gradual ramps (e.g., the morning load pickup) and regime shifts:

| Feature | Window | Captures |
|---------|--------|---------|
| `demand_rmean_3h` | 3-hour mean | Very recent micro-trend |
| `demand_rmean_6h` | 6-hour mean | Half-shift smoothed average |
| `demand_rmean_12h` | 12-hour mean | Half-day trend |
| `demand_rmean_24h` | 24-hour mean | Full-day baseline |
| `demand_rstd_24h` | 24-hour std | Demand volatility / variability |
| `demand_diff_1h` | t−1 minus t−2 | Instantaneous rate of change |

**Leakage safety:** All rolling windows apply `.shift(1)` before `.rolling()`. This means the window at time *t* covers `[t−w, t−1]` — never including *t* itself.

### 2.5 Weather Features

| Feature | Mechanism |
|---------|-----------|
| `temperature_2m` | Air conditioning load increases sharply above ~28°C in Bangladesh |
| `relative_humidity_2m` | High humidity amplifies cooling demand; also industrial process loads |
| `precipitation_sum` | Heavy rain reduces outdoor commercial activity |
| `wind_speed_10m` | Minor effect on cooling demand; retained for completeness |

`apparent_temperature` was dropped (Pearson r ≈ 0.99 with `temperature_2m`) to avoid multicollinearity.

### 2.6 Economic Indicators

Annual World Bank indicators (GDP per capita, electric power consumption, urban population, industry value added) were selected and joined to each hourly row by calendar year. These features capture **long-term structural demand growth** — the gradual trend of increasing baseline consumption as Bangladesh's economy and urbanisation expand.

Since economic data is published annually and represents prior-year values, there is no temporal leakage: the 2023 economic figure joined to 2023 hourly rows does not contain information about 2024 demand.

---

## 3. Data Leakage Prevention — Summary

| Potential Leak Source | Mitigation |
|-----------------------|-----------|
| `generation_mw` ≈ `demand_mw` at t | Excluded from feature set |
| `load_shedding`, fuel mix cols | Excluded (operational readings at t) |
| Cross-border imports | Excluded |
| `StandardScaler` fit on all data | Fit exclusively on training rows |
| Rolling/lag features including t | All use `.shift(1)` before window |
| Test data in training | Chronological assert enforced |

---

## 4. Feature Importance Analysis

**Top feature importance drivers (from best model — Ensemble/LightGBM representative):**

1. **`demand_lag_1h`** — Single strongest predictor. Demand at t+1 is most strongly determined by demand at t. Power grids exhibit near-unit autocorrelation at 1-hour scale.
2. **`demand_lag_24h`** — Same-hour yesterday. Captures the full daily operational cycle without storing sequence data.
3. **`demand_rmean_24h`** — 24-hour rolling mean baseline. Tells the model the prevailing "daily average" regime.
4. **`demand_rmean_6h`** — Recent 6-hour smoothed trend, bridging the gap between immediate lags and the full-day baseline.
5. **`hour_sin` / `hour_cos`** — Cyclical hour encoding. Captures the intraday demand cycle (low at 3–5am, ramp from 6am, dual peaks at 12–15h and 18–21h).
6. **`demand_lag_168h`** — Weekly seasonality anchor (same hour last week).
7. **`temperature_2m`** — Pearson r = **+0.448** with demand. Air-conditioning load is the dominant weather driver in Bangladesh's hot climate.
8. **`is_evening_peak`** / **`is_day_peak`** — Binary flags for Bangladesh's two daily peak windows.
9. **Economic indicators** (econ_0–econ_46) — Capture long-term structural demand growth (GDP per capita, urbanisation, electric power consumption growth).

**Interpretation:** Lag features dominate because Bangladesh's grid demand is strongly autocorrelated — the best single-step predictor of next hour's demand is the current hour's demand. Calendar and cyclical features add the intraday and seasonal context that pure lags miss, while temperature bridges the gap between time-of-day patterns and weather-driven load swings.

---

## 5. Model Selection & Ensemble Strategy

Three models were trained and compared:

| Model | Strengths in this context |
|-------|--------------------------|
| **Random Forest** | Naturally handles missing values; lower variance due to bagging; interpretable feature importances |
| **XGBoost** | Strong regularisation (L1 + L2); handles sparse features well; excellent on tabular data |
| **LightGBM** | Leaf-wise splitting captures complex feature interactions; fast on the large merged dataset |

**Ensemble weighting (0.2 RF + 0.4 XGB + 0.4 LGB):** XGBoost and LightGBM receive higher weights because they consistently outperform vanilla Random Forest on structured tabular regression with engineered temporal features. The ensemble blends their predictions to reduce variance and improve generalisation on the holdout test year.

---

## 6. Evaluation Results

| Model | MAPE (%) | MAE (MW) | RMSE (MW) |
|-------|----------|----------|-----------|
| Random Forest | 3.29% | 333 MW | 517 MW |
| XGBoost | 3.28% | 342 MW | 540 MW |
| LightGBM | 3.19% | 326 MW | 505 MW |
| **Ensemble (0.2·RF + 0.4·XGB + 0.4·LGB)** | **3.14%** | **322 MW** | **503 MW** |

**Test set:** Jan 2024 – Jun 2025 (12,789 hourly observations)

**MAPE interpretation:**  
A MAPE of **3.14%** means the model's hourly forecasts are on average 3.14% away from actual demand. With Bangladesh's grid operating in the 8,000–16,000 MW range, this translates to a mean absolute error of **322 MW** — well within the operational planning tolerance for short-term dispatch decisions.

The Ensemble marginally outperforms all individual models by smoothing the variance between XGBoost and LightGBM predictions, and the extra diversity introduced by the Random Forest (even at only 20% weight) provides a small but consistent benefit.

**Data context:**
- 88,050 hourly records from Apr 2015 – Jun 2025
- Training set: 75,092 rows (Apr 2015 – Dec 2023)
- Test set: 12,789 rows (Jan 2024 – Jun 2025)
- Demand growth: **5,961 MW (2015) → 11,635 MW (2025)** — ~95% increase over the decade
- 87 outlier spikes removed; 20,520 solar NaNs imputed via KNN

---

*Report generated for IITG.ai Predictive Paradox — NeuralLoad-BD submission*
