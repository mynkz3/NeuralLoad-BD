# NeuralLoad-BD 🇧🇩⚡
### Short-Term Power Demand Forecasting for Bangladesh's National Grid

> **Submission for:** IITG.ai Recruitment Task — *Predictive Paradox*

---

## 📌 Problem Statement

Accurate electricity demand forecasting is critical for grid stability. Overestimating leads to wasted generation and financial losses; underestimating causes load shedding. This project builds a robust, classical ML pipeline to predict the **next hour's demand (`demand_mw`)** on Bangladesh's national grid using historical consumption, weather, and macroeconomic data.

---

## 🗂️ Project Structure

```

NeuralLoad-BD/
├── config.py / .ipynb              # Constants & hyperparams
├── data_loader.py / .ipynb         # Load, clean, merge, EDA plots
├── feature_engineering.py / .ipynb # Features + train/test split
├── model.py / .ipynb               # Training + evaluation + plots
├── main.py / .ipynb                # End-to-end pipeline (fully executed)
├── README.md                       # Project overview + results
└── REPORT.md                       # Methodology report (Deliverable B)

```


## 🔬 Approach & Methodology

### 1. Data Preparation

| Dataset | Challenges | Solution |
|---------|-----------|---------|
| PGCB (hourly demand) | Mixed 30-min/1-hour frequencies, duplicate timestamps, extreme spikes | Resample to strict 1H mean; IQR + 25,000 MW physical cap for outlier removal; KNN imputation |
| Weather (Open-Meteo) | Highly correlated features (`apparent_temperature` ≈ `temperature_2m`) | Drop redundant cols; filter to PGCB date range |
| Economic (World Bank) | Annual granularity vs hourly demand | Keyword-filtered indicator selection; melt → pivot → forward-fill; join by calendar year |

### 2. Feature Engineering

Since classical ML models treat each row independently, the concept of **time** must be explicitly built into the feature set:

- **Calendar features**: hour, day-of-week, month, quarter, weekend flag, peak-hour flags (12–15h, 18–21h), cyclical sin/cos encodings
- **Lag features**: demand at t−1, t−2, t−3, t−24, t−48, t−168 hours
- **Rolling aggregates**: 3h, 6h, 12h, 24h rolling means; 24h rolling std; 1h rate of change

All rolling and lag features use `.shift(1)` before the window — **no information from time t leaks into the feature at t**.

### 3. Train / Test Strategy

- **Strict chronological split**: training on all data before `2024-01-01`, test on all data from `2024-01-01` onward
- `StandardScaler` **fit on training data only** — transform applied separately to test set
- Leakage assertion (`assert train.index.max() < test.index.min()`) enforced in code

### 4. Leaky Column Exclusion

The following columns are excluded from features — they represent **real-time operational readings at time t** that would not be available when forecasting t+1:

`generation_mw`, `load_shedding`, `gas`, `liquid_fuel`, `coal`, `hydro`, `solar`, `india_bheramara_hvdc`, `india_tripura`

### 5. Models Trained

| Model | Rationale |
|-------|-----------|
| **Random Forest** | Strong baseline; robust to outliers and noise |
| **XGBoost** | Gradient boosting with L1/L2 regularisation; handles feature interactions |
| **LightGBM** | Leaf-wise splitting; faster training on large datasets |
| **Weighted Ensemble** | Blends RF (20%) + XGBoost (40%) + LightGBM (40%) for variance reduction |

---

## 📊 Results

| Model | Test MAPE (%) | MAE (MW) | RMSE (MW) |
|-------|--------------|----------|-----------|
| Random Forest | 3.29% | 333 MW | 517 MW |
| XGBoost | 3.28% | 342 MW | 540 MW |
| LightGBM | 3.19% | 326 MW | 505 MW |
| **Ensemble** | **3.14%** | **322 MW** | **503 MW** |

**Primary metric:** Mean Absolute Percentage Error (MAPE)  
**Secondary metrics:** MAE (MW), RMSE (MW)  
**Test set:** Jan 2024 – Jun 2025 (12,789 hourly observations)

---

## 🧠 Key Insights

**Data highlights:**
- 88,050 hourly records spanning Apr 2015 – Jun 2025
- Demand grew from **5,961 MW (2015)** → **11,635 MW (2025)** — nearly 2× in a decade
- 87 outlier spikes detected and cleaned via IQR + 25,000 MW physical cap
- Temperature–Demand Pearson r = **+0.448** (strong positive in hot months due to A/C load)

**Feature importance drivers:**
- **`demand_lag_1h`** — strongest single predictor; demand is highly autocorrelated at 1-hour scale
- **`demand_lag_24h`** — same hour yesterday; captures the daily cycle
- **`demand_rmean_24h`** — 24-hour rolling baseline
- **`hour_sin` / `hour_cos`** — intraday periodicity (morning ramp, evening peak 18–21h)
- **`demand_lag_168h`** — weekly seasonality (same hour last week)
- **`temperature_2m`** — weather-driven cooling/heating load (r = 0.448 with demand)
- **Economic indicators** — long-term structural growth trend (GDP, urbanisation)

---

## ⚠️ Constraints Respected

- ❌ No deep learning (LSTMs, Transformers)
- ❌ No autoregressive packages (ARIMA, Prophet)
- ✅ Classical ML only (tree-based regressors)
- ✅ Zero data leakage
- ✅ Chronological train/test split
- ✅ MAPE as primary evaluation metric

---

## 📦 Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm openpyxl
```

