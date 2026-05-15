# %% [markdown]
# # 🚀 NeuralLoad-BD — Main Pipeline
# **Bangladesh National Grid Demand Forecasting**
#
# End-to-end orchestration of the full ML pipeline:
#
# | Step | Module | What happens |
# |------|--------|-------------|
# | 1 | `data_loader` | Load + clean PGCB, weather, economic data; merge into one frame |
# | 2 | `feature_engineering` | Add calendar, lag, rolling features; split train/test; scale |
# | 3 | `model` | Train RF, XGBoost, LightGBM; build ensemble; evaluate & plot |
#
# > **Submission note**: This file was written as a modular Python script
# > and converted to a Jupyter Notebook for submission.

# %% [markdown]
# ## ⚙️ Setup

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import config
import data_loader as dl
import feature_engineering as fe
import model as mdl

sns.set_theme(style=config.PLOT_STYLE)
plt.rcParams["figure.dpi"] = config.PLOT_DPI

print("=" * 60)
print("  NeuralLoad-BD — Bangladesh Demand Forecasting Pipeline")
print("=" * 60)
print(f"  PGCB path     : {config.PGCB_PATH}")
print(f"  Weather path  : {config.WEATHER_PATH}")
print(f"  Economic path : {config.ECON_PATH}")
print(f"  Test split    : from {config.TEST_START_DATE}")

# %% [markdown]
# ---
# ## 📦 Step 1 — Load Raw Data

# %%
pgcb_raw    = dl.load_pgcb(config.PGCB_PATH)
weather_raw = dl.load_weather(config.WEATHER_PATH)
econ_raw    = dl.load_economic(config.ECON_PATH)

# %% [markdown]
# ---
# ## 📊 Step 2 — Exploratory Data Analysis (Raw Data)
#
# All plots below are generated **before** any cleaning is applied,
# so they reveal the raw anomalies (outlier spikes, missing values)
# that the pipeline will subsequently handle.

# %%
# Missing values bar chart
dl.plot_missing_bar(pgcb_raw)

# %%
# Demand distribution: histogram + boxplot (with IQR upper bound marked)
dl.plot_demand_distribution(pgcb_raw)

# %%
# Full time series — RAW (before cleaning — note the outlier spikes)
dl.plot_time_series_raw(pgcb_raw)

# %%
# Seasonality patterns: hour of day / day of week / month
dl.plot_seasonality(pgcb_raw)

# %%
# Year-wise demand growth
dl.plot_yearly_growth(pgcb_raw)

# %%
# Demand vs temperature & humidity
dl.plot_demand_vs_weather(pgcb_raw, weather_raw)

# %% [markdown]
# ---
# ## 🧹 Step 3 — Clean & Merge

# %%
# Clean PGCB: resample to 1H, remove outliers, KNN impute
pgcb_clean = dl.clean_pgcb(pgcb_raw)

# %%
# Post-cleaning time series — compare with the raw plot above
# (outlier spikes are now gone, replaced by smooth KNN-imputed values)
dl.plot_time_series_clean(pgcb_clean)

# %%
# Clean weather: filter date range, drop redundant features
weather_clean = dl.clean_weather(weather_raw)

# %%
# Process economic indicators: keyword filter → melt → pivot → ffill
econ_pivot = dl.process_economic(econ_raw, config.ECON_KEYWORDS)

# %%
# Merge all three sources into a single hourly DataFrame
df_merged = dl.merge_all(pgcb_clean, weather_clean, econ_pivot)

# %% [markdown]
# ---
# ## ⚙️ Step 4 — Feature Engineering

# %%
# Build all features: calendar, lags, rolling aggregates
df_features = fe.build_features(df_merged)

# %%
# Visualise lag and rolling features on a 2-week sample
fe.plot_lag_rolling(df_features, n_days=14)

# %%
# Add supervised target (demand at t+1) and drop NaN rows
df_final = fe.add_target(df_features)

# %%
# Chronological split + StandardScaler (fit on train only)
X_train, X_test, y_train, y_test, feature_cols, scaler = fe.split_and_scale(df_final)

# %% [markdown]
# ---
# ## 🤖 Step 5 — Model Training

# %%
# 5.1  Random Forest
rf_model = mdl.train_random_forest(X_train, y_train)

# %%
# 5.2  XGBoost
xgb_model = mdl.train_xgboost(X_train, y_train, X_test, y_test)

# %%
# 5.3  LightGBM
lgb_model = mdl.train_lightgbm(X_train, y_train, X_test, y_test)

# %% [markdown]
# ---
# ## 📈 Step 6 — Evaluation

# %%
# Generate predictions from all models + ensemble
rf_pred  = rf_model.predict(X_test)
xgb_pred = xgb_model.predict(X_test)
lgb_pred = lgb_model.predict(X_test)
ens_pred = mdl.ensemble_predict(rf_pred, xgb_pred, lgb_pred)

# %%
# Compare all models on MAPE
results = mdl.evaluate_models(y_test, rf_pred, xgb_pred, lgb_pred, ens_pred)

# %%
# Model comparison bar chart
mdl.plot_model_comparison(results)

# %%
# Feature importance from the best individual model
best_name = results.iloc[0]["Model"]
best_pred = {"Random Forest": rf_pred, "XGBoost": xgb_pred,
             "LightGBM": lgb_pred, "Ensemble": ens_pred}[best_name]
best_mape = results.iloc[0]["MAPE (%)"]

best_model_map = {"Random Forest": rf_model, "XGBoost": xgb_model, "LightGBM": lgb_model}
fi_model      = best_model_map.get(best_name, lgb_model)
fi_label      = best_name if best_name in best_model_map else "LightGBM (representative)"
mdl.plot_feature_importance(fi_model, feature_cols, fi_label, top_n=20)

# %%
# Predicted vs Actual — full test set + zoomed 2-week view
mdl.plot_predictions(y_test, best_pred, best_name, best_mape, zoom_days=14)

# %%
# Residuals distribution
mdl.plot_residuals(y_test, best_pred, best_name)

# %% [markdown]
# ---
# ## 🏁 Step 7 — Final Results

# %%
mdl.print_final_metrics(y_test, best_pred, best_name)
