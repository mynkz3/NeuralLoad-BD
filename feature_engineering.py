# %% [markdown]
# # ⚙️ Feature Engineering
# Transforms the merged, cleaned dataset into a supervised learning frame:
# - Calendar / cyclical time features
# - Lag features (historical demand look-back)
# - Rolling aggregate features (short-term trends)
# - Target definition (next-hour demand)
# - Chronological train / test split with leakage verification
# - StandardScaler (fit on train only)

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

import config

# ── Receive the merged dataframe from data_loader ────────────────────────────
# When running standalone, import from data_loader:
# from data_loader import df_merged
# When running from main.py, df_merged is passed in directly.

# %% [markdown]
# ## 1. Calendar & Cyclical Time Features
#
# Tree-based models treat each observation independently, so we must
# explicitly encode temporal structure. Cyclical encoding (sin/cos)
# helps the model recognise that hour 23 and hour 0 are adjacent.

# %%
def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add hour, day-of-week, month, quarter, peak flags, and cyclical encodings."""
    df = df.copy()

    df["hour"]            = df.index.hour
    df["dayofweek"]       = df.index.dayofweek
    df["month"]           = df.index.month
    df["quarter"]         = df.index.quarter

    # Bangladesh weekends: Friday (4) and Saturday (5)
    df["is_weekend"]      = df["dayofweek"].isin(config.WEEKEND_DAYS).astype(int)

    # Typical Bangladesh peak demand windows
    df["is_day_peak"]     = df["hour"].between(12, 15).astype(int)   # 12–15h
    df["is_evening_peak"] = df["hour"].between(18, 21).astype(int)   # 18–21h

    # Cyclical encoding
    df["hour_sin"]  = np.sin(2 * np.pi * df["hour"]  / 24)
    df["hour_cos"]  = np.cos(2 * np.pi * df["hour"]  / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    print("Calendar features added:")
    print(df[["hour", "dayofweek", "month", "quarter",
              "is_weekend", "is_day_peak", "is_evening_peak"]].head(5))
    return df

# %% [markdown]
# ## 2. Lag Features
#
# Lags allow the model to "see" recent demand history:
# - Short lags (1h, 2h, 3h): immediate trend
# - Daily lags (24h, 48h): same hour yesterday / two days ago
# - Weekly lag (168h): same hour last week (strong weekly seasonality)

# %%
def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create demand lag features using config.LAG_HOURS."""
    df = df.copy()
    for lag in config.LAG_HOURS:
        df[f"demand_lag_{lag}h"] = df["demand_mw"].shift(lag)
    print(f"Lag features added : {[f'demand_lag_{h}h' for h in config.LAG_HOURS]}")
    return df

# %% [markdown]
# ## 3. Rolling Aggregate Features
#
# Rolling statistics summarise recent demand patterns:
# - Rolling mean over 3/6/12/24 hours captures smoothed recent trend
# - Rolling std (24h) captures demand volatility
# - 1-hour diff captures the rate of change
#
# All rolling ops use `.shift(1)` before rolling so NO information
# from time t leaks into the feature at time t.

# %%
def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create rolling mean, rolling std, and demand rate-of-change features."""
    df = df.copy()

    for w in config.ROLLING_WINDOWS:
        df[f"demand_rmean_{w}h"] = (
            df["demand_mw"].shift(1).rolling(w, min_periods=1).mean()
        )

    df["demand_rstd_24h"] = (
        df["demand_mw"].shift(1).rolling(24, min_periods=1).std()
    )

    # Rate of change: demand at (t-1) minus demand at (t-2)
    df["demand_diff_1h"] = df["demand_mw"].shift(1) - df["demand_mw"].shift(2)

    print(f"Rolling features added : rmean{config.ROLLING_WINDOWS} + rstd_24h + diff_1h")
    return df

# %% [markdown]
# ## 4. Apply Feature Engineering

# %%
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Orchestrate all feature engineering steps."""
    df = add_calendar_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    return df

# %% [markdown]
# ## 5. Define Supervised Target
#
# The goal is to predict **next hour's demand** (t+1).
# We achieve this by shifting `demand_mw` back by 1 row.
# Rows where the target is NaN (last row) are dropped.

# %%
def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create the supervised learning target: demand at time t+1.
    Drops rows where target or any lag feature is NaN.
    """
    df = df.copy()
    df["target"] = df["demand_mw"].shift(-1)

    lag_cols = [f"demand_lag_{h}h" for h in config.LAG_HOURS]
    df = df.dropna(subset=["target"] + lag_cols)

    print(f"After dropping NaN rows : {df.shape[0]:,} rows remaining")
    return df

# %% [markdown]
# ## 6. Train / Test Split
#
# **Strict chronological split** — no shuffling.
# All data before `TEST_START_DATE` → training set.
# All data from `TEST_START_DATE` onward → hold-out test set.
# An assertion verifies zero temporal overlap.

# %%
def split_and_scale(df: pd.DataFrame):
    """
    Split into train/test chronologically and apply StandardScaler.

    Returns:
        X_train, X_test, y_train, y_test, feature_cols, scaler
    """
    # Identify feature columns: exclude target, raw demand, and leaky operational cols
    feature_cols = [
        c for c in df.columns
        if c not in ["target", "demand_mw"] + config.LEAKY_COLS
    ]

    leaky_found = [c for c in config.LEAKY_COLS if c in df.columns]
    print(f"Leaky columns excluded : {leaky_found}")
    print(f"Feature count          : {len(feature_cols)}")

    # Chronological split
    train = df[df.index < config.TEST_START_DATE]
    test  = df[df.index >= config.TEST_START_DATE]

    X_train, y_train = train[feature_cols], train["target"]
    X_test,  y_test  = test[feature_cols],  test["target"]

    print(f"\nTrain : {X_train.shape[0]:,} rows  "
          f"({train.index.min().date()} → {train.index.max().date()})")
    print(f"Test  : {X_test.shape[0]:,} rows  "
          f"({test.index.min().date()} → {test.index.max().date()})")
    print(f"Target — Train mean: {y_train.mean():,.0f} MW | Test mean: {y_test.mean():,.0f} MW")

    # Leakage assertion
    assert train.index.max() < test.index.min(), "⚠️ DATA LEAKAGE DETECTED!"
    print("\n✅ Zero leakage confirmed — train ends strictly before test starts.")

    # StandardScaler — fit on train only
    scaler = StandardScaler()
    X_train_sc = pd.DataFrame(
        scaler.fit_transform(X_train), columns=feature_cols, index=X_train.index
    )
    X_test_sc = pd.DataFrame(
        scaler.transform(X_test), columns=feature_cols, index=X_test.index
    )
    print("✅ StandardScaler fitted on train only, applied to both splits.")

    return X_train_sc, X_test_sc, y_train, y_test, feature_cols, scaler

# %% [markdown]
# ## 7. Visualise Engineered Features

# %%
def plot_lag_rolling(df: pd.DataFrame, n_days: int = 14) -> None:
    """Plot a sample of lag and rolling features vs actual demand."""
    sample = df.iloc[:n_days * 24]

    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)

    axes[0].plot(sample.index, sample["demand_mw"], lw=1, color="steelblue", label="demand_mw (t)")
    axes[0].plot(sample.index, sample["demand_lag_24h"], lw=1, ls="--", color="tomato", label="lag 24h")
    axes[0].plot(sample.index, sample["demand_lag_168h"], lw=1, ls=":", color="purple", label="lag 168h")
    axes[0].set_title(f"Lag Features vs Actual Demand (first {n_days} days)")
    axes[0].set_ylabel("MW"); axes[0].legend()

    axes[1].plot(sample.index, sample["demand_rmean_6h"], lw=1, color="teal", label="rmean 6h")
    axes[1].plot(sample.index, sample["demand_rmean_24h"], lw=1, color="darkorange", label="rmean 24h")
    axes[1].set_title("Rolling Mean Features")
    axes[1].set_ylabel("MW"); axes[1].legend()

    axes[2].plot(sample.index, sample["demand_diff_1h"], lw=0.8, color="gray")
    axes[2].axhline(0, color="black", lw=0.5, ls="--")
    axes[2].set_title("Demand Rate of Change (diff_1h)")
    axes[2].set_ylabel("ΔMW"); axes[2].set_xlabel("Date")

    plt.tight_layout(); plt.show()

# %%
print("\n✅ feature_engineering.py loaded — call build_features(), add_target(), split_and_scale().")
