# %% [markdown]
# # 📦 Data Loader
# Handles loading, cleaning, and merging all three raw datasets:
# - `PGCB_date_power_demand.xlsx` — hourly grid demand/generation
# - `weather_data.xlsx` — hourly environmental variables
# - `economic_full_1.csv` — annual World Bank macroeconomic indicators

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import KNNImputer

import config

sns.set_theme(style=config.PLOT_STYLE)
plt.rcParams["figure.dpi"] = config.PLOT_DPI

# %% [markdown]
# ## 1. Load Raw Datasets

# %%
def load_pgcb(path: str) -> pd.DataFrame:
    """Load and parse the PGCB hourly demand/generation Excel file."""
    df = pd.read_excel(path, parse_dates=["datetime"])
    print(f"PGCB loaded       : {df.shape[0]:,} rows × {df.shape[1]} cols")
    print(f"  Date range      : {df['datetime'].min()} → {df['datetime'].max()}")
    return df


def load_weather(path: str) -> pd.DataFrame:
    """Load the Open-Meteo hourly weather Excel file.
    The file has a 3-row header; column names are trimmed to the first token
    (removing the unit suffix appended by Open-Meteo exports).
    """
    df = pd.read_excel(path, skiprows=3)
    df.columns = [col.split(" ")[0] for col in df.columns]
    df["time"] = pd.to_datetime(df["time"])
    print(f"Weather loaded    : {df.shape[0]:,} rows × {df.shape[1]} cols")
    print(f"  Date range      : {df['time'].min()} → {df['time'].max()}")
    return df


def load_economic(path: str) -> pd.DataFrame:
    """Load the World Bank annual economic indicators CSV."""
    df = pd.read_csv(path)
    print(f"Economic loaded   : {df.shape[0]:,} rows × {df.shape[1]} cols")
    return df

# %% [markdown]
# ## 2. Clean & Preprocess PGCB Data

# %%
def clean_pgcb(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full PGCB cleaning pipeline:
    1. Drop columns that are mostly missing or irrelevant.
    2. Sort chronologically.
    3. Resample to strict 1-hour mean (resolves duplicates & mixed 30-min rows).
    4. Flag and NaN-replace IQR outliers in demand_mw.
    5. KNN-impute all remaining NaNs.
    """
    # ── 2.1  Drop unusable columns ──────────────────────────────────────────
    drop_cols = ["remarks", "india_adani", "nepal", "wind"]
    existing_drops = [c for c in drop_cols if c in df.columns]
    df = df.drop(columns=existing_drops)
    print(f"Dropped columns   : {existing_drops}")

    # ── 2.2  Sort & resample to 1H mean ─────────────────────────────────────
    df = df.sort_values("datetime").set_index("datetime")
    df = df.resample("1h").mean()
    df = df.dropna(how="all")                    # remove hours with zero data
    print(f"After 1H resample : {df.shape[0]:,} rows")
    print(f"  Frequency check : {pd.infer_freq(df.index[:100])}")

    # ── 2.3  Outlier flagging on demand_mw (IQR + physical cap) ─────────────
    Q1 = df["demand_mw"].quantile(0.25)
    Q3 = df["demand_mw"].quantile(0.75)
    IQR = Q3 - Q1
    lower_b = Q1 - config.IQR_MULTIPLIER * IQR
    upper_b = Q3 + config.IQR_MULTIPLIER * IQR

    # Physical sanity cap: Bangladesh grid cannot exceed 25,000 MW
    upper_b = min(upper_b, config.GRID_PHYSICAL_MAX_MW)

    outlier_mask = (df["demand_mw"] < lower_b) | (df["demand_mw"] > upper_b)
    n_out = outlier_mask.sum()
    df.loc[outlier_mask, "demand_mw"] = np.nan
    print(f"Outliers (demand_mw → NaN): {n_out}")

    # ── 2.4  KNN imputation (fit on whole pre-split frame — no split yet) ───
    print(f"NaNs before KNN impute:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
    imputer = KNNImputer(n_neighbors=config.KNN_NEIGHBOURS, weights="distance")
    df_imp = pd.DataFrame(
        imputer.fit_transform(df),
        columns=df.columns,
        index=df.index,
    )
    print(f"KNN imputation done. Remaining NaNs: {df_imp.isnull().sum().sum()}")
    return df_imp

# %% [markdown]
# ## 3. Clean & Preprocess Weather Data

# %%
def clean_weather(df: pd.DataFrame, start_date: str = "2015-01-01") -> pd.DataFrame:
    """
    Weather cleaning pipeline:
    1. Filter to PGCB coverage window (2015 onward).
    2. Rename 'time' → 'datetime' and set as index.
    3. Drop features highly correlated with temperature_2m (redundant).
    """
    df = df[df["time"] >= start_date].copy()
    df = df.rename(columns={"time": "datetime"})

    # Drop temporary EDA columns if they exist
    for col in ["month_period"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    df = df.set_index("datetime")

    # Drop apparent_temperature (r~0.99 with temperature_2m) and soil_temperature
    redundant = ["apparent_temperature", "soil_temperature_0_to_7cm"]
    existing_redundant = [c for c in redundant if c in df.columns]
    df = df.drop(columns=existing_redundant, errors="ignore")

    print(f"Weather cleaned   : {df.shape[0]:,} rows")
    print(f"  Date range      : {df.index.min()} → {df.index.max()}")
    print(f"  Features kept   : {list(df.columns)}")
    return df

# %% [markdown]
# ## 4. Process Economic Indicators

# %%
def process_economic(df: pd.DataFrame, keywords: list, year_range: tuple = (2015, 2025)) -> pd.DataFrame:
    """
    Economic data processing pipeline:
    1. Filter indicators by keyword relevance.
    2. Melt wide → long format.
    3. Pivot so each indicator is a column, indexed by year.
    4. Forward-fill + backward-fill missing years.
    5. Clip to the relevant year range.

    Returns a DataFrame indexed by year with one column per indicator.
    The columns are renamed to short econ_0, econ_1, ... aliases to
    avoid join issues with long indicator names.
    """
    mask = df["Indicator Name"].str.contains("|".join(keywords), case=False, na=False)
    econ_sel = df[mask].copy()
    print(f"Economic indicators selected: {len(econ_sel)}")

    year_cols = [c for c in econ_sel.columns if c.isdigit()]
    econ_long = econ_sel.melt(
        id_vars=["Indicator Name", "Indicator Code"],
        value_vars=year_cols,
        var_name="year",
        value_name="value",
    )
    econ_long["year"] = econ_long["year"].astype(int)

    econ_pivot = econ_long.pivot_table(index="year", columns="Indicator Name", values="value")
    econ_pivot = econ_pivot.sort_index().ffill().bfill()
    econ_pivot = econ_pivot.loc[year_range[0]: year_range[1]]

    # Rename to short aliases
    econ_pivot.columns = [f"econ_{i}" for i in range(len(econ_pivot.columns))]

    print(f"Economic pivot    : {econ_pivot.shape} (years {econ_pivot.index.min()}–{econ_pivot.index.max()})")
    print(f"  Remaining NaNs  : {econ_pivot.isnull().sum().sum()}")
    return econ_pivot

# %% [markdown]
# ## 5. Merge All Three Datasets

# %%
def merge_all(pgcb: pd.DataFrame, weather: pd.DataFrame, econ: pd.DataFrame) -> pd.DataFrame:
    """
    Merge PGCB (hourly) + Weather (hourly) + Economic (annual):
    - PGCB ← weather: left join on DatetimeIndex
    - Result ← economic: map by calendar year (no temporal leakage — annual values
      represent long-term macro context, not future operational data)
    """
    # PGCB + Weather (both hourly, same DatetimeIndex)
    df = pgcb.join(weather, how="left")
    print(f"After PGCB + Weather merge : {df.shape}")

    # Economic by year
    df["year"] = df.index.year
    for col in econ.columns:
        df[col] = df["year"].map(econ[col])
    df = df.drop(columns=["year"])

    # Final NaN cleanup (weather join edges, economic gaps)
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing) > 0:
        print(f"Residual NaNs after merge (filling with ffill/bfill):\n{missing}")
        df = df.ffill().bfill()

    print(f"Final merged shape : {df.shape[0]:,} rows × {df.shape[1]} cols")
    print(f"  Date range       : {df.index.min()} → {df.index.max()}")
    print(f"  Remaining NaNs   : {df.isnull().sum().sum()}")
    return df

# %% [markdown]
# ## 6. EDA Visualisation Functions

# %%
def plot_missing_bar(pgcb_raw: pd.DataFrame) -> None:
    """Bar chart of missing values by column in the raw PGCB data."""
    missing = pgcb_raw.isnull().sum().reset_index()
    missing.columns = ["Column", "Missing"]
    missing["Pct"] = (missing["Missing"] / len(pgcb_raw) * 100).round(2)
    miss_cols = missing[missing["Missing"] > 0].sort_values("Missing", ascending=True)

    if len(miss_cols) > 0:
        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.barh(miss_cols["Column"], miss_cols["Pct"], color="salmon", edgecolor="darkred")
        ax.set_xlabel("% Missing")
        ax.set_title("PGCB — Missing Values by Column")
        for bar, val in zip(bars, miss_cols["Pct"]):
            ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                    f"{val}%", va="center", fontsize=9)
        plt.tight_layout(); plt.show()
    else:
        print("No missing values in raw PGCB data.")


def plot_demand_distribution(pgcb_raw: pd.DataFrame) -> None:
    """Histogram and boxplot of raw demand_mw distribution."""
    Q1 = pgcb_raw["demand_mw"].quantile(0.25)
    Q3 = pgcb_raw["demand_mw"].quantile(0.75)
    IQR = Q3 - Q1
    upper_bound = Q3 + config.IQR_MULTIPLIER * IQR

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(pgcb_raw["demand_mw"].dropna(), bins=80, color="steelblue", edgecolor="white", alpha=0.8)
    axes[0].axvline(upper_bound, color="red", ls="--", label=f"IQR upper = {upper_bound:.0f} MW")
    axes[0].set_title("demand_mw — Distribution (Raw)")
    axes[0].set_xlabel("MW"); axes[0].legend()

    axes[1].boxplot(pgcb_raw["demand_mw"].dropna(), vert=True, patch_artist=True,
                    boxprops=dict(facecolor="lightblue"))
    axes[1].set_title("demand_mw — Boxplot (Raw)")
    axes[1].set_ylabel("MW")
    plt.tight_layout(); plt.show()


def plot_time_series_raw(pgcb_raw: pd.DataFrame) -> None:
    """Full time series + daily resampled trend — RAW data (before cleaning)."""
    pgcb_sorted = pgcb_raw.sort_values("datetime")
    fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)

    axes[0].plot(pgcb_sorted["datetime"], pgcb_sorted["demand_mw"],
                 lw=0.3, alpha=0.6, color="steelblue")
    axes[0].set_title("Hourly Demand (MW) — BEFORE Cleaning (raw data, includes outlier spikes)")
    axes[0].set_ylabel("MW")

    daily = pgcb_sorted.set_index("datetime")["demand_mw"].resample("D").mean()
    axes[1].plot(daily.index, daily.values, lw=0.8, color="tomato")
    axes[1].set_title("Daily Average Demand (MW) — BEFORE Cleaning")
    axes[1].set_ylabel("MW"); axes[1].set_xlabel("Date")
    plt.tight_layout(); plt.show()


def plot_time_series_clean(pgcb_clean: pd.DataFrame) -> None:
    """Full time series + daily resampled trend — AFTER cleaning."""
    fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)

    axes[0].plot(pgcb_clean.index, pgcb_clean["demand_mw"],
                 lw=0.3, alpha=0.6, color="green")
    axes[0].set_title("Hourly Demand (MW) — AFTER Cleaning (outliers removed, KNN imputed)")
    axes[0].set_ylabel("MW")

    daily = pgcb_clean["demand_mw"].resample("D").mean()
    axes[1].plot(daily.index, daily.values, lw=0.8, color="darkorange")
    axes[1].set_title("Daily Average Demand (MW) — AFTER Cleaning")
    axes[1].set_ylabel("MW"); axes[1].set_xlabel("Date")
    plt.tight_layout(); plt.show()


def plot_seasonality(pgcb_raw: pd.DataFrame) -> None:
    """Seasonality charts: hour, day-of-week, month."""
    pgcb_sorted = pgcb_raw.sort_values("datetime")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    hourly_avg = pgcb_sorted.groupby(pgcb_sorted["datetime"].dt.hour)["demand_mw"].mean()
    axes[0].plot(hourly_avg.index, hourly_avg.values, marker="o", color="steelblue")
    axes[0].set_title("Avg Demand by Hour of Day"); axes[0].set_xlabel("Hour"); axes[0].set_ylabel("MW")

    dow_avg = pgcb_sorted.groupby(pgcb_sorted["datetime"].dt.dayofweek)["demand_mw"].mean()
    day_labels = ["Mon", "Tue", "Wed", "Thu", "Fri*", "Sat*", "Sun"]
    colors = ["steelblue"] * 4 + ["tomato", "tomato"] + ["steelblue"]
    axes[1].bar(dow_avg.index, dow_avg.values, color=colors, edgecolor="white")
    axes[1].set_xticks(range(7)); axes[1].set_xticklabels(day_labels)
    axes[1].set_title("Avg Demand by Day (* = Bangladesh weekend)"); axes[1].set_ylabel("MW")

    monthly_avg = pgcb_sorted.groupby(pgcb_sorted["datetime"].dt.month)["demand_mw"].mean()
    axes[2].bar(monthly_avg.index, monthly_avg.values, color="teal", edgecolor="white")
    axes[2].set_title("Avg Demand by Month"); axes[2].set_xticks(range(1, 13)); axes[2].set_ylabel("MW")
    plt.tight_layout(); plt.show()


def plot_yearly_growth(pgcb_raw: pd.DataFrame) -> None:
    """Year-wise demand growth bar chart."""
    pgcb_sorted = pgcb_raw.sort_values("datetime")
    yearly_avg = pgcb_sorted.groupby(pgcb_sorted["datetime"].dt.year)["demand_mw"].mean()
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(yearly_avg.index, yearly_avg.values, color="darkcyan", edgecolor="white")
    for bar, val in zip(bars, yearly_avg.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 80,
                f"{val:,.0f}", ha="center", va="bottom", fontsize=9)
    ax.set_title("Year-wise Average Demand (MW)")
    ax.set_xlabel("Year"); ax.set_ylabel("MW")
    plt.tight_layout(); plt.show()
    print(f"Growth: {yearly_avg.iloc[0]:,.0f} MW ({yearly_avg.index[0]}) → "
          f"{yearly_avg.iloc[-1]:,.0f} MW ({yearly_avg.index[-1]})")


def plot_demand_vs_weather(pgcb_raw: pd.DataFrame, weather_raw: pd.DataFrame) -> None:
    """Scatter: demand vs temperature & humidity."""
    pgcb_sorted = pgcb_raw.sort_values("datetime")
    weather_subset = weather_raw.rename(columns={"time": "datetime"})
    merged_preview = pd.merge_asof(
        pgcb_sorted.sort_values("datetime"),
        weather_subset[["datetime", "temperature_2m", "relative_humidity_2m"]].sort_values("datetime"),
        on="datetime", direction="nearest", tolerance=pd.Timedelta("1h"),
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].scatter(merged_preview["temperature_2m"], merged_preview["demand_mw"],
                    s=1, alpha=0.2, color="orangered")
    axes[0].set_title("Demand vs Temperature"); axes[0].set_xlabel("°C"); axes[0].set_ylabel("MW")

    axes[1].scatter(merged_preview["relative_humidity_2m"], merged_preview["demand_mw"],
                    s=1, alpha=0.2, color="teal")
    axes[1].set_title("Demand vs Relative Humidity"); axes[1].set_xlabel("RH (%)"); axes[1].set_ylabel("MW")
    plt.tight_layout(); plt.show()

    r_temp = merged_preview[["temperature_2m", "demand_mw"]].dropna().corr().iloc[0, 1]
    r_hum  = merged_preview[["relative_humidity_2m", "demand_mw"]].dropna().corr().iloc[0, 1]
    print(f"Pearson r — Temperature vs Demand : {r_temp:.3f}")
    print(f"Pearson r — Humidity vs Demand    : {r_hum:.3f}")

# %%
print("✅ data_loader.py loaded — functions ready.")
