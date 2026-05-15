# %% [markdown]
# # ⚙️ Configuration
# Central configuration file for the NeuralLoad-BD forecasting pipeline.
# All file paths, constants, and model hyperparameters are defined here.

# %%
import os

# ─── File Paths ───────────────────────────────────────────────────────────────
PGCB_PATH    = "PGCB_date_power_demand.xlsx"
WEATHER_PATH = "weather_data.xlsx"
ECON_PATH    = "economic_full_1.csv"

# ─── Data Integrity Constants ─────────────────────────────────────────────────
# Physical upper limit for Bangladesh national grid (MW)
GRID_PHYSICAL_MAX_MW = 25_000

# IQR multiplier for outlier detection on demand_mw
IQR_MULTIPLIER = 1.5

# KNN imputer neighbours
KNN_NEIGHBOURS = 5

# ─── Feature Engineering ─────────────────────────────────────────────────────
# Lag hours to create as historical demand features
LAG_HOURS = [1, 2, 3, 24, 48, 168]

# Rolling window sizes (in hours) for rolling mean features
ROLLING_WINDOWS = [3, 6, 12, 24]

# Bangladesh weekends: Friday (4) and Saturday (5)
WEEKEND_DAYS = [4, 5]

# ─── Train / Test Split ───────────────────────────────────────────────────────
# All data from TEST_START_DATE onward is held out as the test set
TEST_START_DATE = "2024-01-01"

# ─── Leaky Columns ───────────────────────────────────────────────────────────
# These columns are real-time operational readings only available AT time t,
# not before it — including them would leak future information into the model.
LEAKY_COLS = [
    "generation_mw",         # essentially == demand_mw at t (biggest leaker)
    "load_shedding",         # operational dispatch variable at t
    "gas", "liquid_fuel",    # generation mix at t
    "coal", "hydro", "solar",
    "india_bheramara_hvdc",  # cross-border imports at t
    "india_tripura",
]

# ─── Economic Keyword Filters ─────────────────────────────────────────────────
ECON_KEYWORDS = [
    "GDP per capita",
    "Electric power consumption",
    "Energy use",
    "Population, total",
    "Urban population",
    "Industry",
]

# ─── Model Hyperparameters ────────────────────────────────────────────────────
RF_PARAMS = dict(
    n_estimators=300,
    max_depth=20,
    min_samples_leaf=5,
    n_jobs=-1,
    random_state=42,
)

XGB_PARAMS = dict(
    n_estimators=800,
    max_depth=8,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    reg_alpha=0.1,
    reg_lambda=1.0,
    tree_method="hist",
    random_state=42,
    n_jobs=-1,
)

LGB_PARAMS = dict(
    n_estimators=800,
    max_depth=10,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_samples=20,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)

# Ensemble weights: RF, XGB, LGB
ENSEMBLE_WEIGHTS = (0.2, 0.4, 0.4)

# ─── Plot Style ───────────────────────────────────────────────────────────────
PLOT_DPI = 120
PLOT_STYLE = "whitegrid"

# %%
print("✅ Config loaded.")
print(f"   PGCB path     : {PGCB_PATH}")
print(f"   Test split at : {TEST_START_DATE}")
print(f"   Lag hours     : {LAG_HOURS}")
print(f"   Rolling windows: {ROLLING_WINDOWS}")
