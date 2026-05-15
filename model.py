# %% [markdown]
# # 🤖 Model Training & Evaluation
# Trains three classical regression models and an ensemble on the engineered features:
# - **Random Forest** — robust baseline
# - **XGBoost** — gradient boosted trees, tuned with regularisation
# - **LightGBM** — fast leaf-wise boosting
# - **Weighted Ensemble** — blends all three predictions
#
# Primary evaluation metric: **MAPE (Mean Absolute Percentage Error)**

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error
import xgboost as xgb
import lightgbm as lgb

import config

# %% [markdown]
# ## 1. Model Training Functions

# %%
def train_random_forest(X_train, y_train):
    """Train a Random Forest regressor using hyperparameters from config."""
    print("Training Random Forest …")
    rf = RandomForestRegressor(**config.RF_PARAMS)
    rf.fit(X_train, y_train)
    print("  ✅ Random Forest training complete.")
    return rf


def train_xgboost(X_train, y_train, X_test, y_test):
    """Train an XGBoost regressor with early stopping via eval_set."""
    print("Training XGBoost …")
    model = xgb.XGBRegressor(**config.XGB_PARAMS)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=100,
    )
    print("  ✅ XGBoost training complete.")
    return model


def train_lightgbm(X_train, y_train, X_test, y_test):
    """Train a LightGBM regressor."""
    print("Training LightGBM …")
    model = lgb.LGBMRegressor(**config.LGB_PARAMS)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
    )
    print("  ✅ LightGBM training complete.")
    return model

# %% [markdown]
# ## 2. Ensemble Prediction

# %%
def ensemble_predict(rf_pred, xgb_pred, lgb_pred, weights=None):
    """
    Weighted average of three model predictions.
    Default weights from config: (RF=0.2, XGB=0.4, LGB=0.4).
    """
    if weights is None:
        weights = config.ENSEMBLE_WEIGHTS
    w_rf, w_xgb, w_lgb = weights
    return w_rf * rf_pred + w_xgb * xgb_pred + w_lgb * lgb_pred

# %% [markdown]
# ## 3. Evaluate All Models

# %%
def evaluate_models(y_test, rf_pred, xgb_pred, lgb_pred, ens_pred):
    """
    Compute MAPE for all four models and return a sorted results DataFrame.
    Also prints a formatted comparison table.
    """
    mapes = {
        "Random Forest": mean_absolute_percentage_error(y_test, rf_pred) * 100,
        "XGBoost":       mean_absolute_percentage_error(y_test, xgb_pred) * 100,
        "LightGBM":      mean_absolute_percentage_error(y_test, lgb_pred) * 100,
        "Ensemble":      mean_absolute_percentage_error(y_test, ens_pred) * 100,
    }

    results = (
        pd.DataFrame({"Model": list(mapes.keys()), "MAPE (%)": list(mapes.values())})
        .sort_values("MAPE (%)")
        .reset_index(drop=True)
    )

    print("\n" + "=" * 45)
    print("   MODEL COMPARISON — Test MAPE")
    print("=" * 45)
    print(results.to_string(index=False))
    print(f"\n🏆 Best Model : {results.iloc[0]['Model']}  →  MAPE = {results.iloc[0]['MAPE (%)']:.2f}%")
    return results

# %% [markdown]
# ## 4. Visualisation Functions

# %%
def plot_model_comparison(results: pd.DataFrame) -> None:
    """Horizontal bar chart of MAPE scores, highlighting the best model in gold."""
    best_name = results.iloc[0]["Model"]
    colors = ["gold" if m == best_name else "steelblue" for m in results["Model"]]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.barh(results["Model"], results["MAPE (%)"], color=colors, edgecolor="white")
    for i, v in enumerate(results["MAPE (%)"]):
        ax.text(v + 0.05, i, f"{v:.2f}%", va="center", fontsize=11)
    ax.set_xlabel("MAPE (%)")
    ax.set_title("Model Comparison — Test MAPE (lower is better)")
    ax.invert_yaxis()
    plt.tight_layout(); plt.show()


def plot_feature_importance(model, feature_cols: list, model_name: str, top_n: int = 20) -> None:
    """Plot top-N feature importances from a tree-based model."""
    if not hasattr(model, "feature_importances_"):
        print(f"{model_name} does not expose feature_importances_.")
        return

    imp = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=True)
    top = imp.tail(top_n)

    fig, ax = plt.subplots(figsize=(10, 8))
    top.plot(kind="barh", ax=ax, color="darkcyan", edgecolor="white")
    ax.set_title(f"Top {top_n} Feature Importances — {model_name}")
    ax.set_xlabel("Importance (gain)")
    plt.tight_layout(); plt.show()


def plot_predictions(y_test, pred, model_name: str, mape: float, zoom_days: int = 14) -> None:
    """
    Two-panel plot:
    - Full test period: actual vs predicted
    - Zoomed first N days for fine-grained visual inspection
    """
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))

    # Full test period
    axes[0].plot(y_test.index, y_test.values, lw=0.6, alpha=0.7,
                 label="Actual", color="steelblue")
    axes[0].plot(y_test.index, pred, lw=0.6, alpha=0.7,
                 label=f"{model_name} Predicted", color="tomato")
    axes[0].set_title(f"Predicted vs Actual — Test Set  (MAPE = {mape:.2f}%)")
    axes[0].set_ylabel("MW"); axes[0].legend(); axes[0].set_xlabel("Date")

    # Zoomed first N days
    zoom_end = y_test.index.min() + pd.Timedelta(days=zoom_days)
    mask = y_test.index <= zoom_end
    axes[1].plot(y_test.index[mask], y_test.values[mask], lw=1,
                 marker=".", markersize=2, label="Actual", color="steelblue")
    axes[1].plot(y_test.index[mask], pred[mask], lw=1,
                 marker=".", markersize=2, label="Predicted", color="tomato")
    axes[1].set_title(f"Zoomed — First {zoom_days} Days of Test Set")
    axes[1].set_ylabel("MW"); axes[1].legend(); axes[1].set_xlabel("Date")

    plt.tight_layout(); plt.show()


def plot_residuals(y_test, pred, model_name: str) -> None:
    """Plot prediction error distribution."""
    residuals = y_test.values - pred
    pct_errors = (residuals / y_test.values) * 100

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(residuals, bins=80, color="orchid", edgecolor="white", alpha=0.8)
    axes[0].axvline(0, color="black", lw=1, ls="--")
    axes[0].set_title(f"Residuals Distribution — {model_name}")
    axes[0].set_xlabel("Actual − Predicted (MW)")

    axes[1].hist(pct_errors, bins=80, color="slateblue", edgecolor="white", alpha=0.8)
    axes[1].axvline(0, color="black", lw=1, ls="--")
    axes[1].set_title(f"Percentage Error Distribution — {model_name}")
    axes[1].set_xlabel("% Error")

    plt.tight_layout(); plt.show()

# %% [markdown]
# ## 5. Final Metrics Summary

# %%
def print_final_metrics(y_test, pred, model_name: str) -> None:
    """Print MAPE, MAE, and RMSE for the best model."""
    mape = mean_absolute_percentage_error(y_test, pred) * 100
    mae  = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))

    print("\n" + "=" * 50)
    print(f"  FINAL RESULTS — {model_name}")
    print("=" * 50)
    print(f"  MAPE : {mape:.2f}%")
    print(f"  MAE  : {mae:,.0f} MW")
    print(f"  RMSE : {rmse:,.0f} MW")
    print("=" * 50)

# %%
print("✅ model.py loaded — train_random_forest(), train_xgboost(), train_lightgbm() ready.")
