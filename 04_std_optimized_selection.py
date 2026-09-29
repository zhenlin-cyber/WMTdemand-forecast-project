"""
04_std_optimized_selection.py
--------------------------------
Fixes a methodological gap surfaced after the fact: 02_forecasting_models.py
selected a single "best" model for the whole network by lowest pooled WMAE
(point accuracy), then 03_inventory_planning.py used that one model's
backtest error std to compute every series' safety stock. But safety stock
depends on std (forecast-error variance), not on WMAE/bias - so the model
that's "most accurate on average" is not necessarily the model that
produces the smallest safety-stock requirement for any given series.

This script checks that gap directly: for each of the 15 series, does the
model with the lowest backtest MAE also have the lowest backtest std? Where
they disagree, using the accuracy-best model for planning purposes is
leaving inventory savings on the table.

Fix: select the forecasting model PER SERIES by lowest backtest error std
(the property that actually drives safety stock), not by a single
network-wide WMAE ranking, and rebuild the inventory plan on that basis.
"""
import numpy as np
import pandas as pd
from scipy.stats import norm

Z = norm.ppf(0.95)
LEAD_TIME_WEEKS = 2
MODELS = ["Naive (lag-1)", "Seasonal Naive (lag-52)", "Holt-Winters (ETS)", "LightGBM (global)"]

preds = pd.read_csv("output/forecast_vs_actual.csv", parse_dates=["date"])

# ---- per-series, per-model accuracy AND consistency ----
stats = []
for uid, g in preds.groupby("unique_id"):
    for model in MODELS:
        m = g[g.model == model]
        err = m.actual - m.forecast
        stats.append({"unique_id": uid, "model": model,
                       "mae": err.abs().mean(), "std": err.std(), "bias": err.mean()})
stats = pd.DataFrame(stats)

mae_best = stats.loc[stats.groupby("unique_id")["mae"].idxmin()][["unique_id", "model"]] \
    .rename(columns={"model": "accuracy_best_model"})
std_best = stats.loc[stats.groupby("unique_id")["std"].idxmin()][["unique_id", "model"]] \
    .rename(columns={"model": "consistency_best_model"})
agreement = mae_best.merge(std_best, on="unique_id")
agreement["same_model"] = agreement.accuracy_best_model == agreement.consistency_best_model

print(f"Accuracy-best and consistency-best model agree on {agreement.same_model.mean():.0%} "
      f"of series ({agreement.same_model.sum()} of {len(agreement)})")
print(agreement.to_string(index=False))

# ---- rebuild the inventory plan using the consistency-best model per series ----
rows = []
for uid in preds.unique_id.unique():
    g = preds[preds.unique_id == uid]
    avg_demand = g["actual"].mean()  # actuals are identical regardless of model
    model = std_best.set_index("unique_id").loc[uid, "consistency_best_model"]
    sigma_e = stats[(stats.unique_id == uid) & (stats.model == model)]["std"].values[0]
    safety_stock = Z * sigma_e * np.sqrt(LEAD_TIME_WEEKS)
    rop = avg_demand * LEAD_TIME_WEEKS + safety_stock
    rows.append({
        "unique_id": uid, "model_used": model,
        "avg_weekly_demand": round(avg_demand, 0),
        "forecast_error_std": round(sigma_e, 0),
        "safety_stock_units": round(safety_stock, 0),
        "reorder_point_units": round(rop, 0),
        "safety_stock_weeks_of_cover": round(safety_stock / avg_demand, 2),
    })

plan_v2 = pd.DataFrame(rows)
plan_v1 = pd.read_csv("output/inventory_plan.csv")  # single-model (ETS) plan

comparison = plan_v1[["unique_id", "safety_stock_units"]].rename(columns={"safety_stock_units": "ss_single_model_ets"}) \
    .merge(plan_v2[["unique_id", "model_used", "safety_stock_units"]]
           .rename(columns={"safety_stock_units": "ss_per_series_optimized"}), on="unique_id")
comparison["savings_units"] = comparison.ss_single_model_ets - comparison.ss_per_series_optimized

plan_v2.to_csv("output/inventory_plan_v2_std_optimized.csv", index=False)
comparison.to_csv("output/selection_method_comparison.csv", index=False)

total_v1 = comparison.ss_single_model_ets.sum()
total_v2 = comparison.ss_per_series_optimized.sum()
print(f"\nTotal safety stock, single model (ETS for all 15 series): {total_v1:,.0f} units")
print(f"Total safety stock, per-series consistency-optimized selection: {total_v2:,.0f} units")
print(f"Additional reduction from fixing the selection objective: {(total_v1-total_v2)/total_v1*100:.1f}%")
