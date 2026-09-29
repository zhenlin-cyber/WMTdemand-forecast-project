"""
03_inventory_planning.py  (real-data version)
------------------------------------------------
Same logic as the original version, adapted to weekly cadence:

    Safety Stock (SS)   = z * sigma_e * sqrt(L)
    Reorder Point (ROP) = (avg weekly demand * L) + SS

  - z: 95% service level -> z = 1.645
  - sigma_e: backtest forecast-error std (weekly), from the WINNING model
    (selected in 02_forecasting_models.py by WMAE - here, Holt-Winters ETS
    beat the global LightGBM model; see README for why, and why that's a
    reportable result rather than a failure of the analysis)
  - L: lead time, assumed 2 weeks (a plausible general-merchandise DC-to-
     store replenishment cadence; not in the dataset, so stated as an
     explicit assumption)

Baseline for the "what did better forecasting buy us" comparison: naive
(lag-1), the simplest possible forecast, so the safety-stock reduction
number reflects the full value of doing any real modeling at all - not just
the marginal gap between two sophisticated methods.
"""
import numpy as np
import pandas as pd
from scipy.stats import norm

LEAD_TIME_WEEKS = 2
SERVICE_LEVEL = 0.95
Z = norm.ppf(SERVICE_LEVEL)
BASELINE_MODEL = "Naive (lag-1)"

preds = pd.read_csv("output/forecast_vs_actual.csv", parse_dates=["date"])
with open("output/best_model.txt") as f:
    best_model = f.read().strip()

rows = []
for uid, g in preds.groupby("unique_id"):
    avg_weekly_demand = g[g["model"] == best_model]["actual"].mean()
    for model_name in [BASELINE_MODEL, best_model]:
        m = g[g["model"] == model_name]
        errors = m["actual"] - m["forecast"]
        sigma_e = errors.std()
        safety_stock = Z * sigma_e * np.sqrt(LEAD_TIME_WEEKS)
        rop = avg_weekly_demand * LEAD_TIME_WEEKS + safety_stock
        rows.append({
            "unique_id": uid, "model": model_name,
            "avg_weekly_demand": round(avg_weekly_demand, 0),
            "forecast_error_std": round(sigma_e, 0),
            "safety_stock_units": round(safety_stock, 0),
            "reorder_point_units": round(rop, 0),
            "safety_stock_weeks_of_cover": round(safety_stock / avg_weekly_demand, 2),
        })

inv = pd.DataFrame(rows)
pivot = inv.pivot_table(index="unique_id", columns="model", values="safety_stock_units").reset_index()
pivot["ss_reduction_pct"] = (pivot[BASELINE_MODEL] - pivot[best_model]) / pivot[BASELINE_MODEL] * 100

best = inv[inv["model"] == best_model].copy()
best["error_cv"] = best["forecast_error_std"] / best["avg_weekly_demand"]
best["risk_flag"] = pd.qcut(best["error_cv"], 4, labels=["Low", "Moderate", "Elevated", "High"])
best = best.merge(pivot[["unique_id", "ss_reduction_pct"]], on="unique_id")
best = best.sort_values("error_cv", ascending=False)

best.to_csv("output/inventory_plan.csv", index=False)
inv.to_csv("output/inventory_plan_both_models.csv", index=False)

print(f"Best model (used for planning): {best_model}")
print(f"Service level: {SERVICE_LEVEL:.0%} | Lead time: {LEAD_TIME_WEEKS} weeks | z = {Z:.3f}")
print(f"\nMean safety-stock change vs. naive baseline, same service level: "
      f"{pivot['ss_reduction_pct'].mean():.1f}%")
print(f"\nHigh-risk (top quartile forecast uncertainty):")
print(best[best["risk_flag"] == "High"][["unique_id", "avg_weekly_demand", "error_cv", "safety_stock_units"]]
      .to_string(index=False))
print(f"\nTotal safety stock ({best_model}): {best['safety_stock_units'].sum():,.0f} units | "
      f"Total under naive baseline: {pivot[BASELINE_MODEL].sum():,.0f} units")
