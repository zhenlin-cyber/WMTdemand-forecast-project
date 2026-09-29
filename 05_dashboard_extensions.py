"""
05_dashboard_extensions.py
-----------------------------
Builds the additional data the dashboard's new sections need, on top of the
core pipeline (01-04):

  1. A CLEAN per-series x per-model comparison table (WMAE, MAE, std, bias
     for all 60 series-model combinations), recomputed directly from
     output/forecast_vs_actual.csv rather than reusing an earlier, buggy
     intermediate file. Powers the accuracy-vs-consistency scatter plot and
     the full comparison table.

  2. A genuine 12-MONTH-AHEAD forecast (52 weeks past the end of the
     dataset, 2012-10-26 -> 2013-10-25) for each series, using that
     series' own selection-corrected model (output/inventory_plan_v2_std_
     optimized.csv). This is real forecasting into the unknown, not a
     backtest - which raises a real problem the backtest never had: the
     LightGBM model needs future values of real-world covariates (holiday
     flag, markdown, temperature, fuel price, CPI, unemployment) that don't
     exist yet.

     Disclosed assumption: every exogenous input for a future week is
     carried forward from the SAME CALENDAR WEEK ONE YEAR EARLIER (i.e.
     "this year's holidays/promos/macro conditions look like last year's").
     This is the same logic Seasonal Naive already uses for demand itself,
     just extended to the covariates LightGBM needs. It's a simplification,
     stated plainly rather than hidden - real deployments would pull an
     actual promotional calendar and macro forecast instead of a proxy.

     Uncertainty band: forecast +/- z*sigma_e, reusing each series' own
     backtest error std (not refit) - the same sigma_e that sizes its
     safety stock, so the band shown here is literally "how much the
     safety stock this project computed is meant to absorb."
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from scipy.stats import norm
import warnings
warnings.filterwarnings("ignore")

Z = norm.ppf(0.95)
FUTURE_HORIZON = 52
MODELS = ["Naive (lag-1)", "Seasonal Naive (lag-52)", "Holt-Winters (ETS)", "LightGBM (global)"]

# =====================================================================
# PART 1: clean per-series x per-model stats (recomputed from raw backtest)
# =====================================================================
fva = pd.read_csv("output/forecast_vs_actual.csv", parse_dates=["date"])
rd_holiday = pd.read_csv("data/retail_demand.csv", parse_dates=["date"]) \
    .drop_duplicates(subset=["unique_id", "date"]).set_index(["unique_id", "date"])["holiday_flag"]
fva["holiday_flag"] = fva.set_index(["unique_id", "date"]).index.map(rd_holiday).fillna(0).astype(int).values
fva["w"] = np.where(fva["holiday_flag"] == 1, 5, 1)
fva["abs_err"] = (fva["actual"] - fva["forecast"]).abs()
fva["err"] = fva["actual"] - fva["forecast"]
fva["w_abs_err"] = fva["w"] * fva["abs_err"]

rows = []
for (uid, model), g in fva.groupby(["unique_id", "model"], observed=True):
    rows.append({
        "unique_id": uid, "model": model,
        "wmae": g["w_abs_err"].sum() / g["w"].sum(),
        "mae": g["abs_err"].mean(),
        "std": g["err"].std(),
        "bias": g["err"].mean(),
    })
comparison_full = pd.DataFrame(rows)
comparison_full["model"] = pd.Categorical(comparison_full["model"], categories=MODELS, ordered=True)
comparison_full = comparison_full.sort_values(["unique_id", "model"]).reset_index(drop=True)
comparison_full.to_csv("output/model_comparison_full.csv", index=False)
print(f"Wrote output/model_comparison_full.csv ({len(comparison_full)} rows)")

# =====================================================================
# PART 2: genuine 52-week-ahead future forecast, per series, selected model
# =====================================================================
df = pd.read_csv("data/retail_demand.csv", parse_dates=["date"]).sort_values(["unique_id", "date"]).reset_index(drop=True)
plan = pd.read_csv("output/inventory_plan_v2_std_optimized.csv")
model_used = plan.set_index("unique_id")["model_used"].to_dict()
sigma_e = plan.set_index("unique_id")["forecast_error_std"].to_dict()

last_date = df["date"].max()
future_dates = [last_date + pd.Timedelta(weeks=i) for i in range(1, FUTURE_HORIZON + 1)]
print(f"Last actual date: {last_date.date()}  ->  future horizon: {future_dates[0].date()} to {future_dates[-1].date()}")

future_rows = []

# ---- Naive, Seasonal Naive, ETS: computed directly per series ----
for uid in df["unique_id"].unique():
    model = model_used[uid]
    if model == "LightGBM (global)":
        continue  # handled below, needs the pooled global model
    s = df[df["unique_id"] == uid].sort_values("date").reset_index(drop=True)
    y = s["units_sold"].values

    if model == "Naive (lag-1)":
        fc = np.full(FUTURE_HORIZON, y[-1])
    elif model == "Seasonal Naive (lag-52)":
        fc = y[-52:]  # each future week's "same week last year" is real, already-observed data
    elif model == "Holt-Winters (ETS)":
        ets = ExponentialSmoothing(
            y, trend="add", seasonal="add", seasonal_periods=52,
            damped_trend=True, initialization_method="estimated"
        ).fit(optimized=True)
        fc = np.clip(ets.forecast(FUTURE_HORIZON), 0, None)
    else:
        raise ValueError(model)

    sig = sigma_e[uid]
    for i, d in enumerate(future_dates):
        future_rows.append({"unique_id": uid, "date": d, "model_used": model,
                             "forecast": float(fc[i]), "lower": max(float(fc[i]) - Z*sig, 0),
                             "upper": float(fc[i]) + Z*sig})

# ---- LightGBM: retrain global model on FULL data, recursive future forecast ----
lgbm_series = [uid for uid in df["unique_id"].unique() if model_used[uid] == "LightGBM (global)"]
print("Series using LightGBM (global):", lgbm_series)

feat = df.copy()
feat[["store", "dept"]] = feat["unique_id"].str.extract(r"(\d+)_Dept(\d+)")
feat["store"] = feat["store"].astype("category")
feat["dept"] = feat["dept"].astype("category")
for lag in [1, 4, 8, 52]:
    feat[f"lag_{lag}"] = feat.groupby("unique_id")["units_sold"].shift(lag)
feat["roll_mean_4"] = feat.groupby("unique_id")["units_sold"].transform(lambda x: x.shift(1).rolling(4).mean())
feat["roll_mean_8"] = feat.groupby("unique_id")["units_sold"].transform(lambda x: x.shift(1).rolling(8).mean())
feat["weekofyear"] = feat["date"].dt.isocalendar().week.astype(int)
feature_cols = ["store", "dept", "lag_1", "lag_4", "lag_8", "lag_52", "roll_mean_4",
                 "roll_mean_8", "weekofyear", "holiday_flag", "promo_flag",
                 "total_markdown", "Temperature", "Fuel_Price", "CPI", "Unemployment"]

train_feat = feat.dropna(subset=feature_cols)  # full data, no holdout cutoff - this model is going into real production use
gbm_full = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.04, num_leaves=20,
                              min_child_samples=8, random_state=42, verbosity=-1)
gbm_full.fit(train_feat[feature_cols], train_feat["units_sold"], categorical_feature=["store", "dept"])

for uid in lgbm_series:
    s = df[df["unique_id"] == uid].sort_values("date").reset_index(drop=True)
    store, dept = uid.split("_Dept")
    history = s[["date", "units_sold"]].copy()
    fc = []
    for d in future_dates:
        proxy_date = d - pd.Timedelta(weeks=52)  # "assume this year's conditions match last year's same week"
        proxy = s[s["date"] == proxy_date]
        if proxy.empty:
            proxy_row = s.iloc[-1]  # fallback, shouldn't trigger given the horizon matches one lag_52
        else:
            proxy_row = proxy.iloc[0]
        hv = history["units_sold"].values
        x = pd.DataFrame([{
            "store": store, "dept": dept, "lag_1": hv[-1], "lag_4": hv[-4], "lag_8": hv[-8],
            "lag_52": hv[-52], "roll_mean_4": hv[-4:].mean(), "roll_mean_8": hv[-8:].mean(),
            "weekofyear": d.isocalendar().week,
            "holiday_flag": proxy_row["holiday_flag"], "promo_flag": proxy_row["promo_flag"],
            "total_markdown": proxy_row["total_markdown"], "Temperature": proxy_row["Temperature"],
            "Fuel_Price": proxy_row["Fuel_Price"], "CPI": proxy_row["CPI"],
            "Unemployment": proxy_row["Unemployment"],
        }])
        x["store"] = x["store"].astype("category")
        x["dept"] = x["dept"].astype("category")
        pred = max(gbm_full.predict(x[feature_cols])[0], 0)
        fc.append(pred)
        history = pd.concat([history, pd.DataFrame({"date": [d], "units_sold": [pred]})])

    sig = sigma_e[uid]
    for i, d in enumerate(future_dates):
        future_rows.append({"unique_id": uid, "date": d, "model_used": "LightGBM (global)",
                             "forecast": float(fc[i]), "lower": max(float(fc[i]) - Z*sig, 0),
                             "upper": float(fc[i]) + Z*sig})

future_df = pd.DataFrame(future_rows).sort_values(["unique_id", "date"])
future_df.to_csv("output/future_forecast.csv", index=False)
print(f"Wrote output/future_forecast.csv ({len(future_df)} rows, {future_df.unique_id.nunique()} series)")
