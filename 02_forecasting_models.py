"""
02_forecasting_models.py  (real-data version)
------------------------------------------------
Backtests four forecasting approaches on the real Walmart weekly Store x Dept
sales data, from simplest to most capable.

  1. Naive (lag-1)        - "same as last week." The floor.
  2. Seasonal naive (lag-52) - "same week last year." At weekly grain, the
     dominant seasonal cycle is annual (Thanksgiving, Christmas, back-to-
     school), not weekly - this is the more meaningful naive baseline than
     lag-1 for this data, and is reported separately from it.
  3. Holt-Winters (ETS), fit independently per series - trend + additive
     seasonality. A LOCAL model: only ever sees its own series' history.
  4. LightGBM, fit GLOBALLY - one model pooled across all 15 series, with
     store/dept as categorical features plus lag/rolling demand history and
     the real exogenous covariates (holiday flag, markdown/promo total,
     temperature, fuel price, CPI, unemployment). Pooling lets the model
     borrow signal across series (a holiday effect learned from one dept
     transfers to another) - the same reason global tree-based models won
     the actual M5 forecasting competition rather than 3,000+ independent
     local models. This is a deliberate local-vs-global contrast with ETS.

Evaluation metric: WMAE, defined exactly as the real Kaggle competition
defined it for this dataset - weight = 5 on holiday weeks, 1 otherwise -
because the business cost of a miss isn't uniform across weeks; a miss the
week of Thanksgiving matters more than a miss in a quiet March week, and the
metric should reflect that rather than treating every week equally.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings
warnings.filterwarnings("ignore")

HORIZON = 12  # weeks (~one quarter)

df = pd.read_csv("data/retail_demand.csv", parse_dates=["date"])
df = df.sort_values(["unique_id", "date"]).reset_index(drop=True)
series_ids = df["unique_id"].unique()
cutoff = sorted(df["date"].unique())[-HORIZON]

def wmae(y_true, y_pred, is_holiday):
    w = np.where(is_holiday == 1, 5, 1)
    return np.sum(w * np.abs(y_true - y_pred)) / np.sum(w)

# ---------------- 1-2-3: per-series naive / seasonal-naive / ETS ----------------
results, all_preds = [], []
for uid in series_ids:
    s = df[df["unique_id"] == uid].reset_index(drop=True)
    train = s[s["date"] < cutoff]
    test = s[s["date"] >= cutoff].reset_index(drop=True)
    y_train, y_test = train["units_sold"].values, test["units_sold"].values
    hol_test = test["holiday_flag"].values

    naive_fc = np.full(HORIZON, y_train[-1])
    seas_fc = y_train[-52:-52 + HORIZON] if len(y_train) >= 52 + HORIZON else np.tile(y_train[-4:], 3)[:HORIZON]

    try:
        ets = ExponentialSmoothing(
            y_train, trend="add", seasonal="add", seasonal_periods=52,
            damped_trend=True, initialization_method="estimated"
        ).fit(optimized=True)
        ets_fc = np.clip(ets.forecast(HORIZON), 0, None)
    except Exception:
        ets_fc = seas_fc.astype(float)

    for name, fc in [("Naive (lag-1)", naive_fc), ("Seasonal Naive (lag-52)", seas_fc),
                      ("Holt-Winters (ETS)", ets_fc)]:
        results.append({"unique_id": uid, "model": name,
                         "wmae": wmae(y_test, fc, hol_test),
                         "mae": np.mean(np.abs(y_test - fc)),
                         "bias": np.mean(fc - y_test)})
        for i in range(HORIZON):
            all_preds.append({"unique_id": uid, "date": test.iloc[i]["date"], "model": name,
                               "actual": y_test[i], "forecast": fc[i]})

# ---------------- 4: global LightGBM, pooled across all 15 series ----------------
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

train_feat = feat[feat["date"] < cutoff].dropna(subset=feature_cols)
gbm = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.04, num_leaves=20,
                         min_child_samples=8, random_state=42, verbosity=-1)
gbm.fit(train_feat[feature_cols], train_feat["units_sold"],
        categorical_feature=["store", "dept"])

for uid in series_ids:
    s = df[df["unique_id"] == uid].reset_index(drop=True)
    train = s[s["date"] < cutoff]
    test = s[s["date"] >= cutoff].reset_index(drop=True)
    store, dept = uid.split("_Dept")
    history = train[["date", "units_sold"]].copy()
    gbm_fc = []
    for i in range(HORIZON):
        row = test.iloc[i]
        hv = history["units_sold"].values
        x = pd.DataFrame([{
            "store": store, "dept": dept, "lag_1": hv[-1], "lag_4": hv[-4], "lag_8": hv[-8],
            "lag_52": hv[-52], "roll_mean_4": hv[-4:].mean(), "roll_mean_8": hv[-8:].mean(),
            "weekofyear": row["date"].isocalendar().week, "holiday_flag": row["holiday_flag"],
            "promo_flag": row["promo_flag"], "total_markdown": row["total_markdown"],
            "Temperature": row["Temperature"], "Fuel_Price": row["Fuel_Price"],
            "CPI": row["CPI"], "Unemployment": row["Unemployment"],
        }])
        x["store"] = x["store"].astype("category")
        x["dept"] = x["dept"].astype("category")
        pred = max(gbm.predict(x[feature_cols])[0], 0)
        gbm_fc.append(pred)
        history = pd.concat([history, pd.DataFrame({"date": [row["date"]], "units_sold": [pred]})])
    gbm_fc = np.array(gbm_fc)
    y_test, hol_test = test["units_sold"].values, test["holiday_flag"].values
    results.append({"unique_id": uid, "model": "LightGBM (global)",
                     "wmae": wmae(y_test, gbm_fc, hol_test),
                     "mae": np.mean(np.abs(y_test - gbm_fc)),
                     "bias": np.mean(gbm_fc - y_test)})
    for i in range(HORIZON):
        all_preds.append({"unique_id": uid, "date": test.iloc[i]["date"], "model": "LightGBM (global)",
                           "actual": y_test[i], "forecast": gbm_fc[i]})

results_df = pd.DataFrame(results)
preds_df = pd.DataFrame(all_preds)
results_df.to_csv("output/model_accuracy.csv", index=False)
preds_df.to_csv("output/forecast_vs_actual.csv", index=False)

summary = results_df.groupby("model")[["wmae", "mae", "bias"]].mean().round(1).sort_values("wmae")
print("=== Model comparison (mean WMAE across 15 series, 12-week holdout) ===")
print(summary)
summary.to_csv("output/model_comparison_summary.csv")
best_model = summary.index[0]
print(f"\nBest model by WMAE: {best_model}")
with open("output/best_model.txt", "w") as f:
    f.write(best_model)
