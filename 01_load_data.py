"""
01_load_data.py
-----------------
Loads and cleans the real Walmart Recruiting: Store Sales Forecasting dataset
(Kaggle, 2014) - 421,570 weekly Store x Dept sales records, Feb 2010-Oct 2012,
with real store-level covariates (temperature, fuel price, CPI, unemployment,
five promotional markdown series) and holiday flags.

Source: https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting
(train.csv + features.csv, retrieved from a public GitHub mirror of the
competition files since the raw files aren't hosted on Kaggle without auth).

Scope: the full dataset is 45 stores x 81 depts = up to 3,645 series - too
many to backtest three models against by hand in a weekend and still read
the results. Scoped down to the 5 highest-volume stores x 3 highest-volume,
fully-populated departments = 15 series (all confirmed to have complete,
gap-free 143-week history - no interpolation needed). This mirrors a real
practical constraint: an analyst with limited time prioritizes the
highest-revenue-impact combinations first.

Data-quality handling (the real-world part synthetic data doesn't have):
  - MarkDown1-5 are only populated from Nov 2011 onward (per the official
    competition documentation) - NaN before that isn't missing data, it's
    "no markdown program existed yet." Filled with 0, not imputed, since 0
    is the correct value, not a guess.
  - IsHoliday appears in both files (train + features); the features.csv
    version is used since it also covers the columns we merge on.
"""
import pandas as pd
import numpy as np

TOP_STORES = [20, 4, 14, 13, 2]     # 5 highest total-sales stores
TOP_DEPTS = [92, 95, 38]            # 3 highest total-sales, fully-populated depts

train = pd.read_csv("data/train.csv", parse_dates=["Date"])
feat = pd.read_csv("data/features.csv", parse_dates=["Date"])

df = train[train["Store"].isin(TOP_STORES) & train["Dept"].isin(TOP_DEPTS)].copy()
df = df.drop(columns=["IsHoliday"])  # keep the features.csv version after merge
df = df.merge(
    feat[["Store", "Date", "Temperature", "Fuel_Price", "MarkDown1", "MarkDown2",
          "MarkDown3", "MarkDown4", "MarkDown5", "CPI", "Unemployment", "IsHoliday"]],
    on=["Store", "Date"], how="left"
)

markdown_cols = ["MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5"]
df[markdown_cols] = df[markdown_cols].fillna(0)
df["total_markdown"] = df[markdown_cols].sum(axis=1)
df["promo_flag"] = (df["total_markdown"] > 0).astype(int)
df["holiday_flag"] = df["IsHoliday"].astype(int)

# CPI/Unemployment occasionally null at series edges - forward/back fill per store
df = df.sort_values(["Store", "Date"])
for col in ["CPI", "Unemployment", "Temperature", "Fuel_Price"]:
    df[col] = df.groupby("Store")[col].transform(lambda s: s.ffill().bfill())

df["unique_id"] = df["Store"].astype(str) + "_Dept" + df["Dept"].astype(str)
df = df.rename(columns={"Weekly_Sales": "units_sold", "Date": "date"})
df = df[["unique_id", "Store", "Dept", "date", "units_sold", "promo_flag",
         "holiday_flag", "total_markdown", "Temperature", "Fuel_Price", "CPI",
         "Unemployment"]].sort_values(["unique_id", "date"]).reset_index(drop=True)

out_path = "data/retail_demand.csv"
df.to_csv(out_path, index=False)

print(f"Wrote {len(df):,} rows, {df['unique_id'].nunique()} series, "
      f"{df['date'].nunique()} weeks ({df['date'].min().date()} to {df['date'].max().date()})")
print(f"\nHoliday weeks in data: {df[df['holiday_flag']==1]['date'].nunique()} of {df['date'].nunique()}")
print(f"Weeks with any markdown/promo activity: {df[df['promo_flag']==1]['date'].nunique()} "
      f"(markdown data only exists from Nov 2011 onward)")
print(f"\nPer-series weekly sales summary:")
print(df.groupby("unique_id")["units_sold"].agg(["mean", "std"]).round(0))
