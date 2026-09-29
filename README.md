# Demand Forecasting & Inventory Optimization — Walmart Store-Department Network

A weekend project translating real Walmart store-department demand forecasts
into safety stock and reorder point recommendations — the forecasting →
inventory-policy translation that sits at the center of S&OP, and the piece
pure dashboard/KPI work doesn't cover.

**Live dashboard:** interactive artifact — forecast explorer, model
comparison, inventory plan table, a network-wide safety-stock comparison
across all three approaches, an accuracy-vs-consistency scatter plot (all
4 models × 15 series), the full 60-row backtest comparison table
(searchable/sortable), and a genuine 12-month-ahead forecast per series
with an uncertainty band.
**Documentation (this file), code, and data:** bundled for reproduction.

---

## 1. Problem statement

Given historical weekly demand for a network of store-department combinations,
produce a forecast accurate enough to drive a replenishment decision, then
convert it into two numbers a planner acts on: **safety stock** (buffer
inventory for a target service level) and **reorder point** (the trigger
level for a new order). Evaluated the way a real S&OP cycle is: a fixed
holdout window, a fixed lead-time assumption, a fixed service-level target,
scored against actuals the model never saw during training.

## 2. Data

**Source: real data.** [Walmart Recruiting – Store Sales Forecasting](https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting)
(Kaggle, 2014). 421,570 weekly Store × Dept sales records across 45 stores
and 81 departments, Feb 2010–Oct 2012, with real store-level covariates:
temperature, fuel price, CPI, unemployment, and five markdown/promotion
series. This replaces an earlier version of this project that used
synthetic data — noted here for transparency, and because the switch itself
surfaced a real methodological lesson (Section 3.4).

**Scope:** the full dataset is up to 3,645 series — more than a weekend
allows to backtest four models against and still read the results. Scoped
to the **5 highest-total-sales stores × 3 highest-total-sales, fully
populated departments = 15 series**, each confirmed to have complete,
gap-free 143-week history (no interpolation needed). This is a real,
disclosed simplification, not a hidden one: an analyst with limited time
prioritizes the highest-revenue-impact combinations first, and that's the
selection logic used here.

**Data quality handling** (the part synthetic data doesn't require):
`MarkDown1–5` are only populated from Nov 2011 onward per the competition
documentation — NaN before that date means "no markdown program existed
yet," not missing data, so it's filled with 0 rather than imputed. This
was verified against the data directly: markdown activity appears in
exactly 51 of the 143 weeks, matching the Nov 2011–Oct 2012 window the
documentation describes.

## 3. Methodology

### 3.1 Models — what, and why, in this order

1. **Naive (lag-1)** — "same as last week." The floor.
2. **Seasonal naive (lag-52)** — "same week last year." At weekly grain,
   the dominant seasonal driver is annual (Thanksgiving, Christmas,
   back-to-school), not weekly, so this is the more theoretically
   appropriate naive baseline — reported separately from lag-1 rather than
   assumed superior.
3. **Holt-Winters (ETS)**, fit independently **per series** (local) — trend
   + additive weekly seasonality, damped trend. Fully interpretable;
   decomposes into level/trend/season.
4. **LightGBM**, fit **globally** — one model pooled across all 15 series,
   with store and department as categorical features plus lag/rolling
   demand history and the real exogenous covariates (holiday flag,
   markdown total, temperature, fuel price, CPI, unemployment). Pooling
   lets the model borrow signal across series — the same reasoning behind
   why global tree-based models won the actual M5 forecasting competition
   rather than thousands of independent local models. This is a deliberate
   local-vs-global contrast against ETS, not just "add more models."

### 3.2 Evaluation

- **Holdout:** last 12 weeks per series (~one quarter), trained only on
  data before the cutoff.
- **Metric: WMAE**, defined exactly as the real Kaggle competition defined
  it — weight = 5 on holiday weeks, 1 otherwise. Adopted deliberately
  instead of a self-chosen metric: the business cost of a miss isn't
  uniform across weeks (a miss the week of a major sales event matters more
  than a miss in a quiet week), and using the metric the actual business
  problem was scored on is more defensible than inventing one.
- Caveat, disclosed rather than hidden: only one holiday week (Labor Day,
  Sept 2012) falls inside this particular 12-week holdout — the dataset
  ends before Thanksgiving/Christmas 2012 — so the holiday weighting had
  limited opportunity to differentiate models in this specific backtest
  window.

**Results** (mean across 15 series):

| Model | WMAE | MAE (units) | Bias |
|---|---|---|---|
| **Holt-Winters (ETS)** | **6,626** | 6,765 | +11 |
| LightGBM (global) | 7,617 | 8,239 | +4,780 |
| Naive (lag-1) | 9,083 | 9,162 | −3,571 |
| Seasonal Naive (lag-52) | 13,176 | 12,770 | +1,984 |

ETS won. Two results worth calling out rather than smoothing over:

- **The global LightGBM model did not beat ETS.** It beat both naive
  baselines, but its own bias (+4,780) suggests the recursive 12-step
  forecast drifted upward — plausible with only ~1,900 pooled training rows
  across 15 series with quite different scale, and no damping mechanism to
  temper long-horizon extrapolation the way ETS's damped trend does. Worth
  investigating further, not a result to paper over.
- **Seasonal naive (lag-52) was the worst model**, despite weekly retail
  data usually having strong annual seasonality. Most likely explanation:
  real year-over-year growth in this network during 2010–2012 (post-recession
  recovery) means "same week last year" undershoots current demand levels —
  lag-1, despite ignoring seasonality entirely, tracks the current level
  better than a stale year-old comparison point does.

**Holt-Winters (ETS) was selected as the production model.**

### 3.3 Inventory translation

```
Safety Stock (SS)   = z · σₑ · √L
Reorder Point (ROP) = (avg weekly demand · L) + SS
```

- **z = 1.645** (95% target service level — a reasonable retail default; a
  pharma/medical-device planner would set this higher given a stockout's
  downstream cost).
- **σₑ** — backtest forecast-error standard deviation of the **selected
  model** (ETS), not demand variance itself.
- **L = 2 weeks** — assumed replenishment lead time (not in the dataset;
  stated explicitly as an assumption, not fitted).
- **Baseline for comparison: Naive (lag-1)**, the simplest possible
  forecast — so the safety-stock delta reflects the value of doing any real
  modeling at all, not just the gap between two sophisticated methods.

### 3.4 The finding that matters most in this project

Moving from naive to ETS cut WMAE by 27% and removed almost all forecast
bias. The resulting **safety-stock reduction**, however, is only **11.4%
network-wide** (290,267 → 257,070 total units across the 15 series) — far
smaller than the accuracy improvement alone would suggest, and for **6 of
the 15 series, ETS actually required *more* safety stock than naive**, not
less.

**Why:** safety stock scales with forecast-error *variance* (σₑ), not with
*bias*. Naive (lag-1)'s large WMAE penalty comes almost entirely from a
systematic negative bias — it consistently under-forecasts by ~3,571
units/week on average, which is bad for MAE but doesn't necessarily mean
noisy week-to-week errors. ETS corrects that bias almost completely, but for
several series its period-to-period forecast *variance* is actually higher
than naive's — e.g. for `13_Dept38`, naive's error std is 4,134 units vs.
ETS's 8,137. A model can have much better point accuracy and still need
*more* buffer stock, because point accuracy and forecast consistency are
different properties, and safety stock only cares about the second one.

This is the single most important thing to be able to explain about this
project in an interview: a lower-MAE forecast is not automatically a
lower-inventory forecast, and conflating the two is a common junior-analyst
mistake this project deliberately surfaces rather than avoids.

**Risk flagging:** each series ranked by forecast-error coefficient of
variation (σₑ / average weekly demand) into quartiles. Notably, all three
`Dept 38` series across different stores land in the top-two risk
quartiles — a specific, investigable pattern (worth checking whether Dept 38
has a structural trend break or a promotional dynamic ETS's damped-trend
assumption doesn't fit well), not just noise.

### 3.5 A selection-objective mismatch, found and fixed

Section 3.2 picked one "best" model for the whole network by lowest **WMAE**
(point accuracy). Section 3.3 then used that single model's backtest error
std to size every series' safety stock. But safety stock is driven by std
(consistency), not by WMAE or bias (accuracy) — those are different
statistical properties (Section 3.4) — so a model selected for accuracy
isn't necessarily the model that minimizes any given series' safety stock.

Checked directly: for each of the 15 series, does the model with the lowest
backtest **MAE** also have the lowest backtest **std**? They agree on only
**7 of 15 series (47%)**. For the other 8, the "most accurate" model and the
"most consistent" model are different models entirely.

**Fix:** rather than applying one network-wide model to every series,
`04_std_optimized_selection.py` selects the forecasting model **per series**
by lowest backtest error std — the property safety stock actually depends
on — and rebuilds the inventory plan on that basis.

| Approach | Total network safety stock |
|---|---|
| Naive (lag-1) for every series | 290,267 units |
| Single "best" model (ETS) for every series, selected by WMAE | 257,070 units |
| **Model selected per series by lowest std** | **207,170 units** |

Fixing the selection objective alone — no new modeling, just choosing among
the four models already built using the right criterion per series — cuts
an additional **19.4%** off the plan that was already using the "best"
model. Cumulatively, from the naive baseline to the corrected per-series
plan: **28.6% less network-wide safety stock**, at the same 95% service
level throughout.

The per-series winners aren't dominated by one model, either — `Naive
(lag-1)` wins the consistency criterion for 6 series (its earlier flaw, bias,
doesn't matter for this decision; its variance is what matters, and it's
often low), ETS for 4, seasonal naive for 3, and global LightGBM for 2. No
single model is "best" for this network — the right model is a per-series
question, and treating it as a single network-wide choice was the actual
error, not any individual model's performance.

## 4. Limitations & honest next steps

- **Department-level granularity, not item/SKU-level.** The Kaggle dataset
  only goes down to Store × Department × Week — it contains no individual
  item/UPC data at all (not a scoping choice made here; the competition
  organizers never released it). `Weekly_Sales` for a series like
  `13_Dept92` is hundreds of distinct items summed together, not any single
  product. This matters for how the results should be read: a real
  replenishment planner orders at the item level, not "Department 92 in
  general," so these safety stock and reorder point numbers describe
  department-wide buffer, not a cuttable purchase order for a specific SKU.
  It also likely **understates true variability** — demand across many
  items partially cancels out when summed into a department total, so
  item-level forecast error (and the safety stock it implies, as a % of
  demand) would almost certainly be higher than what's shown here.
- **Scoped to 15 of up to 3,645 possible series.** The selection criterion
  (highest historical volume) is defensible but means the plan doesn't cover
  long-tail departments, which often have the least regular demand and the
  most to gain from a real forecasting process.
- **Lead time is an assumption, not a fitted value** — the dataset doesn't
  contain real replenishment lead times.
- **Only one holiday week in the holdout window** limits how much this
  specific backtest can say about holiday-period performance, which is
  arguably when forecast accuracy matters most for this business.
- **Global LightGBM likely under-tuned relative to ETS** — worth trying
  quantile regression (to get an error-distribution estimate directly rather
  than backing into one from residuals), non-recursive direct multi-step
  forecasting, or more pooled training history to see whether the gap
  closes.
- **Normality assumption in the safety-stock formula** — real forecast-error
  distributions, especially for the higher-risk Dept 38 series, are likely
  not symmetric; empirical error quantiles would be more rigorous than a
  fixed z-score.
- **Per-series model selection (Section 3.5) still picks by std on the same
  12-week backtest used everywhere else** — with only 12 points per series,
  that std estimate is itself noisy (small-sample variance estimates are
  unstable), so the "winning" model per series should be treated as a
  reasonable choice, not a statistically certain one. A longer backtest
  window, or cross-validating over several rolling windows, would give more
  stable per-series picks.
- **No intermittent-demand method (e.g. Croston's method / TSB) was tried.**
  All four models here assume roughly continuous, non-zero weekly demand,
  which holds at department aggregation but wouldn't necessarily hold if this
  were pushed down to item level (Limitation #1) — many individual SKUs sell
  in sporadic bursts with long zero-demand stretches, a pattern standard
  forecasting methods handle poorly. Croston's method (or its TSB variant)
  is built specifically for that case and would be the right next step
  before any item-level extension of this project.
- **No hierarchical reconciliation across levels.** Each of the 15 series is
  forecast independently — there's no step that rolls store-level forecasts
  up to a region or network total and reconciles them so the levels agree
  (top-down, bottom-up, or optimal-reconciliation methods like MinT). A real
  S&OP process usually forecasts at multiple levels simultaneously and needs
  them to tie out; this project only ever operates at one level at a time.

### 3.6 A genuine 12-month-ahead forecast, and the assumption it requires

Everything above is a *backtest* — scoring a model against weeks that
already happened. The dashboard also includes a real forecast 52 weeks past
the end of the dataset (2012-10-26 → 2013-10-25), for each series, using
that series' own selection-corrected model refit on its **full** history
(no holdout this time — there's nothing left to hold out for).

Naive, Seasonal Naive, and Holt-Winters need no new assumptions to do this:
Naive holds its last value flat, Seasonal Naive's "same week last year"
lookback is still real observed data even 52 weeks out, and ETS just
extrapolates its own fitted trend/seasonal decomposition further.

**LightGBM is the exception, and it's a real problem, not a minor one.** It
was trained on real covariates — holiday timing, markdown activity,
temperature, fuel price, CPI, unemployment — and a genuine future forecast
needs *future* values of all of them, which don't exist yet. The fix used
here: every covariate for a future week is carried forward from **the same
calendar week one year earlier** (i.e., "assume this year's holidays,
promotions, and macro conditions look like last year's"). This is the exact
same logic Seasonal Naive already uses for demand itself, just extended to
the inputs LightGBM needs — not a new kind of assumption, the same one
applied more broadly. It's disclosed here and directly in the dashboard,
not hidden: a real deployment would pull an actual forward promotional
calendar and a macro forecast instead of a one-year-back proxy.

**Uncertainty band:** shown as forecast ± z·σₑ, reusing each series' own
backtest error std rather than fitting a new one — the same σₑ that sizes
that series' safety stock, so the band on the chart is literally the
demand variability that safety stock number is meant to absorb.

## 5. Reproducing this

```
pip install -r requirements.txt
python3 01_load_data.py                  # merges/cleans real train.csv + features.csv -> data/retail_demand.csv
python3 02_forecasting_models.py         # backtests 4 models, writes output/*.csv
python3 03_inventory_planning.py         # single-model (ETS) plan, writes output/inventory_plan.csv
python3 04_std_optimized_selection.py    # corrects the selection objective; writes output/inventory_plan_v2_std_optimized.csv
python3 05_dashboard_extensions.py       # clean per-series x per-model stats + genuine 12-month-ahead forecast, for the dashboard
```

All scripts use paths relative to the project root, so this runs as-is from
a fresh clone or unzip — no path edits needed. Verified end-to-end from a
clean copy before packaging.
