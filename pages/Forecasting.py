import os
import warnings
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from utils.github_storage import read_csv


# -----------------------------------------------------------
# LOAD CSS
# -----------------------------------------------------------
css_path = ".streamlit/styles.css"
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.set_page_config(layout="wide")
st.title("AI Forecasting Intelligence")


# -----------------------------------------------------------
# FORECAST HELPERS
# -----------------------------------------------------------
def mean_absolute_error(actual, predicted):
    actual_arr = np.asarray(actual, dtype=float)
    pred_arr = np.asarray(predicted, dtype=float)
    return float(np.mean(np.abs(actual_arr - pred_arr)))


def clip_forecast(values, upper_bound):
    arr = np.asarray(values, dtype=float)
    return np.clip(arr, 0, max(float(upper_bound), 0.0))


def seasonal_naive_forecast(train_series, horizon, season_length):
    values = train_series.to_numpy(dtype=float)

    if len(values) == 0:
        return np.zeros(horizon)

    if len(values) < season_length:
        return np.full(horizon, values[-1])

    last_season = values[-season_length:]
    repeats = int(np.ceil(horizon / season_length))
    return np.tile(last_season, repeats)[:horizon]


def moving_average_forecast(train_series, horizon, window, weighted=False):
    values = train_series.tail(min(window, len(train_series))).to_numpy(dtype=float)

    if len(values) == 0:
        base_value = 0.0
    elif weighted and len(values) > 1:
        weights = np.arange(1, len(values) + 1, dtype=float)
        base_value = float(np.average(values, weights=weights))
    else:
        base_value = float(values.mean())

    return np.full(horizon, max(base_value, 0.0), dtype=float)


def linear_trend_forecast(train_series, horizon, window):
    values = train_series.tail(min(window, len(train_series))).to_numpy(dtype=float)

    if len(values) == 0:
        return np.zeros(horizon)
    if len(values) == 1:
        return np.full(horizon, max(float(values[-1]), 0.0), dtype=float)

    x = np.arange(len(values), dtype=float)
    slope, intercept = np.polyfit(x, values, 1)
    future_x = np.arange(len(values), len(values) + horizon, dtype=float)
    forecast = slope * future_x + intercept

    return np.clip(forecast, 0, None)


def robust_holt_winters_forecast(train_series, horizon, season_length):
    series = train_series.astype(float)

    if series.empty:
        return np.zeros(horizon)

    capped = series.clip(lower=0.0)
    if len(capped) >= 30:
        capped = capped.clip(upper=float(capped.quantile(0.95)))

    transformed = np.log1p(capped)
    trend = "add" if len(transformed) >= 4 else None
    use_seasonality = len(series) >= season_length * 2
    seasonal = "add" if use_seasonality else None

    model_kwargs = {
        "trend": trend,
        "seasonal": seasonal,
        "initialization_method": "estimated",
    }

    if trend:
        model_kwargs["damped_trend"] = True

    if seasonal:
        model_kwargs["seasonal_periods"] = season_length

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = ExponentialSmoothing(transformed, **model_kwargs)
            fitted = model.fit(optimized=True, use_brute=False)

        forecast = np.expm1(np.asarray(fitted.forecast(horizon), dtype=float))
        upper_bound = float(series.tail(min(max(season_length * 4, 28), len(series))).quantile(0.98))
        if not np.isfinite(upper_bound) or upper_bound <= 0:
            upper_bound = float(series.clip(lower=0.0).max())

        return clip_forecast(forecast, upper_bound)
    except Exception:
        return moving_average_forecast(series, horizon, window=min(max(season_length, 7), len(series)))


def monthly_hybrid_forecast(train_series, horizon):
    weighted_recent = moving_average_forecast(train_series, horizon, window=3, weighted=True)
    medium_term = moving_average_forecast(train_series, horizon, window=6, weighted=False)
    trend = linear_trend_forecast(train_series, horizon, window=6)

    forecast = (0.50 * weighted_recent) + (0.35 * medium_term) + (0.15 * trend)
    upper_bound = float(train_series.tail(min(12, len(train_series))).quantile(0.98))
    if not np.isfinite(upper_bound) or upper_bound <= 0:
        upper_bound = float(train_series.clip(lower=0.0).max())

    return clip_forecast(forecast, upper_bound)


def build_future_index(last_timestamp, periods, freq):
    offset = pd.tseries.frequencies.to_offset(freq)
    start = last_timestamp + offset
    return pd.date_range(start=start, periods=periods, freq=freq)


def evaluate_candidate_windows(series, candidate_fn, validation_horizon, min_train_size, step=1):
    errors = []

    last_end = len(series) - validation_horizon + 1
    for end in range(min_train_size, last_end, step):
        train = series.iloc[:end]
        test = series.iloc[end : end + validation_horizon]

        if len(test) < validation_horizon:
            continue

        preds = candidate_fn(train, validation_horizon)
        errors.append(mean_absolute_error(test, preds))

    return errors


def select_best_forecast_model(series, horizon, freq):
    if freq == "D":
        validation_horizon = min(14, max(7, horizon))
        min_train_size = max(56, validation_horizon * 4)
        step = 7
        season_length = 7
        candidates = {
            "Robust Holt-Winters": lambda train, h: robust_holt_winters_forecast(train, h, season_length),
            "Seasonal Naive": lambda train, h: seasonal_naive_forecast(train, h, season_length),
            "Weighted 28-Day Average": lambda train, h: moving_average_forecast(train, h, window=28, weighted=True),
        }
        baseline_name = "Seasonal Naive"
    else:
        validation_horizon = min(3, horizon)
        min_train_size = max(9, validation_horizon * 3)
        step = 1
        season_length = 12
        candidates = {
            "Hybrid Recent-Level": lambda train, h: monthly_hybrid_forecast(train, h),
            "Weighted 3-Month Average": lambda train, h: moving_average_forecast(train, h, window=3, weighted=True),
            "Seasonal Naive": lambda train, h: seasonal_naive_forecast(train, h, season_length),
        }
        baseline_name = "Weighted 3-Month Average"

    baseline_errors = evaluate_candidate_windows(
        series,
        candidates[baseline_name],
        validation_horizon=validation_horizon,
        min_train_size=min_train_size,
        step=step,
    )

    scored_candidates = []
    for model_name, candidate_fn in candidates.items():
        candidate_errors = evaluate_candidate_windows(
            series,
            candidate_fn,
            validation_horizon=validation_horizon,
            min_train_size=min_train_size,
            step=step,
        )
        mae = float(np.mean(candidate_errors)) if candidate_errors else None
        scored_candidates.append((model_name, candidate_fn, mae, len(candidate_errors)))

    valid_candidates = [row for row in scored_candidates if row[2] is not None]

    if not valid_candidates:
        final_pred = candidates[baseline_name](series, horizon)
        future_idx = build_future_index(series.index.max(), horizon, freq)
        forecast = pd.Series(np.clip(final_pred, 0, None), index=future_idx, name="forecast")

        return {
            "model": baseline_name,
            "holdout": 0,
            "mae": None,
            "baseline_mae": None,
            "improvement_pct": None,
            "forecast": forecast,
        }

    selected, selected_fn, selected_mae, windows = min(valid_candidates, key=lambda row: row[2])
    baseline_mae = float(np.mean(baseline_errors)) if baseline_errors else None
    improvement_pct = None
    if baseline_mae and selected_mae is not None and baseline_mae > 0:
        improvement_pct = ((baseline_mae - selected_mae) / baseline_mae) * 100.0

    final_pred = selected_fn(series, horizon)

    future_idx = build_future_index(series.index.max(), horizon, freq)
    forecast = pd.Series(np.clip(final_pred, 0, None), index=future_idx, name="forecast")

    return {
        "model": selected,
        "holdout": validation_horizon,
        "windows": windows,
        "mae": selected_mae,
        "baseline_mae": baseline_mae,
        "improvement_pct": improvement_pct,
        "forecast": forecast,
    }


def render_forecast_chart(history_series, forecast_series, x_title):
    hist_df = history_series.reset_index()
    hist_df.columns = ["Date", "Actual"]

    fc_df = forecast_series.reset_index()
    fc_df.columns = ["Date", "Forecast"]

    history_line = (
        alt.Chart(hist_df)
        .mark_line(color="#4FC3F7", point=True)
        .encode(
            x=alt.X("Date:T", title=x_title),
            y=alt.Y("Actual:Q", title="Amount"),
            tooltip=["Date:T", alt.Tooltip("Actual:Q", format=",.2f")],
        )
    )

    forecast_line = (
        alt.Chart(fc_df)
        .mark_line(color="#FFC107", point=True, strokeDash=[5, 4])
        .encode(
            x=alt.X("Date:T", title=x_title),
            y=alt.Y("Forecast:Q", title="Amount"),
            tooltip=["Date:T", alt.Tooltip("Forecast:Q", format=",.2f")],
        )
    )

    st.altair_chart(history_line + forecast_line, use_container_width=True)


# -----------------------------------------------------------
# LOAD DATA
# -----------------------------------------------------------
df = read_csv()
df["period"] = pd.to_datetime(df["period"])
df = df.sort_values("period", ascending=False).reset_index(drop=True)


# -----------------------------------------------------------
# COMPACT SIDEBAR FILTERS
# -----------------------------------------------------------
st.sidebar.markdown("### Filters")

c1, c2 = st.sidebar.columns(2)

with c1:
    f_year = st.multiselect("Year", sorted(df.year.unique()))
    f_acc = st.multiselect("Account", sorted(df.accounts.unique()))

with c2:
    f_month = st.multiselect("Month", sorted(df.year_month.unique()))
    include_cat = st.multiselect("Include Category", sorted(df.category.unique()))
    exclude_cat = st.multiselect("Exclude Category", sorted(df.category.unique()))


# -----------------------------------------------------------
# APPLY FILTERS
# -----------------------------------------------------------
filtered = df.copy()

if f_year:
    filtered = filtered[filtered.year.isin(f_year)]
if f_month:
    filtered = filtered[filtered.year_month.isin(f_month)]
if include_cat:
    filtered = filtered[filtered.category.isin(include_cat)]
if exclude_cat:
    filtered = filtered[~filtered.category.isin(exclude_cat)]
if f_acc:
    filtered = filtered[filtered.accounts.isin(f_acc)]

if filtered.empty:
    st.warning("No data available for forecasting.")
    st.stop()


# -----------------------------------------------------------
# CONTEXT KPIS (WHAT MODEL SEES)
# -----------------------------------------------------------
st.markdown("### Dataset Snapshot (Model Input View)")

total = filtered["amount"].sum()
months = filtered["year_month"].nunique()
avg_month = total / months if months else 0
txns = len(filtered)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Spend", f"INR {total:,.0f}")
k2.metric("Months Covered", months)
k3.metric("Avg / Month", f"INR {avg_month:,.0f}")
k4.metric("Transactions", txns)


# -----------------------------------------------------------
# RECENT TREND (LAST 6 MONTHS)
# -----------------------------------------------------------
st.markdown("### Recent Spending Trend")

monthly_preview = (
    filtered.groupby("year_month")["amount"]
    .sum()
    .sort_index()
)

st.line_chart(monthly_preview.tail(6))


# -----------------------------------------------------------
# CATEGORY CONTRIBUTION
# -----------------------------------------------------------
st.markdown("### Category Contribution")

cat_share = (
    filtered.groupby("category")["amount"]
    .sum()
    .sort_values(ascending=False)
)

st.bar_chart(cat_share)


# -----------------------------------------------------------
# FORECAST SECTION
# -----------------------------------------------------------
st.markdown("### Forecast Prediction (Backtested Time-Series)")
st.caption(
    "Forecasts are selected by rolling backtests. Daily uses a robust outlier-capped Holt-Winters option, while monthly favors recent-level models for short histories."
)

fc1, fc2 = st.columns(2)
with fc1:
    daily_horizon = st.slider("Daily Forecast Horizon", min_value=7, max_value=90, value=30, step=1)
with fc2:
    monthly_horizon = st.slider("Monthly Forecast Horizon", min_value=3, max_value=12, value=6, step=1)

# Daily series with full day continuity (missing days treated as zero spend)
daily_series = (
    filtered.set_index("period")
    .sort_index()["amount"]
    .groupby(pd.Grouper(freq="D"))
    .sum()
    .asfreq("D", fill_value=0.0)
)

# Monthly series with continuous monthly index
monthly_series = (
    filtered.set_index("period")
    .sort_index()["amount"]
    .groupby(pd.Grouper(freq="MS"))
    .sum()
    .asfreq("MS", fill_value=0.0)
)

st.markdown("#### Daily Forecast")
if len(daily_series) < 14:
    st.warning("Need at least 14 days of data for a reliable daily forecast.")
else:
    daily_result = select_best_forecast_model(
        series=daily_series,
        horizon=daily_horizon,
        freq="D",
    )

    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Selected Model", daily_result["model"])
    d2.metric(
        "Holdout MAE",
        "N/A" if daily_result["mae"] is None else f"INR {daily_result['mae']:,.0f}",
    )
    d3.metric("Validation Window", "N/A" if daily_result["holdout"] == 0 else f"{daily_result['holdout']} days")
    d4.metric(
        "Vs Baseline",
        "N/A" if daily_result["improvement_pct"] is None else f"{daily_result['improvement_pct']:.1f}%",
    )

    render_forecast_chart(daily_series, daily_result["forecast"], "Date")

    daily_table = daily_result["forecast"].reset_index()
    daily_table.columns = ["Date", "Forecast"]
    daily_table["Forecast"] = daily_table["Forecast"].round(2)
    st.dataframe(daily_table, use_container_width=True)

st.markdown("#### Monthly Forecast")
if len(monthly_series) < 3:
    st.warning("Need at least 3 months of data for a monthly forecast.")
else:
    monthly_result = select_best_forecast_model(
        series=monthly_series,
        horizon=monthly_horizon,
        freq="MS",
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Selected Model", monthly_result["model"])
    m2.metric(
        "Holdout MAE",
        "N/A" if monthly_result["mae"] is None else f"INR {monthly_result['mae']:,.0f}",
    )
    m3.metric(
        "Validation Window",
        "N/A" if monthly_result["holdout"] == 0 else f"{monthly_result['holdout']} months",
    )
    m4.metric(
        "Vs Baseline",
        "N/A" if monthly_result["improvement_pct"] is None else f"{monthly_result['improvement_pct']:.1f}%",
    )

    render_forecast_chart(monthly_series, monthly_result["forecast"], "Month")

    monthly_table = monthly_result["forecast"].reset_index()
    monthly_table.columns = ["Month", "Forecast"]
    monthly_table["Forecast"] = monthly_table["Forecast"].round(2)
    st.dataframe(monthly_table, use_container_width=True)

    projected_total = monthly_result["forecast"].sum()
    st.metric("Projected Spend (Forecast Window)", f"INR {projected_total:,.0f}")
