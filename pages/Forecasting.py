import os

import altair as alt
import pandas as pd
import streamlit as st

from utils.forecast_xgboost import (
    build_fixed_history_series,
    build_future_fixed_series,
    evaluate_latest_holdout,
    forecast_with_xgboost_bundle,
    load_saved_bundle,
    train_xgboost_bundle,
)
from utils.github_storage import read_csv


css_path = ".streamlit/styles.css"
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.set_page_config(layout="wide")
st.title("AI Forecasting Intelligence")


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


def render_model_metrics(result, unit_label):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Model", result["model"])
    c2.metric(
        "Holdout MAE",
        "N/A" if result["mae"] is None else f"INR {result['mae']:,.0f}",
    )
    c3.metric(
        "Validation Window",
        "N/A" if result["holdout"] == 0 else f"{result['holdout']} {unit_label}",
    )
    c4.metric(
        "Vs Baseline",
        "N/A" if result["improvement_pct"] is None else f"{result['improvement_pct']:.1f}%",
    )


df = read_csv()
df["period"] = pd.to_datetime(df["period"])
df = df.sort_values("period", ascending=False).reset_index(drop=True)

st.markdown("### Filters")

c1, c2 = st.columns(2)

with c1:
    f_year = st.multiselect("Year", sorted(df.year.unique()))
    f_acc = st.multiselect("Account", sorted(df.accounts.unique()))

with c2:
    f_month = st.multiselect("Month", sorted(df.year_month.unique()))
    include_cat = st.multiselect("Include Category", sorted(df.category.unique()))
    exclude_cat = st.multiselect("Exclude Category", sorted(df.category.unique()))

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

using_full_history = not any([f_year, f_acc, f_month, include_cat, exclude_cat])

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

st.markdown("### Recent Spending Trend")
monthly_preview = (
    filtered.groupby("year_month")["amount"]
    .sum()
    .sort_index()
)
st.line_chart(monthly_preview.tail(6))

st.markdown("### Category Contribution")
cat_share = (
    filtered.groupby("category")["amount"]
    .sum()
    .sort_values(ascending=False)
)
st.bar_chart(cat_share)

st.markdown("### Forecast Prediction (XGBoost)")
st.caption(
    "When no filters are applied, this page uses the saved repo XGBoost models trained locally on the latest data. "
    "If you filter the dataset, it retrains XGBoost on that history when enough completed data is available; otherwise it uses the full completed history."
)
st.caption(
    "Fixed costs are handled separately before training: Rent contributes INR 4,000 per month and bike_emi contributes INR 5,333 per month. "
    "XGBoost predicts the remaining variable spend, then the fixed amounts are added back into the final forecast."
)

fc1, fc2 = st.columns(2)
with fc1:
    daily_horizon = st.slider("Daily Forecast Horizon", min_value=7, max_value=90, value=30, step=1)
with fc2:
    monthly_horizon = st.slider("Monthly Forecast Horizon", min_value=3, max_value=12, value=6, step=1)

daily_series = (
    filtered.set_index("period")
    .sort_index()["amount"]
    .groupby(pd.Grouper(freq="D"))
    .sum()
    .asfreq("D", fill_value=0.0)
)

monthly_series = (
    filtered.set_index("period")
    .sort_index()["amount"]
    .groupby(pd.Grouper(freq="MS"))
    .sum()
    .asfreq("MS", fill_value=0.0)
)

fixed_daily_history, fixed_daily_context = build_fixed_history_series(filtered, "D")
fixed_monthly_history, fixed_monthly_context = build_fixed_history_series(filtered, "MS")

fixed_daily_history = fixed_daily_history.reindex(daily_series.index, fill_value=0.0)
fixed_monthly_history = fixed_monthly_history.reindex(monthly_series.index, fill_value=0.0)

variable_daily_series = (daily_series - fixed_daily_history).clip(lower=0.0)
variable_monthly_series = (monthly_series - fixed_monthly_history).clip(lower=0.0)

current_month_start = pd.Timestamp.today().to_period("M").start_time
training_daily_series = variable_daily_series[
    variable_daily_series.index < current_month_start
]
training_monthly_series = variable_monthly_series[
    variable_monthly_series.index < current_month_start
]

full_daily_series = (
    df.set_index("period")
    .sort_index()["amount"]
    .groupby(pd.Grouper(freq="D"))
    .sum()
    .asfreq("D", fill_value=0.0)
)
full_monthly_series = (
    df.set_index("period")
    .sort_index()["amount"]
    .groupby(pd.Grouper(freq="MS"))
    .sum()
    .asfreq("MS", fill_value=0.0)
)
full_daily_fixed, _ = build_fixed_history_series(df, "D")
full_monthly_fixed, _ = build_fixed_history_series(df, "MS")
full_daily_fixed = full_daily_fixed.reindex(full_daily_series.index, fill_value=0.0)
full_monthly_fixed = full_monthly_fixed.reindex(full_monthly_series.index, fill_value=0.0)
full_training_daily_series = (full_daily_series - full_daily_fixed).clip(lower=0.0)
full_training_monthly_series = (full_monthly_series - full_monthly_fixed).clip(lower=0.0)
full_training_daily_series = full_training_daily_series[
    full_training_daily_series.index < current_month_start
]
full_training_monthly_series = full_training_monthly_series[
    full_training_monthly_series.index < current_month_start
]

daily_model_series = (
    training_daily_series
    if len(training_daily_series) >= 36
    else full_training_daily_series
)
monthly_model_series = (
    training_monthly_series
    if len(training_monthly_series) >= 20
    else full_training_monthly_series
)

st.markdown("#### Daily Forecast")
if len(daily_model_series) < 36:
    st.warning("Need at least 36 days of data for the daily XGBoost forecaster.")
else:
    daily_bundle = load_saved_bundle("D") if using_full_history else None
    if daily_bundle is None:
        daily_bundle = train_xgboost_bundle(daily_model_series, "D")
    daily_result = evaluate_latest_holdout(daily_model_series, "D")
    daily_variable_forecast = forecast_with_xgboost_bundle(daily_bundle, variable_daily_series, daily_horizon)
    daily_fixed_future = build_future_fixed_series(variable_daily_series.index.max(), daily_horizon, "D", fixed_daily_context)
    daily_forecast = (daily_variable_forecast.add(daily_fixed_future, fill_value=0.0)).rename("forecast")

    render_model_metrics(daily_result, "days")
    render_forecast_chart(daily_series, daily_forecast, "Date")

    daily_table = daily_forecast.reset_index()
    daily_table.columns = ["Date", "Forecast"]
    daily_table["Forecast"] = daily_table["Forecast"].round(2)
    st.dataframe(daily_table, use_container_width=True)

st.markdown("#### Monthly Forecast")
if len(monthly_model_series) < 20:
    st.warning("Need at least 20 months of data for the monthly XGBoost forecaster.")
else:
    monthly_bundle = load_saved_bundle("MS") if using_full_history else None
    if monthly_bundle is None:
        monthly_bundle = train_xgboost_bundle(monthly_model_series, "MS")
    monthly_result = evaluate_latest_holdout(monthly_model_series, "MS")
    monthly_variable_forecast = forecast_with_xgboost_bundle(monthly_bundle, monthly_model_series, monthly_horizon)
    monthly_fixed_future = build_future_fixed_series(monthly_model_series.index.max(), monthly_horizon, "MS", fixed_monthly_context)
    monthly_forecast = (monthly_variable_forecast.add(monthly_fixed_future, fill_value=0.0)).rename("forecast")

    render_model_metrics(monthly_result, "months")
    render_forecast_chart(monthly_series, monthly_forecast, "Month")

    monthly_table = monthly_forecast.reset_index()
    monthly_table.columns = ["Month", "Forecast"]
    monthly_table["Forecast"] = monthly_table["Forecast"].round(2)
    st.dataframe(monthly_table, use_container_width=True)

    projected_total = monthly_forecast.sum()
    st.metric("Projected Spend (Forecast Window)", f"INR {projected_total:,.0f}")
