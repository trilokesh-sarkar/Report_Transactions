# =======================================================================
#  📊 KPI DASHBOARD MODULE — Import in app.py
# =======================================================================

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import altair as alt
import os, joblib

from utils.forecast_xgboost import (
    build_fixed_history_series,
    build_future_fixed_series,
    forecast_with_xgboost_bundle,
)


# =======================================================================
# 🔥 MINI SPARKLINE (embedded KPI chart)
# =======================================================================

def sparkline(data, color="#ffbf00"):
    """Generates tiny mini trend chart for KPI"""
    if len(data) < 2:
        return None

    df = data.reset_index(drop=True).rename(columns={data.name: "value"})

    return (
        alt.Chart(df.reset_index())
        .mark_line(size=2, interpolate="monotone", color=color)
        .encode(x="index:Q", y="value:Q")
        .properties(width=120, height=30)
    )


# =======================================================================
# 💰 FORMATTING HELPERS
# =======================================================================

def fmt_k(n):
    try:
        n = float(n)
    except:
        return n

    if abs(n) >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    elif abs(n) >= 1_000:
        return f"{n/1_000:.1f}K"
    return f"{n:,.0f}"

def rup(n):
    return f"₹{fmt_k(n)}"


# =======================================================================
# 💼 INCOME LOGIC
# =======================================================================

def calc_income(year_month: str) -> float:
    try:
        ym = pd.to_datetime(year_month, format="%Y-%m")
    except:
        return 0.0

    if ym == pd.Timestamp(2024, 10, 1):
        return 12000
    elif ym == pd.Timestamp(2024, 11, 1):
        return 14112
    elif ym >= pd.Timestamp(2026, 5, 1):
        return 32000
    elif ym >= pd.Timestamp(2024, 12, 1):
        return 24200
    return 0.0

# Backward compatibility helper (used in app.py)
def get_income(month):
    return calc_income(month)


# =======================================================================
# 🔮 CURRENT MONTH FORECAST (FROM DAILY ML MODEL)
# =======================================================================

def get_current_month_forecast(
    filtered,
    DAILY_MODEL_PATH="models/daily_forecast_model.pkl"
):
    if not os.path.exists(DAILY_MODEL_PATH):
        return None

    try:
        model = joblib.load(DAILY_MODEL_PATH)
    except Exception:
        # If the serialized model cannot be imported in the current runtime,
        # keep the dashboard usable and skip the optional KPI forecast.
        return None

    df = filtered.copy()
    df["period"] = pd.to_datetime(df["period"], errors="coerce").dt.normalize()
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)
    df = df.dropna(subset=["period"])

    daily = df.groupby("period")["amount"].sum().sort_index()

    if daily.empty:
        return None

    today = pd.Timestamp.today().normalize()
    end_of_month = today + pd.offsets.MonthEnd(0)

    try:
        if isinstance(model, dict) and "model" in model:
            fixed_history, fixed_context = build_fixed_history_series(df, "D")
            fixed_history = fixed_history.reindex(daily.index, fill_value=0.0)
            variable_daily = (daily - fixed_history).clip(lower=0.0)
            if daily.index.max() >= end_of_month:
                preds = pd.Series(dtype=float)
            else:
                horizon = (end_of_month - daily.index.max()).days
                variable_forecast = forecast_with_xgboost_bundle(
                    model, variable_daily, horizon
                )
                fixed_forecast = build_future_fixed_series(
                    variable_daily.index.max(), horizon, "D", fixed_context
                )
                predicted = variable_forecast.add(fixed_forecast, fill_value=0.0)
                preds = predicted[predicted.index >= today]
        else:
            future_dates = pd.date_range(today, end_of_month)
            future = pd.DataFrame({"period": future_dates})
            future["day"] = future["period"].dt.day
            future["dow"] = future["period"].dt.dayofweek
            future["month"] = future["period"].dt.month
            preds = pd.Series(model.predict(future[["day", "dow", "month"]]))
    except Exception:
        return None

    if preds.empty:
        remaining_forecast = 0.0
    else:
        # ---------------- SAFETY CAPS ----------------
        max_daily = daily.quantile(0.95)
        avg_daily = daily.mean()

        # Hard cap
        preds = preds.clip(0, max_daily)

        # Conservative sanity check
        if preds.mean() > avg_daily * 2:
            preds = preds.clip(0, avg_daily * 1.5)

        remaining_forecast = preds.sum()

    spent_so_far = df[
        df["period"].dt.to_period("M") == today.to_period("M")
    ]["amount"].sum()

    return {
        "spent_so_far": round(spent_so_far, 2),
        "remaining_forecast": round(remaining_forecast, 2),
        "forecast_total": round(spent_so_far + remaining_forecast, 2)
    }


# =======================================================================
#               🔥 MAIN KPI RENDER FUNCTION
# =======================================================================

def render_kpis(filtered: pd.DataFrame, df: pd.DataFrame, MONTHLY_BUDGET: float):

    if filtered is None or filtered.empty:
        st.warning("⚠ No data available for KPI dashboard.")
        return

    # =====================================================
    # 🔧 PREP
    # =====================================================
    f = filtered.copy()
    f["period"] = pd.to_datetime(f["period"], errors="coerce")
    f["amount"] = pd.to_numeric(f["amount"], errors="coerce").fillna(0.0)
    f["year_month"] = f["period"].dt.to_period("M").astype(str)

    today = pd.Timestamp.today().date()
    now_ts = pd.Timestamp.now()

    # =====================================================
    # 📊 GLOBAL METRICS (ALL TIME)
    # =====================================================
    total_spend = f["amount"].sum()
    today_spend = f[f["period"].dt.date == today]["amount"].sum()

    month_totals = f.groupby("year_month")["amount"].sum().sort_index()
    cat_sum = f.groupby("category")["amount"].sum().sort_values(ascending=False)

    # =====================================================
    # 📆 CURRENT MONTH (DATE-DRIVEN)
    # =====================================================
    current_month_start = now_ts.replace(day=1).normalize()
    current_month_end = current_month_start + pd.offsets.MonthEnd(0)
    current_month_key = current_month_start.strftime("%Y-%m")

    current_month_df = f[
        (f["period"] >= current_month_start) &
        (f["period"] <= current_month_end)
    ]

    current_month_spend = current_month_df["amount"].sum()

    # =====================================================
    # 💰 INCOME
    # =====================================================
    total_income = sum(calc_income(m) for m in month_totals.index)
    current_month_income = calc_income(current_month_key)

    pct_spent = (
        (current_month_spend / current_month_income) * 100
        if current_month_income > 0 else 0
    )

    # =====================================================
    # 💼 BUDGET LOGIC — MONTHLY RESET
    # =====================================================
    TOTAL_MONTHLY_BUDGET = MONTHLY_BUDGET

    days_total = pd.Period(now_ts, freq="M").days_in_month
    days_left = max(days_total - now_ts.day + 1, 1)

    budget_left = max(TOTAL_MONTHLY_BUDGET - current_month_spend, 0)
    daily_allowed_left = budget_left / days_left

    spend_velocity = (
        current_month_spend / max(now_ts.day, 1)
    )

    # =====================================================
    # 📈 TRENDS (MoM & WoW)
    # =====================================================
    mom = (
        ((month_totals.iloc[-1] - month_totals.iloc[-2]) / month_totals.iloc[-2] * 100)
        if len(month_totals) > 1 and month_totals.iloc[-2] > 0 else 0
    )

    f["year_week"] = f["period"].dt.strftime("%Y-W%U")
    weekly = f.groupby("year_week")["amount"].sum().sort_index()

    wow = (
        ((weekly.iloc[-1] - weekly.iloc[-2]) / weekly.iloc[-2] * 100)
        if len(weekly) > 1 and weekly.iloc[-2] > 0 else 0
    )

    # =====================================================
    # 🔮 FORECAST (CURRENT MONTH)
    # =====================================================
    forecast = get_current_month_forecast(df)

    # =====================================================
    # ========== ROW 1 — CORE KPIs ==========
    # =====================================================
    st.subheader("📊 Financial KPI Overview")
    a1, a2, a3, a4 = st.columns(4)

    a1.metric("💰 Total Income", rup(total_income))
    a2.metric("💸 Total Spend", rup(total_spend))
    a3.metric("🛒 Today Spend", rup(today_spend))
    a4.metric("⚡ % Spent (Income)", f"{pct_spent:.1f}%")

    # =====================================================
    # ========== ROW 2 — BUDGET HEALTH ==========
    # =====================================================
    st.markdown("### 💼 Monthly Budget Health")

    b1, b2, b3, b4, b5 = st.columns(5)

    b1.metric("💰 Budget Left", rup(budget_left))
    b2.metric("📅 Days Left", days_left)
    b3.metric("⚡ Daily Allowed", rup(daily_allowed_left))
    b4.metric("📆 Month Spend", rup(current_month_spend))
    b5.metric("🚀 Spend Velocity", rup(spend_velocity))

    # =====================================================
    # ========== ROW 3 — FORECAST KPIs ==========
    # =====================================================
    st.markdown("### 🔮 Forecast Outlook (AI)")

    f1, f2, f3 = st.columns(3)

    if forecast:
        f1.metric("🤖 Forecasted Month Spend", rup(forecast["forecast_total"]))
        f2.metric(
            "📊 Forecast vs Budget",
            rup(forecast["forecast_total"] - TOTAL_MONTHLY_BUDGET),
            delta="Over Budget"
            if forecast["forecast_total"] > TOTAL_MONTHLY_BUDGET
            else "Under Budget",
        )
        f3.metric(
            "⚡ Forecast Daily Avg (Remaining)",
            rup(forecast["remaining_forecast"] / days_left if days_left > 0 else 0),
        )
    else:
        f1.metric("🤖 Forecasted Month Spend", "—")
        f2.metric("📊 Forecast vs Budget", "—")
        f3.metric("⚡ Forecast Daily Avg", "—")

    # =====================================================
    # ========== ROW 4 — TRENDS ==========
    # =====================================================
    st.markdown("### 📈 Trends & Growth")

    t1, t2 = st.columns(2)
    t1.metric("📆 MoM Growth", f"{mom:.1f}%")
    t2.metric("🔄 WoW Change", f"{wow:.1f}%")

    # =====================================================
    # ========== ROW 5 — CATEGORY INSIGHTS ==========
    # =====================================================
    st.markdown("### 🏷 Category Insights")

    st.metric(
        "🏆 Highest Spend Category",
        cat_sum.idxmax() if not cat_sum.empty else "-",
    )

    # =====================================================
    # 📊 Category Spend Summary Table (With Current Month)
    # =====================================================

    # Base aggregation (all-time)
    share = cat_sum.reset_index().rename(columns={"amount": "Total Spend"})

    # Number of unique months in filtered data
    months_count = f["year_month"].nunique()

    # Average monthly spend per category
    share["Avg Monthly Spend"] = (
        share["Total Spend"] / months_count if months_count > 0 else 0
    )

    # Share percentage (all-time)
    share["Share %"] = (
        (share["Total Spend"] / total_spend * 100).round(2)
        if total_spend > 0 else 0
    )
    # =====================================================
    # 📊 Category Spend Summary Table (With Current Month)
    # =====================================================

    # Base aggregation (all-time)
    share = cat_sum.reset_index().rename(columns={"amount": "Total Spend"})

    # Number of unique months in filtered data
    months_count = f["year_month"].nunique()

    # Average monthly spend per category
    share["Avg Monthly Spend"] = (
        share["Total Spend"] / months_count if months_count > 0 else 0
    )

    # Share percentage (all-time)
    share["Share %"] = (
        (share["Total Spend"] / total_spend * 100).round(2)
        if total_spend > 0 else 0
    )

    # Current month spend per category
    current_month_cat = (
        current_month_df
        .groupby("category")["amount"]
        .sum()
        .reset_index(name="Current Month Spend")
    )

    share = share.merge(
        current_month_cat,
        on="category",
        how="left"
    )

    share["Current Month Spend"] = share["Current Month Spend"].fillna(0)

    # Formatting
    share["Total Spend"] = share["Total Spend"].apply(rup)
    share["Avg Monthly Spend"] = share["Avg Monthly Spend"].apply(rup)
    share["Current Month Spend"] = share["Current Month Spend"].apply(rup)

    # 🔥 REORDER COLUMNS (Share % LAST)
    share = share[
        [
            "category",
            "Total Spend",
            "Avg Monthly Spend",
            "Current Month Spend",
            "Share %"
        ]
    ]

    # Sort by impact
    share = share.sort_values("Share %", ascending=False)

    # Display
    st.dataframe(
        share,
        use_container_width=True,
        hide_index=True
    )

    st.success("KPI Dashboard Loaded ✅")
