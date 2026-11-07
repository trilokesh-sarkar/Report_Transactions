import os

import pandas as pd
import streamlit as st
from utils.filter_state import persistent_multiselect, reset_filter_button

from utils.github_storage import read_csv
from utils.kpi_dashboard import get_income


def fmt_currency(value: float) -> str:
    return f"INR {value:,.0f}"


def fmt_month(ts: pd.Timestamp) -> str:
    return ts.strftime("%b %Y")


# -----------------------------------------------------------
# PAGE CONFIG + CSS
# -----------------------------------------------------------
st.set_page_config(layout="wide")

css_path = ".streamlit/styles.css"
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title("Monthly Deep Dive - Smart Analytics")


# -----------------------------------------------------------
# LOAD DATA
# -----------------------------------------------------------
df = read_csv()
df["period"] = pd.to_datetime(df["period"])
df["year_month"] = df["period"].dt.to_period("M").astype(str)


# -----------------------------------------------------------
# SIDEBAR FILTERS
# -----------------------------------------------------------
st.sidebar.markdown("### Filters")

c1, c2 = st.sidebar.columns(2)

with c1:
    f_year = persistent_multiselect("Year", sorted(df.year.unique()), "years")
    f_acc = persistent_multiselect("Account", sorted(df.accounts.unique()), "accounts")

with c2:
    f_month = persistent_multiselect("Month", sorted(df.year_month.unique()), "months")
    include_cat = persistent_multiselect(
        "Include Category",
        sorted(df.category.unique()),
        "include_categories",
        placeholder="Select categories...",
    )
    exclude_cat = persistent_multiselect(
        "Exclude Category",
        sorted(df.category.unique()),
        "exclude_categories",
        placeholder="Select categories...",
    )

reset_filter_button()

st.sidebar.markdown("### Savings Settings")
income_mode = st.sidebar.radio(
    "Income Source",
    ["Use mapped income", "Use custom monthly income"],
    index=0,
)

custom_monthly_income = None
if income_mode == "Use custom monthly income":
    custom_monthly_income = st.sidebar.number_input(
        "Monthly Income",
        min_value=0.0,
        value=25000.0,
        step=500.0,
    )


# -----------------------------------------------------------
# APPLY FILTERS
# -----------------------------------------------------------
filtered = df.copy()

if f_year:
    filtered = filtered[filtered.year.isin(f_year)]
if f_month:
    filtered = filtered[filtered.year_month.isin(f_month)]
if f_acc:
    filtered = filtered[filtered.accounts.isin(f_acc)]
if include_cat:
    filtered = filtered[filtered.category.isin(include_cat)]
if exclude_cat:
    filtered = filtered[~filtered.category.isin(exclude_cat)]

if filtered.empty:
    st.warning("No data available after applying filters.")
    st.stop()


# -----------------------------------------------------------
# AGGREGATIONS
# -----------------------------------------------------------
monthly_df = (
    filtered.groupby("year_month", as_index=False)["amount"]
    .sum()
    .rename(columns={"amount": "spend"})
)
monthly_df["year_month"] = pd.to_datetime(monthly_df["year_month"])
monthly_df = monthly_df.sort_values("year_month").reset_index(drop=True)

monthly = monthly_df.set_index("year_month")["spend"]
total_spend = float(monthly.sum())
months_count = len(monthly)

best_month_ts = monthly.idxmin()
worst_month_ts = monthly.idxmax()
best_month_value = float(monthly.min())
worst_month_value = float(monthly.max())

monthly_avg = float(monthly.mean())
volatility = float(monthly.std()) if months_count > 1 else 0.0
consistency = max(0.0, 100 - ((volatility / monthly_avg) * 100)) if monthly_avg > 0 else 0.0

latest_spend = float(monthly.iloc[-1])
prev_spend = float(monthly.iloc[-2]) if months_count > 1 else None

if prev_spend and prev_spend != 0:
    mom_growth_pct = ((latest_spend - prev_spend) / prev_spend) * 100
else:
    mom_growth_pct = None


# Category insights
cat_share = filtered.groupby("category")["amount"].sum().sort_values(ascending=False)
top_cat_name = cat_share.index[0]
top_cat_spend = float(cat_share.iloc[0])
top_cat_share_pct = (top_cat_spend / total_spend * 100) if total_spend > 0 else 0.0
top3_share_pct = (cat_share.head(3).sum() / total_spend * 100) if total_spend > 0 else 0.0
active_categories = int(cat_share.size)

cat_monthly = (
    filtered.groupby(["year_month", "category"], as_index=False)["amount"]
    .sum()
    .rename(columns={"amount": "spend"})
)
cat_monthly["year_month"] = pd.to_datetime(cat_monthly["year_month"])
cat_pivot = cat_monthly.pivot_table(
    index="year_month",
    columns="category",
    values="spend",
    fill_value=0.0,
)

most_volatile_cat = "N/A"
most_stable_cat = "N/A"

if len(cat_pivot) >= 2:
    cat_means = cat_pivot.mean()
    valid_cols = cat_means[cat_means > 0].index.tolist()

    if valid_cols:
        cat_cv = (cat_pivot[valid_cols].std() / cat_means[valid_cols]).sort_values(ascending=False)
        most_volatile_cat = str(cat_cv.index[0])
        most_stable_cat = str(cat_cv.index[-1])


# Savings insights
monthly_df["year_month_str"] = monthly_df["year_month"].dt.strftime("%Y-%m")
if custom_monthly_income is not None:
    monthly_df["income"] = float(custom_monthly_income)
else:
    monthly_df["income"] = monthly_df["year_month_str"].apply(get_income).astype(float)

monthly_df["savings"] = monthly_df["income"] - monthly_df["spend"]

income_known_months = int((monthly_df["income"] > 0).sum())
unknown_income_months = int((monthly_df["income"] <= 0).sum())

if custom_monthly_income is not None:
    savings_base = monthly_df.copy()
else:
    savings_base = monthly_df[monthly_df["income"] > 0].copy()

savings_available = not savings_base.empty

if savings_available:
    total_income = float(savings_base["income"].sum())
    total_savings = float(savings_base["savings"].sum())
    avg_savings = float(savings_base["savings"].mean())
    latest_savings = float(savings_base["savings"].iloc[-1])
    positive_savings_months = int((savings_base["savings"] >= 0).sum())
    deficit_months = int((savings_base["savings"] < 0).sum())
    savings_rate = (total_savings / total_income * 100) if total_income > 0 else None
else:
    total_income = 0.0
    total_savings = 0.0
    avg_savings = 0.0
    latest_savings = 0.0
    positive_savings_months = 0
    deficit_months = 0
    savings_rate = None


# -----------------------------------------------------------
# KPI ROWS
# -----------------------------------------------------------
st.markdown("### Monthly Performance KPIs")

k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Best Month (Low Spend)", fmt_month(best_month_ts), fmt_currency(best_month_value))
k2.metric("Worst Month (High Spend)", fmt_month(worst_month_ts), fmt_currency(worst_month_value))
k3.metric("Average Monthly Spend", fmt_currency(monthly_avg))
k4.metric("Latest Month Spend", fmt_currency(latest_spend))
k5.metric("MoM Growth", "N/A" if mom_growth_pct is None else f"{mom_growth_pct:+.1f}%")
k6.metric("Consistency Score", f"{consistency:.1f} / 100")

st.markdown("### Category Intelligence KPIs")

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Top Category", top_cat_name, fmt_currency(top_cat_spend))
c2.metric("Top Category Share", f"{top_cat_share_pct:.1f}%")
c3.metric("Top 3 Category Share", f"{top3_share_pct:.1f}%")
c4.metric("Active Categories", active_categories)
c5.metric("Most Volatile Category", most_volatile_cat)
c6.metric("Most Stable Category", most_stable_cat)

st.markdown("### Savings Intelligence KPIs")

if not savings_available:
    st.warning(
        "Savings insights need income data. Use the sidebar option 'Use custom monthly income' "
        "if mapped income is missing for selected months."
    )
else:
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Total Income", fmt_currency(total_income))
    s2.metric("Total Savings", fmt_currency(total_savings))
    s3.metric("Average Savings / Month", fmt_currency(avg_savings))
    s4.metric("Savings Rate", "N/A" if savings_rate is None else f"{savings_rate:.1f}%")

    s5, s6, s7 = st.columns(3)
    s5.metric("Latest Month Savings", fmt_currency(latest_savings))
    s6.metric("Positive Savings Months", f"{positive_savings_months} / {len(savings_base)}")
    s7.metric("Deficit Months", deficit_months)

if custom_monthly_income is None and unknown_income_months > 0:
    st.info(
        f"Income mapping was missing for {unknown_income_months} month(s). "
        f"Savings KPIs used {income_known_months} month(s) with known income."
    )


# -----------------------------------------------------------
# INSIGHT SUMMARY
# -----------------------------------------------------------
st.markdown("### Insight Summary")

insight_1 = (
    f"Monthly spend ranges from {fmt_currency(best_month_value)} in {fmt_month(best_month_ts)} "
    f"to {fmt_currency(worst_month_value)} in {fmt_month(worst_month_ts)}."
)
insight_2 = (
    f"The top category is {top_cat_name}, contributing {top_cat_share_pct:.1f}% of total spend. "
    f"Top 3 categories together contribute {top3_share_pct:.1f}%."
)

if mom_growth_pct is None:
    insight_3 = "MoM growth needs at least 2 months of data."
else:
    direction = "up" if mom_growth_pct > 0 else "down"
    insight_3 = f"Latest month spend is {direction} by {abs(mom_growth_pct):.1f}% versus the previous month."

if savings_available:
    if latest_savings >= 0:
        insight_4 = f"Latest month shows positive savings of {fmt_currency(latest_savings)}."
    else:
        insight_4 = f"Latest month shows a savings deficit of {fmt_currency(abs(latest_savings))}."
else:
    insight_4 = "Savings insight is unavailable because income data is missing."

st.write(f"- {insight_1}")
st.write(f"- {insight_2}")
st.write(f"- {insight_3}")
st.write(f"- {insight_4}")


# -----------------------------------------------------------
# CHART GRID
# -----------------------------------------------------------
c1, c2 = st.columns(2)

with c1:
    st.markdown("#### Monthly Spend Trend")
    st.line_chart(monthly)

    st.markdown("#### MoM Growth %")
    mom = (monthly.pct_change() * 100).round(2)
    st.line_chart(mom)

with c2:
    st.markdown("#### Cumulative Spend")
    st.line_chart(monthly.cumsum())

    st.markdown("#### Category Contribution")
    st.bar_chart(cat_share)


# -----------------------------------------------------------
# ADDITIONAL INSIGHT CHARTS
# -----------------------------------------------------------
st.markdown("#### Average Spend per Day by Month")
days = filtered.groupby("year_month")["period"].nunique()
days.index = pd.to_datetime(days.index)
avg_day = (monthly / days.reindex(monthly.index)).round(2)
st.bar_chart(avg_day)

st.markdown("#### Top 5 Category Trend by Month")
top5_categories = cat_share.head(5).index.tolist()
top5_trend = cat_monthly[cat_monthly["category"].isin(top5_categories)].pivot_table(
    index="year_month",
    columns="category",
    values="spend",
    fill_value=0.0,
)
st.area_chart(top5_trend)

if savings_available:
    st.markdown("#### Savings Trend by Month")
    savings_plot = savings_base.set_index("year_month")[["spend", "income", "savings"]]
    st.line_chart(savings_plot)


# -----------------------------------------------------------
# RUNNING TOTAL TREND
# -----------------------------------------------------------
st.markdown("#### Running Total Trend")
dcat_sorted = filtered.sort_values("period")
dcat_sorted["running_total"] = dcat_sorted["amount"].cumsum()
st.line_chart(dcat_sorted.set_index("period")["running_total"])
