import streamlit as st
from utils.filter_state import persistent_multiselect, reset_filter_button
import pandas as pd
from utils.github_storage import read_csv
import os

# -----------------------------------------------------------
# LOAD CSS
# -----------------------------------------------------------
css_path = ".streamlit/styles.css"
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.set_page_config(layout="wide")
st.title("📊 Spending Insights")

# -----------------------------------------------------------
# LOAD DATA
# -----------------------------------------------------------
df = read_csv()
df["period"] = pd.to_datetime(df["period"])
df["year_month"] = df["period"].dt.to_period("M").astype(str)

# -----------------------------------------------------------
# SIDEBAR FILTERS (compact)
# -----------------------------------------------------------
st.sidebar.markdown("### 🔍 Filters")

c1, c2 = st.sidebar.columns(2)

with c1:
    f_year = persistent_multiselect("Year", sorted(df.year.unique()), "years")
    f_acc  = persistent_multiselect("Account", sorted(df.accounts.unique()), "accounts")

with c2:
    f_month = persistent_multiselect("Month", sorted(df.year_month.unique()), "months")
    include_cat = persistent_multiselect(
        "Include Category",
        sorted(df.category.unique()),
        "include_categories",
        placeholder="Include category..."
    )
    exclude_cat = persistent_multiselect(
        "Exclude Category",
        sorted(df.category.unique()),
        "exclude_categories",
        placeholder="Exclude category..."
    )

reset_filter_button()

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
    st.warning("No data after filters.")
    st.stop()

# -----------------------------------------------------------
# KPI STRIP
# -----------------------------------------------------------
daily = filtered.groupby("period")["amount"].sum()
total = filtered["amount"].sum()

k1, k2, k3, k4 = st.columns(4)
k1.metric("💰 Total Spend", f"₹{total:,.0f}")
k2.metric("📆 Avg Daily Spend", f"₹{daily.mean():.0f}")
k3.metric("💸 Highest Day", f"{daily.idxmax().date()}", f"₹{daily.max():.0f}")
k4.metric("🧾 Transactions", len(filtered))

# -----------------------------------------------------------
# TOP CATEGORIES + DAILY TREND
# -----------------------------------------------------------
c1, c2 = st.columns(2)

with c1:
    st.subheader("🏆 Top Categories")
    cat_spend = filtered.groupby("category")["amount"].sum().sort_values(ascending=False)
    st.bar_chart(cat_spend)

with c2:
    st.subheader("📈 Daily Spending Trend")
    st.line_chart(daily)

# -----------------------------------------------------------
# WEEKDAY vs WEEKEND + CATEGORY %
# -----------------------------------------------------------
filtered["dow"] = filtered["period"].dt.dayofweek
filtered["type"] = filtered["dow"].apply(lambda x: "Weekend" if x >= 5 else "Weekday")

c3, c4 = st.columns(2)

with c3:
    st.subheader("📅 Weekday vs Weekend (Avg)")
    ww = filtered.groupby("type")["amount"].mean()
    st.bar_chart(ww)

with c4:
    st.subheader("📂 Category Contribution %")
    total_spend = filtered["amount"].sum()
    cat_pct = (
        filtered.groupby("category")["amount"]
        .sum()
        .sort_values(ascending=False) / total_spend * 100
    )
    st.bar_chart(cat_pct)

# -----------------------------------------------------------
# MONTHLY TREND + MOVING AVERAGE
# -----------------------------------------------------------
st.subheader("📈 Monthly Trend with Smoothing")

monthly = filtered.groupby("year_month")["amount"].sum()
ma = monthly.rolling(3).mean()

st.line_chart(pd.DataFrame({
    "Monthly Spend": monthly,
    "3-Month Moving Avg": ma
}))

# -----------------------------------------------------------
# HEATMAP + CONSISTENCY
# -----------------------------------------------------------
c5, c6 = st.columns(2)

with c5:
    st.subheader("🗓 Day vs Month Pattern")
    filtered["day"] = filtered["period"].dt.day
    filtered["month_name"] = filtered["period"].dt.month_name()

    heat = filtered.pivot_table(
        values="amount",
        index="day",
        columns="month_name",
        aggfunc="sum",
        fill_value=0
    )
    st.dataframe(heat, height=300)

with c6:
    st.subheader("🎯 Spend Consistency")
    std = daily.std()
    mean = daily.mean()
    score = max(0, 100 - (std / mean) * 100)
    st.metric("Consistency Score", f"{score:.1f} / 100")

# -----------------------------------------------------------
# TOP TRANSACTIONS + BEST CATEGORY PER MONTH
# -----------------------------------------------------------
c7, c8 = st.columns(2)

with c7:
    st.subheader("💎 Top 5 Transactions")
    top_txn = filtered.sort_values("amount", ascending=False).head(5)[
        ["period", "category", "accounts", "amount"]
    ]
    st.dataframe(top_txn, height=250)

with c8:
    st.subheader("🏅 Most Expensive Category Each Month")
    monthly_cat = (
        filtered.groupby(["year_month", "category"])["amount"]
        .sum()
        .reset_index()
    )
    idx = monthly_cat.groupby("year_month")["amount"].idxmax()
    best_cat_month = monthly_cat.loc[idx]
    st.dataframe(best_cat_month.sort_values("year_month"), height=250)
