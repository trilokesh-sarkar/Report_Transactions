import streamlit as st
from utils.filter_state import persistent_multiselect, reset_filter_button
import pandas as pd
import os
from utils.github_storage import read_csv
from utils.kpi_dashboard import render_kpis, get_income
from utils.kpi_drilldown import render_kpi_suite

DEFAULT_TOTAL_CATEGORY_BUDGET = 20000.0
DEFAULT_RENT_BUDGET = 7000.0


def format_currency(value):
    return f"₹{value:,.0f}"


def round_to_step(value, step=500):
    if value <= 0:
        return 0.0
    return float(step * round(value / step))


def ensure_budget_map(source_df):
    categories = sorted(source_df["category"].dropna().unique())
    category_totals = source_df.groupby("category")["amount"].sum()
    defaults = {category: 0.0 for category in categories}

    if not categories:
        return {}

    rent_category = next((category for category in categories if str(category).strip().lower() == "rent"), None)
    remaining_budget = DEFAULT_TOTAL_CATEGORY_BUDGET

    if rent_category is not None:
        defaults[rent_category] = DEFAULT_RENT_BUDGET
        remaining_budget -= DEFAULT_RENT_BUDGET

    non_rent_categories = [category for category in categories if category != rent_category]
    non_rent_total = float(category_totals.reindex(non_rent_categories).fillna(0.0).sum())

    if non_rent_categories:
        if non_rent_total > 0:
            for category in non_rent_categories:
                share = float(category_totals.get(category, 0.0)) / non_rent_total
                defaults[category] = round_to_step(remaining_budget * share)
        else:
            equal_budget = round_to_step(remaining_budget / len(non_rent_categories))
            for category in non_rent_categories:
                defaults[category] = equal_budget

    allocated_total = sum(defaults.values())
    gap = DEFAULT_TOTAL_CATEGORY_BUDGET - allocated_total
    if categories and abs(gap) >= 1:
        adjust_target = rent_category if rent_category is not None else max(defaults, key=defaults.get)
        defaults[adjust_target] = max(defaults[adjust_target] + gap, 0.0)

    if "kpi_category_budget_map" not in st.session_state:
        st.session_state["kpi_category_budget_map"] = defaults
    else:
        saved_budget_map = st.session_state["kpi_category_budget_map"]
        for category, budget in defaults.items():
            if category not in saved_budget_map:
                saved_budget_map[category] = budget

        if rent_category is not None:
            saved_budget_map[rent_category] = DEFAULT_RENT_BUDGET

    return st.session_state["kpi_category_budget_map"]


def safe_pct_change(current_value, previous_value):
    if previous_value in (None, 0):
        return float("nan")
    return ((current_value - previous_value) / previous_value) * 100


# -----------------------------------------------------------
# LOAD GLOBAL CSS
# -----------------------------------------------------------
css_path = ".streamlit/styles.css"
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.set_page_config(layout="wide")
st.title("📊 KPI Intelligence Dashboard")

# -----------------------------------------------------------
# LOAD DATA
# -----------------------------------------------------------
df = read_csv()
df["period"] = pd.to_datetime(df["period"])
df = df.sort_values("period", ascending=False).reset_index(drop=True)

# -----------------------------------------------------------
# SIDEBAR FILTERS (Compact + Exclude Category)
# -----------------------------------------------------------
st.sidebar.markdown("### 🔍 Smart Filters")

c1, c2 = st.sidebar.columns(2)

with c1:
    f_year = persistent_multiselect("Year", sorted(df.year.unique(), reverse=True), "years")
    f_acc  = persistent_multiselect("Account", sorted(df.accounts.unique()), "accounts")

with c2:
    f_month = persistent_multiselect("Month", sorted(df.year_month.unique(), reverse=True), "months")
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
historical_filtered = df.copy()

if f_year:
    filtered = filtered[filtered.year.isin(f_year)]
    historical_filtered = historical_filtered[historical_filtered.year.isin(f_year)]

if f_month:
    filtered = filtered[filtered.year_month.isin(f_month)]

if f_acc:
    filtered = filtered[filtered.accounts.isin(f_acc)]
    historical_filtered = historical_filtered[historical_filtered.accounts.isin(f_acc)]

if include_cat:
    filtered = filtered[filtered.category.isin(include_cat)]
    historical_filtered = historical_filtered[historical_filtered.category.isin(include_cat)]

# 👉 Exclude category logic
if exclude_cat:
    filtered = filtered[~filtered.category.isin(exclude_cat)]
    historical_filtered = historical_filtered[~historical_filtered.category.isin(exclude_cat)]

if filtered.empty:
    st.warning("No data available after applying filters.")
    st.stop()

if historical_filtered.empty:
    historical_filtered = filtered.copy()

# -----------------------------------------------------------
# ADVANCED KPI STRIP
# -----------------------------------------------------------
st.markdown("### 🧠 Financial Health Indicators")

total_spend = filtered["amount"].sum()
txn_count = len(filtered)
avg_txn = total_spend / txn_count if txn_count else 0

# Most expensive category
top_cat = (
    filtered.groupby("category")["amount"]
    .sum()
    .sort_values(ascending=False)
)

top_cat_name = top_cat.index[0] if not top_cat.empty else "-"
top_cat_value = top_cat.iloc[0] if not top_cat.empty else 0
top_cat_share = (top_cat_value / total_spend * 100) if total_spend else 0

category_stats = (
    filtered.groupby("category")["amount"]
    .agg(["sum", "count", "mean"])
    .sort_values("sum", ascending=False)
)

active_categories = int(category_stats.shape[0]) if not category_stats.empty else 0
top3_share = (category_stats["sum"].head(3).sum() / total_spend * 100) if total_spend else 0
most_frequent_category = category_stats["count"].idxmax() if not category_stats.empty else "-"
most_frequent_count = int(category_stats["count"].max()) if not category_stats.empty else 0
highest_avg_category = category_stats["mean"].idxmax() if not category_stats.empty else "-"
highest_avg_value = float(category_stats["mean"].max()) if not category_stats.empty else 0
smallest_category = category_stats.index[-1] if not category_stats.empty else "-"
smallest_category_value = float(category_stats["sum"].iloc[-1]) if not category_stats.empty else 0

# Most active account
top_acc = (
    filtered.groupby("accounts")["amount"]
    .sum()
    .sort_values(ascending=False)
)

top_acc_name = top_acc.index[0] if not top_acc.empty else "-"
top_acc_value = top_acc.iloc[0] if not top_acc.empty else 0

# Monthly burn rate
monthly = filtered.groupby("year_month")["amount"].sum()
burn_rate = monthly.mean() if not monthly.empty else 0

# Spending volatility
volatility = monthly.std() if len(monthly) > 1 else 0

# Best & Worst month
best_month = monthly.idxmin() if not monthly.empty else "-"
worst_month = monthly.idxmax() if not monthly.empty else "-"

k1, k2, k3, k4 = st.columns(4)
k1.metric("💳 Avg Transaction", f"₹{avg_txn:,.0f}")
k2.metric("🔥 Monthly Burn Rate", f"₹{burn_rate:,.0f}")
k3.metric("📂 Top Category", top_cat_name, f"₹{top_cat_value:,.0f}")
k4.metric("🏦 Top Account", top_acc_name, f"₹{top_acc_value:,.0f}")

k5, k6, k7 = st.columns(3)
k5.metric("🧾 Total Transactions", txn_count)
k6.metric("🟢 Best Month", best_month)
k7.metric("🔴 Worst Month", worst_month)

st.caption(f"Spending Volatility (std dev): ₹{volatility:,.0f}")

# -----------------------------------------------------------
# CATEGORY INTELLIGENCE
# -----------------------------------------------------------
st.markdown("### 📂 Category Intelligence")

c1, c2, c3, c4 = st.columns(4)
c1.metric("📁 Active Categories", active_categories)
c2.metric("🥇 Top Category Share", f"{top_cat_share:.1f}%")
c3.metric("📊 Top 3 Share", f"{top3_share:.1f}%")
c4.metric("🔁 Most Frequent Category", most_frequent_category, f"{most_frequent_count} txns")

c5, c6, c7 = st.columns(3)
c5.metric("💸 Highest Avg Ticket", highest_avg_category, f"₹{highest_avg_value:,.0f}")
c6.metric("🪶 Smallest Category", smallest_category, f"₹{smallest_category_value:,.0f}")
c7.metric("🏆 Top Category", top_cat_name, f"₹{top_cat_value:,.0f}")

st.dataframe(
    category_stats.rename(
        columns={
            "sum": "Total Spend",
            "count": "Transactions",
            "mean": "Avg Transaction",
        }
    ).round({"Total Spend": 2, "Avg Transaction": 2}),
    width="stretch",
    height=260,
)

# -----------------------------------------------------------
# CATEGORY BUDGET VS ACTUAL
# -----------------------------------------------------------
st.markdown("### 🎯 Budget vs Actual by Category")

budget_map = ensure_budget_map(df)
available_budget_months = sorted(filtered["year_month"].unique())
comparison_month = st.selectbox(
    "Budget Comparison Month",
    options=available_budget_months,
    index=len(available_budget_months) - 1,
)

budget_categories = sorted(filtered["category"].dropna().unique())
budget_editor = pd.DataFrame(
    {
        "Category": budget_categories,
        "Monthly Budget": [float(budget_map.get(category, 0.0)) for category in budget_categories],
    }
)

edited_budgets = st.data_editor(
    budget_editor,
    hide_index=True,
    width="stretch",
    column_config={
        "Category": st.column_config.TextColumn("Category", disabled=True),
        "Monthly Budget": st.column_config.NumberColumn(
            "Monthly Budget",
            min_value=0.0,
            step=500.0,
            format="%.2f",
        ),
    },
    key="kpi_budget_editor",
)

for _, row in edited_budgets.iterrows():
    budget_map[row["Category"]] = float(row["Monthly Budget"] or 0.0)

month_actuals = (
    filtered[filtered["year_month"] == comparison_month]
    .groupby("category")["amount"]
    .sum()
)

budget_vs_actual = edited_budgets.copy()
budget_vs_actual["Actual Spend"] = budget_vs_actual["Category"].map(month_actuals).fillna(0.0)
budget_vs_actual["Variance"] = budget_vs_actual["Monthly Budget"] - budget_vs_actual["Actual Spend"]
budget_vs_actual["Overspend %"] = budget_vs_actual.apply(
    lambda row: (
        ((row["Actual Spend"] - row["Monthly Budget"]) / row["Monthly Budget"]) * 100
        if row["Monthly Budget"] > 0 and row["Actual Spend"] > row["Monthly Budget"]
        else (100.0 if row["Monthly Budget"] == 0 and row["Actual Spend"] > 0 else 0.0)
    ),
    axis=1,
)
budget_vs_actual["Remaining Budget"] = budget_vs_actual["Variance"].clip(lower=0.0)
budget_vs_actual["Status"] = budget_vs_actual["Variance"].apply(
    lambda value: "Red Flag" if value < 0 else "Within Budget"
)
budget_vs_actual = budget_vs_actual.sort_values(["Overspend %", "Actual Spend"], ascending=[False, False])

total_budget = float(budget_vs_actual["Monthly Budget"].sum())
actual_budget_spend = float(budget_vs_actual["Actual Spend"].sum())
over_budget_count = int((budget_vs_actual["Status"] == "Red Flag").sum())
remaining_budget_total = float(budget_vs_actual["Variance"].clip(lower=0.0).sum())

b1, b2, b3, b4 = st.columns(4)
b1.metric("Budgeted Categories", len(budget_vs_actual))
b2.metric("Monthly Budget Total", format_currency(total_budget))
b3.metric("Actual Spend In Month", format_currency(actual_budget_spend))
b4.metric("Red-Flag Categories", over_budget_count)

st.caption(f"Budget comparison is based on {comparison_month}. Red flags mark categories where actual spend is above the monthly limit.")

budget_display = budget_vs_actual.copy()
for column in ["Monthly Budget", "Actual Spend", "Variance", "Overspend %", "Remaining Budget"]:
    budget_display[column] = budget_display[column].round(2)
st.dataframe(budget_display, width="stretch", height=280)

# -----------------------------------------------------------
# SAVINGS OPPORTUNITY DETECTOR
# -----------------------------------------------------------
st.markdown("### 💡 Savings Opportunity Detector")

analysis_month = max(f_month) if f_month else max(historical_filtered["year_month"])
analysis_month_ts = pd.to_datetime(analysis_month)

historical_scope = historical_filtered[
    pd.to_datetime(historical_filtered["year_month"]) <= analysis_month_ts
].copy()

category_monthly = (
    historical_scope.groupby(["year_month", "category"])["amount"]
    .sum()
    .reset_index()
)
category_monthly["month_ts"] = pd.to_datetime(category_monthly["year_month"])

if not category_monthly.empty:
    category_pivot = category_monthly.pivot_table(
        index="month_ts",
        columns="category",
        values="amount",
        fill_value=0.0,
    ).sort_index()
    full_month_index = pd.date_range(category_pivot.index.min(), analysis_month_ts, freq="MS")
    category_pivot = category_pivot.reindex(full_month_index, fill_value=0.0)
else:
    category_pivot = pd.DataFrame(index=pd.DatetimeIndex([analysis_month_ts]))

analysis_month_label = analysis_month_ts.strftime("%Y-%m")
analysis_month_position = category_pivot.index.get_loc(analysis_month_ts) if analysis_month_ts in category_pivot.index else len(category_pivot.index) - 1

growth_rows = []
savings_rows = []

for category in category_pivot.columns:
    series = category_pivot[category]
    latest_value = float(series.iloc[analysis_month_position]) if len(series) > 0 else 0.0
    previous_value = float(series.iloc[analysis_month_position - 1]) if analysis_month_position >= 1 else 0.0
    growth_amount = latest_value - previous_value
    growth_pct = safe_pct_change(latest_value, previous_value)

    if growth_amount > 0:
        growth_rows.append(
            {
                "Category": category,
                "Latest Month Spend": latest_value,
                "Previous Month Spend": previous_value,
                "Growth Amount": growth_amount,
                "Growth %": growth_pct,
            }
        )

    prior_window = series.iloc[max(analysis_month_position - 3, 0):analysis_month_position]
    baseline = float(prior_window.mean()) if not prior_window.empty else previous_value
    possible_saving = max(latest_value - baseline, 0.0)

    if possible_saving > 0:
        savings_rows.append(
            {
                "Category": category,
                "Latest Month Spend": latest_value,
                "Baseline Spend": baseline,
                "Possible Monthly Savings": possible_saving,
            }
        )

growth_df = (
    pd.DataFrame(growth_rows).sort_values("Growth Amount", ascending=False)
    if growth_rows and analysis_month_position >= 1
    else pd.DataFrame()
)
savings_df = pd.DataFrame(savings_rows).sort_values("Possible Monthly Savings", ascending=False) if savings_rows else pd.DataFrame()

latest_month_transactions = historical_scope[historical_scope["year_month"] == analysis_month_label].copy()
repeat_summary = (
    latest_month_transactions.groupby("category")
    .agg(
        Transactions=("amount", "size"),
        Active_Days=("period", lambda s: pd.to_datetime(s).dt.date.nunique()),
        Total_Spend=("amount", "sum"),
        Avg_Ticket=("amount", "mean"),
    )
    .reset_index()
    .rename(columns={"category": "Category"})
)

repeat_threshold = repeat_summary["Transactions"].median() if not repeat_summary.empty else 0
avg_ticket_threshold = repeat_summary["Avg_Ticket"].median() if not repeat_summary.empty else 0

repeat_candidates = repeat_summary[
    (repeat_summary["Transactions"] >= max(repeat_threshold, 3))
    & (repeat_summary["Avg_Ticket"] <= avg_ticket_threshold)
].sort_values(["Transactions", "Total_Spend"], ascending=[False, False])

top_growth_category = growth_df.iloc[0]["Category"] if not growth_df.empty else "-"
top_growth_amount = float(growth_df.iloc[0]["Growth Amount"]) if not growth_df.empty else 0.0
top_savings_category = savings_df.iloc[0]["Category"] if not savings_df.empty else "-"
top_savings_amount = float(savings_df.iloc[0]["Possible Monthly Savings"]) if not savings_df.empty else 0.0

s1, s2, s3, s4 = st.columns(4)
s1.metric("Highest Growth Category", top_growth_category, format_currency(top_growth_amount))
s2.metric("Top Savings Opportunity", top_savings_category, format_currency(top_savings_amount))
s3.metric("Repeat-Spend Categories", len(repeat_candidates))
s4.metric("Potential Savings Pool", format_currency(float(savings_df["Possible Monthly Savings"].head(5).sum()) if not savings_df.empty else 0.0))

st.caption(
    f"These detectors use historical data up to {analysis_month_label} after applying Year, Account, and Category filters."
)

tab1, tab2, tab3 = st.tabs(["Growth Watchlist", "Savings Opportunities", "Repeat Spend Heuristic"])

with tab1:
    if growth_df.empty:
        st.info(f"Need at least 2 months of category history before or up to {analysis_month_label} to detect growth categories.")
    else:
        growth_display = growth_df.head(8).copy()
        for column in ["Latest Month Spend", "Previous Month Spend", "Growth Amount", "Growth %"]:
            growth_display[column] = growth_display[column].round(2)
        st.dataframe(growth_display, width="stretch", height=260)

with tab2:
    if savings_df.empty:
        st.info("No category is currently above its recent baseline.")
    else:
        savings_display = savings_df.head(8).copy()
        for column in ["Latest Month Spend", "Baseline Spend", "Possible Monthly Savings"]:
            savings_display[column] = savings_display[column].round(2)
        st.dataframe(savings_display, width="stretch", height=260)

with tab3:
    st.caption(f"Heuristic: categories with frequent low-ticket transactions in {analysis_month_label} may be easier to trim.")
    if repeat_candidates.empty:
        st.info("No obvious repeat-spend pattern detected in the latest month.")
    else:
        repeat_display = repeat_candidates.head(8).copy()
        for column in ["Total_Spend", "Avg_Ticket"]:
            repeat_display[column] = repeat_display[column].round(2)
        st.dataframe(repeat_display, width="stretch", height=260)

# -----------------------------------------------------------
# CATEGORY TREND KPIS
# -----------------------------------------------------------
st.markdown("### 📈 Category Trend KPIs")

major_categories = category_stats.head(5).index.tolist()
trend_rows = []

for category in major_categories:
    series = category_pivot[category] if category in category_pivot.columns else pd.Series(dtype=float)
    latest_value = float(series.iloc[analysis_month_position]) if len(series) > 0 else 0.0
    previous_value = float(series.iloc[analysis_month_position - 1]) if analysis_month_position >= 1 else 0.0
    trend_rows.append(
        {
            "Category": category,
            "Latest Month Spend": latest_value,
            "MoM Change %": safe_pct_change(latest_value, previous_value),
            "3-Month Avg": float(series.iloc[max(analysis_month_position - 2, 0):analysis_month_position + 1].mean()) if not series.empty else 0.0,
            "Peak Month": series.iloc[:analysis_month_position + 1].idxmax().strftime("%Y-%m") if not series.empty else "-",
            "Peak Spend": float(series.iloc[:analysis_month_position + 1].max()) if not series.empty else 0.0,
            "Volatility": float(series.iloc[:analysis_month_position + 1].std()) if analysis_month_position >= 1 else 0.0,
        }
    )

trend_df = pd.DataFrame(trend_rows)
trend_display = trend_df.copy()
for column in ["Latest Month Spend", "MoM Change %", "3-Month Avg", "Peak Spend", "Volatility"]:
    trend_display[column] = trend_display[column].round(2)

t1, t2, t3 = st.columns(3)
t1.metric("Tracked Major Categories", len(trend_display))
t2.metric("Latest Trend Month", analysis_month_label)
t3.metric("Avg Category Volatility", format_currency(float(trend_df["Volatility"].mean()) if not trend_df.empty else 0.0))

st.dataframe(trend_display, width="stretch", height=260)

# -----------------------------------------------------------
# KPI RENDER (your original KPIs)
# -----------------------------------------------------------
render_kpis(filtered=filtered, df=df, MONTHLY_BUDGET=20000)
render_kpi_suite(filtered, get_income)
