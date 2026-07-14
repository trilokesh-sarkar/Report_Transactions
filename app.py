import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv

from utils.github_storage import read_csv, write_csv, write_savings_csv
from utils.kpi_dashboard import render_kpis, get_income

GOAL_TRACKING_START_PERIOD = "2026-04"


def format_currency(value):
    return f"₹{value:,.0f}"


def estimate_goal_completion(current_saved, target_amount, monthly_contribution, start_period):
    remaining = max(float(target_amount) - float(current_saved), 0.0)

    if remaining <= 0:
        return "Completed", 0

    if monthly_contribution <= 0:
        return "No projection", None

    months_needed = int((remaining + monthly_contribution - 1) // monthly_contribution)
    completion_month = pd.Period(start_period, freq="M") + months_needed - 1
    return completion_month.strftime("%b %Y"), months_needed


def months_until_target(current_period, target_period):
    current = pd.Period(current_period, freq="M")
    target = pd.Period(target_period, freq="M")
    month_gap = (target.year - current.year) * 12 + (target.month - current.month)
    return max(month_gap, 0) + 1


def prepare_goal_tracker_items(goal_items):
    base_columns = ["Goal", "Target Amount", "Saved at Start", "Target Month", "Priority"]

    if goal_items is None or len(goal_items) == 0:
        return pd.DataFrame(columns=base_columns)

    prepared = pd.DataFrame(goal_items).copy()

    # Migrate older session rows that stored the manual value as "Current Saved".
    if "Saved at Start" not in prepared.columns:
        legacy_saved = prepared["Current Saved"] if "Current Saved" in prepared.columns else 0.0
        prepared["Saved at Start"] = pd.to_numeric(legacy_saved, errors="coerce").fillna(0.0)

    for column in base_columns:
        if column not in prepared.columns:
            prepared[column] = "Medium" if column == "Priority" else 0.0 if column == "Saved at Start" else None

    return prepared[base_columns].copy()


def allocate_savings_to_goals(goal_rows, savings_pool):
    if goal_rows.empty:
        return pd.Series(dtype=float)

    allocation_order = {"High": 0, "Medium": 1, "Low": 2}
    allocation_view = goal_rows.copy()
    allocation_view["_priority_sort"] = allocation_view["Priority"].map(allocation_order).fillna(9)
    allocation_view["_target_ts"] = pd.to_datetime(allocation_view["Target Month"], format="%Y-%m", errors="coerce")
    allocation_view = allocation_view.sort_values(
        ["_priority_sort", "_target_ts", "Goal"],
        ascending=[True, True, True],
        na_position="last",
    )

    remaining_pool = max(float(savings_pool), 0.0)
    allocations = {}

    for idx, row in allocation_view.iterrows():
        baseline_saved = max(float(row["Saved at Start"]), 0.0)
        target_amount = max(float(row["Target Amount"]), 0.0)
        remaining_need = max(target_amount - baseline_saved, 0.0)
        auto_saved = min(remaining_need, remaining_pool)
        allocations[idx] = auto_saved
        remaining_pool -= auto_saved

    return pd.Series(allocations, dtype=float)


def build_monthly_savings_export(monthly_goal_view):
    export_columns = [
        "year_month",
        "month_start",
        "monthly_earnings",
        "monthly_expense",
        "monthly_savings",
        "cumulative_savings",
    ]

    if monthly_goal_view.empty:
        return pd.DataFrame(columns=export_columns)

    export_df = monthly_goal_view.copy()
    export_df["month_start"] = export_df["month_ts"].dt.strftime("%Y-%m-%d")
    export_df["cumulative_savings"] = export_df["savings"].cumsum().round(2)
    export_df = export_df.rename(
        columns={
            "income": "monthly_earnings",
            "spend": "monthly_expense",
            "savings": "monthly_savings",
        }
    )
    export_df = export_df[export_columns].copy()

    for column in ["monthly_earnings", "monthly_expense", "monthly_savings", "cumulative_savings"]:
        export_df[column] = pd.to_numeric(export_df[column], errors="coerce").fillna(0.0).round(2)

    return export_df


def sync_monthly_savings_export(export_df):
    export_signature = export_df.to_csv(index=False)
    if st.session_state.get("monthly_savings_export_signature") == export_signature:
        return

    write_savings_csv(export_df, "Sync monthly savings summary")
    st.session_state["monthly_savings_export_signature"] = export_signature


# -----------------------------------------------------------
# CONFIG
# -----------------------------------------------------------
load_dotenv()

st.set_page_config(
    page_title="💰 Finance Analytics",
    page_icon="📊",
    layout="wide"
)

# -----------------------------------------------------------
# LOAD CUSTOM CSS
# -----------------------------------------------------------
css_path = ".streamlit/styles.css"
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.markdown("<h1 class='title-main'>💰 Personal Finance Intelligence Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<h5 class='subtitle'>Track • Analyze • Forecast • Optimize</h5>", unsafe_allow_html=True)

# -----------------------------------------------------------
# PASSWORD
# -----------------------------------------------------------
APP_PASSWORD = os.getenv("APP_PASSWORD")
password = st.sidebar.text_input("🔑 Enter Access Password", type="password")

if APP_PASSWORD and password != APP_PASSWORD:
    st.stop()

st.success("🔓 Access Granted")

# -----------------------------------------------------------
# LOAD DATA FROM GITHUB CSV
# -----------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data():
    return read_csv()

def refresh():
    load_data.clear()
    st.rerun()

df = load_data()
df["period"] = pd.to_datetime(df["period"])
df = df.sort_values("period", ascending=False).reset_index(drop=True)

# -----------------------------------------------------------
# SIDEBAR FILTERS
# -----------------------------------------------------------

st.sidebar.markdown("### 🔍 Filters")

c1, c2 = st.sidebar.columns(2)

with c1:
    f_year = st.multiselect("Year", sorted(df.year.unique()))
    f_acc  = st.multiselect("Account", sorted(df.accounts.unique()))

with c2:
    f_month = st.multiselect("Month", sorted(df.year_month.unique()))
    include_cat = st.multiselect(
        "Include Category",
        sorted(df.category.unique()),
        placeholder="Include category...",
    )
    exclude_cat = st.multiselect(
        "Exclude Category",
        sorted(df.category.unique()),
        placeholder="Exclude category..."
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

# 👉 Inverse category filter
if exclude_cat:
    filtered = filtered[~filtered.category.isin(exclude_cat)]

if filtered.empty:
    st.warning("No data available after applying filters.")
    st.stop()

# -----------------------------------------------------------
# ADD EXPENSE
# -----------------------------------------------------------
st.markdown("<h3>➕ Add Expense Entry</h3>", unsafe_allow_html=True)

with st.expander("Add Expense Form"):

    if st.button("🔄 Refresh data"):
        refresh()

    with st.form("expense_form", clear_on_submit=True):
        d = st.date_input("📅 Date")

        categories = sorted( set(df.category.dropna().astype(str).tolist()) | {"bike_emi", "Trip"} )
        default_cat_index = categories.index("Food") if "Food" in categories else 0
        cat = st.selectbox("📂 Category", categories, index=default_cat_index)

        acc = st.text_input("🏦 Account / UPI / Card", value="UPI")
        amt = st.number_input("💰 Amount", min_value=0.0, value=11.0)

        submit = st.form_submit_button("💾 Save Entry")

    if submit:
        dt = pd.to_datetime(d)

        year = dt.year
        month = dt.strftime("%B")
        year_month = dt.strftime("%Y-%m")

        last_total = (
            df["running_total"].max()
            if "running_total" in df.columns and not df.empty
            else 0
        )
        running_total = last_total + amt

        new_row = pd.DataFrame([{
            "period": dt,
            "accounts": acc,
            "category": cat,
            "amount": amt,
            "year": year,
            "month": month,
            "year_month": year_month,
            "running_total": running_total
        }])

        df_new = pd.concat([df, new_row], ignore_index=True)

        write_csv(df_new, f"Added ₹{amt} in {cat}")
        st.success(f"Added ₹{amt} to {cat}")
        st.balloons()
        refresh()

# -----------------------------------------------------------
# TRANSACTIONS TABLE
# -----------------------------------------------------------
st.markdown("<h3>📄 Transactions</h3>", unsafe_allow_html=True)

df_show = filtered.copy()
df_show["period"] = df_show["period"].dt.date
df_show = df_show.sort_values("period", ascending=False)

st.dataframe(df_show, use_container_width=True, height=260)

csv = df_show.to_csv(index=False).encode()
st.download_button("📥 Export CSV", csv, "finance_data.csv")

# -----------------------------------------------------------
# DELETE TRANSACTION
# -----------------------------------------------------------
st.markdown("<h3>🗑 Delete Transaction</h3>", unsafe_allow_html=True)

df_del = df.copy().reset_index()
df_del["period"] = df_del["period"].dt.date

st.dataframe(df_del[["index", "period", "accounts", "category", "amount"]], height=220)

del_id = st.number_input("Row ID to Delete", min_value=0, step=1)

if st.button("🗑 Delete"):
    df_new = df_del.drop(index=del_id).drop(columns=["index"])
    write_csv(df_new, f"Deleted row {del_id}")
    st.success("Deleted Successfully")
    refresh()

# -----------------------------------------------------------
# KPIs
# -----------------------------------------------------------
render_kpis(filtered=filtered, df=df, MONTHLY_BUDGET=20000)

# -----------------------------------------------------------
# GOAL TRACKING
# -----------------------------------------------------------
st.markdown("<h3>🎯 Goal Tracking</h3>", unsafe_allow_html=True)
st.caption(
    "Savings projections reset from April 2026 and use mapped income, including the assumption that income from May 2026 onward is ₹32,000."
)

goal_source = df.copy()
goal_source["period"] = pd.to_datetime(goal_source["period"])
goal_source["year_month"] = goal_source["period"].dt.to_period("M").astype(str)

monthly_goal_view = (
    goal_source.groupby("year_month", as_index=False)["amount"]
    .sum()
    .rename(columns={"amount": "spend"})
)
monthly_goal_view["income"] = monthly_goal_view["year_month"].apply(get_income).astype(float)
monthly_goal_view = monthly_goal_view[monthly_goal_view["income"] > 0].copy()
monthly_goal_view = monthly_goal_view[
    pd.to_datetime(monthly_goal_view["year_month"]) >= pd.Timestamp(GOAL_TRACKING_START_PERIOD)
].copy()
monthly_goal_view["savings"] = monthly_goal_view["income"] - monthly_goal_view["spend"]
monthly_goal_view["month_ts"] = pd.to_datetime(monthly_goal_view["year_month"])
monthly_goal_view = monthly_goal_view.sort_values("month_ts").reset_index(drop=True)
monthly_savings_export = build_monthly_savings_export(monthly_goal_view)

try:
    sync_monthly_savings_export(monthly_savings_export)
except Exception as exc:
    st.warning(f"Unable to sync monthly savings CSV to GitHub: {exc}")

if monthly_goal_view.empty:
    st.info("Goal tracking needs at least one month with mapped income.")
else:
    recent_window = min(3, len(monthly_goal_view))
    avg_recent_savings = float(monthly_goal_view["savings"].tail(recent_window).mean())
    avg_all_savings = float(monthly_goal_view["savings"].mean())
    latest_savings = float(monthly_goal_view["savings"].iloc[-1])
    cumulative_savings = max(float(monthly_goal_view["savings"].sum()), 0.0)
    projected_monthly_contribution = max(avg_recent_savings, 0.0)
    current_goal_period = monthly_goal_view["year_month"].iloc[-1]

    default_goals = pd.DataFrame(
        [
            {"Goal": "Emergency Fund", "Target Amount": 150000.0, "Saved at Start": 0.0, "Target Month": "2026-12", "Priority": "High"},
            {"Goal": "Trip", "Target Amount": 60000.0, "Saved at Start": 0.0, "Target Month": "2026-10", "Priority": "Medium"},
            {"Goal": "Gadget", "Target Amount": 80000.0, "Saved at Start": 0.0, "Target Month": "2027-02", "Priority": "Low"},
            {"Goal": "Loan Closure Target", "Target Amount": 200000.0, "Saved at Start": 0.0, "Target Month": "2027-06", "Priority": "High"},
        ]
    )

    if "goal_tracker_items" not in st.session_state:
        st.session_state["goal_tracker_items"] = default_goals
    else:
        st.session_state["goal_tracker_items"] = prepare_goal_tracker_items(st.session_state["goal_tracker_items"])

    g1, g2, g3, g4, g5 = st.columns(5)
    g1.metric("Avg Savings / Month", format_currency(avg_all_savings))
    g2.metric("Recent Savings Pace", format_currency(avg_recent_savings))
    g3.metric("Latest Month Savings", format_currency(latest_savings))
    g4.metric("Projection Basis", f"{recent_window}-month avg")
    g5.metric("Auto Savings Pool", format_currency(cumulative_savings))

    st.caption(
        "Enter the amount already saved outside this tracker in `Saved at Start`. "
        "`Current Saved` is updated automatically using net savings since April 2026, "
        "allocated by priority and nearest target month."
    )
    st.caption(
        "Monthly savings, earnings, and expense data are also synced to `monthly_savings_data.csv` in GitHub for reuse."
    )

    goal_editor = st.data_editor(
        prepare_goal_tracker_items(st.session_state["goal_tracker_items"]),
        num_rows="dynamic",
        hide_index=True,
        width="stretch",
        column_config={
            "Goal": st.column_config.TextColumn("Goal"),
            "Target Amount": st.column_config.NumberColumn("Target Amount", min_value=0.0, step=5000.0, format="%.2f"),
            "Saved at Start": st.column_config.NumberColumn("Saved at Start", min_value=0.0, step=1000.0, format="%.2f"),
            "Target Month": st.column_config.TextColumn("Target Month", help="Use YYYY-MM format, for example 2026-12"),
            "Priority": st.column_config.SelectboxColumn("Priority", options=["High", "Medium", "Low"]),
        },
        key="goal_tracker_editor",
    )
    st.session_state["goal_tracker_items"] = prepare_goal_tracker_items(goal_editor)

    goal_results = st.session_state["goal_tracker_items"].copy()
    goal_results = goal_results.dropna(subset=["Goal", "Target Amount", "Saved at Start", "Target Month"])
    goal_results = goal_results[goal_results["Goal"].astype(str).str.strip() != ""].copy()

    if goal_results.empty:
        st.info("Add one or more goals to start tracking progress.")
    else:
        goal_results["Target Amount"] = pd.to_numeric(goal_results["Target Amount"], errors="coerce").fillna(0.0)
        goal_results["Saved at Start"] = pd.to_numeric(goal_results["Saved at Start"], errors="coerce").fillna(0.0)
        goal_results["Priority"] = goal_results["Priority"].fillna("Medium")
        goal_results["Target Month"] = goal_results["Target Month"].astype(str).str.strip()
        goal_results = goal_results[goal_results["Target Month"].str.match(r"^\d{4}-\d{2}$", na=False)].copy()
        goal_results["Auto Saved"] = allocate_savings_to_goals(goal_results, cumulative_savings).reindex(goal_results.index, fill_value=0.0)
        goal_results["Current Saved"] = (goal_results["Saved at Start"] + goal_results["Auto Saved"]).clip(upper=goal_results["Target Amount"])
        goal_results["Remaining"] = (goal_results["Target Amount"] - goal_results["Current Saved"]).clip(lower=0.0)
        goal_results["Progress %"] = goal_results.apply(
            lambda row: ((row["Current Saved"] / row["Target Amount"]) * 100) if row["Target Amount"] > 0 else 0.0,
            axis=1,
        )

        completion_labels = []
        months_needed_list = []
        target_months_left = []
        required_monthly_list = []
        target_gap_list = []
        status_list = []
        for _, row in goal_results.iterrows():
            completion_label, months_needed = estimate_goal_completion(
                current_saved=row["Current Saved"],
                target_amount=row["Target Amount"],
                monthly_contribution=projected_monthly_contribution,
                start_period=current_goal_period,
            )
            completion_labels.append(completion_label)
            months_needed_list.append(months_needed)
            months_left = months_until_target(current_goal_period, row["Target Month"])
            target_months_left.append(months_left)
            required_monthly = (row["Remaining"] / months_left) if months_left > 0 else row["Remaining"]
            required_monthly_list.append(required_monthly)
            target_gap = projected_monthly_contribution - required_monthly
            target_gap_list.append(target_gap)
            if row["Remaining"] <= 0:
                status_list.append("Completed")
            elif projected_monthly_contribution <= 0:
                status_list.append("No savings pace")
            elif target_gap >= 0:
                status_list.append("On Track")
            else:
                status_list.append("At Risk")

        goal_results["Projected Completion"] = completion_labels
        goal_results["Months Needed"] = months_needed_list
        goal_results["Months To Target"] = target_months_left
        goal_results["Required / Month"] = required_monthly_list
        goal_results["Target Gap"] = target_gap_list
        goal_results["Status"] = status_list
        goal_results["Suggested Monthly Savings"] = goal_results.apply(
            lambda row: (row["Remaining"] / max(row["Months Needed"], 1)) if row["Months Needed"] not in (None, 0) else 0.0,
            axis=1,
        )

        total_goal_target = float(goal_results["Target Amount"].sum())
        total_goal_saved = float(goal_results["Current Saved"].sum())
        total_goal_remaining = float(goal_results["Remaining"].sum())
        completed_goals = int((goal_results["Status"] == "Completed").sum())
        at_risk_goals = int((goal_results["Status"] == "At Risk").sum())
        total_auto_saved = float(goal_results["Auto Saved"].sum())

        summary1, summary2, summary3, summary4, summary5, summary6 = st.columns(6)
        summary1.metric("Goal Corpus", format_currency(total_goal_target))
        summary2.metric("Saved So Far", format_currency(total_goal_saved))
        summary3.metric("Still Needed", format_currency(total_goal_remaining))
        summary4.metric("Completed Goals", completed_goals)
        summary5.metric("At-Risk Goals", at_risk_goals)
        summary6.metric("Auto Applied", format_currency(total_auto_saved))

        goal_summary = goal_results.copy()
        priority_order = {"High": 0, "Medium": 1, "Low": 2}
        goal_summary["_priority_sort"] = goal_summary["Priority"].map(priority_order).fillna(9)
        goal_summary = goal_summary.sort_values(["_priority_sort", "Target Month", "Remaining"], ascending=[True, True, False]).drop(columns=["_priority_sort"])

        for column in [
            "Target Amount",
            "Saved at Start",
            "Auto Saved",
            "Current Saved",
            "Remaining",
            "Progress %",
            "Suggested Monthly Savings",
            "Required / Month",
            "Target Gap",
        ]:
            goal_summary[column] = goal_summary[column].round(2)

        st.dataframe(goal_summary, width="stretch", height=260)

        priority_order = {"High": 0, "Medium": 1, "Low": 2}
        goal_cards = goal_results.assign(_priority_sort=goal_results["Priority"].map(priority_order).fillna(9))
        goal_cards = goal_cards.sort_values(["_priority_sort", "Target Month", "Remaining"]).drop(columns=["_priority_sort"])

        progress_cols = st.columns(min(4, len(goal_cards)))
        for idx, (_, row) in enumerate(goal_cards.head(4).iterrows()):
            progress_cols[idx].metric(
                row["Goal"],
                f"{row['Progress %']:.1f}%",
                f"{row['Status']} • {format_currency(row['Remaining'])} left",
            )
            progress_cols[idx].progress(min(max(row["Progress %"] / 100, 0.0), 1.0))

        breakdown_col1, breakdown_col2 = st.columns(2)

        with breakdown_col1:
            status_counts = goal_results["Status"].value_counts().rename_axis("Status").reset_index(name="Goals")
            st.markdown("#### Goal Status Mix")
            st.bar_chart(status_counts.set_index("Status"))

        with breakdown_col2:
            required_savings = goal_results[["Goal", "Required / Month"]].copy().sort_values("Required / Month", ascending=False)
            required_savings["Required / Month"] = required_savings["Required / Month"].round(2)
            st.markdown("#### Monthly Savings Needed By Goal")
            st.dataframe(required_savings, width="stretch", height=220)

        savings_plot = monthly_goal_view[["month_ts", "savings", "income", "spend"]].rename(
            columns={"month_ts": "Month", "savings": "Savings", "income": "Income", "spend": "Spend"}
        ).set_index("Month")
        st.markdown("#### Savings Trend Supporting Goals")
        st.line_chart(savings_plot)
