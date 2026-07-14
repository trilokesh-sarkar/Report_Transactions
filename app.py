import os

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from utils.finance_agent import get_ai_configuration_status, run_finance_agent
from utils.github_storage import read_csv, write_csv, write_savings_csv
from utils.kpi_dashboard import get_income, render_kpis

GOAL_TRACKING_START_PERIOD = "2026-04"
MONTHLY_BUDGET = 20000
PRIORITY_ORDER = {"High": 0, "Medium": 1, "Low": 2}
DEFAULT_GOALS = pd.DataFrame(
    [
        {"Goal": "Emergency Fund", "Target Amount": 150000.0, "Saved at Start": 0.0, "Target Month": "2026-12", "Priority": "High"},
        {"Goal": "Trip", "Target Amount": 60000.0, "Saved at Start": 0.0, "Target Month": "2026-10", "Priority": "Medium"},
        {"Goal": "Gadget", "Target Amount": 80000.0, "Saved at Start": 0.0, "Target Month": "2027-02", "Priority": "Low"},
        {"Goal": "Loan Closure Target", "Target Amount": 200000.0, "Saved at Start": 0.0, "Target Month": "2027-06", "Priority": "High"},
    ]
)


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

    allocation_view = goal_rows.copy()
    allocation_view["_priority_sort"] = allocation_view["Priority"].map(PRIORITY_ORDER).fillna(9)
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


@st.cache_data(show_spinner=False)
def load_data():
    return read_csv()


def refresh():
    load_data.clear()
    st.rerun()


def configure_page():
    load_dotenv(dotenv_path=".env", override=False)
    st.set_page_config(
        page_title="💰 Finance Analytics",
        page_icon="📊",
        layout="wide",
    )


def load_custom_css():
    css_path = ".streamlit/styles.css"
    if os.path.exists(css_path):
        with open(css_path) as css_file:
            st.markdown(f"<style>{css_file.read()}</style>", unsafe_allow_html=True)


def render_page_header():
    st.markdown("<h1 class='title-main'>💰 Personal Finance Intelligence Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("<h5 class='subtitle'>Track • Analyze • Forecast • Optimize</h5>", unsafe_allow_html=True)


def require_password():
    app_password = os.getenv("APP_PASSWORD")
    password = st.sidebar.text_input("🔑 Enter Access Password", type="password")

    if app_password and password != app_password:
        st.stop()

    st.success("🔓 Access Granted")


def prepare_main_dataframe():
    df = load_data().copy()
    df["period"] = pd.to_datetime(df["period"], errors="coerce")
    return df.sort_values("period", ascending=False).reset_index(drop=True)


def render_sidebar_filters(df):
    st.sidebar.markdown("### 🔍 Filters")

    col1, col2 = st.sidebar.columns(2)
    with col1:
        selected_years = st.multiselect("Year", sorted(df["year"].dropna().unique()))
        selected_accounts = st.multiselect("Account", sorted(df["accounts"].dropna().unique()))

    with col2:
        selected_months = st.multiselect("Month", sorted(df["year_month"].dropna().unique()))
        include_categories = st.multiselect(
            "Include Category",
            sorted(df["category"].dropna().unique()),
            placeholder="Include category...",
        )
        exclude_categories = st.multiselect(
            "Exclude Category",
            sorted(df["category"].dropna().unique()),
            placeholder="Exclude category...",
        )

    return {
        "years": selected_years,
        "accounts": selected_accounts,
        "months": selected_months,
        "include_categories": include_categories,
        "exclude_categories": exclude_categories,
    }


def apply_sidebar_filters(df, filters):
    filtered = df.copy()

    if filters["years"]:
        filtered = filtered[filtered["year"].isin(filters["years"])]
    if filters["months"]:
        filtered = filtered[filtered["year_month"].isin(filters["months"])]
    if filters["accounts"]:
        filtered = filtered[filtered["accounts"].isin(filters["accounts"])]
    if filters["include_categories"]:
        filtered = filtered[filtered["category"].isin(filters["include_categories"])]
    if filters["exclude_categories"]:
        filtered = filtered[~filtered["category"].isin(filters["exclude_categories"])]

    return filtered


def summarize_active_filters(filters):
    summary = {}
    for key, value in filters.items():
        if value:
            summary[key] = value
    return summary


def render_expense_entry(df):
    st.markdown("<h3>➕ Add Expense Entry</h3>", unsafe_allow_html=True)

    with st.expander("Add Expense Form"):
        if st.button("🔄 Refresh data"):
            refresh()

        with st.form("expense_form", clear_on_submit=True):
            entry_date = st.date_input("📅 Date")
            categories = sorted(set(df["category"].dropna().astype(str).tolist()) | {"bike_emi", "Trip"})
            default_cat_index = categories.index("Food") if "Food" in categories else 0
            category = st.selectbox("📂 Category", categories, index=default_cat_index)
            account = st.text_input("🏦 Account / UPI / Card", value="UPI")
            amount = st.number_input("💰 Amount", min_value=0.0, value=11.0)
            submitted = st.form_submit_button("💾 Save Entry")

        if submitted:
            entry_ts = pd.to_datetime(entry_date)
            last_total = (
                df["running_total"].max()
                if "running_total" in df.columns and not df.empty
                else 0
            )

            new_row = pd.DataFrame(
                [
                    {
                        "period": entry_ts,
                        "accounts": account,
                        "category": category,
                        "amount": amount,
                        "year": entry_ts.year,
                        "month": entry_ts.strftime("%B"),
                        "year_month": entry_ts.strftime("%Y-%m"),
                        "running_total": last_total + amount,
                    }
                ]
            )

            df_new = pd.concat([df, new_row], ignore_index=True)
            write_csv(df_new, f"Added ₹{amount} in {category}")
            st.success(f"Added ₹{amount} to {category}")
            st.balloons()
            refresh()


def render_transactions_section(filtered):
    st.markdown("<h3>📄 Transactions</h3>", unsafe_allow_html=True)

    df_show = filtered.copy()
    df_show["period"] = df_show["period"].dt.date
    df_show = df_show.sort_values("period", ascending=False)

    st.dataframe(df_show, use_container_width=True, height=260)
    st.download_button("📥 Export CSV", df_show.to_csv(index=False).encode(), "finance_data.csv")


def render_delete_section(df):
    st.markdown("<h3>🗑 Delete Transaction</h3>", unsafe_allow_html=True)

    df_del = df.copy().reset_index()
    df_del["period"] = df_del["period"].dt.date

    st.dataframe(df_del[["index", "period", "accounts", "category", "amount"]], height=220)
    delete_id = st.number_input("Row ID to Delete", min_value=0, step=1)

    if st.button("🗑 Delete"):
        df_new = df_del.drop(index=delete_id).drop(columns=["index"])
        write_csv(df_new, f"Deleted row {delete_id}")
        st.success("Deleted Successfully")
        refresh()


def build_monthly_goal_view(df):
    goal_source = df.copy()
    goal_source["period"] = pd.to_datetime(goal_source["period"], errors="coerce")
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

    return monthly_goal_view.sort_values("month_ts").reset_index(drop=True)


def sync_goal_savings_export(monthly_goal_view):
    monthly_savings_export = build_monthly_savings_export(monthly_goal_view)

    try:
        sync_monthly_savings_export(monthly_savings_export)
    except Exception as exc:
        st.warning(f"Unable to sync monthly savings CSV to GitHub: {exc}")


def initialize_goal_tracker_items():
    if "goal_tracker_items" not in st.session_state:
        st.session_state["goal_tracker_items"] = DEFAULT_GOALS.copy()
    else:
        st.session_state["goal_tracker_items"] = prepare_goal_tracker_items(st.session_state["goal_tracker_items"])


def calculate_goal_results(goal_items, cumulative_savings, projected_monthly_contribution, current_goal_period):
    goal_results = goal_items.copy()
    goal_results = goal_results.dropna(subset=["Goal", "Target Amount", "Saved at Start", "Target Month"])
    goal_results = goal_results[goal_results["Goal"].astype(str).str.strip() != ""].copy()

    if goal_results.empty:
        return goal_results

    goal_results["Target Amount"] = pd.to_numeric(goal_results["Target Amount"], errors="coerce").fillna(0.0)
    goal_results["Saved at Start"] = pd.to_numeric(goal_results["Saved at Start"], errors="coerce").fillna(0.0)
    goal_results["Priority"] = goal_results["Priority"].fillna("Medium")
    goal_results["Target Month"] = goal_results["Target Month"].astype(str).str.strip()
    goal_results = goal_results[goal_results["Target Month"].str.match(r"^\d{4}-\d{2}$", na=False)].copy()

    if goal_results.empty:
        return goal_results

    goal_results["Auto Saved"] = allocate_savings_to_goals(goal_results, cumulative_savings).reindex(goal_results.index, fill_value=0.0)
    goal_results["Current Saved"] = (
        goal_results["Saved at Start"] + goal_results["Auto Saved"]
    ).clip(upper=goal_results["Target Amount"])
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
        required_monthly = (row["Remaining"] / months_left) if months_left > 0 else row["Remaining"]
        target_gap = projected_monthly_contribution - required_monthly

        target_months_left.append(months_left)
        required_monthly_list.append(required_monthly)
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

    return goal_results


def render_goal_summary(goal_results):
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


def render_goal_table(goal_results):
    goal_summary = goal_results.copy()
    goal_summary["_priority_sort"] = goal_summary["Priority"].map(PRIORITY_ORDER).fillna(9)
    goal_summary = goal_summary.sort_values(
        ["_priority_sort", "Target Month", "Remaining"],
        ascending=[True, True, False],
    ).drop(columns=["_priority_sort"])

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


def render_goal_highlights(goal_results):
    goal_cards = goal_results.assign(_priority_sort=goal_results["Priority"].map(PRIORITY_ORDER).fillna(9))
    goal_cards = goal_cards.sort_values(["_priority_sort", "Target Month", "Remaining"]).drop(columns=["_priority_sort"])

    progress_cols = st.columns(min(4, len(goal_cards)))
    for idx, (_, row) in enumerate(goal_cards.head(4).iterrows()):
        progress_cols[idx].metric(
            row["Goal"],
            f"{row['Progress %']:.1f}%",
            f"{row['Status']} • {format_currency(row['Remaining'])} left",
        )
        progress_cols[idx].progress(min(max(row["Progress %"] / 100, 0.0), 1.0))


def render_goal_breakdowns(goal_results):
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


def render_goal_tracking(df):
    st.markdown("<h3>🎯 Goal Tracking</h3>", unsafe_allow_html=True)
    st.caption(
        "Savings projections reset from April 2026 and use mapped income, including the assumption that income from May 2026 onward is ₹32,000."
    )

    monthly_goal_view = build_monthly_goal_view(df)
    sync_goal_savings_export(monthly_goal_view)

    if monthly_goal_view.empty:
        st.info("Goal tracking needs at least one month with mapped income.")
        return {
            "df": df.copy(),
            "monthly_goal_view": monthly_goal_view,
            "goal_results": pd.DataFrame(),
            "projected_monthly_contribution": 0.0,
            "current_goal_period": None,
        }

    recent_window = min(3, len(monthly_goal_view))
    avg_recent_savings = float(monthly_goal_view["savings"].tail(recent_window).mean())
    avg_all_savings = float(monthly_goal_view["savings"].mean())
    latest_savings = float(monthly_goal_view["savings"].iloc[-1])
    cumulative_savings = max(float(monthly_goal_view["savings"].sum()), 0.0)
    projected_monthly_contribution = max(avg_recent_savings, 0.0)
    current_goal_period = monthly_goal_view["year_month"].iloc[-1]

    initialize_goal_tracker_items()

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

    goal_results = calculate_goal_results(
        st.session_state["goal_tracker_items"],
        cumulative_savings=cumulative_savings,
        projected_monthly_contribution=projected_monthly_contribution,
        current_goal_period=current_goal_period,
    )

    if goal_results.empty:
        st.info("Add one or more goals to start tracking progress.")
        return {
            "df": df.copy(),
            "monthly_goal_view": monthly_goal_view,
            "goal_results": goal_results,
            "projected_monthly_contribution": projected_monthly_contribution,
            "current_goal_period": current_goal_period,
        }

    render_goal_summary(goal_results)
    render_goal_table(goal_results)
    render_goal_highlights(goal_results)
    render_goal_breakdowns(goal_results)

    savings_plot = monthly_goal_view[["month_ts", "savings", "income", "spend"]].rename(
        columns={"month_ts": "Month", "savings": "Savings", "income": "Income", "spend": "Spend"}
    ).set_index("Month")
    st.markdown("#### Savings Trend Supporting Goals")
    st.line_chart(savings_plot)

    return {
        "df": df.copy(),
        "monthly_goal_view": monthly_goal_view,
        "goal_results": goal_results,
        "projected_monthly_contribution": projected_monthly_contribution,
        "current_goal_period": current_goal_period,
    }


def _run_agent_prompt(session_key, prompt, finance_context, agent_mode):
    history = st.session_state.setdefault(session_key, [])
    history.append({"role": "user", "content": prompt})
    try:
        reply = run_finance_agent(
            prompt=prompt,
            finance_context=finance_context,
            history=history[:-1],
            agent_mode=agent_mode,
        )
    except Exception as exc:
        reply = f"AI request failed: {exc}"
    history.append({"role": "assistant", "content": reply})


def render_agentic_ai_section(finance_context):
    st.markdown("<h3>🤖 Agentic AI</h3>", unsafe_allow_html=True)
    st.caption(
        "Finance Copilot and Goal Planning Agent use your OpenRouter model from `.env` and the same transaction and goal data shown above."
    )

    is_configured, status_message = get_ai_configuration_status()
    if not is_configured:
        st.info(status_message)
        return

    st.caption(status_message)
    st.caption("The AI reads the full finance dataset plus the live goal tracker state. It suggests changes, but does not edit your data automatically.")

    finance_context = finance_context.copy()
    finance_context["active_filters"] = st.session_state.get("active_filter_summary", {})

    copilot_tab, planner_tab = st.tabs(["Finance Copilot", "Goal Planning Agent"])

    with copilot_tab:
        st.caption("Try: `Why did savings drop in July?`, `Which 3 categories hurt my goal progress most?`, or `Can I afford a ₹15k trip next month?`")
        copilot_history = st.session_state.setdefault("finance_copilot_history", [])

        for message in copilot_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        copilot_prompt = st.chat_input("Ask the finance copilot", key="finance_copilot_input")
        if copilot_prompt:
            with st.spinner("Finance Copilot is reviewing your data..."):
                _run_agent_prompt(
                    session_key="finance_copilot_history",
                    prompt=copilot_prompt,
                    finance_context=finance_context,
                    agent_mode="finance_copilot",
                )
            st.rerun()

    with planner_tab:
        st.caption("Use the buttons for quick planning runs, or ask your own goal-planning question.")
        planner_history = st.session_state.setdefault("goal_planner_history", [])

        button_cols = st.columns(4)
        preset_prompts = [
            ("Risk Review", "Review my goals, identify which ones are at risk, and explain why."),
            ("Target Months", "Suggest more realistic target months for goals that are currently at risk."),
            ("Monthly Plan", "Build a month-by-month savings plan for the next 6 months that protects high priority goals first."),
            ("Rebalance", "Suggest how to rebalance my savings allocations across goals while protecting high priority goals."),
        ]

        selected_prompt = None
        for idx, (label, prompt) in enumerate(preset_prompts):
            if button_cols[idx].button(label, key=f"goal_planner_preset_{idx}"):
                selected_prompt = prompt

        custom_prompt = st.text_area(
            "Ask the goal planner",
            value="",
            placeholder="Example: Give me a realistic savings plan for my emergency fund and trip goal.",
            key="goal_planner_custom_prompt",
        )
        run_custom_prompt = st.button("Run Goal Planner", key="goal_planner_custom_submit")

        if selected_prompt:
            with st.spinner("Goal Planning Agent is building a plan..."):
                _run_agent_prompt(
                    session_key="goal_planner_history",
                    prompt=selected_prompt,
                    finance_context=finance_context,
                    agent_mode="goal_planner",
                )
            st.rerun()

        if run_custom_prompt and custom_prompt.strip():
            with st.spinner("Goal Planning Agent is building a plan..."):
                _run_agent_prompt(
                    session_key="goal_planner_history",
                    prompt=custom_prompt.strip(),
                    finance_context=finance_context,
                    agent_mode="goal_planner",
                )
            st.session_state["goal_planner_custom_prompt"] = ""
            st.rerun()

        for message in planner_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])


def main():
    configure_page()
    load_custom_css()
    render_page_header()
    require_password()

    df = prepare_main_dataframe()
    filters = render_sidebar_filters(df)
    st.session_state["active_filter_summary"] = summarize_active_filters(filters)
    filtered = apply_sidebar_filters(df, filters)

    if filtered.empty:
        st.warning("No data available after applying filters.")
        st.stop()

    render_expense_entry(df)
    render_transactions_section(filtered)
    render_delete_section(df)
    render_kpis(filtered=filtered, df=df, MONTHLY_BUDGET=MONTHLY_BUDGET)
    finance_context = render_goal_tracking(df)
    render_agentic_ai_section(finance_context)


main()
