import streamlit as st
from dotenv import load_dotenv

from app import build_finance_context, load_custom_css, prepare_main_dataframe, render_agentic_ai_section


def render_context_snapshot(finance_context):
    monthly_goal_view = finance_context["monthly_goal_view"]
    goal_results = finance_context["goal_results"]
    projected_monthly_contribution = float(finance_context["projected_monthly_contribution"])

    snapshot_cols = st.columns(4)
    snapshot_cols[0].metric("Months Loaded", len(monthly_goal_view))
    snapshot_cols[1].metric("Projected Monthly Savings", f"₹{projected_monthly_contribution:,.0f}")
    snapshot_cols[2].metric("Active Goals", int((goal_results["Status"] != "Completed").sum()) if not goal_results.empty else 0)
    snapshot_cols[3].metric("At-Risk Goals", int((goal_results["Status"] == "At Risk").sum()) if not goal_results.empty else 0)


def main():
    load_dotenv(dotenv_path=".env", override=False)
    st.set_page_config(page_title="AI Chatbot", page_icon="🤖", layout="wide")
    load_custom_css()

    st.title("🤖 Finance AI Chatbot")
    st.caption("Use a dedicated page for Finance Copilot and Goal Planning Agent without scrolling through the main dashboard.")

    df = prepare_main_dataframe()
    st.session_state["active_filter_summary"] = {}
    finance_context = build_finance_context(df)

    if finance_context["monthly_goal_view"].empty:
        st.warning("The chatbot needs at least one month with mapped income before it can answer finance questions.")
        st.stop()

    render_context_snapshot(finance_context)
    st.markdown("---")
    render_agentic_ai_section(finance_context)


if __name__ == "__main__":
    main()
