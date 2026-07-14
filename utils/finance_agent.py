import json
import os
from typing import Any

import pandas as pd
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode


DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_OPENROUTER_MODEL = "openai/gpt-oss-20b:free"


def _get_setting(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value:
        return str(value).strip()

    try:
        secret_value = st.secrets.get(name)
        if secret_value is None and "general" in st.secrets:
            secret_value = st.secrets["general"].get(name)
    except Exception:
        secret_value = None

    if secret_value is None:
        return default

    return str(secret_value).strip()


def get_ai_configuration_status() -> tuple[bool, str]:
    api_key = _get_setting("OPENROUTER_API_KEY")
    model = _get_setting("OPENROUTER_MODEL", DEFAULT_OPENROUTER_MODEL)

    if not api_key:
        return False, "Add OPENROUTER_API_KEY to `.env` or `.streamlit/secrets.toml` to enable Finance Copilot and Goal Planning Agent."

    return True, f"Using OpenRouter model `{model}`."


def _json_payload(payload: Any) -> str:
    return json.dumps(payload, default=str, ensure_ascii=True)


def _message_to_text(message: Any) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    return str(content)


def _history_to_messages(history: list[dict[str, str]] | None) -> list[Any]:
    messages = []
    for item in history or []:
        role = item.get("role")
        content = item.get("content", "")
        if role == "assistant":
            messages.append(AIMessage(content=content))
        elif role == "user":
            messages.append(HumanMessage(content=content))
    return messages


def _previous_month(month: str) -> str | None:
    try:
        return (pd.Period(month, freq="M") - 1).strftime("%Y-%m")
    except Exception:
        return None


def _future_months(start_month: str, count: int) -> list[str]:
    start_period = pd.Period(start_month, freq="M")
    return [(start_period + step).strftime("%Y-%m") for step in range(max(count, 0))]


def _build_finance_tools(context: dict[str, Any]):
    df = context["df"].copy()
    monthly_goal_view = context["monthly_goal_view"].copy()
    goal_results = context["goal_results"].copy()
    projected_monthly_contribution = float(context.get("projected_monthly_contribution", 0.0))
    current_goal_period = str(context.get("current_goal_period", ""))
    active_filters = context.get("active_filters", {})

    monthly_goal_view["year_month"] = monthly_goal_view["year_month"].astype(str)
    monthly_goal_lookup = monthly_goal_view.set_index("year_month").to_dict("index") if not monthly_goal_view.empty else {}
    available_months = sorted(df["year_month"].dropna().astype(str).unique().tolist())

    def category_totals_for_month(month: str) -> pd.Series:
        month_df = df[df["year_month"].astype(str) == str(month)].copy()
        if month_df.empty:
            return pd.Series(dtype=float)
        return (
            month_df.groupby("category")["amount"]
            .sum()
            .sort_values(ascending=False)
        )

    @tool
    def list_available_months() -> str:
        """List the months available in the finance dataset and the active sidebar filters."""
        return _json_payload(
            {
                "available_months": available_months,
                "active_filters": active_filters,
                "latest_month": available_months[-1] if available_months else None,
            }
        )

    @tool
    def get_monthly_summary(month: str) -> str:
        """Get earnings, expense, savings, and top spending categories for a month in YYYY-MM format."""
        month_key = str(month).strip()
        month_categories = category_totals_for_month(month_key)
        month_info = monthly_goal_lookup.get(month_key)

        if month_info is None and month_categories.empty:
            return _json_payload({"error": f"No data found for {month_key}.", "available_months": available_months[-6:]})

        if month_info is None:
            total_expense = float(month_categories.sum())
            total_income = 0.0
            total_savings = total_income - total_expense
        else:
            total_expense = float(month_info["spend"])
            total_income = float(month_info["income"])
            total_savings = float(month_info["savings"])

        return _json_payload(
            {
                "month": month_key,
                "earnings": round(total_income, 2),
                "expense": round(total_expense, 2),
                "savings": round(total_savings, 2),
                "top_categories": [
                    {"category": category, "amount": round(float(amount), 2)}
                    for category, amount in month_categories.head(8).items()
                ],
            }
        )

    @tool
    def compare_months(current_month: str, previous_month: str) -> str:
        """Compare two months in YYYY-MM format, including savings deltas and category changes."""
        current_key = str(current_month).strip()
        previous_key = str(previous_month).strip()

        current_summary = json.loads(get_monthly_summary.invoke({"month": current_key}))
        previous_summary = json.loads(get_monthly_summary.invoke({"month": previous_key}))

        if "error" in current_summary or "error" in previous_summary:
            return _json_payload({"error": "One or both months are unavailable.", "current": current_summary, "previous": previous_summary})

        current_categories = category_totals_for_month(current_key)
        previous_categories = category_totals_for_month(previous_key)
        category_delta = current_categories.sub(previous_categories, fill_value=0.0).sort_values(ascending=False)

        return _json_payload(
            {
                "current_month": current_key,
                "previous_month": previous_key,
                "earnings_delta": round(current_summary["earnings"] - previous_summary["earnings"], 2),
                "expense_delta": round(current_summary["expense"] - previous_summary["expense"], 2),
                "savings_delta": round(current_summary["savings"] - previous_summary["savings"], 2),
                "top_increases": [
                    {"category": category, "delta": round(float(delta), 2)}
                    for category, delta in category_delta.head(5).items()
                    if float(delta) > 0
                ],
                "top_decreases": [
                    {"category": category, "delta": round(float(delta), 2)}
                    for category, delta in category_delta.sort_values().head(5).items()
                    if float(delta) < 0
                ],
            }
        )

    @tool
    def top_category_drivers(month: str, top_n: int = 3) -> str:
        """Find the categories that most increased expense for a given month compared with the previous month."""
        month_key = str(month).strip()
        prev_key = _previous_month(month_key)
        current_categories = category_totals_for_month(month_key)

        if current_categories.empty:
            return _json_payload({"error": f"No category data found for {month_key}."})

        if not prev_key:
            return _json_payload(
                {
                    "month": month_key,
                    "drivers": [
                        {"category": category, "amount": round(float(amount), 2)}
                        for category, amount in current_categories.head(top_n).items()
                    ],
                    "note": "Previous month unavailable, so this is the top spend list.",
                }
            )

        previous_categories = category_totals_for_month(prev_key)
        deltas = current_categories.sub(previous_categories, fill_value=0.0).sort_values(ascending=False)

        return _json_payload(
            {
                "month": month_key,
                "previous_month": prev_key,
                "drivers": [
                    {"category": category, "delta": round(float(delta), 2)}
                    for category, delta in deltas.head(top_n).items()
                ],
            }
        )

    @tool
    def get_goal_snapshot() -> str:
        """Get the current goal tracking snapshot, including statuses, target months, and required monthly savings."""
        if goal_results.empty:
            return _json_payload({"error": "No goal data is available yet."})

        return _json_payload(
            {
                "projection_basis_month": current_goal_period,
                "projected_monthly_contribution": round(projected_monthly_contribution, 2),
                "goals": [
                    {
                        "goal": str(row["Goal"]),
                        "priority": str(row["Priority"]),
                        "target_month": str(row["Target Month"]),
                        "status": str(row["Status"]),
                        "target_amount": round(float(row["Target Amount"]), 2),
                        "current_saved": round(float(row["Current Saved"]), 2),
                        "remaining": round(float(row["Remaining"]), 2),
                        "required_per_month": round(float(row["Required / Month"]), 2),
                        "suggested_monthly_savings": round(float(row["Suggested Monthly Savings"]), 2),
                        "projected_completion": str(row["Projected Completion"]),
                    }
                    for _, row in goal_results.iterrows()
                ],
            }
        )

    @tool
    def check_purchase_affordability(amount: float, purpose: str = "general purchase") -> str:
        """Estimate whether a planned purchase can fit next month without seriously hurting goal progress."""
        amount = max(float(amount), 0.0)

        if goal_results.empty:
            return _json_payload(
                {
                    "purpose": purpose,
                    "purchase_amount": round(amount, 2),
                    "projected_monthly_savings": round(projected_monthly_contribution, 2),
                    "affordable": amount <= projected_monthly_contribution,
                    "note": "No goal tracker data is available, so this is based only on recent savings pace.",
                }
            )

        active_goals = goal_results[goal_results["Status"] != "Completed"].copy()
        high_priority_required = float(
            active_goals.loc[active_goals["Priority"] == "High", "Required / Month"].sum()
        )
        all_required = float(active_goals["Required / Month"].sum())
        after_high = projected_monthly_contribution - high_priority_required
        after_all = projected_monthly_contribution - all_required

        if amount <= max(after_all, 0.0):
            comfort = "Comfortable"
        elif amount <= max(after_high, 0.0):
            comfort = "Possible, but it could pressure lower-priority goals."
        elif amount <= projected_monthly_contribution:
            comfort = "Tight. It fits the pace only if you delay goal contributions."
        else:
            comfort = "Risky. It exceeds the projected monthly savings pace."

        return _json_payload(
            {
                "purpose": purpose,
                "purchase_amount": round(amount, 2),
                "projected_monthly_savings": round(projected_monthly_contribution, 2),
                "high_priority_goal_commitment": round(high_priority_required, 2),
                "all_goal_commitment": round(all_required, 2),
                "available_after_high_priority_goals": round(after_high, 2),
                "available_after_all_goals": round(after_all, 2),
                "affordable": bool(amount <= max(after_high, 0.0)),
                "assessment": comfort,
            }
        )

    @tool
    def build_goal_plan(months: int = 6) -> str:
        """Build a month-by-month savings plan based on current goal priorities and the recent savings pace."""
        if goal_results.empty:
            return _json_payload({"error": "No goal data is available yet."})

        plan_months = max(1, min(int(months), 12))
        working_goals = goal_results.copy()
        working_goals["_priority_sort"] = working_goals["Priority"].map({"High": 0, "Medium": 1, "Low": 2}).fillna(9)
        working_goals = working_goals.sort_values(["_priority_sort", "Target Month", "Remaining", "Goal"]).reset_index(drop=True)

        remaining_by_goal = {
            str(row["Goal"]): max(float(row["Remaining"]), 0.0)
            for _, row in working_goals.iterrows()
        }
        plan_rows = []

        for month_label in _future_months(current_goal_period, plan_months):
            month_budget = max(projected_monthly_contribution, 0.0)
            allocations = []

            for _, row in working_goals.iterrows():
                goal_name = str(row["Goal"])
                remaining_need = remaining_by_goal.get(goal_name, 0.0)
                if remaining_need <= 0 or month_budget <= 0:
                    continue

                allocation = min(remaining_need, month_budget)
                remaining_by_goal[goal_name] = max(remaining_need - allocation, 0.0)
                month_budget -= allocation
                allocations.append(
                    {
                        "goal": goal_name,
                        "allocated": round(float(allocation), 2),
                        "remaining_after_allocation": round(float(remaining_by_goal[goal_name]), 2),
                    }
                )

            plan_rows.append(
                {
                    "month": month_label,
                    "planned_savings": round(float(projected_monthly_contribution - month_budget), 2),
                    "unallocated_buffer": round(float(month_budget), 2),
                    "allocations": allocations,
                }
            )

        recommendations = []
        for _, row in working_goals.iterrows():
            recommended_target = row["Projected Completion"]
            if row["Status"] == "At Risk" and projected_monthly_contribution > 0:
                recommended_target = str(row["Projected Completion"])
            recommendations.append(
                {
                    "goal": str(row["Goal"]),
                    "status": str(row["Status"]),
                    "current_target_month": str(row["Target Month"]),
                    "recommended_target_month": recommended_target,
                    "required_per_month": round(float(row["Required / Month"]), 2),
                    "projected_completion": str(row["Projected Completion"]),
                }
            )

        return _json_payload(
            {
                "projection_basis_month": current_goal_period,
                "projected_monthly_contribution": round(projected_monthly_contribution, 2),
                "months_planned": plan_months,
                "monthly_plan": plan_rows,
                "goal_recommendations": recommendations,
            }
        )

    return [
        list_available_months,
        get_monthly_summary,
        compare_months,
        top_category_drivers,
        get_goal_snapshot,
        check_purchase_affordability,
        build_goal_plan,
    ]


def _build_system_prompt(agent_mode: str) -> str:
    if agent_mode == "goal_planner":
        return (
            "You are a goal planning agent for a personal finance dashboard. "
            "Use the available tools to inspect savings pace, goal risks, and target dates. "
            "Give numeric, practical recommendations. Protect high-priority goals first. "
            "When suggesting changes, explain tradeoffs clearly. "
            "Never invent financial numbers; only use tool results."
        )

    return (
        "You are a finance copilot for a personal finance dashboard. "
        "Use tools before answering questions about savings, monthly trends, categories, affordability, or goals. "
        "Always cite the specific months and amounts you relied on. "
        "Be concise, practical, and grounded in the tool outputs."
    )


def _build_llm(system_prompt: str, tools: list[Any]):
    api_key = _get_setting("OPENROUTER_API_KEY")
    base_url = _get_setting("OPENROUTER_BASE_URL", DEFAULT_OPENROUTER_BASE_URL)
    model = _get_setting("OPENROUTER_MODEL", DEFAULT_OPENROUTER_MODEL)

    llm = ChatOpenAI(
        api_key=api_key,
        base_url=base_url,
        model=model,
        temperature=0.1,
    )
    llm_with_tools = llm.bind_tools(tools)

    def call_model(state: MessagesState):
        response = llm_with_tools.invoke([SystemMessage(content=system_prompt), *state["messages"]])
        return {"messages": [response]}

    return call_model


def _should_continue(state: MessagesState):
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "tool_node"
    return END


def run_finance_agent(
    prompt: str,
    finance_context: dict[str, Any],
    history: list[dict[str, str]] | None = None,
    agent_mode: str = "finance_copilot",
) -> str:
    tools = _build_finance_tools(finance_context)
    system_prompt = _build_system_prompt(agent_mode)
    llm_call = _build_llm(system_prompt, tools)
    tool_node = ToolNode(tools)

    graph_builder = StateGraph(MessagesState)
    graph_builder.add_node("llm_call", llm_call)
    graph_builder.add_node("tool_node", tool_node)
    graph_builder.add_edge(START, "llm_call")
    graph_builder.add_conditional_edges("llm_call", _should_continue, ["tool_node", END])
    graph_builder.add_edge("tool_node", "llm_call")
    graph = graph_builder.compile()

    history_messages = _history_to_messages(history)
    result = graph.invoke({"messages": [*history_messages, HumanMessage(content=prompt)]})

    for message in reversed(result["messages"]):
        if isinstance(message, AIMessage):
            text = _message_to_text(message).strip()
            if text:
                return text

    return "I could not generate a grounded answer from the available finance data."
