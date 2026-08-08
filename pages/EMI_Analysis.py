import math
import os
from datetime import date
from decimal import Decimal, ROUND_HALF_UP

import altair as alt
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# Try to import GitHub storage utilities, but handle gracefully if they don't exist
try:
    from utils.github_storage import write_emi_analysis_csv, read_emi_analysis_csv
    GITHUB_AVAILABLE = True
except ImportError:
    GITHUB_AVAILABLE = False
    # Define placeholder functions
    def write_emi_analysis_csv(*args, **kwargs):
        return None
    def read_emi_analysis_csv(*args, **kwargs):
        return None


# -----------------------------------------------------------
# HELPERS
# -----------------------------------------------------------
def round_half_up(value, digits=0):
    quant = "1" if digits == 0 else f"1.{'0' * digits}"
    return float(Decimal(str(value)).quantize(Decimal(quant), rounding=ROUND_HALF_UP))


def ceil_to_rupee(value):
    return float(math.ceil(value))


def format_currency(value, digits=0):
    return f"\u20b9{value:,.{digits}f}"


def calculate_emi(principal, annual_rate, tenure_months):
    if tenure_months <= 0:
        return 0.0

    if annual_rate <= 0:
        return principal / tenure_months

    monthly_rate = annual_rate / 1200
    factor = (1 + monthly_rate) ** tenure_months
    return principal * monthly_rate * factor / (factor - 1)


def calculate_remaining_months(balance, monthly_rate, emi_amount):
    if balance <= 0.01:
        return 0

    if emi_amount <= 0:
        return math.inf

    if monthly_rate <= 0:
        return math.ceil(balance / emi_amount)

    minimum_interest = balance * monthly_rate
    if emi_amount <= minimum_interest:
        return math.inf

    months = math.log(emi_amount / (emi_amount - minimum_interest)) / math.log(1 + monthly_rate)
    return math.ceil(months)


def normalise_part_payments(payment_df):
    cleaned = payment_df.copy()

    if cleaned.empty:
        return []

    cleaned["Payment Month"] = pd.to_numeric(cleaned["Payment Month"], errors="coerce")
    cleaned["Payment Amount"] = pd.to_numeric(cleaned["Payment Amount"], errors="coerce")
    cleaned = cleaned.dropna(subset=["Payment Month", "Payment Amount"])
    cleaned = cleaned[(cleaned["Payment Month"] >= 1) & (cleaned["Payment Amount"] > 0)]

    if cleaned.empty:
        return []

    cleaned["Payment Month"] = cleaned["Payment Month"].astype(int)

    return [
        {
            "row_id": int(idx),
            "month": int(row["Payment Month"]),
            "amount": float(row["Payment Amount"]),
        }
        for idx, row in cleaned.iterrows()
    ]


def calculate_completed_emis(start_date_value, as_of_date_value, total_months):
    start_ts = pd.Timestamp(start_date_value).normalize()
    as_of_ts = pd.Timestamp(as_of_date_value).normalize()

    if as_of_ts < start_ts:
        return 0

    month_gap = (as_of_ts.year - start_ts.year) * 12 + (as_of_ts.month - start_ts.month)
    completed = month_gap + (1 if as_of_ts.day >= start_ts.day else 0)

    return max(0, min(int(completed), int(total_months)))


def add_schedule_dates(schedule_df, start_date_value):
    if schedule_df.empty:
        return schedule_df

    dated_df = schedule_df.copy()
    start_ts = pd.Timestamp(start_date_value).normalize()
    dated_df.insert(
        1,
        "EMI Date",
        [(start_ts + pd.DateOffset(months=month - 1)).date() for month in dated_df["Month"]],
    )
    return dated_df


def build_month_options(schedule_df):
    month_options = []

    for _, row in schedule_df.iterrows():
        emi_date = pd.Timestamp(row["EMI Date"]).strftime("%d %b %Y")
        label = f"Month {int(row['Month'])} - {emi_date}"
        month_options.append((label, int(row["Month"])))

    return month_options


def build_schedule(
    principal,
    annual_rate,
    tenure_months,
    part_payments=None,
    monthly_extra_payment=0.0,
):
    part_payments = sorted(part_payments or [], key=lambda item: (item["month"], item["row_id"]))
    monthly_rate = annual_rate / 1200
    starting_emi = ceil_to_rupee(calculate_emi(principal, annual_rate, tenure_months))
    current_emi = starting_emi
    balance = float(principal)
    monthly_extra_payment = max(float(monthly_extra_payment), 0.0)

    payments_by_month = {}
    for payment in part_payments:
        payments_by_month.setdefault(payment["month"], []).append(payment)

    schedule_rows = []
    payment_events = []
    applied_row_ids = set()
    max_months = max(tenure_months + len(part_payments) + 24, tenure_months * 3)

    for month in range(1, max_months + 1):
        if balance <= 0.01:
            break

        opening_balance = balance
        interest_paid = opening_balance * monthly_rate if monthly_rate > 0 else 0.0
        emi_paid = min(current_emi, opening_balance + interest_paid)
        principal_paid = max(emi_paid - interest_paid, 0.0)
        balance = max(opening_balance + interest_paid - emi_paid, 0.0)

        recurring_extra_paid = min(monthly_extra_payment, balance)
        balance = max(balance - recurring_extra_paid, 0.0)
        total_part_payment = recurring_extra_paid
        month_events = payments_by_month.get(month, [])

        for payment in month_events:
            applied_amount = min(payment["amount"], balance)
            balance = max(balance - applied_amount, 0.0)
            total_part_payment += applied_amount
            applied_row_ids.add(payment["row_id"])

            revised_total_tenure = month + calculate_remaining_months(balance, monthly_rate, current_emi)

            payment_events.append(
                {
                    "Input Row": payment["row_id"] + 1,
                    "Payment Month": month,
                    "Part Payment": applied_amount,
                    "Balance After Payment": balance,
                    "EMI After Payment": current_emi if balance > 0.01 else 0.0,
                    "Projected Total Tenure": revised_total_tenure,
                    "Months Saved": max(tenure_months - revised_total_tenure, 0),
                }
            )

        emi_for_next_month = current_emi if balance > 0.01 else 0.0

        schedule_rows.append(
            {
                "Month": month,
                "Opening Principal": opening_balance,
                "EMI Paid": emi_paid,
                "Principal Paid": principal_paid,
                "Interest Paid": interest_paid,
                "Extra EMI Payment": recurring_extra_paid,
                "Part Payment": total_part_payment,
                "Closing Principal": balance,
                "EMI Next Month": emi_for_next_month,
            }
        )

    schedule_df = pd.DataFrame(schedule_rows)
    events_df = pd.DataFrame(payment_events)
    unused_payments = [payment for payment in part_payments if payment["row_id"] not in applied_row_ids]

    return schedule_df, events_df, starting_emi, current_emi, unused_payments


def prepare_schedule_display(schedule_df):
    if schedule_df.empty:
        return schedule_df

    display_df = schedule_df.copy()
    currency_columns = [
        "Opening Principal",
        "EMI Paid",
        "Principal Paid",
        "Interest Paid",
        "Extra EMI Payment",
        "Part Payment",
        "Closing Principal",
        "EMI Next Month",
    ]

    for column in currency_columns:
        display_df[column] = display_df[column].round(2)

    return display_df


def prepare_event_display(events_df):
    if events_df.empty:
        return events_df

    display_df = events_df.copy()
    for column in ["Part Payment", "Balance After Payment", "EMI After Payment"]:
        display_df[column] = display_df[column].round(2)

    return display_df


def build_balance_chart(balance_df, original_months, updated_months):
    colors = alt.Scale(domain=["Original", "Updated"], range=["#d6a84a", "#5ad0ff"])

    base = (
        alt.Chart(balance_df)
        .mark_line(point=True, strokeWidth=3)
        .encode(
            x=alt.X("Month:Q", title="Month"),
            y=alt.Y("Closing Principal:Q", title="Outstanding Balance"),
            color=alt.Color("Scenario:N", scale=colors),
            tooltip=[
                alt.Tooltip("Scenario:N"),
                alt.Tooltip("Month:Q"),
                alt.Tooltip("Closing Principal:Q", format=",.2f"),
            ],
        )
        .properties(height=360)
        .interactive()
    )

    payoff_markers = pd.DataFrame(
        {
            "Month": [original_months, updated_months],
            "Scenario": ["Original", "Updated"],
        }
    )

    rules = (
        alt.Chart(payoff_markers)
        .mark_rule(strokeDash=[5, 4], opacity=0.65)
        .encode(
            x="Month:Q",
            color=alt.Color("Scenario:N", scale=colors, legend=None),
            tooltip=["Scenario:N", alt.Tooltip("Month:Q", title="Payoff Month")],
        )
    )

    return base + rules


def build_cumulative_interest_chart(interest_df):
    colors = alt.Scale(domain=["Original", "Updated"], range=["#ffb84d", "#7ce38b"])

    return (
        alt.Chart(interest_df)
        .mark_line(point=True, strokeWidth=3)
        .encode(
            x=alt.X("Month:Q", title="Month"),
            y=alt.Y("Cumulative Interest:Q", title="Cumulative Interest Paid"),
            color=alt.Color("Scenario:N", scale=colors),
            tooltip=[
                alt.Tooltip("Scenario:N"),
                alt.Tooltip("Month:Q"),
                alt.Tooltip("Cumulative Interest:Q", format=",.2f"),
            ],
        )
        .properties(height=340)
        .interactive()
    )


# -----------------------------------------------------------
# GITHUB STORAGE FUNCTIONS (with fallback)
# -----------------------------------------------------------
def save_part_payments_to_github(part_payments_df):
    """Save part payments to GitHub with commit message"""
    if not GITHUB_AVAILABLE:
        return False, "GitHub storage not available"
    
    try:
        from utils.github_storage import write_emi_part_payments_csv
        timestamp = pd.Timestamp.now(tz="Asia/Kolkata").strftime("%Y-%m-%d %H:%M:%S")
        write_emi_part_payments_csv(
            part_payments_df,
            commit_message=f"Part payments updated on {timestamp}"
        )
        return True, "Part payments saved to GitHub!"
    except Exception as e:
        return False, f"Failed to save part payments: {str(e)}"


def load_part_payments_from_github():
    """Load part payments from GitHub"""
    if not GITHUB_AVAILABLE:
        return pd.DataFrame(columns=["Payment Month", "Payment Amount"])
    
    try:
        from utils.github_storage import read_emi_part_payments_csv
        df = read_emi_part_payments_csv()
        if df is not None and not df.empty:
            return df
        return pd.DataFrame(columns=["Payment Month", "Payment Amount"])
    except Exception as e:
        # Silently fail - just return empty dataframe
        return pd.DataFrame(columns=["Payment Month", "Payment Amount"])


def save_emi_analysis_to_github(
    comparison_df,
    original_schedule,
    updated_schedule,
    payment_events,
    loan_amount,
    interest_rate,
    tenure_months,
    original_emi,
    monthly_extra_payment,
    part_payments,
    loan_start_date,
    as_of_date,
):
    """Save EMI analysis results to GitHub"""
    if not GITHUB_AVAILABLE:
        return False, "GitHub storage not available"
    
    timestamp = pd.Timestamp.now(tz="Asia/Kolkata").strftime("%Y-%m-%d %H:%M:%S")
    
    summary_data = {
        "Analysis Date": timestamp,
        "Loan Amount": loan_amount,
        "Interest Rate (%)": interest_rate,
        "Tenure (Months)": tenure_months,
        "Original EMI": original_emi,
        "Monthly Extra Payment": monthly_extra_payment,
        "Loan Start Date": pd.Timestamp(loan_start_date).strftime("%Y-%m-%d"),
        "As Of Date": pd.Timestamp(as_of_date).strftime("%Y-%m-%d"),
        "Number of Part Payments": len(part_payments),
        "Total Part Payments": sum(p["amount"] for p in part_payments),
        "Updated Tenure (Months)": len(updated_schedule),
        "Original Tenure (Months)": len(original_schedule),
        "Tenure Reduction (Months)": len(original_schedule) - len(updated_schedule),
        "Original Total Interest": original_schedule["Interest Paid"].sum() if not original_schedule.empty else 0,
        "Updated Total Interest": updated_schedule["Interest Paid"].sum() if not updated_schedule.empty else 0,
        "Interest Saved": max(original_schedule["Interest Paid"].sum() - updated_schedule["Interest Paid"].sum(), 0) if not original_schedule.empty and not updated_schedule.empty else 0,
        "Loan Closed Date": updated_schedule.iloc[-1]["EMI Date"] if not updated_schedule.empty else None,
    }
    
    summary_df = pd.DataFrame([summary_data])
    
    original_schedule_export = original_schedule.copy()
    if not original_schedule_export.empty:
        original_schedule_export["Scenario"] = "Original"
        original_schedule_export["EMI Date"] = original_schedule_export["EMI Date"].astype(str)
    
    updated_schedule_export = updated_schedule.copy()
    if not updated_schedule_export.empty:
        updated_schedule_export["Scenario"] = "Updated"
        updated_schedule_export["EMI Date"] = updated_schedule_export["EMI Date"].astype(str)
    
    comparison_export = comparison_df.copy()
    for col in comparison_export.columns:
        if col != "Month" and pd.api.types.is_numeric_dtype(comparison_export[col]):
            comparison_export[col] = comparison_export[col].round(2)
    comparison_export["EMI Date"] = comparison_export["EMI Date"].astype(str) if "EMI Date" in comparison_export.columns else None
    
    payment_events_export = payment_events.copy()
    if not payment_events_export.empty:
        for col in payment_events_export.columns:
            if col != "Input Row" and pd.api.types.is_numeric_dtype(payment_events_export[col]):
                payment_events_export[col] = payment_events_export[col].round(2)
    
    part_payments_df = pd.DataFrame(part_payments) if part_payments else pd.DataFrame(columns=["row_id", "month", "amount"])
    
    try:
        from utils.github_storage import write_emi_analysis_csv
        write_emi_analysis_csv(
            summary_df=summary_df,
            comparison_df=comparison_export,
            original_schedule=original_schedule_export,
            updated_schedule=updated_schedule_export,
            payment_events=payment_events_export,
            part_payments=part_payments_df,
            commit_message=f"EMI Analysis saved on {timestamp}"
        )
        return True, "EMI analysis saved to GitHub successfully!"
    except Exception as e:
        return False, f"Failed to save to GitHub: {str(e)}"


def load_emi_analysis_from_github():
    """Load saved EMI analysis from GitHub"""
    if not GITHUB_AVAILABLE:
        return None
    
    try:
        from utils.github_storage import read_emi_analysis_csv
        return read_emi_analysis_csv()
    except Exception as e:
        return None


# -----------------------------------------------------------
# PAGE SETUP
# -----------------------------------------------------------
load_dotenv(dotenv_path=".env", override=False)

st.set_page_config(
    layout="wide",
    page_title="EMI Analysis",
    page_icon="💰",
)

css_path = ".streamlit/styles.css"
if os.path.exists(css_path):
    with open(css_path, encoding="utf-8") as css_file:
        st.markdown(f"<style>{css_file.read()}</style>", unsafe_allow_html=True)

st.markdown("<h1 class='page-title'>💰 EMI Analysis &amp; Part Payment Planner</h1>", unsafe_allow_html=True)
st.caption("Model how extra payments change your balance, interest outflow, EMI, and loan closure date.")

# -----------------------------------------------------------
# INITIALIZE DEFAULTS
# -----------------------------------------------------------
default_payments = pd.DataFrame(
    {
        "Payment Month": pd.Series(dtype="int"),
        "Payment Amount": pd.Series(dtype="float"),
    }
)

# -----------------------------------------------------------
# LOAD SAVED PART PAYMENTS ON STARTUP
# -----------------------------------------------------------
if "emi_part_payments" not in st.session_state:
    # Try to load from GitHub on startup
    saved_payments = load_part_payments_from_github()
    if saved_payments is not None and not saved_payments.empty:
        st.session_state["emi_part_payments"] = saved_payments
        st.sidebar.success("✅ Loaded saved part payments from GitHub")
    else:
        st.session_state["emi_part_payments"] = default_payments.copy()

# -----------------------------------------------------------
# INPUTS
# -----------------------------------------------------------
st.markdown("### EMI Overview")

input_col1, input_col2, input_col3, input_col4 = st.columns([1.25, 1, 1, 1.4])

with input_col1:
    loan_amount = st.number_input("Loan Amount", min_value=1000.0, value=194351.0, step=1000.0)

with input_col2:
    interest_rate = st.number_input("Interest Rate (% p.a.)", min_value=0.0, value=14.51, step=0.05)

with input_col3:
    tenure_months = st.number_input("Tenure (Months)", min_value=1, value=48, step=1)

with input_col4:
    add_extra_with_emi = st.checkbox("Add part payment with each EMI", value=False)
    monthly_extra_payment = (
        st.number_input("Extra Amount / Month", min_value=0.0, value=1000.0, step=500.0)
        if add_extra_with_emi
        else 0.0
    )

date_col1, date_col2, date_col3 = st.columns([1, 1, 1.9])

with date_col1:
    loan_start_date = st.date_input("Loan Start Date", value=date(2026, 4, 3))

with date_col2:
    as_of_date = st.date_input("Auto Update As Of", value=pd.Timestamp.today().date())

with date_col3:
    st.markdown("")
    st.caption(
        "EMI auto-posting rule: one EMI is treated as paid on the 3rd of every month, starting April 3, 2026. One-time lump-sum part payments keep EMI fixed and reduce tenure."
    )

original_schedule, _, original_emi, _, _ = build_schedule(
    principal=loan_amount,
    annual_rate=interest_rate,
    tenure_months=tenure_months,
    part_payments=[],
    monthly_extra_payment=0.0,
)

original_total_interest = original_schedule["Interest Paid"].sum()
original_total_payment = original_schedule["EMI Paid"].sum()
original_schedule = add_schedule_dates(original_schedule, loan_start_date)
month_options = build_month_options(original_schedule)

overview_cols = st.columns(7)
overview_cols[0].metric("Loan Amount", format_currency(loan_amount, 0))
overview_cols[1].metric("Interest Rate", f"{interest_rate:.2f}%")
overview_cols[2].metric("Tenure", f"{tenure_months} months")
overview_cols[3].metric("EMI Amount", format_currency(original_emi, 0))
overview_cols[4].metric("Extra With EMI", format_currency(monthly_extra_payment, 0))
overview_cols[5].metric("Total Interest Payable", format_currency(original_total_interest, 0))
overview_cols[6].metric("Total Payment", format_currency(original_total_payment, 0))

st.caption(
    "Assumption: any extra amount added with EMI and any one-time lump-sum part payment are both applied after the regular EMI of that month."
)

# -----------------------------------------------------------
# PART PAYMENT INPUTS
# -----------------------------------------------------------
st.divider()
st.markdown("### Part Payment Inputs")
st.caption(
    "Add lump-sum part payments using the form below. Each payment will reduce your loan tenure while keeping EMI fixed."
)

add_payment_col1, add_payment_col2, add_payment_col3 = st.columns([1.8, 1, 0.8])

with add_payment_col1:
    selected_month_label = st.selectbox(
        "Select Month For Lump-Sum Payment",
        options=[label for label, _ in month_options],
        index=0,
    )

with add_payment_col2:
    selected_lumpsum_amount = st.number_input(
        "Lump-Sum Amount",
        min_value=0.0,
        value=10000.0,
        step=1000.0,
    )

with add_payment_col3:
    st.markdown("")
    st.markdown("")
    if st.button("Add Lump-Sum", type="primary"):
        selected_month_value = dict(month_options)[selected_month_label]
        if selected_lumpsum_amount > 0:
            updated_entries = pd.concat(
                [
                    st.session_state["emi_part_payments"],
                    pd.DataFrame(
                        [{"Payment Month": selected_month_value, "Payment Amount": selected_lumpsum_amount}]
                    ),
                ],
                ignore_index=True,
            )
            st.session_state["emi_part_payments"] = updated_entries
            
            # Auto-save to GitHub when adding a payment
            if GITHUB_AVAILABLE:
                success, message = save_part_payments_to_github(updated_entries)
                if success:
                    st.success(f"✅ Part payment added and saved to GitHub")
                else:
                    st.warning(f"⚠️ Part payment added but could not save to GitHub: {message}")
            else:
                st.info("💡 GitHub storage not configured - part payments saved in session only")
            
            st.rerun()

# Display current payments with delete buttons
if not st.session_state["emi_part_payments"].empty:
    st.markdown("#### Current Part Payments")
    
    # Create columns for header
    header_cols = st.columns([2, 2, 0.8])
    with header_cols[0]:
        st.markdown("**Month**")
    with header_cols[1]:
        st.markdown("**Amount**")
    with header_cols[2]:
        st.markdown("**Action**")
    
    # Display each payment with delete button
    for idx in st.session_state["emi_part_payments"].index:
        row_cols = st.columns([2, 2, 0.8])
        with row_cols[0]:
            month_val = int(st.session_state["emi_part_payments"].loc[idx, "Payment Month"])
            # Find the month label
            month_label = next((label for label, val in month_options if val == month_val), f"Month {month_val}")
            st.write(month_label)
        with row_cols[1]:
            st.write(format_currency(st.session_state["emi_part_payments"].loc[idx, "Payment Amount"], 0))
        with row_cols[2]:
            if st.button("🗑️", key=f"delete_{idx}"):
                # Remove the payment
                updated_df = st.session_state["emi_part_payments"].drop(idx).reset_index(drop=True)
                st.session_state["emi_part_payments"] = updated_df
                
                # Auto-save to GitHub after deletion
                if GITHUB_AVAILABLE:
                    success, message = save_part_payments_to_github(updated_df)
                    if success:
                        st.success(f"✅ Part payment deleted and saved to GitHub")
                    else:
                        st.warning(f"⚠️ Part payment deleted but could not save to GitHub: {message}")
                else:
                    st.info("💡 Part payment deleted from session")
                
                st.rerun()
    
    st.markdown("---")
    
    # Clear all button
    clear_col1, clear_col2, clear_col3 = st.columns([1, 2, 3])
    with clear_col1:
        if st.button("🗑️ Clear All Payments", type="secondary"):
            st.session_state["emi_part_payments"] = default_payments.copy()
            if GITHUB_AVAILABLE:
                success, message = save_part_payments_to_github(default_payments.copy())
                if success:
                    st.success("✅ All payments cleared and saved to GitHub")
                else:
                    st.warning(f"⚠️ Payments cleared but could not save to GitHub: {message}")
            else:
                st.info("💡 All payments cleared from session")
            st.rerun()

# Add manual save/load buttons if GitHub is available
if GITHUB_AVAILABLE:
    save_col1, save_col2, save_col3 = st.columns([1, 2, 3])
    with save_col1:
        if st.button("💾 Save Part Payments", type="primary"):
            with st.spinner("Saving to GitHub..."):
                success, message = save_part_payments_to_github(st.session_state["emi_part_payments"])
                if success:
                    st.success(f"✅ {message}")
                    st.balloons()
                else:
                    st.error(f"❌ {message}")

    with save_col2:
        if st.button("🔄 Load Saved Payments"):
            saved_payments = load_part_payments_from_github()
            if saved_payments is not None and not saved_payments.empty:
                st.session_state["emi_part_payments"] = saved_payments
                st.success("✅ Loaded saved part payments from GitHub")
                st.rerun()
            else:
                st.warning("No saved part payments found")
    
    st.caption("💡 Part payments are automatically saved to GitHub when you add or delete them.")
else:
    st.info("💡 GitHub storage not configured. Part payments will be saved in session only and will not persist after restart.")

part_payments = normalise_part_payments(st.session_state["emi_part_payments"])

# -----------------------------------------------------------
# CALCULATIONS
# -----------------------------------------------------------
updated_schedule, payment_events, _, latest_emi, unused_payments = build_schedule(
    principal=loan_amount,
    annual_rate=interest_rate,
    tenure_months=tenure_months,
    part_payments=part_payments,
    monthly_extra_payment=monthly_extra_payment,
)

updated_total_interest = updated_schedule["Interest Paid"].sum()
updated_regular_payment = updated_schedule["EMI Paid"].sum()
recurring_extra_total = updated_schedule["Extra EMI Payment"].sum()
total_part_payment = updated_schedule["Part Payment"].sum()
one_time_part_payment_total = max(total_part_payment - recurring_extra_total, 0.0)
updated_total_payment = updated_regular_payment + total_part_payment
updated_schedule = add_schedule_dates(updated_schedule, loan_start_date)

original_months = int(len(original_schedule))
updated_months = int(len(updated_schedule))
tenure_reduction = max(original_months - updated_months, 0)
interest_saved = max(original_total_interest - updated_total_interest, 0.0)
revised_emi = original_emi
completed_emis = calculate_completed_emis(loan_start_date, as_of_date, updated_months)

if completed_emis > 0:
    current_outstanding = float(updated_schedule.iloc[completed_emis - 1]["Closing Principal"])
else:
    current_outstanding = float(loan_amount)

if completed_emis < updated_months:
    next_emi_row = updated_schedule.iloc[completed_emis]
    next_emi_date = next_emi_row["EMI Date"]
    next_total_due = float(next_emi_row["EMI Paid"] + next_emi_row["Part Payment"])
    next_base_emi = float(next_emi_row["EMI Paid"])
    next_extra_emi = float(next_emi_row["Extra EMI Payment"])
    next_one_time_part_payment = max(float(next_emi_row["Part Payment"] - next_emi_row["Extra EMI Payment"]), 0.0)
else:
    next_emi_date = None
    next_total_due = 0.0
    next_base_emi = 0.0
    next_extra_emi = 0.0
    next_one_time_part_payment = 0.0

loan_closed_date = updated_schedule.iloc[-1]["EMI Date"] if updated_months else loan_start_date
loan_month_label = min(completed_emis + 1, updated_months) if updated_months else 0

if unused_payments:
    first_unused = min(payment["month"] for payment in unused_payments)
    st.info(
        f"Part payment entries scheduled from month {first_unused} onward were not applied because the loan closes earlier."
    )


# -----------------------------------------------------------
# RESULTS & COMPARISON
# -----------------------------------------------------------
st.divider()
st.markdown("### Results & Comparison")

auto_cols = st.columns(6)
auto_cols[0].metric("EMIs Auto Posted", f"{completed_emis} / {updated_months}")
auto_cols[1].metric("Current Outstanding", format_currency(current_outstanding, 0))
auto_cols[2].metric("Current Loan Month", f"Month {loan_month_label}" if next_emi_date else "Closed")
auto_cols[3].metric("Next EMI Date", next_emi_date.strftime("%d %b %Y") if next_emi_date else "Completed")
auto_cols[4].metric("Next EMI Due", format_currency(next_total_due, 0))
auto_cols[5].metric("Projected Closure", loan_closed_date.strftime("%d %b %Y"))

st.caption(
    f"As of {pd.Timestamp(as_of_date).strftime('%d %b %Y')}, the calculator auto-updates the live loan position using the 3rd of each month as the EMI date."
)

result_cols = st.columns(6)
result_cols[0].metric(
    "Updated Tenure",
    f"{updated_months} months",
    f"-{tenure_reduction} months",
    delta_color="inverse",
)
result_cols[1].metric(
    "Interest Saved",
    format_currency(interest_saved, 0),
    f"{(interest_saved / original_total_interest):.1%}" if original_total_interest else "0.0%",
)
result_cols[2].metric("Updated Interest", format_currency(updated_total_interest, 0))
result_cols[3].metric("Extra In EMI", format_currency(recurring_extra_total, 0))
result_cols[4].metric(
    "EMI After Part Payment",
    format_currency(revised_emi, 0),
    "No change",
)
result_cols[5].metric("Updated Total Payment", format_currency(updated_total_payment, 0))

summary_cols = st.columns(3)
summary_cols[0].metric("One-Time Part Payments", format_currency(one_time_part_payment_total, 0))
summary_cols[1].metric("Total Extra Payments", format_currency(total_part_payment, 0))
summary_cols[2].metric("Initial EMI + Extra", format_currency(original_emi + monthly_extra_payment, 0))

detail_cols = st.columns(4)
detail_cols[0].metric("Next Base EMI", format_currency(next_base_emi, 0))
detail_cols[1].metric("Next Extra EMI", format_currency(next_extra_emi, 0))
detail_cols[2].metric("Next One-Time Payment", format_currency(next_one_time_part_payment, 0))
detail_cols[3].metric("Loan Start", pd.Timestamp(loan_start_date).strftime("%d %b %Y"))

if payment_events.empty and monthly_extra_payment <= 0:
    st.info("Add one or more part payment rows or enable extra payment with EMI to see revised tenure, EMI, and savings.")
elif payment_events.empty:
    st.caption("No one-time part payments added yet. The recurring extra EMI payment is already reflected in the schedules and graphs.")
else:
    st.markdown("#### Part Payment Impact Timeline")
    st.dataframe(prepare_event_display(payment_events), width="stretch", height=220)

comparison_df = pd.merge(
    original_schedule[["Month", "EMI Date", "Closing Principal", "Interest Paid"]].rename(
        columns={
            "EMI Date": "EMI Date",
            "Closing Principal": "Original Closing Principal",
            "Interest Paid": "Original Interest Paid",
        }
    ),
    updated_schedule[["Month", "Closing Principal", "Interest Paid", "Extra EMI Payment", "Part Payment", "EMI Paid"]].rename(
        columns={
            "Closing Principal": "Updated Closing Principal",
            "Interest Paid": "Updated Interest Paid",
            "Extra EMI Payment": "Updated Extra EMI Payment",
            "EMI Paid": "Updated EMI Paid",
        }
    ),
    on="Month",
    how="outer",
).sort_values("Month")

comparison_df["Original Closing Principal"] = comparison_df["Original Closing Principal"].fillna(0.0)
comparison_df["Updated Closing Principal"] = comparison_df["Updated Closing Principal"].fillna(0.0)
comparison_df["Original Interest Paid"] = comparison_df["Original Interest Paid"].fillna(0.0)
comparison_df["Updated Interest Paid"] = comparison_df["Updated Interest Paid"].fillna(0.0)
comparison_df["Updated Extra EMI Payment"] = comparison_df["Updated Extra EMI Payment"].fillna(0.0)
comparison_df["Updated EMI Paid"] = comparison_df["Updated EMI Paid"].fillna(0.0)
comparison_df["Part Payment"] = comparison_df["Part Payment"].fillna(0.0)
comparison_df["Balance Difference"] = (
    comparison_df["Original Closing Principal"] - comparison_df["Updated Closing Principal"]
)
comparison_df["Cumulative Interest Saved"] = (
    comparison_df["Original Interest Paid"].cumsum() - comparison_df["Updated Interest Paid"].cumsum()
)

tab_compare, tab_original, tab_updated, tab_save = st.tabs(
    ["Comparison Table", "Original Schedule", "Updated Schedule", "💾 Save to GitHub"]
)

with tab_compare:
    comparison_display = comparison_df.copy()
    for column in comparison_display.columns:
        if column != "Month" and pd.api.types.is_numeric_dtype(comparison_display[column]):
            comparison_display[column] = comparison_display[column].round(2)
    st.dataframe(comparison_display, width="stretch", height=360)

with tab_original:
    st.dataframe(prepare_schedule_display(original_schedule), width="stretch", height=360)

with tab_updated:
    st.dataframe(prepare_schedule_display(updated_schedule), width="stretch", height=360)

with tab_save:
    st.markdown("#### Save Complete EMI Analysis to GitHub")
    st.caption("Save the current EMI analysis results, schedules, and comparisons to GitHub for future reference.")
    
    save_full_col1, save_full_col2 = st.columns([1, 3])
    
    with save_full_col1:
        save_full_analysis = st.button("💾 Save Full Analysis", type="primary")
    
    with save_full_col2:
        if save_full_analysis:
            with st.spinner("Saving full analysis to GitHub..."):
                success, message = save_emi_analysis_to_github(
                    comparison_df=comparison_df,
                    original_schedule=original_schedule,
                    updated_schedule=updated_schedule,
                    payment_events=payment_events,
                    loan_amount=loan_amount,
                    interest_rate=interest_rate,
                    tenure_months=tenure_months,
                    original_emi=original_emi,
                    monthly_extra_payment=monthly_extra_payment,
                    part_payments=part_payments,
                    loan_start_date=loan_start_date,
                    as_of_date=as_of_date,
                )
                if success:
                    st.success(message)
                    st.balloons()
                else:
                    st.error(message)
    
    st.markdown("---")
    st.markdown("#### Load Saved Analysis")
    st.caption("Load previously saved EMI analysis from GitHub.")
    
    if st.button("📂 Load Saved Analysis"):
        with st.spinner("Loading from GitHub..."):
            saved_data = load_emi_analysis_from_github()
            if saved_data:
                st.success("Analysis loaded successfully!")
                
                # Display summary
                if saved_data.get("summary") is not None:
                    st.dataframe(saved_data["summary"], use_container_width=True)
                
                load_tabs = st.tabs(["Comparison", "Original Schedule", "Updated Schedule", "Payment Events"])
                
                with load_tabs[0]:
                    if saved_data.get("comparison") is not None:
                        st.dataframe(saved_data["comparison"])
                
                with load_tabs[1]:
                    if saved_data.get("original_schedule") is not None:
                        st.dataframe(saved_data["original_schedule"])
                
                with load_tabs[2]:
                    if saved_data.get("updated_schedule") is not None:
                        st.dataframe(saved_data["updated_schedule"])
                
                with load_tabs[3]:
                    if saved_data.get("payment_events") is not None and not saved_data["payment_events"].empty:
                        st.dataframe(saved_data["payment_events"])
                    else:
                        st.info("No payment events found in saved data.")
            else:
                st.warning("No saved analysis found or unable to load.")


# -----------------------------------------------------------
# GRAPHS
# -----------------------------------------------------------
st.divider()
st.markdown("### Graphs")

balance_long = pd.concat(
    [
        original_schedule[["Month", "Closing Principal"]].assign(Scenario="Original"),
        updated_schedule[["Month", "Closing Principal"]].assign(Scenario="Updated"),
    ],
    ignore_index=True,
)

cumulative_interest_long = pd.concat(
    [
        original_schedule[["Month", "Interest Paid"]]
        .assign(Scenario="Original")
        .rename(columns={"Interest Paid": "Monthly Interest"}),
        updated_schedule[["Month", "Interest Paid"]]
        .assign(Scenario="Updated")
        .rename(columns={"Interest Paid": "Monthly Interest"}),
    ],
    ignore_index=True,
)
cumulative_interest_long["Cumulative Interest"] = cumulative_interest_long.groupby("Scenario")[
    "Monthly Interest"
].cumsum()

chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.markdown("#### Loan Balance Over Time")
    st.altair_chart(
        build_balance_chart(balance_long, original_months=original_months, updated_months=updated_months),
        width="stretch",
    )
    st.caption(
        f"Loan closes in month {updated_months} after extra payments versus month {original_months} in the original plan."
    )

with chart_col2:
    st.markdown("#### Cumulative Interest Paid")
    st.altair_chart(build_cumulative_interest_chart(cumulative_interest_long), width="stretch")
    st.caption(
        f"Projected interest saving: {format_currency(interest_saved, 0)} with {format_currency(total_part_payment, 0)} in total extra payments."
    )