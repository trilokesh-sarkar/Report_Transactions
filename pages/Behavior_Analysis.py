import streamlit as st
import pandas as pd
from utils.github_storage import read_csv
from utils.filter_state import persistent_multiselect, reset_filter_button
import os

# -----------------------------------------------------------
# LOAD GLOBAL CSS
# -----------------------------------------------------------
css_path = ".streamlit/styles.css"
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.set_page_config(layout="wide")
st.title("🧠 Spending Behavior Analysis")

# -----------------------------------------------------------
# LOAD DATA
# -----------------------------------------------------------
df = read_csv()
df["period"] = pd.to_datetime(df["period"])

# -----------------------------------------------------------
# SIDEBAR FILTERS
# -----------------------------------------------------------
st.sidebar.markdown("### Filters")

c1, c2 = st.sidebar.columns(2)

with c1:
    f_year = persistent_multiselect("Year", sorted(df.year.unique(), reverse=True), "years")
    f_acc = persistent_multiselect("Account", sorted(df.accounts.unique()), "accounts")

with c2:
    f_month = persistent_multiselect("Month", sorted(df.year_month.unique(), reverse=True), "months")
    include_cat = persistent_multiselect("Include Category", sorted(df.category.unique()), "include_categories")
    exclude_cat = persistent_multiselect("Exclude Category", sorted(df.category.unique()), "exclude_categories")

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
    st.warning("No data available after applying filters.")
    st.stop()

daily = filtered.groupby("period")["amount"].sum().reset_index()

# -----------------------------------------------------------
# MOST FREQUENT CATEGORY
# -----------------------------------------------------------
st.subheader("🔁 Most Frequent Category")
freq_cat = filtered["category"].value_counts().idxmax()
st.success(f"You spend most frequently on **{freq_cat}**")

# -----------------------------------------------------------
# MOST USED ACCOUNT
# -----------------------------------------------------------
st.subheader("🏦 Most Used Account")
freq_acc = filtered["accounts"].value_counts().idxmax()
st.info(f"Most used payment method: **{freq_acc}**")

# -----------------------------------------------------------
# SPEND SPIKE DETECTION
# -----------------------------------------------------------
st.subheader("⚡ Spend Spike Detection")

threshold = daily["amount"].mean() + 2 * daily["amount"].std()
spikes = daily[daily["amount"] > threshold]

if not spikes.empty:
    st.write("High spending days detected:")
    st.dataframe(spikes)
else:
    st.write("No abnormal spikes detected.")

# -----------------------------------------------------------
# LOW SPEND DAYS
# -----------------------------------------------------------
st.subheader("🧘 Low Spend Days")

low_days = daily[daily["amount"] < daily["amount"].mean() * 0.5]
st.write(f"{len(low_days)} low-spend days identified.")

# -----------------------------------------------------------
# DAY OF WEEK PATTERN
# -----------------------------------------------------------
st.subheader("📆 Day-of-Week Spending Pattern")

filtered["dow"] = filtered["period"].dt.day_name()
dow_spend = filtered.groupby("dow")["amount"].mean()
st.bar_chart(dow_spend)

# -----------------------------------------------------------
# SPENDER PERSONALITY
# -----------------------------------------------------------
st.subheader("🧬 Your Spender Type")

avg_daily = daily["amount"].mean()

if avg_daily < 300:
    st.success("You are a **Minimalist Spender** 🧘 — controlled and disciplined.")
elif avg_daily < 800:
    st.info("You are a **Balanced Spender** ⚖ — practical with occasional splurges.")
else:
    st.warning("You are a **High Lifestyle Spender** 💎 — comfort and quality matter.")

# -----------------------------------------------------------
# HABIT CATEGORY (highest avg spend)
# -----------------------------------------------------------
st.subheader("🎯 Your Habit Category")

habit_cat = filtered.groupby("category")["amount"].mean().idxmax()
st.write(f"You tend to spend the highest per transaction on **{habit_cat}**.")

# -----------------------------------------------------------
# CONSECUTIVE SPEND STREAK
# -----------------------------------------------------------
st.subheader("🔥 Longest Spending Streak")

daily_sorted = daily.sort_values("period")
daily_sorted["gap"] = daily_sorted["period"].diff().dt.days

streak = 1
max_streak = 1

for g in daily_sorted["gap"].fillna(1):
    if g == 1:
        streak += 1
        max_streak = max(max_streak, streak)
    else:
        streak = 1

st.metric("Longest Consecutive Spending Streak", f"{max_streak} days")

# -----------------------------------------------------------
# NO SPEND DAYS
# -----------------------------------------------------------
st.subheader("🛑 No-Spend Discipline Days")

all_days = pd.date_range(filtered["period"].min(), filtered["period"].max())
spent_days = set(daily["period"])
no_spend_days = [d for d in all_days if d not in spent_days]

st.write(f"You had **{len(no_spend_days)}** no-spend days.")

# -----------------------------------------------------------
# BEGINNING vs END OF MONTH
# -----------------------------------------------------------
st.subheader("📅 Beginning vs End of Month Spending")

filtered["day"] = filtered["period"].dt.day
filtered["month_part"] = filtered["day"].apply(
    lambda x: "Start (1-10)" if x <= 10 else ("Middle (11-20)" if x <= 20 else "End (21-31)")
)

month_part_spend = filtered.groupby("month_part")["amount"].mean()
st.bar_chart(month_part_spend)

# -----------------------------------------------------------
# RISKY SPEND DAYS (top 10%)
# -----------------------------------------------------------
st.subheader("⚠ Risky Spend Days")

risky = daily[daily["amount"] > daily["amount"].quantile(0.90)]
st.write(f"{len(risky)} unusually high spend days detected.")
st.dataframe(risky)

# -----------------------------------------------------------
# MOST COMMON SPEND DAY
# -----------------------------------------------------------
st.subheader("📆 Most Common Spend Day")

common_day = filtered["period"].dt.day_name().value_counts().idxmax()
st.write(f"You most frequently spend on **{common_day}**.")
