import sys

import pandas as pd
from dotenv import load_dotenv

from utils.forecast_xgboost import (
    build_fixed_history_series,
    summarize_training_result,
    train_and_save_bundle,
)
from utils.github_storage import read_csv


load_dotenv()


def load_data():
    df = read_csv().copy()
    df.columns = df.columns.str.lower()
    df["period"] = pd.to_datetime(df["period"], errors="coerce")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)
    df = df.dropna(subset=["period"])
    return df


def build_daily_series(df):
    return (
        df.groupby(pd.Grouper(key="period", freq="D"))["amount"]
        .sum()
        .asfreq("D", fill_value=0.0)
    )


def build_monthly_series(df):
    return (
        df.groupby(pd.Grouper(key="period", freq="MS"))["amount"]
        .sum()
        .asfreq("MS", fill_value=0.0)
    )


def build_variable_series(df, freq):
    if freq == "D":
        total_series = build_daily_series(df)
    else:
        total_series = build_monthly_series(df)

    fixed_series, _ = build_fixed_history_series(df, freq)
    fixed_series = fixed_series.reindex(total_series.index, fill_value=0.0)
    variable_series = (total_series - fixed_series).clip(lower=0.0)

    return total_series, fixed_series, variable_series


def print_summary(label, summary):
    print(f"{label} points       : {summary['points']}")
    print(f"{label} date range   : {summary['start']} -> {summary['end']}")
    if summary["mae"] is None:
        print(f"{label} holdout MAE  : N/A")
        print(f"{label} baseline     : N/A")
        print(f"{label} improvement  : N/A")
    else:
        print(f"{label} holdout MAE  : INR {summary['mae']:,.2f}")
        print(f"{label} baseline MAE : INR {summary['baseline_mae']:,.2f}")
        print(f"{label} improvement  : {summary['improvement_pct']:.2f}%")


def main():
    print("Loading latest finance data from GitHub...")
    df = load_data()

    if df.empty:
        print("No data found. Cannot train forecast models.")
        return 1

    daily_total, daily_fixed, daily_series = build_variable_series(df, "D")
    monthly_total, monthly_fixed, monthly_series = build_variable_series(df, "MS")

    print("\nTraining daily XGBoost forecast model...")
    train_and_save_bundle(daily_series, "D")
    daily_summary = summarize_training_result(daily_series, "D")
    print_summary("Daily", daily_summary)
    print(f"Daily fixed component learned separately : INR {daily_fixed.sum():,.2f}")

    print("\nTraining monthly XGBoost forecast model...")
    train_and_save_bundle(monthly_series, "MS")
    monthly_summary = summarize_training_result(monthly_series, "MS")
    print_summary("Monthly", monthly_summary)
    print(f"Monthly fixed component learned separately: INR {monthly_fixed.sum():,.2f}")

    print("\nForecast models trained and saved in models/.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
