import os
from typing import Any

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb


MODEL_DIR = "models"
DAILY_MODEL_PATH = os.path.join(MODEL_DIR, "daily_forecast_model.pkl")
MONTHLY_MODEL_PATH = os.path.join(MODEL_DIR, "monthly_forecast_model.pkl")

FIXED_COST_RULES = {
    "rent": {"monthly_amount": 4000.0, "default_day": 1},
    "bike_emi": {"monthly_amount": 5333.0, "default_day": 1},
}


FREQ_CONFIGS = {
    "D": {
        "name": "Daily XGBoost",
        "lags": [1, 2, 3, 7, 14, 21, 28],
        "roll_windows": [7, 14, 28],
        "holdout": 14,
        "baseline_window": 28,
        "baseline_season": 7,
        "use_log_target": True,
        "params": {
            "n_estimators": 320,
            "learning_rate": 0.04,
            "max_depth": 4,
            "min_child_weight": 3,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "reg_alpha": 0.2,
            "reg_lambda": 1.5,
            "objective": "reg:squarederror",
            "random_state": 42,
        },
    },
    "MS": {
        "name": "Monthly XGBoost",
        "lags": [1, 2, 3, 6, 12],
        "roll_windows": [3, 6],
        "holdout": 3,
        "baseline_window": 3,
        "baseline_season": 12,
        "use_log_target": False,
        "include_quarter_features": False,
        "params": {
            "n_estimators": 60,
            "learning_rate": 0.06,
            "max_depth": 1,
            "min_child_weight": 2,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "reg_alpha": 0.8,
            "reg_lambda": 2.0,
            "objective": "reg:squarederror",
            "random_state": 42,
        },
    },
}


def mean_absolute_error(actual, predicted):
    actual_arr = np.asarray(actual, dtype=float)
    pred_arr = np.asarray(predicted, dtype=float)
    return float(np.mean(np.abs(actual_arr - pred_arr)))


def clip_forecast(values, upper_bound):
    arr = np.asarray(values, dtype=float)
    return np.clip(arr, 0, max(float(upper_bound), 0.0))


def moving_average_forecast(train_series, horizon, window, weighted=False):
    values = train_series.tail(min(window, len(train_series))).to_numpy(dtype=float)

    if len(values) == 0:
        base_value = 0.0
    elif weighted and len(values) > 1:
        weights = np.arange(1, len(values) + 1, dtype=float)
        base_value = float(np.average(values, weights=weights))
    else:
        base_value = float(values.mean())

    return np.full(horizon, max(base_value, 0.0), dtype=float)


def seasonal_naive_forecast(train_series, horizon, season_length):
    values = train_series.to_numpy(dtype=float)

    if len(values) == 0:
        return np.zeros(horizon)

    if len(values) < season_length:
        return np.full(horizon, max(float(values[-1]), 0.0), dtype=float)

    last_season = values[-season_length:]
    repeats = int(np.ceil(horizon / season_length))
    return np.tile(last_season, repeats)[:horizon]


def build_future_index(last_timestamp, periods, freq):
    offset = pd.tseries.frequencies.to_offset(freq)
    return pd.date_range(start=last_timestamp + offset, periods=periods, freq=freq)


def get_frequency_config(freq):
    if freq not in FREQ_CONFIGS:
        raise ValueError(f"Unsupported forecast frequency: {freq}")
    return FREQ_CONFIGS[freq]


def get_min_history(freq):
    config = get_frequency_config(freq)
    return max(config["lags"]) + 8


def model_path_for_freq(freq):
    return DAILY_MODEL_PATH if freq == "D" else MONTHLY_MODEL_PATH


def _normalize_category(value):
    return str(value).strip().lower()


def build_fixed_cost_context(source_df):
    df = source_df.copy()
    df["period"] = pd.to_datetime(df["period"], errors="coerce")
    df = df.dropna(subset=["period"]).copy()
    df["category_key"] = df["category"].map(_normalize_category)
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)
    df["month_key"] = df["period"].dt.to_period("M").astype(str)

    allocation_rows = []
    active_rules = {}

    for category_key, rule in FIXED_COST_RULES.items():
        category_df = df[df["category_key"] == category_key].sort_values("period")
        if category_df.empty:
            continue

        active_rules[category_key] = dict(rule)

        for _, month_df in category_df.groupby("month_key"):
            remaining = float(rule["monthly_amount"])
            for row in month_df.itertuples():
                if remaining <= 0:
                    break
                allocation = min(max(float(row.amount), 0.0), remaining)
                if allocation <= 0:
                    continue
                allocation_rows.append(
                    {
                        "period": pd.Timestamp(row.period).normalize(),
                        "category_key": category_key,
                        "allocated_amount": allocation,
                    }
                )
                remaining -= allocation

    allocations = pd.DataFrame(allocation_rows)
    scheduled_days = {}

    for category_key, rule in active_rules.items():
        category_allocations = allocations[allocations["category_key"] == category_key] if not allocations.empty else pd.DataFrame()
        if category_allocations.empty:
            scheduled_days[category_key] = int(rule["default_day"])
            continue
        mode_days = category_allocations["period"].dt.day.mode()
        scheduled_days[category_key] = int(mode_days.iloc[0]) if not mode_days.empty else int(rule["default_day"])

    return {
        "active_rules": active_rules,
        "allocations": allocations,
        "scheduled_days": scheduled_days,
    }


def build_fixed_history_series(source_df, freq):
    context = build_fixed_cost_context(source_df)
    allocations = context["allocations"]

    if allocations.empty:
        return pd.Series(dtype=float), context

    if freq == "D":
        fixed_series = (
            allocations.groupby("period")["allocated_amount"]
            .sum()
            .sort_index()
        )
    else:
        allocations = allocations.copy()
        allocations["period"] = allocations["period"].dt.to_period("M").dt.to_timestamp()
        fixed_series = (
            allocations.groupby("period")["allocated_amount"]
            .sum()
            .sort_index()
        )

    return fixed_series.astype(float), context


def build_future_fixed_series(last_timestamp, periods, freq, fixed_context):
    active_rules = fixed_context.get("active_rules", {})
    if not active_rules:
        future_index = build_future_index(last_timestamp, periods, freq)
        return pd.Series(0.0, index=future_index, dtype=float)

    future_index = build_future_index(last_timestamp, periods, freq)

    if freq == "MS":
        fixed_amount = sum(rule["monthly_amount"] for rule in active_rules.values())
        return pd.Series(float(fixed_amount), index=future_index, dtype=float)

    future_series = pd.Series(0.0, index=future_index, dtype=float)
    scheduled_days = fixed_context.get("scheduled_days", {})

    month_starts = pd.date_range(start=future_index.min().replace(day=1), end=future_index.max().replace(day=1), freq="MS")
    for month_start in month_starts:
        month_end = month_start + pd.offsets.MonthEnd(0)
        for category_key, rule in active_rules.items():
            scheduled_day = int(scheduled_days.get(category_key, rule["default_day"]))
            scheduled_date = month_start + pd.offsets.Day(min(scheduled_day, month_end.day) - 1)
            if scheduled_date in future_series.index:
                future_series.loc[scheduled_date] += float(rule["monthly_amount"])

    return future_series


def _feature_row(history_series, timestamp, freq):
    config = get_frequency_config(freq)
    history = history_series.astype(float)
    row = {}

    for lag in config["lags"]:
        row[f"lag_{lag}"] = float(history.iloc[-lag]) if len(history) >= lag else np.nan

    for window in config["roll_windows"]:
        tail = history.iloc[-window:] if len(history) >= window else history
        row[f"roll_mean_{window}"] = float(tail.mean()) if len(tail) else 0.0
        row[f"roll_std_{window}"] = float(tail.std(ddof=0)) if len(tail) > 1 else 0.0

    row["time_idx"] = len(history)
    row["month"] = int(timestamp.month)
    row["month_sin"] = float(np.sin(2 * np.pi * timestamp.month / 12))
    row["month_cos"] = float(np.cos(2 * np.pi * timestamp.month / 12))

    if freq == "D":
        day_of_week = int(timestamp.dayofweek)
        row["day"] = int(timestamp.day)
        row["dow"] = day_of_week
        row["weekofyear"] = int(timestamp.isocalendar().week)
        row["is_weekend"] = int(day_of_week >= 5)
        row["dow_sin"] = float(np.sin(2 * np.pi * day_of_week / 7))
        row["dow_cos"] = float(np.cos(2 * np.pi * day_of_week / 7))
    else:
        if config.get("include_quarter_features", True):
            quarter = int(timestamp.quarter)
            row["quarter"] = quarter
            row["quarter_sin"] = float(np.sin(2 * np.pi * quarter / 4))
            row["quarter_cos"] = float(np.cos(2 * np.pi * quarter / 4))

    return row


def build_supervised_frame(series, freq):
    config = get_frequency_config(freq)
    rows = []
    targets = []
    index = []
    min_lag = max(config["lags"])

    for pos in range(min_lag, len(series)):
        history = series.iloc[:pos]
        timestamp = series.index[pos]
        rows.append(_feature_row(history, timestamp, freq))
        targets.append(float(series.iloc[pos]))
        index.append(timestamp)

    X = pd.DataFrame(rows, index=index)
    y = pd.Series(targets, index=index, dtype=float)
    return X, y


def _calculate_upper_bound(series, freq):
    recent_points = 56 if freq == "D" else 12
    recent = series.tail(min(recent_points, len(series))).clip(lower=0.0)
    if recent.empty:
        return 0.0

    quantile_bound = float(recent.quantile(0.98))
    max_bound = float(recent.max())
    return max(quantile_bound, max_bound)


def train_xgboost_bundle(series, freq):
    series = series.astype(float)
    if len(series) < get_min_history(freq):
        raise ValueError(
            f"Need at least {get_min_history(freq)} {freq} observations to train the XGBoost forecaster."
        )

    config = get_frequency_config(freq)
    X_train, y_train = build_supervised_frame(series, freq)
    use_log_target = bool(config.get("use_log_target", False))
    training_target = np.log1p(y_train.clip(lower=0.0)) if use_log_target else y_train.clip(lower=0.0)

    model = xgb.XGBRegressor(**config["params"])
    model.fit(X_train, training_target, verbose=False)

    return {
        "model": model,
        "freq": freq,
        "feature_columns": list(X_train.columns),
        "upper_bound": _calculate_upper_bound(series, freq),
        "use_log_target": use_log_target,
        "train_end": str(series.index.max()),
        "history_points": int(len(series)),
    }


def forecast_with_xgboost_bundle(bundle, history_series, horizon):
    freq = bundle["freq"]
    feature_columns = bundle["feature_columns"]
    history = history_series.astype(float).copy()
    predictions = []

    future_index = build_future_index(history.index.max(), horizon, freq)
    for timestamp in future_index:
        feature_frame = pd.DataFrame([_feature_row(history, timestamp, freq)])
        feature_frame = feature_frame.reindex(columns=feature_columns, fill_value=0.0)
        raw_prediction = float(bundle["model"].predict(feature_frame)[0])
        if bundle.get("use_log_target", False):
            prediction = max(float(np.expm1(raw_prediction)), 0.0)
        else:
            prediction = max(raw_prediction, 0.0)
        predictions.append(prediction)
        history.loc[timestamp] = prediction

    clipped = clip_forecast(predictions, bundle.get("upper_bound", history.max()))
    return pd.Series(clipped, index=future_index, name="forecast")


def train_and_save_bundle(series, freq, model_path=None):
    bundle = train_xgboost_bundle(series, freq)
    output_path = model_path or model_path_for_freq(freq)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(bundle, output_path)
    return bundle


def load_saved_bundle(freq, model_path=None):
    input_path = model_path or model_path_for_freq(freq)
    if not os.path.exists(input_path):
        return None
    return joblib.load(input_path)


def baseline_forecast(train_series, horizon, freq):
    config = get_frequency_config(freq)
    if freq == "D":
        return seasonal_naive_forecast(train_series, horizon, config["baseline_season"])
    return moving_average_forecast(train_series, horizon, config["baseline_window"], weighted=True)


def evaluate_latest_holdout(series, freq):
    config = get_frequency_config(freq)
    holdout = min(config["holdout"], max(1, len(series) // 5))
    min_history = get_min_history(freq)

    if len(series) < (min_history + holdout):
        return {
            "model": config["name"],
            "holdout": 0,
            "mae": None,
            "baseline_mae": None,
            "improvement_pct": None,
        }

    train = series.iloc[:-holdout]
    test = series.iloc[-holdout:]
    bundle = train_xgboost_bundle(train, freq)
    predictions = forecast_with_xgboost_bundle(bundle, train, holdout)
    baseline = baseline_forecast(train, holdout, freq)

    model_mae = mean_absolute_error(test, predictions)
    baseline_mae = mean_absolute_error(test, baseline)
    improvement_pct = None
    if baseline_mae > 0:
        improvement_pct = ((baseline_mae - model_mae) / baseline_mae) * 100.0

    return {
        "model": config["name"],
        "holdout": holdout,
        "mae": model_mae,
        "baseline_mae": baseline_mae,
        "improvement_pct": improvement_pct,
    }


def summarize_training_result(series, freq):
    metrics = evaluate_latest_holdout(series, freq)
    return {
        "points": int(len(series)),
        "start": str(series.index.min().date()) if len(series) else None,
        "end": str(series.index.max().date()) if len(series) else None,
        **metrics,
    }
