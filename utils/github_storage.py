import base64
import os
import requests
import pandas as pd
from io import StringIO
from zoneinfo import ZoneInfo
from dotenv import load_dotenv

load_dotenv()


def _get_cfg(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name, default)
    if value is None:
        return None
    return str(value).strip()


# -----------------------------------------------------------
# CONFIG
# -----------------------------------------------------------
GITHUB_TOKEN = _get_cfg("GITHUB_TOKEN")
if not GITHUB_TOKEN:
    raise ValueError("Missing GITHUB_TOKEN. Set it in .env or Streamlit secrets.")

OWNER = _get_cfg("GITHUB_OWNER", "trilokesh-sarkar")
REPO = _get_cfg("GITHUB_REPO", "Report_Transactions")
BRANCH = _get_cfg("GITHUB_BRANCH", "main")
TRANSACTIONS_FILE_PATH = _get_cfg("GITHUB_FILE_PATH", "finance_data.csv")
SAVINGS_FILE_PATH = _get_cfg("GITHUB_SAVINGS_FILE_PATH", "monthly_savings_data.csv")
CHAT_HISTORY_FILE_PATH = _get_cfg("GITHUB_CHAT_FILE_PATH", "agent_chat_history.csv")

BASE_CONTENTS_URL = f"https://api.github.com/repos/{OWNER}/{REPO}/contents"

HEADERS = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github+json",
}

RECURRING_BIKE_EMI_START = pd.Timestamp(2026, 4, 1)
RECURRING_BIKE_EMI_AMOUNT = 5333.0
RECURRING_BIKE_EMI_CATEGORY = "bike_emi"
RECURRING_BIKE_EMI_ACCOUNT = "Auto Debit"
RECURRING_TIMEZONE = ZoneInfo("Asia/Kolkata")


def _build_file_url(file_path: str) -> str:
    return f"{BASE_CONTENTS_URL}/{file_path}"


def _build_github_error(action: str, status_code: int, response_text: str, file_path: str) -> str:
    hint = "Check GitHub config and token permissions."
    if status_code == 401:
        hint = "Invalid or expired GITHUB_TOKEN."
    elif status_code == 403:
        hint = "Token lacks required permissions (Contents: write)."
    elif status_code == 404:
        hint = (
            "Target not found or token has no access. Verify GITHUB_OWNER, "
            "GITHUB_REPO, GITHUB_BRANCH, and GITHUB_FILE_PATH exactly."
        )

    return (
        f"GitHub {action} Failed: {status_code}. {hint} "
        f"[OWNER={OWNER}, REPO={REPO}, BRANCH={BRANCH}, FILE={file_path}] "
        f"API={response_text[:300]}"
    )


def _read_csv_from_path(file_path: str, missing_ok: bool = False) -> pd.DataFrame:
    response = requests.get(_build_file_url(file_path), headers=HEADERS, params={"ref": BRANCH})

    if response.status_code == 404 and missing_ok:
        return pd.DataFrame()

    if response.status_code != 200:
        raise Exception(_build_github_error("Read", response.status_code, response.text, file_path))

    content = response.json()["content"]
    decoded = base64.b64decode(content).decode("utf-8")
    return pd.read_csv(StringIO(decoded))


def _write_csv_to_path(
    df: pd.DataFrame,
    file_path: str,
    message: str,
    skip_if_unchanged: bool = True,
):
    file_url = _build_file_url(file_path)
    response = requests.get(file_url, headers=HEADERS, params={"ref": BRANCH})

    if response.status_code == 404:
        existing_sha = None
        existing_content = None
    elif response.status_code == 200:
        body = response.json()
        existing_sha = body["sha"]
        existing_content = base64.b64decode(body["content"]).decode("utf-8")
    else:
        raise Exception(_build_github_error("SHA Fetch", response.status_code, response.text, file_path))

    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)
    new_content = csv_buffer.getvalue()

    if skip_if_unchanged and existing_content == new_content:
        return False

    payload = {
        "message": message,
        "content": base64.b64encode(new_content.encode()).decode(),
        "branch": BRANCH,
    }
    if existing_sha:
        payload["sha"] = existing_sha

    response = requests.put(file_url, headers=HEADERS, json=payload)

    if response.status_code not in [200, 201]:
        raise Exception(_build_github_error("Write", response.status_code, response.text, file_path))

    return True


def get_current_month_start() -> pd.Timestamp:
    now_ist = pd.Timestamp.now(tz=RECURRING_TIMEZONE)
    return pd.Timestamp(year=now_ist.year, month=now_ist.month, day=1)


def apply_recurring_transactions(df: pd.DataFrame) -> pd.DataFrame:
    updated = df.copy()
    updated["period"] = pd.to_datetime(updated["period"], errors="coerce")

    current_month_start = get_current_month_start()
    if current_month_start < RECURRING_BIKE_EMI_START:
        return updated

    recurring_months = pd.date_range(
        start=RECURRING_BIKE_EMI_START,
        end=current_month_start,
        freq="MS",
    )

    existing_periods = set(
        updated.loc[
            updated["category"].astype(str).str.lower() == RECURRING_BIKE_EMI_CATEGORY,
            "period",
        ]
        .dropna()
        .dt.normalize()
    )

    last_running_total = (
        pd.to_numeric(updated["running_total"], errors="coerce").max()
        if "running_total" in updated.columns and not updated.empty
        else 0.0
    )
    last_running_total = 0.0 if pd.isna(last_running_total) else float(last_running_total)

    missing_rows = []
    for period in recurring_months:
        normalized_period = period.normalize()
        if normalized_period in existing_periods:
            continue

        missing_rows.append(
            {
                "period": normalized_period,
                "accounts": RECURRING_BIKE_EMI_ACCOUNT,
                "category": RECURRING_BIKE_EMI_CATEGORY,
                "amount": RECURRING_BIKE_EMI_AMOUNT,
                "month": normalized_period.strftime("%B"),
                "running_total": last_running_total,
                "year": normalized_period.year,
                "year_month": str(normalized_period.to_period("M")),
            }
        )

    if missing_rows:
        updated = pd.concat([updated, pd.DataFrame(missing_rows)], ignore_index=True)

    return updated


# -----------------------------------------------------------
# READ CSV
# -----------------------------------------------------------
def read_csv():
    df = _read_csv_from_path(TRANSACTIONS_FILE_PATH)
    df = apply_recurring_transactions(df)

    df["period"] = pd.to_datetime(df["period"], errors="coerce")
    df["year"] = df.period.dt.year
    df["year_month"] = df.period.dt.to_period("M").astype(str)

    return df


def read_savings_csv():
    return _read_csv_from_path(SAVINGS_FILE_PATH)


def read_chat_history_csv():
    return _read_csv_from_path(CHAT_HISTORY_FILE_PATH, missing_ok=True)


# -----------------------------------------------------------
# WRITE CSV
# -----------------------------------------------------------
def write_csv(df, message="update csv"):
    return _write_csv_to_path(df, TRANSACTIONS_FILE_PATH, message)


def write_savings_csv(df, message="update monthly savings csv"):
    return _write_csv_to_path(df, SAVINGS_FILE_PATH, message)


def write_chat_history_csv(df, message="update agent chat history csv"):
    return _write_csv_to_path(df, CHAT_HISTORY_FILE_PATH, message)

# Add these functions to utils/github_storage.py

def write_emi_part_payments_csv(df, commit_message):
    """
    Write EMI part payments to GitHub as CSV.
    """
    from io import StringIO
    from .github_utils import github_upload_content
    
    if df.empty:
        # Create empty dataframe with proper columns
        df = pd.DataFrame(columns=["Payment Month", "Payment Amount"])
    
    csv_content = df.to_csv(index=False)
    file_path = "emi_part_payments.csv"
    
    try:
        github_upload_content(file_path, csv_content, commit_message)
        return True
    except Exception as e:
        raise Exception(f"Failed to upload part payments: {str(e)}")


def read_emi_part_payments_csv():
    """
    Read EMI part payments from GitHub.
    """
    from io import StringIO
    from .github_utils import github_download_content
    
    file_path = "emi_part_payments.csv"
    
    try:
        content = github_download_content(file_path)
        if content:
            df = pd.read_csv(StringIO(content))
            return df
        return pd.DataFrame(columns=["Payment Month", "Payment Amount"])
    except Exception as e:
        # File might not exist yet
        return pd.DataFrame(columns=["Payment Month", "Payment Amount"])


def write_emi_analysis_csv(summary_df, comparison_df, original_schedule, updated_schedule, payment_events, part_payments, commit_message):
    """
    Write EMI analysis dataframes to GitHub as separate CSV files.
    """
    from io import StringIO
    from .github_utils import github_upload_content
    import json
    
    # Create a folder for EMI analysis
    folder_path = "emi_analysis/"
    
    # Convert each dataframe to CSV string
    def df_to_csv_string(df):
        if df is None or df.empty:
            return ""
        return df.to_csv(index=False)
    
    files_to_upload = {
        f"{folder_path}summary.csv": df_to_csv_string(summary_df),
        f"{folder_path}comparison.csv": df_to_csv_string(comparison_df),
        f"{folder_path}original_schedule.csv": df_to_csv_string(original_schedule),
        f"{folder_path}updated_schedule.csv": df_to_csv_string(updated_schedule),
        f"{folder_path}payment_events.csv": df_to_csv_string(payment_events),
        f"{folder_path}part_payments.csv": df_to_csv_string(part_payments),
    }
    
    # Upload each file
    for file_path, content in files_to_upload.items():
        if content:  # Only upload non-empty files
            try:
                github_upload_content(file_path, content, commit_message)
            except Exception as e:
                raise Exception(f"Failed to upload {file_path}: {str(e)}")
    
    # Save timestamp file to track when analysis was saved
    timestamp = pd.Timestamp.now(tz="Asia/Kolkata").strftime("%Y-%m-%d %H:%M:%S")
    timestamp_df = pd.DataFrame({"saved_at": [timestamp]})
    try:
        github_upload_content(
            f"{folder_path}last_save_timestamp.csv",
            timestamp_df.to_csv(index=False),
            commit_message
        )
    except Exception as e:
        raise Exception(f"Failed to upload timestamp: {str(e)}")


def read_emi_analysis_csv():
    """
    Read EMI analysis CSV files from GitHub and return as dictionary of dataframes.
    """
    from io import StringIO
    from .github_utils import github_download_content
    
    folder_path = "emi_analysis/"
    files = ["summary.csv", "comparison.csv", "original_schedule.csv", 
             "updated_schedule.csv", "payment_events.csv", "part_payments.csv"]
    
    result = {}
    
    for file_name in files:
        try:
            content = github_download_content(f"{folder_path}{file_name}")
            if content:
                df = pd.read_csv(StringIO(content))
                if not df.empty:
                    result[file_name.replace(".csv", "")] = df
        except Exception as e:
            # File might not exist yet
            continue
    
    return result if result else None