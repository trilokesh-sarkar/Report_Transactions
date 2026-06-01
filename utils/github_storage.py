import base64
import os
import requests
import pandas as pd
from io import StringIO
from zoneinfo import ZoneInfo
from dotenv import load_dotenv

load_dotenv()

# -----------------------------------------------------------
# CONFIG FROM ENV / STREAMLIT SECRETS
# -----------------------------------------------------------
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
if not GITHUB_TOKEN:
    raise ValueError("Missing GITHUB_TOKEN. Set it in .env.")

OWNER = os.getenv("GITHUB_OWNER", "trilokesh-sarkar")
REPO = os.getenv("GITHUB_REPO", "Report_Transactions")
BRANCH = os.getenv("GITHUB_BRANCH", "main")
FILE_PATH = os.getenv("GITHUB_FILE_PATH", "finance_data.csv")

BASE_URL = f"https://api.github.com/repos/{OWNER}/{REPO}/contents/{FILE_PATH}"

HEADERS = {
    "Authorization": f"token {GITHUB_TOKEN}",  # IMPORTANT: token not Bearer
    "Accept": "application/vnd.github+json",
}

RECURRING_BIKE_EMI_START = pd.Timestamp(2026, 4, 1)
RECURRING_BIKE_EMI_AMOUNT = 5333.0
RECURRING_BIKE_EMI_CATEGORY = "bike_emi"
RECURRING_BIKE_EMI_ACCOUNT = "Auto Debit"
RECURRING_TIMEZONE = ZoneInfo("Asia/Kolkata")


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
    r = requests.get(BASE_URL, headers=HEADERS)

    if r.status_code != 200:
        raise Exception(
            "GitHub Read Failed: "
            f"{r.status_code} - {r.text}. "
            f"Check GITHUB_OWNER={OWNER}, GITHUB_REPO={REPO}, "
            f"GITHUB_BRANCH={BRANCH}, GITHUB_FILE_PATH={FILE_PATH}."
        )

    content = r.json()["content"]
    decoded = base64.b64decode(content).decode("utf-8")

    df = pd.read_csv(StringIO(decoded))
    df = apply_recurring_transactions(df)

    df["period"] = pd.to_datetime(df["period"], errors="coerce")
    df["year"] = df.period.dt.year
    df["year_month"] = df.period.dt.to_period("M").astype(str)

    return df


# -----------------------------------------------------------
# WRITE CSV
# -----------------------------------------------------------
def write_csv(df, message="update csv"):
    # 1️⃣ Get latest SHA
    r = requests.get(BASE_URL, headers=HEADERS)

    if r.status_code != 200:
        raise Exception(
            "GitHub SHA Fetch Failed: "
            f"{r.status_code} - {r.text}. "
            f"Check GITHUB_OWNER={OWNER}, GITHUB_REPO={REPO}, "
            f"GITHUB_BRANCH={BRANCH}, GITHUB_FILE_PATH={FILE_PATH}."
        )

    sha = r.json()["sha"]

    # 2️⃣ Convert DF to base64
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)
    encoded = base64.b64encode(csv_buffer.getvalue().encode()).decode()

    payload = {
        "message": message,
        "content": encoded,
        "sha": sha,
        "branch": BRANCH,
    }

    r = requests.put(BASE_URL, headers=HEADERS, json=payload)

    if r.status_code not in [200, 201]:
        raise Exception(
            "GitHub Write Failed: "
            f"{r.status_code} - {r.text}. "
            f"Check GITHUB_OWNER={OWNER}, GITHUB_REPO={REPO}, "
            f"GITHUB_BRANCH={BRANCH}, GITHUB_FILE_PATH={FILE_PATH}, and token permissions."
        )

    return True
