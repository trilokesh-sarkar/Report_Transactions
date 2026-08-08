# Add these functions to utils/github_storage.py

def write_emi_part_payments_csv(df, commit_message):
    """
    Write EMI part payments to GitHub as CSV.
    """
    import base64
    from github import Github
    import os
    
    # Get GitHub credentials from environment
    github_token = os.getenv("GITHUB_TOKEN")
    github_repo = os.getenv("GITHUB_REPO")
    
    if not github_token or not github_repo:
        raise Exception("GitHub credentials not configured")
    
    # Initialize GitHub client
    g = Github(github_token)
    repo = g.get_repo(github_repo)
    
    # Convert DataFrame to CSV
    csv_content = df.to_csv(index=False)
    
    # Encode content
    encoded_content = base64.b64encode(csv_content.encode()).decode()
    
    file_path = "emi_part_payments.csv"
    
    try:
        # Try to get existing file
        contents = repo.get_contents(file_path)
        repo.update_file(
            contents.path,
            commit_message,
            encoded_content,
            contents.sha,
            branch="main"
        )
    except:
        # File doesn't exist, create it
        repo.create_file(
            file_path,
            commit_message,
            encoded_content,
            branch="main"
        )


def read_emi_part_payments_csv():
    """
    Read EMI part payments from GitHub.
    """
    import base64
    from github import Github
    import os
    from io import StringIO
    
    # Get GitHub credentials from environment
    github_token = os.getenv("GITHUB_TOKEN")
    github_repo = os.getenv("GITHUB_REPO")
    
    if not github_token or not github_repo:
        return pd.DataFrame(columns=["Payment Month", "Payment Amount"])
    
    # Initialize GitHub client
    g = Github(github_token)
    repo = g.get_repo(github_repo)
    
    file_path = "emi_part_payments.csv"
    
    try:
        contents = repo.get_contents(file_path)
        content = base64.b64decode(contents.content).decode()
        df = pd.read_csv(StringIO(content))
        return df
    except:
        # File doesn't exist
        return pd.DataFrame(columns=["Payment Month", "Payment Amount"])


def write_emi_analysis_csv(summary_df, comparison_df, original_schedule, updated_schedule, payment_events, part_payments, commit_message):
    """
    Write EMI analysis dataframes to GitHub as separate CSV files.
    """
    import base64
    from github import Github
    import os
    
    # Get GitHub credentials from environment
    github_token = os.getenv("GITHUB_TOKEN")
    github_repo = os.getenv("GITHUB_REPO")
    
    if not github_token or not github_repo:
        raise Exception("GitHub credentials not configured")
    
    # Initialize GitHub client
    g = Github(github_token)
    repo = g.get_repo(github_repo)
    
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
            encoded_content = base64.b64encode(content.encode()).decode()
            
            try:
                # Try to get existing file
                contents = repo.get_contents(file_path)
                repo.update_file(
                    contents.path,
                    commit_message,
                    encoded_content,
                    contents.sha,
                    branch="main"
                )
            except:
                # File doesn't exist, create it
                repo.create_file(
                    file_path,
                    commit_message,
                    encoded_content,
                    branch="main"
                )


def read_emi_analysis_csv():
    """
    Read EMI analysis CSV files from GitHub and return as dictionary of dataframes.
    """
    import base64
    from github import Github
    import os
    from io import StringIO
    
    # Get GitHub credentials from environment
    github_token = os.getenv("GITHUB_TOKEN")
    github_repo = os.getenv("GITHUB_REPO")
    
    if not github_token or not github_repo:
        return None
    
    # Initialize GitHub client
    g = Github(github_token)
    repo = g.get_repo(github_repo)
    
    folder_path = "emi_analysis/"
    files = ["summary.csv", "comparison.csv", "original_schedule.csv", 
             "updated_schedule.csv", "payment_events.csv", "part_payments.csv"]
    
    result = {}
    
    for file_name in files:
        try:
            contents = repo.get_contents(f"{folder_path}{file_name}")
            content = base64.b64decode(contents.content).decode()
            df = pd.read_csv(StringIO(content))
            if not df.empty:
                result[file_name.replace(".csv", "")] = df
        except:
            # File might not exist yet
            continue
    
    return result if result else None