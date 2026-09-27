import pandas as pd
import numpy as np

INPUT = r"S:\sih\1\data\PAIMANA_PROJECT_DATA_UNIQUE.csv"
OUTPUT = r"S:\sih\1\data\PAIMANA_FEATURE_DATA.csv"

df = pd.read_csv(INPUT)

print("=" * 70)
print("PREPARING FEATURE DATA")
print("=" * 70)


def extract_first_number(value):
    if pd.isna(value):
        return np.nan

    value = str(value).strip()
    value = value.split("\n")[0].strip()

    try:
        return float(value)
    except:
        return np.nan


def parse_approval_date(value):
    if pd.isna(value):
        return pd.NaT

    value = str(value).strip()
    value = value.split("\n")[0].strip()

    try:
        return pd.to_datetime(value, format="%m/%Y")
    except:
        return pd.NaT


# --------------------------------------------------
# BASIC NUMERIC FEATURES
# --------------------------------------------------

df["Cost Numeric"] = df["Cost"].apply(
    extract_first_number
)

df["Expenditure Numeric"] = df[
    "Cumulative Expenditure"
].apply(
    extract_first_number
)

df["Progress Numeric"] = pd.to_numeric(
    df["Physical Progress"],
    errors="coerce"
)


# --------------------------------------------------
# REPORT DATE
# --------------------------------------------------

df["Report Date"] = pd.to_datetime(
    df["Report Date"],
    errors="coerce"
)


# --------------------------------------------------
# SORT PROJECT HISTORY
# --------------------------------------------------

df = df.sort_values(
    ["Unique Project Key", "Report Date"]
).reset_index(drop=True)

group = df.groupby(
    "Unique Project Key"
)


# --------------------------------------------------
# PREVIOUS MONTH FEATURES
# --------------------------------------------------

df["Previous Progress"] = group[
    "Progress Numeric"
].shift(1)

df["Progress Change"] = (
    df["Progress Numeric"]
    - df["Previous Progress"]
)

df["Previous Expenditure"] = group[
    "Expenditure Numeric"
].shift(1)

df["Expenditure Change"] = (
    df["Expenditure Numeric"]
    - df["Previous Expenditure"]
)

df["Previous Cost"] = group[
    "Cost Numeric"
].shift(1)


# --------------------------------------------------
# EFFICIENCY FEATURE
# --------------------------------------------------

df["Progress per Crore"] = np.where(
    df["Expenditure Numeric"] > 0,
    df["Progress Numeric"]
    / df["Expenditure Numeric"],
    np.nan
)


# --------------------------------------------------
# PROJECT HISTORY
# --------------------------------------------------

df["Months Observed"] = group[
    "Report Date"
].transform("count")


# --------------------------------------------------
# APPROVAL DATE
# --------------------------------------------------

df["Approval Date Parsed"] = df[
    "Date of Approval"
].apply(
    parse_approval_date
)


# --------------------------------------------------
# PROJECT AGE
# --------------------------------------------------

df["Project Age Months"] = (
    (
        df["Report Date"].dt.year
        - df["Approval Date Parsed"].dt.year
    ) * 12
    +
    (
        df["Report Date"].dt.month
        - df["Approval Date Parsed"].dt.month
    )
)


# --------------------------------------------------
# STAGNATION FEATURE
# --------------------------------------------------

df["Progress Stagnant"] = (
    (
        df["Progress Change"].abs() < 0.1
    )
    &
    df["Previous Progress"].notna()
).astype(int)


# --------------------------------------------------
# SAVE
# --------------------------------------------------

df.to_csv(
    OUTPUT,
    index=False,
    encoding="utf-8-sig"
)


# --------------------------------------------------
# OUTPUT SUMMARY
# --------------------------------------------------

print("Rows:", len(df))
print("Columns:", len(df.columns))

print("\nNew feature columns:")

print([
    "Cost Numeric",
    "Expenditure Numeric",
    "Progress Numeric",
    "Previous Progress",
    "Progress Change",
    "Previous Expenditure",
    "Expenditure Change",
    "Previous Cost",
    "Progress per Crore",
    "Months Observed",
    "Approval Date Parsed",
    "Project Age Months",
    "Progress Stagnant"
])

print("\nMissing values:")

print(
    df[
        [
            "Cost Numeric",
            "Expenditure Numeric",
            "Progress Numeric",
            "Previous Progress",
            "Progress Change",
            "Expenditure Change",
            "Project Age Months"
        ]
    ].isna().sum()
)

print("\nApproval date sample:")

print(
    df[
        [
            "Date of Approval",
            "Approval Date Parsed",
            "Report Date",
            "Project Age Months"
        ]
    ].head(10).to_string(index=False)
)

print("\nSaved to:")
print(OUTPUT)

print("=" * 70)
print("FEATURE PREPARATION COMPLETE")
print("=" * 70)