import pandas as pd
import numpy as np
import re

INPUT = r"S:\sih\1\data\PAIMANA_PROJECT_DATA_UNIQUE.csv"
OUTPUT = r"S:\sih\1\data\PAIMANA_PARSED_DATA.csv"

df = pd.read_csv(INPUT)

print("=" * 80)
print("PARSING PROJECT COST AND COMMISSIONING FIELDS")
print("=" * 80)


def clean_value(value):
    if pd.isna(value):
        return ""

    return str(value).strip()


def parse_three_values(value):
    value = clean_value(value)

    if not value:
        return np.nan, np.nan, np.nan

    parts = value.split("\n")

    while len(parts) < 3:
        parts.append("")

    original = parts[0].strip()
    revised = parts[1].strip()
    anticipated = parts[2].strip()

    return original, revised, anticipated


def parse_number(value):
    if pd.isna(value):
        return np.nan

    value = str(value).strip()

    if not value:
        return np.nan

    value = value.replace(",", "")

    match = re.search(r"-?\d+(?:\.\d+)?", value)

    if not match:
        return np.nan

    try:
        return float(match.group())
    except:
        return np.nan


def parse_date(value):
    if pd.isna(value):
        return pd.NaT

    value = str(value).strip()

    if not value:
        return pd.NaT

    match = re.search(r"(\d{1,2})/(\d{4})", value)

    if not match:
        return pd.NaT

    month = int(match.group(1))
    year = int(match.group(2))

    if month < 1 or month > 12:
        return pd.NaT

    try:
        return pd.Timestamp(year=year, month=month, day=1)
    except:
        return pd.NaT


# ============================================================
# COMMISSIONING DATES
# ============================================================

commissioning_values = df[
    "Date of Commissioning"
].apply(
    parse_three_values
)

df["Original Commissioning"] = commissioning_values.apply(
    lambda x: x[0]
)

df["Revised Commissioning"] = commissioning_values.apply(
    lambda x: x[1]
)

df["Anticipated Commissioning"] = commissioning_values.apply(
    lambda x: x[2]
)

df["Original Commissioning Date"] = df[
    "Original Commissioning"
].apply(parse_date)

df["Revised Commissioning Date"] = df[
    "Revised Commissioning"
].apply(parse_date)

df["Anticipated Commissioning Date"] = df[
    "Anticipated Commissioning"
].apply(parse_date)


# ============================================================
# COSTS
# ============================================================

cost_values = df[
    "Cost"
].apply(
    parse_three_values
)

df["Original Cost"] = cost_values.apply(
    lambda x: x[0]
)

df["Revised Cost"] = cost_values.apply(
    lambda x: x[1]
)

df["Anticipated Cost"] = cost_values.apply(
    lambda x: x[2]
)

df["Original Cost Numeric"] = df[
    "Original Cost"
].apply(parse_number)

df["Revised Cost Numeric"] = df[
    "Revised Cost"
].apply(parse_number)

df["Anticipated Cost Numeric"] = df[
    "Anticipated Cost"
].apply(parse_number)


# ============================================================
# COST ESCALATION FEATURES
# ============================================================

df["Cost Increase From Original"] = (
    df["Anticipated Cost Numeric"]
    - df["Original Cost Numeric"]
)

df["Cost Increase Percent"] = np.where(
    df["Original Cost Numeric"] > 0,
    (
        df["Anticipated Cost Numeric"]
        - df["Original Cost Numeric"]
    )
    / df["Original Cost Numeric"]
    * 100,
    np.nan
)


# ============================================================
# COMMISSIONING DELAY FEATURES
# ============================================================

df["Original To Anticipated Delay Months"] = np.where(
    df["Original Commissioning Date"].notna()
    &
    df["Anticipated Commissioning Date"].notna(),
    (
        (
            df["Anticipated Commissioning Date"].dt.year
            -
            df["Original Commissioning Date"].dt.year
        ) * 12
        +
        (
            df["Anticipated Commissioning Date"].dt.month
            -
            df["Original Commissioning Date"].dt.month
        )
    ),
    np.nan
)


# ============================================================
# REPORT DATE
# ============================================================

df["Report Date"] = pd.to_datetime(
    df["Report Date"],
    errors="coerce"
)


# ============================================================
# SORT
# ============================================================

df = df.sort_values(
    [
        "Unique Project Key",
        "Report Date"
    ]
).reset_index(drop=True)


# ============================================================
# HISTORICAL FEATURES
# ============================================================

group = df.groupby(
    "Unique Project Key"
)

df["Previous Progress"] = group[
    "Physical Progress"
].shift(1)

df["Previous Progress"] = pd.to_numeric(
    df["Previous Progress"],
    errors="coerce"
)

df["Progress Numeric"] = pd.to_numeric(
    df["Physical Progress"],
    errors="coerce"
)

df["Progress Change"] = (
    df["Progress Numeric"]
    -
    df["Previous Progress"]
)

df["Previous Anticipated Cost"] = group[
    "Anticipated Cost Numeric"
].shift(1)

df["Cost Change From Previous"] = (
    df["Anticipated Cost Numeric"]
    -
    df["Previous Anticipated Cost"]
)

df["Previous Anticipated Commissioning"] = group[
    "Anticipated Commissioning Date"
].shift(1)


# ============================================================
# SAVE
# ============================================================

df.to_csv(
    OUTPUT,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# SUMMARY
# ============================================================

print("\nRows:", len(df))
print("Columns:", len(df.columns))

print("\nNew parsed columns:")

print([
    "Original Commissioning",
    "Revised Commissioning",
    "Anticipated Commissioning",
    "Original Commissioning Date",
    "Revised Commissioning Date",
    "Anticipated Commissioning Date",
    "Original Cost",
    "Revised Cost",
    "Anticipated Cost",
    "Original Cost Numeric",
    "Revised Cost Numeric",
    "Anticipated Cost Numeric",
    "Cost Increase From Original",
    "Cost Increase Percent",
    "Original To Anticipated Delay Months",
    "Previous Progress",
    "Progress Numeric",
    "Progress Change",
    "Previous Anticipated Cost",
    "Cost Change From Previous",
    "Previous Anticipated Commissioning"
])


print("\nMissing values:")

print(
    df[
        [
            "Original Commissioning Date",
            "Revised Commissioning Date",
            "Anticipated Commissioning Date",
            "Original Cost Numeric",
            "Revised Cost Numeric",
            "Anticipated Cost Numeric",
            "Cost Increase Percent",
            "Original To Anticipated Delay Months"
        ]
    ].isna().sum()
)


print("\n" + "=" * 80)
print("SAMPLE PARSED RECORDS")
print("=" * 80)

print(
    df[
        [
            "Unique Project Key",
            "Date of Commissioning",
            "Original Commissioning Date",
            "Revised Commissioning Date",
            "Anticipated Commissioning Date",
            "Cost",
            "Original Cost Numeric",
            "Revised Cost Numeric",
            "Anticipated Cost Numeric",
            "Original To Anticipated Delay Months"
        ]
    ].head(15).to_string(index=False)
)


print("\nSaved to:")
print(OUTPUT)

print("=" * 80)
print("PARSING COMPLETE")
print("=" * 80)