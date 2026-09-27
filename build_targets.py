import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(r"S:\sih\1")
INPUT_FILE = BASE_DIR / "data" / "PAIMANA_PARSED_DATA.csv"
OUTPUT_FILE = BASE_DIR / "data" / "PAIMANA_ML_DATA.csv"

print("=" * 80)
print("BUILDING FUTURE ML TARGETS")
print("=" * 80)

df = pd.read_csv(INPUT_FILE)

print(f"\nInput rows: {len(df)}")

# ---------------------------------------------------------
# DATE CONVERSION
# ---------------------------------------------------------

df["Report Date"] = pd.to_datetime(df["Report Date"], errors="coerce")

date_columns = [
    "Original Commissioning Date",
    "Revised Commissioning Date",
    "Anticipated Commissioning Date"
]

for col in date_columns:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")

# ---------------------------------------------------------
# SORT
# ---------------------------------------------------------

df = df.sort_values(
    ["Unique Project Key", "Report Date"]
).reset_index(drop=True)

# ---------------------------------------------------------
# CURRENT LATEST PLANNED COST
# Anticipated > Revised > Original
# ---------------------------------------------------------

df["Latest Planned Cost"] = (
    df["Anticipated Cost Numeric"]
    .combine_first(df["Revised Cost Numeric"])
    .combine_first(df["Original Cost Numeric"])
)

# ---------------------------------------------------------
# CURRENT LATEST PLANNED COMMISSIONING
# Anticipated > Revised > Original
# ---------------------------------------------------------

df["Latest Planned Commissioning"] = (
    df["Anticipated Commissioning Date"]
    .combine_first(df["Revised Commissioning Date"])
    .combine_first(df["Original Commissioning Date"])
)

# ---------------------------------------------------------
# NEXT MONTH VALUES
# ---------------------------------------------------------

group = df.groupby("Unique Project Key", sort=False)

df["Next Report Date"] = group["Report Date"].shift(-1)

df["Next Progress"] = group["Progress Numeric"].shift(-1)

df["Next Planned Cost"] = group["Latest Planned Cost"].shift(-1)

df["Next Planned Commissioning"] = group[
    "Latest Planned Commissioning"
].shift(-1)

# ---------------------------------------------------------
# CHECK WHETHER NEXT RECORD IS EXACTLY ONE MONTH LATER
# ---------------------------------------------------------

expected_next_month = (
    df["Report Date"] + pd.DateOffset(months=1)
)

df["Has Next Month"] = (
    df["Next Report Date"].notna()
    & (
        df["Next Report Date"].dt.year
        == expected_next_month.dt.year
    )
    & (
        df["Next Report Date"].dt.month
        == expected_next_month.dt.month
    )
)

# ---------------------------------------------------------
# FUTURE PROGRESS CHANGE
# ---------------------------------------------------------

df["Future Progress Change"] = (
    df["Next Progress"] - df["Progress Numeric"]
)

# ---------------------------------------------------------
# FUTURE COST CHANGE %
# ---------------------------------------------------------

df["Future Cost Change Percent"] = np.where(
    (df["Latest Planned Cost"].notna())
    & (df["Latest Planned Cost"] != 0)
    & (df["Next Planned Cost"].notna()),
    (
        (df["Next Planned Cost"] - df["Latest Planned Cost"])
        / df["Latest Planned Cost"]
    ) * 100,
    np.nan
)

# ---------------------------------------------------------
# FUTURE COMMISSIONING CHANGE
# ---------------------------------------------------------

df["Future Commissioning Change Months"] = np.where(
    df["Latest Planned Commissioning"].notna()
    & df["Next Planned Commissioning"].notna(),
    (
        (
            df["Next Planned Commissioning"].dt.year
            - df["Latest Planned Commissioning"].dt.year
        ) * 12
        +
        (
            df["Next Planned Commissioning"].dt.month
            - df["Latest Planned Commissioning"].dt.month
        )
    ),
    np.nan
)

# =========================================================
# TARGET 1
# FUTURE STAGNATION RISK
#
# 1 = progress changes by <= 0.1 percentage point
# 0 = progress changes by > 0.1 percentage point
# =========================================================

valid_stagnation = (
    df["Has Next Month"]
    & df["Progress Numeric"].notna()
    & df["Next Progress"].notna()
)

df["Future Stagnation Risk"] = np.nan

df.loc[valid_stagnation, "Future Stagnation Risk"] = np.where(
    df.loc[valid_stagnation, "Future Progress Change"].abs() <= 0.1,
    1,
    0
)

# =========================================================
# TARGET 2
# FUTURE PROGRESS DECLINE
#
# 1 = progress decreases
# 0 = progress does not decrease
# =========================================================

valid_decline = (
    df["Has Next Month"]
    & df["Progress Numeric"].notna()
    & df["Next Progress"].notna()
)

df["Future Progress Decline"] = np.nan

df.loc[valid_decline, "Future Progress Decline"] = np.where(
    df.loc[valid_decline, "Next Progress"]
    < df.loc[valid_decline, "Progress Numeric"],
    1,
    0
)

# =========================================================
# TARGET 3
# FUTURE COST ESCALATION RISK
#
# 1 = reported planned cost increases
# 0 = planned cost stays same or decreases
# =========================================================

valid_cost = (
    df["Has Next Month"]
    & df["Latest Planned Cost"].notna()
    & df["Next Planned Cost"].notna()
)

df["Future Cost Escalation Risk"] = np.nan

df.loc[valid_cost, "Future Cost Escalation Risk"] = np.where(
    df.loc[valid_cost, "Next Planned Cost"]
    > df.loc[valid_cost, "Latest Planned Cost"],
    1,
    0
)

# =========================================================
# TARGET 4
# FUTURE DELAY RISK
#
# 1 = latest planned commissioning date moves later
# 0 = date stays same or moves earlier
# =========================================================

valid_delay = (
    df["Has Next Month"]
    & df["Latest Planned Commissioning"].notna()
    & df["Next Planned Commissioning"].notna()
)

df["Future Delay Risk"] = np.nan

df.loc[valid_delay, "Future Delay Risk"] = np.where(
    df.loc[valid_delay, "Next Planned Commissioning"]
    > df.loc[valid_delay, "Latest Planned Commissioning"],
    1,
    0
)

# =========================================================
# TARGET SUMMARY
# =========================================================

print("\n" + "=" * 80)
print("TARGET SUMMARY")
print("=" * 80)

targets = [
    "Future Stagnation Risk",
    "Future Progress Decline",
    "Future Cost Escalation Risk",
    "Future Delay Risk"
]

for target in targets:
    print(f"\n{target}")

    valid = df[target].dropna()

    print(f"Valid rows: {len(valid)}")

    if len(valid) > 0:
        print(
            valid.astype(int)
            .value_counts()
            .sort_index()
            .rename_axis(target)
        )

        print("\nPercentage:")
        print(
            (
                valid.astype(int)
                .value_counts(normalize=True)
                .sort_index()
                * 100
            ).round(2)
        )

# =========================================================
# NEXT MONTH COVERAGE
# =========================================================

print("\n" + "=" * 80)
print("NEXT MONTH COVERAGE")
print("=" * 80)

print(
    df["Has Next Month"]
    .value_counts()
    .rename({
        True: "Rows with next month",
        False: "Rows without next month"
    })
)

# =========================================================
# TARGET MISSING VALUES
# =========================================================

print("\n" + "=" * 80)
print("TARGET MISSING VALUES")
print("=" * 80)

print(df[targets].isna().sum())

# =========================================================
# EXAMPLE TARGET RECORDS
# =========================================================

print("\n" + "=" * 80)
print("EXAMPLE TARGET RECORDS")
print("=" * 80)

example_columns = [
    "Unique Project Key",
    "Report Date",
    "Progress Numeric",
    "Next Progress",
    "Future Progress Change",
    "Latest Planned Cost",
    "Next Planned Cost",
    "Future Cost Change Percent",
    "Latest Planned Commissioning",
    "Next Planned Commissioning",
    "Future Commissioning Change Months",
    "Future Stagnation Risk",
    "Future Progress Decline",
    "Future Cost Escalation Risk",
    "Future Delay Risk"
]

available_columns = [
    col for col in example_columns
    if col in df.columns
]

print(
    df.loc[
        df["Has Next Month"],
        available_columns
    ].head(20).to_string(index=False)
)

# =========================================================
# SAVE
# =========================================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n" + "=" * 80)
print("TARGET BUILDING COMPLETE")
print("=" * 80)

print(f"\nRows: {len(df)}")
print(f"Columns: {len(df.columns)}")

print("\nSaved to:")
print(OUTPUT_FILE)