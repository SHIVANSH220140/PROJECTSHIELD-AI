import pandas as pd

INPUT = r"S:\sih\1\data\PAIMANA_PROJECT_DATA_UNIQUE.csv"

df = pd.read_csv(INPUT)

print("=" * 80)
print("INSPECTING TARGET-RELATED FIELDS")
print("=" * 80)

print("\nTotal rows:", len(df))

print("\n" + "=" * 80)
print("COMMISSIONING DATE EXAMPLES")
print("=" * 80)

commissioning = (
    df["Date of Commissioning"]
    .dropna()
    .astype(str)
    .drop_duplicates()
)

for value in commissioning.head(30):
    print(repr(value))

print("\n" + "=" * 80)
print("COST EXAMPLES")
print("=" * 80)

cost = (
    df["Cost"]
    .dropna()
    .astype(str)
    .drop_duplicates()
)

for value in cost.head(30):
    print(repr(value))

print("\n" + "=" * 80)
print("PHYSICAL PROGRESS EXAMPLES")
print("=" * 80)

progress = (
    df["Physical Progress"]
    .dropna()
    .astype(str)
    .drop_duplicates()
)

for value in progress.head(30):
    print(repr(value))

print("\n" + "=" * 80)
print("MISSING VALUES")
print("=" * 80)

print(
    df[
        [
            "Date of Approval",
            "Date of Commissioning",
            "Cost",
            "Cumulative Expenditure",
            "Physical Progress"
        ]
    ].isna().sum()
)

print("\n" + "=" * 80)
print("SAMPLE COMPLETE RECORDS")
print("=" * 80)

print(
    df[
        [
            "Unique Project Key",
            "Project Name",
            "Date of Approval",
            "Date of Commissioning",
            "Cost",
            "Cumulative Expenditure",
            "Physical Progress",
            "Report Month"
        ]
    ].head(20).to_string(index=False)
)

print("\n" + "=" * 80)
print("INSPECTION COMPLETE")
print("=" * 80)