import pandas as pd

INPUT = r"S:\sih\1\data\PAIMANA_PROJECT_DATA.csv"

df = pd.read_csv(INPUT)

duplicates = df[
    df.duplicated(
        subset=["Project ID", "Report Month"],
        keep=False
    )
].copy()

duplicates = duplicates.sort_values(
    ["Project ID", "Report Month"]
)

print("=" * 70)
print("DUPLICATE PROJECT ID + MONTH ANALYSIS")
print("=" * 70)

print("Duplicate records:", len(duplicates))
print("Duplicate Project ID + Month groups:",
      duplicates.groupby(["Project ID", "Report Month"]).ngroups)

print("\nDuplicates by month:")
print(
    duplicates["Report Month"]
    .value_counts()
    .sort_index()
)

print("\nFirst 50 duplicate groups:")
print(
    duplicates[
        [
            "Project ID",
            "Report Month",
            "Project Name",
            "Sector",
            "Cost",
            "Cumulative Expenditure",
            "Physical Progress"
        ]
    ]
    .head(50)
    .to_string(index=False)
)

# Save for detailed inspection
duplicates.to_csv(
    r"S:\sih\1\data\PAIMANA_DUPLICATES.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nSaved duplicate records to:")
print(r"S:\sih\1\data\PAIMANA_DUPLICATES.csv")