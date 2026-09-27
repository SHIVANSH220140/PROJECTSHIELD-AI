import pandas as pd

INPUT = r"S:\sih\1\data\PAIMANA_PROJECT_DATA.csv"

df = pd.read_csv(INPUT)

print("=" * 60)
print("PROJECT HISTORY VALIDATION")
print("=" * 60)

print("Total records:", len(df))
print("Unique projects:", df["Project ID"].nunique())
print("Unique months:", df["Report Month"].nunique())

print("\nRecords per month:")
print(df["Report Month"].value_counts().sort_index())

# Number of months available for each project
history = (
    df.groupby("Project ID")["Report Month"]
    .nunique()
    .value_counts()
    .sort_index()
)

print("\nNumber of projects by months of history:")
print(history)

# Projects appearing in multiple months
project_months = (
    df.groupby("Project ID")["Report Month"]
    .nunique()
)

print("\nProjects appearing in:")
print("1 month :", (project_months == 1).sum())
print("2 months:", (project_months == 2).sum())
print("3-5 months:", ((project_months >= 3) & (project_months <= 5)).sum())
print("6-10 months:", ((project_months >= 6) & (project_months <= 10)).sum())
print("11-15 months:", ((project_months >= 11) & (project_months <= 15)).sum())
print("16 months:", (project_months == 16).sum())

# Check duplicate Project ID + Report Month
duplicates = df.duplicated(
    subset=["Project ID", "Report Month"],
    keep=False
)

print("\nDuplicate Project ID + Month records:", duplicates.sum())

if duplicates.sum() > 0:
    print("\nDuplicate examples:")
    print(
        df.loc[
            duplicates,
            ["Project ID", "Report Month", "Project Name"]
        ].head(20).to_string(index=False)
    )

# Check whether a project has different names across months
name_counts = (
    df.groupby("Project ID")["Project Name"]
    .nunique()
)

print("\nProjects with different names across reports:",
      (name_counts > 1).sum())

# Check whether a project has different sectors
sector_counts = (
    df.groupby("Project ID")["Sector"]
    .nunique()
)

print("Projects with different sectors:",
      (sector_counts > 1).sum())

print("\n" + "=" * 60)
print("VALIDATION COMPLETE")
print("=" * 60)