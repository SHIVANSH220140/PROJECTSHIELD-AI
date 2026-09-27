import pandas as pd
import re

INPUT = r"S:\sih\1\data\PAIMANA_MASTER_CLEAN.csv"
OUTPUT = r"S:\sih\1\data\PAIMANA_PROJECT_DATA.csv"

df = pd.read_csv(INPUT)

def extract_project_id(name):
    if pd.isna(name):
        return None

    name = str(name).strip()

    # Find all parenthesized values
    matches = re.findall(r"\(([^()]*)\)", name)

    if not matches:
        return None

    # Look from right to left.
    # Project codes usually contain letters/numbers and start with N or a digit.
    for value in reversed(matches):
        value = value.strip().upper()

        if re.fullmatch(r"[A-Z]?\d{5,}", value):
            return value

    return None


df["Project ID"] = df["Project Name"].apply(extract_project_id)

# Remove summary rows
before = len(df)

df = df[
    ~df["Project Name"]
    .str.strip()
    .str.lower()
    .eq("total")
].copy()

removed_totals = before - len(df)

print("=" * 60)
print("PROJECT ID ANALYSIS")
print("=" * 60)

print("Total records after removing Total rows:", len(df))
print("Total rows removed:", removed_totals)

print("Project IDs found:", df["Project ID"].notna().sum())
print("Missing Project IDs:", df["Project ID"].isna().sum())
print("Unique Project IDs:", df["Project ID"].nunique())

print("\nSample Project IDs:")
print(
    df[
        ["Project ID", "Project Name", "Report Month"]
    ].head(15).to_string(index=False)
)

print("\nMissing Project ID examples:")
print(
    df.loc[
        df["Project ID"].isna(),
        ["Project Name", "Report Month"]
    ].head(15).to_string(index=False)
)

df.to_csv(
    OUTPUT,
    index=False,
    encoding="utf-8-sig"
)

print("\nSaved to:")
print(OUTPUT)