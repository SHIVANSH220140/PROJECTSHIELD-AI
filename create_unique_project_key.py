import pandas as pd
import re

INPUT = r"S:\sih\1\data\PAIMANA_PROJECT_DATA.csv"
OUTPUT = r"S:\sih\1\data\PAIMANA_PROJECT_DATA_UNIQUE.csv"

df = pd.read_csv(INPUT)

def extract_codes(name):
    if pd.isna(name):
        return None, None

    name = str(name).strip()

    matches = re.findall(r"\(([^()]*)\)", name)

    if len(matches) < 2:
        return None, None

    values = [x.strip().upper() for x in matches]

    project_id = None
    package_id = None

    # Find the final main project ID
    for value in reversed(values):
        if re.fullmatch(r"[A-Z]?\d{5,}", value):
            project_id = value
            break

    if project_id is None:
        return None, None

    # Find numeric code immediately before project ID
    for i in range(len(values) - 1, 0, -1):
        if values[i] == project_id:
            previous = values[i - 1]

            if re.fullmatch(r"\d{5,}", previous):
                package_id = previous
            break

    return project_id, package_id


df[["Project ID", "Package ID"]] = df["Project Name"].apply(
    lambda x: pd.Series(extract_codes(x))
)

# Create the actual unique key
df["Unique Project Key"] = (
    df["Project ID"].astype(str)
    + "_"
    + df["Package ID"].astype(str)
)

# Where Package ID is missing, don't create "nan"
df.loc[
    df["Package ID"].isna(),
    "Unique Project Key"
] = df.loc[
    df["Package ID"].isna(),
    "Project ID"
]

print("=" * 70)
print("UNIQUE PROJECT KEY ANALYSIS")
print("=" * 70)

print("Total records:", len(df))

print("Project IDs:", df["Project ID"].nunique())

print("Package IDs found:", df["Package ID"].notna().sum())
print("Package IDs missing:", df["Package ID"].isna().sum())

print("Unique Project Keys:", df["Unique Project Key"].nunique())

duplicates = df.duplicated(
    subset=["Unique Project Key", "Report Month"],
    keep=False
)

print(
    "Duplicate Unique Project Key + Month records:",
    duplicates.sum()
)

print("\nSample:")
print(
    df[
        [
            "Unique Project Key",
            "Project ID",
            "Package ID",
            "Report Month",
            "Project Name"
        ]
    ].head(20).to_string(index=False)
)

if duplicates.sum() > 0:

    print("\nRemaining duplicate examples:")

    print(
        df.loc[
            duplicates,
            [
                "Unique Project Key",
                "Project ID",
                "Package ID",
                "Report Month",
                "Project Name"
            ]
        ]
        .head(30)
        .to_string(index=False)
    )

df.to_csv(
    OUTPUT,
    index=False,
    encoding="utf-8-sig"
)

print("\nSaved to:")
print(OUTPUT)