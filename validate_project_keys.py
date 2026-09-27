import pandas as pd
import re

INPUT = r"S:\sih\1\data\PAIMANA_PROJECT_DATA_UNIQUE.csv"

df = pd.read_csv(INPUT)

print("=" * 80)
print("VALIDATING PROJECT KEYS")
print("=" * 80)

print("\nTotal records:", len(df))

print("\nUnique Project IDs:", df["Project ID"].nunique())
print("Unique Project Keys:", df["Unique Project Key"].nunique())

print("\n" + "=" * 80)
print("PROJECT ID FORMAT")
print("=" * 80)

def check_project_id(value):
    value = str(value).strip()

    if re.fullmatch(r"[A-Z]\d{8}", value):
        return "LETTER + 8 DIGITS"

    if re.fullmatch(r"[A-Z]\d{7}", value):
        return "LETTER + 7 DIGITS"

    if re.fullmatch(r"\d+", value):
        return "NUMERIC ONLY"

    return "OTHER"


id_types = df["Project ID"].apply(check_project_id)

print(id_types.value_counts())

print("\nExamples of non-standard Project IDs:")

bad_ids = df.loc[
    id_types != "LETTER + 8 DIGITS",
    "Project ID"
].drop_duplicates()

print(bad_ids.head(50).to_string(index=False))


print("\n" + "=" * 80)
print("PROJECT KEY FORMAT")
print("=" * 80)

def check_key(value):
    value = str(value).strip()

    if re.fullmatch(r"[A-Z]\d{8}", value):
        return "PARENT ID"

    if re.fullmatch(r"[A-Z]\d{8}_\d+", value):
        return "PARENT + PACKAGE"

    return "OTHER"


key_types = df["Unique Project Key"].apply(check_key)

print(key_types.value_counts())

print("\nExamples of OTHER keys:")

other_keys = df.loc[
    key_types == "OTHER",
    [
        "Project ID",
        "Unique Project Key",
        "Project Name"
    ]
].drop_duplicates()

print(other_keys.head(50).to_string(index=False))


print("\n" + "=" * 80)
print("PROJECT KEY SAMPLES")
print("=" * 80)

print(
    df[
        [
            "Project ID",
            "Unique Project Key",
            "Project Name"
        ]
    ]
    .drop_duplicates(
        subset=["Unique Project Key"]
    )
    .head(30)
    .to_string(index=False)
)


print("\n" + "=" * 80)
print("PACKAGE ID SUMMARY")
print("=" * 80)

print(
    "Package IDs found:",
    df["Package ID"].notna().sum()
)

print(
    "Package IDs missing:",
    df["Package ID"].isna().sum()
)

print(
    "Unique package IDs:",
    df["Package ID"].nunique()
)

print("\n" + "=" * 80)
print("VALIDATION COMPLETE")
print("=" * 80)