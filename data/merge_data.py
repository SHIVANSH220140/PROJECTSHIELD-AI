import csv
import os
import glob
import pandas as pd

DATA_FOLDER = r"S:\sih\1\data"
OUTPUT_FILE = r"S:\sih\1\data\PAIMANA_MASTER.csv"

files = sorted(
    glob.glob(os.path.join(DATA_FOLDER, "*_done.csv")),
    key=lambda x: int(os.path.basename(x).split("_")[0])
)

all_data = []

for file in files:
    print(f"Processing: {os.path.basename(file)}")

    rows = []

    with open(file, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)

        header = next(reader)

        for row in reader:
            if len(row) == 9:
                rows.append(row)

    df = pd.DataFrame(rows, columns=[
        "State",
        "Sector",
        "Sl No",
        "Project Name",
        "Date of Approval",
        "Date of Commissioning",
        "Cost",
        "Cumulative Expenditure",
        "Physical Progress"
    ])

    month_number = int(os.path.basename(file).split("_")[0])
    month_name = os.path.basename(file).split("_")[1]
    year = int(os.path.basename(file).split("_")[2])

    df["Report Month"] = f"{month_name.capitalize()} {year}"
    df["Report Date"] = pd.to_datetime(
        f"01-{month_name}-{year}",
        format="%d-%B-%Y"
    )

    all_data.append(df)

master = pd.concat(all_data, ignore_index=True)

# Clean text fields
for col in ["State", "Sector", "Project Name"]:
    master[col] = (
        master[col]
        .astype(str)
        .str.replace("\n", " ", regex=False)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

# Convert numeric fields
master["Sl No"] = pd.to_numeric(master["Sl No"], errors="coerce")

master["Physical Progress"] = pd.to_numeric(
    master["Physical Progress"],
    errors="coerce"
)

master["Cumulative Expenditure"] = pd.to_numeric(
    master["Cumulative Expenditure"],
    errors="coerce"
)

# Remove completely empty rows
master = master.dropna(
    subset=["State", "Sector", "Project Name"],
    how="all"
)

# Remove accidental header rows
master = master[
    master["Sector"].str.strip().str.lower() != "sector"
]

master.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)

print("\n" + "=" * 60)
print("MASTER DATASET CREATED")
print("=" * 60)

print(f"Files processed : {len(files)}")
print(f"Total rows      : {len(master)}")
print(f"Total columns   : {len(master.columns)}")
print(f"Output          : {OUTPUT_FILE}")

print("\nColumns:")
for i, col in enumerate(master.columns, 1):
    print(f"{i}. {col}")

print("\nMissing values:")
print(master.isnull().sum())

print("\nUnique states:", master["State"].nunique())
print("Unique sectors:", master["Sector"].nunique())
print("Unique projects:", master["Project Name"].nunique())