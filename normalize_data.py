import pandas as pd

INPUT = r"S:\sih\1\data\PAIMANA_MASTER.csv"
OUTPUT = r"S:\sih\1\data\PAIMANA_MASTER_CLEAN.csv"

df = pd.read_csv(INPUT)

# Normalize whitespace
df["State"] = (
    df["State"]
    .astype(str)
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)

# Standardize simple state/UT names
state_map = {
    "ANDHRA PRADESH": "Andhra Pradesh",
    "ARUNACHAL PRADESH": "Arunachal Pradesh",
    "ASSAM": "Assam",
    "BIHAR": "Bihar",
    "CHHATTISGARH": "Chhattisgarh",
    "GOA": "Goa",
    "GUJARAT": "Gujarat",
    "HARYANA": "Haryana",
    "HIMACHAL PRADESH": "Himachal Pradesh",
    "JAMMU AND KASHMIR": "Jammu and Kashmir",
    "JHARKHAND": "Jharkhand",
    "KARNATAKA": "Karnataka",
    "KERALA": "Kerala",
    "MADHYA PRADESH": "Madhya Pradesh",
    "MAHARASHTRA": "Maharashtra",
    "MAHARASTRA": "Maharashtra",
    "MANIPUR": "Manipur",
    "MEGHALAYA": "Meghalaya",
    "MIZORAM": "Mizoram",
    "NAGALAND": "Nagaland",
    "ODISHA": "Odisha",
    "PUNJAB": "Punjab",
    "RAJASTHAN": "Rajasthan",
    "SIKKIM": "Sikkim",
    "TAMIL NADU": "Tamil Nadu",
    "TELANGANA": "Telangana",
    "TRIPURA": "Tripura",
    "UTTAR PRADESH": "Uttar Pradesh",
    "UTTARAKHAND": "Uttarakhand",
    "WEST BENGAL": "West Bengal",
    "DELHI": "Delhi",
    "LADAKH": "Ladakh",
    "GOA": "Goa",
    "PUDUCHERRY": "Puducherry",
    "ANDAMAN AND NICOBAR ISLAND": "Andaman & Nicobar",
    "ANDAMAN AND NICOBAR ISLANDS": "Andaman & Nicobar"
}

df["State"] = df["State"].replace(state_map)

# Save cleaned dataset
df.to_csv(OUTPUT, index=False, encoding="utf-8-sig")

print("=" * 60)
print("CLEAN DATASET CREATED")
print("=" * 60)

print("Rows:", len(df))
print("Columns:", len(df.columns))
print("Unique State values:", df["State"].nunique())

print("\nState values:")
print(df["State"].value_counts().to_string())

print("\nSaved to:")
print(OUTPUT)