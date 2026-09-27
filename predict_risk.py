import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path

BASE_DIR = Path(r"S:\sih\1")
DATA_FILE = BASE_DIR / "data" / "PAIMANA_ML_DATA.csv"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_FILE = BASE_DIR / "data" / "PAIMANA_RISK_SCORES.csv"

print("=" * 80)
print("PAIMANA PROJECT RISK SCORING")
print("=" * 80)

# =========================================================
# LOAD
# =========================================================

df = pd.read_csv(DATA_FILE)

df["Report Date"] = pd.to_datetime(
    df["Report Date"],
    errors="coerce"
)

print(f"\nInput rows: {len(df)}")

# =========================================================
# LOAD FEATURES
# =========================================================

with open(
    MODEL_DIR / "features.json",
    "r",
    encoding="utf-8"
) as f:
    features = json.load(f)

print("\nFeatures loaded:")
for feature in features:
    print(" -", feature)

# =========================================================
# PREPARE FEATURES
# =========================================================

df[features] = df[features].replace(
    [np.inf, -np.inf],
    np.nan
)

imputer = joblib.load(
    MODEL_DIR / "feature_imputer.pkl"
)

X = imputer.transform(
    df[features]
)

# =========================================================
# LOAD MODELS
# =========================================================

stagnation_model = joblib.load(
    MODEL_DIR / "stagnation_model.pkl"
)

delay_model = joblib.load(
    MODEL_DIR / "delay_model.pkl"
)

# =========================================================
# PREDICT
# =========================================================

print("\nGenerating risk probabilities...")

df["Stagnation Risk"] = (
    stagnation_model.predict_proba(X)[:, 1] * 100
)

df["Delay Risk"] = (
    delay_model.predict_proba(X)[:, 1] * 100
)

# =========================================================
# OVERALL RISK
#
# Equal weighting for prototype:
# 50% stagnation
# 50% delay
# =========================================================

df["Overall Risk"] = (
    0.50 * df["Stagnation Risk"]
    +
    0.50 * df["Delay Risk"]
)

# =========================================================
# RISK CATEGORY
# =========================================================

def risk_category(score):

    if score >= 70:
        return "HIGH"

    if score >= 40:
        return "MEDIUM"

    return "LOW"


df["Risk Level"] = df["Overall Risk"].apply(
    risk_category
)

# =========================================================
# RISK FLAGS
# =========================================================

df["Stagnation Flag"] = np.where(
    df["Stagnation Risk"] >= 50,
    "YES",
    "NO"
)

df["Delay Flag"] = np.where(
    df["Delay Risk"] >= 50,
    "YES",
    "NO"
)

# =========================================================
# CURRENT PROJECT SNAPSHOT
#
# Keep only the latest available record per project.
# =========================================================

df = df.sort_values(
    [
        "Unique Project Key",
        "Report Date"
    ]
)

latest = (
    df.groupby(
        "Unique Project Key",
        as_index=False
    )
    .tail(1)
    .copy()
)

latest = latest.sort_values(
    "Overall Risk",
    ascending=False
)

# =========================================================
# SAVE FULL HISTORY
# =========================================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)

# =========================================================
# SAVE LATEST PROJECT SNAPSHOT
# =========================================================

LATEST_FILE = (
    BASE_DIR
    / "data"
    / "PAIMANA_LATEST_RISK.csv"
)

latest.to_csv(
    LATEST_FILE,
    index=False
)

# =========================================================
# SUMMARY
# =========================================================

print("\n" + "=" * 80)
print("RISK SUMMARY")
print("=" * 80)

print(
    latest["Risk Level"]
    .value_counts()
)

print(
    f"\nProjects: {len(latest)}"
)

print(
    f"Average overall risk: "
    f"{latest['Overall Risk'].mean():.2f}"
)

print(
    f"Highest risk: "
    f"{latest['Overall Risk'].max():.2f}"
)

# =========================================================
# TOP 20
# =========================================================

print("\n" + "=" * 80)
print("TOP 20 HIGH-RISK PROJECTS")
print("=" * 80)

display_columns = [
    "Unique Project Key",
    "Project Name",
    "State",
    "Sector",
    "Report Date",
    "Progress Numeric",
    "Stagnation Risk",
    "Delay Risk",
    "Overall Risk",
    "Risk Level"
]

available = [
    col
    for col in display_columns
    if col in latest.columns
]

print(
    latest[available]
    .head(20)
    .to_string(index=False)
)

# =========================================================
# SAVE SUMMARY
# =========================================================

summary = {
    "total_projects": int(len(latest)),
    "average_risk": float(
        latest["Overall Risk"].mean()
    ),
    "maximum_risk": float(
        latest["Overall Risk"].max()
    ),
    "high_risk_projects": int(
        (latest["Risk Level"] == "HIGH").sum()
    ),
    "medium_risk_projects": int(
        (latest["Risk Level"] == "MEDIUM").sum()
    ),
    "low_risk_projects": int(
        (latest["Risk Level"] == "LOW").sum()
    )
}

with open(
    MODEL_DIR / "risk_summary.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        indent=4
    )

print("\n" + "=" * 80)
print("RISK SCORING COMPLETE")
print("=" * 80)

print("\nFull history:")
print(OUTPUT_FILE)

print("\nLatest project risk:")
print(LATEST_FILE)

print("\nSummary:")
print(
    MODEL_DIR / "risk_summary.json"
)