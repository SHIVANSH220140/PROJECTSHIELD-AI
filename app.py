import os
import json
import pickle
import warnings
from pathlib import Path

import pandas as pd
import numpy as np
from flask import Flask, jsonify, request, render_template
from flask_cors import CORS

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
TEMPLATE_DIR = BASE_DIR / "templates"

app = Flask(__name__, template_folder=str(TEMPLATE_DIR))
CORS(app)


# ============================================================
# HELPERS
# ============================================================

def safe_float(value, default=0.0):
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def safe_int(value, default=0):
    try:
        if pd.isna(value):
            return default
        return int(value)
    except Exception:
        return default


def first_numeric(value):
    if pd.isna(value):
        return np.nan

    text = str(value).strip()

    if not text:
        return np.nan

    text = text.split("\n")[0].strip()
    text = text.replace(",", "")

    try:
        return float(text)
    except Exception:
        return np.nan


def clean_value(value):
    if pd.isna(value):
        return None

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating, float)):
        if np.isnan(value):
            return None
        return float(value)

    return str(value)


def normalize_columns(df):
    df.columns = [
        str(col).strip().replace("\ufeff", "")
        for col in df.columns
    ]
    return df


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 80)
print("PAIMANA AI DASHBOARD")
print("=" * 80)
print()

parsed_file = DATA_DIR / "PAIMANA_PARSED_DATA.csv"
risk_file = DATA_DIR / "PAIMANA_LATEST_RISK.csv"

print("Loading PAIMANA data...")

if not parsed_file.exists():
    raise FileNotFoundError(
        f"Missing data file: {parsed_file}"
    )

df = pd.read_csv(
    parsed_file,
    low_memory=False
)

df = normalize_columns(df)

print(f"Loaded records: {len(df):,}")

# ------------------------------------------------------------
# CREATE REQUIRED NUMERIC COLUMNS IF MISSING
# ------------------------------------------------------------

if "Cost Numeric" not in df.columns:

    if "Cost" in df.columns:
        df["Cost Numeric"] = df["Cost"].apply(first_numeric)

    elif "Original Cost Numeric" in df.columns:
        df["Cost Numeric"] = pd.to_numeric(
            df["Original Cost Numeric"],
            errors="coerce"
        )

    else:
        df["Cost Numeric"] = np.nan


if "Expenditure Numeric" not in df.columns:

    if "Cumulative Expenditure" in df.columns:
        df["Expenditure Numeric"] = df[
            "Cumulative Expenditure"
        ].apply(first_numeric)

    else:
        df["Expenditure Numeric"] = np.nan


if "Progress Numeric" not in df.columns:

    if "Physical Progress" in df.columns:
        df["Progress Numeric"] = pd.to_numeric(
            df["Physical Progress"],
            errors="coerce"
        )

    elif "Progress" in df.columns:
        df["Progress Numeric"] = pd.to_numeric(
            df["Progress"],
            errors="coerce"
        )

    else:
        df["Progress Numeric"] = np.nan


# ------------------------------------------------------------
# REPORT DATE
# ------------------------------------------------------------

if "Report Date" in df.columns:
    df["Report Date"] = pd.to_datetime(
        df["Report Date"],
        errors="coerce"
    )


# ------------------------------------------------------------
# UNIQUE PROJECT KEY
# ------------------------------------------------------------

if "Unique Project Key" not in df.columns:

    if "Project ID" in df.columns:
        df["Unique Project Key"] = (
            df["Project ID"].astype(str)
        )

    elif "Project ID" in df.columns:
        df["Unique Project Key"] = (
            df["Project ID"].astype(str)
        )

    else:
        df["Unique Project Key"] = (
            df.index.astype(str)
        )


# ------------------------------------------------------------
# PROJECT ID
# ------------------------------------------------------------

if "Project ID" not in df.columns:

    if "Parent Project ID" in df.columns:
        df["Project ID"] = df["Parent Project ID"]

    else:
        df["Project ID"] = (
            df["Unique Project Key"]
            .astype(str)
            .str.split("_")
            .str[0]
        )


# ------------------------------------------------------------
# PROJECT NAME
# ------------------------------------------------------------

if "Project Name" not in df.columns:
    df["Project Name"] = "Unknown Project"


if "State" not in df.columns:
    df["State"] = "Unknown"


if "Sector" not in df.columns:
    df["Sector"] = "Unknown"


# ============================================================
# LOAD RISK DATA
# ============================================================

risk_df = None

if risk_file.exists():

    try:
        risk_df = pd.read_csv(
            risk_file,
            low_memory=False
        )

        risk_df = normalize_columns(risk_df)

        print("ML risk scores loaded.")

    except Exception as e:

        print(
            f"Warning: Could not load risk data: {e}"
        )

else:

    print(
        "Warning: Risk file not found."
    )


# ============================================================
# MERGE RISK DATA
# ============================================================

if risk_df is not None and len(risk_df) > 0:

    if "Unique Project Key" in risk_df.columns:

        risk_columns = [
            "Unique Project Key",
            "Stagnation Risk",
            "Delay Risk",
            "Overall Risk",
            "Risk Level"
        ]

        available_risk_columns = [
            c for c in risk_columns
            if c in risk_df.columns
        ]

        risk_merge = risk_df[
            available_risk_columns
        ].copy()

        risk_merge = risk_merge.drop_duplicates(
            subset=["Unique Project Key"],
            keep="last"
        )

        df = df.merge(
            risk_merge,
            on="Unique Project Key",
            how="left",
            suffixes=("", "_risk")
        )


# ============================================================
# RISK COLUMNS
# ============================================================

if "Stagnation Risk" not in df.columns:
    df["Stagnation Risk"] = 0.0

if "Delay Risk" not in df.columns:
    df["Delay Risk"] = 0.0

if "Overall Risk" not in df.columns:
    df["Overall Risk"] = (
        pd.to_numeric(
            df["Stagnation Risk"],
            errors="coerce"
        ).fillna(0)
        * 0.5
        +
        pd.to_numeric(
            df["Delay Risk"],
            errors="coerce"
        ).fillna(0)
        * 0.5
    )


# ------------------------------------------------------------
# CONVERT RISK VALUES
# ------------------------------------------------------------

for col in [
    "Stagnation Risk",
    "Delay Risk",
    "Overall Risk"
]:

    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    ).fillna(0)


# ============================================================
# PRESENTATION-SAFE RISK
# ============================================================

df["Display Stagnation Risk"] = (
    df["Stagnation Risk"]
)

completed_mask = (
    df["Progress Numeric"].fillna(0) >= 99.5
)

df.loc[
    completed_mask,
    "Display Stagnation Risk"
] = 0.0


df["Display Overall Risk"] = (
    df["Display Stagnation Risk"] * 0.5
    +
    df["Delay Risk"] * 0.5
)


def risk_level(score):

    score = safe_float(score)

    if score >= 70:
        return "HIGH"

    if score >= 40:
        return "MEDIUM"

    return "LOW"


df["Display Risk Level"] = (
    df["Display Overall Risk"]
    .apply(risk_level)
)


# ============================================================
# LATEST RECORD PER PROJECT
# ============================================================

sort_columns = []

if "Report Date" in df.columns:
    sort_columns.append("Report Date")

if "Unique Project Key" in df.columns:
    sort_columns.append("Unique Project Key")

if sort_columns:
    df = df.sort_values(
        sort_columns
    )


latest_df = (
    df
    .groupby(
        "Unique Project Key",
        as_index=False
    )
    .tail(1)
    .copy()
)


print(
    f"Unique projects: {len(latest_df):,}"
)

print()

print("=" * 80)
print("SERVER READY")
print("=" * 80)
print()

# ============================================================
# SERIALIZER
# ============================================================

def project_to_dict(row):

    result = {}

    for column in row.index:

        value = row[column]

        if isinstance(value, pd.Timestamp):
            if pd.isna(value):
                result[column] = None
            else:
                result[column] = value.strftime(
                    "%Y-%m-%d"
                )

        else:
            result[column] = clean_value(value)

    # Friendly aliases for frontend

    result["unique_project_key"] = clean_value(
        row.get("Unique Project Key")
    )

    result["project_id"] = clean_value(
        row.get("Project ID")
    )

    result["project_name"] = clean_value(
        row.get("Project Name")
    )

    result["state"] = clean_value(
        row.get("State")
    )

    result["sector"] = clean_value(
        row.get("Sector")
    )

    result["progress"] = safe_float(
        row.get("Progress Numeric")
    )

    result["risk_score"] = safe_float(
        row.get("Display Overall Risk")
    )

    result["overall_risk"] = safe_float(
        row.get("Display Overall Risk")
    )

    result["risk_level"] = row.get(
        "Display Risk Level",
        "LOW"
    )

    return result


# ============================================================
# HOME
# ============================================================

@app.route("/")
def home():

    return render_template(
        "dashboard.html"
    )


# ============================================================
# HEALTH
# ============================================================

@app.route("/api/health")
def health():

    return jsonify({
        "status": "online",
        "records": int(len(df)),
        "projects": int(len(latest_df))
    })


# ============================================================
# SUMMARY
# ============================================================

@app.route("/api/summary")
def summary():

    try:

        total_projects = len(latest_df)

        total_records = len(df)

        high_risk = int(
            (
                latest_df["Display Risk Level"]
                == "HIGH"
            ).sum()
        )

        medium_risk = int(
            (
                latest_df["Display Risk Level"]
                == "MEDIUM"
            ).sum()
        )

        low_risk = int(
            (
                latest_df["Display Risk Level"]
                == "LOW"
            ).sum()
        )

        average_risk = safe_float(
            latest_df[
                "Display Overall Risk"
            ].mean()
        )

        # ----------------------------------------------------
        # COST
        # ----------------------------------------------------

        cost_series = pd.to_numeric(
            latest_df["Cost Numeric"],
            errors="coerce"
        )

        total_cost = safe_float(
            cost_series.sum()
        )

        # ----------------------------------------------------
        # EXPENDITURE
        # ----------------------------------------------------

        expenditure_series = pd.to_numeric(
            latest_df["Expenditure Numeric"],
            errors="coerce"
        )

        total_expenditure = safe_float(
            expenditure_series.sum()
        )

        # ----------------------------------------------------
        # PROGRESS
        # ----------------------------------------------------

        progress_series = pd.to_numeric(
            latest_df["Progress Numeric"],
            errors="coerce"
        )

        average_progress = safe_float(
            progress_series.mean()
        )

        # ----------------------------------------------------
        # STATES / SECTORS
        # ----------------------------------------------------

        states = int(
            latest_df["State"]
            .dropna()
            .nunique()
        )

        sectors = int(
            latest_df["Sector"]
            .dropna()
            .nunique()
        )

        # ----------------------------------------------------
        # LATEST REPORT
        # ----------------------------------------------------

        latest_report = None

        if "Report Date" in df.columns:

            valid_dates = df[
                "Report Date"
            ].dropna()

            if len(valid_dates) > 0:

                latest_report = (
                    valid_dates.max()
                    .strftime("%B %Y")
                )

        # ----------------------------------------------------
        # TOP RISK PROJECTS
        # ----------------------------------------------------

        top_risk_df = (
            latest_df
            .sort_values(
                "Display Overall Risk",
                ascending=False
            )
            .head(8)
        )

        top_projects = [
            project_to_dict(row)
            for _, row in top_risk_df.iterrows()
        ]

        response = {

            "status": "success",

            "total_projects": int(
                total_projects
            ),

            "projects": int(
                total_projects
            ),

            "total_records": int(
                total_records
            ),

            "records": int(
                total_records
            ),

            "high_risk": high_risk,

            "medium_risk": medium_risk,

            "low_risk": low_risk,

            "average_risk": round(
                average_risk,
                2
            ),

            "total_cost": round(
                total_cost,
                2
            ),

            "total_expenditure": round(
                total_expenditure,
                2
            ),

            "average_progress": round(
                average_progress,
                2
            ),

            "states": states,

            "sectors": sectors,

            "latest_report": latest_report,

            "risk_distribution": {

                "HIGH": high_risk,

                "MEDIUM": medium_risk,

                "LOW": low_risk
            },

            "top_risk_projects":
                top_projects
        }

        return jsonify(response)

    except Exception as e:

        print(
            "\nSUMMARY ERROR:"
        )

        import traceback
        traceback.print_exc()

        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500


# ============================================================
# PROJECTS
# ============================================================

@app.route("/api/projects")
def projects():

    try:

        query = request.args.get(
            "q",
            ""
        ).strip().lower()

        state = request.args.get(
            "state",
            ""
        ).strip()

        sector = request.args.get(
            "sector",
            ""
        ).strip()

        risk = request.args.get(
            "risk",
            ""
        ).strip().upper()

        limit = request.args.get(
            "limit",
            100
        )

        try:
            limit = int(limit)
        except:
            limit = 100

        result = latest_df.copy()

        if query:

            mask = (
                result["Project Name"]
                .fillna("")
                .astype(str)
                .str.lower()
                .str.contains(
                    query,
                    na=False
                )
                |
                result["Project ID"]
                .fillna("")
                .astype(str)
                .str.lower()
                .str.contains(
                    query,
                    na=False
                )
                |
                result["Unique Project Key"]
                .fillna("")
                .astype(str)
                .str.lower()
                .str.contains(
                    query,
                    na=False
                )
            )

            result = result[mask]

        if state:

            result = result[
                result["State"]
                .fillna("")
                .astype(str)
                .str.lower()
                == state.lower()
            ]

        if sector:

            result = result[
                result["Sector"]
                .fillna("")
                .astype(str)
                .str.lower()
                == sector.lower()
            ]

        if risk in [
            "HIGH",
            "MEDIUM",
            "LOW"
        ]:

            result = result[
                result["Display Risk Level"]
                == risk
            ]

        result = result.sort_values(
            "Display Overall Risk",
            ascending=False
        )

        result = result.head(
            max(1, min(limit, 1000))
        )

        output = [
            project_to_dict(row)
            for _, row in result.iterrows()
        ]

        return jsonify({
            "status": "success",
            "count": len(output),
            "projects": output
        })

    except Exception as e:

        import traceback
        traceback.print_exc()

        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500


# ============================================================
# PROJECT DETAILS
# ============================================================

@app.route("/api/project/<path:project_id>")
def project_details(project_id):

    try:

        project_id = str(project_id)

        history = df[
            (
                df["Unique Project Key"]
                .astype(str)
                == project_id
            )
            |
            (
                df["Project ID"]
                .astype(str)
                == project_id
            )
        ].copy()

        if len(history) == 0:

            return jsonify({
                "status": "error",
                "error": "Project not found"
            }), 404

        if "Report Date" in history.columns:

            history = history.sort_values(
                "Report Date"
            )

        latest = history.iloc[-1]

        history_output = []

        for _, row in history.iterrows():

            history_output.append({

                "report_date":
                    clean_value(
                        row.get(
                            "Report Date"
                        )
                    ),

                "progress":
                    safe_float(
                        row.get(
                            "Progress Numeric"
                        )
                    ),

                "expenditure":
                    safe_float(
                        row.get(
                            "Expenditure Numeric"
                        )
                    ),

                "cost":
                    safe_float(
                        row.get(
                            "Cost Numeric"
                        )
                    )
            })

        result = project_to_dict(
            latest
        )

        result["history"] = history_output

        return jsonify({
            "status": "success",
            "project": result
        })

    except Exception as e:

        import traceback
        traceback.print_exc()

        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500


# ============================================================
# STATES
# ============================================================

@app.route("/api/states")
def states():

    values = sorted(
        latest_df["State"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    return jsonify({
        "states": values
    })


# ============================================================
# SECTORS
# ============================================================

@app.route("/api/sectors")
def sectors():

    values = sorted(
        latest_df["Sector"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    return jsonify({
        "sectors": values
    })


# ============================================================
# ALERTS
# ============================================================

@app.route("/api/alerts")
def alerts():

    try:

        result = latest_df[
            latest_df["Display Risk Level"]
            == "HIGH"
        ].copy()

        result = result.sort_values(
            "Display Overall Risk",
            ascending=False
        )

        output = []

        for _, row in result.iterrows():

            output.append({

                "project_id":
                    clean_value(
                        row.get(
                            "Project ID"
                        )
                    ),

                "unique_project_key":
                    clean_value(
                        row.get(
                            "Unique Project Key"
                        )
                    ),

                "project_name":
                    clean_value(
                        row.get(
                            "Project Name"
                        )
                    ),

                "state":
                    clean_value(
                        row.get(
                            "State"
                        )
                    ),

                "sector":
                    clean_value(
                        row.get(
                            "Sector"
                        )
                    ),

                "risk_score":
                    safe_float(
                        row.get(
                            "Display Overall Risk"
                        )
                    ),

                "risk_level":
                    row.get(
                        "Display Risk Level"
                    ),

                "stagnation_risk":
                    safe_float(
                        row.get(
                            "Display Stagnation Risk"
                        )
                    ),

                "delay_risk":
                    safe_float(
                        row.get(
                            "Delay Risk"
                        )
                    ),

                "progress":
                    safe_float(
                        row.get(
                            "Progress Numeric"
                        )
                    )
            })

        return jsonify({
            "status": "success",
            "count": len(output),
            "alerts": output
        })

    except Exception as e:

        import traceback
        traceback.print_exc()

        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500


# ============================================================
# ANALYTICS - STATES
# ============================================================

@app.route("/api/analytics/states")
def analytics_states():

    try:

        grouped = (
            latest_df
            .groupby(
                "State",
                dropna=False
            )
            .agg(
                Projects=(
                    "Unique Project Key",
                    "count"
                ),
                Average_Risk=(
                    "Display Overall Risk",
                    "mean"
                ),
                Average_Progress=(
                    "Progress Numeric",
                    "mean"
                ),
                Total_Cost=(
                    "Cost Numeric",
                    "sum"
                )
            )
            .reset_index()
        )

        output = []

        for _, row in grouped.iterrows():

            output.append({

                "state":
                    clean_value(
                        row["State"]
                    ),

                "projects":
                    safe_int(
                        row["Projects"]
                    ),

                "average_risk":
                    round(
                        safe_float(
                            row["Average_Risk"]
                        ),
                        2
                    ),

                "average_progress":
                    round(
                        safe_float(
                            row["Average_Progress"]
                        ),
                        2
                    ),

                "total_cost":
                    round(
                        safe_float(
                            row["Total_Cost"]
                        ),
                        2
                    )
            })

        output.sort(
            key=lambda x:
            x["average_risk"],
            reverse=True
        )

        return jsonify({
            "status": "success",
            "data": output
        })

    except Exception as e:

        import traceback
        traceback.print_exc()

        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500


# ============================================================
# ANALYTICS - SECTORS
# ============================================================

@app.route("/api/analytics/sectors")
def analytics_sectors():

    try:

        grouped = (
            latest_df
            .groupby(
                "Sector",
                dropna=False
            )
            .agg(
                Projects=(
                    "Unique Project Key",
                    "count"
                ),
                Average_Risk=(
                    "Display Overall Risk",
                    "mean"
                ),
                Average_Progress=(
                    "Progress Numeric",
                    "mean"
                ),
                Total_Cost=(
                    "Cost Numeric",
                    "sum"
                )
            )
            .reset_index()
        )

        output = []

        for _, row in grouped.iterrows():

            output.append({

                "sector":
                    clean_value(
                        row["Sector"]
                    ),

                "projects":
                    safe_int(
                        row["Projects"]
                    ),

                "average_risk":
                    round(
                        safe_float(
                            row["Average_Risk"]
                        ),
                        2
                    ),

                "average_progress":
                    round(
                        safe_float(
                            row["Average_Progress"]
                        ),
                        2
                    ),

                "total_cost":
                    round(
                        safe_float(
                            row["Total_Cost"]
                        ),
                        2
                    )
            })

        output.sort(
            key=lambda x:
            x["average_risk"],
            reverse=True
        )

        return jsonify({
            "status": "success",
            "data": output
        })

    except Exception as e:

        import traceback
        traceback.print_exc()

        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500


# ============================================================
# ANALYTICS - MONTHLY
# ============================================================

@app.route("/api/analytics/monthly")
def analytics_monthly():

    try:

        if "Report Date" not in df.columns:

            return jsonify({
                "status": "success",
                "data": []
            })

        monthly = (
            df.dropna(
                subset=["Report Date"]
            )
            .groupby(
                "Report Date"
            )
            .agg(
                Records=(
                    "Unique Project Key",
                    "count"
                ),
                Projects=(
                    "Unique Project Key",
                    "nunique"
                ),
                Average_Progress=(
                    "Progress Numeric",
                    "mean"
                ),
                Average_Expenditure=(
                    "Expenditure Numeric",
                    "mean"
                )
            )
            .reset_index()
            .sort_values(
                "Report Date"
            )
        )

        output = []

        for _, row in monthly.iterrows():

            output.append({

                "month":
                    row["Report Date"]
                    .strftime(
                        "%b %Y"
                    ),

                "records":
                    safe_int(
                        row["Records"]
                    ),

                "projects":
                    safe_int(
                        row["Projects"]
                    ),

                "average_progress":
                    round(
                        safe_float(
                            row[
                                "Average_Progress"
                            ]
                        ),
                        2
                    ),

                "average_expenditure":
                    round(
                        safe_float(
                            row[
                                "Average_Expenditure"
                            ]
                        ),
                        2
                    )
            })

        return jsonify({
            "status": "success",
            "data": output
        })

    except Exception as e:

        import traceback
        traceback.print_exc()

        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500


# ============================================================
# RUN SERVER
# ============================================================

if __name__ == "__main__":

    print(
        "Dashboard:"
    )

    print(
        "http://127.0.0.1:5000"
    )

    print()

    print(
        "API health:"
    )

    print(
        "http://127.0.0.1:5000/api/health"
    )

    print()

    print(
        "Starting Flask..."
    )

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )