import pandas as pd
import numpy as np
import json
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import joblib

BASE_DIR = Path(r"S:\sih\1")
DATA_FILE = BASE_DIR / "data" / "PAIMANA_ML_DATA.csv"
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("PAIMANA ML MODEL TRAINING")
print("=" * 80)

# =========================================================
# LOAD DATA
# =========================================================

df = pd.read_csv(DATA_FILE)

print(f"\nDataset rows: {len(df)}")
print(f"Dataset columns: {len(df.columns)}")

df["Report Date"] = pd.to_datetime(
    df["Report Date"],
    errors="coerce"
)

df = df.sort_values("Report Date").reset_index(drop=True)

# =========================================================
# CREATE SAFE FEATURES
# =========================================================

feature_candidates = [
    "Progress Numeric",
    "Previous Progress",
    "Progress Change",
    "Previous Expenditure",
    "Expenditure Change",
    "Cost Numeric",
    "Previous Cost",
    "Progress per Crore",
    "Months Observed",
    "Project Age Months",
    "Progress Stagnant",
    "Original Cost Numeric",
    "Revised Cost Numeric",
    "Anticipated Cost Numeric",
    "Cost Increase From Original",
    "Cost Increase Percent"
]

features = [
    col for col in feature_candidates
    if col in df.columns
]

print("\nFeatures used:")
for feature in features:
    print(" -", feature)

# =========================================================
# REMOVE INVALID / INFINITE VALUES
# =========================================================

df[features] = df[features].replace(
    [np.inf, -np.inf],
    np.nan
)

# =========================================================
# CHRONOLOGICAL SPLIT
#
# Train:      <= Feb 2026
# Validation: Mar-Apr 2026
# Test:       May-Jun 2026
#
# July cannot be used because we need next-month labels.
# =========================================================

train_df = df[
    df["Report Date"] <= pd.Timestamp("2026-02-01")
].copy()

val_df = df[
    (df["Report Date"] >= pd.Timestamp("2026-03-01"))
    & (df["Report Date"] <= pd.Timestamp("2026-04-01"))
].copy()

test_df = df[
    (df["Report Date"] >= pd.Timestamp("2026-05-01"))
    & (df["Report Date"] <= pd.Timestamp("2026-06-01"))
].copy()

print("\n" + "=" * 80)
print("DATA SPLIT")
print("=" * 80)

print(f"Training rows:   {len(train_df)}")
print(f"Validation rows: {len(val_df)}")
print(f"Testing rows:    {len(test_df)}")

# =========================================================
# IMPUTER
# =========================================================

imputer = SimpleImputer(strategy="median")

X_train_all = imputer.fit_transform(
    train_df[features]
)

X_val_all = imputer.transform(
    val_df[features]
)

X_test_all = imputer.transform(
    test_df[features]
)

joblib.dump(
    imputer,
    MODEL_DIR / "feature_imputer.pkl"
)

# =========================================================
# MODEL FUNCTION
# =========================================================

def train_model(target_name, filename):

    print("\n" + "=" * 80)
    print(f"TRAINING: {target_name}")
    print("=" * 80)

    train_mask = train_df[target_name].notna()
    val_mask = val_df[target_name].notna()
    test_mask = test_df[target_name].notna()

    X_train = X_train_all[train_mask.values]
    X_val = X_val_all[val_mask.values]
    X_test = X_test_all[test_mask.values]

    y_train = train_df.loc[
        train_mask,
        target_name
    ].astype(int).values

    y_val = val_df.loc[
        val_mask,
        target_name
    ].astype(int).values

    y_test = test_df.loc[
        test_mask,
        target_name
    ].astype(int).values

    print(f"\nTrain samples: {len(y_train)}")
    print(f"Validation samples: {len(y_val)}")
    print(f"Test samples: {len(y_test)}")

    print("\nTraining class distribution:")
    print(pd.Series(y_train).value_counts().sort_index())

    if len(np.unique(y_train)) < 2:
        print("ERROR: Training data contains only one class.")
        return None

    positive = np.sum(y_train == 1)
    negative = np.sum(y_train == 0)

    scale_pos_weight = (
        negative / positive
        if positive > 0
        else 1
    )

    print(
        f"\nScale positive weight: "
        f"{scale_pos_weight:.2f}"
    )

    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[
            (X_val, y_val)
        ],
        verbose=False
    )

    # -----------------------------------------------------
    # PREDICTIONS
    # -----------------------------------------------------

    val_probability = model.predict_proba(X_val)[:, 1]
    test_probability = model.predict_proba(X_test)[:, 1]

    val_prediction = (
        val_probability >= 0.5
    ).astype(int)

    test_prediction = (
        test_probability >= 0.5
    ).astype(int)

    # -----------------------------------------------------
    # METRICS
    # -----------------------------------------------------

    def calculate_metrics(y_true, prediction, probability):

        result = {
            "accuracy": float(
                accuracy_score(
                    y_true,
                    prediction
                )
            ),
            "precision": float(
                precision_score(
                    y_true,
                    prediction,
                    zero_division=0
                )
            ),
            "recall": float(
                recall_score(
                    y_true,
                    prediction,
                    zero_division=0
                )
            ),
            "f1": float(
                f1_score(
                    y_true,
                    prediction,
                    zero_division=0
                )
            )
        }

        if len(np.unique(y_true)) == 2:
            result["roc_auc"] = float(
                roc_auc_score(
                    y_true,
                    probability
                )
            )

            result["pr_auc"] = float(
                average_precision_score(
                    y_true,
                    probability
                )
            )
        else:
            result["roc_auc"] = None
            result["pr_auc"] = None

        result["confusion_matrix"] = (
            confusion_matrix(
                y_true,
                prediction
            ).tolist()
        )

        return result

    val_metrics = calculate_metrics(
        y_val,
        val_prediction,
        val_probability
    )

    test_metrics = calculate_metrics(
        y_test,
        test_prediction,
        test_probability
    )

    print("\nVALIDATION METRICS")

    for key, value in val_metrics.items():
        print(f"{key}: {value}")

    print("\nTEST METRICS")

    for key, value in test_metrics.items():
        print(f"{key}: {value}")

    # -----------------------------------------------------
    # FEATURE IMPORTANCE
    # -----------------------------------------------------

    importance = pd.DataFrame({
        "Feature": features,
        "Importance": model.feature_importances_
    }).sort_values(
        "Importance",
        ascending=False
    )

    print("\nTOP FEATURES")

    print(
        importance.head(10).to_string(
            index=False
        )
    )

    importance.to_csv(
        MODEL_DIR / (
            filename.replace(
                ".pkl",
                "_feature_importance.csv"
            )
        ),
        index=False
    )

    # -----------------------------------------------------
    # SAVE MODEL
    # -----------------------------------------------------

    model_path = MODEL_DIR / filename

    joblib.dump(
        model,
        model_path
    )

    print(
        f"\nModel saved: {model_path}"
    )

    return {
        "model": model,
        "validation": val_metrics,
        "test": test_metrics,
        "test_indices": test_df.index[test_mask].tolist(),
        "test_probability": test_probability,
        "test_prediction": test_prediction
    }


# =========================================================
# TRAIN MODELS
# =========================================================

results = {}

results["stagnation"] = train_model(
    "Future Stagnation Risk",
    "stagnation_model.pkl"
)

results["delay"] = train_model(
    "Future Delay Risk",
    "delay_model.pkl"
)

results["progress_decline"] = train_model(
    "Future Progress Decline",
    "progress_decline_model.pkl"
)

# =========================================================
# SAVE MODEL METRICS
# =========================================================

metrics_output = {}

for name, result in results.items():

    if result is None:
        continue

    metrics_output[name] = {
        "validation": result["validation"],
        "test": result["test"]
    }

with open(
    MODEL_DIR / "model_metrics.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        metrics_output,
        f,
        indent=4
    )

# =========================================================
# SAVE FEATURE LIST
# =========================================================

with open(
    MODEL_DIR / "features.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        features,
        f,
        indent=4
    )

# =========================================================
# GENERATE TEST PREDICTIONS
# =========================================================

prediction_df = test_df[
    [
        "Unique Project Key",
        "Project Name",
        "State",
        "Sector",
        "Report Date"
    ]
].copy()

for name, result in results.items():

    if result is None:
        continue

    model_target = {
        "stagnation":
            "Future Stagnation Risk",
        "delay":
            "Future Delay Risk",
        "progress_decline":
            "Future Progress Decline"
    }[name]

    mask = test_df[model_target].notna()

    temp = test_df.loc[
        mask,
        [
            "Unique Project Key",
            "Report Date"
        ]
    ].copy()

    temp[f"{name}_risk_probability"] = (
        result["test_probability"]
    )

    temp[f"{name}_prediction"] = (
        result["test_prediction"]
    )

    prediction_df = prediction_df.merge(
        temp,
        on=[
            "Unique Project Key",
            "Report Date"
        ],
        how="left"
    )

prediction_file = (
    BASE_DIR
    / "data"
    / "ML_TEST_PREDICTIONS.csv"
)

prediction_df.to_csv(
    prediction_file,
    index=False
)

print("\n" + "=" * 80)
print("MODEL TRAINING COMPLETE")
print("=" * 80)

print("\nModels saved in:")
print(MODEL_DIR)

print("\nMetrics:")
print(
    MODEL_DIR / "model_metrics.json"
)

print("\nTest predictions:")
print(prediction_file)