from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "dataset" / "loan_data_new.csv"
MODEL_PATH = ROOT / "model" / "loan_approval_xgboost.pkl"
METRICS_PATH = ROOT / "model" / "metrics.json"

TARGET = "Loan Status"
REQUIRED_COLUMNS = [
    "Age", "Gender", "Education", "Person Income", "Employee Experience",
    "Home Onwership", "Loan Amount", "Loan Intent", "Loan interest Rate",
    "Loan percentage", "Credit History", "Credit Score", "Previous Loan",
    TARGET
]

def clean_target(series):
    if pd.api.types.is_numeric_dtype(series):
        target = pd.to_numeric(series, errors="coerce")
    else:
        mapping = {
            "1": 1, "0": 0,
            "approved": 1, "rejected": 0,
            "yes": 1, "no": 0
        }
        target = series.astype(str).str.strip().str.lower().map(mapping)

    if target.isna().any():
        raise ValueError("Loan Status must contain only 1/0 or Approved/Rejected.")
    if set(target.astype(int).unique()) != {0, 1}:
        raise ValueError("Loan Status must contain both classes: 0 and 1.")
    return target.astype(int)

def add_features(data):
    data = data.copy()
    income = pd.to_numeric(data["Person Income"], errors="coerce").replace(0, np.nan)
    amount = pd.to_numeric(data["Loan Amount"], errors="coerce")
    data["Loan_to_Income_Ratio"] = amount / income
    return data

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Put loan_data_new.csv inside the dataset folder."
        )

    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()

    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}")

    df = df[REQUIRED_COLUMNS].drop_duplicates().copy()
    df[TARGET] = clean_target(df[TARGET])
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    df = df[df["Age"].between(18, 100)].copy()

    y = df[TARGET]
    # Exclude Loan interest Rate to avoid possible target leakage.
    X = df.drop(columns=[TARGET, "Loan interest Rate"])
    X = add_features(X)

    numeric_columns = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = [c for c in X.columns if c not in numeric_columns]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_columns),
        ("categorical", categorical_pipeline, categorical_columns)
    ])

    classifier = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, probabilities)), 4),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "classification_report": classification_report(
            y_test, predictions, output_dict=True, zero_division=0
        ),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "feature_columns": list(X.columns)
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "pipeline": pipeline,
        "feature_columns": list(X.columns)
    }, MODEL_PATH)

    with open(METRICS_PATH, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    print("Training completed.")
    print(f"Model saved: {MODEL_PATH}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, predictions))
    print("\nClassification report:")
    print(classification_report(y_test, predictions, zero_division=0))

if __name__ == "__main__":
    main()
