"""
Train a Random Forest Regressor on ds_salaries.csv.
Based on the EDA notebook — mirrors the same cleaning, feature engineering,
and encoding steps, then trains with cross-validation.

Changes vs notebook (intentional fixes):
  - Each categorical column gets its OWN LabelEncoder saved separately.
    The notebook reused one `le` instance for all columns which silently
    overwrites previous fits — meaning encoders.pkl would only remember
    the last column. Fixed here.
  - encoders dict is properly defined and saved (notebook cell 35 had a
    NameError because `encoders` was never built).

Saves to model/saved/:
  model.pkl       ← trained RandomForestRegressor
  encoders.pkl    ← dict of {col: LabelEncoder} for all categorical columns
"""

import os
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble         import RandomForestRegressor
from sklearn.model_selection  import train_test_split, cross_val_predict
from sklearn.preprocessing    import LabelEncoder
from sklearn.metrics          import mean_absolute_error, mean_squared_error, r2_score

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "ds_salaries.csv")
SAVE_DIR  = os.path.join(os.path.dirname(__file__), "saved")

TARGET      = "salary_in_usd"
CATEGORICAL = [
    "experience_level",
    "employment_type",
    "job_title",
    "company_size",
    "company_region",
    "employee_region",
]
FEATURES = CATEGORICAL + ["remote_ratio"]


# ── Job title grouping (50 raw → 6 groups) ────────────────────────────────────
JOB_GROUPS: dict[str, list[str]] = {
    "Data Scientist": [
        "Data Scientist", "Applied Data Scientist", "Staff Data Scientist",
        "Lead Data Scientist", "Principal Data Scientist", "Data Science Consultant",
        "Data Science Engineer", "Data Science Manager", "Director of Data Science",
        "Head of Data Science",
    ],
    "Data Engineer": [
        "Data Engineer", "Big Data Engineer", "Cloud Data Engineer",
        "Lead Data Engineer", "Principal Data Engineer", "Data Engineering Manager",
        "Director of Data Engineering", "ETL Developer", "Analytics Engineer",
        "Data Architect", "Big Data Architect",
    ],
    "ML Engineer": [
        "Machine Learning Engineer", "ML Engineer", "Lead Machine Learning Engineer",
        "Machine Learning Developer", "Machine Learning Infrastructure Engineer",
        "Machine Learning Manager", "Head of Machine Learning",
        "Applied Machine Learning Scientist",
    ],
    "Data Analyst": [
        "Data Analyst", "Business Data Analyst", "Lead Data Analyst",
        "BI Data Analyst", "Marketing Data Analyst", "Financial Data Analyst",
        "Finance Data Analyst", "Product Data Analyst", "Principal Data Analyst",
        "Data Analytics Engineer", "Data Analytics Manager", "Data Analytics Lead",
        "Data Specialist",
    ],
    "Research Scientist": [
        "Research Scientist", "Machine Learning Scientist", "AI Scientist",
        "NLP Engineer", "Computer Vision Engineer", "3D Computer Vision Researcher",
        "Computer Vision Software Engineer",
    ],
}


def group_job_title(title: str) -> str:
    for group, titles in JOB_GROUPS.items():
        if title in titles:
            return group
    return "Other"


# ── Region mapping (57 country codes → 6 regions) ────────────────────────────
REGION_MAP: dict[str, str] = {
    # North America
    "US": "North America", "CA": "North America", "MX": "North America",
    "PR": "North America",
    # Europe
    "DE": "Europe", "GB": "Europe", "FR": "Europe", "ES": "Europe",
    "PT": "Europe", "IT": "Europe", "NL": "Europe", "BE": "Europe",
    "CH": "Europe", "AT": "Europe", "PL": "Europe", "RO": "Europe",
    "DK": "Europe", "CZ": "Europe", "IE": "Europe", "GR": "Europe",
    "HU": "Europe", "HR": "Europe", "LU": "Europe", "SI": "Europe",
    "EE": "Europe", "MT": "Europe", "UA": "Europe", "BG": "Europe",
    "RS": "Europe", "JE": "Europe",
    # Asia
    "IN": "Asia", "JP": "Asia", "CN": "Asia", "PK": "Asia",
    "SG": "Asia", "AE": "Asia", "IL": "Asia", "IR": "Asia",
    "IQ": "Asia", "TR": "Asia", "VN": "Asia", "MY": "Asia",
    "PH": "Asia", "HK": "Asia",
    # South America
    "BR": "South America", "CL": "South America", "CO": "South America",
    "AR": "South America", "BO": "South America",
    # Africa
    "NG": "Africa", "KE": "Africa", "DZ": "Africa", "TN": "Africa",
    # Oceania
    "AU": "Oceania", "NZ": "Oceania",
}


def map_region(code: str) -> str:
    return REGION_MAP.get(code, "Other")


# ── Evaluation helper ─────────────────────────────────────────────────────────
def evaluate(label: str, y_true, y_pred) -> dict:
    mae  = mean_absolute_error(y_true, y_pred)
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_true, y_pred)
    print(f"\n{label}")
    print(f"  MAE  : ${mae:>10,.0f}")
    print(f"  RMSE : ${rmse:>10,.0f}")
    print(f"  R²   : {r2:.4f}")
    return {"model": label, "mae": mae, "mse": mse, "rmse": rmse, "r2": r2}


# ── Main ──────────────────────────────────────────────────────────────────────
def load_and_clean() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)

    # Group job titles
    df["job_title"] = df["job_title"].apply(group_job_title)

    # Map country codes → regions (reduces cardinality from 57 → 6)
    df["company_region"]  = df["company_location"].map(map_region)
    df["employee_region"] = df["employee_residence"].map(map_region)

    # Keep only the columns we need
    df = df[FEATURES + [TARGET]].copy()

    # Drop duplicates and nulls
    df = df.drop_duplicates()
    df = df.dropna()

    return df


def encode(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:
    """
    Fit a separate LabelEncoder per categorical column.
    Returns the encoded dataframe and the encoders dict for saving.
    """
    encoders: dict[str, LabelEncoder] = {}
    for col in CATEGORICAL:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    return df, encoders


def train() -> None:
    os.makedirs(SAVE_DIR, exist_ok=True)

    # ── Load & clean ──────────────────────────────────────────────────────────
    df = load_and_clean()
    print(f"Dataset shape after cleaning: {df.shape}")
    print("\nJob title distribution:")
    print(df["job_title"].value_counts().to_string())

    # ── Encode ────────────────────────────────────────────────────────────────
    df, encoders = encode(df)

    X = df[FEATURES]
    y = df[TARGET]

    # ── Split ─────────────────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"\nTrain size: {len(X_train)}  |  Test size: {len(X_test)}")

    # ── Random Forest with Cross-Validation ───────────────────────────────────
    rf = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    )

    # CV on training data only 
    print("\nRunning 5-fold cross-validation on training set...")
    cv_preds = cross_val_predict(rf, X_train, y_train, cv=5)
    evaluate("RandomForest — CV (train)", y_train, cv_preds)

    # Train final model on full training set
    rf.fit(X_train, y_train)

    # Evaluate on held-out test set
    test_preds = rf.predict(X_test)
    evaluate("RandomForest — Test set", y_test, test_preds)

    # ── Feature importance plot ───────────────────────────────────────────────
    importances = pd.Series(rf.feature_importances_, index=FEATURES).sort_values()
    fig, ax = plt.subplots(figsize=(8, 5))
    importances.plot(kind="barh", ax=ax, color="#4DB6AC")
    ax.set_title("Feature Importances — Random Forest", fontweight="bold")
    ax.set_xlabel("Importance")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "feature_importance.png"), dpi=120)
    plt.close(fig)
    print("\nSaved: feature_importance.png")

    # ── Save artifacts ────────────────────────────────────────────────────────
    joblib.dump(rf,       os.path.join(SAVE_DIR, "model.pkl"))
    joblib.dump(encoders, os.path.join(SAVE_DIR, "encoders.pkl"))
    print("Saved: model.pkl  encoders.pkl")


if __name__ == "__main__":
    train()