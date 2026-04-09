import os
import numpy as np
import joblib
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_DIR = os.path.join(BASE_DIR, "model", "saved")

# Feature order must match train.py FEATURES list exactly
FEATURE_ORDER = [
    "experience_level",
    "employment_type",
    "job_title",
    "company_size",
    "company_region",
    "employee_region",
    "remote_ratio",   # numeric — not encoded
]

CATEGORICAL = [
    "experience_level",
    "employment_type",
    "job_title",
    "company_size",
    "company_region",
    "employee_region",
]

_model    = None
_encoders = None


def load() -> None:
    """Called once at API startup via FastAPI lifespan."""
    global _model, _encoders
    _model    = joblib.load(os.path.join(MODEL_DIR, "model.pkl"))
    _encoders = joblib.load(os.path.join(MODEL_DIR, "encoders.pkl"))
    print(f"Model loaded from  : {MODEL_DIR}")
    print(f"Encoders available : {list(_encoders.keys())}")


def _safe_encode(col: str, value: str) -> int:
    """
    Encode a single categorical value using its dedicated LabelEncoder.
    Falls back to the first known class if value was never seen during training.
    This prevents crashes on unknown inputs.
    """
    encoder = _encoders[col]
    if value not in encoder.classes_:
        print(f"  [predictor] Unknown value '{value}' for '{col}' — using fallback")
        value = encoder.classes_[0]
    return int(encoder.transform([value])[0])


def predict(
    experience_level: str,
    employment_type:  str,
    job_title:        str,
    company_size:     str,
    remote_ratio:     int,
    company_region:   str,
    employee_region:  str,
) -> float:
    features = np.array([[
        _safe_encode("experience_level", experience_level),
        _safe_encode("employment_type",  employment_type),
        _safe_encode("job_title",        job_title),
        _safe_encode("company_size",     company_size),
        _safe_encode("company_region",   company_region),
        _safe_encode("employee_region",  employee_region),
        remote_ratio,   # numeric — no encoding needed
    ]])
    return float(_model.predict(features)[0])