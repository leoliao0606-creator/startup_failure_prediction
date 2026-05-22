from __future__ import annotations

from datetime import date
from pathlib import Path

PREDICTION_HORIZON_YEARS = 3

DATA_DIR = Path("data")
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")

COMPANIES_RAW_PATH = DATA_DIR / "companies_raw.csv"
SNAPSHOTS_PATH = DATA_DIR / "snapshots.csv"
LEGACY_SEED_PATH = DATA_DIR / "cleaned_startups.csv"

MODEL_PATH = MODELS_DIR / "startup_failure_model.joblib"
REPORT_PATH = REPORTS_DIR / "evaluation_report.md"

SNAPSHOT_YEARLY_OFFSETS = (1, 2, 3, 4, 5, 6, 7)
SNAPSHOT_DAYS_AFTER_FUNDING = 30
SNAPSHOT_MIN_AGE_DAYS = 180

REFERENCE_TODAY = date(2026, 5, 21)

TIME_SPLIT_FRACTION = 0.75
