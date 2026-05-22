from __future__ import annotations

from typing import Any

import pandas as pd

NUMERIC_FEATURES = [
    "funding_total_usd",
    "funding_rounds",
    "founded_year",
    "operating_years",
    "market_score",
    "scalability_score",
]

CATEGORICAL_FEATURES = [
    "industry",
    "product_type",
    "country",
]

TEXT_FEATURES = [
    "company_description",
    "founder_statement",
]

MODEL_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES + ["early_text"]

EXCLUDED_LEAKAGE_FIELDS = [
    "failure_reason",
]


def records_to_frame(records: list[dict[str, Any]] | pd.DataFrame) -> pd.DataFrame:
    frame = records.copy() if isinstance(records, pd.DataFrame) else pd.DataFrame(records)

    for field in NUMERIC_FEATURES:
        if field not in frame:
            frame[field] = pd.NA
        frame[field] = pd.to_numeric(frame[field], errors="coerce")

    for field in CATEGORICAL_FEATURES:
        if field not in frame:
            frame[field] = "__unknown__"
        frame[field] = frame[field].fillna("__unknown__").astype(str).replace("", "__unknown__")

    for field in TEXT_FEATURES:
        if field not in frame:
            frame[field] = ""
        frame[field] = frame[field].fillna("").astype(str)

    frame["early_text"] = frame[TEXT_FEATURES].agg(" ".join, axis=1)
    return frame[MODEL_COLUMNS]


def numeric_baselines(frame: pd.DataFrame) -> dict[str, float]:
    baselines: dict[str, float] = {}
    for field in NUMERIC_FEATURES:
        values = pd.to_numeric(frame[field], errors="coerce")
        baselines[field] = float(values.mean()) if values.notna().any() else 0.0
    return baselines
