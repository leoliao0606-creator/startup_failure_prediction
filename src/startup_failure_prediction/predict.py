from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Any

from .config import MODEL_PATH, PREDICTION_HORIZON_YEARS, REFERENCE_TODAY
from .model import StartupRiskModel, risk_level

DEFAULT_EXAMPLE = {
    "company_name": "Example Startup",
    "snapshot_date": REFERENCE_TODAY.isoformat(),
    "industry": "Ecommerce",
    "product_type": "Marketplace",
    "country": "USA",
    "age_years_at_snapshot": 2.5,
    "funding_total_usd_at_snapshot": 18000000,
    "funding_rounds_at_snapshot": 3,
    "days_since_last_round": 120,
    "market_score": 44,
    "scalability_score": 58,
    "company_description": "A marketplace using subsidies to grow in a crowded category with weak retention.",
    "founder_statement": "We are still searching for repeat usage after the launch campaign.",
}


def predict_payload(model: StartupRiskModel, payload: dict[str, Any]) -> dict[str, Any]:
    enriched = _enrich_payload(payload)
    probability = model.predict_proba(enriched)
    horizon_years = model.metadata.get("horizon_years", PREDICTION_HORIZON_YEARS)
    snapshot_date = enriched.get("snapshot_date") or REFERENCE_TODAY.isoformat()
    return {
        "risk_probability": round(probability, 4),
        "risk_level": risk_level(probability),
        "horizon_years": int(horizon_years),
        "snapshot_date": snapshot_date,
        "interpretation": (
            f"Estimated probability the company fails within {horizon_years} years of "
            f"{snapshot_date}."
        ),
        "top_risk_factors": model.explain(enriched),
        "model_metadata": model.metadata,
    }


def _enrich_payload(payload: dict[str, Any]) -> dict[str, Any]:
    enriched = dict(payload)
    if "snapshot_date" not in enriched or not enriched["snapshot_date"]:
        enriched["snapshot_date"] = REFERENCE_TODAY.isoformat()

    if "age_years_at_snapshot" not in enriched and enriched.get("founded_date"):
        try:
            founded = date.fromisoformat(str(enriched["founded_date"]))
            snap = date.fromisoformat(str(enriched["snapshot_date"]))
            enriched["age_years_at_snapshot"] = round((snap - founded).days / 365.25, 3)
        except ValueError:
            pass

    if "funding_total_usd_at_snapshot" not in enriched and "funding_total_usd" in enriched:
        enriched["funding_total_usd_at_snapshot"] = enriched["funding_total_usd"]
    if "funding_rounds_at_snapshot" not in enriched and "funding_rounds" in enriched:
        enriched["funding_rounds_at_snapshot"] = enriched["funding_rounds"]
    if "days_since_last_round" not in enriched:
        enriched["days_since_last_round"] = -1

    return enriched


def load_payload(input_file: str | None, input_json: str | None) -> dict[str, Any]:
    if input_file:
        return json.loads(Path(input_file).read_text(encoding="utf-8"))
    if input_json:
        return json.loads(input_json)
    return DEFAULT_EXAMPLE


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict startup failure risk at a snapshot date.")
    parser.add_argument("--model", default=str(MODEL_PATH))
    parser.add_argument("--input-file")
    parser.add_argument("--input-json")
    args = parser.parse_args()

    model = StartupRiskModel.load(args.model)
    payload = load_payload(args.input_file, args.input_json)
    print(json.dumps(predict_payload(model, payload), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
