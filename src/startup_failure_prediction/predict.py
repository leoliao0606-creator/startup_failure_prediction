from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .model import StartupRiskModel, risk_level
from .train import DEFAULT_MODEL_PATH

DEFAULT_EXAMPLE = {
    "company_name": "Example Startup",
    "industry": "Ecommerce",
    "product_type": "Marketplace",
    "country": "USA",
    "funding_total_usd": 18000000,
    "funding_rounds": 3,
    "founded_year": 2022,
    "operating_years": 3,
    "market_score": 44,
    "scalability_score": 58,
    "company_description": "A marketplace using subsidies to grow in a crowded category with weak retention.",
    "founder_statement": "We are still searching for repeat usage after the launch campaign.",
}


def predict_payload(model: StartupRiskModel, payload: dict[str, Any]) -> dict[str, Any]:
    probability = model.predict_proba(payload)
    return {
        "risk_probability": round(probability, 4),
        "risk_level": risk_level(probability),
        "top_risk_factors": model.explain(payload),
        "model_metadata": model.metadata,
    }


def load_payload(input_file: str | None, input_json: str | None) -> dict[str, Any]:
    if input_file:
        return json.loads(Path(input_file).read_text(encoding="utf-8"))
    if input_json:
        return json.loads(input_json)
    return DEFAULT_EXAMPLE


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict startup failure risk.")
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--input-file")
    parser.add_argument("--input-json")
    args = parser.parse_args()

    model = StartupRiskModel.load(args.model)
    payload = load_payload(args.input_file, args.input_json)
    print(json.dumps(predict_payload(model, payload), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
