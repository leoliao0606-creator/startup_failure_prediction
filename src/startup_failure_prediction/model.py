from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from .embeddings import encode_texts
from .features import records_to_frame


@dataclass
class StartupRiskModel:
    structured_preprocessor: Any
    classifier: Any
    metadata: dict[str, Any]
    feature_names: list[str]
    coefficients: list[float]
    numeric_baselines: dict[str, float]
    text_model_name: str
    text_embedding_dim: int

    def predict_proba(self, record: dict[str, Any]) -> float:
        features = self.transform_records([record])
        return float(self.classifier.predict_proba(features)[0][1])

    def predict(self, record: dict[str, Any], threshold: float = 0.5) -> int:
        return int(self.predict_proba(record) >= threshold)

    def explain(self, record: dict[str, Any], top_n: int = 5) -> list[dict[str, Any]]:
        values = self.transform_records([record])[0]

        feature_contributions = [
            (name, float(value), coefficient * float(value))
            for name, value, coefficient in zip(self.feature_names, values, self.coefficients)
            if abs(float(value)) > 1e-9
        ]
        structured = [
            item for item in feature_contributions if not item[0].startswith("text_embedding__")
        ]
        text_impact = sum(
            impact
            for name, _value, impact in feature_contributions
            if name.startswith("text_embedding__")
        )

        contributions = structured[:]
        if abs(text_impact) > 1e-9:
            contributions.append(("text_embedding__semantic_profile", 1.0, text_impact))

        positive = sorted(
            (item for item in contributions if item[2] > 0),
            key=lambda item: item[2],
            reverse=True,
        )
        selected = positive[:top_n]
        if not selected:
            selected = sorted(contributions, key=lambda item: abs(item[2]), reverse=True)[:top_n]

        return [
            {
                "feature": name,
                "signal": humanize_signal(name, record, self.numeric_baselines),
                "impact": round(impact, 4),
            }
            for name, _value, impact in selected
        ]

    def transform_records(self, records: list[dict[str, Any]]) -> np.ndarray:
        frame = records_to_frame(records)
        structured = self.structured_preprocessor.transform(frame)
        if hasattr(structured, "toarray"):
            structured = structured.toarray()
        text_embeddings = encode_texts(
            frame["early_text"].fillna("").astype(str).tolist(),
            model_name=self.text_model_name,
        )
        return np.hstack([np.asarray(structured, dtype=np.float32), text_embeddings])

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "structured_preprocessor": self.structured_preprocessor,
                "classifier": self.classifier,
                "metadata": self.metadata,
                "feature_names": self.feature_names,
                "coefficients": self.coefficients,
                "numeric_baselines": self.numeric_baselines,
                "text_model_name": self.text_model_name,
                "text_embedding_dim": self.text_embedding_dim,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "StartupRiskModel":
        payload = joblib.load(path)
        return cls(
            structured_preprocessor=payload["structured_preprocessor"],
            classifier=payload["classifier"],
            metadata=payload.get("metadata", {}),
            feature_names=list(payload["feature_names"]),
            coefficients=[float(value) for value in payload["coefficients"]],
            numeric_baselines={
                key: float(value) for key, value in payload.get("numeric_baselines", {}).items()
            },
            text_model_name=payload["text_model_name"],
            text_embedding_dim=int(payload["text_embedding_dim"]),
        )


def risk_level(probability: float) -> str:
    if probability >= 0.75:
        return "high"
    if probability >= 0.45:
        return "medium"
    return "low"


def humanize_signal(
    feature_name: str,
    record: dict[str, Any],
    baselines: dict[str, float],
) -> str:
    if feature_name.startswith("numeric__"):
        field = feature_name.removeprefix("numeric__")
        raw = record.get(field)
        try:
            value = float(raw)
        except (TypeError, ValueError):
            value = baselines.get(field, 0.0)
        average = baselines.get(field, 0.0)
        direction = "above" if value >= average else "below"
        label = field.replace("_", " ")
        return f"{label} is {direction} the training baseline ({value:g} vs {average:g})"

    if feature_name.startswith("categorical__"):
        field_value = feature_name.removeprefix("categorical__")
        for field in ("industry", "product_type", "country"):
            prefix = f"{field}_"
            if field_value.startswith(prefix):
                value = field_value.removeprefix(prefix)
                label = field.replace("_", " ")
                return f"{label} is {value}"
        return field_value.replace("_", " ")

    if feature_name.startswith("text__"):
        token = feature_name.removeprefix("text__")
        return f"description contains risk-associated term '{token}'"

    if feature_name == "text_embedding__semantic_profile":
        return "Transformer embedding is aligned with failed-company semantic patterns"

    if feature_name.startswith("text_embedding__"):
        return "Transformer embedding dimension contributes to the risk score"

    return feature_name
