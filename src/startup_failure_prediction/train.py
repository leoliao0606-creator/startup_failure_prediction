from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .embeddings import default_text_model_name, encode_texts
from .features import (
    CATEGORICAL_FEATURES,
    EXCLUDED_LEAKAGE_FIELDS,
    NUMERIC_FEATURES,
    numeric_baselines,
    records_to_frame,
)
from .model import StartupRiskModel

DEFAULT_DATA_PATH = Path("data/cleaned_startups.csv")
DEFAULT_MODEL_PATH = Path("models/startup_failure_model.joblib")
DEFAULT_REPORT_PATH = Path("reports/evaluation_report.md")


def train_model(
    data_path: str | Path = DEFAULT_DATA_PATH,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    report_path: str | Path = DEFAULT_REPORT_PATH,
    seed: int = 42,
    test_ratio: float = 0.25,
    text_model_name: str | None = None,
) -> dict[str, Any]:
    data = pd.read_csv(data_path)
    if data.empty:
        raise ValueError("training data is empty")

    y = data["label"].astype(int)
    x = records_to_frame(data)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_ratio,
        random_state=seed,
        stratify=y,
    )

    selected_text_model = text_model_name or default_text_model_name()
    structured_preprocessor = build_structured_preprocessor()
    structured_preprocessor.fit(x_train, y_train)

    x_train_features, feature_names, embedding_dim = build_model_matrix(
        x_train,
        structured_preprocessor,
        text_model_name=selected_text_model,
    )
    x_test_features, _feature_names, _embedding_dim = build_model_matrix(
        x_test,
        structured_preprocessor,
        text_model_name=selected_text_model,
    )

    classifier = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=seed,
    )
    classifier.fit(x_train_features, y_train)

    train_scores = classifier.predict_proba(x_train_features)[:, 1]
    test_scores = classifier.predict_proba(x_test_features)[:, 1]
    train_predictions = (train_scores >= 0.5).astype(int)
    test_predictions = (test_scores >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, test_predictions, labels=[0, 1]).ravel()

    metrics = {
        "train_accuracy": float(accuracy_score(y_train, train_predictions)),
        "test_accuracy": float(accuracy_score(y_test, test_predictions)),
        "train_roc_auc": float(roc_auc_score(y_train, train_scores)),
        "test_roc_auc": float(roc_auc_score(y_test, test_scores)),
        "test_confusion_matrix": {
            "tp": int(tp),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
        },
    }

    coefficients = [
        float(value)
        for value in classifier.coef_[0].tolist()
    ]
    versions = dependency_versions()
    metadata = {
        "model_type": "SentenceTransformer embeddings + scikit-learn logistic regression",
        "text_model_name": selected_text_model,
        "text_embedding_dim": int(embedding_dim),
        "dependency_versions": versions,
        "sklearn_version": sklearn.__version__,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "training_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "seed": int(seed),
        "metrics": metrics,
        "feature_count": int(len(feature_names)),
        "label_mean": float(y.mean()),
    }

    model = StartupRiskModel(
        structured_preprocessor=structured_preprocessor,
        classifier=classifier,
        metadata=metadata,
        feature_names=feature_names,
        coefficients=coefficients,
        numeric_baselines=numeric_baselines(data),
        text_model_name=selected_text_model,
        text_embedding_dim=embedding_dim,
    )
    model.save(model_path)
    write_report(report_path, model, metrics, len(x_train), len(x_test))

    return {
        "model_path": str(model_path),
        "report_path": str(report_path),
        "metrics": metrics,
        "metadata": metadata,
    }


def build_structured_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )


def build_model_matrix(
    frame: pd.DataFrame,
    structured_preprocessor: ColumnTransformer,
    text_model_name: str,
) -> tuple[np.ndarray, list[str], int]:
    structured = structured_preprocessor.transform(frame)
    if hasattr(structured, "toarray"):
        structured = structured.toarray()
    text_embeddings = encode_texts(
        frame["early_text"].fillna("").astype(str).tolist(),
        model_name=text_model_name,
    )
    feature_names = [
        str(name)
        for name in structured_preprocessor.get_feature_names_out()
    ]
    embedding_names = [
        f"text_embedding__dim_{index:03d}"
        for index in range(text_embeddings.shape[1])
    ]
    matrix = np.hstack([np.asarray(structured, dtype=np.float32), text_embeddings])
    return matrix, feature_names + embedding_names, int(text_embeddings.shape[1])


def dependency_versions() -> dict[str, str]:
    import sentence_transformers
    import torch
    import transformers

    return {
        "scikit_learn": sklearn.__version__,
        "sentence_transformers": sentence_transformers.__version__,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
    }


def write_report(
    path: str | Path,
    model: StartupRiskModel,
    metrics: dict[str, Any],
    train_rows: int,
    test_rows: int,
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    feature_weights = sorted(
        zip(model.feature_names, model.coefficients),
        key=lambda item: item[1],
        reverse=True,
    )
    top_risk = feature_weights[:12]
    top_protective = list(reversed(feature_weights[-12:]))

    lines = [
        "# Startup Failure Prediction MVP Evaluation",
        "",
        "This report is generated from the local synthetic seed dataset. Replace `data/cleaned_startups.csv` with a real balanced dataset before using the score for decisions.",
        "",
        "## Data",
        "",
        f"- Training rows: {train_rows}",
        f"- Test rows: {test_rows}",
        f"- Leakage guard: excluded post-outcome fields: {', '.join(EXCLUDED_LEAKAGE_FIELDS)}",
        f"- Feature count: {len(model.feature_names)}",
        f"- Text encoder: `{model.text_model_name}`",
        f"- Text embedding dimensions: {model.text_embedding_dim}",
        "",
        "## Metrics",
        "",
        f"- Train accuracy: {metrics['train_accuracy']:.3f}",
        f"- Test accuracy: {metrics['test_accuracy']:.3f}",
        f"- Train ROC-AUC: {metrics['train_roc_auc']:.3f}",
        f"- Test ROC-AUC: {metrics['test_roc_auc']:.3f}",
        f"- Test confusion matrix: {metrics['test_confusion_matrix']}",
        "",
        "## Transformer Channel",
        "",
        f"- Model: `{model.text_model_name}`",
        f"- Embedding dimensions: {model.text_embedding_dim}",
        f"- Versions: {model.metadata.get('dependency_versions', {})}",
        "",
        "## Top Failure-Risk Weights",
        "",
    ]
    lines.extend(f"- `{name}`: {weight:.4f}" for name, weight in top_risk)
    lines.extend(["", "## Top Protective Weights", ""])
    lines.extend(f"- `{name}`: {weight:.4f}" for name, weight in top_protective)
    lines.extend(
        [
            "",
            "## MVP Notes",
            "",
            "- This model uses a SentenceTransformer encoder for text semantics, then concatenates those embeddings with numeric and categorical features before logistic regression.",
            "- The current dataset is intentionally small and illustrative; it validates the API and training pipeline but is not a production model.",
            "- The next production step is to replace the seed data with real failed and successful companies, then rerun training and compare against gradient boosting and calibrated classifiers.",
        ]
    )

    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the startup failure prediction MVP.")
    parser.add_argument("--data", default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--report", default=str(DEFAULT_REPORT_PATH))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--text-model", default=None)
    args = parser.parse_args()

    result = train_model(
        args.data,
        args.model,
        args.report,
        seed=args.seed,
        text_model_name=args.text_model,
    )
    metrics = result["metrics"]
    print(f"model: {result['model_path']}")
    print(f"report: {result['report_path']}")
    print(f"test_accuracy: {metrics['test_accuracy']:.3f}")
    print(f"test_roc_auc: {metrics['test_roc_auc']:.3f}")


if __name__ == "__main__":
    main()
