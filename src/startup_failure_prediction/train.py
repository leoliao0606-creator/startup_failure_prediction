from __future__ import annotations

import argparse
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import (
    COMPANIES_RAW_PATH,
    MODEL_PATH,
    PREDICTION_HORIZON_YEARS,
    REPORT_PATH,
    TIME_SPLIT_FRACTION,
)
from .embeddings import default_text_model_name, encode_texts
from .features import (
    CATEGORICAL_FEATURES,
    EXCLUDED_LEAKAGE_FIELDS,
    NUMERIC_FEATURES,
    numeric_baselines,
    records_to_frame,
)
from .model import StartupRiskModel
from .snapshots import build_snapshots, load_companies

FUNDING_EVENTS_PATH = COMPANIES_RAW_PATH.parent / "funding_events.csv"


def train_model(
    companies_path: str | Path = COMPANIES_RAW_PATH,
    funding_events_path: str | Path = FUNDING_EVENTS_PATH,
    model_path: str | Path = MODEL_PATH,
    report_path: str | Path = REPORT_PATH,
    horizon_years: int = PREDICTION_HORIZON_YEARS,
    seed: int = 42,
    time_split_fraction: float = TIME_SPLIT_FRACTION,
    text_model_name: str | None = None,
    calibrate: bool = True,
) -> dict[str, Any]:
    companies = load_companies(companies_path, funding_events_path)
    snapshots = build_snapshots(companies, horizon_years=horizon_years)
    if snapshots.empty:
        raise ValueError("no usable snapshots produced from raw data")

    train_frame, test_frame, split_date = time_based_split(snapshots, time_split_fraction)
    if train_frame.empty or test_frame.empty:
        raise ValueError(
            "time split produced empty partition; check snapshot coverage or split fraction"
        )

    y_train = train_frame["label"].astype(int).to_numpy()
    y_test = test_frame["label"].astype(int).to_numpy()

    x_train_frame = records_to_frame(train_frame)
    x_test_frame = records_to_frame(test_frame)

    selected_text_model = text_model_name or default_text_model_name()
    structured_preprocessor = build_structured_preprocessor()
    structured_preprocessor.fit(x_train_frame, y_train)

    x_train_features, feature_names, embedding_dim = build_model_matrix(
        x_train_frame, structured_preprocessor, text_model_name=selected_text_model
    )
    x_test_features, _names, _dim = build_model_matrix(
        x_test_frame, structured_preprocessor, text_model_name=selected_text_model
    )

    base_classifier = LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        random_state=seed,
    )

    if calibrate and len(np.unique(y_train)) > 1 and len(y_train) >= 10:
        try:
            classifier = CalibratedClassifierCV(
                base_classifier,
                cv=min(3, int(min(np.bincount(y_train)))),
                method="sigmoid",
            )
            classifier.fit(x_train_features, y_train)
        except ValueError:
            classifier = base_classifier
            classifier.fit(x_train_features, y_train)
    else:
        classifier = base_classifier
        classifier.fit(x_train_features, y_train)

    train_scores = classifier.predict_proba(x_train_features)[:, 1]
    test_scores = classifier.predict_proba(x_test_features)[:, 1]
    train_predictions = (train_scores >= 0.5).astype(int)
    test_predictions = (test_scores >= 0.5).astype(int)

    metrics = {
        "train_accuracy": float(accuracy_score(y_train, train_predictions)),
        "test_accuracy": float(accuracy_score(y_test, test_predictions)),
        "train_roc_auc": _safe_roc_auc(y_train, train_scores),
        "test_roc_auc": _safe_roc_auc(y_test, test_scores),
        "train_pr_auc": _safe_avg_precision(y_train, train_scores),
        "test_pr_auc": _safe_avg_precision(y_test, test_scores),
        "test_recall_failures": _safe_recall(y_test, test_predictions),
        "test_precision_failures": _safe_precision(y_test, test_predictions),
        "test_brier_score": float(brier_score_loss(y_test, test_scores)) if len(y_test) > 0 else float("nan"),
        "test_confusion_matrix": _confusion_dict(y_test, test_predictions),
        "split_date": split_date.isoformat(),
        "label_mean_train": float(np.mean(y_train)) if len(y_train) > 0 else float("nan"),
        "label_mean_test": float(np.mean(y_test)) if len(y_test) > 0 else float("nan"),
        "calibration_bins": calibration_bins(y_test, test_scores),
    }

    coefficients = _coefficients_from(classifier)
    metadata = {
        "model_type": "SentenceTransformer embeddings + calibrated logistic regression on snapshot features",
        "horizon_years": int(horizon_years),
        "snapshot_count_train": int(len(train_frame)),
        "snapshot_count_test": int(len(test_frame)),
        "company_count": int(snapshots["company_id"].nunique()),
        "text_model_name": selected_text_model,
        "text_embedding_dim": int(embedding_dim),
        "dependency_versions": dependency_versions(),
        "sklearn_version": sklearn.__version__,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "split_date": split_date.isoformat(),
        "seed": int(seed),
        "metrics": metrics,
        "feature_count": int(len(feature_names)),
    }

    model = StartupRiskModel(
        structured_preprocessor=structured_preprocessor,
        classifier=classifier,
        metadata=metadata,
        feature_names=feature_names,
        coefficients=coefficients,
        numeric_baselines=numeric_baselines(snapshots),
        text_model_name=selected_text_model,
        text_embedding_dim=embedding_dim,
    )
    model.save(model_path)
    write_report(report_path, model, metrics, train_frame, test_frame, horizon_years)

    return {
        "model_path": str(model_path),
        "report_path": str(report_path),
        "metrics": metrics,
        "metadata": metadata,
    }


def time_based_split(
    snapshots: pd.DataFrame,
    train_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame, date]:
    ordered = snapshots.sort_values("snapshot_date").reset_index(drop=True)
    if len(ordered) == 0:
        raise ValueError("cannot split empty snapshot frame")

    cutoff_index = max(1, min(len(ordered) - 1, int(round(len(ordered) * train_fraction))))
    split_date_str = ordered.loc[cutoff_index, "snapshot_date"]
    split_date = pd.to_datetime(split_date_str).date()
    train = ordered[pd.to_datetime(ordered["snapshot_date"]).dt.date < split_date].copy()
    test = ordered[pd.to_datetime(ordered["snapshot_date"]).dt.date >= split_date].copy()
    return train, test, split_date


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
    feature_names = [str(name) for name in structured_preprocessor.get_feature_names_out()]
    embedding_names = [
        f"text_embedding__dim_{index:03d}" for index in range(text_embeddings.shape[1])
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


def calibration_bins(y: np.ndarray, scores: np.ndarray, bins: int = 5) -> list[dict[str, float]]:
    if len(y) == 0:
        return []
    edges = np.linspace(0.0, 1.0, bins + 1)
    output: list[dict[str, float]] = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (scores >= lo) & (scores < hi if hi < 1.0 else scores <= hi)
        if not np.any(in_bin):
            continue
        output.append(
            {
                "range": f"[{lo:.2f}, {hi:.2f}]",
                "count": int(in_bin.sum()),
                "mean_predicted": float(scores[in_bin].mean()),
                "fraction_positive": float(y[in_bin].mean()),
            }
        )
    return output


def _safe_roc_auc(y: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, scores))


def _safe_avg_precision(y: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(average_precision_score(y, scores))


def _safe_recall(y: np.ndarray, predictions: np.ndarray) -> float:
    if len(y) == 0 or y.sum() == 0:
        return float("nan")
    return float(recall_score(y, predictions, zero_division=0))


def _safe_precision(y: np.ndarray, predictions: np.ndarray) -> float:
    if len(y) == 0 or predictions.sum() == 0:
        return float("nan")
    return float(precision_score(y, predictions, zero_division=0))


def _confusion_dict(y: np.ndarray, predictions: np.ndarray) -> dict[str, int]:
    if len(y) == 0:
        return {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
    tn, fp, fn, tp = confusion_matrix(y, predictions, labels=[0, 1]).ravel()
    return {"tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn)}


def _coefficients_from(classifier: Any) -> list[float]:
    if hasattr(classifier, "coef_"):
        return [float(value) for value in classifier.coef_[0].tolist()]
    if hasattr(classifier, "calibrated_classifiers_"):
        coefs = []
        for member in classifier.calibrated_classifiers_:
            inner = getattr(member, "estimator", None) or getattr(member, "base_estimator", None)
            if inner is not None and hasattr(inner, "coef_"):
                coefs.append(np.asarray(inner.coef_[0], dtype=np.float64))
        if coefs:
            return [float(value) for value in np.mean(coefs, axis=0).tolist()]
    return []


def write_report(
    path: str | Path,
    model: StartupRiskModel,
    metrics: dict[str, Any],
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    horizon_years: int,
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    feature_weights = sorted(
        zip(model.feature_names, model.coefficients),
        key=lambda item: item[1] if not np.isnan(item[1]) else 0.0,
        reverse=True,
    ) if model.coefficients else []
    top_risk = feature_weights[:10]
    top_protective = list(reversed(feature_weights[-10:])) if len(feature_weights) > 10 else []

    train_pos = int(train_frame["label"].sum()) if not train_frame.empty else 0
    test_pos = int(test_frame["label"].sum()) if not test_frame.empty else 0

    lines = [
        "# Startup Failure Prediction Evaluation",
        "",
        f"Prediction horizon: **{horizon_years} years** after snapshot_date.",
        "",
        "Snapshots are generated per company (yearly after founding and ~30 days post-funding). "
        "Each snapshot only uses information known before snapshot_date.",
        "",
        "## Data",
        "",
        f"- Companies: {model.metadata['company_count']}",
        f"- Training snapshots: {len(train_frame)} (positives: {train_pos})",
        f"- Test snapshots: {len(test_frame)} (positives: {test_pos})",
        f"- Time split date: snapshots before {metrics['split_date']} are train, on/after are test",
        f"- Label rule: 1 if failed within {horizon_years}y of snapshot_date; 0 if confirmed alive at snapshot_date + {horizon_years}y; censored snapshots are excluded.",
        f"- Leakage guard: excluded post-outcome fields {EXCLUDED_LEAKAGE_FIELDS}",
        f"- Feature count: {len(model.feature_names)}",
        f"- Text encoder: `{model.text_model_name}` ({model.text_embedding_dim} dims)",
        "",
        "## Metrics (test split)",
        "",
        f"- ROC-AUC: {metrics['test_roc_auc']:.3f}",
        f"- PR-AUC (average precision): {metrics['test_pr_auc']:.3f}",
        f"- Recall on failures (positives): {metrics['test_recall_failures']:.3f}",
        f"- Precision on failures: {metrics['test_precision_failures']:.3f}",
        f"- Brier score (lower is better calibrated): {metrics['test_brier_score']:.3f}",
        f"- Accuracy: {metrics['test_accuracy']:.3f}",
        f"- Confusion matrix: {metrics['test_confusion_matrix']}",
        f"- Positive class rate (train / test): {metrics['label_mean_train']:.3f} / {metrics['label_mean_test']:.3f}",
        "",
        "## Calibration (test split)",
        "",
        "Reliability bins compare model probability to observed failure rate. Well-calibrated bins have `fraction_positive ≈ mean_predicted`.",
        "",
    ]
    if metrics["calibration_bins"]:
        lines.append("| Probability range | Count | Mean predicted | Fraction positive |")
        lines.append("|---|---|---|---|")
        for entry in metrics["calibration_bins"]:
            lines.append(
                f"| {entry['range']} | {entry['count']} | {entry['mean_predicted']:.3f} | {entry['fraction_positive']:.3f} |"
            )
    else:
        lines.append("_No calibration bins to report (insufficient test data)._")

    lines.extend(
        [
            "",
            "## Top Failure-Risk Coefficients",
            "",
        ]
    )
    if top_risk:
        lines.extend(f"- `{name}`: {weight:.4f}" for name, weight in top_risk)
    else:
        lines.append("_Coefficients unavailable for this classifier._")

    if top_protective:
        lines.extend(["", "## Top Protective Coefficients", ""])
        lines.extend(f"- `{name}`: {weight:.4f}" for name, weight in top_protective)

    lines.extend(
        [
            "",
            "## Notes",
            "",
            f"- Each company contributes multiple snapshots (1y after founding, after funding rounds, and yearly thereafter).",
            f"- Time-based split: train on early snapshots, test on later ones. This catches temporal drift but with the current tiny dataset the test ROC-AUC is noisy.",
            f"- Replace `data/companies_raw.csv` and `data/funding_events.csv` with real records (loot-drop failures + Crunchbase survivors) before reading metrics seriously.",
        ]
    )

    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the snapshot-based startup failure prediction model."
    )
    parser.add_argument("--companies", default=str(COMPANIES_RAW_PATH))
    parser.add_argument("--events", default=str(FUNDING_EVENTS_PATH))
    parser.add_argument("--model", default=str(MODEL_PATH))
    parser.add_argument("--report", default=str(REPORT_PATH))
    parser.add_argument("--horizon-years", type=int, default=PREDICTION_HORIZON_YEARS)
    parser.add_argument("--time-split", type=float, default=TIME_SPLIT_FRACTION)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--text-model", default=None)
    parser.add_argument("--no-calibrate", action="store_true")
    args = parser.parse_args()

    result = train_model(
        companies_path=args.companies,
        funding_events_path=args.events,
        model_path=args.model,
        report_path=args.report,
        horizon_years=args.horizon_years,
        time_split_fraction=args.time_split,
        seed=args.seed,
        text_model_name=args.text_model,
        calibrate=not args.no_calibrate,
    )
    metrics = result["metrics"]
    print(f"model: {result['model_path']}")
    print(f"report: {result['report_path']}")
    print(f"split_date: {metrics['split_date']}")
    print(f"test_roc_auc: {metrics['test_roc_auc']:.3f}")
    print(f"test_pr_auc: {metrics['test_pr_auc']:.3f}")
    print(f"test_recall_failures: {metrics['test_recall_failures']:.3f}")
    print(f"test_brier_score: {metrics['test_brier_score']:.3f}")


if __name__ == "__main__":
    main()
