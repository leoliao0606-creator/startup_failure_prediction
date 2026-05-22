from __future__ import annotations

import tempfile
import unittest
from datetime import date
from pathlib import Path

from startup_failure_prediction.config import COMPANIES_RAW_PATH, PREDICTION_HORIZON_YEARS
from startup_failure_prediction.model import StartupRiskModel
from startup_failure_prediction.predict import DEFAULT_EXAMPLE, predict_payload
from startup_failure_prediction.snapshots import (
    Company,
    FundingEvent,
    build_snapshots,
    candidate_snapshot_dates,
    label_for_snapshot,
    load_companies,
)
from startup_failure_prediction.train import FUNDING_EVENTS_PATH, train_model


def _make_company(
    outcome: str,
    founded: date,
    outcome_date: date | None = None,
    last_observed: date = date(2026, 5, 21),
) -> Company:
    return Company(
        company_id="co_test",
        company_name="Test Co",
        industry="SaaS",
        product_type="SaaS",
        country="USA",
        founded_date=founded,
        outcome=outcome,
        outcome_date=outcome_date,
        last_observed_date=last_observed,
        market_score=60,
        scalability_score=70,
        company_description="Generic test description.",
        founder_statement="Generic founder note.",
        funding_events=(
            FundingEvent(round_date=date(founded.year + 1, 1, 1), round_name="Seed", amount_usd=1_000_000),
        ),
    )


class LabelRuleTest(unittest.TestCase):
    def test_failed_within_horizon_is_positive(self) -> None:
        company = _make_company(
            outcome="failed",
            founded=date(2018, 1, 1),
            outcome_date=date(2020, 6, 1),
        )
        label, censored, _ = label_for_snapshot(company, date(2019, 1, 1), horizon_years=3)
        self.assertEqual(label, 1)
        self.assertFalse(censored)

    def test_failed_after_horizon_is_negative(self) -> None:
        company = _make_company(
            outcome="failed",
            founded=date(2010, 1, 1),
            outcome_date=date(2023, 1, 1),
        )
        label, censored, _ = label_for_snapshot(company, date(2015, 1, 1), horizon_years=3)
        self.assertEqual(label, 0)
        self.assertFalse(censored)

    def test_operating_with_long_observation_is_negative(self) -> None:
        company = _make_company(
            outcome="operating",
            founded=date(2015, 1, 1),
            last_observed=date(2026, 5, 21),
        )
        label, censored, _ = label_for_snapshot(company, date(2020, 1, 1), horizon_years=3)
        self.assertEqual(label, 0)
        self.assertFalse(censored)

    def test_operating_too_recent_is_censored(self) -> None:
        company = _make_company(
            outcome="operating",
            founded=date(2024, 1, 1),
            last_observed=date(2026, 5, 21),
        )
        label, censored, _ = label_for_snapshot(company, date(2025, 1, 1), horizon_years=3)
        self.assertIsNone(label)
        self.assertTrue(censored)


class SnapshotGenerationTest(unittest.TestCase):
    def test_candidate_dates_skip_before_min_age_and_after_outcome(self) -> None:
        company = _make_company(
            outcome="failed",
            founded=date(2018, 1, 1),
            outcome_date=date(2020, 6, 1),
        )
        candidates = candidate_snapshot_dates(company)
        for snapshot_date in candidates:
            self.assertGreaterEqual((snapshot_date - company.founded_date).days, 180)
            self.assertLess(snapshot_date, company.outcome_date)

    def test_build_snapshots_excludes_censored_by_default(self) -> None:
        companies = [
            _make_company(
                outcome="operating",
                founded=date(2024, 1, 1),
                last_observed=date(2026, 5, 21),
            ),
            _make_company(
                outcome="failed",
                founded=date(2018, 1, 1),
                outcome_date=date(2020, 6, 1),
            ),
        ]
        frame = build_snapshots(companies, horizon_years=3)
        self.assertFalse(frame["censored"].any())

    def test_load_companies_returns_real_dataset(self) -> None:
        companies = load_companies(COMPANIES_RAW_PATH, FUNDING_EVENTS_PATH)
        self.assertGreater(len(companies), 30)
        outcomes = {c.outcome for c in companies}
        self.assertEqual(outcomes, {"failed", "operating"})


class PipelineTest(unittest.TestCase):
    def test_train_save_load_predict(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.joblib"
            report_path = Path(tmpdir) / "report.md"
            result = train_model(
                model_path=model_path,
                report_path=report_path,
                seed=7,
            )

            self.assertTrue(model_path.exists())
            self.assertTrue(report_path.exists())
            self.assertIn("test_roc_auc", result["metrics"])
            self.assertEqual(result["metadata"]["horizon_years"], PREDICTION_HORIZON_YEARS)

            model = StartupRiskModel.load(model_path)
            prediction = predict_payload(model, DEFAULT_EXAMPLE)
            self.assertGreaterEqual(prediction["risk_probability"], 0.0)
            self.assertLessEqual(prediction["risk_probability"], 1.0)
            self.assertEqual(prediction["horizon_years"], PREDICTION_HORIZON_YEARS)


if __name__ == "__main__":
    unittest.main()
