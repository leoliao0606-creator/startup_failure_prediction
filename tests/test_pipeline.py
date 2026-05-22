from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from startup_failure_prediction.model import StartupRiskModel
from startup_failure_prediction.predict import DEFAULT_EXAMPLE, predict_payload
from startup_failure_prediction.train import train_model


class PipelineTest(unittest.TestCase):
    def test_train_save_load_predict(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.json"
            report_path = Path(tmpdir) / "report.md"
            result = train_model(
                data_path="data/cleaned_startups.csv",
                model_path=model_path,
                report_path=report_path,
                seed=7,
            )

            self.assertTrue(model_path.exists())
            self.assertTrue(report_path.exists())
            self.assertIn("test_roc_auc", result["metrics"])

            model = StartupRiskModel.load(model_path)
            prediction = predict_payload(model, DEFAULT_EXAMPLE)
            self.assertGreaterEqual(prediction["risk_probability"], 0.0)
            self.assertLessEqual(prediction["risk_probability"], 1.0)
            self.assertGreater(len(prediction["top_risk_factors"]), 0)


if __name__ == "__main__":
    unittest.main()
