# Startup Failure Prediction Evaluation

Prediction horizon: **3 years** after snapshot_date.

Snapshots are generated per company (yearly after founding and ~30 days post-funding). Each snapshot only uses information known before snapshot_date.

## Data

- Companies: 50
- Training snapshots: 278 (positives: 66)
- Test snapshots: 93 (positives: 51)
- Time split date: snapshots before 2021-12-12 are train, on/after are test
- Label rule: 1 if failed within 3y of snapshot_date; 0 if confirmed alive at snapshot_date + 3y; censored snapshots are excluded.
- Leakage guard: excluded post-outcome fields ['failure_reason', 'outcome', 'outcome_date']
- Feature count: 433
- Text encoder: `sentence-transformers/all-MiniLM-L6-v2` (384 dims)

## Metrics (test split)

- ROC-AUC: 0.996
- PR-AUC (average precision): 0.997
- Recall on failures (positives): 0.902
- Precision on failures: 0.979
- Brier score (lower is better calibrated): 0.050
- Accuracy: 0.935
- Confusion matrix: {'tp': 46, 'tn': 41, 'fp': 1, 'fn': 5}
- Positive class rate (train / test): 0.237 / 0.548

## Calibration (test split)

Reliability bins compare model probability to observed failure rate. Well-calibrated bins have `fraction_positive ≈ mean_predicted`.

| Probability range | Count | Mean predicted | Fraction positive |
|---|---|---|---|
| [0.00, 0.20] | 35 | 0.040 | 0.000 |
| [0.20, 0.40] | 8 | 0.279 | 0.250 |
| [0.40, 0.60] | 4 | 0.480 | 1.000 |
| [0.60, 0.80] | 16 | 0.705 | 0.938 |
| [0.80, 1.00] | 30 | 0.884 | 1.000 |

## Top Failure-Risk Coefficients

- `numeric__funding_rounds_at_snapshot`: 1.1040
- `categorical__industry_HR Tech`: 0.7774
- `categorical__industry_Legaltech`: 0.7460
- `categorical__product_type_Consumer App`: 0.5814
- `numeric__funding_total_usd_at_snapshot`: 0.5518
- `categorical__product_type_Virtual Platform`: 0.5482
- `categorical__industry_Collaboration`: 0.5262
- `categorical__industry_Edtech`: 0.5147
- `categorical__industry_Crypto`: 0.4726
- `categorical__product_type_SaaS`: 0.4677

## Top Protective Coefficients

- `numeric__market_score`: -2.4123
- `categorical__industry_Proptech`: -1.2011
- `categorical__industry_Healthcare`: -1.0553
- `categorical__product_type_Hardware`: -1.0550
- `categorical__product_type_Marketplace`: -0.6635
- `categorical__country_USA`: -0.6279
- `categorical__country_Japan`: -0.5894
- `categorical__industry_Gaming`: -0.5894
- `categorical__industry_Logistics`: -0.5668
- `text_embedding__dim_073`: -0.4082

## Notes

- Each company contributes multiple snapshots (1y after founding, after funding rounds, and yearly thereafter).
- Time-based split: train on early snapshots, test on later ones. This catches temporal drift but with the current tiny dataset the test ROC-AUC is noisy.
- Replace `data/companies_raw.csv` and `data/funding_events.csv` with real records (loot-drop failures + Crunchbase survivors) before reading metrics seriously.
