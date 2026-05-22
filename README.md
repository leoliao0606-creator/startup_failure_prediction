# Startup Failure Prediction MVP

This repository is a runnable MVP for the system described in `startup_failure_prediction_brief.md`.

The current version is built for the existing `projects` conda environment. It includes:

- a cleaned seed dataset at `data/cleaned_startups.csv`
- feature engineering for numeric, categorical, and early text fields
- a SentenceTransformer text encoder plus numeric scaling, categorical one-hot encoding, and logistic regression
- model persistence to `models/startup_failure_model.joblib`
- an evaluation report at `reports/evaluation_report.md`
- a JSON prediction API with a minimal browser UI

If `conda` is not available in a non-interactive shell, use `/home/cliao/miniconda3/bin/conda` in place of `conda` in the commands below.

## Setup

Activate the existing environment and install this repo in editable mode once:

```bash
conda activate projects
python -m pip install -e .
```

The default text model is `sentence-transformers/all-MiniLM-L6-v2`. The first training run downloads it into the local HuggingFace cache. To use a different encoder:

```bash
python -m startup_failure_prediction.train --text-model sentence-transformers/all-mpnet-base-v2
```

After that, the package is importable from this environment and `python -m startup_failure_prediction.train` works from the project directory.

## Important Data Note

The included dataset is synthetic seed data for proving the pipeline. It is not a real loot-drop or Crunchbase export, and the model should not be used for investment or operating decisions until the dataset is replaced with real balanced failed and successful startup examples.

The pipeline keeps `failure_reason` in the CSV for later analysis, but excludes it from model features because it is usually known only after a company fails.

## Train

```bash
python -m startup_failure_prediction.train
```

This writes:

- `models/startup_failure_model.joblib`
- `reports/evaluation_report.md`

## Predict From CLI

```bash
python -m startup_failure_prediction.predict
```

Or pass a JSON file:

```bash
python -m startup_failure_prediction.predict --input-file path/to/company.json
```

## Run API

Train once, then start the API:

```bash
python -m startup_failure_prediction.api --port 8000
```

Open `http://127.0.0.1:8000` for the UI, or call the API directly:

```bash
curl -s http://127.0.0.1:8000/health
curl -s http://127.0.0.1:8000/example
curl -s -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"industry":"Ecommerce","product_type":"Marketplace","country":"USA","funding_total_usd":18000000,"funding_rounds":3,"founded_year":2022,"operating_years":3,"market_score":44,"scalability_score":58,"company_description":"A marketplace using subsidies to grow in a crowded category with weak retention.","founder_statement":"We are still searching for repeat usage after launch."}'
```

## Test

```bash
python -m unittest discover -s tests
```

## Next Steps

1. Replace `data/cleaned_startups.csv` with real failed and successful startup records.
2. Add a scraper or import job for loot-drop failed cases and a separate source for successful companies.
3. Re-train and compare this baseline against gradient boosting and calibrated classifiers.
4. Add calibration checks before exposing the probability as a decision metric.
