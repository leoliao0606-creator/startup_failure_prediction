# Startup Failure Prediction MVP Evaluation

This report is generated from the local synthetic seed dataset. Replace `data/cleaned_startups.csv` with a real balanced dataset before using the score for decisions.

## Data

- Training rows: 22
- Test rows: 8
- Leakage guard: excluded post-outcome fields: failure_reason
- Feature count: 424
- Text encoder: `sentence-transformers/all-MiniLM-L6-v2`
- Text embedding dimensions: 384

## Metrics

- Train accuracy: 1.000
- Test accuracy: 1.000
- Train ROC-AUC: 1.000
- Test ROC-AUC: 1.000
- Test confusion matrix: {'tp': 4, 'tn': 4, 'fp': 0, 'fn': 0}

## Transformer Channel

- Model: `sentence-transformers/all-MiniLM-L6-v2`
- Embedding dimensions: 384
- Versions: {'scikit_learn': '1.8.0', 'sentence_transformers': '5.5.0', 'torch': '2.12.0+cu130', 'transformers': '5.8.1'}

## Top Failure-Risk Weights

- `numeric__funding_total_usd`: 0.2081
- `categorical__product_type_Consumer App`: 0.1503
- `numeric__funding_rounds`: 0.1376
- `categorical__industry_Legaltech`: 0.1229
- `categorical__product_type_Hardware`: 0.1083
- `categorical__country_Brazil`: 0.0910
- `categorical__industry_HR Tech`: 0.0823
- `categorical__industry_Fintech`: 0.0592
- `categorical__industry_Foodtech`: 0.0504
- `categorical__product_type_Subscription`: 0.0504
- `text_embedding__dim_016`: 0.0378
- `text_embedding__dim_364`: 0.0355

## Top Protective Weights

- `numeric__market_score`: -1.0392
- `numeric__scalability_score`: -0.9677
- `numeric__operating_years`: -0.8814
- `categorical__product_type_SaaS`: -0.3013
- `categorical__industry_Healthcare`: -0.0959
- `categorical__country_Australia`: -0.0939
- `categorical__industry_Collaboration`: -0.0939
- `categorical__industry_Martech`: -0.0782
- `categorical__industry_Agtech`: -0.0740
- `categorical__country_USA`: -0.0653
- `categorical__product_type_Marketplace`: -0.0437
- `text_embedding__dim_271`: -0.0407

## MVP Notes

- This model uses a SentenceTransformer encoder for text semantics, then concatenates those embeddings with numeric and categorical features before logistic regression.
- The current dataset is intentionally small and illustrative; it validates the API and training pipeline but is not a production model.
- The next production step is to replace the seed data with real failed and successful companies, then rerun training and compare against gradient boosting and calibrated classifiers.
