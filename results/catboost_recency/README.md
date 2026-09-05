# CatBoost point-model and temporal-recency results

This directory contains the frozen machine-readable outputs from the original CatBoost
point-model / temporal-recency analysis.

## Key file

`catboost_recency_predictions.csv.gz`

This is the frozen prediction table used by the downstream CatBoost uncertainty,
SHAP/ALE, and hierarchical-attribution scripts for numerical reconciliation.

## Other files

- `catboost_recency_metrics.csv` — device-level and publication-balanced performance
- `catboost_recency_paired_comparison.csv` — paired RF/CatBoost and recency comparisons
- `rolling_origin_recency_lambda_results.csv` — historical pseudo-future lambda grid results
- `rolling_origin_recency_lambda_selection.csv` — one-standard-error lambda selection
- `chronological_PCE_catboost_recency_calibration.csv` — chronological PCE calibration
- `chronological_high_PCE_comparison.csv` — measured-PCE >=20% comparison
- `catboost_recency_fit_diagnostics.csv` — model-fit diagnostics
- `catboost_model_selection.csv` — training-only CatBoost candidate comparison
- `run_manifest.json` — frozen inputs, software versions, hashes, and selected settings
- `verification_report.json` — integrity checks from the original analysis

## Frozen analysis summary

- Final cohort: 33,175 records
- Normalized DOI groups: 6,368
- Prediction rows: 411,256
- Duplicate prediction keys: 0
- Grouped DOI overlap: 0
- Chronological DOI overlap: 0
- Rolling pseudo-futures: 2016, 2017, 2018
- Selected Random-Forest recency lambda: 0
- Selected CatBoost recency lambda: 0.025 year^-1
- The 2019–2021 holdout was not used for lambda selection

The final manuscript point predictor retains the simpler full inverse-DOI CatBoost model
without the recency term because the selected recency weighting did not produce a
statistically resolved PCE improvement on the untouched chronological holdout.
