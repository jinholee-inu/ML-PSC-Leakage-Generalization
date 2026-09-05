# CatBoost configuration

This directory contains frozen configuration and run metadata associated with the CatBoost analyses.

## Files

- `catboost_model_selection.csv` — candidate-model comparison used during training-only model selection
- `catboost_point_model_run_manifest.json` — archived point-model / recency run configuration, software versions, input hashes, and selected model settings
- `baseline_run_manifest.json` — baseline-analysis manifest used for cross-checking the CatBoost workflow

These files should be treated as frozen provenance records and should not be edited when reproducing the published analysis.
