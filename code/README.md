# Analysis code

This directory contains the analysis scripts used for the publication-aware perovskite solar-cell machine-learning benchmark.

The frozen cohort and validation assignments are provided in `../data/`.

## Baseline and DOI-weighting analyses

- `psc_baseline_validation.py` — feature construction, Elastic Net / Random Forest baselines, row-wise, DOI-grouped, and chronological validation
- `doi_balanced_weighting.py` — tempered and full inverse-DOI sample weighting
- `verify_results.py` — deterministic integrity checks for the baseline analysis
- `verify_weighting_results.py` — deterministic integrity checks for the DOI-weighting analysis

## CatBoost analyses

The `catboost/` subdirectory contains the recovered scripts for:

- CatBoost quantile uncertainty and supported-domain diagnostics;
- held-out SHAP and ALE analyses;
- hierarchical within-family attribution;
- RF–CatBoost robustness comparison; and
- preparation of final figure / Origin-ready source tables.

See `catboost/README.md` for details and remaining external dependencies.
