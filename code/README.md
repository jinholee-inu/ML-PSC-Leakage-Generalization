# Analysis code

This directory contains the analysis scripts used for the
publication-aware perovskite solar-cell machine-learning benchmark.

The frozen cohort and validation assignments are provided in `../data/`.

Core analysis:
- `psc_baseline_validation.py`: baseline preprocessing, model fitting, and validation
- `doi_balanced_weighting.py`: publication-balanced training analysis

Verification:
- `verify_results.py`
- `verify_weighting_results.py`

Additional CatBoost, uncertainty, explainability, and domain-reliability
scripts will be added before the archived release.
