# CatBoost analysis code

This folder contains the recovered CatBoost point-model, temporal-recency, uncertainty,
explainability, hierarchical-attribution, and RF–CatBoost comparison scripts used in the
manuscript analysis.

## Scripts

- `catboost_recency_analysis.py`
  - training-only CatBoost model selection
  - full inverse-DOI point prediction
  - historical rolling-origin recency-lambda selection
  - chronological comparison with the frozen Random Forest baseline

- `run_catboost_uncertainty.py`
  - CatBoost MultiQuantile / calibrated interval and supported-domain analysis

- `catboost_shap_ale.py`
  - held-out feature-family SHAP and accumulated-local-effect analysis

- `catboost_within_family.py`
  - hierarchical within-family attribution

- `compare_rf_catboost.py`
  - Random Forest–CatBoost robustness comparison

- `create_final_figure5.py`
  - final uncertainty / OOD figure generation

- `prepare_origin_sources.py`
  - Origin-ready source-table preparation

- `baseline-code/psc_baseline_validation.py`
  - baseline feature-construction module retained with the recovered archive

## Frozen point-model outputs

The original point-model / recency prediction table and audit summaries are provided in:

```text
../../results/catboost_recency/
```

In particular, downstream CatBoost analyses use:

```text
../../results/catboost_recency/catboost_recency_predictions.csv.gz
```

for numerical reconciliation.

## Random-Forest uncertainty/OOD dependency

`run_catboost_uncertainty.py` imports the original RF uncertainty/OOD helper module from:

```text
../../uncertainty_rf/PSC_uncertainty_OOD_analysis_package/code/
```

That dependency is now included in the public repository.

## Frozen configuration

The corresponding CatBoost model-selection and run-manifest files are in:

```text
../../config/catboost/
```

The archived CatBoost run used Python 3.12.13, NumPy 2.3.5, pandas 2.2.3,
scikit-learn 1.8.0, and CatBoost 1.2.10.

See `requirements-catboost.txt`.

## Reproducibility status

The source-code chain for the baseline, DOI-balanced weighting, CatBoost point/recency,
CatBoost uncertainty, SHAP/ALE, and hierarchical-attribution analyses is now recovered.

The remaining manuscript-specific code to add is the locked composition-domain,
Sn-subgroup / mixture-of-experts, and mixed Pb–Sn warning-policy analysis.
