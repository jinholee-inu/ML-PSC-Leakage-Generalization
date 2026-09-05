# Publication Leakage and Temporal Generalization in Machine Learning for Perovskite Solar Cells

This repository contains frozen data indices, validation assignments, analysis code,
configuration files, and selected machine-readable outputs supporting the manuscript:

**Publication Leakage and Temporal Generalization in Machine Learning for Perovskite Solar Cells**

## Repository structure

```text
ML-PSC-Leakage-Generalization/
├── data/
├── code/
│   ├── catboost/
│   └── reliability/
│       ├── composition_domain/
│       └── sn_subgroup/
├── config/
├── results/
│   ├── catboost_recency/
│   ├── composition_domain/
│   └── sn_subgroup/
└── uncertainty_rf/
```

## Recovered analysis chain

The repository currently contains source code and frozen inputs/outputs for:

- baseline Elastic Net and Random-Forest validation;
- publication-leakage analysis;
- DOI-balanced sample weighting;
- Random-Forest uncertainty and OOD analysis;
- CatBoost point-model and temporal-recency analysis;
- CatBoost uncertainty;
- held-out SHAP and ALE;
- hierarchical within-family attribution;
- RF–CatBoost robustness comparison;
- locked composition-domain reliability analysis; and
- Sn-only / mixed Pb–Sn subgroup calibration and mixture-of-experts analysis.

## Remaining reliability analysis

The remaining manuscript-specific source package to add is the locked mixed Pb–Sn
warning-policy analysis.

After that package is added, the repository can receive a final manuscript-to-code audit
and be prepared for a versioned archival release / persistent DOI.
