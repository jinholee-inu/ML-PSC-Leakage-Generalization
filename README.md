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
│       ├── sn_subgroup/
│       └── mixed_pb_sn_warning/
├── config/
├── results/
│   ├── catboost_recency/
│   ├── composition_domain/
│   ├── sn_subgroup/
│   ├── uncertainty_rf/
│   └── mixed_pb_sn_warning/
└── uncertainty_rf/
```

## Recovered analysis chain

The repository contains the recovered source code and frozen analysis inputs/outputs for:

- baseline Elastic Net and Random-Forest validation;
- row-wise publication-leakage analysis;
- DOI-grouped and chronological validation;
- DOI-balanced sample weighting;
- Random-Forest uncertainty and OOD analysis;
- CatBoost point-model and temporal-recency analysis;
- CatBoost uncertainty;
- held-out SHAP and ALE;
- hierarchical within-family attribution;
- RF–CatBoost robustness comparison;
- locked composition-domain reliability analysis;
- Sn-only / mixed Pb–Sn subgroup calibration and mixture-of-experts analysis; and
- the locked mixed Pb–Sn Green/Amber/Red warning-policy analysis.

## Reproducibility status

The manuscript-specific scientific analysis chain has now been recovered.

Before the final archival release, the repository should receive one final manuscript-to-code
and data-availability audit, any remaining compact result tables promised by the manuscript
should be added, and a versioned release should be archived with a persistent DOI.

The raw 31 March 2022 Perovskite Database snapshot is not redistributed here; provenance
and checksums are provided under `data/`.
