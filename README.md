# Publication Leakage and Temporal Generalization in Machine Learning for Perovskite Solar Cells

This repository contains data indices, frozen validation assignments, analysis code, and configuration files supporting the manuscript:

**Publication Leakage and Temporal Generalization in Machine Learning for Perovskite Solar Cells**

The study evaluates how publication-level dependence, chronological distribution shift, publication-balanced training, predictive uncertainty, and composition-domain support affect claims of generalization in literature-derived perovskite solar-cell machine learning.

## Repository structure

```text
ML-PSC-Leakage-Generalization/
├── data/                  # frozen cohort index, validation splits, and provenance
├── code/                  # baseline, DOI-weighting, and CatBoost analysis scripts
└── config/                # frozen CatBoost model-selection and run manifests
```

## Data

The analysis uses the frozen **31 March 2022** snapshot of the public Perovskite Database Project.

The raw database CSV is not redistributed here. The repository instead provides:

- the final 33,175-record cohort index;
- normalized DOI-group assignments;
- frozen row-wise, DOI-grouped, and chronological validation assignments;
- data-audit counts; and
- SHA-256 provenance information.

See [`data/README.md`](data/README.md).

## Analysis code

The current code release includes:

- baseline Elastic Net and Random Forest validation;
- row-wise versus DOI-grouped publication-leakage analysis;
- chronological validation;
- DOI-balanced sample weighting;
- CatBoost uncertainty analysis;
- held-out SHAP and ALE analysis;
- hierarchical within-family attribution; and
- RF–CatBoost robustness comparison.

See [`code/README.md`](code/README.md) and [`code/catboost/README.md`](code/catboost/README.md).

## Reproducibility status

The frozen cohort and split assignments are public in this repository. Baseline and DOI-weighting verification scripts are included.

The recovered CatBoost archive is authentic and hash-traceable, but two earlier dependencies still need to be added before the repository is considered fully self-contained for an end-to-end rerun:

1. the Random-Forest uncertainty/OOD helper module used by the CatBoost uncertainty script; and
2. the CatBoost point-model/recency analysis package that generated the frozen point predictions used for reconciliation.

These components will be added from the original archived analysis files before the final versioned release.

## Citation

A manuscript citation and archived DOI will be added after acceptance / final repository archiving.

## Contact

Soonil Hong — Korea Research Institute of Chemical Technology  
Jinho Lee — Incheon National University
