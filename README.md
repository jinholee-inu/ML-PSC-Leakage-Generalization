# Publication Leakage and Temporal Generalization in Machine Learning for Perovskite Solar Cells

This repository contains frozen data indices, validation assignments, analysis code,
configuration files, and selected machine-readable outputs supporting the manuscript:

**Publication Leakage and Temporal Generalization in Machine Learning for Perovskite Solar Cells**

The study evaluates how publication-level dependence, chronological distribution shift,
publication-balanced training, predictive uncertainty, and composition-domain support affect
claims of generalization in literature-derived perovskite solar-cell machine learning.

## Repository structure

```text
ML-PSC-Leakage-Generalization/
├── data/                  # frozen cohort index, validation splits, and provenance
├── code/                  # baseline, DOI-weighting, and CatBoost analysis scripts
├── config/                # frozen CatBoost model-selection and run manifests
├── results/               # selected frozen machine-readable outputs
└── uncertainty_rf/        # recovered Random-Forest uncertainty/OOD dependency
```

## Data

The analysis uses the frozen **31 March 2022** snapshot of the public Perovskite Database Project.

The raw database CSV is not redistributed here. The repository provides the final
33,175-record cohort index, normalized DOI-group assignments, frozen validation assignments,
data-audit counts, and SHA-256 provenance information.

See `data/README.md`.

## Analysis code currently recovered

- baseline Elastic Net and Random Forest validation;
- row-wise versus DOI-grouped publication-leakage analysis;
- chronological validation;
- DOI-balanced sample weighting;
- CatBoost training-only model selection and temporal-recency analysis;
- CatBoost uncertainty analysis;
- held-out SHAP and ALE analysis;
- hierarchical within-family attribution; and
- RF–CatBoost robustness comparison.

The original RF uncertainty/OOD helper package required by the CatBoost uncertainty analysis
is also included.

## Frozen outputs

`results/catboost_recency/` contains the original CatBoost point-model / recency prediction
table and compact audit summaries used by downstream CatBoost analyses.

## Remaining manuscript-specific code

Before the final archived release, the repository should also include the locked
composition-domain, Sn-subgroup / mixture-of-experts, and mixed Pb–Sn warning-policy
analyses, together with the final selected result tables needed by the data-availability statement.

## Citation

A manuscript citation and persistent archived DOI will be added with the final release.

## Contact

Soonil Hong — Korea Research Institute of Chemical Technology  
Jinho Lee — Incheon National University
