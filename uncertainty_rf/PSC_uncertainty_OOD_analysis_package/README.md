# Random-Forest uncertainty and OOD analysis

This directory contains the authentic Random-Forest uncertainty/OOD source package recovered
from the original manuscript analysis.

It is kept at this repository path intentionally:

```text
uncertainty_rf/PSC_uncertainty_OOD_analysis_package/code/
```

because the recovered CatBoost uncertainty script imports
`uncertainty_ood_analysis.py` from this location.

## Core files

- `code/uncertainty_ood_analysis.py`
  - refits the fully DOI-balanced Random Forest under frozen DOI-grouped and chronological partitions
  - extracts between-tree uncertainty
  - calibrates DOI-disjoint prediction intervals
  - calculates target-free feature-space OOD and forest-leaf support
  - generates coverage, error-association, OOD-stratified, and selective-prediction outputs

- `code/verify_uncertainty_ood_results.py`
  - independently checks prediction keys, archived-prediction reconciliation,
    interval ordering, finite outputs, and metric recalculation

Supporting baseline and DOI-weighting source files from the recovered package are retained in
`code/` so that this directory can be used as the original standalone code dependency.

## Archived run integrity

The recovered full run recorded:

- Python 3.12.13
- NumPy 2.3.5
- pandas 2.2.3
- scikit-learn 1.8.0
- 169,164 target-level uncertainty/OOD prediction rows
- 0 duplicate prediction keys
- 0 DOI overlap across grouped and chronological train/test boundaries
- 1,000 DOI-cluster bootstrap replicates
- maximum archived RF prediction difference: 1.4210854715202004e-14
- independent metric-recalculation difference: 7.105427357601002e-15

The corresponding archived manifests and verification reports are retained in `provenance/`.

## Data and large result files

Large archived result tables and duplicate copies of the cohort/split files are intentionally not
included in this GitHub staging package. The public repository already contains the frozen cohort
and split information under `data/`.

A full independent rerun of the RF uncertainty analysis additionally requires the archived
full-DOI-weighted Random-Forest predictions (`publication_weighting_predictions.csv.gz`).
That file should be added later under the repository `results/` package or archived with the
final Zenodo release.

## Raw database

The raw Perovskite Database CSV is not redistributed. The expected frozen 31 March 2022
snapshot SHA-256 is:

`9d30614b3a9228f2d66d4b09791e3210316fb717a17c11d3b70f18302ff074bb`
