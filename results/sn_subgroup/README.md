# Sn-subgroup calibration and mixture-of-experts outputs

This directory contains the frozen machine-readable outputs from the Sn-only / mixed Pb–Sn
subgroup reliability analysis.

## Core outputs

- `subgroup_future_predictions.csv.gz`
  - frozen future predictions for all domains, targets, and evaluated methods

- `subgroup_future_metrics.csv`
  - device-level and publication-balanced R², MAE, RMSE, bias, and confidence intervals

- `subgroup_paired_comparisons.csv`
  - paired changes relative to the frozen global Random Forest

- `subgroup_support.csv`
  - historical and future device / DOI support for Sn-only and mixed Pb–Sn domains

## Historical-only model selection

- `subgroup_calibrator_selection.csv`
- `subgroup_development_selection.csv`
- `subgroup_mixture_weight_selection.csv`
- `subgroup_policy_selection.csv`

## Robustness summaries

- `subgroup_PCE_calibration_bins.csv`
- `subgroup_PCE_upper_tail.csv`
- `subgroup_robustness_strata.csv`

## Provenance and verification

- `subgroup_calibration_moe_run_manifest.json`
- `independent_subgroup_calibration_moe_verification.json`

Integrity summary:
- prediction rows: 5,060
- duplicate prediction keys: 0
- historical/future DOI overlap: 0
- DOI-cluster bootstrap replicates: 1,000
- independent verification: 22 / 22 checks passed
