# Random-Forest uncertainty calibration outputs

This directory contains the compact RF uncertainty-calibration table needed by the
mixed Pb–Sn warning-policy analysis.

- `uncertainty_calibration_diagnostics.csv`

For the locked warning policy, the target-specific 95% calibrated floor is calculated from the
chronological 95% calibration row as:

`sigma_floor × normalized_residual_quantile`

The full RF uncertainty/OOD source package is retained under `uncertainty_rf/`.
