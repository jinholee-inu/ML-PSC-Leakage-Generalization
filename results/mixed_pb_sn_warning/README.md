# Mixed Pb–Sn warning-policy outputs

This directory contains the frozen machine-readable outputs from the locked mixed Pb–Sn
Green/Amber/Red warning-policy analysis.

## Files

- `mixed_pb_sn_warning_assignments.csv.gz`
  - row- and target-level warning tiers, trigger flags, OOD values, interval widths, and actions

- `mixed_pb_sn_warning_policy.csv`
  - machine-readable locked threshold table

- `mixed_pb_sn_warning_metrics.csv`
  - device-level and publication-balanced tier metrics with bootstrap intervals

- `mixed_pb_sn_warning_comparisons.csv`
  - Red-versus-Amber paired DOI-cluster bootstrap comparisons

- `mixed_pb_sn_warning_trigger_summary.csv`
  - trigger counts and publication-balanced prevalence

- `mixed_pb_sn_warning_sensitivity.csv`
  - permissive, locked-primary, and conservative sensitivity variants

- `mixed_pb_sn_warning_run_manifest.json`
  - software versions, input hashes, locked design, and integrity summary

- `independent_mixed_pb_sn_warning_verification.json`
  - independent 24-check verification report

## Integrity summary

- future mixed Pb–Sn records: 98
- future DOI groups: 25
- target-level assignment rows: 392
- duplicate assignment keys: 0
- warning tiers present: Amber and Red
- independent verification: 24 / 24 checks passed
