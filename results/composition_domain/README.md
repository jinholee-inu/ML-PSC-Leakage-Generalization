# Composition-domain reliability outputs

This directory contains the frozen machine-readable outputs from the composition-domain
reliability audit.

## Key downstream files

- `composition_domain_assignments.csv.gz`
  - composition labels, historical/future assignment, and exact-formula novelty for all 33,175 records

- `composition_domain_predictions.csv.gz`
  - frozen 2019–2021 paired Random-Forest prediction / interval / OOD table used by the audit
  - important input for the Sn-subgroup and mixed Pb–Sn warning analyses

- `composition_domain_support.csv`
  - historical and future device/DOI support by composition domain

## Audit summaries

- `composition_domain_performance.csv`
- `composition_domain_comparisons.csv`
- `composition_domain_high_efficiency_PCE.csv`
- `composition_domain_run_manifest.json`
- `composition_domain_verification_report.json`
- `independent_composition_domain_verification_report.json`

## Integrity

- composition rows: 33,175
- normalized DOI groups: 6,368
- future paired prediction rows: 72,928
- duplicate prediction keys: 0
- missing composition joins: 0
- independent verification: 19 / 19 checks passed
