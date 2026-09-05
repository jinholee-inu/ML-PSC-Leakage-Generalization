# Mixed Pb–Sn warning-policy analysis

This folder contains the authentic locked Green/Amber/Red warning-policy analysis for
mixed Pb–Sn perovskite solar-cell predictions from the fully DOI-balanced Random Forest.

## Scripts

- `mixed_pb_sn_warning_policy.py`
  - does not retrain the Random Forest
  - applies the locked warning rule to 2019–2021 mixed Pb–Sn predictions
  - uses only historical composition support, historical exact-formula novelty,
    training-referenced OOD percentiles, and calibrated interval-width diagnostics
    for tier assignment
  - uses future measured outcomes only to evaluate warning enrichment

- `verify_mixed_pb_sn_warning_policy.py`
  - independently reconstructs the tier assignments
  - checks the frozen prediction join
  - checks composition-cell support and calibration floors
  - recomputes metrics, Red/Amber comparisons, sensitivity selections, and bootstrap settings

The original report builder and Figure 10 exports are intentionally excluded from this public
code folder because they do not generate the underlying scientific results.

## Locked policy

Mixed Pb–Sn domain support:
- Green eligibility: >=100 historical DOI groups
- Amber: 10–99 historical DOI groups
- Red: <10 historical DOI groups

A-site × B-site composition-cell support:
- Green: >=10 historical DOI groups
- Amber: 5–9 historical DOI groups
- Red: <5 historical DOI groups

Additional Red triggers:
- exact short-form formula unseen historically
- feature-space OOD percentile >=0.95
- model-support OOD percentile >=0.95
- 95% interval half-width >=2.5× target-specific calibrated floor

Additional Amber triggers:
- feature/model OOD percentile >=0.90 and <0.95
- 95% interval half-width >=1.5× and <2.5× calibrated floor

The final tier is the more severe of the domain-support and uncertainty tiers.

## Calibrated interval floor

For each target, the 95% calibration floor used here is computed from the chronological
95% RF calibration row as:

`calibrated_floor_95_half_width = sigma_floor × normalized_residual_quantile`

The warning multiplier is:

`interval_95_half_width / calibrated_floor_95_half_width`

The required calibration table is provided at:

`../../../results/uncertainty_rf/uncertainty_calibration_diagnostics.csv`

## Upstream inputs already present in the repository

- `results/composition_domain/composition_domain_predictions.csv.gz`
- `results/composition_domain/composition_domain_support.csv`
- `results/uncertainty_rf/uncertainty_calibration_diagnostics.csv`

The input SHA-256 hashes in the archived warning-policy run manifest match the recovered
composition-domain and RF-uncertainty source packages exactly.

## Archived integrity

- mixed Pb–Sn historical support: 274 records / 57 DOI groups
- future chronological support: 98 records / 25 DOI groups
- target-level warning assignments: 392
- duplicate prediction keys: 0
- Green assignments: 0
- Red counts: PCE 46, Voc 72, Jsc 91, FF 45
- DOI-cluster bootstrap replicates: 1,000
- independent verification: 24 / 24 checks passed
