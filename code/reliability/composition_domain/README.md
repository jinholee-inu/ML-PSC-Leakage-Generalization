# Composition-domain reliability analysis

This folder contains the authentic composition-domain reliability analysis used for the
locked, fully DOI-balanced Random-Forest stress test.

## Scripts

- `composition_domain_audit.py`
  - assigns FA/MA/Cs A-site and Pb/Sn B-site domains
  - evaluates exact-formula novelty using publications through 2018
  - joins frozen Random-Forest prediction, interval, and OOD quantities
  - compares DOI-grouped and chronological performance on the same 2019–2021 records
  - reports publication-balanced metrics and DOI-cluster bootstrap confidence intervals

- `verify_composition_domain_audit.py`
  - independently reconstructs composition labels and formula novelty
  - verifies the archived prediction join
  - recomputes domain support and publication-balanced metrics
  - checks confidence-interval ordering and frozen-input hashes

The Word/PDF report builder from the original package is intentionally excluded because it
does not generate scientific results.

## Archived design

- final cohort: 33,175 device records
- normalized DOI groups: 6,368
- historical records through 2018: 24,059
- future records from 2019–2021: 9,116
- future DOI groups: 1,735
- paired future prediction rows: 72,928
- duplicate prediction keys: 0
- DOI-cluster bootstrap replicates: 1,000
- minimum descriptive support: 5 future DOI groups
- minimum primary inferential support: 20 future DOI groups

The independent verifier passed 19/19 checks. The maximum difference from the archived
prediction source was 3.553e-15, and the maximum independently recomputed metric difference
was 7.105e-15.

## Inputs

The analysis requires:

1. the frozen 31 March 2022 raw Perovskite Database snapshot;
2. `data/final_cohort_index.csv`; and
3. the archived Random-Forest uncertainty/OOD prediction table
   (`uncertainty_ood_predictions.csv.gz`).

The large upstream uncertainty/OOD prediction table is not duplicated in this folder. It can
be retained in the final archival data release; the derived frozen composition-domain prediction
table used by downstream reliability analyses is provided in `results/composition_domain/`.
