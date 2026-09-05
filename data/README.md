# Data

This directory contains the minimal data metadata and frozen analysis indices for the manuscript
**Publication Leakage and Temporal Generalization in Machine Learning for Perovskite Solar Cells**
(project repository: ML-PSC-Leakage-Generalization).

## Source dataset

The study uses the frozen **31 March 2022** snapshot of the public Perovskite Database Project.
The raw database CSV is **not redistributed in this repository**. Reproduction should use the
same snapshot and verify its SHA-256 checksum recorded in `source_snapshot.json`.

## Final analysis cohort

- Raw database records: 42,459
- Final quality-controlled cohort: 33,175 device records
- Normalized DOI groups: 6,368
- Chronological training cohort (publication year <= 2018): 24,059 records / 4,633 DOI groups
- Chronological test cohort (2019-2021): 9,116 records / 1,735 DOI groups

## Files

- `source_snapshot.json`  
  Source metadata, dataset checksum, and cohort checksum.

- `filter_counts.csv`  
  Sequential cohort-retention counts used for the data-audit summary and Figure 1.

- `final_cohort_index.csv`  
  **To be added from the frozen analysis workspace.** This is the authoritative 33,175-record
  cohort index used by the manuscript analyses. Do not reconstruct this file manually from the manuscript.

- `split_manifest.csv`  
  **To be added from the frozen baseline-analysis workspace.** This is the authoritative frozen
  row-wise, DOI-grouped, and chronological split assignment table. Do not regenerate the folds
  after publication.

## Reproducibility note

The two files marked “To be added” are required before the repository is considered complete.
They should be copied from the original analysis outputs so that record identifiers and split
assignments exactly match the published predictions.
