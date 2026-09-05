# Data

This directory contains the frozen data indices and provenance information used in the manuscript **Publication Leakage and Temporal Generalization in Machine Learning for Perovskite Solar Cells**.

## Source dataset

The study uses the frozen **31 March 2022** snapshot of the public Perovskite Database Project. The raw database CSV is not redistributed in this repository. Reproduction should use the same snapshot and verify the SHA-256 checksum recorded in `source_snapshot.json`.

## Final analysis cohort

- Raw database records: 42,459
- Final quality-controlled cohort: 33,175 device records
- Normalized DOI groups: 6,368
- Chronological training cohort (publication year <= 2018): 24,059 records / 4,633 DOI groups
- Chronological test cohort (2019–2021): 9,116 records / 1,735 DOI groups

## Files

### `final_cohort_index.csv`
Authoritative 33,175-record cohort used for the manuscript analyses.

### `split_manifest.csv`
Frozen validation assignments for the final cohort, including row-wise, DOI-grouped, and chronological partitions. These assignments should not be regenerated when reproducing the published analysis.

### `filter_counts.csv`
Sequential cohort-retention counts used in the data-audit summary.

### `source_snapshot.json`
Source-snapshot metadata and SHA-256 provenance information.

## Split integrity

The frozen split manifest contains 6,368 normalized DOI groups. No DOI crosses a DOI-grouped train/test boundary or the chronological training/test boundary.
