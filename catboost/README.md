PSC CatBoost uncertainty and explainability reproducibility package
==================================================================

Scope
-----
This package recalculates PCE uncertainty/OOD, grouped SHAP, ALE, and nested
within-family attribution for the final full 1/n_DOI weighted CatBoost model.
It preserves the audited 33,175-record cohort, 6,368 normalized DOI groups,
frozen DOI-grouped five-fold manifest, and chronological holdout (train <=2018;
test 2019--2021).

Main conclusions
----------------
1. CatBoost remains the final point predictor: chronological PCE R2 = 0.445
   and MAE = 3.024 percentage points.
2. CatBoost CQR is calibrated but conservative: publication-balanced future
   95% coverage = 0.963 and mean width = 17.36 PCE points.
3. CatBoost uncertainty ranks future error weakly (DOI-level rho = 0.123);
   retaining the least-uncertain 50% reduces publication-balanced MAE by 4.2%.
4. RF and CatBoost feature-family PCE explanations are highly stable
   (rho = 0.982; top-five overlap = 5/5).
5. Continuous-variable ALE rankings are model-sensitive (rho = 0.10;
   top-three overlap = 2/3) and should not be interpreted as causal design rules.

Directory guide
---------------
analysis/
  Full analysis and deliverable-generation scripts.
inputs/
  Frozen cohort index, split manifest, CatBoost model selection, and manifests.
results/uncertainty/
  CatBoost CQR predictions, coverage, OOD, selective-prediction tables, and Figure 5.
results/shap/
  Grouped SHAP, ALE tables, diagnostics, and Figure 6.
results/hierarchical/
  Nested within-family attribution, diagnostics, and Figure 7.
results/comparison/
  Paired RF--CatBoost DOI-bootstrap comparisons and robustness figure.
origin_source_csv/
  Flat CSV copies of every table used for Origin plotting.
deliverables/
  Word report, PDF preview, and Origin-ready Excel workbook.

Frozen input hashes
-------------------
Raw database CSV:
9d30614b3a9228f2d66d4b09791e3210316fb717a17c11d3b70f18302ff074bb

Final cohort index:
c8e637a7f6e2e47749642b313590377b0a6a4450f2a6519e04f1fbc9fc0f0bd6

Split manifest:
771d200195a14370a42d8f8ab7e1a39c712413e57858954b8103e3931f4ac330

Software
--------
Python 3.12.13; NumPy 2.3.5; pandas 2.2.3; scikit-learn 1.8.0;
CatBoost 1.2.10.

Integrity
---------
- CatBoost uncertainty predictions: 42,291 / 42,291 expected rows
- Duplicate prediction keys: 0
- Maximum archived point-prediction difference: 3.553e-15
- CatBoost grouped SHAP local rows: 46,200
- Maximum SHAP additivity residual: 4.974e-14
- CatBoost hierarchical local rows: 483,000
- Maximum child-to-parent telescoping difference: 3.908e-14
- Grouped and chronological DOI boundary overlap: 0
- DOI-cluster bootstrap replicates: 1,000

The raw public database CSV is not duplicated in this package. Supply the
snapshot matching the SHA-256 hash above when rerunning the scripts.
