#!/usr/bin/env python3
"""Deterministic integrity checks for DOI-balanced PSC weighting outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from doi_balanced_weighting import (  # noqa: E402
    CHRONO_SCHEME,
    DEVICE_LENS,
    FULL,
    GROUPED_SCHEME,
    LENS_ORDER,
    PUBLICATION_LENS,
    TEMPERED,
    UNWEIGHTED,
    WEIGHT_ORDER,
    error_cluster_stats,
    error_metrics_from_totals,
    evaluation_weights,
    high_efficiency_effects_from_totals,
    metric_values,
    paired_cluster_stats,
    paired_effects_from_totals,
    paired_high_efficiency_stats,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", required=True, type=Path)
    parser.add_argument("--baseline-results-dir", required=True, type=Path)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    args = parse_args()
    results = args.results_dir
    manifest = json.loads((results / "weighting_run_manifest.json").read_text())
    predictions = pd.read_csv(results / "publication_weighting_predictions.csv.gz")
    metrics = pd.read_csv(results / "publication_weighting_metrics.csv")
    paired = pd.read_csv(results / "publication_weighting_paired_comparison.csv")
    calibration = pd.read_csv(
        results / "chronological_PCE_weighting_calibration.csv"
    )
    high = pd.read_csv(
        results / "chronological_PCE_high_efficiency_paired_comparison.csv"
    )
    diagnostics = pd.read_csv(results / "training_weight_diagnostics.csv")
    split = pd.read_csv(args.baseline_results_dir / "split_manifest.csv")
    frozen = pd.read_csv(args.baseline_results_dir / "baseline_predictions.csv.gz")
    frozen = frozen.loc[frozen["scheme"].isin([GROUPED_SCHEME, CHRONO_SCHEME])]

    checks: dict[str, object] = {}
    require(len(predictions) == manifest["prediction_rows"], "Prediction row mismatch")
    checks["prediction_rows"] = int(len(predictions))

    prediction_key = [
        "Ref_ID",
        "scheme",
        "training_weighting",
        "model",
        "target",
    ]
    duplicate_count = int(predictions.duplicated(prediction_key).sum())
    require(duplicate_count == 0, "Duplicate held-out prediction keys")
    checks["duplicate_prediction_keys"] = duplicate_count

    expected_records = {GROUPED_SCHEME: 33175, CHRONO_SCHEME: 9116}
    coverage = predictions.groupby(
        ["scheme", "training_weighting", "model", "target"], sort=False
    )["Ref_ID"].nunique()
    for (scheme, _weighting, _model, _target), count in coverage.items():
        require(int(count) == expected_records[scheme], "Incomplete prediction coverage")
    checks["coverage_groups_checked"] = int(len(coverage))

    target_nunique = predictions.groupby(
        ["Ref_ID", "scheme", "target"], sort=False
    )["y_true"].nunique(dropna=False)
    require(int(target_nunique.max()) == 1, "Target values differ across predictions")
    checks["target_consistency_max_unique_values"] = int(target_nunique.max())

    frozen = frozen.assign(training_weighting=UNWEIGHTED)
    frozen_keys = ["Ref_ID", "scheme", "fold", "model", "target"]
    current_unweighted = predictions.loc[
        predictions["training_weighting"].eq(UNWEIGHTED)
    ]
    compared = frozen.merge(
        current_unweighted,
        on=frozen_keys,
        how="inner",
        suffixes=("_frozen", "_current"),
        validate="one_to_one",
    )
    require(len(compared) == len(frozen) == len(current_unweighted), "Frozen baseline mismatch")
    require(
        compared["doi_norm_frozen"].astype("string").eq(
            compared["doi_norm_current"].astype("string")
        ).all(),
        "Frozen DOI labels changed",
    )
    frozen_y_error = float(
        np.max(np.abs(compared["y_true_frozen"] - compared["y_true_current"]))
    )
    frozen_pred_error = float(
        np.max(np.abs(compared["y_pred_frozen"] - compared["y_pred_current"]))
    )
    require(frozen_y_error < 1e-12 and frozen_pred_error < 1e-12, "Frozen values changed")
    checks["frozen_unweighted_rows"] = int(len(compared))
    checks["frozen_y_true_max_abs_difference"] = frozen_y_error
    checks["frozen_prediction_max_abs_difference"] = frozen_pred_error

    grouped_fold_counts = split.groupby("doi_norm")["grouped_fold"].nunique()
    require(int(grouped_fold_counts.max()) == 1, "DOI fragmented across grouped folds")
    historical_doi = set(
        split.loc[
            split["chronological_role"].eq("train_through_2018"), "doi_norm"
        ]
    )
    future_doi = set(
        split.loc[
            split["chronological_role"].eq("test_2019_onward"), "doi_norm"
        ]
    )
    chrono_overlap = len(historical_doi.intersection(future_doi))
    require(chrono_overlap == 0, "Chronological DOI overlap")
    checks["grouped_max_folds_per_DOI"] = int(grouped_fold_counts.max())
    checks["chronological_train_test_DOI_overlap"] = int(chrono_overlap)

    metric_error = 0.0
    for row in metrics.itertuples(index=False):
        frame = predictions.loc[
            predictions["scheme"].eq(row.scheme)
            & predictions["training_weighting"].eq(row.training_weighting)
            & predictions["model"].eq(row.model)
            & predictions["target"].eq(row.target)
        ]
        recalculated = metric_values(
            frame["y_true"].to_numpy(),
            frame["y_pred"].to_numpy(),
            evaluation_weights(frame, row.evaluation_lens),
        )
        for metric in ["R2", "MAE", "RMSE"]:
            metric_error = max(metric_error, abs(recalculated[metric] - getattr(row, metric)))
    require(metric_error < 1e-12, "Metric recalculation mismatch")
    checks["metric_recalculation_max_abs_difference"] = float(metric_error)

    paired_error = 0.0
    for row in paired.itertuples(index=False):
        baseline = predictions.loc[
            predictions["scheme"].eq(row.scheme)
            & predictions["training_weighting"].eq(UNWEIGHTED)
            & predictions["model"].eq(row.model)
            & predictions["target"].eq(row.target)
        ]
        candidate = predictions.loc[
            predictions["scheme"].eq(row.scheme)
            & predictions["training_weighting"].eq(row.training_weighting)
            & predictions["model"].eq(row.model)
            & predictions["target"].eq(row.target)
        ]
        effects = paired_effects_from_totals(
            paired_cluster_stats(baseline, candidate, row.evaluation_lens).sum(axis=0)
        )
        stored = [
            row.delta_R2_weighted_minus_unweighted,
            row.MAE_change_fraction_weighted_vs_unweighted,
            row.RMSE_change_fraction_weighted_vs_unweighted,
        ]
        paired_error = max(
            paired_error,
            float(np.max(np.abs(np.asarray(effects) - np.asarray(stored)))),
        )
    require(paired_error < 1e-12, "Paired comparison mismatch")
    checks["paired_effect_recalculation_max_abs_difference"] = float(paired_error)

    bin_edges = [-np.inf, 5, 10, 15, 20, np.inf]
    bin_labels = ["0–5", "5–10", "10–15", "15–20", "≥20"]
    calibration_source = predictions.loc[
        predictions["scheme"].eq(CHRONO_SCHEME)
        & predictions["target"].eq("PCE")
    ].copy()
    calibration_source["measured_PCE_bin"] = pd.cut(
        calibration_source["y_true"],
        bins=bin_edges,
        labels=bin_labels,
        right=False,
    ).astype("string")
    calibration_error = 0.0
    for row in calibration.itertuples(index=False):
        frame = calibration_source.loc[
            calibration_source["training_weighting"].eq(row.training_weighting)
            & calibration_source["model"].eq(row.model)
            & calibration_source["measured_PCE_bin"].eq(row.measured_PCE_bin)
        ]
        values = error_metrics_from_totals(
            error_cluster_stats(frame, row.evaluation_lens).sum(axis=0)
        )
        stored = [
            row.measured_PCE_mean,
            row.predicted_PCE_mean,
            row.mean_bias_predicted_minus_measured,
            row.MAE,
            row.RMSE,
        ]
        calibration_error = max(
            calibration_error,
            float(np.max(np.abs(np.asarray(values) - np.asarray(stored)))),
        )
    require(calibration_error < 1e-12, "Calibration recalculation mismatch")
    checks["calibration_recalculation_max_abs_difference"] = float(calibration_error)

    high_source = calibration_source.loc[calibration_source["y_true"].ge(20.0)]
    high_unique = high_source[["Ref_ID", "doi_norm"]].drop_duplicates()
    require(len(high_unique) == 529, "High-PCE record count must include exact 20% values")
    require(high_unique["doi_norm"].nunique() == 260, "High-PCE DOI count mismatch")
    high_error = 0.0
    for row in high.itertuples(index=False):
        baseline = high_source.loc[
            high_source["training_weighting"].eq(UNWEIGHTED)
            & high_source["model"].eq(row.model)
        ]
        candidate = high_source.loc[
            high_source["training_weighting"].eq(row.training_weighting)
            & high_source["model"].eq(row.model)
        ]
        values = high_efficiency_effects_from_totals(
            paired_high_efficiency_stats(
                baseline, candidate, row.evaluation_lens
            ).sum(axis=0)
        )
        stored = [
            row.unweighted_mean_bias,
            row.weighted_mean_bias,
            row.delta_bias_weighted_minus_unweighted,
            row.unweighted_MAE,
            row.weighted_MAE,
            row.MAE_change_fraction_weighted_vs_unweighted,
            row.unweighted_RMSE,
            row.weighted_RMSE,
            row.RMSE_change_fraction_weighted_vs_unweighted,
        ]
        high_error = max(
            high_error,
            float(np.max(np.abs(np.asarray(values) - np.asarray(stored)))),
        )
    require(high_error < 1e-12, "High-efficiency comparison mismatch")
    checks["high_PCE_records"] = int(len(high_unique))
    checks["high_PCE_DOI_groups"] = int(high_unique["doi_norm"].nunique())
    checks["high_efficiency_recalculation_max_abs_difference"] = float(high_error)

    warnings_count = int(diagnostics["warnings"].fillna("").ne("").sum())
    require(warnings_count == 0, "Model fitting warnings present")
    rf_diag = diagnostics.loc[
        diagnostics["model"].eq("preprocessor_and_random_forest")
    ]
    mean_weight_error = float(np.max(np.abs(rf_diag["sample_weight_mean"] - 1.0)))
    require(mean_weight_error < 1e-12, "Training weights not normalized")
    require(float(rf_diag["sample_weight_min"].min()) > 0, "Nonpositive weights")
    full_cv = float(
        rf_diag.loc[
            rf_diag["training_weighting"].eq(FULL), "DOI_total_weight_CV"
        ].max()
    )
    require(full_cv < 1e-12, "Full inverse weighting is not equal-publication")
    checks["fit_warning_rows"] = warnings_count
    checks["sample_weight_mean_max_abs_difference_from_one"] = mean_weight_error
    checks["full_inverse_DOI_total_weight_CV_max"] = full_cv

    required_files = [
        "publication_weighting_metrics.csv",
        "publication_weighting_paired_comparison.csv",
        "weighting_performance_by_DOI_size.csv",
        "chronological_PCE_weighting_calibration.csv",
        "chronological_PCE_high_efficiency_paired_comparison.csv",
        "DOI_size_distribution.csv",
        "publication_weighting_predictions.csv.gz",
        "training_weight_diagnostics.csv",
        "Figure4_DOI_balanced_weighting.png",
        "Figure4_DOI_balanced_weighting.pdf",
        "Figure4_DOI_balanced_weighting.svg",
        "weighting_run_manifest.json",
    ]
    checks["output_files"] = {
        name: {
            "bytes": int((results / name).stat().st_size),
            "sha256": file_sha256(results / name),
        }
        for name in required_files
    }
    checks["status"] = "passed"
    checks["weighting_conditions"] = WEIGHT_ORDER
    checks["evaluation_lenses"] = LENS_ORDER
    (results / "verification_report.json").write_text(
        json.dumps(checks, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "passed", **{k: checks[k] for k in [
        "prediction_rows",
        "duplicate_prediction_keys",
        "frozen_unweighted_rows",
        "metric_recalculation_max_abs_difference",
        "fit_warning_rows",
        "high_PCE_records",
    ]}}))


if __name__ == "__main__":
    main()
