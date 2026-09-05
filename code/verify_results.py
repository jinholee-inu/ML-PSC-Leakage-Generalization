#!/usr/bin/env python3
"""Integrity checks for the PSC baseline-validation deliverables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = args.results_dir
    predictions = pd.read_csv(results / "baseline_predictions.csv.gz")
    summary = pd.read_csv(results / "baseline_metrics_summary.csv")
    split = pd.read_csv(results / "split_manifest.csv")
    diagnostics = pd.read_csv(results / "fit_diagnostics.csv")
    gaps = pd.read_csv(results / "baseline_generalization_gap.csv")
    inflation = pd.read_csv(results / "publication_leakage_inflation.csv")
    leakage = json.loads((results / "row_random_leakage_summary.json").read_text())

    expected_rows = 33175 * 3 * 4 * 2 + 9116 * 3 * 4
    checks: dict[str, object] = {
        "prediction_rows_expected": expected_rows,
        "prediction_rows_observed": int(len(predictions)),
        "prediction_key_duplicates": int(
            predictions.duplicated(
                ["Ref_ID", "scheme", "fold", "model", "target"]
            ).sum()
        ),
        "grouped_records_per_model_target": sorted(
            predictions.loc[predictions["scheme"].eq("DOI-grouped 5-fold")]
            .groupby(["model", "target"])
            .size()
            .unique()
            .tolist()
        ),
        "row_random_records_per_model_target": sorted(
            predictions.loc[predictions["scheme"].eq("Row-wise random 5-fold")]
            .groupby(["model", "target"])
            .size()
            .unique()
            .tolist()
        ),
        "chronological_records_per_model_target": sorted(
            predictions.loc[predictions["scheme"].eq("Chronological >2018")]
            .groupby(["model", "target"])
            .size()
            .unique()
            .tolist()
        ),
        "normalized_DOI_groups": int(split["doi_norm"].nunique()),
        "grouped_fold_count": int(split["grouped_fold"].nunique()),
        "row_random_fold_count": int(split["row_random_fold"].nunique()),
        "row_random_records_with_same_DOI_in_training": int(
            split["row_random_DOI_seen_in_train"].sum()
        ),
        "chronological_train_records": int(
            split["chronological_role"].eq("train_through_2018").sum()
        ),
        "chronological_test_records": int(
            split["chronological_role"].eq("test_2019_onward").sum()
        ),
    }

    doi_fold_count = split.groupby("doi_norm")["grouped_fold"].nunique()
    doi_random_fold_count = split.groupby("doi_norm")["row_random_fold"].nunique()
    doi_role_count = split.groupby("doi_norm")["chronological_role"].nunique()
    checks["DOI_groups_crossing_grouped_folds"] = int((doi_fold_count > 1).sum())
    checks["DOI_groups_crossing_row_random_folds"] = int(
        (doi_random_fold_count > 1).sum()
    )
    checks["DOI_groups_crossing_chronological_roles"] = int((doi_role_count > 1).sum())

    maximum_metric_difference = 0.0
    for _, row in summary.iterrows():
        group = predictions.loc[
            predictions["scheme"].eq(row["scheme"])
            & predictions["model"].eq(row["model"])
            & predictions["target"].eq(row["target"])
        ]
        recomputed = {
            "R2": r2_score(group["y_true"], group["y_pred"]),
            "MAE": mean_absolute_error(group["y_true"], group["y_pred"]),
            "RMSE": float(np.sqrt(mean_squared_error(group["y_true"], group["y_pred"]))),
        }
        for metric, value in recomputed.items():
            maximum_metric_difference = max(
                maximum_metric_difference, abs(float(row[metric]) - float(value))
            )
    checks["maximum_summary_metric_recompute_difference"] = maximum_metric_difference

    elastic = diagnostics.loc[diagnostics["model"].eq("Elastic Net")]
    checks["elastic_fit_count"] = int(len(elastic))
    checks["elastic_warning_count"] = int(elastic["warnings"].fillna("").ne("").sum())
    checks["elastic_max_dual_gap"] = float(elastic["dual_gap"].max())

    checks["gap_point_estimates_inside_CI"] = bool(
        (
            gaps["delta_R2_chrono_minus_grouped"].between(
                gaps["delta_R2_CI_low"], gaps["delta_R2_CI_high"]
            )
            & gaps["MAE_ratio_chrono_over_grouped"].between(
                gaps["MAE_ratio_CI_low"], gaps["MAE_ratio_CI_high"]
            )
        ).all()
    )
    checks["inflation_point_estimates_inside_CI"] = bool(
        (
            inflation["delta_R2_random_minus_grouped"].between(
                inflation["delta_R2_CI_low"], inflation["delta_R2_CI_high"]
            )
            & inflation["MAE_reduction_fraction"].between(
                inflation["MAE_reduction_CI_low"],
                inflation["MAE_reduction_CI_high"],
            )
            & inflation["RMSE_reduction_fraction"].between(
                inflation["RMSE_reduction_CI_low"],
                inflation["RMSE_reduction_CI_high"],
            )
        ).all()
    )
    checks["leakage_summary_matches_split"] = bool(
        leakage["DOI_groups_fragmented_across_folds"]
        == checks["DOI_groups_crossing_row_random_folds"]
        and leakage["records_with_same_DOI_in_training"]
        == checks["row_random_records_with_same_DOI_in_training"]
    )

    png = (results / "Figure3_baseline_validation.png").read_bytes()
    pdf = (results / "Figure3_baseline_validation.pdf").read_bytes()
    svg = (results / "Figure3_baseline_validation.svg").read_text(encoding="utf-8")
    checks["PNG_complete"] = png.startswith(b"\x89PNG\r\n\x1a\n") and b"IEND" in png[-32:]
    checks["PDF_complete"] = pdf.startswith(b"%PDF-") and pdf.rstrip().endswith(b"%%EOF")
    checks["SVG_complete"] = svg.lstrip().startswith("<?xml") and svg.rstrip().endswith("</svg>")

    failures = []
    expected = {
        "prediction_rows_observed": expected_rows,
        "prediction_key_duplicates": 0,
        "grouped_records_per_model_target": [33175],
        "row_random_records_per_model_target": [33175],
        "chronological_records_per_model_target": [9116],
        "normalized_DOI_groups": 6368,
        "grouped_fold_count": 5,
        "row_random_fold_count": 5,
        "row_random_records_with_same_DOI_in_training": 31870,
        "chronological_train_records": 24059,
        "chronological_test_records": 9116,
        "DOI_groups_crossing_grouped_folds": 0,
        "DOI_groups_crossing_row_random_folds": 5372,
        "DOI_groups_crossing_chronological_roles": 0,
        "elastic_fit_count": 44,
        "elastic_warning_count": 0,
        "gap_point_estimates_inside_CI": True,
        "inflation_point_estimates_inside_CI": True,
        "leakage_summary_matches_split": True,
        "PNG_complete": True,
        "PDF_complete": True,
        "SVG_complete": True,
    }
    for key, value in expected.items():
        if checks.get(key) != value:
            failures.append(f"{key}: expected {value!r}, observed {checks.get(key)!r}")
    if maximum_metric_difference > 1e-12:
        failures.append(
            "maximum_summary_metric_recompute_difference exceeds 1e-12: "
            f"{maximum_metric_difference}"
        )
    if checks["elastic_max_dual_gap"] > 1e-3:
        failures.append(
            f"elastic_max_dual_gap exceeds tolerance: {checks['elastic_max_dual_gap']}"
        )

    report = {
        "status": "passed" if not failures else "failed",
        "checks": checks,
        "failures": failures,
    }
    output_path = results / "verification_report.json"
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
