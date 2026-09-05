#!/usr/bin/env python3
"""Independent verification of mixed Pb-Sn warning-policy outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


TARGETS = ["PCE", "Voc", "Jsc", "FF"]
CHRONO_SCHEME = "Chronological >2018"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", required=True, type=Path)
    parser.add_argument("--source-predictions", required=True, type=Path)
    parser.add_argument("--support", required=True, type=Path)
    parser.add_argument("--calibration", required=True, type=Path)
    return parser.parse_args()


def metric_bundle(frame: pd.DataFrame, publication_balanced: bool) -> dict[str, float]:
    if publication_balanced:
        counts = frame["doi_norm"].value_counts()
        weights = 1.0 / frame["doi_norm"].map(counts).to_numpy(float)
    else:
        weights = np.ones(len(frame), dtype=float)
    y = frame["y_true"].to_numpy(float)
    p = frame["y_pred"].to_numpy(float)
    residual = p - y
    mean_y = np.average(y, weights=weights)
    denominator = np.sum(weights * (y - mean_y) ** 2)
    bias = np.average(residual, weights=weights)
    return {
        "mean_measured": float(mean_y),
        "mean_predicted": float(np.average(p, weights=weights)),
        "MAE": float(np.average(np.abs(residual), weights=weights)),
        "RMSE": float(np.sqrt(np.average(residual**2, weights=weights))),
        "bias": float(bias),
        "absolute_bias": float(abs(bias)),
        "R2": float(1.0 - np.sum(weights * residual**2) / denominator) if denominator > 0 else np.nan,
        "coverage_95": float(np.average(frame["interval_95_covered"].astype(float), weights=weights)),
        "mean_interval_95_half_width": float(np.average(frame["interval_95_half_width"], weights=weights)),
        "large_error_fraction": float(np.average(frame["large_error"].astype(float), weights=weights)),
        "mean_feature_OOD_percentile": float(np.average(frame["feature_ood_percentile"], weights=weights)),
        "mean_model_OOD_percentile": float(np.average(frame["model_ood_percentile"], weights=weights)),
        "formula_unseen_fraction": float(np.average(frame["formula_unseen_historical"].astype(float), weights=weights)),
        "mean_uncertainty_multiplier": float(np.average(frame["uncertainty_multiplier"], weights=weights)),
    }


def main() -> None:
    args = parse_args()
    root = args.results_dir
    checks: list[dict[str, object]] = []

    def check(name: str, passed: bool, detail: object) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    required = [
        "mixed_pb_sn_warning_assignments.csv.gz",
        "mixed_pb_sn_warning_policy.csv",
        "mixed_pb_sn_warning_metrics.csv",
        "mixed_pb_sn_warning_comparisons.csv",
        "mixed_pb_sn_warning_trigger_summary.csv",
        "mixed_pb_sn_warning_sensitivity.csv",
        "Figure10_Mixed_PbSn_warning_policy.png",
        "Figure10_Mixed_PbSn_warning_policy.pdf",
        "Figure10_Mixed_PbSn_warning_policy.svg",
        "mixed_pb_sn_warning_run_manifest.json",
    ]
    missing = [name for name in required if not (root / name).exists()]
    check("required outputs exist", not missing, missing)

    assignments = pd.read_csv(root / "mixed_pb_sn_warning_assignments.csv.gz", low_memory=False)
    metrics = pd.read_csv(root / "mixed_pb_sn_warning_metrics.csv")
    comparisons = pd.read_csv(root / "mixed_pb_sn_warning_comparisons.csv")
    triggers = pd.read_csv(root / "mixed_pb_sn_warning_trigger_summary.csv")
    sensitivity = pd.read_csv(root / "mixed_pb_sn_warning_sensitivity.csv")
    manifest = json.loads((root / "mixed_pb_sn_warning_run_manifest.json").read_text(encoding="utf-8"))

    check("assignment row count", len(assignments) == 392, len(assignments))
    check("record and DOI support", assignments["Ref_ID"].nunique() == 98 and assignments["doi_norm"].nunique() == 25,
          {"records": assignments["Ref_ID"].nunique(), "DOI": assignments["doi_norm"].nunique()})
    duplicate_count = int(assignments.duplicated(["Ref_ID", "target"]).sum())
    check("prediction key uniqueness", duplicate_count == 0, duplicate_count)
    check("target set", set(assignments["target"]) == set(TARGETS), sorted(assignments["target"].unique()))
    check("mixed Pb-Sn subset only", assignments["b_site_pattern"].eq("Pb+Sn").all(), assignments["b_site_pattern"].value_counts().to_dict())

    source = pd.read_csv(args.source_predictions, low_memory=False)
    source = source.loc[
        source["scheme"].eq(CHRONO_SCHEME) & source["b_site_pattern"].eq("Pb+Sn"),
        ["Ref_ID", "target", "y_true", "y_pred", "feature_ood_percentile", "model_ood_percentile", "interval_95_half_width"],
    ]
    joined = assignments.merge(source, on=["Ref_ID", "target"], suffixes=("_new", "_source"), validate="one_to_one")
    frozen_difference = max(
        float(np.max(np.abs(joined[f"{column}_new"] - joined[f"{column}_source"])))
        for column in ["y_true", "y_pred", "feature_ood_percentile", "model_ood_percentile", "interval_95_half_width"]
    )
    check("frozen prediction and diagnostic identity", frozen_difference <= 1e-12, frozen_difference)

    support = pd.read_csv(args.support)
    mixed = support.loc[(support["domain_type"] == "B-site pattern") & (support["domain"] == "Pb+Sn")]
    mixed_historical_doi = int(mixed.iloc[0]["historical_DOI"])
    check("mixed historical support", mixed_historical_doi == 57, mixed_historical_doi)
    cell_map = support.loc[support["domain_type"] == "A x B domain"].set_index("domain")["historical_DOI"]
    mapped = assignments["composition_domain"].map(cell_map).to_numpy(int)
    check("composition-cell support mapping", np.array_equal(mapped, assignments["composition_cell_historical_DOI"].to_numpy(int)),
          int(np.max(np.abs(mapped - assignments["composition_cell_historical_DOI"].to_numpy(int)))))

    calibration = pd.read_csv(args.calibration)
    calibration = calibration.loc[(calibration["scheme"] == CHRONO_SCHEME) & (calibration["nominal_coverage"] == 0.95)]
    floors = dict(zip(calibration["target"], calibration["sigma_floor"] * calibration["normalized_residual_quantile"]))
    recalculated_floor = assignments["target"].map(floors).to_numpy(float)
    recalculated_multiplier = assignments["interval_95_half_width"].to_numpy(float) / recalculated_floor
    check("calibration-floor identity", np.max(np.abs(recalculated_floor - assignments["calibrated_floor_95_half_width"])) <= 1e-12,
          float(np.max(np.abs(recalculated_floor - assignments["calibrated_floor_95_half_width"]))))
    check("uncertainty-multiplier identity", np.max(np.abs(recalculated_multiplier - assignments["uncertainty_multiplier"])) <= 1e-12,
          float(np.max(np.abs(recalculated_multiplier - assignments["uncertainty_multiplier"]))))

    red_recomputed = (
        assignments["formula_unseen_historical"].astype(bool)
        | assignments["feature_ood_percentile"].ge(0.95)
        | assignments["model_ood_percentile"].ge(0.95)
        | assignments["uncertainty_multiplier"].ge(2.5)
        | assignments["composition_cell_historical_DOI"].lt(5)
    )
    expected_tier = np.where(red_recomputed, "Red", "Amber")
    check("independent tier recomputation", np.array_equal(expected_tier, assignments["warning_tier"]),
          int(np.sum(expected_tier != assignments["warning_tier"])))
    check("Green gate closed", not assignments["warning_tier"].eq("Green").any(), assignments["warning_tier"].value_counts().to_dict())
    expected_red = {"PCE": 46, "Voc": 72, "Jsc": 91, "FF": 45}
    observed_red = assignments.loc[assignments["warning_tier"] == "Red"].groupby("target").size().to_dict()
    check("target-specific Red counts", observed_red == expected_red, observed_red)

    max_metric_difference = 0.0
    metric_names = [
        "mean_measured", "mean_predicted", "MAE", "RMSE", "bias", "absolute_bias", "R2", "coverage_95",
        "mean_interval_95_half_width", "large_error_fraction", "mean_feature_OOD_percentile",
        "mean_model_OOD_percentile", "formula_unseen_fraction", "mean_uncertainty_multiplier",
    ]
    for row in metrics.itertuples():
        block = assignments.loc[assignments["target"].eq(row.target)]
        if row.warning_tier != "All":
            block = block.loc[block["warning_tier"].eq(row.warning_tier)]
        recalculated = metric_bundle(block, row.evaluation == "Publication-balanced")
        for metric in metric_names:
            left = recalculated[metric]
            right = float(getattr(row, metric))
            if np.isfinite(left) and np.isfinite(right):
                max_metric_difference = max(max_metric_difference, abs(left - right))
    check("metric recomputation", max_metric_difference <= 1e-12, max_metric_difference)

    ci_order = True
    for metric in metric_names:
        ci_order &= (metrics[f"{metric}_CI_low"] <= metrics[f"{metric}_CI_high"]).all()
    check("metric CI ordering", ci_order, "all low <= high")
    comparison_ci_order = True
    for prefix in ["delta_MAE", "delta_RMSE", "delta_absolute_bias", "delta_large_error_fraction", "delta_coverage_95", "delta_mean_interval_95_half_width", "MAE_ratio"]:
        comparison_ci_order &= (comparisons[f"{prefix}_CI_low"] <= comparisons[f"{prefix}_CI_high"]).all()
    check("comparison CI ordering", comparison_ci_order, "all low <= high")

    publication = comparisons.loc[comparisons["evaluation"] == "Publication-balanced"].set_index("target")
    check("Jsc and FF Red MAE enrichment", publication.loc["Jsc", "MAE_ratio_CI_low"] > 1 and publication.loc["FF", "MAE_ratio_CI_low"] > 1,
          publication[["MAE_ratio_Red_over_Amber", "MAE_ratio_CI_low", "MAE_ratio_CI_high"]].to_dict("index"))
    check("PCE comparison classified descriptive", not bool(publication.loc["PCE", "inferential_eligible"]),
          {"Amber_DOI": int(publication.loc["PCE", "Amber_DOI"]), "Red_DOI": int(publication.loc["PCE", "Red_DOI"])})

    unseen = triggers.loc[triggers["trigger"] == "Exact formula unseen historically"].set_index("target")["records"].to_dict()
    check("exact-formula novelty count", unseen == {target: 34 for target in TARGETS}, unseen)
    model_critical = triggers.loc[triggers["trigger"] == "Model-support OOD >=P95", "records"]
    check("model-support P95 non-trigger in this cohort", int(model_critical.sum()) == 0, int(model_critical.sum()))
    selected = sensitivity.loc[sensitivity["selected_policy"]]
    check("one locked primary sensitivity row per target", len(selected) == 4 and set(selected["target"]) == set(TARGETS), selected[["variant", "target"]].to_dict("records"))
    check("future outcomes excluded from threshold setting", manifest["design"]["thresholds_locked_without_future_outcomes"] is True and manifest["design"]["future_outcomes_role"] == "evaluation only",
          manifest["design"])
    check("bootstrap replicate count", manifest["design"]["bootstrap_replicates"] == 1000, manifest["design"]["bootstrap_replicates"])

    passed = sum(bool(item["passed"]) for item in checks)
    report = {
        "status": "PASS" if passed == len(checks) else "FAIL",
        "checks_passed": passed,
        "checks_total": len(checks),
        "checks": checks,
    }
    (root / "independent_mixed_pb_sn_warning_verification.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
