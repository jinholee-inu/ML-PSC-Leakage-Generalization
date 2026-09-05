#!/usr/bin/env python3
"""Independent integrity checks for subgroup calibration / MoE outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DOMAINS = {"Sn-only": "Sn (no Pb)", "Mixed Pb-Sn": "Pb+Sn"}
TARGETS = ["PCE", "Voc", "Jsc", "FF"]
METHODS = [
    "Frozen global",
    "Subgroup calibrator",
    "Domain expert",
    "Convex mixture",
    "Development-selected policy",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", required=True, type=Path)
    parser.add_argument("--assignments", required=True, type=Path)
    parser.add_argument("--weighting-predictions", required=True, type=Path)
    return parser.parse_args()


def publication_metric(frame: pd.DataFrame) -> dict[str, float]:
    counts = frame["doi_norm"].value_counts()
    weights = frame["doi_norm"].map(counts).to_numpy(float) ** -1.0
    y = frame["y_true"].to_numpy(float)
    p = frame["y_pred"].to_numpy(float)
    mean_y = np.average(y, weights=weights)
    residual = p - y
    return {
        "R2": float(1.0 - np.sum(weights * residual**2) / np.sum(weights * (y - mean_y) ** 2)),
        "MAE": float(np.average(np.abs(residual), weights=weights)),
        "RMSE": float(np.sqrt(np.average(residual**2, weights=weights))),
        "bias": float(np.average(residual, weights=weights)),
    }


def device_metric(frame: pd.DataFrame) -> dict[str, float]:
    y = frame["y_true"].to_numpy(float)
    p = frame["y_pred"].to_numpy(float)
    residual = p - y
    return {
        "R2": float(1.0 - np.sum(residual**2) / np.sum((y - y.mean()) ** 2)),
        "MAE": float(np.mean(np.abs(residual))),
        "RMSE": float(np.sqrt(np.mean(residual**2))),
        "bias": float(np.mean(residual)),
    }


def main() -> None:
    args = parse_args()
    root = args.results_dir
    checks: list[dict[str, object]] = []

    def check(name: str, passed: bool, detail: object) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    required = [
        "subgroup_support.csv",
        "subgroup_development_selection.csv",
        "subgroup_calibrator_selection.csv",
        "subgroup_mixture_weight_selection.csv",
        "subgroup_policy_selection.csv",
        "subgroup_future_predictions.csv.gz",
        "subgroup_future_metrics.csv",
        "subgroup_paired_comparisons.csv",
        "subgroup_PCE_calibration_bins.csv",
        "subgroup_PCE_upper_tail.csv",
        "subgroup_robustness_strata.csv",
        "Figure9_subgroup_calibration_moe.png",
        "Figure9_subgroup_calibration_moe.pdf",
        "Figure9_subgroup_calibration_moe.svg",
        "subgroup_calibration_moe_run_manifest.json",
    ]
    missing = [name for name in required if not (root / name).exists()]
    check("required outputs exist", not missing, missing)

    predictions = pd.read_csv(root / "subgroup_future_predictions.csv.gz", low_memory=False)
    metrics = pd.read_csv(root / "subgroup_future_metrics.csv")
    paired = pd.read_csv(root / "subgroup_paired_comparisons.csv")
    support = pd.read_csv(root / "subgroup_support.csv")
    calibrators = pd.read_csv(root / "subgroup_calibrator_selection.csv")
    mixtures = pd.read_csv(root / "subgroup_mixture_weight_selection.csv")
    policies = pd.read_csv(root / "subgroup_policy_selection.csv")
    upper = pd.read_csv(root / "subgroup_PCE_upper_tail.csv")
    manifest = json.loads((root / "subgroup_calibration_moe_run_manifest.json").read_text(encoding="utf-8"))

    check("prediction row count", len(predictions) == 5060, len(predictions))
    duplicate_count = int(predictions.duplicated(["Ref_ID", "target", "method"]).sum())
    check("prediction key uniqueness", duplicate_count == 0, duplicate_count)
    check("target set", set(predictions["target"]) == set(TARGETS), sorted(predictions["target"].unique()))
    check("method set", set(predictions["method"]) == set(METHODS), sorted(predictions["method"].unique()))
    check("domain set", set(predictions["domain"]) == set(DOMAINS), sorted(predictions["domain"].unique()))

    expected_support = {
        ("Sn-only", "Historical <=2018"): (389, 83),
        ("Sn-only", "Future 2019-2021"): (155, 30),
        ("Mixed Pb-Sn", "Historical <=2018"): (274, 57),
        ("Mixed Pb-Sn", "Future 2019-2021"): (98, 25),
    }
    observed_support = {
        (row.domain, row.period): (int(row.records), int(row.DOI_groups))
        for row in support.itertuples()
    }
    check(
        "subgroup support counts",
        observed_support == expected_support,
        {f"{key[0]} | {key[1]}": value for key, value in observed_support.items()},
    )

    sets_match = True
    for (domain, target), block in predictions.groupby(["domain", "target"]):
        reference = set(block.loc[block["method"].eq("Frozen global"), "Ref_ID"])
        for method in METHODS:
            sets_match &= set(block.loc[block["method"].eq(method), "Ref_ID"]) == reference
    check("paired record sets", sets_match, "all methods share exact future Ref_ID sets")

    archive = pd.read_csv(args.weighting_predictions, low_memory=False)
    archive = archive.loc[
        archive["scheme"].eq("Chronological >2018")
        & archive["training_weighting"].eq("Full 1/n_DOI")
        & archive["model"].eq("Random Forest"),
        ["Ref_ID", "target", "y_true", "y_pred"],
    ]
    frozen = predictions.loc[predictions["method"].eq("Frozen global")]
    comparison = frozen.merge(archive, on=["Ref_ID", "target"], suffixes=("_new", "_archive"), validate="one_to_one")
    pred_diff = float(np.max(np.abs(comparison["y_pred_new"] - comparison["y_pred_archive"])))
    truth_diff = float(np.max(np.abs(comparison["y_true_new"] - comparison["y_true_archive"])))
    check("frozen prediction identity", pred_diff <= 1e-12, pred_diff)
    check("outcome identity", truth_diff <= 1e-12, truth_diff)

    assignments = pd.read_csv(args.assignments, usecols=["doi_norm", "publication_year", "b_site_pattern"])
    overlap = 0
    for pattern in DOMAINS.values():
        block = assignments.loc[assignments["b_site_pattern"].eq(pattern)]
        historical = set(block.loc[block["publication_year"] <= 2018, "doi_norm"])
        future = set(block.loc[block["publication_year"] > 2018, "doi_norm"])
        overlap += len(historical & future)
    check("historical-future DOI separation", overlap == 0, overlap)

    selected_calibrators = calibrators.loc[calibrators["selected"]]
    check(
        "one-SE calibrator selection",
        len(selected_calibrators) == 8 and selected_calibrators["candidate"].eq("Identity").all(),
        selected_calibrators[["domain", "target", "candidate"]].to_dict("records"),
    )
    selected_mixtures = mixtures.loc[mixtures["selected"]]
    check(
        "one-SE mixture shrinkage",
        len(selected_mixtures) == 8 and np.allclose(selected_mixtures["alpha"], 0.0),
        selected_mixtures[["domain", "target", "alpha"]].to_dict("records"),
    )
    selected_policies = policies.loc[policies["selected"]]
    check(
        "development policy selection",
        len(selected_policies) == 8 and selected_policies["candidate"].eq("Frozen global").all(),
        selected_policies[["domain", "target", "candidate"]].to_dict("records"),
    )

    max_metric_difference = 0.0
    for row in metrics.itertuples():
        frame = predictions.loc[
            predictions["domain"].eq(row.domain)
            & predictions["target"].eq(row.target)
            & predictions["method"].eq(row.method)
        ]
        recalculated = publication_metric(frame) if row.evaluation == "Publication-balanced" else device_metric(frame)
        max_metric_difference = max(
            max_metric_difference,
            *[abs(recalculated[name] - float(getattr(row, name))) for name in ["R2", "MAE", "RMSE", "bias"]],
        )
    check("metric recomputation", max_metric_difference <= 1e-12, max_metric_difference)

    ci_ok = True
    for base in ["R2", "MAE", "RMSE", "bias"]:
        ci_ok &= (metrics[f"{base}_CI_low"] <= metrics[f"{base}_CI_high"]).all()
    check("metric confidence interval ordering", ci_ok, "all low <= high")
    paired_ci_ok = True
    for base in ["delta_R2", "delta_MAE", "MAE_change_percent", "delta_RMSE", "delta_absolute_bias"]:
        paired_ci_ok &= (paired[f"{base}_CI_low"] <= paired[f"{base}_CI_high"]).all()
    check("paired confidence interval ordering", paired_ci_ok, "all low <= high")

    pce_expert = paired.loc[
        paired["target"].eq("PCE")
        & paired["method"].eq("Domain expert")
        & paired["evaluation"].eq("Publication-balanced")
    ]
    check(
        "domain experts worsen future PCE MAE",
        len(pce_expert) == 2 and (pce_expert["MAE_change_percent_CI_low"] > 0).all(),
        pce_expert[["domain", "MAE_change_percent", "MAE_change_percent_CI_low", "MAE_change_percent_CI_high"]].to_dict("records"),
    )

    conventional = upper.loc[
        upper["subset"].eq("Conventional PCE >=20%") & upper["method"].eq("Frozen global")
    ]
    counts = dict(zip(conventional["domain"], conventional["DOI_groups"]))
    check("PCE >=20 support classified descriptive", counts == {"Sn-only": 0, "Mixed Pb-Sn": 2}, counts)
    check("bootstrap replicate count", manifest.get("bootstrap_replicates") == 1000, manifest.get("bootstrap_replicates"))
    check(
        "future outcomes excluded from tuning",
        manifest.get("temporal_design", {}).get("future_outcomes_used_for_tuning") is False,
        manifest.get("temporal_design"),
    )
    check("manifest prediction integrity", manifest.get("prediction_rows") == 5060 and manifest.get("duplicate_prediction_keys") == 0, {"rows": manifest.get("prediction_rows"), "duplicates": manifest.get("duplicate_prediction_keys")})

    passed = sum(bool(item["passed"]) for item in checks)
    report = {
        "status": "PASS" if passed == len(checks) else "FAIL",
        "checks_passed": passed,
        "checks_total": len(checks),
        "checks": checks,
    }
    (root / "independent_subgroup_calibration_moe_verification.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
