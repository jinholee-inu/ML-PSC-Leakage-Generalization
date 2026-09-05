#!/usr/bin/env python3
"""Independent checks for the PSC composition-domain reliability audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


GROUPED = "DOI-grouped 5-fold"
CHRONO = "Chronological >2018"
TARGETS = ["PCE", "Voc", "Jsc", "FF"]
OOD_THRESHOLD = 0.95


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", required=True, type=Path)
    parser.add_argument("--source-predictions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def add_status_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    for ion in ["FA", "MA", "Cs", "Pb", "Sn"]:
        frame[f"{ion}_status"] = np.where(
            frame[f"{ion}_present"].astype(bool),
            f"{ion} present",
            f"{ion} absent",
        )
    return frame


def expected_a(row: pd.Series) -> str:
    fa, ma, cs = bool(row.FA_present), bool(row.MA_present), bool(row.Cs_present)
    if fa and ma and cs:
        return "FA+MA+Cs"
    if fa and ma:
        return "FA+MA"
    if fa and cs:
        return "FA+Cs"
    if ma and cs:
        return "MA+Cs"
    if fa:
        return "FA (no MA/Cs)"
    if ma:
        return "MA (no FA/Cs)"
    if cs:
        return "Cs (no FA/MA)"
    return "Other/unknown"


def expected_b(row: pd.Series) -> str:
    pb, sn = bool(row.Pb_present), bool(row.Sn_present)
    if pb and sn:
        return "Pb+Sn"
    if pb:
        return "Pb (no Sn)"
    if sn:
        return "Sn (no Pb)"
    return "Other/unknown"


def publication_balanced_metrics(frame: pd.DataFrame) -> dict[str, float]:
    work = frame.assign(
        measured=frame["y_true"].astype(float),
        predicted=frame["y_pred"].astype(float),
        absolute_error=(frame["y_pred"] - frame["y_true"]).abs(),
        residual=frame["y_pred"] - frame["y_true"],
        squared_error=(frame["y_pred"] - frame["y_true"]) ** 2,
        measured_sq=frame["y_true"].astype(float) ** 2,
        feature_ood=frame["feature_ood_percentile"].astype(float),
        high_feature_ood=frame["feature_ood_percentile"].ge(OOD_THRESHOLD).astype(float),
        model_ood=frame["model_ood_percentile"].astype(float),
        high_model_ood=frame["model_ood_percentile"].ge(OOD_THRESHOLD).astype(float),
        formula_unseen=frame["formula_unseen_historical"].astype(float),
        coverage_90=frame["interval_90_covered"].astype(float),
        coverage_95=frame["interval_95_covered"].astype(float),
        width_90=2 * frame["interval_90_half_width"].astype(float),
        width_95=2 * frame["interval_95_half_width"].astype(float),
    )
    columns = [
        "measured",
        "predicted",
        "absolute_error",
        "residual",
        "squared_error",
        "measured_sq",
        "feature_ood",
        "high_feature_ood",
        "model_ood",
        "high_model_ood",
        "formula_unseen",
        "coverage_90",
        "coverage_95",
        "width_90",
        "width_95",
    ]
    doi = work.groupby("doi_norm", sort=False)[columns].mean()
    mean = doi.mean()
    variance = max(float(mean.measured_sq - mean.measured**2), 0.0)
    mse = float(mean.squared_error)
    sd = math.sqrt(variance)
    return {
        "mean_measured": float(mean.measured),
        "mean_predicted": float(mean.predicted),
        "MAE": float(mean.absolute_error),
        "bias": float(mean.residual),
        "RMSE": math.sqrt(mse),
        "R2": 1 - mse / variance if variance > 0 else np.nan,
        "target_SD": sd,
        "MAE_over_target_SD": float(mean.absolute_error / sd) if sd > 0 else np.nan,
        "mean_feature_OOD_percentile": float(mean.feature_ood),
        "high_feature_OOD_fraction": float(mean.high_feature_ood),
        "mean_model_OOD_percentile": float(mean.model_ood),
        "high_model_OOD_fraction": float(mean.high_model_ood),
        "formula_unseen_fraction": float(mean.formula_unseen),
        "coverage_90": float(mean.coverage_90),
        "coverage_95": float(mean.coverage_95),
        "interval_90_mean_width": float(mean.width_90),
        "interval_95_mean_width": float(mean.width_95),
    }


def main() -> None:
    args = parse_args()
    assignments = pd.read_csv(
        args.results_dir / "composition_domain_assignments.csv.gz", low_memory=False
    )
    assignments = add_status_columns(assignments)
    predictions = pd.read_csv(
        args.results_dir / "composition_domain_predictions.csv.gz", low_memory=False
    )
    predictions = add_status_columns(predictions)
    performance = pd.read_csv(
        args.results_dir / "composition_domain_performance.csv", low_memory=False
    )
    support = pd.read_csv(
        args.results_dir / "composition_domain_support.csv", low_memory=False
    )
    manifest = json.loads(
        (args.results_dir / "composition_domain_run_manifest.json").read_text()
    )

    expected_a_labels = assignments.apply(expected_a, axis=1)
    expected_b_labels = assignments.apply(expected_b, axis=1)
    label_a_ok = bool(expected_a_labels.eq(assignments["a_site_pattern"]).all())
    label_b_ok = bool(expected_b_labels.eq(assignments["b_site_pattern"]).all())
    domain_ok = bool(
        assignments["composition_domain"].eq(
            assignments["a_site_pattern"] + " / " + assignments["b_site_pattern"]
        ).all()
    )

    historical_formulas = set(
        assignments.loc[
            assignments["publication_year"].le(2018), "absorber_short_form_clean"
        ]
    )
    independently_unseen = ~assignments["absorber_short_form_clean"].isin(
        historical_formulas
    )
    formula_novelty_ok = bool(
        independently_unseen.eq(assignments["formula_unseen_historical"].astype(bool)).all()
    )

    source = pd.read_csv(args.source_predictions, low_memory=False)
    source = source.loc[
        source["publication_year"].ge(2019)
        & source["scheme"].isin([GROUPED, CHRONO])
    ]
    joined = predictions.merge(
        source[
            [
                "Ref_ID",
                "scheme",
                "target",
                "y_true",
                "y_pred",
                "feature_ood_percentile",
                "model_ood_percentile",
                "interval_90_half_width",
                "interval_95_half_width",
            ]
        ],
        on=["Ref_ID", "scheme", "target"],
        suffixes=("_result", "_source"),
        validate="one_to_one",
    )
    archive_diffs = {}
    for column in [
        "y_true",
        "y_pred",
        "feature_ood_percentile",
        "model_ood_percentile",
        "interval_90_half_width",
        "interval_95_half_width",
    ]:
        archive_diffs[column] = float(
            np.max(
                np.abs(
                    joined[f"{column}_result"].to_numpy(dtype=float)
                    - joined[f"{column}_source"].to_numpy(dtype=float)
                )
            )
        )

    domain_columns = {
        "A-site pattern": "a_site_pattern",
        "B-site pattern": "b_site_pattern",
        "A x B domain": "composition_domain",
        "FA presence": "FA_status",
        "MA presence": "MA_status",
        "Cs presence": "Cs_status",
        "Pb presence": "Pb_status",
        "Sn presence": "Sn_status",
    }
    metric_names = [
        "mean_measured",
        "mean_predicted",
        "MAE",
        "bias",
        "RMSE",
        "R2",
        "target_SD",
        "MAE_over_target_SD",
        "mean_feature_OOD_percentile",
        "high_feature_OOD_fraction",
        "mean_model_OOD_percentile",
        "high_model_OOD_fraction",
        "formula_unseen_fraction",
        "coverage_90",
        "coverage_95",
        "interval_90_mean_width",
        "interval_95_mean_width",
    ]
    max_metric_difference = 0.0
    max_count_difference = 0
    for row in performance.itertuples(index=False):
        column = domain_columns[row.domain_type]
        group = predictions.loc[
            predictions["scheme"].eq(row.scheme)
            & predictions["target"].eq(row.target)
            & predictions[column].eq(row.domain)
        ]
        independent = publication_balanced_metrics(group)
        max_count_difference = max(
            max_count_difference,
            abs(len(group) - int(row.records)),
            abs(group["doi_norm"].nunique() - int(row.DOI)),
        )
        for metric in metric_names:
            left, right = float(getattr(row, metric)), float(independent[metric])
            if np.isfinite(left) and np.isfinite(right):
                max_metric_difference = max(max_metric_difference, abs(left - right))

    max_support_count_difference = 0
    for row in support.itertuples(index=False):
        column = domain_columns[row.domain_type]
        train = assignments.loc[
            assignments["publication_year"].le(2018)
            & assignments[column].eq(row.domain)
        ]
        future = assignments.loc[
            assignments["publication_year"].ge(2019)
            & assignments[column].eq(row.domain)
        ]
        differences = [
            abs(len(train) - int(row.historical_records)),
            abs(train["doi_norm"].nunique() - int(row.historical_DOI)),
            abs(len(future) - int(row.future_records)),
            abs(future["doi_norm"].nunique() - int(row.future_DOI)),
        ]
        max_support_count_difference = max(max_support_count_difference, *differences)

    grouped = predictions.loc[
        predictions["scheme"].eq(GROUPED), ["Ref_ID", "target", "y_true"]
    ]
    chrono = predictions.loc[
        predictions["scheme"].eq(CHRONO), ["Ref_ID", "target", "y_true"]
    ]
    paired = grouped.merge(
        chrono,
        on=["Ref_ID", "target"],
        suffixes=("_grouped", "_chrono"),
        validate="one_to_one",
    )
    max_paired_y_difference = float(
        np.max(np.abs(paired["y_true_grouped"] - paired["y_true_chrono"]))
    )

    ci_low = [column for column in performance.columns if column.endswith("_CI_low")]
    ci_order_ok = all(
        bool(
            (
                performance[low].isna()
                | performance[low.replace("_CI_low", "_CI_high")].isna()
                | performance[low].le(
                    performance[low.replace("_CI_low", "_CI_high")]
                )
            ).all()
        )
        for low in ci_low
    )
    hashes_ok = (
        sha256(args.source_predictions)
        == manifest["inputs"]["predictions_sha256"]
    )
    figures_ok = all(
        (args.results_dir / f"Figure8_composition_domain_reliability.{suffix}").exists()
        for suffix in ["png", "pdf", "svg"]
    )
    checks = {
        "assignment_row_count": len(assignments) == 33175,
        "normalized_DOI_count": assignments["doi_norm"].nunique() == 6368,
        "historical_future_partition_counts": bool(
            assignments["publication_year"].le(2018).sum() == 24059
            and assignments["publication_year"].ge(2019).sum() == 9116
        ),
        "A_site_labels": label_a_ok,
        "B_site_labels": label_b_ok,
        "cross_domain_labels": domain_ok,
        "historical_formula_novelty": formula_novelty_ok,
        "future_prediction_row_count": len(predictions) == 72928,
        "prediction_key_uniqueness": not predictions.duplicated(
            ["Ref_ID", "scheme", "target"]
        ).any(),
        "source_archive_row_match": len(joined) == len(predictions),
        "source_archive_numeric_match": max(archive_diffs.values()) <= 1e-12,
        "paired_future_outcomes": max_paired_y_difference <= 1e-12,
        "performance_counts_recomputed": max_count_difference == 0,
        "performance_metrics_recomputed": max_metric_difference <= 1e-12,
        "support_counts_recomputed": max_support_count_difference == 0,
        "confidence_interval_order": ci_order_ok,
        "source_hash_match": hashes_ok,
        "figure_files_present": figures_ok,
        "primary_verification_passed": manifest["verification"]["status"] == "passed",
    }
    report = {
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "details": {
            "checks_passed": int(sum(checks.values())),
            "checks_total": int(len(checks)),
            "assignment_rows": int(len(assignments)),
            "prediction_rows": int(len(predictions)),
            "maximum_archive_difference": float(max(archive_diffs.values())),
            "archive_differences": archive_diffs,
            "maximum_performance_metric_difference": float(max_metric_difference),
            "maximum_performance_count_difference": int(max_count_difference),
            "maximum_support_count_difference": int(max_support_count_difference),
            "maximum_paired_y_true_difference": max_paired_y_difference,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["status"] != "passed":
        raise RuntimeError("Independent composition-domain verification failed.")


if __name__ == "__main__":
    main()
