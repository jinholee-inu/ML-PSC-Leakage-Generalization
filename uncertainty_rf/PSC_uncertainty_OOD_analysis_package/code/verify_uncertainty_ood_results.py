#!/usr/bin/env python3
"""Independent integrity checks for completed PSC uncertainty/OOD outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", required=True, type=Path)
    parser.add_argument("--archived-weighting-predictions", required=True, type=Path)
    return parser.parse_args()


def evaluation_mean(frame: pd.DataFrame, column: str, lens: str) -> float:
    if lens == "Device-level":
        return float(frame[column].mean())
    if lens == "Publication-balanced":
        return float(frame.groupby("doi_norm", sort=False)[column].mean().mean())
    raise ValueError(lens)


def main() -> None:
    args = parse_args()
    predictions = pd.read_csv(args.results_dir / "uncertainty_ood_predictions.csv.gz")
    metrics = pd.read_csv(args.results_dir / "uncertainty_coverage_metrics.csv")
    archived = pd.read_csv(args.archived_weighting_predictions)
    archived = archived.loc[
        archived["training_weighting"].eq("Full 1/n_DOI")
        & archived["model"].eq("Random Forest")
        & archived["scheme"].isin(["DOI-grouped 5-fold", "Chronological >2018"])
    ].copy()

    duplicate_keys = int(
        predictions.duplicated(["scheme", "fold", "Ref_ID", "target"]).sum()
    )
    merged = predictions[["Ref_ID", "scheme", "fold", "target", "y_true", "y_pred"]].merge(
        archived[["Ref_ID", "scheme", "fold", "target", "y_true", "y_pred"]],
        on=["Ref_ID", "scheme", "fold", "target"],
        how="outer",
        suffixes=("_new", "_archived"),
        indicator=True,
        validate="one_to_one",
    )
    if not merged["_merge"].eq("both").all():
        raise AssertionError("Prediction keys do not match archived weighted results")
    max_target_difference = float(
        np.max(np.abs(merged["y_true_new"] - merged["y_true_archived"]))
    )
    max_prediction_difference = float(
        np.max(np.abs(merged["y_pred_new"] - merged["y_pred_archived"]))
    )

    recalculation_differences: list[float] = []
    for row in metrics.itertuples(index=False):
        frame = predictions.loc[
            predictions["scheme"].eq(row.scheme)
            & predictions["target"].eq(row.target)
        ].copy()
        label = int(round(float(row.nominal_coverage) * 100))
        frame["coverage_value"] = frame[f"interval_{label}_covered"].astype(float)
        frame["width_value"] = 2.0 * frame[f"interval_{label}_half_width"]
        frame["score_value"] = frame[f"interval_{label}_score"]
        observed = [
            evaluation_mean(frame, "coverage_value", row.evaluation_lens),
            evaluation_mean(frame, "width_value", row.evaluation_lens),
            evaluation_mean(frame, "score_value", row.evaluation_lens),
        ]
        reported = [
            float(row.empirical_coverage),
            float(row.mean_interval_width),
            float(row.mean_interval_score),
        ]
        recalculation_differences.extend(
            abs(left - right) for left, right in zip(observed, reported, strict=True)
        )

    interval_order_valid = bool(
        (predictions["interval_95_lower"] <= predictions["interval_90_lower"]).all()
        and (predictions["interval_90_lower"] <= predictions["y_pred"]).all()
        and (predictions["y_pred"] <= predictions["interval_90_upper"]).all()
        and (predictions["interval_90_upper"] <= predictions["interval_95_upper"]).all()
    )
    report = {
        "status": "passed",
        "prediction_rows": int(len(predictions)),
        "duplicate_prediction_keys": duplicate_keys,
        "max_target_difference_from_archive": max_target_difference,
        "max_prediction_difference_from_archive": max_prediction_difference,
        "max_metric_recalculation_difference": float(max(recalculation_differences)),
        "interval_order_valid": interval_order_valid,
        "finite_numeric_outputs": bool(
            np.isfinite(
                predictions[
                    [
                        "y_true",
                        "y_pred",
                        "ensemble_std",
                        "interval_90_lower",
                        "interval_90_upper",
                        "interval_95_lower",
                        "interval_95_upper",
                        "feature_ood_percentile",
                        "model_ood_percentile",
                    ]
                ].to_numpy(dtype=float)
            ).all()
        ),
    }
    if duplicate_keys:
        raise AssertionError("Duplicate prediction keys")
    if max_target_difference > 1e-12 or max_prediction_difference > 1e-10:
        raise AssertionError("Archived prediction mismatch")
    if report["max_metric_recalculation_difference"] > 1e-12:
        raise AssertionError("Metric recalculation mismatch")
    if not interval_order_valid or not report["finite_numeric_outputs"]:
        raise AssertionError("Invalid interval or non-finite output")
    (args.results_dir / "independent_verification_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
