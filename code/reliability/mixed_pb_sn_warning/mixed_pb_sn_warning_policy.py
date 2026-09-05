#!/usr/bin/env python3
"""Leakage-controlled warning policy for mixed Pb-Sn PSC predictions.

The script does not retrain or modify the frozen DOI-balanced Random Forest.
It applies a prospective Green/Amber/Red policy to the archived 2019--2021
chronological predictions using only training-support, target-free OOD, exact
formula novelty, and interval-width diagnostics whose thresholds were locked
without future outcomes. Future outcomes are used only for policy evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
import sklearn


CHRONO_SCHEME = "Chronological >2018"
TARGETS = ["PCE", "Voc", "Jsc", "FF"]
TARGET_LABELS = {"PCE": "PCE", "Voc": r"$V_{OC}$", "Jsc": r"$J_{SC}$", "FF": "FF"}
TARGET_UNITS = {
    "PCE": "percentage point",
    "Voc": "V",
    "Jsc": "mA cm^-2",
    "FF": "percentage point",
}

# These thresholds are prospective and are not optimized on 2019--2021 outcomes.
DOMAIN_ESTABLISHED_DOI = 100
CELL_SPARSE_DOI = 10
CELL_CRITICAL_DOI = 5
OOD_MODERATE = 0.90
OOD_CRITICAL = 0.95
UNCERTAINTY_MODERATE = 1.5
UNCERTAINTY_CRITICAL = 2.5
MIN_DESCRIPTIVE_DOI = 5
MIN_INFERENTIAL_DOI = 20
BOOTSTRAP_REPLICATES = 1000
SEED = 20260829

# Error thresholds are used only to evaluate warning enrichment, never to assign tiers.
LARGE_ERROR_THRESHOLD = {"PCE": 4.0, "Voc": 0.10, "Jsc": 4.0, "FF": 10.0}

METRICS = [
    "mean_measured",
    "mean_predicted",
    "MAE",
    "RMSE",
    "bias",
    "absolute_bias",
    "R2",
    "coverage_95",
    "mean_interval_95_half_width",
    "large_error_fraction",
    "mean_feature_OOD_percentile",
    "mean_model_OOD_percentile",
    "formula_unseen_fraction",
    "mean_uncertainty_multiplier",
]

TIER_RANK = {"Green": 0, "Amber": 1, "Red": 2}
TIER_COLORS = {"Green": "#2E8B57", "Amber": "#E6A23C", "Red": "#C4473A"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--support", required=True, type=Path)
    parser.add_argument("--calibration", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--bootstrap", type=int, default=BOOTSTRAP_REPLICATES)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ci(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan, np.nan
    return tuple(np.quantile(values, [0.025, 0.975]).tolist())


def publication_weights(frame: pd.DataFrame) -> np.ndarray:
    counts = frame["doi_norm"].value_counts()
    return 1.0 / frame["doi_norm"].map(counts).to_numpy(dtype=float)


def metric_bundle(frame: pd.DataFrame, weights: np.ndarray) -> dict[str, float]:
    y = frame["y_true"].to_numpy(dtype=float)
    p = frame["y_pred"].to_numpy(dtype=float)
    residual = p - y
    y_mean = float(np.average(y, weights=weights))
    denominator = float(np.sum(weights * (y - y_mean) ** 2))
    mse = float(np.average(residual**2, weights=weights))
    bias = float(np.average(residual, weights=weights))
    return {
        "mean_measured": y_mean,
        "mean_predicted": float(np.average(p, weights=weights)),
        "MAE": float(np.average(np.abs(residual), weights=weights)),
        "RMSE": math.sqrt(mse),
        "bias": bias,
        "absolute_bias": abs(bias),
        "R2": float(1.0 - np.sum(weights * residual**2) / denominator)
        if denominator > 0
        else np.nan,
        "coverage_95": float(
            np.average(frame["interval_95_covered"].to_numpy(dtype=float), weights=weights)
        ),
        "mean_interval_95_half_width": float(
            np.average(frame["interval_95_half_width"].to_numpy(dtype=float), weights=weights)
        ),
        "large_error_fraction": float(
            np.average(frame["large_error"].to_numpy(dtype=float), weights=weights)
        ),
        "mean_feature_OOD_percentile": float(
            np.average(frame["feature_ood_percentile"].to_numpy(dtype=float), weights=weights)
        ),
        "mean_model_OOD_percentile": float(
            np.average(frame["model_ood_percentile"].to_numpy(dtype=float), weights=weights)
        ),
        "formula_unseen_fraction": float(
            np.average(frame["formula_unseen_historical"].to_numpy(dtype=float), weights=weights)
        ),
        "mean_uncertainty_multiplier": float(
            np.average(frame["uncertainty_multiplier"].to_numpy(dtype=float), weights=weights)
        ),
    }


def bootstrap_metric_bundle(
    frame: pd.DataFrame,
    all_dois: pd.Index,
    bootstrap_counts: np.ndarray,
    evaluation: str,
) -> dict[str, np.ndarray]:
    doi_position = pd.Series(np.arange(len(all_dois)), index=all_dois)
    row_positions = doi_position.loc[frame["doi_norm"]].to_numpy(dtype=int)
    within_counts = frame["doi_norm"].value_counts()
    divisor = frame["doi_norm"].map(within_counts).to_numpy(dtype=float)
    output = {metric: np.full(len(bootstrap_counts), np.nan) for metric in METRICS}
    for replicate, counts in enumerate(bootstrap_counts):
        weights = counts[row_positions].astype(float)
        if evaluation == "Publication-balanced":
            weights = weights / divisor
        if weights.sum() <= 0:
            continue
        values = metric_bundle(frame, weights)
        for metric in METRICS:
            output[metric][replicate] = values[metric]
    return output


def warning_reasons(row: pd.Series) -> str:
    reasons: list[str] = []
    if row["warning_tier"] == "Red":
        if row["flag_formula_unseen"]:
            reasons.append("exact formula unseen historically")
        if row["flag_feature_OOD_critical"]:
            reasons.append("feature OOD >= training P95")
        if row["flag_model_OOD_critical"]:
            reasons.append("model-support OOD >= training P95")
        if row["flag_cell_support_critical"]:
            reasons.append("A x B cell has <5 historical DOI")
        if row["flag_uncertainty_critical"]:
            reasons.append("95% interval half-width >=2.5x calibrated floor")
    else:
        reasons.append("mixed Pb-Sn domain has 10-99 historical DOI")
        if row["flag_cell_support_sparse"]:
            reasons.append("A x B cell has 5-9 historical DOI")
        if row["flag_feature_OOD_moderate"]:
            reasons.append("feature OOD >= training P90")
        if row["flag_model_OOD_moderate"]:
            reasons.append("model-support OOD >= training P90")
        if row["flag_uncertainty_moderate"]:
            reasons.append("95% interval half-width >=1.5x calibrated floor")
    return "; ".join(reasons)


def assign_policy(
    predictions: pd.DataFrame,
    support: pd.DataFrame,
    calibration: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float], int]:
    mixed = support.loc[
        support["domain_type"].eq("B-site pattern") & support["domain"].eq("Pb+Sn")
    ]
    if len(mixed) != 1:
        raise ValueError("Expected exactly one mixed Pb-Sn support row")
    mixed_historical_doi = int(mixed.iloc[0]["historical_DOI"])

    cells = support.loc[support["domain_type"].eq("A x B domain")].set_index("domain")
    cell_historical_doi = cells["historical_DOI"].to_dict()

    calibration = calibration.loc[
        calibration["scheme"].eq(CHRONO_SCHEME)
        & calibration["nominal_coverage"].eq(0.95)
    ].copy()
    if set(calibration["target"]) != set(TARGETS) or len(calibration) != len(TARGETS):
        raise ValueError("Chronological 95% calibration rows are incomplete")
    floor_half_width = dict(
        zip(
            calibration["target"],
            calibration["sigma_floor"] * calibration["normalized_residual_quantile"],
        )
    )

    frame = predictions.loc[
        predictions["scheme"].eq(CHRONO_SCHEME)
        & predictions["b_site_pattern"].eq("Pb+Sn")
    ].copy()
    frame["mixed_domain_historical_DOI"] = mixed_historical_doi
    frame["composition_cell_historical_DOI"] = (
        frame["composition_domain"].map(cell_historical_doi).astype(int)
    )
    frame["calibrated_floor_95_half_width"] = frame["target"].map(floor_half_width)
    frame["uncertainty_multiplier"] = (
        frame["interval_95_half_width"] / frame["calibrated_floor_95_half_width"]
    )
    frame["flag_limited_mixed_domain"] = mixed_historical_doi < DOMAIN_ESTABLISHED_DOI
    frame["flag_cell_support_sparse"] = (
        frame["composition_cell_historical_DOI"] < CELL_SPARSE_DOI
    )
    frame["flag_cell_support_critical"] = (
        frame["composition_cell_historical_DOI"] < CELL_CRITICAL_DOI
    )
    frame["flag_formula_unseen"] = frame["formula_unseen_historical"].astype(bool)
    frame["flag_feature_OOD_moderate"] = frame["feature_ood_percentile"].ge(OOD_MODERATE)
    frame["flag_feature_OOD_critical"] = frame["feature_ood_percentile"].ge(OOD_CRITICAL)
    frame["flag_model_OOD_moderate"] = frame["model_ood_percentile"].ge(OOD_MODERATE)
    frame["flag_model_OOD_critical"] = frame["model_ood_percentile"].ge(OOD_CRITICAL)
    frame["flag_uncertainty_moderate"] = frame["uncertainty_multiplier"].ge(
        UNCERTAINTY_MODERATE
    )
    frame["flag_uncertainty_critical"] = frame["uncertainty_multiplier"].ge(
        UNCERTAINTY_CRITICAL
    )

    domain_critical = (
        frame["flag_cell_support_critical"]
        | frame["flag_formula_unseen"]
        | frame["flag_feature_OOD_critical"]
        | frame["flag_model_OOD_critical"]
    )
    if mixed_historical_doi < CELL_SPARSE_DOI:
        domain_tier = np.repeat("Red", len(frame))
    elif mixed_historical_doi >= DOMAIN_ESTABLISHED_DOI:
        domain_tier = np.where(domain_critical, "Red", "Green")
        moderate_domain = (
            frame["flag_cell_support_sparse"]
            | frame["flag_feature_OOD_moderate"]
            | frame["flag_model_OOD_moderate"]
        )
        domain_tier = np.where((domain_tier == "Green") & moderate_domain, "Amber", domain_tier)
    else:
        domain_tier = np.where(domain_critical, "Red", "Amber")
    frame["domain_support_tier"] = domain_tier
    frame["uncertainty_tier"] = np.select(
        [frame["flag_uncertainty_critical"], frame["flag_uncertainty_moderate"]],
        ["Red", "Amber"],
        default="Green",
    )
    frame["warning_tier"] = [
        max((d, u), key=lambda item: TIER_RANK[item])
        for d, u in zip(frame["domain_support_tier"], frame["uncertainty_tier"])
    ]
    frame["warning_reason"] = frame.apply(warning_reasons, axis=1)
    frame["large_error_threshold"] = frame["target"].map(LARGE_ERROR_THRESHOLD)
    frame["large_error"] = frame["absolute_error"].ge(frame["large_error_threshold"])
    frame["recommended_action"] = frame["warning_tier"].map(
        {
            "Green": "Report point estimate and 95% interval with routine model caveats.",
            "Amber": "Report only with 95% interval plus OOD/support fields; do not use as the sole ranking criterion.",
            "Red": "Do not report a point estimate alone or use for automated ranking; require experimental confirmation or domain update.",
        }
    )

    policy = pd.DataFrame(
        [
            {
                "component": "Mixed Pb-Sn domain support",
                "Green": f">={DOMAIN_ESTABLISHED_DOI} historical DOI",
                "Amber": f"{CELL_SPARSE_DOI}-{DOMAIN_ESTABLISHED_DOI-1} historical DOI",
                "Red": f"<{CELL_SPARSE_DOI} historical DOI",
                "locked_source": "historical training metadata",
                "current_value": mixed_historical_doi,
            },
            {
                "component": "A x B composition-cell support",
                "Green": f">={CELL_SPARSE_DOI} historical DOI",
                "Amber": f"{CELL_CRITICAL_DOI}-{CELL_SPARSE_DOI-1} historical DOI",
                "Red": f"<{CELL_CRITICAL_DOI} historical DOI",
                "locked_source": "historical training metadata",
                "current_value": "record-specific",
            },
            {
                "component": "Exact short-form composition",
                "Green": "seen historically",
                "Amber": "not used",
                "Red": "unseen historically",
                "locked_source": "historical composition vocabulary",
                "current_value": "record-specific",
            },
            {
                "component": "Feature-space OOD percentile",
                "Green": f"<{OOD_MODERATE:.2f}",
                "Amber": f"{OOD_MODERATE:.2f}-<{OOD_CRITICAL:.2f}",
                "Red": f">={OOD_CRITICAL:.2f}",
                "locked_source": "training-partition prototype distances",
                "current_value": "record-specific",
            },
            {
                "component": "Model-support OOD percentile",
                "Green": f"<{OOD_MODERATE:.2f}",
                "Amber": f"{OOD_MODERATE:.2f}-<{OOD_CRITICAL:.2f}",
                "Red": f">={OOD_CRITICAL:.2f}",
                "locked_source": "training-partition RF leaf support",
                "current_value": "record-specific",
            },
            {
                "component": "95% interval half-width / calibrated floor",
                "Green": f"<{UNCERTAINTY_MODERATE:.1f}x",
                "Amber": f"{UNCERTAINTY_MODERATE:.1f}-<{UNCERTAINTY_CRITICAL:.1f}x",
                "Red": f">={UNCERTAINTY_CRITICAL:.1f}x",
                "locked_source": "2018 DOI-disjoint calibration partition",
                "current_value": "target- and record-specific",
            },
        ]
    )
    return frame, policy, floor_half_width, mixed_historical_doi


def build_bootstrap_counts(dois: pd.Index, replicates: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(dois), size=(replicates, len(dois)))
    counts = np.zeros((replicates, len(dois)), dtype=np.int16)
    rows = np.repeat(np.arange(replicates), len(dois))
    np.add.at(counts, (rows, draws.ravel()), 1)
    return counts


def performance_tables(
    frame: pd.DataFrame,
    bootstrap_counts: np.ndarray,
    all_dois: pd.Index,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[tuple[str, str, str], dict[str, np.ndarray]]]:
    rows: list[dict[str, object]] = []
    cache: dict[tuple[str, str, str], dict[str, np.ndarray]] = {}
    for target in TARGETS:
        target_frame = frame.loc[frame["target"].eq(target)]
        for tier in ["Amber", "Red", "All"]:
            subset = target_frame if tier == "All" else target_frame.loc[target_frame["warning_tier"].eq(tier)]
            if subset.empty:
                continue
            for evaluation in ["Publication-balanced", "Device-level"]:
                weights = publication_weights(subset) if evaluation == "Publication-balanced" else np.ones(len(subset))
                points = metric_bundle(subset, weights)
                boots = bootstrap_metric_bundle(subset, all_dois, bootstrap_counts, evaluation)
                cache[(target, tier, evaluation)] = boots
                record: dict[str, object] = {
                    "target": target,
                    "unit": TARGET_UNITS[target],
                    "warning_tier": tier,
                    "evaluation": evaluation,
                    "records": int(len(subset)),
                    "DOI_groups": int(subset["doi_norm"].nunique()),
                    "record_fraction": float(len(subset) / len(target_frame)),
                    "descriptive_eligible": int(subset["doi_norm"].nunique()) >= MIN_DESCRIPTIVE_DOI,
                    "inferential_eligible": int(subset["doi_norm"].nunique()) >= MIN_INFERENTIAL_DOI,
                }
                for metric in METRICS:
                    record[metric] = points[metric]
                    low, high = ci(boots[metric]) if record["descriptive_eligible"] else (np.nan, np.nan)
                    record[f"{metric}_CI_low"] = low
                    record[f"{metric}_CI_high"] = high
                rows.append(record)

    metrics = pd.DataFrame(rows)
    comparisons: list[dict[str, object]] = []
    for target in TARGETS:
        for evaluation in ["Publication-balanced", "Device-level"]:
            amber = metrics.loc[
                metrics["target"].eq(target)
                & metrics["warning_tier"].eq("Amber")
                & metrics["evaluation"].eq(evaluation)
            ]
            red = metrics.loc[
                metrics["target"].eq(target)
                & metrics["warning_tier"].eq("Red")
                & metrics["evaluation"].eq(evaluation)
            ]
            if amber.empty or red.empty:
                continue
            amber = amber.iloc[0]
            red = red.iloc[0]
            amber_boot = cache[(target, "Amber", evaluation)]
            red_boot = cache[(target, "Red", evaluation)]
            record = {
                "target": target,
                "unit": TARGET_UNITS[target],
                "evaluation": evaluation,
                "comparison": "Red minus Amber",
                "Amber_records": int(amber["records"]),
                "Amber_DOI": int(amber["DOI_groups"]),
                "Red_records": int(red["records"]),
                "Red_DOI": int(red["DOI_groups"]),
                "inferential_eligible": min(int(amber["DOI_groups"]), int(red["DOI_groups"]))
                >= MIN_INFERENTIAL_DOI,
            }
            difference_metrics = [
                "MAE",
                "RMSE",
                "absolute_bias",
                "large_error_fraction",
                "coverage_95",
                "mean_interval_95_half_width",
            ]
            for metric in difference_metrics:
                point = float(red[metric] - amber[metric])
                boot = red_boot[metric] - amber_boot[metric]
                low, high = ci(boot)
                record[f"delta_{metric}"] = point
                record[f"delta_{metric}_CI_low"] = low
                record[f"delta_{metric}_CI_high"] = high
            ratio = float(red["MAE"] / amber["MAE"])
            with np.errstate(invalid="ignore", divide="ignore"):
                ratio_boot = red_boot["MAE"] / amber_boot["MAE"]
            low, high = ci(ratio_boot)
            record["MAE_ratio_Red_over_Amber"] = ratio
            record["MAE_ratio_CI_low"] = low
            record["MAE_ratio_CI_high"] = high
            record["MAE_change_percent"] = 100.0 * (ratio - 1.0)
            record["MAE_change_percent_CI_low"] = 100.0 * (low - 1.0)
            record["MAE_change_percent_CI_high"] = 100.0 * (high - 1.0)
            comparisons.append(record)
    return metrics, pd.DataFrame(comparisons), cache


def trigger_summary(frame: pd.DataFrame) -> pd.DataFrame:
    triggers = {
        "Limited mixed-domain support (<100 historical DOI)": "flag_limited_mixed_domain",
        "Sparse A x B cell (<10 historical DOI)": "flag_cell_support_sparse",
        "Critical A x B cell (<5 historical DOI)": "flag_cell_support_critical",
        "Exact formula unseen historically": "flag_formula_unseen",
        "Feature OOD >=P90": "flag_feature_OOD_moderate",
        "Feature OOD >=P95": "flag_feature_OOD_critical",
        "Model-support OOD >=P90": "flag_model_OOD_moderate",
        "Model-support OOD >=P95": "flag_model_OOD_critical",
        "Uncertainty multiplier >=1.5x": "flag_uncertainty_moderate",
        "Uncertainty multiplier >=2.5x": "flag_uncertainty_critical",
        "Final Amber tier": None,
        "Final Red tier": None,
    }
    rows: list[dict[str, object]] = []
    for target in TARGETS:
        block = frame.loc[frame["target"].eq(target)]
        eval_weights = publication_weights(block)
        for name, column in triggers.items():
            if name == "Final Amber tier":
                flag = block["warning_tier"].eq("Amber")
            elif name == "Final Red tier":
                flag = block["warning_tier"].eq("Red")
            else:
                flag = block[column].astype(bool)
            rows.append(
                {
                    "target": target,
                    "trigger": name,
                    "records": int(flag.sum()),
                    "DOI_groups_with_trigger": int(block.loc[flag, "doi_norm"].nunique()),
                    "device_fraction": float(flag.mean()),
                    "publication_balanced_fraction": float(
                        np.average(flag.to_numpy(dtype=float), weights=eval_weights)
                    ),
                }
            )
    return pd.DataFrame(rows)


def sensitivity_table(frame: pd.DataFrame) -> pd.DataFrame:
    variants = {
        "Permissive": {"ood": 0.99, "uncertainty": 3.0, "cell": 2},
        "Primary (locked)": {"ood": 0.95, "uncertainty": 2.5, "cell": 5},
        "Conservative": {"ood": 0.90, "uncertainty": 2.0, "cell": 10},
    }
    rows: list[dict[str, object]] = []
    for variant, settings in variants.items():
        red = (
            frame["formula_unseen_historical"].astype(bool)
            | frame["feature_ood_percentile"].ge(settings["ood"])
            | frame["model_ood_percentile"].ge(settings["ood"])
            | frame["uncertainty_multiplier"].ge(settings["uncertainty"])
            | frame["composition_cell_historical_DOI"].lt(settings["cell"])
        )
        for target in TARGETS:
            block = frame.loc[frame["target"].eq(target)].copy()
            block_red = red.loc[block.index]
            amber_frame = block.loc[~block_red]
            red_frame = block.loc[block_red]
            eval_weights = publication_weights(block)
            record: dict[str, object] = {
                "variant": variant,
                "selected_policy": variant == "Primary (locked)",
                "target": target,
                "OOD_critical_threshold": settings["ood"],
                "uncertainty_critical_multiplier": settings["uncertainty"],
                "critical_cell_historical_DOI_less_than": settings["cell"],
                "Red_records": int(block_red.sum()),
                "Red_DOI": int(block.loc[block_red, "doi_norm"].nunique()),
                "Red_publication_balanced_fraction": float(
                    np.average(block_red.to_numpy(dtype=float), weights=eval_weights)
                ),
            }
            for label, subset in [("Amber", amber_frame), ("Red", red_frame)]:
                if subset.empty:
                    record[f"{label}_MAE"] = np.nan
                    record[f"{label}_large_error_fraction"] = np.nan
                else:
                    values = metric_bundle(subset, publication_weights(subset))
                    record[f"{label}_MAE"] = values["MAE"]
                    record[f"{label}_large_error_fraction"] = values["large_error_fraction"]
            record["MAE_ratio_Red_over_Amber"] = (
                record["Red_MAE"] / record["Amber_MAE"]
                if record["Amber_MAE"] and np.isfinite(record["Amber_MAE"])
                else np.nan
            )
            rows.append(record)
    return pd.DataFrame(rows)


def draw_figure(
    frame: pd.DataFrame,
    metrics: pd.DataFrame,
    comparisons: pd.DataFrame,
    output_dir: Path,
    mixed_historical_doi: int,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.2,
            "axes.titlesize": 11,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.2,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 9.6))
    fig.suptitle(
        "Prospective OOD and uncertainty warnings for mixed Pb-Sn PSC predictions",
        x=0.06,
        y=0.985,
        ha="left",
        fontsize=15,
        fontweight="bold",
    )

    ax = axes[0, 0]
    ax.set_axis_off()
    ax.set_title("(a) Locked prospective decision rule", loc="left", fontweight="bold")

    def box(x: float, y: float, width: float, height: float, text: str, fill: str, edge: str) -> None:
        patch = FancyBboxPatch(
            (x, y), width, height,
            boxstyle="round,pad=0.012,rounding_size=0.015",
            facecolor=fill, edgecolor=edge, linewidth=1.2,
            transform=ax.transAxes,
        )
        ax.add_patch(patch)
        ax.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=8.4, transform=ax.transAxes)

    box(0.05, 0.74, 0.90, 0.17, f"Mixed Pb-Sn gate\n{mixed_historical_doi} historical DOI -> minimum Amber", "#FFF3D6", TIER_COLORS["Amber"])
    triggers = [
        "Exact formula\nunseen",
        "Feature/model OOD\n>= training P95",
        "A x B cell\n<5 historical DOI",
        "95% half-width\n>=2.5x floor",
    ]
    for index, text in enumerate(triggers):
        x = 0.035 + index * 0.242
        box(x, 0.42, 0.215, 0.18, text, "#FBE7E4", TIER_COLORS["Red"])
        ax.annotate("", xy=(x + 0.107, 0.42), xytext=(0.50, 0.74), xycoords=ax.transAxes,
                    arrowprops=dict(arrowstyle="-|>", color="#7A7F87", lw=0.8))
    box(0.05, 0.08, 0.40, 0.17, "Amber\ninterval + OOD/support required", "#FFF3D6", TIER_COLORS["Amber"])
    box(0.55, 0.08, 0.40, 0.17, "Red\nno point-only ranking; confirm experimentally", "#FBE7E4", TIER_COLORS["Red"])
    ax.text(0.05, 0.015, "Green remains closed until mixed Pb-Sn support reaches >=100 historical DOI.",
            transform=ax.transAxes, fontsize=7.8, color="#4B5563")

    ax = axes[0, 1]
    counts = (
        frame.groupby(["target", "warning_tier"], observed=True)
        .size().unstack(fill_value=0).reindex(TARGETS).fillna(0)
    )
    y = np.arange(len(TARGETS))
    amber = counts.get("Amber", pd.Series(0, index=TARGETS)).to_numpy(dtype=float)
    red = counts.get("Red", pd.Series(0, index=TARGETS)).to_numpy(dtype=float)
    total = amber + red
    ax.barh(y, 100 * amber / total, color=TIER_COLORS["Amber"], label="Amber")
    ax.barh(y, 100 * red / total, left=100 * amber / total, color=TIER_COLORS["Red"], label="Red")
    ax.set_yticks(y, [TARGET_LABELS[target] for target in TARGETS])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Future mixed Pb-Sn device records (%)")
    ax.set_title("(b) Target-specific final warning tier", loc="left", fontweight="bold")
    ax.grid(axis="x", color="#E5E7EB", linewidth=0.7)
    ax.legend(frameon=False, loc="lower right")
    for index, (a_count, r_count) in enumerate(zip(amber, red)):
        if a_count:
            ax.text(50 * a_count / total[index], index, f"{int(a_count)}", ha="center", va="center", color="#3D2B00", fontweight="bold")
        if r_count:
            ax.text(100 - 50 * r_count / total[index], index, f"{int(r_count)}", ha="center", va="center", color="white", fontweight="bold")

    ax = axes[1, 0]
    comparison = comparisons.loc[comparisons["evaluation"].eq("Publication-balanced")].set_index("target").reindex(TARGETS)
    ratio = comparison["MAE_ratio_Red_over_Amber"].to_numpy(dtype=float)
    low = comparison["MAE_ratio_CI_low"].to_numpy(dtype=float)
    high = comparison["MAE_ratio_CI_high"].to_numpy(dtype=float)
    ax.errorbar(ratio, y, xerr=[ratio - low, high - ratio], fmt="o", color="#A7362D", ecolor="#A7362D", capsize=3, markersize=6)
    ax.axvline(1.0, color="#555B63", linestyle="--", linewidth=1)
    ax.set_yticks(y, [TARGET_LABELS[target] for target in TARGETS])
    ax.invert_yaxis()
    ax.set_xlabel("Publication-balanced MAE ratio (Red / Amber)")
    ax.set_title("(c) Error enrichment in Red warnings", loc="left", fontweight="bold")
    ax.grid(axis="x", color="#E5E7EB", linewidth=0.7)
    for yi, value in zip(y, ratio):
        ax.text(value + 0.08, yi, f"{value:.2f}x", ha="left", va="center", fontsize=7.7, color="#7A241E")

    ax = axes[1, 1]
    pce = frame.loc[frame["target"].eq("PCE")].copy()
    for tier in ["Amber", "Red"]:
        block = pce.loc[pce["warning_tier"].eq(tier)]
        ax.scatter(block["y_true"], block["y_pred"], s=30, alpha=0.72, color=TIER_COLORS[tier], edgecolor="white", linewidth=0.45, label=tier)
    lower = min(pce["y_true"].min(), pce["y_pred"].min()) - 0.5
    upper = max(pce["y_true"].max(), pce["y_pred"].max()) + 0.5
    ax.plot([lower, upper], [lower, upper], linestyle="--", color="#4B5563", linewidth=1)
    ax.axvspan(20, upper, color="#E9EDF2", alpha=0.55, zorder=0)
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.set_xlabel("Measured PCE (%)")
    ax.set_ylabel("Predicted PCE (%)")
    ax.set_title("(d) Chronological PCE predictions", loc="left", fontweight="bold")
    ax.grid(color="#E5E7EB", linewidth=0.65)
    ax.legend(frameon=False, loc="upper left")
    pce_metrics = metrics.loc[
        metrics["target"].eq("PCE") & metrics["evaluation"].eq("Publication-balanced")
    ].set_index("warning_tier")
    ax.text(
        0.98,
        0.04,
        f"Amber MAE {pce_metrics.loc['Amber','MAE']:.2f}\nRed MAE {pce_metrics.loc['Red','MAE']:.2f}\nAll bias {pce_metrics.loc['All','bias']:.2f}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#D1D5DB", alpha=0.9),
    )

    fig.text(
        0.06,
        0.012,
        "Thresholds were locked without 2019-2021 outcomes. Error bars are global DOI-cluster bootstrap 95% CIs (1,000 replicates). "
        "Tier-specific estimates with <20 DOI groups are descriptive.",
        fontsize=8,
        color="#4B5563",
    )
    fig.tight_layout(rect=[0.04, 0.045, 0.99, 0.95], h_pad=2.2, w_pad=2.0)
    for suffix, dpi in [("png", 600), ("pdf", None), ("svg", None)]:
        kwargs: dict[str, object] = {"bbox_inches": "tight"}
        if dpi is not None:
            kwargs["dpi"] = dpi
        fig.savefig(output_dir / f"Figure10_Mixed_PbSn_warning_policy.{suffix}", **kwargs)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    started = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    predictions = pd.read_csv(args.predictions, low_memory=False)
    support = pd.read_csv(args.support)
    calibration = pd.read_csv(args.calibration)
    frame, policy, floor_half_width, mixed_historical_doi = assign_policy(
        predictions, support, calibration
    )

    all_dois = pd.Index(sorted(frame["doi_norm"].unique()))
    bootstrap_counts = build_bootstrap_counts(all_dois, args.bootstrap, args.seed)
    metrics, comparisons, _ = performance_tables(frame, bootstrap_counts, all_dois)
    triggers = trigger_summary(frame)
    sensitivity = sensitivity_table(frame)

    assignment_columns = [
        "Ref_ID", "doi_norm", "publication_year", "target", "y_true", "y_pred", "residual", "absolute_error",
        "a_site_pattern", "b_site_pattern", "composition_domain", "Sn_fraction_among_Pb_Sn",
        "mixed_domain_historical_DOI", "composition_cell_historical_DOI", "formula_unseen_historical",
        "feature_ood_distance_ratio", "feature_ood_percentile", "model_leaf_support", "model_ood_percentile",
        "interval_90_half_width", "interval_90_covered", "interval_95_half_width", "interval_95_covered",
        "calibrated_floor_95_half_width", "uncertainty_multiplier",
        "flag_limited_mixed_domain", "flag_cell_support_sparse", "flag_cell_support_critical", "flag_formula_unseen",
        "flag_feature_OOD_moderate", "flag_feature_OOD_critical", "flag_model_OOD_moderate", "flag_model_OOD_critical",
        "flag_uncertainty_moderate", "flag_uncertainty_critical", "domain_support_tier", "uncertainty_tier",
        "warning_tier", "warning_reason", "recommended_action", "large_error_threshold", "large_error",
    ]
    frame[assignment_columns].to_csv(
        args.output_dir / "mixed_pb_sn_warning_assignments.csv.gz", index=False, compression="gzip"
    )
    policy.to_csv(args.output_dir / "mixed_pb_sn_warning_policy.csv", index=False)
    metrics.to_csv(args.output_dir / "mixed_pb_sn_warning_metrics.csv", index=False)
    comparisons.to_csv(args.output_dir / "mixed_pb_sn_warning_comparisons.csv", index=False)
    triggers.to_csv(args.output_dir / "mixed_pb_sn_warning_trigger_summary.csv", index=False)
    sensitivity.to_csv(args.output_dir / "mixed_pb_sn_warning_sensitivity.csv", index=False)
    draw_figure(frame, metrics, comparisons, args.output_dir, mixed_historical_doi)

    expected_rows = 98 * len(TARGETS)
    duplicate_keys = int(frame.duplicated(["Ref_ID", "target"]).sum())
    tier_values = sorted(frame["warning_tier"].unique())
    future_outcomes_used_for_thresholds = False
    verification = {
        "status": "passed",
        "prediction_rows": int(len(frame)),
        "expected_prediction_rows": expected_rows,
        "records": int(frame["Ref_ID"].nunique()),
        "DOI_groups": int(frame["doi_norm"].nunique()),
        "duplicate_prediction_keys": duplicate_keys,
        "targets": sorted(frame["target"].unique()),
        "warning_tiers_present": tier_values,
        "missing_policy_values": int(frame[assignment_columns].isna().sum().sum()),
        "bootstrap_replicates": int(args.bootstrap),
        "bootstrap_row_sums_valid": bool(np.all(bootstrap_counts.sum(axis=1) == len(all_dois))),
        "future_outcomes_used_for_thresholds": future_outcomes_used_for_thresholds,
        "all_mixed_pb_sn_minimum_amber": bool((frame["warning_tier"] != "Green").all()),
        "calibrated_floor_half_widths": floor_half_width,
        "figure_files_present": all(
            (args.output_dir / f"Figure10_Mixed_PbSn_warning_policy.{suffix}").exists()
            for suffix in ["png", "pdf", "svg"]
        ),
    }
    if not (
        len(frame) == expected_rows
        and frame["Ref_ID"].nunique() == 98
        and frame["doi_norm"].nunique() == 25
        and duplicate_keys == 0
        and set(frame["target"]) == set(TARGETS)
        and set(tier_values).issubset({"Amber", "Red"})
        and verification["bootstrap_row_sums_valid"]
        and not future_outcomes_used_for_thresholds
    ):
        verification["status"] = "failed"

    manifest = {
        "status": verification["status"],
        "runtime_seconds": time.time() - started,
        "software": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "inputs": {
            "predictions_path": str(args.predictions.resolve()),
            "predictions_sha256": sha256(args.predictions),
            "support_path": str(args.support.resolve()),
            "support_sha256": sha256(args.support),
            "calibration_path": str(args.calibration.resolve()),
            "calibration_sha256": sha256(args.calibration),
        },
        "design": {
            "model_retrained": False,
            "future_cohort": "2019-2021",
            "subgroup": "mixed Pb-Sn",
            "mixed_historical_DOI": mixed_historical_doi,
            "primary_evaluation": "publication-balanced",
            "thresholds_locked_without_future_outcomes": True,
            "future_outcomes_role": "evaluation only",
            "domain_established_DOI": DOMAIN_ESTABLISHED_DOI,
            "cell_sparse_DOI": CELL_SPARSE_DOI,
            "cell_critical_DOI": CELL_CRITICAL_DOI,
            "OOD_moderate": OOD_MODERATE,
            "OOD_critical": OOD_CRITICAL,
            "uncertainty_moderate_multiplier": UNCERTAINTY_MODERATE,
            "uncertainty_critical_multiplier": UNCERTAINTY_CRITICAL,
            "large_error_thresholds": LARGE_ERROR_THRESHOLD,
            "minimum_descriptive_DOI": MIN_DESCRIPTIVE_DOI,
            "minimum_inferential_DOI": MIN_INFERENTIAL_DOI,
            "bootstrap_replicates": args.bootstrap,
            "seed": args.seed,
        },
        "verification": verification,
    }
    (args.output_dir / "mixed_pb_sn_warning_run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    if verification["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
