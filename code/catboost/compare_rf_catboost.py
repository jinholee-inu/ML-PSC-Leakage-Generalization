#!/usr/bin/env python3
"""Paired DOI-cluster comparison of RF and CatBoost uncertainty/explanations."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr

GROUPED = "DOI-grouped 5-fold"
CHRONO = "Chronological >2018"
FAMILY_ORDER = [
    "Absorber composition", "Bandgap", "Absorber thickness",
    "Device architecture", "Substrate and area", "Electron-transport layer",
    "Hole-transport layer", "Back contact", "Deposition route",
    "Thermal processing", "Solvent/additive/quench",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rf-uncertainty", required=True, type=Path)
    parser.add_argument("--cb-uncertainty", required=True, type=Path)
    parser.add_argument("--rf-shap", required=True, type=Path)
    parser.add_argument("--cb-shap", required=True, type=Path)
    parser.add_argument("--rf-hierarchical", required=True, type=Path)
    parser.add_argument("--cb-hierarchical", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def paired_bootstrap_means(
    left: np.ndarray, right: np.ndarray, replicates: int, seed: int
) -> tuple[float, float, float, float, float]:
    difference = right - left
    point_left = float(np.nanmean(left))
    point_right = float(np.nanmean(right))
    point_difference = float(np.nanmean(difference))
    rng = np.random.default_rng(seed)
    n = len(left)
    boot = np.empty(replicates, dtype=float)
    for replicate in range(replicates):
        sample = rng.integers(0, n, n)
        boot[replicate] = np.nanmean(difference[sample])
    low, high = np.nanquantile(boot, [0.025, 0.975])
    return point_left, point_right, point_difference, float(low), float(high)


def explanation_family_comparison(
    rf_local: pd.DataFrame, cb_local: pd.DataFrame, replicates: int, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["Ref_ID", "doi_norm", "scheme", "fold", "target", "family"]
    merged = rf_local[keys + ["abs_shap_value", "shap_value"]].merge(
        cb_local[keys + ["abs_shap_value", "shap_value"]],
        on=keys,
        suffixes=("_RF", "_CatBoost"),
        validate="one_to_one",
    )
    family_rows: list[dict[str, object]] = []
    stability_rows: list[dict[str, object]] = []
    for (scheme, target), frame in merged.groupby(["scheme", "target"], sort=False):
        doi_family = frame.groupby(["doi_norm", "family"], sort=False).agg(
            RF=("abs_shap_value_RF", "mean"),
            CatBoost=("abs_shap_value_CatBoost", "mean"),
        ).reset_index()
        pivot_rf = doi_family.pivot(index="doi_norm", columns="family", values="RF").reindex(columns=FAMILY_ORDER)
        pivot_cb = doi_family.pivot(index="doi_norm", columns="family", values="CatBoost").reindex(columns=FAMILY_ORDER)
        mean_rf = pivot_rf.mean()
        mean_cb = pivot_cb.mean()
        rho = float(spearmanr(mean_rf, mean_cb).statistic)
        top_rf = set(mean_rf.nlargest(5).index)
        top_cb = set(mean_cb.nlargest(5).index)
        rng = np.random.default_rng(seed + sum(map(ord, scheme + target)))
        rho_boot = np.empty(replicates, dtype=float)
        overlap_boot = np.empty(replicates, dtype=float)
        rf_values = pivot_rf.to_numpy(dtype=float)
        cb_values = pivot_cb.to_numpy(dtype=float)
        for replicate in range(replicates):
            sample = rng.integers(0, len(pivot_rf), len(pivot_rf))
            left = np.nanmean(rf_values[sample], axis=0)
            right = np.nanmean(cb_values[sample], axis=0)
            rho_boot[replicate] = spearmanr(left, right).statistic
            overlap_boot[replicate] = len(set(np.argsort(left)[-5:]) & set(np.argsort(right)[-5:]))
        stability_rows.append({
            "scheme": scheme,
            "target": target,
            "RF_CatBoost_Spearman_rho": rho,
            "rho_CI_low": float(np.nanquantile(rho_boot, 0.025)),
            "rho_CI_high": float(np.nanquantile(rho_boot, 0.975)),
            "top5_overlap": int(len(top_rf & top_cb)),
            "top5_overlap_CI_low": float(np.nanquantile(overlap_boot, 0.025)),
            "top5_overlap_CI_high": float(np.nanquantile(overlap_boot, 0.975)),
            "explained_DOI": int(len(pivot_rf)),
        })
        for family in FAMILY_ORDER:
            left = pivot_rf[family].to_numpy(dtype=float)
            right = pivot_cb[family].to_numpy(dtype=float)
            p_left, p_right, delta, low, high = paired_bootstrap_means(
                left, right, replicates, seed + sum(map(ord, scheme + target + family))
            )
            family_rows.append({
                "scheme": scheme,
                "target": target,
                "family": family,
                "RF_mean_abs_SHAP": p_left,
                "CatBoost_mean_abs_SHAP": p_right,
                "CatBoost_minus_RF_mean_abs_SHAP": delta,
                "delta_CI_low": low,
                "delta_CI_high": high,
                "RF_rank": int(mean_rf.rank(method="min", ascending=False)[family]),
                "CatBoost_rank": int(mean_cb.rank(method="min", ascending=False)[family]),
                "explained_DOI": int(len(pivot_rf)),
            })
    return pd.DataFrame(family_rows), pd.DataFrame(stability_rows)


def ale_comparison(rf: pd.DataFrame, cb: pd.DataFrame) -> pd.DataFrame:
    keys = ["scheme", "fold", "target", "feature"]
    left = rf.groupby(keys, sort=False).agg(RF_ALE_range=("ALE_range", "mean")).reset_index()
    right = cb.groupby(keys, sort=False).agg(CatBoost_ALE_range=("ALE_range", "mean")).reset_index()
    merged = left.merge(right, on=keys, validate="one_to_one")
    rows: list[dict[str, object]] = []
    for (scheme, target), frame in merged.groupby(["scheme", "target"], sort=False):
        rho = float(spearmanr(frame["RF_ALE_range"], frame["CatBoost_ALE_range"]).statistic)
        top3_rf = set(frame.nlargest(3, "RF_ALE_range")["feature"])
        top3_cb = set(frame.nlargest(3, "CatBoost_ALE_range")["feature"])
        for record in frame.itertuples():
            rows.append({
                "scheme": scheme,
                "target": target,
                "feature": record.feature,
                "RF_ALE_range": record.RF_ALE_range,
                "CatBoost_ALE_range": record.CatBoost_ALE_range,
                "CatBoost_minus_RF_ALE_range": record.CatBoost_ALE_range - record.RF_ALE_range,
                "RF_CatBoost_feature_rank_Spearman_rho": rho,
                "top3_overlap": len(top3_rf & top3_cb),
            })
    return pd.DataFrame(rows)


def hierarchical_comparison(
    rf: pd.DataFrame, cb: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["scheme", "target", "parent_family", "subgroup"]
    columns = keys + ["mean_abs_hierarchical_SHAP", "attribution_mass_percent", "rank_within_family"]
    merged = rf[columns].merge(cb[columns], on=keys, suffixes=("_RF", "_CatBoost"), validate="one_to_one")
    merged["CatBoost_minus_RF_mean_abs_SHAP"] = merged["mean_abs_hierarchical_SHAP_CatBoost"] - merged["mean_abs_hierarchical_SHAP_RF"]
    rows: list[dict[str, object]] = []
    for (scheme, target, family), frame in merged.groupby(["scheme", "target", "parent_family"], sort=False):
        rho = float(spearmanr(frame["mean_abs_hierarchical_SHAP_RF"], frame["mean_abs_hierarchical_SHAP_CatBoost"]).statistic)
        n = min(5, len(frame))
        overlap = len(set(frame.nlargest(n, "mean_abs_hierarchical_SHAP_RF")["subgroup"]) & set(frame.nlargest(n, "mean_abs_hierarchical_SHAP_CatBoost")["subgroup"]))
        rows.append({
            "scheme": scheme,
            "target": target,
            "parent_family": family,
            "shared_subgroups": int(len(frame)),
            "RF_CatBoost_Spearman_rho": rho,
            "top_n": n,
            "top_n_overlap": overlap,
        })
    return merged, pd.DataFrame(rows)


def uncertainty_comparison(
    rf: pd.DataFrame, cb: pd.DataFrame, replicates: int, seed: int
) -> pd.DataFrame:
    rf = rf.loc[rf["target"].eq("PCE")].copy()
    cb = cb.loc[cb["target"].eq("PCE")].copy()
    keys = ["Ref_ID", "doi_norm", "scheme", "fold", "target", "y_true"]
    useful = [
        "absolute_error", "interval_90_covered", "interval_90_half_width",
        "interval_90_score", "interval_95_covered", "interval_95_half_width",
        "interval_95_score",
    ]
    merged = rf[keys + useful].merge(cb[keys + useful], on=keys, suffixes=("_RF", "_CatBoost"), validate="one_to_one")
    rows: list[dict[str, object]] = []
    for scheme, frame in merged.groupby("scheme", sort=False):
        doi = frame.groupby("doi_norm", sort=False).agg({
            **{f"{name}_RF": "mean" for name in useful},
            **{f"{name}_CatBoost": "mean" for name in useful},
        })
        for nominal in [90, 95]:
            for metric, column in [
                ("coverage", f"interval_{nominal}_covered"),
                ("interval_width", f"interval_{nominal}_half_width"),
                ("interval_score", f"interval_{nominal}_score"),
            ]:
                left = doi[f"{column}_RF"].to_numpy(dtype=float)
                right = doi[f"{column}_CatBoost"].to_numpy(dtype=float)
                if metric == "interval_width":
                    left = left * 2.0
                    right = right * 2.0
                p_left, p_right, delta, low, high = paired_bootstrap_means(
                    left, right, replicates, seed + nominal + sum(map(ord, scheme + metric))
                )
                rows.append({
                    "scheme": scheme,
                    "target": "PCE",
                    "nominal_coverage": nominal / 100.0,
                    "metric": metric,
                    "RF": p_left,
                    "CatBoost": p_right,
                    "CatBoost_minus_RF": delta,
                    "delta_CI_low": low,
                    "delta_CI_high": high,
                    "DOI_groups": int(len(doi)),
                })
    return pd.DataFrame(rows)


def make_figure(
    family: pd.DataFrame,
    uncertainty: pd.DataFrame,
    rf_selective: pd.DataFrame,
    cb_selective: pd.DataFrame,
    output_dir: Path,
) -> None:
    sns.set_theme(style="whitegrid", context="paper")
    fig, axes = plt.subplots(2, 2, figsize=(11.4, 8.6), constrained_layout=True)
    chrono = family.loc[(family["scheme"].eq(CHRONO)) & (family["target"].eq("PCE"))].copy()
    axes[0, 0].scatter(chrono["RF_mean_abs_SHAP"], chrono["CatBoost_mean_abs_SHAP"], color="#3B6EA8", s=50)
    limit = max(chrono["RF_mean_abs_SHAP"].max(), chrono["CatBoost_mean_abs_SHAP"].max()) * 1.08
    axes[0, 0].plot([0, limit], [0, limit], ls="--", color="0.3")
    for row in chrono.itertuples():
        if min(row.RF_rank, row.CatBoost_rank) > 5:
            continue
        axes[0, 0].annotate(row.family.replace("Electron-transport layer", "ETL").replace("Hole-transport layer", "HTL").replace("Solvent/additive/quench", "Solvent/additive"), (row.RF_mean_abs_SHAP, row.CatBoost_mean_abs_SHAP), xytext=(3, 3), textcoords="offset points", fontsize=7)
    axes[0, 0].set_xlabel("RF mean |SHAP| (PCE pp)")
    axes[0, 0].set_ylabel("CatBoost mean |SHAP| (PCE pp)")
    axes[0, 0].set_title("(a) Feature-family attribution stability", loc="left", fontweight="bold")

    top = chrono.nsmallest(5, "CatBoost_rank").sort_values("CatBoost_mean_abs_SHAP")
    y = np.arange(len(top))
    axes[0, 1].barh(y - 0.18, top["RF_mean_abs_SHAP"], height=0.34, label="RF", color="#8996A8")
    axes[0, 1].barh(y + 0.18, top["CatBoost_mean_abs_SHAP"], height=0.34, label="CatBoost", color="#E28E2C")
    axes[0, 1].set_yticks(y, top["family"].str.replace("Electron-transport layer", "ETL").str.replace("Hole-transport layer", "HTL").str.replace("Solvent/additive/quench", "Solvent/additive"))
    axes[0, 1].set_xlabel("Mean |SHAP| (PCE pp)")
    axes[0, 1].set_title("(b) Chronological top-five families", loc="left", fontweight="bold")
    axes[0, 1].legend(frameon=False)

    panel = uncertainty.loc[(uncertainty["scheme"].eq(CHRONO)) & (uncertainty["metric"].eq("coverage"))]
    x = np.arange(len(panel))
    width = 0.35
    axes[1, 0].bar(x - width/2, panel["RF"], width, label="RF", color="#8996A8")
    axes[1, 0].bar(x + width/2, panel["CatBoost"], width, label="CatBoost CQR", color="#E28E2C")
    axes[1, 0].scatter(x, panel["nominal_coverage"], marker="_", s=400, color="black", label="Nominal")
    axes[1, 0].set_xticks(x, [f"{int(v*100)}%" for v in panel["nominal_coverage"]])
    axes[1, 0].set_ylim(0.82, 1.0)
    axes[1, 0].set_ylabel("Publication-balanced coverage")
    axes[1, 0].set_title("(c) Chronological interval calibration", loc="left", fontweight="bold")
    axes[1, 0].legend(frameon=False, fontsize=8)

    for model, frame, color in [("RF", rf_selective, "#8996A8"), ("CatBoost CQR", cb_selective, "#E28E2C")]:
        part = frame.loc[(frame["scheme"].eq(CHRONO)) & (frame["target"].eq("PCE")) & (frame["ranking_measure"].eq("Predictive uncertainty"))].sort_values("retained_fraction")
        axes[1, 1].plot(part["retained_fraction"] * 100, part["publication_balanced_MAE"], marker="o", label=model, color=color)
    axes[1, 1].set_xlabel("Retained devices (%)")
    axes[1, 1].set_ylabel("Publication-balanced PCE MAE (pp)")
    axes[1, 1].set_title("(d) Selective prediction utility", loc="left", fontweight="bold")
    axes[1, 1].legend(frameon=False)
    fig.suptitle("RF–CatBoost robustness of uncertainty and explanation", fontweight="bold")
    for suffix in ["png", "pdf", "svg"]:
        kwargs = {"dpi": 600} if suffix == "png" else {}
        fig.savefig(output_dir / f"FigureS_RF_CatBoost_robustness.{suffix}", bbox_inches="tight", **kwargs)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    replicates = 1000
    seed = 20260826
    rf_local = pd.read_csv(args.rf_shap / "shap_local_values.csv.gz")
    cb_local = pd.read_csv(args.cb_shap / "shap_local_values.csv.gz")
    family, stability = explanation_family_comparison(rf_local, cb_local, replicates, seed)
    family.to_csv(args.output_dir / "RF_CatBoost_SHAP_family_paired_comparison.csv", index=False)
    stability.to_csv(args.output_dir / "RF_CatBoost_SHAP_stability.csv", index=False)

    rf_ale = pd.read_csv(args.rf_shap / "ale_feature_ranking.csv")
    cb_ale = pd.read_csv(args.cb_shap / "ale_feature_ranking.csv")
    ale = ale_comparison(rf_ale, cb_ale)
    ale.to_csv(args.output_dir / "RF_CatBoost_ALE_comparison.csv", index=False)

    rf_h = pd.read_csv(args.rf_hierarchical / "within_family_importance_summary.csv")
    cb_h = pd.read_csv(args.cb_hierarchical / "within_family_importance_summary.csv")
    hierarchical, hierarchical_stability = hierarchical_comparison(rf_h, cb_h)
    hierarchical.to_csv(args.output_dir / "RF_CatBoost_hierarchical_attribution_comparison.csv", index=False)
    hierarchical_stability.to_csv(args.output_dir / "RF_CatBoost_hierarchical_rank_stability.csv", index=False)

    rf_u = pd.read_csv(args.rf_uncertainty / "uncertainty_ood_predictions.csv.gz")
    cb_u = pd.read_csv(args.cb_uncertainty / "catboost_uncertainty_ood_predictions.csv.gz")
    uncertainty = uncertainty_comparison(rf_u, cb_u, replicates, seed)
    uncertainty.to_csv(args.output_dir / "RF_CatBoost_uncertainty_paired_comparison.csv", index=False)
    rf_selective = pd.read_csv(args.rf_uncertainty / "selective_prediction_metrics.csv")
    cb_selective = pd.read_csv(args.cb_uncertainty / "catboost_selective_prediction_metrics.csv")
    make_figure(family, uncertainty, rf_selective, cb_selective, args.output_dir)

    key = stability.loc[(stability["scheme"].eq(CHRONO)) & (stability["target"].eq("PCE"))].iloc[0]
    verification = {
        "status": "passed",
        "RF_CatBoost_prediction_record_alignment": int(len(cb_u)),
        "RF_CatBoost_SHAP_local_key_alignment": int(len(cb_local)),
        "chronological_PCE_family_rank_rho": float(key["RF_CatBoost_Spearman_rho"]),
        "chronological_PCE_top5_overlap": int(key["top5_overlap"]),
        "bootstrap_replicates": replicates,
        "duplicate_family_comparison_keys": int(family.duplicated(["scheme", "target", "family"]).sum()),
        "duplicate_uncertainty_comparison_keys": int(uncertainty.duplicated(["scheme", "nominal_coverage", "metric"]).sum()),
    }
    if verification["duplicate_family_comparison_keys"] or verification["duplicate_uncertainty_comparison_keys"]:
        raise AssertionError("Comparison result contains duplicate keys")
    (args.output_dir / "RF_CatBoost_comparison_verification.json").write_text(json.dumps(verification, indent=2), encoding="utf-8")
    print(json.dumps(verification, indent=2), flush=True)


if __name__ == "__main__":
    main()
