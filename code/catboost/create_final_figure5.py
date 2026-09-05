#!/usr/bin/env python3
"""Create evidence-calibrated final Figure 5 from CatBoost uncertainty results."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

GROUPED = "DOI-grouped 5-fold"
CHRONO = "Chronological >2018"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    coverage = pd.read_csv(args.results_dir / "catboost_uncertainty_coverage_metrics.csv")
    quantiles = pd.read_csv(args.results_dir / "catboost_uncertainty_ood_quintile_performance.csv")
    selective = pd.read_csv(args.results_dir / "catboost_selective_prediction_metrics.csv")
    sns.set_theme(style="whitegrid", context="paper")
    palette = {GROUPED: "#3B6EA8", CHRONO: "#C44E52"}
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 8.4), constrained_layout=True)

    panel = quantiles.loc[(quantiles["target"].eq("PCE")) & (quantiles["ranking_measure"].eq("Predictive uncertainty"))]
    for scheme in [GROUPED, CHRONO]:
        part = panel.loc[panel["scheme"].eq(scheme)].sort_values("quantile")
        axes[0, 0].errorbar(part["quantile"], part["publication_balanced_MAE"], yerr=np.vstack([part["publication_balanced_MAE"] - part["MAE_CI_low"], part["MAE_CI_high"] - part["publication_balanced_MAE"]]), marker="o", capsize=3, linewidth=1.5, color=palette[scheme], label=scheme.replace(" 5-fold", ""))
    axes[0, 0].set_title("(a) Uncertainty weakly stratifies PCE error", loc="left", fontweight="bold")
    axes[0, 0].set_xlabel("Predictive-uncertainty quintile")
    axes[0, 0].set_ylabel("Publication-balanced PCE MAE (pp)")
    axes[0, 0].set_xticks(range(1, 6))
    axes[0, 0].legend(frameon=False, fontsize=8)

    panel = coverage.loc[(coverage["target"].eq("PCE")) & (coverage["evaluation_lens"].eq("Publication-balanced"))]
    offsets = {GROUPED: -0.003, CHRONO: 0.003}
    for scheme in [GROUPED, CHRONO]:
        part = panel.loc[panel["scheme"].eq(scheme)].sort_values("nominal_coverage")
        axes[0, 1].errorbar(part["nominal_coverage"] + offsets[scheme], part["empirical_coverage"], yerr=np.vstack([part["empirical_coverage"] - part["coverage_CI_low"], part["coverage_CI_high"] - part["empirical_coverage"]]), marker="o", capsize=3, linestyle="none", color=palette[scheme], label=scheme.replace(" 5-fold", ""))
    axes[0, 1].plot([0.885, 0.965], [0.885, 0.965], "--", color="#555555", linewidth=1)
    axes[0, 1].set_xlim(0.885, 0.965)
    axes[0, 1].set_ylim(0.80, 1.00)
    axes[0, 1].set_xticks([0.90, 0.95], ["90%", "95%"])
    axes[0, 1].set_yticks([0.80, 0.85, 0.90, 0.95, 1.00], ["80%", "85%", "90%", "95%", "100%"])
    axes[0, 1].set_title("(b) Conservative publication-balanced coverage", loc="left", fontweight="bold")
    axes[0, 1].set_xlabel("Nominal coverage")
    axes[0, 1].set_ylabel("Empirical coverage")
    axes[0, 1].legend(frameon=False, fontsize=8)

    panel = quantiles.loc[(quantiles["target"].eq("PCE")) & (quantiles["ranking_measure"].eq("Model-support OOD"))]
    for scheme in [GROUPED, CHRONO]:
        part = panel.loc[panel["scheme"].eq(scheme)].sort_values("quantile")
        axes[1, 0].errorbar(part["quantile"], part["publication_balanced_MAE"], yerr=np.vstack([part["publication_balanced_MAE"] - part["MAE_CI_low"], part["MAE_CI_high"] - part["publication_balanced_MAE"]]), marker="o", capsize=3, linewidth=1.5, color=palette[scheme], label=scheme.replace(" 5-fold", ""))
    axes[1, 0].set_title("(c) Leaf-support OOD is not an error proxy", loc="left", fontweight="bold")
    axes[1, 0].set_xlabel("Model-support OOD quintile")
    axes[1, 0].set_ylabel("Publication-balanced PCE MAE (pp)")
    axes[1, 0].set_xticks(range(1, 6))
    axes[1, 0].legend(frameon=False, fontsize=8)

    panel = selective.loc[(selective["scheme"].eq(CHRONO)) & (selective["target"].eq("PCE"))]
    measure_palette = {"Predictive uncertainty": "#4C72B0", "Feature-space OOD": "#DD8452", "Model-support OOD": "#55A868"}
    for measure, color in measure_palette.items():
        part = panel.loc[panel["ranking_measure"].eq(measure)].sort_values("retained_fraction")
        axes[1, 1].plot(100 * part["retained_fraction"], part["publication_balanced_MAE"], marker="o", linewidth=1.6, color=color, label=measure)
    axes[1, 1].set_title("(d) Selective prediction in future publications", loc="left", fontweight="bold")
    axes[1, 1].set_xlabel("Predictions retained (%)")
    axes[1, 1].set_ylabel("Publication-balanced PCE MAE (pp)")
    axes[1, 1].set_xticks([50, 60, 75, 90, 100])
    axes[1, 1].legend(frameon=False, fontsize=8)
    fig.suptitle("CatBoost uncertainty calibration and supported-domain diagnostics", fontsize=14, fontweight="bold")
    for suffix in ["png", "pdf", "svg"]:
        kwargs = {"dpi": 600} if suffix == "png" else {}
        fig.savefig(args.results_dir / f"Figure5_CatBoost_uncertainty_OOD.{suffix}", bbox_inches="tight", **kwargs)
    plt.close(fig)


if __name__ == "__main__":
    main()
