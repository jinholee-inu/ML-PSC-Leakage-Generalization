#!/usr/bin/env python3
"""Prepare compact, flat, Origin-ready tables for final CatBoost Figures 5--7."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

CHRONO = "Chronological >2018"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--rf-uncertainty", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def save(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    u = args.project / "results" / "uncertainty"
    s = args.project / "results" / "shap"
    h = args.project / "results" / "hierarchical"
    c = args.project / "results" / "comparison"

    mapping = {
        "F5_Coverage.csv": u / "catboost_uncertainty_coverage_metrics.csv",
        "F5_Uncertainty_Quintiles.csv": u / "catboost_uncertainty_ood_quintile_performance.csv",
        "F5_Selective_Prediction.csv": u / "catboost_selective_prediction_metrics.csv",
        "F5_Error_Association.csv": u / "catboost_uncertainty_error_association.csv",
        "F5_OOD_Strata.csv": u / "catboost_ood_stratified_performance.csv",
        "F5_High_PCE.csv": u / "catboost_high_PCE_uncertainty_metrics.csv",
        "F6_SHAP_Family_Summary.csv": s / "shap_family_importance_summary.csv",
        "F6_SHAP_Rank_Stability.csv": s / "shap_rank_stability.csv",
        "F6_ALE_Curves.csv": s / "ale_curves.csv",
        "F6_ALE_Bootstrap.csv": s / "ale_chronological_PCE_bootstrap.csv",
        "F6_ALE_Ranking.csv": s / "ale_feature_ranking.csv",
        "F7_Within_Family_Summary.csv": h / "within_family_importance_summary.csv",
        "F7_Within_Family_Stability.csv": h / "within_family_rank_stability.csv",
        "F7_Material_Vocabulary.csv": h / "historical_training_material_vocabulary.csv",
        "RF_CB_SHAP_Comparison.csv": c / "RF_CatBoost_SHAP_family_paired_comparison.csv",
        "RF_CB_SHAP_Stability.csv": c / "RF_CatBoost_SHAP_stability.csv",
        "RF_CB_ALE_Comparison.csv": c / "RF_CatBoost_ALE_comparison.csv",
        "RF_CB_Hierarchical.csv": c / "RF_CatBoost_hierarchical_attribution_comparison.csv",
        "RF_CB_Hierarchical_Stability.csv": c / "RF_CatBoost_hierarchical_rank_stability.csv",
        "RF_CB_Uncertainty.csv": c / "RF_CatBoost_uncertainty_paired_comparison.csv",
    }
    for output_name, source in mapping.items():
        save(pd.read_csv(source), args.output_dir / output_name)

    local = pd.read_csv(s / "shap_local_values.csv.gz")
    summary = pd.read_csv(s / "shap_family_importance_summary.csv")
    top = summary.loc[(summary["scheme"].eq(CHRONO)) & (summary["target"].eq("PCE"))].nsmallest(6, "rank")["family"]
    signed = local.loc[(local["scheme"].eq(CHRONO)) & (local["target"].eq("PCE")) & (local["family"].isin(top)), ["Ref_ID", "doi_norm", "publication_year", "family", "shap_value", "abs_shap_value", "shap_mcse", "y_true", "y_pred"]]
    signed = signed.merge(summary.loc[(summary["scheme"].eq(CHRONO)) & (summary["target"].eq("PCE")), ["family", "rank"]], on="family", validate="many_to_one").sort_values(["rank", "doi_norm"])
    save(signed, args.output_dir / "F6_Signed_PCE_Future.csv")

    rf_selective = pd.read_csv(args.rf_uncertainty / "selective_prediction_metrics.csv")
    cb_selective = pd.read_csv(u / "catboost_selective_prediction_metrics.csv")
    rf_selective["model"] = "Random Forest"
    cb_selective["model"] = "CatBoost CQR"
    save(pd.concat([rf_selective, cb_selective], ignore_index=True), args.output_dir / "RF_CB_Selective_Prediction.csv")

    key = {
        "frozen_records": 33175,
        "normalized_DOI_groups": 6368,
        "CatBoost_chronological_PCE_R2": 0.445211,
        "CatBoost_chronological_PCE_MAE": 3.0239,
        "CatBoost_chronological_95_coverage_publication_balanced": 0.962845,
        "CatBoost_chronological_95_width": 17.361955,
        "CatBoost_chronological_uncertainty_error_rho": 0.123331,
        "CatBoost_chronological_50pct_selective_MAE": 2.896526,
        "CatBoost_chronological_overall_publication_balanced_MAE": 3.024738,
        "RF_CatBoost_chronological_PCE_family_rank_rho": 0.981818,
        "RF_CatBoost_chronological_PCE_top5_overlap": 5,
        "CatBoost_SHAP_additivity_max_residual": 4.973799150320701e-14,
        "CatBoost_hierarchical_child_parent_max_difference": 3.907985046680551e-14,
    }
    (args.output_dir / "key_results.json").write_text(json.dumps(key, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
