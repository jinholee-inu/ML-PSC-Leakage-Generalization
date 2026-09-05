#!/usr/bin/env python3
"""Leakage-controlled grouped SHAP and ALE analysis for CatBoost PSC prediction.

The selected full 1/n_DOI weighted CatBoost model is refitted under the frozen
DOI-grouped and chronological partitions. Explanations are calculated only for
held-out records. Because the execution environment intentionally has no SHAP
dependency, this script computes interventional Monte-Carlo Shapley values for
11 scientifically defined parent feature blocks. Continuous-variable ALE is
calculated using training-derived quantile bins and held-out DOI-balanced
records. The implementation verifies Shapley additivity and exact agreement of
refitted predictions with the archived weighting analysis.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import re
import sys
import time
import warnings
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from scipy import sparse
from scipy.stats import spearmanr
import catboost
from catboost import CatBoostRegressor, Pool

warnings.filterwarnings(
    "ignore",
    message=r"`sklearn\.utils\.parallel\.delayed` should be used.*",
    category=UserWarning,
)
warnings.filterwarnings("ignore", category=PendingDeprecationWarning)


SCRIPT_DIR = Path(__file__).resolve().parent
BASELINE_CODE_DIR = SCRIPT_DIR / "baseline-code"
sys.path.insert(0, str(BASELINE_CODE_DIR))

from psc_baseline_validation import (  # noqa: E402
    CATEGORICAL_FEATURES,
    RAW_REQUIRED,
    TARGETS,
    ModelConfig,
    build_features,
    make_preprocessor,
    normalize_doi,
    sha256,
)


GROUPED_SCHEME = "DOI-grouped 5-fold"
CHRONO_SCHEME = "Chronological >2018"
FULL_WEIGHTING = "Full 1/n_DOI"
MODEL = "CatBoost"
CONDITION = "CatBoost | Full DOI"
FAMILY_ORDER = [
    "Absorber composition",
    "Bandgap",
    "Absorber thickness",
    "Device architecture",
    "Substrate and area",
    "Electron-transport layer",
    "Hole-transport layer",
    "Back contact",
    "Deposition route",
    "Thermal processing",
    "Solvent/additive/quench",
]
ALE_FEATURE_LABELS = {
    "bandgap_eV": "Bandgap",
    "thickness_mean_nm": "Absorber thickness",
    "anneal_temp_mean_C": "Annealing temperature",
    "anneal_time_mean_min": "Annealing time",
    "substrate_temp_mean_C": "Substrate temperature",
    "log10_cell_area_cm2": "Log cell area",
    "comp_A_Cs_fraction": "Cs fraction at A site",
    "comp_B_Sn_fraction": "Sn fraction at B site",
    "comp_X_Br_fraction": "Br fraction at X site",
}
ALE_FEATURES = list(ALE_FEATURE_LABELS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", required=True, type=Path)
    parser.add_argument("--cohort", required=True, type=Path)
    parser.add_argument("--baseline-results-dir", required=True, type=Path)
    parser.add_argument("--weighting-results-dir", required=True, type=Path)
    parser.add_argument("--catboost-model-selection", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def training_weights(dois: pd.Series) -> np.ndarray:
    counts = dois.value_counts()
    raw = dois.map(counts).to_numpy(dtype=float) ** -1.0
    return raw / raw.mean()


def weighted_target_location_scale(
    y_train: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    mean = np.average(y_train, axis=0, weights=weights)
    var = np.average((y_train - mean) ** 2, axis=0, weights=weights)
    std = np.sqrt(np.maximum(var, 0.0))
    return mean, np.where(std > 0, std, 1.0)


def as_model_matrix(matrix: object) -> object:
    if sparse.issparse(matrix):
        return matrix.tocsr().astype(np.float32)
    return np.asarray(matrix, dtype=np.float32)


def model_predict(
    forest: CatBoostRegressor,
    matrix: object,
    y_mean: np.ndarray,
    y_std: np.ndarray,
) -> np.ndarray:
    return forest.predict(matrix) * y_std + y_mean


def selected_catboost_config(
    selection: pd.DataFrame, scheme: str, fold: str
) -> tuple[dict[str, float | int | str], int]:
    context = (
        f"outer_grouped_fold_{int(str(fold).split('_')[-1])}"
        if scheme == GROUPED_SCHEME
        else "chronological_train_through_2018"
    )
    rows = selection.loc[selection["context"].eq(context)]
    if rows.empty:
        raise AssertionError(f"Missing CatBoost selection for {context}")
    selected_name = str(rows["selected_candidate"].iloc[0])
    chosen = rows.loc[rows["candidate"].eq(selected_name)].iloc[0]
    candidate = {
        "candidate": selected_name,
        "depth": int(selected_name.split("_")[0].replace("depth", "")),
        "learning_rate": float(selected_name.split("_")[1].replace("lr", "")),
        "l2_leaf_reg": float(selected_name.split("_")[2].replace("l2-", "")),
    }
    return candidate, int(chosen["selected_final_iterations"])


def build_catboost_model(
    candidate: dict[str, float | int | str], iterations: int, random_state: int
) -> CatBoostRegressor:
    return CatBoostRegressor(
        loss_function="MultiRMSE",
        eval_metric="MultiRMSE",
        iterations=int(iterations),
        depth=int(candidate["depth"]),
        learning_rate=float(candidate["learning_rate"]),
        l2_leaf_reg=float(candidate["l2_leaf_reg"]),
        bootstrap_type="Bernoulli",
        subsample=0.80,
        rsm=0.50,
        random_strength=1.0,
        random_seed=int(random_state),
        thread_count=-1,
        allow_writing_files=False,
        verbose=False,
    )


def choose_one_record_per_doi(
    indices: np.ndarray,
    metadata: pd.DataFrame,
    maximum: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    frame = metadata.iloc[indices][["doi_norm"]].copy()
    frame["row_index"] = indices
    selected: list[int] = []
    for _doi, group in frame.groupby("doi_norm", sort=True):
        candidates = group["row_index"].to_numpy(dtype=int)
        selected.append(int(rng.choice(candidates)))
    selected_arr = np.asarray(selected, dtype=int)
    if len(selected_arr) > maximum:
        selected_arr = rng.choice(selected_arr, size=maximum, replace=False)
    return np.sort(selected_arr)


def family_for_feature(name: str) -> str:
    if name.startswith("materials__"):
        token = name[len("materials__") :]
        prefix = token.split("__", 1)[0]
        return {
            "a": "Absorber composition",
            "b": "Absorber composition",
            "x": "Absorber composition",
            "sub": "Substrate and area",
            "etl": "Electron-transport layer",
            "htl": "Hole-transport layer",
            "back": "Back contact",
            "solv": "Solvent/additive/quench",
            "add": "Solvent/additive/quench",
            "quench": "Solvent/additive/quench",
        }[prefix]

    if name.startswith("categorical__"):
        raw = name[len("categorical__") :]
        for column in sorted(CATEGORICAL_FEATURES, key=len, reverse=True):
            if raw == column or raw.startswith(column + "_"):
                return {
                    "architecture": "Device architecture",
                    "absorber_short_form": "Absorber composition",
                    "substrate_stack": "Substrate and area",
                    "etl_stack": "Electron-transport layer",
                    "htl_stack": "Hole-transport layer",
                    "backcontact_stack": "Back contact",
                    "absorber_deposition": "Deposition route",
                    "absorber_atmosphere": "Deposition route",
                    "annealing_atmosphere": "Thermal processing",
                    "quenching_medium": "Solvent/additive/quench",
                    "solvent_system": "Solvent/additive/quench",
                }[column]
        raise KeyError(f"Unmapped categorical feature: {name}")

    if not name.startswith("numeric__"):
        raise KeyError(f"Unknown encoded feature: {name}")
    raw = name[len("numeric__") :].replace("missingindicator_", "")
    if raw in {
        "is_inorganic",
        "is_leadfree",
        "dimension_0D",
        "dimension_2D",
        "dimension_2D3D",
        "dimension_3D",
        "dimension_3D_2Dcap",
    } or raw.startswith("comp_"):
        return "Absorber composition"
    if raw == "bandgap_eV":
        return "Bandgap"
    if raw.startswith("thickness_"):
        return "Absorber thickness"
    if raw in {"cell_area_cm2", "log10_cell_area_cm2", "substrate_layer_count"}:
        return "Substrate and area"
    if raw == "etl_layer_count":
        return "Electron-transport layer"
    if raw == "htl_layer_count":
        return "Hole-transport layer"
    if raw == "backcontact_layer_count":
        return "Back contact"
    if raw == "deposition_step_count":
        return "Deposition route"
    if raw.startswith("anneal_") or raw.startswith("substrate_temp_"):
        return "Thermal processing"
    if raw in {"solvent_count", "additive_count", "quench_count"}:
        return "Solvent/additive/quench"
    raise KeyError(f"Unmapped numeric feature: {name}")


def family_columns(feature_names: np.ndarray) -> dict[str, np.ndarray]:
    mapping: dict[str, list[int]] = {name: [] for name in FAMILY_ORDER}
    for index, feature in enumerate(feature_names):
        mapping[family_for_feature(str(feature))].append(index)
    output = {key: np.asarray(value, dtype=int) for key, value in mapping.items()}
    assigned = np.concatenate(list(output.values()))
    if len(assigned) != len(feature_names) or len(np.unique(assigned)) != len(feature_names):
        raise AssertionError("Encoded features were not assigned to exactly one family")
    if any(not len(value) for value in output.values()):
        raise AssertionError("Every scientific feature family must contain encoded columns")
    return output


def monte_carlo_group_shap(
    forest: CatBoostRegressor,
    explain_matrix: object,
    background_matrix: object,
    columns: dict[str, np.ndarray],
    y_mean: np.ndarray,
    y_std: np.ndarray,
    permutations: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    rng = np.random.default_rng(seed)
    x = explain_matrix.toarray() if sparse.issparse(explain_matrix) else np.asarray(explain_matrix)
    background = (
        background_matrix.toarray()
        if sparse.issparse(background_matrix)
        else np.asarray(background_matrix)
    )
    x = np.asarray(x, dtype=np.float32)
    background = np.asarray(background, dtype=np.float32)
    n_rows = x.shape[0]
    n_groups = len(FAMILY_ORDER)
    n_targets = len(TARGETS)
    shap_values = np.zeros((n_rows, n_groups, n_targets), dtype=np.float64)
    shap_squares = np.zeros((n_rows, n_groups, n_targets), dtype=np.float64)
    base_values = np.zeros((n_rows, n_targets), dtype=np.float64)
    final_prediction = model_predict(forest, x, y_mean, y_std)
    final_path_difference = 0.0

    for _replicate in range(permutations):
        background_index = rng.integers(0, len(background), size=n_rows)
        current = background[background_index].copy()
        row_permutations = np.vstack([rng.permutation(n_groups) for _ in range(n_rows)])
        states = [current.copy()]
        for step in range(n_groups):
            for group_index, family in enumerate(FAMILY_ORDER):
                rows = np.flatnonzero(row_permutations[:, step] == group_index)
                if not len(rows):
                    continue
                cols = columns[family]
                current[np.ix_(rows, cols)] = x[np.ix_(rows, cols)]
            states.append(current.copy())
        stacked = np.vstack(states).astype(np.float32, copy=False)
        prediction = model_predict(forest, stacked, y_mean, y_std).reshape(
            n_groups + 1, n_rows, n_targets
        )
        base_values += prediction[0]
        final_path_difference = max(
            final_path_difference,
            float(np.max(np.abs(prediction[-1] - final_prediction))),
        )
        for step in range(n_groups):
            delta = prediction[step + 1] - prediction[step]
            for group_index in range(n_groups):
                rows = np.flatnonzero(row_permutations[:, step] == group_index)
                shap_values[rows, group_index, :] += delta[rows]
                shap_squares[rows, group_index, :] += delta[rows] ** 2

    shap_values /= float(permutations)
    if permutations > 1:
        variance = np.maximum(
            (shap_squares - permutations * shap_values**2) / (permutations - 1),
            0.0,
        )
        shap_mcse = np.sqrt(variance / permutations)
    else:
        shap_mcse = np.full_like(shap_values, np.nan)
    base_values /= float(permutations)
    residual = final_prediction - (base_values + shap_values.sum(axis=1))
    return (
        shap_values,
        shap_mcse,
        base_values,
        final_prediction,
        float(np.max(np.abs(residual))),
        final_path_difference,
    )


def archived_prediction_difference(
    archived: pd.DataFrame,
    scheme: str,
    fold: str,
    metadata: pd.DataFrame,
    test_index: np.ndarray,
    prediction: np.ndarray,
) -> float:
    rows: list[pd.DataFrame] = []
    for target_index, target in enumerate(TARGETS):
        rows.append(
            pd.DataFrame(
                {
                    "Ref_ID": metadata.iloc[test_index]["Ref_ID"].to_numpy(),
                    "scheme": scheme,
                    "fold": fold,
                    "target": target,
                    "y_pred_refit": prediction[:, target_index],
                }
            )
        )
    refit = pd.concat(rows, ignore_index=True)
    frozen = archived.loc[
        archived["scheme"].eq(scheme) & archived["fold"].astype(str).eq(str(fold)),
        ["Ref_ID", "scheme", "fold", "target", "y_pred"],
    ].copy()
    merged = refit.merge(
        frozen,
        on=["Ref_ID", "scheme", "fold", "target"],
        how="left",
        validate="one_to_one",
    )
    if merged["y_pred"].isna().any():
        raise AssertionError("Archived prediction alignment failed")
    return float(np.max(np.abs(merged["y_pred_refit"] - merged["y_pred"])))


def ale_for_partition(
    processor,
    forest: CatBoostRegressor,
    features: pd.DataFrame,
    metadata: pd.DataFrame,
    train_index: np.ndarray,
    explain_index: np.ndarray,
    feature: str,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    bins: int,
    scheme: str,
    fold: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_values = pd.to_numeric(features.iloc[train_index][feature], errors="coerce")
    train_values = train_values[np.isfinite(train_values)]
    if len(train_values) < 50 or train_values.nunique() < 5:
        return pd.DataFrame(), pd.DataFrame()
    # Restrict ALE to the central training support so a single extreme maximum
    # does not define an unrealistically broad terminal bin.
    edges = np.unique(np.quantile(train_values, np.linspace(0.025, 0.975, bins + 1)))
    if len(edges) < 4:
        return pd.DataFrame(), pd.DataFrame()

    explain = features.iloc[explain_index].copy()
    values = pd.to_numeric(explain[feature], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(values) & (values >= edges[0]) & (values <= edges[-1])
    if valid.sum() < max(20, len(edges) * 2):
        return pd.DataFrame(), pd.DataFrame()
    selected_index = explain_index[valid]
    selected = explain.loc[valid].copy()
    selected_values = values[valid]
    bin_index = np.searchsorted(edges, selected_values, side="right") - 1
    bin_index = np.clip(bin_index, 0, len(edges) - 2)

    low = edges[bin_index]
    high = edges[bin_index + 1]
    low_frame = selected.copy()
    high_frame = selected.copy()
    low_frame[feature] = low
    high_frame[feature] = high
    low_matrix = as_model_matrix(processor.transform(low_frame))
    high_matrix = as_model_matrix(processor.transform(high_frame))
    delta = model_predict(forest, high_matrix, y_mean, y_std) - model_predict(
        forest, low_matrix, y_mean, y_std
    )

    individual_rows: list[dict[str, object]] = []
    curve_rows: list[dict[str, object]] = []
    for target_index, target in enumerate(TARGETS):
        bin_effect = np.full(len(edges) - 1, np.nan, dtype=float)
        counts = np.zeros(len(edges) - 1, dtype=int)
        for bin_number in range(len(edges) - 1):
            mask = bin_index == bin_number
            counts[bin_number] = int(mask.sum())
            if mask.any():
                bin_effect[bin_number] = float(np.mean(delta[mask, target_index]))
        filled = pd.Series(bin_effect).interpolate(limit_direction="both").fillna(0.0).to_numpy()
        accumulated = np.cumsum(filled)
        row_ale = accumulated[bin_index]
        center = float(np.mean(row_ale))
        centered = accumulated - center
        for bin_number in range(len(edges) - 1):
            curve_rows.append(
                {
                    "scheme": scheme,
                    "fold": fold,
                    "target": target,
                    "feature": feature,
                    "feature_label": ALE_FEATURE_LABELS[feature],
                    "bin": bin_number + 1,
                    "quantile_midpoint": (bin_number + 0.5) / (len(edges) - 1),
                    "x_low": float(edges[bin_number]),
                    "x_high": float(edges[bin_number + 1]),
                    "x_mid": float((edges[bin_number] + edges[bin_number + 1]) / 2.0),
                    "n_records": int(counts[bin_number]),
                    "local_effect": float(filled[bin_number]),
                    "ALE": float(centered[bin_number]),
                }
            )
        for row_position, original_index in enumerate(selected_index):
            individual_rows.append(
                {
                    "Ref_ID": metadata.iloc[original_index]["Ref_ID"],
                    "doi_norm": metadata.iloc[original_index]["doi_norm"],
                    "scheme": scheme,
                    "fold": fold,
                    "target": target,
                    "feature": feature,
                    "feature_value": float(selected_values[row_position]),
                    "bin": int(bin_index[row_position] + 1),
                    "delta": float(delta[row_position, target_index]),
                }
            )
    return pd.DataFrame(curve_rows), pd.DataFrame(individual_rows)


def importance_summaries(
    local: pd.DataFrame, bootstrap_replicates: int, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fold_rows: list[dict[str, object]] = []
    for (scheme, fold, target, family), group in local.groupby(
        ["scheme", "fold", "target", "family"], sort=False
    ):
        fold_rows.append(
            {
                "scheme": scheme,
                "fold": fold,
                "target": target,
                "family": family,
                "n_explained_DOI": int(group["doi_norm"].nunique()),
                "mean_abs_SHAP": float(group["abs_shap_value"].mean()),
                "mean_signed_SHAP": float(group["shap_value"].mean()),
                "positive_fraction": float((group["shap_value"] > 0).mean()),
            }
        )
    fold_frame = pd.DataFrame(fold_rows)
    fold_frame["rank"] = fold_frame.groupby(["scheme", "fold", "target"])[
        "mean_abs_SHAP"
    ].rank(method="min", ascending=False)

    rng = np.random.default_rng(seed)
    summary_rows: list[dict[str, object]] = []
    for (scheme, target), group in local.groupby(["scheme", "target"], sort=False):
        pivot = group.pivot(index="doi_norm", columns="family", values="abs_shap_value")
        pivot = pivot.reindex(columns=FAMILY_ORDER)
        signed = group.pivot(index="doi_norm", columns="family", values="shap_value").reindex(
            columns=FAMILY_ORDER
        )
        point = pivot.mean(axis=0)
        boot = np.empty((bootstrap_replicates, len(FAMILY_ORDER)), dtype=float)
        values = pivot.to_numpy(dtype=float)
        for replicate in range(bootstrap_replicates):
            sampled = rng.integers(0, len(values), size=len(values))
            boot[replicate] = np.nanmean(values[sampled], axis=0)
        lower = np.nanquantile(boot, 0.025, axis=0)
        upper = np.nanquantile(boot, 0.975, axis=0)
        total = float(point.sum())
        for family_index, family in enumerate(FAMILY_ORDER):
            summary_rows.append(
                {
                    "scheme": scheme,
                    "target": target,
                    "family": family,
                    "n_explained_DOI": int(len(pivot)),
                    "mean_abs_SHAP": float(point[family]),
                    "mean_abs_SHAP_CI_low": float(lower[family_index]),
                    "mean_abs_SHAP_CI_high": float(upper[family_index]),
                    "relative_importance_percent": float(100.0 * point[family] / total),
                    "mean_signed_SHAP": float(signed[family].mean()),
                    "positive_fraction": float((signed[family] > 0).mean()),
                }
            )
    summary = pd.DataFrame(summary_rows)
    summary["rank"] = summary.groupby(["scheme", "target"])["mean_abs_SHAP"].rank(
        method="min", ascending=False
    )
    return fold_frame, summary


def rank_stability(fold_importance: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    grouped = fold_importance.loc[fold_importance["scheme"].eq(GROUPED_SCHEME)]
    for target in TARGETS:
        target_frame = grouped.loc[grouped["target"].eq(target)]
        folds = sorted(target_frame["fold"].unique())
        for left, right in combinations(folds, 2):
            a = target_frame.loc[target_frame["fold"].eq(left)].set_index("family")
            b = target_frame.loc[target_frame["fold"].eq(right)].set_index("family")
            rho = spearmanr(a.loc[FAMILY_ORDER, "mean_abs_SHAP"], b.loc[FAMILY_ORDER, "mean_abs_SHAP"]).statistic
            top_a = set(a.nsmallest(5, "rank").index)
            top_b = set(b.nsmallest(5, "rank").index)
            rows.append(
                {
                    "target": target,
                    "comparison": "grouped_fold_pair",
                    "left": left,
                    "right": right,
                    "spearman_rho": float(rho),
                    "top5_overlap": int(len(top_a & top_b)),
                    "top5_jaccard": float(len(top_a & top_b) / len(top_a | top_b)),
                }
            )
        a = summary.loc[
            summary["scheme"].eq(GROUPED_SCHEME) & summary["target"].eq(target)
        ].set_index("family")
        b = summary.loc[
            summary["scheme"].eq(CHRONO_SCHEME) & summary["target"].eq(target)
        ].set_index("family")
        rho = spearmanr(a.loc[FAMILY_ORDER, "mean_abs_SHAP"], b.loc[FAMILY_ORDER, "mean_abs_SHAP"]).statistic
        top_a = set(a.nsmallest(5, "rank").index)
        top_b = set(b.nsmallest(5, "rank").index)
        rows.append(
            {
                "target": target,
                "comparison": "grouped_aggregate_vs_chronological",
                "left": GROUPED_SCHEME,
                "right": CHRONO_SCHEME,
                "spearman_rho": float(rho),
                "top5_overlap": int(len(top_a & top_b)),
                "top5_jaccard": float(len(top_a & top_b) / len(top_a | top_b)),
            }
        )
    return pd.DataFrame(rows)


def ale_rankings(curves: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (scheme, fold, target, feature), group in curves.groupby(
        ["scheme", "fold", "target", "feature"], sort=False
    ):
        rows.append(
            {
                "scheme": scheme,
                "fold": fold,
                "target": target,
                "feature": feature,
                "feature_label": ALE_FEATURE_LABELS[feature],
                "ALE_range": float(group["ALE"].max() - group["ALE"].min()),
                "ALE_max_abs": float(group["ALE"].abs().max()),
                "n_bins": int(len(group)),
                "n_records": int(group["n_records"].sum()),
            }
        )
    frame = pd.DataFrame(rows)
    frame["rank"] = frame.groupby(["scheme", "fold", "target"])["ALE_range"].rank(
        method="min", ascending=False
    )
    return frame


def bootstrap_chronological_pce_ale(
    individual: pd.DataFrame,
    curves: pd.DataFrame,
    top_features: list[str],
    replicates: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for feature in top_features:
        ind = individual.loc[
            individual["scheme"].eq(CHRONO_SCHEME)
            & individual["target"].eq("PCE")
            & individual["feature"].eq(feature)
        ].copy()
        curve = curves.loc[
            curves["scheme"].eq(CHRONO_SCHEME)
            & curves["target"].eq("PCE")
            & curves["feature"].eq(feature)
        ].sort_values("bin")
        bins = curve["bin"].to_numpy(dtype=int)
        original = curve["ALE"].to_numpy(dtype=float)
        boot = np.empty((replicates, len(bins)), dtype=float)
        values = ind[["bin", "delta"]].to_numpy(dtype=float)
        for replicate in range(replicates):
            sampled = values[rng.integers(0, len(values), size=len(values))]
            effects = np.full(len(bins), np.nan, dtype=float)
            counts = np.zeros(len(bins), dtype=float)
            for index, bin_number in enumerate(bins):
                mask = sampled[:, 0] == bin_number
                counts[index] = mask.sum()
                if mask.any():
                    effects[index] = sampled[mask, 1].mean()
            effects = pd.Series(effects).interpolate(limit_direction="both").fillna(0.0).to_numpy()
            accumulated = np.cumsum(effects)
            sampled_bins = sampled[:, 0].astype(int) - 1
            center = float(np.mean(accumulated[sampled_bins]))
            boot[replicate] = accumulated - center
        lower = np.quantile(boot, 0.025, axis=0)
        upper = np.quantile(boot, 0.975, axis=0)
        for index, record in curve.reset_index(drop=True).iterrows():
            rows.append(
                {
                    **record.to_dict(),
                    "ALE_CI_low": float(lower[index]),
                    "ALE_CI_high": float(upper[index]),
                    "ALE_original_check": float(original[index]),
                }
            )
    return pd.DataFrame(rows)


def make_figure(
    output_dir: Path,
    local: pd.DataFrame,
    summary: pd.DataFrame,
    stability: pd.DataFrame,
    ale_bootstrap: pd.DataFrame,
) -> None:
    sns.set_theme(style="whitegrid", context="paper")
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 9.6))

    chrono = summary.loc[summary["scheme"].eq(CHRONO_SCHEME)].copy()
    heat = chrono.pivot(index="family", columns="target", values="relative_importance_percent")
    heat = heat.reindex(index=FAMILY_ORDER, columns=list(TARGETS))
    sns.heatmap(
        heat,
        ax=axes[0, 0],
        cmap="Blues",
        annot=True,
        fmt=".1f",
        cbar_kws={"label": "Relative mean |SHAP| (%)"},
        linewidths=0.4,
    )
    axes[0, 0].set_title("(a) Chronological feature-family importance", loc="left", fontweight="bold")
    axes[0, 0].set_xlabel("Prediction target")
    axes[0, 0].set_ylabel("")

    pce_summary = chrono.loc[chrono["target"].eq("PCE")].nsmallest(6, "rank")
    top_families = pce_summary.sort_values("mean_abs_SHAP")["family"].tolist()
    pce_local = local.loc[
        local["scheme"].eq(CHRONO_SCHEME)
        & local["target"].eq("PCE")
        & local["family"].isin(top_families)
    ].copy()
    sns.boxplot(
        data=pce_local,
        x="shap_value",
        y="family",
        order=top_families,
        ax=axes[0, 1],
        color="#5B8DB8",
        showfliers=False,
        width=0.65,
    )
    axes[0, 1].axvline(0.0, color="0.25", linestyle="--", linewidth=1)
    axes[0, 1].set_title("(b) Signed PCE contribution in future publications", loc="left", fontweight="bold")
    axes[0, 1].set_xlabel("Grouped SHAP value (PCE percentage points)")
    axes[0, 1].set_ylabel("")

    palette = sns.color_palette("colorblind", n_colors=max(3, ale_bootstrap["feature"].nunique()))
    for color, (feature, group) in zip(palette, ale_bootstrap.groupby("feature", sort=False)):
        group = group.sort_values("quantile_midpoint")
        label = ALE_FEATURE_LABELS[feature]
        axes[1, 0].plot(group["quantile_midpoint"], group["ALE"], marker="o", label=label, color=color)
        axes[1, 0].fill_between(
            group["quantile_midpoint"],
            group["ALE_CI_low"],
            group["ALE_CI_high"],
            color=color,
            alpha=0.16,
            linewidth=0,
        )
    axes[1, 0].axhline(0.0, color="0.25", linestyle="--", linewidth=1)
    axes[1, 0].set_title("(c) Chronological PCE accumulated local effects", loc="left", fontweight="bold")
    axes[1, 0].set_xlabel("Training-distribution quantile")
    axes[1, 0].set_ylabel("ALE (PCE percentage points)")
    axes[1, 0].legend(frameon=False, fontsize=8)

    grouped_pce = summary.loc[
        summary["scheme"].eq(GROUPED_SCHEME) & summary["target"].eq("PCE")
    ].set_index("family")
    chrono_pce = summary.loc[
        summary["scheme"].eq(CHRONO_SCHEME) & summary["target"].eq("PCE")
    ].set_index("family")
    axes[1, 1].scatter(
        grouped_pce.loc[FAMILY_ORDER, "rank"],
        chrono_pce.loc[FAMILY_ORDER, "rank"],
        s=48,
        color="#C65046",
    )
    abbreviations = {
        "Absorber composition": "Composition",
        "Bandgap": "Bandgap",
        "Absorber thickness": "Thickness",
        "Device architecture": "Architecture",
        "Substrate and area": "Substrate/area",
        "Electron-transport layer": "ETL",
        "Hole-transport layer": "HTL",
        "Back contact": "Back contact",
        "Deposition route": "Deposition",
        "Thermal processing": "Thermal",
        "Solvent/additive/quench": "Solvent/additive",
    }
    for family in FAMILY_ORDER:
        axes[1, 1].annotate(
            abbreviations[family],
            (grouped_pce.loc[family, "rank"], chrono_pce.loc[family, "rank"]),
            xytext=(4, 3),
            textcoords="offset points",
            fontsize=7,
        )
    axes[1, 1].plot([0.5, 11.5], [0.5, 11.5], linestyle="--", color="0.35", linewidth=1)
    rho = stability.loc[
        stability["target"].eq("PCE")
        & stability["comparison"].eq("grouped_aggregate_vs_chronological"),
        "spearman_rho",
    ].iloc[0]
    axes[1, 1].text(0.04, 0.84, f"Spearman rho = {rho:.2f}", transform=axes[1, 1].transAxes, va="top")
    axes[1, 1].set_xlim(0.5, 11.5)
    axes[1, 1].set_ylim(11.5, 0.5)
    axes[1, 1].set_title("(d) PCE importance-rank stability", loc="left", fontweight="bold")
    axes[1, 1].set_xlabel("DOI-grouped rank (1 = highest)")
    axes[1, 1].set_ylabel("Chronological rank (1 = highest)")

    fig.suptitle("Publication-disjoint explainability of perovskite solar-cell prediction", fontsize=15, fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(output_dir / "Figure6_CatBoost_SHAP_ALE_explainability.png", dpi=600, bbox_inches="tight")
    fig.savefig(output_dir / "Figure6_CatBoost_SHAP_ALE_explainability.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "Figure6_CatBoost_SHAP_ALE_explainability.svg", bbox_inches="tight")
    plt.close(fig)


def build_report(
    summary: pd.DataFrame,
    stability: pd.DataFrame,
    ale_ranking: pd.DataFrame,
    verification: dict[str, object],
) -> str:
    chrono_pce = summary.loc[
        summary["scheme"].eq(CHRONO_SCHEME) & summary["target"].eq("PCE")
    ].sort_values("rank")
    top = chrono_pce.head(5)
    pce_stability = stability.loc[
        stability["target"].eq("PCE")
        & stability["comparison"].eq("grouped_aggregate_vs_chronological")
    ].iloc[0]
    chrono_ale = ale_ranking.loc[
        ale_ranking["scheme"].eq(CHRONO_SCHEME) & ale_ranking["target"].eq("PCE")
    ].sort_values("rank")
    top_lines = "\n".join(
        f"- {row.family}: {row.relative_importance_percent:.1f}% of total mean absolute grouped SHAP "
        f"(mean |SHAP| {row.mean_abs_SHAP:.3f} PCE percentage points; 95% CI {row.mean_abs_SHAP_CI_low:.3f}-{row.mean_abs_SHAP_CI_high:.3f})."
        for row in top.itertuples()
    )
    ale_lines = "\n".join(
        f"- {row.feature_label}: ALE range {row.ALE_range:.3f} PCE percentage points."
        for row in chrono_ale.head(5).itertuples()
    )
    return f"""# PSC SHAP and ALE explainability analysis

## Scope

The analysis retained the frozen 33,175-record cohort, 6,368 normalized DOI groups, feature pipeline, validation partitions, and selected full `1/n_DOI` weighted CatBoost model. Every explanation was calculated for a held-out record. Each explained record represented one DOI, so global explanation summaries were publication balanced by construction.

SHAP values were estimated by interventional Monte-Carlo Shapley sampling over 11 parent feature blocks. Every replicate began from a randomly selected one-record-per-DOI training background and added feature blocks in a random order. This approach preserves complete one-hot and material-token blocks, but it remains an interventional association analysis and may create hybrid combinations not jointly observed in the literature.

ALE curves used quantile bins spanning the central 95% of the relevant training-partition distribution. Lower- and upper-bin predictions were evaluated on held-out DOI-balanced records that fell within those training-supported bounds. Thus, ALE describes local model response within observed ranges and should not be interpreted as a causal material or process effect.

## Chronological PCE results

{top_lines}

The PCE feature-family ranking had Spearman rho = **{pce_stability.spearman_rho:.3f}** between the DOI-grouped aggregate and chronological holdout, with **{int(pce_stability.top5_overlap)} of 5** top families shared. This quantifies whether the explanation remained stable when the target changed from an unseen publication to a future publication.

The strongest continuous-variable ALE amplitudes in the chronological cohort were:

{ale_lines}

## Interpretation

The reported quantities are model-based statistical associations. A large SHAP magnitude means that a feature block materially changed the fitted model prediction relative to a DOI-balanced training background; it does not establish that experimentally changing that feature will produce the same photovoltaic response. ALE reduces extrapolative perturbation by comparing adjacent observed bins, but correlated composition, architecture, and processing choices remain inseparable in this observational literature dataset.

The appropriate manuscript use is therefore to emphasize robust feature families that remain highly ranked under both DOI-grouped and chronological validation, while treating fold-sensitive or temporally unstable rankings as evidence that the model explanation depends on the literature domain. The ALE curves should be described as local response profiles, not optimization prescriptions.

## Integrity checks

- Explained DOI groups: {verification['explained_DOI_groups']:,}
- Local SHAP rows: {verification['local_SHAP_rows']:,}
- Maximum archived prediction difference: {verification['max_archived_prediction_difference']:.3e}
- Maximum SHAP additivity residual: {verification['max_SHAP_additivity_residual']:.3e}
- Maximum terminal-path prediction difference: {verification['max_terminal_path_prediction_difference']:.3e}
- Median local SHAP Monte-Carlo standard error: {verification['median_local_SHAP_MCSE']:.3f}
- 95th percentile local SHAP Monte-Carlo standard error: {verification['p95_local_SHAP_MCSE']:.3f}
- Unmapped encoded features: {verification['unmapped_encoded_features']}
- Grouped train-test DOI overlap: {verification['grouped_boundary_DOI_overlap']}
- Chronological train-test DOI overlap: {verification['chronological_boundary_DOI_overlap']}
- DOI-cluster bootstrap replicates: {verification['bootstrap_replicates']}
"""


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    config = ModelConfig()
    explain_per_grouped_fold = 150
    explain_chronological = 300
    background_dois = 400
    shap_permutations = 48
    ale_bins = 8
    bootstrap_replicates = 1000
    if args.quick:
        config = ModelConfig(
            grouped_folds=2,
            bootstrap_replicates=30,
            token_min_df=20,
            token_max_features=800,
            rf_estimators=20,
        )
        explain_per_grouped_fold = 35
        explain_chronological = 50
        background_dois = 60
        shap_permutations = 4
        ale_bins = 4
        bootstrap_replicates = 30

    baseline_manifest = json.loads(
        (args.baseline_results_dir / "run_manifest.json").read_text(encoding="utf-8")
    )
    if not args.quick:
        if sha256(args.raw) != baseline_manifest["inputs"]["raw_sha256"]:
            raise AssertionError("Raw snapshot hash differs from frozen baseline")
        if sha256(args.cohort) != baseline_manifest["inputs"]["cohort_sha256"]:
            raise AssertionError("Cohort hash differs from frozen baseline")

    raw = pd.read_csv(args.raw, usecols=RAW_REQUIRED, low_memory=False)
    cohort = pd.read_csv(args.cohort, low_memory=False)
    raw = cohort[["Ref_ID"]].merge(raw, on="Ref_ID", how="left", validate="one_to_one")
    metadata = pd.DataFrame(
        {
            "Ref_ID": raw["Ref_ID"],
            "doi_norm": normalize_doi(raw["Ref_DOI_number"]),
            "publication_year": pd.to_datetime(raw["Ref_publication_date"], errors="raise").dt.year,
        }
    )
    targets = pd.DataFrame(index=raw.index)
    for target, (source, _unit) in TARGETS.items():
        targets[target] = pd.to_numeric(raw[source], errors="raise")
    targets["FF"] = targets["FF"] * 100.0
    features, numeric_features = build_features(raw)

    split_manifest = pd.read_csv(args.baseline_results_dir / "split_manifest.csv")
    if not split_manifest["Ref_ID"].equals(metadata["Ref_ID"]):
        raise AssertionError("Split manifest does not align to cohort")
    archived = pd.read_csv(
        args.weighting_results_dir / "catboost_recency_predictions.csv.gz"
    )
    archived = archived.loc[
        archived["scheme"].isin([GROUPED_SCHEME, CHRONO_SCHEME])
        & archived["training_weighting"].eq(FULL_WEIGHTING)
        & archived["model"].eq(MODEL)
        & archived["condition"].eq(CONDITION)
    ].copy()
    selection = pd.read_csv(args.catboost_model_selection)

    partitions: list[tuple[str, str, np.ndarray, np.ndarray, int]] = []
    grouped_folds = sorted(split_manifest["grouped_fold"].unique())
    if args.quick:
        grouped_folds = grouped_folds[:2]
    for fold_number in grouped_folds:
        test = np.flatnonzero(split_manifest["grouped_fold"].eq(fold_number).to_numpy())
        train = np.flatnonzero(split_manifest["grouped_fold"].ne(fold_number).to_numpy())
        partitions.append(
            (GROUPED_SCHEME, f"fold_{int(fold_number)}", train, test, config.seed + int(fold_number))
        )
    chrono_train = np.flatnonzero(
        split_manifest["chronological_role"].eq("train_through_2018").to_numpy()
    )
    chrono_test = np.flatnonzero(
        split_manifest["chronological_role"].eq("test_2019_onward").to_numpy()
    )
    partitions.append(
        (
            CHRONO_SCHEME,
            "holdout_2019_onward",
            chrono_train,
            chrono_test,
            config.seed + 2019,
        )
    )

    local_rows: list[dict[str, object]] = []
    ale_curve_frames: list[pd.DataFrame] = []
    ale_individual_frames: list[pd.DataFrame] = []
    diagnostics: list[dict[str, object]] = []
    archived_differences: list[float] = []
    additivity_residuals: list[float] = []
    path_differences: list[float] = []
    grouped_overlap = 0
    chronological_overlap = 0

    for partition_number, (scheme, fold, train_index, test_index, random_state) in enumerate(partitions):
        partition_started = time.perf_counter()
        max_explain = (
            explain_chronological if scheme == CHRONO_SCHEME else explain_per_grouped_fold
        )
        explain_index = choose_one_record_per_doi(
            test_index,
            metadata,
            max_explain,
            config.seed + 41000 + partition_number,
        )
        background_index = choose_one_record_per_doi(
            train_index,
            metadata,
            background_dois,
            config.seed + 42000 + partition_number,
        )
        train_doi = set(metadata.iloc[train_index]["doi_norm"])
        test_doi = set(metadata.iloc[test_index]["doi_norm"])
        overlap = len(train_doi & test_doi)
        if scheme == GROUPED_SCHEME:
            grouped_overlap = max(grouped_overlap, overlap)
        else:
            chronological_overlap = max(chronological_overlap, overlap)
        if overlap:
            raise AssertionError(f"DOI boundary overlap in {scheme} {fold}")

        print(
            f"[{scheme} {fold}] train={len(train_index):,}, test={len(test_index):,}, "
            f"explain DOI={len(explain_index):,}",
            flush=True,
        )
        processor = make_preprocessor(config, numeric_features)
        train_matrix = as_model_matrix(processor.fit_transform(features.iloc[train_index]))
        test_matrix = as_model_matrix(processor.transform(features.iloc[test_index]))
        explain_matrix = as_model_matrix(processor.transform(features.iloc[explain_index]))
        background_matrix = as_model_matrix(processor.transform(features.iloc[background_index]))
        y_train = targets.iloc[train_index].to_numpy(dtype=float)
        weights = training_weights(metadata.iloc[train_index]["doi_norm"].reset_index(drop=True))
        y_mean, y_std = weighted_target_location_scale(y_train, weights)
        y_scaled = (y_train - y_mean) / y_std
        candidate, iterations = selected_catboost_config(selection, scheme, fold)
        if args.quick:
            iterations = min(iterations, 60)
        forest = build_catboost_model(candidate, iterations, random_state)
        forest.fit(Pool(train_matrix, label=y_scaled, weight=weights))
        full_prediction = model_predict(forest, test_matrix, y_mean, y_std)
        archived_difference = archived_prediction_difference(
            archived, scheme, fold, metadata, test_index, full_prediction
        )
        archived_differences.append(archived_difference)

        encoded_names = processor.get_feature_names_out()
        columns = family_columns(encoded_names)
        shap_values, shap_mcse, base_values, explain_prediction, residual, path_difference = monte_carlo_group_shap(
            forest,
            explain_matrix,
            background_matrix,
            columns,
            y_mean,
            y_std,
            shap_permutations,
            config.seed + 43000 + partition_number,
        )
        additivity_residuals.append(residual)
        path_differences.append(path_difference)
        explain_targets = targets.iloc[explain_index].to_numpy(dtype=float)
        for row_position, original_index in enumerate(explain_index):
            for target_index, target in enumerate(TARGETS):
                for family_index, family in enumerate(FAMILY_ORDER):
                    value = float(shap_values[row_position, family_index, target_index])
                    local_rows.append(
                        {
                            "Ref_ID": metadata.iloc[original_index]["Ref_ID"],
                            "doi_norm": metadata.iloc[original_index]["doi_norm"],
                            "publication_year": int(metadata.iloc[original_index]["publication_year"]),
                            "scheme": scheme,
                            "fold": fold,
                            "target": target,
                            "family": family,
                            "shap_value": value,
                            "abs_shap_value": abs(value),
                            "shap_mcse": float(shap_mcse[row_position, family_index, target_index]),
                            "base_value": float(base_values[row_position, target_index]),
                            "y_pred": float(explain_prediction[row_position, target_index]),
                            "y_true": float(explain_targets[row_position, target_index]),
                        }
                    )

        for feature in ALE_FEATURES:
            curve, individual = ale_for_partition(
                processor,
                forest,
                features,
                metadata,
                train_index,
                explain_index,
                feature,
                y_mean,
                y_std,
                ale_bins,
                scheme,
                fold,
            )
            if not curve.empty:
                ale_curve_frames.append(curve)
                ale_individual_frames.append(individual)

        diagnostics.append(
            {
                "scheme": scheme,
                "fold": fold,
                "train_records": int(len(train_index)),
                "test_records": int(len(test_index)),
                "train_DOI": int(len(train_doi)),
                "test_DOI": int(len(test_doi)),
                "explained_DOI": int(len(explain_index)),
                "background_DOI": int(len(background_index)),
                "encoded_features": int(len(encoded_names)),
                "feature_families": int(len(columns)),
                "SHAP_permutations": int(shap_permutations),
                "catboost_candidate": candidate["candidate"],
                "catboost_iterations": int(iterations),
                "archived_prediction_max_difference": archived_difference,
                "SHAP_additivity_max_residual": residual,
                "terminal_path_max_difference": path_difference,
                "runtime_seconds": float(time.perf_counter() - partition_started),
            }
        )

    local = pd.DataFrame(local_rows)
    curves = pd.concat(ale_curve_frames, ignore_index=True)
    ale_individual = pd.concat(ale_individual_frames, ignore_index=True)
    fold_importance, summary = importance_summaries(
        local, bootstrap_replicates, config.seed + 50000
    )
    stability = rank_stability(fold_importance, summary)
    ale_ranking = ale_rankings(curves)
    chrono_pce_ranking = ale_ranking.loc[
        ale_ranking["scheme"].eq(CHRONO_SCHEME)
        & ale_ranking["target"].eq("PCE")
    ].sort_values("rank")
    top_ale_features = chrono_pce_ranking.head(3)["feature"].tolist()
    ale_bootstrap = bootstrap_chronological_pce_ale(
        ale_individual,
        curves,
        top_ale_features,
        bootstrap_replicates,
        config.seed + 51000,
    )

    mcse_rows: list[dict[str, object]] = []
    for (scheme, target), group in local.groupby(["scheme", "target"], sort=False):
        mcse_rows.append(
            {
                "scheme": scheme,
                "target": target,
                "n_local_values": int(len(group)),
                "median_SHAP_MCSE": float(group["shap_mcse"].median()),
                "p95_SHAP_MCSE": float(group["shap_mcse"].quantile(0.95)),
                "mean_SHAP_MCSE": float(group["shap_mcse"].mean()),
                "mean_abs_SHAP": float(group["abs_shap_value"].mean()),
                "mean_MCSE_to_mean_abs_SHAP": float(
                    group["shap_mcse"].mean() / group["abs_shap_value"].mean()
                ),
            }
        )
    mcse_summary = pd.DataFrame(mcse_rows)

    local.to_csv(args.output_dir / "shap_local_values.csv.gz", index=False, compression="gzip")
    fold_importance.to_csv(args.output_dir / "shap_family_importance_by_fold.csv", index=False)
    summary.to_csv(args.output_dir / "shap_family_importance_summary.csv", index=False)
    stability.to_csv(args.output_dir / "shap_rank_stability.csv", index=False)
    mcse_summary.to_csv(args.output_dir / "shap_monte_carlo_diagnostics.csv", index=False)
    curves.to_csv(args.output_dir / "ale_curves.csv", index=False)
    ale_individual.to_csv(
        args.output_dir / "ale_individual_local_effects.csv.gz", index=False, compression="gzip"
    )
    ale_ranking.to_csv(args.output_dir / "ale_feature_ranking.csv", index=False)
    ale_bootstrap.to_csv(args.output_dir / "ale_chronological_PCE_bootstrap.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(
        args.output_dir / "explainability_partition_diagnostics.csv", index=False
    )

    verification = {
        "status": "passed",
        "explained_DOI_groups": int(local["doi_norm"].nunique()),
        "local_SHAP_rows": int(len(local)),
        "duplicate_local_keys": int(
            local.duplicated(["Ref_ID", "scheme", "fold", "target", "family"]).sum()
        ),
        "max_archived_prediction_difference": float(max(archived_differences)),
        "max_SHAP_additivity_residual": float(max(additivity_residuals)),
        "max_terminal_path_prediction_difference": float(max(path_differences)),
        "median_local_SHAP_MCSE": float(np.nanmedian(local["shap_mcse"])),
        "p95_local_SHAP_MCSE": float(np.nanquantile(local["shap_mcse"], 0.95)),
        "chronological_PCE_median_SHAP_MCSE": float(
            mcse_summary.loc[
                mcse_summary["scheme"].eq(CHRONO_SCHEME)
                & mcse_summary["target"].eq("PCE"),
                "median_SHAP_MCSE",
            ].iloc[0]
        ),
        "chronological_PCE_p95_SHAP_MCSE": float(
            mcse_summary.loc[
                mcse_summary["scheme"].eq(CHRONO_SCHEME)
                & mcse_summary["target"].eq("PCE"),
                "p95_SHAP_MCSE",
            ].iloc[0]
        ),
        "unmapped_encoded_features": 0,
        "grouped_boundary_DOI_overlap": int(grouped_overlap),
        "chronological_boundary_DOI_overlap": int(chronological_overlap),
        "bootstrap_replicates": int(bootstrap_replicates),
        "SHAP_permutations": int(shap_permutations),
        "ALE_top_features": top_ale_features,
        "finite_SHAP_values": bool(np.isfinite(local["shap_value"]).all()),
        "finite_ALE_values": bool(np.isfinite(curves["ALE"]).all()),
    }
    if verification["duplicate_local_keys"] != 0:
        raise AssertionError("Duplicate local SHAP keys")
    if not verification["finite_SHAP_values"] or not verification["finite_ALE_values"]:
        raise AssertionError("Non-finite explanation values")
    if not args.quick and verification["max_archived_prediction_difference"] > 1e-8:
        raise AssertionError("Refitted prediction differs from archive")
    if verification["max_SHAP_additivity_residual"] > 1e-10:
        raise AssertionError("SHAP additivity check failed")

    make_figure(args.output_dir, local, summary, stability, ale_bootstrap)
    report = build_report(summary, stability, ale_ranking, verification)
    (args.output_dir / "PSC_CatBoost_SHAP_ALE_explainability_report.md").write_text(
        report, encoding="utf-8"
    )
    (args.output_dir / "explainability_verification_report.json").write_text(
        json.dumps(verification, indent=2), encoding="utf-8"
    )
    manifest = {
        "status": "completed",
        "runtime_seconds": float(time.perf_counter() - started),
        "software": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "inputs": {
            "raw_sha256": sha256(args.raw),
            "cohort_sha256": sha256(args.cohort),
            "split_manifest_sha256": sha256(args.baseline_results_dir / "split_manifest.csv"),
            "archived_catboost_predictions_sha256": sha256(
                args.weighting_results_dir / "catboost_recency_predictions.csv.gz"
            ),
            "catboost_model_selection_sha256": sha256(args.catboost_model_selection),
        },
        "model": {
            "training_weighting": FULL_WEIGHTING,
            "model": MODEL,
            "catboost_version": catboost.__version__,
            "selection_source": str(args.catboost_model_selection.resolve()),
        },
        "SHAP": {
            "method": "interventional Monte-Carlo grouped Shapley values",
            "feature_families": FAMILY_ORDER,
            "permutations_per_record": shap_permutations,
            "background": "one record per training DOI, randomly sampled",
            "evaluation": "one held-out record per DOI",
        },
        "ALE": {
            "candidate_features": ALE_FEATURES,
            "training_quantile_bins": ale_bins,
            "evaluation": "held-out DOI-balanced records",
        },
        "verification": verification,
    }
    (args.output_dir / "explainability_run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(verification, indent=2), flush=True)


if __name__ == "__main__":
    main()
