#!/usr/bin/env python3
"""Uncertainty and out-of-distribution analysis for DOI-balanced PSC models.

The script refits only the previously selected full 1/n_DOI weighted Random
Forest under the frozen DOI-grouped and chronological partitions. It verifies
the refitted means against the archived predictions, extracts between-tree
dispersion, calibrates prediction intervals on DOI-disjoint inner partitions,
and computes a target-free feature-space support score using training-only
preprocessing, truncated SVD, and prototype distance.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import re
import sys
import time
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
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import Normalizer, StandardScaler


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCE_ROOT = SCRIPT_DIR.parent / "source" / "PSC_DOI_balanced_weighting_package"
BASELINE_CODE_DIR = DEFAULT_SOURCE_ROOT / "baseline-code"
if not BASELINE_CODE_DIR.exists():
    # Standalone reproducibility-package layout: supporting baseline code sits
    # beside this script in code/.
    BASELINE_CODE_DIR = SCRIPT_DIR
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


FULL_WEIGHTING = "Full 1/n_DOI"
MODEL = "Random Forest"
GROUPED_SCHEME = "DOI-grouped 5-fold"
CHRONO_SCHEME = "Chronological >2018"
SCHEME_ORDER = [GROUPED_SCHEME, CHRONO_SCHEME]
TARGET_ORDER = list(TARGETS)
NOMINAL_LEVELS = [0.90, 0.95]
OOD_COMPONENTS = 32
OOD_CLUSTERS = 160
RETAINED_FRACTIONS = [0.50, 0.60, 0.75, 0.90, 1.00]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", required=True, type=Path)
    parser.add_argument("--cohort", required=True, type=Path)
    parser.add_argument("--baseline-results-dir", required=True, type=Path)
    parser.add_argument("--weighting-results-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def training_weights(dois: pd.Series) -> np.ndarray:
    counts = dois.value_counts()
    raw = dois.map(counts).to_numpy(dtype=float) ** -1.0
    weights = raw / raw.mean()
    if not np.isfinite(weights).all() or np.any(weights <= 0):
        raise AssertionError("Training weights must be positive and finite")
    return weights


def weighted_target_location_scale(
    y_train: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    mean = np.average(y_train, axis=0, weights=weights)
    variance = np.average((y_train - mean) ** 2, axis=0, weights=weights)
    std = np.sqrt(np.maximum(variance, 0.0))
    std = np.where(std > 0, std, 1.0)
    return mean, std


def weighted_quantile(
    values: np.ndarray, quantile: float, weights: np.ndarray
) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    values = values[valid]
    weights = weights[valid]
    if not len(values):
        return float("nan")
    order = np.argsort(values, kind="mergesort")
    values = values[order]
    weights = weights[order]
    cumulative = np.cumsum(weights) - 0.5 * weights
    cumulative /= weights.sum()
    return float(np.interp(quantile, cumulative, values))


def publication_weights(dois: pd.Series) -> np.ndarray:
    counts = dois.value_counts()
    return dois.map(counts).to_numpy(dtype=float) ** -1.0


def as_model_matrix(matrix: object) -> object:
    if sparse.issparse(matrix):
        return matrix.tocsr().astype(np.float32)
    return np.asarray(matrix, dtype=np.float32)


def build_forest(config: ModelConfig, random_state: int) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=config.rf_estimators,
        max_features=config.rf_max_features,
        min_samples_leaf=config.rf_min_samples_leaf,
        max_samples=config.rf_max_samples,
        bootstrap=True,
        random_state=random_state,
        n_jobs=-1,
    )


def tree_prediction_mean_std(
    forest: RandomForestRegressor,
    matrix: object,
    y_mean: np.ndarray,
    y_std: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    tree_predictions = np.stack(
        [tree.predict(matrix) for tree in forest.estimators_], axis=0
    ).astype(np.float64)
    mean_scaled = tree_predictions.mean(axis=0)
    std_scaled = tree_predictions.std(axis=0, ddof=1)
    forest_scaled = forest.predict(matrix)
    mean_difference = float(np.max(np.abs(mean_scaled - forest_scaled)))
    mean = mean_scaled * y_std + y_mean
    std = std_scaled * y_std
    return mean, std, mean_difference


def fit_weighted_forest(
    features: pd.DataFrame,
    targets: pd.DataFrame,
    metadata: pd.DataFrame,
    train_index: np.ndarray,
    test_index: np.ndarray,
    numeric_features: list[str],
    config: ModelConfig,
    random_state: int,
) -> tuple[object, object, RandomForestRegressor, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    processor = make_preprocessor(config, numeric_features)
    train_matrix = as_model_matrix(processor.fit_transform(features.iloc[train_index]))
    test_matrix = as_model_matrix(processor.transform(features.iloc[test_index]))
    y_train = targets.iloc[train_index].to_numpy(dtype=float)
    weights = training_weights(metadata.iloc[train_index]["doi_norm"].reset_index(drop=True))
    y_mean, y_std = weighted_target_location_scale(y_train, weights)
    y_scaled = (y_train - y_mean) / y_std
    forest = build_forest(config, random_state)
    forest.fit(train_matrix, y_scaled, sample_weight=weights)
    prediction, ensemble_std, tree_mean_difference = tree_prediction_mean_std(
        forest, test_matrix, y_mean, y_std
    )
    return (
        train_matrix,
        test_matrix,
        forest,
        prediction,
        ensemble_std,
        y_mean,
        y_std,
        tree_mean_difference,
    )


def feature_space_ood(
    train_matrix: object,
    test_matrix: object,
    random_state: int,
    components: int,
    clusters: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    normalizer = Normalizer(norm="l2", copy=True)
    normalized_train = normalizer.transform(train_matrix)
    normalized_test = normalizer.transform(test_matrix)
    n_components = max(
        2,
        min(components, normalized_train.shape[0] - 1, normalized_train.shape[1] - 1),
    )
    svd = TruncatedSVD(
        n_components=n_components,
        n_iter=5,
        random_state=random_state,
    )
    train_reduced = svd.fit_transform(normalized_train)
    test_reduced = svd.transform(normalized_test)
    scaler = StandardScaler()
    train_reduced = scaler.fit_transform(train_reduced).astype(np.float32)
    test_reduced = scaler.transform(test_reduced).astype(np.float32)
    n_clusters = max(8, min(clusters, len(train_reduced)))
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        batch_size=min(4096, len(train_reduced)),
        n_init=3,
        max_iter=150,
        random_state=random_state,
    )
    train_labels = kmeans.fit_predict(train_reduced)
    test_labels = kmeans.predict(test_reduced)
    train_distance = np.linalg.norm(
        train_reduced - kmeans.cluster_centers_[train_labels], axis=1
    )
    test_distance = np.linalg.norm(
        test_reduced - kmeans.cluster_centers_[test_labels], axis=1
    )
    sorted_train = np.sort(train_distance)
    percentile = np.searchsorted(sorted_train, test_distance, side="right") / float(
        len(sorted_train)
    )
    median_distance = float(np.median(train_distance))
    ratio = test_distance / max(median_distance, np.finfo(float).eps)
    diagnostics = {
        "svd_components": int(n_components),
        "svd_explained_variance_ratio_sum": float(
            svd.explained_variance_ratio_.sum()
        ),
        "prototype_clusters": int(n_clusters),
        "train_distance_median": median_distance,
        "train_distance_p95": float(np.quantile(train_distance, 0.95)),
        "test_distance_median": float(np.median(test_distance)),
        "test_distance_p95": float(np.quantile(test_distance, 0.95)),
        "test_fraction_above_train_p95": float(np.mean(percentile > 0.95)),
    }
    return ratio, np.clip(percentile, 0.0, 1.0), diagnostics


def forest_leaf_support(
    forest: RandomForestRegressor,
    train_matrix: object,
    test_matrix: object,
    train_weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Measure held-out support from DOI-weighted training mass in RF leaves."""
    train_log_support = np.zeros(train_matrix.shape[0], dtype=np.float64)
    test_log_support = np.zeros(test_matrix.shape[0], dtype=np.float64)
    for tree in forest.estimators_:
        train_leaf = tree.apply(train_matrix)
        test_leaf = tree.apply(test_matrix)
        size = int(max(train_leaf.max(), test_leaf.max())) + 1
        leaf_mass = np.bincount(train_leaf, weights=train_weights, minlength=size)
        train_log_support += np.log1p(leaf_mass[train_leaf])
        test_log_support += np.log1p(leaf_mass[test_leaf])
    train_log_support /= len(forest.estimators_)
    test_log_support /= len(forest.estimators_)
    train_support = np.expm1(train_log_support)
    test_support = np.expm1(test_log_support)
    train_ood = -train_log_support
    test_ood = -test_log_support
    sorted_train = np.sort(train_ood)
    percentile = np.searchsorted(sorted_train, test_ood, side="right") / float(
        len(sorted_train)
    )
    diagnostics = {
        "train_leaf_support_median": float(np.median(train_support)),
        "train_leaf_support_p05": float(np.quantile(train_support, 0.05)),
        "test_leaf_support_median": float(np.median(test_support)),
        "test_leaf_support_p05": float(np.quantile(test_support, 0.05)),
        "test_fraction_model_OOD_above_train_p95": float(
            np.mean(percentile > 0.95)
        ),
    }
    return test_support, np.clip(percentile, 0.0, 1.0), diagnostics


def calibrate_partition(
    features: pd.DataFrame,
    targets: pd.DataFrame,
    metadata: pd.DataFrame,
    fit_index: np.ndarray,
    calibration_index: np.ndarray,
    numeric_features: list[str],
    config: ModelConfig,
    random_state: int,
    scheme: str,
    outer_fold: str,
    calibration_label: str,
) -> tuple[dict[str, dict[str, float]], list[dict[str, object]]]:
    (
        _train_matrix,
        _calibration_matrix,
        _forest,
        prediction,
        ensemble_std,
        _y_mean,
        _y_std,
        tree_mean_difference,
    ) = fit_weighted_forest(
        features,
        targets,
        metadata,
        fit_index,
        calibration_index,
        numeric_features,
        config,
        random_state,
    )
    y_true = targets.iloc[calibration_index].to_numpy(dtype=float)
    dois = metadata.iloc[calibration_index]["doi_norm"].reset_index(drop=True)
    evaluation_weights = publication_weights(dois)
    calibration: dict[str, dict[str, float]] = {}
    diagnostics: list[dict[str, object]] = []
    for target_index, target in enumerate(TARGET_ORDER):
        raw_sigma = ensemble_std[:, target_index]
        floor = weighted_quantile(raw_sigma, 0.05, evaluation_weights)
        scale = np.maximum(raw_sigma, floor)
        score = np.abs(y_true[:, target_index] - prediction[:, target_index]) / scale
        row = {
            "sigma_floor": floor,
            "q90": weighted_quantile(score, 0.90, evaluation_weights),
            "q95": weighted_quantile(score, 0.95, evaluation_weights),
        }
        calibration[target] = row
        for nominal in NOMINAL_LEVELS:
            quantile = row[f"q{int(nominal * 100)}"]
            covered = np.abs(y_true[:, target_index] - prediction[:, target_index]) <= (
                quantile * scale
            )
            diagnostics.append(
                {
                    "scheme": scheme,
                    "outer_fold": outer_fold,
                    "calibration_partition": calibration_label,
                    "target": target,
                    "nominal_coverage": nominal,
                    "fit_records": int(len(fit_index)),
                    "fit_DOI": int(metadata.iloc[fit_index]["doi_norm"].nunique()),
                    "calibration_records": int(len(calibration_index)),
                    "calibration_DOI": int(dois.nunique()),
                    "sigma_floor": floor,
                    "normalized_residual_quantile": quantile,
                    "publication_balanced_calibration_coverage": float(
                        np.average(covered, weights=evaluation_weights)
                    ),
                    "tree_mean_max_difference": tree_mean_difference,
                }
            )
    return calibration, diagnostics


def interval_score(
    y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray, alpha: float
) -> np.ndarray:
    width = upper - lower
    below = (2.0 / alpha) * (lower - y_true) * (y_true < lower)
    above = (2.0 / alpha) * (y_true - upper) * (y_true > upper)
    return width + below + above


def assemble_predictions(
    metadata: pd.DataFrame,
    targets: pd.DataFrame,
    test_index: np.ndarray,
    scheme: str,
    fold: str,
    prediction: np.ndarray,
    ensemble_std: np.ndarray,
    feature_ood_distance_ratio: np.ndarray,
    feature_ood_percentile: np.ndarray,
    model_support: np.ndarray,
    model_ood_percentile: np.ndarray,
    calibration: dict[str, dict[str, float]],
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    meta = metadata.iloc[test_index].reset_index(drop=True)
    y_true = targets.iloc[test_index].reset_index(drop=True)
    for target_index, target in enumerate(TARGET_ORDER):
        frame = meta[["Ref_ID", "doi_norm", "publication_year"]].copy()
        frame["scheme"] = scheme
        frame["fold"] = fold
        frame["training_weighting"] = FULL_WEIGHTING
        frame["model"] = MODEL
        frame["target"] = target
        frame["y_true"] = y_true[target].to_numpy(dtype=float)
        frame["y_pred"] = prediction[:, target_index]
        frame["residual"] = frame["y_pred"] - frame["y_true"]
        frame["absolute_error"] = frame["residual"].abs()
        frame["ensemble_std"] = ensemble_std[:, target_index]
        frame["feature_ood_distance_ratio"] = feature_ood_distance_ratio
        frame["feature_ood_percentile"] = feature_ood_percentile
        frame["model_leaf_support"] = model_support
        frame["model_ood_percentile"] = model_ood_percentile
        sigma_floor = calibration[target]["sigma_floor"]
        adaptive_scale = np.maximum(frame["ensemble_std"].to_numpy(), sigma_floor)
        frame["sigma_floor"] = sigma_floor
        for nominal in NOMINAL_LEVELS:
            label = int(nominal * 100)
            q = calibration[target][f"q{label}"]
            half_width = q * adaptive_scale
            lower = frame["y_pred"].to_numpy() - half_width
            upper = frame["y_pred"].to_numpy() + half_width
            frame[f"interval_{label}_half_width"] = half_width
            frame[f"interval_{label}_lower"] = lower
            frame[f"interval_{label}_upper"] = upper
            frame[f"interval_{label}_covered"] = (
                (frame["y_true"].to_numpy() >= lower)
                & (frame["y_true"].to_numpy() <= upper)
            )
            frame[f"interval_{label}_score"] = interval_score(
                frame["y_true"].to_numpy(), lower, upper, 1.0 - nominal
            )
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def saved_prediction_difference(
    produced: pd.DataFrame,
    archived: pd.DataFrame,
    scheme: str,
    fold: str,
) -> float:
    left = produced[["Ref_ID", "target", "y_true", "y_pred"]].copy()
    right = archived.loc[
        archived["scheme"].eq(scheme)
        & archived["fold"].eq(fold)
        & archived["training_weighting"].eq(FULL_WEIGHTING)
        & archived["model"].eq(MODEL),
        ["Ref_ID", "target", "y_true", "y_pred"],
    ].copy()
    merged = left.merge(
        right,
        on=["Ref_ID", "target"],
        how="outer",
        suffixes=("_new", "_archived"),
        indicator=True,
        validate="one_to_one",
    )
    if not merged["_merge"].eq("both").all():
        raise AssertionError(f"Archived prediction keys differ for {scheme} {fold}")
    y_difference = np.max(np.abs(merged["y_true_new"] - merged["y_true_archived"]))
    if y_difference > 1e-12:
        raise AssertionError("Archived target values differ")
    return float(np.max(np.abs(merged["y_pred_new"] - merged["y_pred_archived"])))


def cluster_mean_ci(
    frame: pd.DataFrame,
    value_column: str,
    lens: str,
    replicates: int,
    seed: int,
) -> tuple[float, float, float]:
    grouped = frame.groupby("doi_norm", sort=False)[value_column].agg(["sum", "count"])
    sums = grouped["sum"].to_numpy(dtype=float)
    counts = grouped["count"].to_numpy(dtype=float)
    if lens == "Device-level":
        point = float(sums.sum() / counts.sum())
    elif lens == "Publication-balanced":
        point = float(np.mean(sums / counts))
    else:
        raise ValueError(lens)
    rng = np.random.default_rng(seed)
    boot = np.empty(replicates, dtype=float)
    n_clusters = len(sums)
    for index in range(replicates):
        sample = rng.integers(0, n_clusters, n_clusters)
        if lens == "Device-level":
            boot[index] = sums[sample].sum() / counts[sample].sum()
        else:
            boot[index] = np.mean(sums[sample] / counts[sample])
    low, high = np.quantile(boot, [0.025, 0.975])
    return point, float(low), float(high)


def coverage_metrics(
    predictions: pd.DataFrame, replicates: int, seed: int
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (scheme, target), frame in predictions.groupby(["scheme", "target"], sort=False):
        for nominal in NOMINAL_LEVELS:
            label = int(nominal * 100)
            work = frame.copy()
            work["coverage_value"] = work[f"interval_{label}_covered"].astype(float)
            work["width_value"] = 2.0 * work[f"interval_{label}_half_width"]
            work["score_value"] = work[f"interval_{label}_score"]
            for lens in ["Device-level", "Publication-balanced"]:
                coverage, coverage_low, coverage_high = cluster_mean_ci(
                    work,
                    "coverage_value",
                    lens,
                    replicates,
                    seed + sum(map(ord, scheme + target + lens)) + label,
                )
                width, width_low, width_high = cluster_mean_ci(
                    work,
                    "width_value",
                    lens,
                    replicates,
                    seed + 1000 + sum(map(ord, scheme + target + lens)) + label,
                )
                score, score_low, score_high = cluster_mean_ci(
                    work,
                    "score_value",
                    lens,
                    replicates,
                    seed + 2000 + sum(map(ord, scheme + target + lens)) + label,
                )
                rows.append(
                    {
                        "scheme": scheme,
                        "target": target,
                        "nominal_coverage": nominal,
                        "evaluation_lens": lens,
                        "empirical_coverage": coverage,
                        "coverage_CI_low": coverage_low,
                        "coverage_CI_high": coverage_high,
                        "mean_interval_width": width,
                        "width_CI_low": width_low,
                        "width_CI_high": width_high,
                        "mean_interval_score": score,
                        "interval_score_CI_low": score_low,
                        "interval_score_CI_high": score_high,
                        "records": int(len(frame)),
                        "DOI_groups": int(frame["doi_norm"].nunique()),
                    }
                )
    return pd.DataFrame(rows)


def correlation_ci(
    x: np.ndarray, y: np.ndarray, replicates: int, seed: int
) -> tuple[float, float, float]:
    point = float(spearmanr(x, y).statistic)
    rng = np.random.default_rng(seed)
    boot = np.empty(replicates, dtype=float)
    for index in range(replicates):
        sample = rng.integers(0, len(x), len(x))
        boot[index] = spearmanr(x[sample], y[sample]).statistic
    low, high = np.nanquantile(boot, [0.025, 0.975])
    return point, float(low), float(high)


def uncertainty_error_association(
    predictions: pd.DataFrame, replicates: int, seed: int
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (scheme, target), frame in predictions.groupby(["scheme", "target"], sort=False):
        doi = frame.groupby("doi_norm", sort=False).agg(
            mean_absolute_error=("absolute_error", "mean"),
            mean_ensemble_std=("ensemble_std", "mean"),
            mean_interval_half_width=("interval_95_half_width", "mean"),
            mean_feature_ood_percentile=("feature_ood_percentile", "mean"),
            mean_model_ood_percentile=("model_ood_percentile", "mean"),
            records=("Ref_ID", "size"),
        )
        for score_name, column in [
            ("Ensemble standard deviation", "mean_ensemble_std"),
            ("Calibrated 95% half-width", "mean_interval_half_width"),
            ("Feature-space OOD percentile", "mean_feature_ood_percentile"),
            ("Model-support OOD percentile", "mean_model_ood_percentile"),
        ]:
            point, low, high = correlation_ci(
                doi[column].to_numpy(dtype=float),
                doi["mean_absolute_error"].to_numpy(dtype=float),
                replicates,
                seed + sum(map(ord, scheme + target + score_name)),
            )
            rows.append(
                {
                    "scheme": scheme,
                    "target": target,
                    "score": score_name,
                    "publication_level_Spearman_rho": point,
                    "rho_CI_low": low,
                    "rho_CI_high": high,
                    "DOI_groups": int(len(doi)),
                }
            )
    return pd.DataFrame(rows)


def quantile_performance(
    predictions: pd.DataFrame, replicates: int, seed: int
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    measures = {
        "Predictive uncertainty": "interval_95_half_width",
        "Feature-space OOD": "feature_ood_percentile",
        "Model-support OOD": "model_ood_percentile",
    }
    for (scheme, target), frame in predictions.groupby(["scheme", "target"], sort=False):
        for measure_name, measure_column in measures.items():
            work = frame.copy()
            ranks = work[measure_column].rank(method="average", pct=True)
            work["quantile"] = np.minimum(np.ceil(ranks * 5.0), 5).astype(int)
            for quantile, subset in work.groupby("quantile", sort=True):
                mae, low, high = cluster_mean_ci(
                    subset,
                    "absolute_error",
                    "Publication-balanced",
                    replicates,
                    seed
                    + sum(map(ord, scheme + target + measure_name))
                    + int(quantile),
                )
                rows.append(
                    {
                        "scheme": scheme,
                        "target": target,
                        "ranking_measure": measure_name,
                        "quantile": int(quantile),
                        "publication_balanced_MAE": mae,
                        "MAE_CI_low": low,
                        "MAE_CI_high": high,
                        "score_median": float(subset[measure_column].median()),
                        "records": int(len(subset)),
                        "DOI_groups": int(subset["doi_norm"].nunique()),
                    }
                )
    return pd.DataFrame(rows)


def ood_stratified_performance(
    predictions: pd.DataFrame, replicates: int, seed: int
) -> pd.DataFrame:
    bins = [-np.inf, 0.50, 0.75, 0.90, 0.95, np.inf]
    labels = ["ID <=50th", "50-75th", "75-90th", "90-95th", "OOD >95th"]
    rows: list[dict[str, object]] = []
    measures = {
        "Feature-space OOD": "feature_ood_percentile",
        "Model-support OOD": "model_ood_percentile",
    }
    for measure_name, column in measures.items():
        work = predictions.copy()
        work["OOD_stratum"] = pd.cut(
            work[column], bins=bins, labels=labels, right=True
        )
        for (scheme, target, stratum), frame in work.groupby(
            ["scheme", "target", "OOD_stratum"], observed=True, sort=False
        ):
            for lens in ["Device-level", "Publication-balanced"]:
                mae, low, high = cluster_mean_ci(
                    frame,
                    "absolute_error",
                    lens,
                    replicates,
                    seed
                    + sum(
                        map(ord, scheme + target + measure_name + str(stratum) + lens)
                    ),
                )
                coverage, coverage_low, coverage_high = cluster_mean_ci(
                    frame.assign(
                        coverage_value=frame["interval_95_covered"].astype(float)
                    ),
                    "coverage_value",
                    lens,
                    replicates,
                    seed
                    + 2000
                    + sum(
                        map(ord, scheme + target + measure_name + str(stratum) + lens)
                    ),
                )
                rows.append(
                    {
                        "scheme": scheme,
                        "target": target,
                        "OOD_measure": measure_name,
                        "OOD_stratum": str(stratum),
                        "evaluation_lens": lens,
                        "MAE": mae,
                        "MAE_CI_low": low,
                        "MAE_CI_high": high,
                        "coverage_95": coverage,
                        "coverage_95_CI_low": coverage_low,
                        "coverage_95_CI_high": coverage_high,
                        "records": int(len(frame)),
                        "DOI_groups": int(frame["doi_norm"].nunique()),
                    }
                )
    return pd.DataFrame(rows)


def selective_prediction_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    measures = {
        "Predictive uncertainty": "interval_95_half_width",
        "Feature-space OOD": "feature_ood_percentile",
        "Model-support OOD": "model_ood_percentile",
    }
    for (scheme, target), frame in predictions.groupby(["scheme", "target"], sort=False):
        for measure_name, column in measures.items():
            ordered = frame.sort_values(column, kind="mergesort")
            for retained in RETAINED_FRACTIONS:
                n_keep = max(1, int(math.ceil(len(ordered) * retained)))
                subset = ordered.iloc[:n_keep]
                device_mae = float(subset["absolute_error"].mean())
                doi_mae = float(
                    subset.groupby("doi_norm", sort=False)["absolute_error"].mean().mean()
                )
                rows.append(
                    {
                        "scheme": scheme,
                        "target": target,
                        "ranking_measure": measure_name,
                        "retained_fraction": retained,
                        "device_level_MAE": device_mae,
                        "publication_balanced_MAE": doi_mae,
                        "records_retained": int(len(subset)),
                        "DOI_groups_retained": int(subset["doi_norm"].nunique()),
                        "score_threshold": float(subset[column].max()),
                    }
                )
    return pd.DataFrame(rows)


def plot_figure5(
    coverage: pd.DataFrame,
    quantiles: pd.DataFrame,
    selective: pd.DataFrame,
    output_dir: Path,
) -> None:
    sns.set_theme(style="whitegrid", context="paper")
    palette = {GROUPED_SCHEME: "#3B6EA8", CHRONO_SCHEME: "#C44E52"}
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 8.4), constrained_layout=True)

    ax = axes[0, 0]
    panel = quantiles.loc[
        quantiles["target"].eq("PCE")
        & quantiles["ranking_measure"].eq("Predictive uncertainty")
    ]
    for scheme in SCHEME_ORDER:
        part = panel.loc[panel["scheme"].eq(scheme)].sort_values("quantile")
        ax.errorbar(
            part["quantile"],
            part["publication_balanced_MAE"],
            yerr=np.vstack(
                [
                    part["publication_balanced_MAE"] - part["MAE_CI_low"],
                    part["MAE_CI_high"] - part["publication_balanced_MAE"],
                ]
            ),
            marker="o",
            capsize=3,
            linewidth=1.5,
            color=palette[scheme],
            label=scheme.replace(" 5-fold", ""),
        )
    ax.set_title("(a) Uncertainty stratifies PCE error", loc="left", fontweight="bold")
    ax.set_xlabel("Predictive-uncertainty quintile")
    ax.set_ylabel("Publication-balanced PCE MAE (pp)")
    ax.set_xticks(range(1, 6))
    ax.legend(frameon=False, fontsize=8)

    ax = axes[0, 1]
    panel = coverage.loc[
        coverage["target"].eq("PCE")
        & coverage["evaluation_lens"].eq("Publication-balanced")
    ]
    offsets = {GROUPED_SCHEME: -0.003, CHRONO_SCHEME: 0.003}
    for scheme in SCHEME_ORDER:
        part = panel.loc[panel["scheme"].eq(scheme)].sort_values("nominal_coverage")
        ax.errorbar(
            part["nominal_coverage"] + offsets[scheme],
            part["empirical_coverage"],
            yerr=np.vstack(
                [
                    part["empirical_coverage"] - part["coverage_CI_low"],
                    part["coverage_CI_high"] - part["empirical_coverage"],
                ]
            ),
            marker="o",
            capsize=3,
            linestyle="none",
            color=palette[scheme],
            label=scheme.replace(" 5-fold", ""),
        )
    ax.plot([0.885, 0.965], [0.885, 0.965], "--", color="#555555", linewidth=1)
    ax.set_xlim(0.885, 0.965)
    ax.set_ylim(0.80, 1.00)
    ax.set_xticks([0.90, 0.95])
    ax.set_xticklabels(["90%", "95%"])
    ax.set_yticks([0.80, 0.85, 0.90, 0.95, 1.00])
    ax.set_yticklabels(["80%", "85%", "90%", "95%", "100%"])
    ax.set_title("(b) Publication-balanced interval coverage", loc="left", fontweight="bold")
    ax.set_xlabel("Nominal coverage")
    ax.set_ylabel("Empirical coverage")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 0]
    panel = quantiles.loc[
        quantiles["target"].eq("PCE")
        & quantiles["ranking_measure"].eq("Model-support OOD")
    ]
    for scheme in SCHEME_ORDER:
        part = panel.loc[panel["scheme"].eq(scheme)].sort_values("quantile")
        ax.errorbar(
            part["quantile"],
            part["publication_balanced_MAE"],
            yerr=np.vstack(
                [
                    part["publication_balanced_MAE"] - part["MAE_CI_low"],
                    part["MAE_CI_high"] - part["publication_balanced_MAE"],
                ]
            ),
            marker="o",
            capsize=3,
            linewidth=1.5,
            color=palette[scheme],
            label=scheme.replace(" 5-fold", ""),
        )
    ax.set_title("(c) Low model support identifies OOD risk", loc="left", fontweight="bold")
    ax.set_xlabel("OOD-score quintile")
    ax.set_ylabel("Publication-balanced PCE MAE (pp)")
    ax.set_xticks(range(1, 6))
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 1]
    panel = selective.loc[
        selective["scheme"].eq(CHRONO_SCHEME) & selective["target"].eq("PCE")
    ]
    measure_palette = {
        "Predictive uncertainty": "#4C72B0",
        "Feature-space OOD": "#DD8452",
        "Model-support OOD": "#55A868",
    }
    for measure in measure_palette:
        part = panel.loc[panel["ranking_measure"].eq(measure)].sort_values(
            "retained_fraction"
        )
        ax.plot(
            100.0 * part["retained_fraction"],
            part["publication_balanced_MAE"],
            marker="o",
            linewidth=1.6,
            color=measure_palette[measure],
            label=measure,
        )
    ax.set_title("(d) Selective prediction in the future cohort", loc="left", fontweight="bold")
    ax.set_xlabel("Predictions retained (%)")
    ax.set_ylabel("Publication-balanced PCE MAE (pp)")
    ax.set_xticks([50, 60, 75, 90, 100])
    ax.legend(frameon=False, fontsize=8)

    fig.suptitle(
        "Uncertainty calibration and supported-domain diagnostics",
        fontsize=14,
        fontweight="bold",
    )
    for extension, kwargs in [
        ("png", {"dpi": 600}),
        ("pdf", {}),
        ("svg", {}),
    ]:
        fig.savefig(
            output_dir / f"Figure5_uncertainty_OOD.{extension}",
            bbox_inches="tight",
            **kwargs,
        )
    plt.close(fig)


def report_text(
    coverage: pd.DataFrame,
    association: pd.DataFrame,
    quantiles: pd.DataFrame,
    selective: pd.DataFrame,
    verification: dict[str, object],
) -> str:
    def cov(scheme: str, nominal: float) -> pd.Series:
        return coverage.loc[
            coverage["scheme"].eq(scheme)
            & coverage["target"].eq("PCE")
            & coverage["nominal_coverage"].eq(nominal)
            & coverage["evaluation_lens"].eq("Publication-balanced")
        ].iloc[0]

    def assoc(scheme: str, score: str) -> pd.Series:
        return association.loc[
            association["scheme"].eq(scheme)
            & association["target"].eq("PCE")
            & association["score"].eq(score)
        ].iloc[0]

    def quintile(scheme: str, measure: str, q: int) -> pd.Series:
        return quantiles.loc[
            quantiles["scheme"].eq(scheme)
            & quantiles["target"].eq("PCE")
            & quantiles["ranking_measure"].eq(measure)
            & quantiles["quantile"].eq(q)
        ].iloc[0]

    def risk(measure: str, retained: float) -> pd.Series:
        return selective.loc[
            selective["scheme"].eq(CHRONO_SCHEME)
            & selective["target"].eq("PCE")
            & selective["ranking_measure"].eq(measure)
            & selective["retained_fraction"].eq(retained)
        ].iloc[0]

    g90, g95 = cov(GROUPED_SCHEME, 0.90), cov(GROUPED_SCHEME, 0.95)
    c90, c95 = cov(CHRONO_SCHEME, 0.90), cov(CHRONO_SCHEME, 0.95)
    c_unc = assoc(CHRONO_SCHEME, "Calibrated 95% half-width")
    c_feature_ood = assoc(CHRONO_SCHEME, "Feature-space OOD percentile")
    c_model_ood = assoc(CHRONO_SCHEME, "Model-support OOD percentile")
    c_u1 = quintile(CHRONO_SCHEME, "Predictive uncertainty", 1)
    c_u5 = quintile(CHRONO_SCHEME, "Predictive uncertainty", 5)
    c_o1 = quintile(CHRONO_SCHEME, "Model-support OOD", 1)
    c_o5 = quintile(CHRONO_SCHEME, "Model-support OOD", 5)
    r50_u, r100_u = risk("Predictive uncertainty", 0.50), risk(
        "Predictive uncertainty", 1.00
    )
    r50_o = risk("Model-support OOD", 0.50)

    return f"""# PSC uncertainty and out-of-distribution analysis

## Scope and design

The analysis retained the frozen 33,175-record cohort, normalized DOI groups, feature pipeline, hyperparameters, and validation partitions. Only the previously selected full `1/n_DOI` weighted Random Forest was refitted. The refitted mean predictions were checked record by record against the archived weighting analysis before uncertainty or OOD statistics were accepted.

Prediction uncertainty was represented by the between-tree standard deviation of the 120-tree Random Forest. This raw dispersion was scaled using DOI-balanced normalized residual quantiles obtained from DOI-disjoint inner calibration partitions. In each grouped outer fold, one of the remaining DOI folds was reserved for calibration and the other three were used for the calibration model. For the chronological analysis, publications through 2017 formed the calibration-model training set and 2018 publications formed the calibration set; the final predictor remained trained through 2018 and was evaluated on 2019-2021 records.

The OOD score was target-free. The training-fitted feature matrix was row-normalized, reduced by truncated SVD, and represented by training-only MiniBatchKMeans prototypes. Each held-out record received a nearest-prototype distance percentile relative to the corresponding training partition. Values above the training 95th percentile were designated strongly OOD.

## Main results

- DOI-grouped PCE intervals achieved publication-balanced empirical coverage of **{g90.empirical_coverage:.3f}** at the 90% nominal level and **{g95.empirical_coverage:.3f}** at the 95% level.
- In the chronological 2019-2021 cohort, publication-balanced PCE coverage was **{c90.empirical_coverage:.3f}** for the 90% interval and **{c95.empirical_coverage:.3f}** for the 95% interval. The difference from the grouped result quantifies interval calibration drift under temporal transfer.
- Across future DOI groups, the calibrated PCE half-width correlated with mean absolute error at Spearman rho = **{c_unc.publication_level_Spearman_rho:.3f}** (95% CI: {c_unc.rho_CI_low:.3f} to {c_unc.rho_CI_high:.3f}). The target-free feature-space OOD percentile showed rho = **{c_feature_ood.publication_level_Spearman_rho:.3f}** (95% CI: {c_feature_ood.rho_CI_low:.3f} to {c_feature_ood.rho_CI_high:.3f}), whereas the model-support OOD percentile showed rho = **{c_model_ood.publication_level_Spearman_rho:.3f}** (95% CI: {c_model_ood.rho_CI_low:.3f} to {c_model_ood.rho_CI_high:.3f}).
- Chronological publication-balanced PCE MAE increased from **{c_u1.publication_balanced_MAE:.3f}** pp in the lowest uncertainty quintile to **{c_u5.publication_balanced_MAE:.3f}** pp in the highest uncertainty quintile. Across model-support OOD strata, it changed from **{c_o1.publication_balanced_MAE:.3f}** pp in quintile 1 to **{c_o5.publication_balanced_MAE:.3f}** pp in quintile 5.
- Retaining the 50% most certain chronological predictions reduced publication-balanced PCE MAE from **{r100_u.publication_balanced_MAE:.3f}** to **{r50_u.publication_balanced_MAE:.3f}** pp. Retaining the 50% most supported predictions by the forest-leaf score yielded **{r50_o.publication_balanced_MAE:.3f}** pp.

## Interpretation

The uncertainty and support scores should be treated as complementary diagnostics. Random-Forest tree dispersion measures model disagreement within the fitted ensemble, prototype distance measures how far a record lies from the observed training-feature manifold, and DOI-weighted forest-leaf mass quantifies model-conditioned local support. A useful score should increase with error and enable selective prediction, but nominal interval coverage can still deteriorate under chronological shift because the temporal cohort is not exchangeable with the historical calibration data.

The practical recommendation is therefore to report the full DOI-balanced Random Forest together with its calibrated interval and OOD percentile, to flag predictions above the training 95th OOD percentile, and to avoid interpreting the interval as a guarantee for record-performance devices. The supported-domain filter improves average reliability but does not establish causal validity or prospective screening performance.

## Integrity checks

- Prediction rows: {verification['prediction_rows']:,}
- Duplicate prediction keys: {verification['duplicate_prediction_keys']}
- Maximum difference from archived full-weighted predictions: {verification['max_archived_prediction_difference']:.3e}
- Maximum tree-mean aggregation difference: {verification['max_tree_mean_difference']:.3e}
- Grouped boundary DOI overlap: {verification['grouped_boundary_DOI_overlap']}
- Chronological boundary DOI overlap: {verification['chronological_boundary_DOI_overlap']}
- DOI-cluster bootstrap replicates: {verification['bootstrap_replicates']}
"""


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    config = ModelConfig()
    ood_components = OOD_COMPONENTS
    ood_clusters = OOD_CLUSTERS
    if args.quick:
        config = ModelConfig(
            grouped_folds=2,
            bootstrap_replicates=30,
            token_min_df=20,
            token_max_features=800,
            rf_estimators=20,
        )
        ood_components = 8
        ood_clusters = 24

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
            "publication_year": pd.to_datetime(
                raw["Ref_publication_date"], errors="raise"
            ).dt.year,
        }
    )
    targets = pd.DataFrame(index=raw.index)
    for target, (source, _unit) in TARGETS.items():
        targets[target] = pd.to_numeric(raw[source], errors="raise")
    targets["FF"] = targets["FF"] * 100.0
    features, numeric_features = build_features(raw)

    split_manifest = pd.read_csv(args.baseline_results_dir / "split_manifest.csv")
    if not split_manifest["Ref_ID"].equals(metadata["Ref_ID"]):
        raise AssertionError("Split manifest does not align to frozen cohort")
    if not split_manifest["doi_norm"].astype("string").eq(
        metadata["doi_norm"].astype("string")
    ).all():
        raise AssertionError("Normalized DOI labels differ from frozen split manifest")

    archived = pd.read_csv(
        args.weighting_results_dir / "publication_weighting_predictions.csv.gz"
    )
    archived = archived.loc[
        archived["scheme"].isin(SCHEME_ORDER)
        & archived["training_weighting"].eq(FULL_WEIGHTING)
        & archived["model"].eq(MODEL)
    ].copy()

    grouped_folds = sorted(split_manifest["grouped_fold"].unique())
    if args.quick:
        grouped_folds = grouped_folds[:2]
    prediction_frames: list[pd.DataFrame] = []
    calibration_rows: list[dict[str, object]] = []
    partition_rows: list[dict[str, object]] = []
    archived_differences: list[float] = []
    tree_mean_differences: list[float] = []

    for fold_number in grouped_folds:
        outer_test = np.flatnonzero(
            split_manifest["grouped_fold"].eq(fold_number).to_numpy()
        )
        outer_train = np.flatnonzero(
            split_manifest["grouped_fold"].ne(fold_number).to_numpy()
        )
        calibration_fold = grouped_folds[
            (grouped_folds.index(fold_number) + 1) % len(grouped_folds)
        ]
        inner_calibration = np.flatnonzero(
            split_manifest["grouped_fold"].eq(calibration_fold).to_numpy()
        )
        inner_fit = np.flatnonzero(
            ~split_manifest["grouped_fold"].isin([fold_number, calibration_fold]).to_numpy()
        )
        fold = f"fold_{int(fold_number)}"
        print(
            f"[{GROUPED_SCHEME} {fold}] calibration fit={len(inner_fit):,}, "
            f"calibration={len(inner_calibration):,}",
            flush=True,
        )
        calibration, diagnostics = calibrate_partition(
            features,
            targets,
            metadata,
            inner_fit,
            inner_calibration,
            numeric_features,
            config,
            config.seed + 10000 + int(fold_number),
            GROUPED_SCHEME,
            fold,
            f"fold_{int(calibration_fold)}",
        )
        calibration_rows.extend(diagnostics)
        print(
            f"[{GROUPED_SCHEME} {fold}] final fit train={len(outer_train):,}, "
            f"test={len(outer_test):,}",
            flush=True,
        )
        partition_started = time.perf_counter()
        (
            train_matrix,
            test_matrix,
            forest,
            prediction,
            ensemble_std,
            _y_mean,
            _y_std,
            tree_mean_difference,
        ) = fit_weighted_forest(
            features,
            targets,
            metadata,
            outer_train,
            outer_test,
            numeric_features,
            config,
            config.seed + int(fold_number),
        )
        ood_ratio, ood_percentile, ood_diag = feature_space_ood(
            train_matrix,
            test_matrix,
            config.seed + 20000 + int(fold_number),
            ood_components,
            ood_clusters,
        )
        model_support, model_ood_percentile, model_ood_diag = forest_leaf_support(
            forest,
            train_matrix,
            test_matrix,
            training_weights(
                metadata.iloc[outer_train]["doi_norm"].reset_index(drop=True)
            ),
        )
        ood_diag.update(model_ood_diag)
        frame = assemble_predictions(
            metadata,
            targets,
            outer_test,
            GROUPED_SCHEME,
            fold,
            prediction,
            ensemble_std,
            ood_ratio,
            ood_percentile,
            model_support,
            model_ood_percentile,
            calibration,
        )
        archived_difference = saved_prediction_difference(
            frame, archived, GROUPED_SCHEME, fold
        )
        archived_differences.append(archived_difference)
        tree_mean_differences.append(tree_mean_difference)
        prediction_frames.append(frame)
        partition_rows.append(
            {
                "scheme": GROUPED_SCHEME,
                "fold": fold,
                "train_records": int(len(outer_train)),
                "test_records": int(len(outer_test)),
                "train_DOI": int(metadata.iloc[outer_train]["doi_norm"].nunique()),
                "test_DOI": int(metadata.iloc[outer_test]["doi_norm"].nunique()),
                "features_after_encoding": int(train_matrix.shape[1]),
                "tree_mean_max_difference": tree_mean_difference,
                "archived_prediction_max_difference": archived_difference,
                "runtime_seconds": float(time.perf_counter() - partition_started),
                **ood_diag,
            }
        )

    chrono_train = np.flatnonzero(
        split_manifest["chronological_role"].eq("train_through_2018").to_numpy()
    )
    chrono_test = np.flatnonzero(
        split_manifest["chronological_role"].eq("test_2019_onward").to_numpy()
    )
    chrono_inner_fit = np.flatnonzero(
        split_manifest["publication_year"].le(2017).to_numpy()
    )
    chrono_inner_calibration = np.flatnonzero(
        split_manifest["publication_year"].eq(2018).to_numpy()
    )
    chrono_fold = "holdout_2019_onward"
    print(
        f"[{CHRONO_SCHEME}] calibration fit={len(chrono_inner_fit):,}, "
        f"calibration={len(chrono_inner_calibration):,}",
        flush=True,
    )
    calibration, diagnostics = calibrate_partition(
        features,
        targets,
        metadata,
        chrono_inner_fit,
        chrono_inner_calibration,
        numeric_features,
        config,
        config.seed + 30000,
        CHRONO_SCHEME,
        chrono_fold,
        "publication_year_2018",
    )
    calibration_rows.extend(diagnostics)
    print(
        f"[{CHRONO_SCHEME}] final fit train={len(chrono_train):,}, "
        f"test={len(chrono_test):,}",
        flush=True,
    )
    partition_started = time.perf_counter()
    (
        train_matrix,
        test_matrix,
        forest,
        prediction,
        ensemble_std,
        _y_mean,
        _y_std,
        tree_mean_difference,
    ) = fit_weighted_forest(
        features,
        targets,
        metadata,
        chrono_train,
        chrono_test,
        numeric_features,
        config,
        config.seed + int(re.sub(r"\D", "", chrono_fold) or 0),
    )
    ood_ratio, ood_percentile, ood_diag = feature_space_ood(
        train_matrix,
        test_matrix,
        config.seed + 40000,
        ood_components,
        ood_clusters,
    )
    model_support, model_ood_percentile, model_ood_diag = forest_leaf_support(
        forest,
        train_matrix,
        test_matrix,
        training_weights(metadata.iloc[chrono_train]["doi_norm"].reset_index(drop=True)),
    )
    ood_diag.update(model_ood_diag)
    frame = assemble_predictions(
        metadata,
        targets,
        chrono_test,
        CHRONO_SCHEME,
        chrono_fold,
        prediction,
        ensemble_std,
        ood_ratio,
        ood_percentile,
        model_support,
        model_ood_percentile,
        calibration,
    )
    archived_difference = saved_prediction_difference(
        frame, archived, CHRONO_SCHEME, chrono_fold
    )
    archived_differences.append(archived_difference)
    tree_mean_differences.append(tree_mean_difference)
    prediction_frames.append(frame)
    partition_rows.append(
        {
            "scheme": CHRONO_SCHEME,
            "fold": chrono_fold,
            "train_records": int(len(chrono_train)),
            "test_records": int(len(chrono_test)),
            "train_DOI": int(metadata.iloc[chrono_train]["doi_norm"].nunique()),
            "test_DOI": int(metadata.iloc[chrono_test]["doi_norm"].nunique()),
            "features_after_encoding": int(train_matrix.shape[1]),
            "tree_mean_max_difference": tree_mean_difference,
            "archived_prediction_max_difference": archived_difference,
            "runtime_seconds": float(time.perf_counter() - partition_started),
            **ood_diag,
        }
    )

    predictions = pd.concat(prediction_frames, ignore_index=True)
    predictions.to_csv(
        args.output_dir / "uncertainty_ood_predictions.csv.gz",
        index=False,
        compression="gzip",
    )
    pd.DataFrame(calibration_rows).to_csv(
        args.output_dir / "uncertainty_calibration_diagnostics.csv", index=False
    )
    pd.DataFrame(partition_rows).to_csv(
        args.output_dir / "uncertainty_ood_partition_diagnostics.csv", index=False
    )

    replicates = config.bootstrap_replicates
    coverage = coverage_metrics(predictions, replicates, config.seed)
    association = uncertainty_error_association(predictions, replicates, config.seed)
    quantiles = quantile_performance(predictions, replicates, config.seed)
    ood_strata = ood_stratified_performance(predictions, replicates, config.seed)
    selective = selective_prediction_metrics(predictions)
    coverage.to_csv(args.output_dir / "uncertainty_coverage_metrics.csv", index=False)
    association.to_csv(
        args.output_dir / "uncertainty_error_association.csv", index=False
    )
    quantiles.to_csv(
        args.output_dir / "uncertainty_ood_quintile_performance.csv", index=False
    )
    ood_strata.to_csv(
        args.output_dir / "ood_stratified_performance.csv", index=False
    )
    selective.to_csv(
        args.output_dir / "selective_prediction_metrics.csv", index=False
    )
    plot_figure5(coverage, quantiles, selective, args.output_dir)

    expected_rows = (
        int(split_manifest["grouped_fold"].isin(grouped_folds).sum())
        + int(split_manifest["chronological_role"].eq("test_2019_onward").sum())
    ) * len(TARGET_ORDER)
    duplicate_keys = int(
        predictions.duplicated(["scheme", "fold", "Ref_ID", "target"]).sum()
    )
    grouped_overlap = 0
    for fold_number in grouped_folds:
        test_doi = set(
            split_manifest.loc[
                split_manifest["grouped_fold"].eq(fold_number), "doi_norm"
            ]
        )
        train_doi = set(
            split_manifest.loc[
                split_manifest["grouped_fold"].ne(fold_number), "doi_norm"
            ]
        )
        grouped_overlap += len(test_doi.intersection(train_doi))
    chronological_overlap = len(
        set(metadata.iloc[chrono_train]["doi_norm"]).intersection(
            set(metadata.iloc[chrono_test]["doi_norm"])
        )
    )
    verification = {
        "status": "passed",
        "prediction_rows": int(len(predictions)),
        "expected_prediction_rows": int(expected_rows),
        "duplicate_prediction_keys": duplicate_keys,
        "max_archived_prediction_difference": float(max(archived_differences)),
        "max_tree_mean_difference": float(max(tree_mean_differences)),
        "grouped_boundary_DOI_overlap": int(grouped_overlap),
        "chronological_boundary_DOI_overlap": int(chronological_overlap),
        "bootstrap_replicates": int(replicates),
        "interval_bounds_finite": bool(
            np.isfinite(
                predictions[
                    [
                        "interval_90_lower",
                        "interval_90_upper",
                        "interval_95_lower",
                        "interval_95_upper",
                    ]
                ].to_numpy()
            ).all()
        ),
        "interval_order_valid": bool(
            (
                predictions["interval_95_lower"]
                <= predictions["interval_90_lower"]
            ).all()
            and (
                predictions["interval_90_lower"] <= predictions["y_pred"]
            ).all()
            and (
                predictions["y_pred"] <= predictions["interval_90_upper"]
            ).all()
            and (
                predictions["interval_90_upper"]
                <= predictions["interval_95_upper"]
            ).all()
        ),
        "feature_ood_percentile_range": [
            float(predictions["feature_ood_percentile"].min()),
            float(predictions["feature_ood_percentile"].max()),
        ],
        "model_ood_percentile_range": [
            float(predictions["model_ood_percentile"].min()),
            float(predictions["model_ood_percentile"].max()),
        ],
    }
    if verification["prediction_rows"] != verification["expected_prediction_rows"]:
        raise AssertionError("Prediction row count mismatch")
    if duplicate_keys:
        raise AssertionError("Duplicate prediction keys")
    if (not args.quick) and verification["max_archived_prediction_difference"] > 1e-10:
        raise AssertionError("Refitted predictions differ from frozen weighted results")
    if grouped_overlap or chronological_overlap:
        raise AssertionError("DOI split boundary leakage detected")
    if not verification["interval_bounds_finite"] or not verification["interval_order_valid"]:
        raise AssertionError("Invalid uncertainty interval bounds")
    (args.output_dir / "uncertainty_ood_verification_report.json").write_text(
        json.dumps(verification, indent=2), encoding="utf-8"
    )
    (args.output_dir / "PSC_uncertainty_OOD_report.md").write_text(
        report_text(coverage, association, quantiles, selective, verification),
        encoding="utf-8",
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
            "split_manifest_sha256": sha256(
                args.baseline_results_dir / "split_manifest.csv"
            ),
            "archived_weighting_predictions_sha256": sha256(
                args.weighting_results_dir / "publication_weighting_predictions.csv.gz"
            ),
        },
        "model": {
            "training_weighting": FULL_WEIGHTING,
            "model": MODEL,
            "estimators": config.rf_estimators,
            "max_features": config.rf_max_features,
            "min_samples_leaf": config.rf_min_samples_leaf,
            "max_samples": config.rf_max_samples,
        },
        "uncertainty": {
            "raw_measure": "between-tree standard deviation",
            "calibration": "DOI-balanced normalized residual quantiles on DOI-disjoint inner partitions",
            "nominal_levels": NOMINAL_LEVELS,
        },
        "ood": {
            "target_free": True,
            "components": ood_components,
            "prototype_clusters": ood_clusters,
            "score": "nearest-prototype distance percentile relative to training partition",
        },
        "verification": verification,
    }
    (args.output_dir / "uncertainty_ood_run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(verification, indent=2), flush=True)


if __name__ == "__main__":
    main()
