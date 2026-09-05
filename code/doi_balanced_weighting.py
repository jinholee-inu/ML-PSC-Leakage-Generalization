#!/usr/bin/env python3
"""DOI-balanced sample-weighting analysis for the audited PSC cohort.

The script preserves the frozen unweighted predictions and fits only two new
training-weight variants under the existing DOI-grouped and chronological
partitions. All feature preprocessing and DOI-size calculations are restricted
to each training partition.
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
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from scipy import sparse
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

SCRIPT_DIR = Path(__file__).resolve().parent
BASELINE_CODE_DIR = SCRIPT_DIR.parent / "psc-baseline-validation"
if not (BASELINE_CODE_DIR / "psc_baseline_validation.py").exists():
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


UNWEIGHTED = "Unweighted"
TEMPERED = "Tempered 1/sqrt(n_DOI)"
FULL = "Full 1/n_DOI"
WEIGHT_EXPONENTS = {TEMPERED: 0.5, FULL: 1.0}
WEIGHT_ORDER = [UNWEIGHTED, TEMPERED, FULL]

GROUPED_SCHEME = "DOI-grouped 5-fold"
CHRONO_SCHEME = "Chronological >2018"
SCHEME_ORDER = [GROUPED_SCHEME, CHRONO_SCHEME]

DEVICE_LENS = "Device-level"
PUBLICATION_LENS = "Publication-balanced"
LENS_ORDER = [DEVICE_LENS, PUBLICATION_LENS]

MODEL_ORDER = ["Dummy mean", "Elastic Net", "Random Forest"]
TARGET_ORDER = list(TARGETS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", required=True, type=Path)
    parser.add_argument("--cohort", required=True, type=Path)
    parser.add_argument("--baseline-results-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def append_predictions(
    storage: list[pd.DataFrame],
    metadata: pd.DataFrame,
    scheme: str,
    fold: str,
    weighting: str,
    model: str,
    target: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    storage.append(
        pd.DataFrame(
            {
                "Ref_ID": metadata["Ref_ID"].to_numpy(),
                "doi_norm": metadata["doi_norm"].to_numpy(),
                "publication_year": metadata["publication_year"].to_numpy(),
                "scheme": scheme,
                "fold": fold,
                "training_weighting": weighting,
                "model": model,
                "target": target,
                "y_true": y_true,
                "y_pred": y_pred,
            }
        )
    )


def training_weights(dois: pd.Series, exponent: float) -> np.ndarray:
    counts = dois.value_counts()
    raw = dois.map(counts).to_numpy(dtype=float) ** (-exponent)
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


def fit_weight_variants_for_partition(
    features: pd.DataFrame,
    targets: pd.DataFrame,
    metadata: pd.DataFrame,
    train_index: np.ndarray,
    test_index: np.ndarray,
    numeric_features: list[str],
    config: ModelConfig,
    scheme: str,
    fold: str,
    prediction_storage: list[pd.DataFrame],
    diagnostics: list[dict[str, object]],
) -> None:
    partition_started = time.perf_counter()
    processor = make_preprocessor(config, numeric_features)
    train_matrix = processor.fit_transform(features.iloc[train_index])
    test_matrix = processor.transform(features.iloc[test_index])
    if sparse.issparse(train_matrix):
        train_matrix = train_matrix.tocsr().astype(np.float32)
        test_matrix = test_matrix.tocsr().astype(np.float32)
    else:
        train_matrix = np.asarray(train_matrix, dtype=np.float32)
        test_matrix = np.asarray(test_matrix, dtype=np.float32)
    linear_train = train_matrix.astype(np.float64)
    linear_test = test_matrix.astype(np.float64)

    y_train = targets.iloc[train_index].to_numpy(dtype=float)
    y_test = targets.iloc[test_index].to_numpy(dtype=float)
    train_dois = metadata.iloc[train_index]["doi_norm"].reset_index(drop=True)
    test_metadata = metadata.iloc[test_index]

    for weighting, exponent in WEIGHT_EXPONENTS.items():
        fit_started = time.perf_counter()
        weights = training_weights(train_dois, exponent)
        y_mean, y_std = weighted_target_location_scale(y_train, weights)
        y_scaled = (y_train - y_mean) / y_std

        doi_weight = pd.DataFrame(
            {"doi_norm": train_dois.to_numpy(), "weight": weights}
        ).groupby("doi_norm")["weight"].sum()
        effective_n = float(weights.sum() ** 2 / np.square(weights).sum())

        for target_index, target in enumerate(TARGETS):
            dummy_pred = np.full(len(test_index), y_mean[target_index], dtype=float)
            append_predictions(
                prediction_storage,
                test_metadata,
                scheme,
                fold,
                weighting,
                "Dummy mean",
                target,
                y_test[:, target_index],
                dummy_pred,
            )

            elastic = ElasticNet(
                alpha=config.elastic_alpha,
                l1_ratio=config.elastic_l1_ratio,
                max_iter=config.elastic_max_iter,
                tol=config.elastic_tolerance,
                selection="cyclic",
                random_state=config.seed + target_index,
            )
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                elastic.fit(
                    linear_train,
                    y_scaled[:, target_index],
                    sample_weight=weights,
                )
            pred = elastic.predict(linear_test) * y_std[target_index] + y_mean[target_index]
            append_predictions(
                prediction_storage,
                test_metadata,
                scheme,
                fold,
                weighting,
                "Elastic Net",
                target,
                y_test[:, target_index],
                pred,
            )
            diagnostics.append(
                {
                    "scheme": scheme,
                    "fold": fold,
                    "training_weighting": weighting,
                    "model": "Elastic Net",
                    "target": target,
                    "n_iter": int(elastic.n_iter_),
                    "dual_gap": float(elastic.dual_gap_),
                    "warnings": " | ".join(str(item.message) for item in caught),
                }
            )

        forest = RandomForestRegressor(
            n_estimators=config.rf_estimators,
            max_features=config.rf_max_features,
            min_samples_leaf=config.rf_min_samples_leaf,
            max_samples=config.rf_max_samples,
            bootstrap=True,
            random_state=config.seed + int(re.sub(r"\D", "", fold) or 0),
            n_jobs=-1,
        )
        forest.fit(train_matrix, y_scaled, sample_weight=weights)
        forest_pred = forest.predict(test_matrix) * y_std + y_mean
        for target_index, target in enumerate(TARGETS):
            append_predictions(
                prediction_storage,
                test_metadata,
                scheme,
                fold,
                weighting,
                "Random Forest",
                target,
                y_test[:, target_index],
                forest_pred[:, target_index],
            )

        diagnostics.append(
            {
                "scheme": scheme,
                "fold": fold,
                "training_weighting": weighting,
                "model": "preprocessor_and_random_forest",
                "target": "all",
                "train_records": int(len(train_index)),
                "test_records": int(len(test_index)),
                "train_DOI": int(train_dois.nunique()),
                "test_DOI": int(test_metadata["doi_norm"].nunique()),
                "features_after_encoding": int(train_matrix.shape[1]),
                "sample_weight_min": float(weights.min()),
                "sample_weight_max": float(weights.max()),
                "sample_weight_mean": float(weights.mean()),
                "sample_weight_effective_N": effective_n,
                "DOI_total_weight_min": float(doi_weight.min()),
                "DOI_total_weight_max": float(doi_weight.max()),
                "DOI_total_weight_CV": float(doi_weight.std(ddof=0) / doi_weight.mean()),
                "fit_seconds": float(time.perf_counter() - fit_started),
                "partition_preprocessing_plus_all_weights_seconds": np.nan,
            }
        )

    elapsed = float(time.perf_counter() - partition_started)
    for row in reversed(diagnostics):
        if row.get("scheme") != scheme or row.get("fold") != fold:
            break
        if row.get("model") == "preprocessor_and_random_forest":
            row["partition_preprocessing_plus_all_weights_seconds"] = elapsed


def metric_values(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_weight: np.ndarray | None = None,
) -> dict[str, float]:
    return {
        "R2": float(r2_score(y_true, y_pred, sample_weight=sample_weight)),
        "MAE": float(mean_absolute_error(y_true, y_pred, sample_weight=sample_weight)),
        "RMSE": float(
            math.sqrt(mean_squared_error(y_true, y_pred, sample_weight=sample_weight))
        ),
    }


def evaluation_weights(frame: pd.DataFrame, lens: str) -> np.ndarray | None:
    if lens == DEVICE_LENS:
        return None
    if lens != PUBLICATION_LENS:
        raise ValueError(f"Unknown evaluation lens: {lens}")
    counts = frame["doi_norm"].map(frame["doi_norm"].value_counts()).to_numpy(dtype=float)
    weights = 1.0 / counts
    return weights / weights.mean()


def doi_cluster_stats(frame: pd.DataFrame, lens: str) -> np.ndarray:
    work = frame[["doi_norm", "y_true", "y_pred"]].copy()
    work["y2"] = work["y_true"] ** 2
    work["abs_error"] = (work["y_pred"] - work["y_true"]).abs()
    work["sq_error"] = (work["y_pred"] - work["y_true"]) ** 2
    if lens == DEVICE_LENS:
        grouped = work.groupby("doi_norm", sort=False).agg(
            mass=("y_true", "size"),
            y_sum=("y_true", "sum"),
            y2_sum=("y2", "sum"),
            abs_sum=("abs_error", "sum"),
            sq_sum=("sq_error", "sum"),
        )
    elif lens == PUBLICATION_LENS:
        grouped = work.groupby("doi_norm", sort=False).agg(
            y_sum=("y_true", "mean"),
            y2_sum=("y2", "mean"),
            abs_sum=("abs_error", "mean"),
            sq_sum=("sq_error", "mean"),
        )
        grouped.insert(0, "mass", 1.0)
    else:
        raise ValueError(f"Unknown evaluation lens: {lens}")
    return grouped[["mass", "y_sum", "y2_sum", "abs_sum", "sq_sum"]].to_numpy(
        dtype=float
    )


def metrics_from_totals(totals: np.ndarray) -> tuple[float, float, float]:
    mass, y_sum, y2_sum, abs_sum, sq_sum = totals
    sst = y2_sum - y_sum * y_sum / mass
    r2 = 1.0 - sq_sum / sst if sst > 0 else np.nan
    mae = abs_sum / mass
    rmse = math.sqrt(sq_sum / mass)
    return float(r2), float(mae), float(rmse)


def cluster_bootstrap_samples(
    frame: pd.DataFrame,
    lens: str,
    replicates: int,
    seed: int,
) -> np.ndarray:
    stats = doi_cluster_stats(frame, lens)
    group_count = len(stats)
    rng = np.random.default_rng(seed)
    boot = np.empty((replicates, 3), dtype=float)
    for replicate in range(replicates):
        totals = stats[rng.integers(0, group_count, size=group_count)].sum(axis=0)
        boot[replicate] = metrics_from_totals(totals)
    return boot


def summarize_predictions(
    predictions: pd.DataFrame, config: ModelConfig
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (scheme, weighting, model, target), frame in predictions.groupby(
        ["scheme", "training_weighting", "model", "target"], sort=False
    ):
        for lens in LENS_ORDER:
            weights = evaluation_weights(frame, lens)
            point = metric_values(
                frame["y_true"].to_numpy(),
                frame["y_pred"].to_numpy(),
                sample_weight=weights,
            )
            boot = cluster_bootstrap_samples(
                frame,
                lens=lens,
                replicates=config.bootstrap_replicates,
                seed=config.seed + sum(map(ord, scheme + weighting + model + target + lens)),
            )
            row: dict[str, object] = {
                "scheme": scheme,
                "training_weighting": weighting,
                "evaluation_lens": lens,
                "model": model,
                "target": target,
                "unit": TARGETS[target][1],
                "records": int(len(frame)),
                "DOI_groups": int(frame["doi_norm"].nunique()),
            }
            for metric_index, metric in enumerate(["R2", "MAE", "RMSE"]):
                low, high = np.nanpercentile(boot[:, metric_index], [2.5, 97.5])
                row[metric] = point[metric]
                row[f"{metric}_CI_low"] = float(low)
                row[f"{metric}_CI_high"] = float(high)
            rows.append(row)
    return pd.DataFrame(rows)


def paired_cluster_stats(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    lens: str,
) -> np.ndarray:
    left = baseline[["Ref_ID", "doi_norm", "y_true", "y_pred"]].rename(
        columns={"doi_norm": "doi_base", "y_true": "y_base", "y_pred": "pred_base"}
    )
    right = candidate[["Ref_ID", "doi_norm", "y_true", "y_pred"]].rename(
        columns={"doi_norm": "doi_new", "y_true": "y_new", "y_pred": "pred_new"}
    )
    paired = left.merge(right, on="Ref_ID", how="inner", validate="one_to_one")
    if len(paired) != len(left) or len(paired) != len(right):
        raise AssertionError("Paired predictions do not cover identical records")
    if not paired["doi_base"].eq(paired["doi_new"]).all():
        raise AssertionError("Paired DOI labels disagree")
    if not np.allclose(paired["y_base"], paired["y_new"], atol=1e-12, rtol=0):
        raise AssertionError("Paired target values disagree")

    paired["y2"] = paired["y_base"] ** 2
    paired["abs_base"] = (paired["pred_base"] - paired["y_base"]).abs()
    paired["sq_base"] = (paired["pred_base"] - paired["y_base"]) ** 2
    paired["abs_new"] = (paired["pred_new"] - paired["y_base"]).abs()
    paired["sq_new"] = (paired["pred_new"] - paired["y_base"]) ** 2
    if lens == DEVICE_LENS:
        grouped = paired.groupby("doi_base", sort=False).agg(
            mass=("y_base", "size"),
            y_sum=("y_base", "sum"),
            y2_sum=("y2", "sum"),
            abs_base=("abs_base", "sum"),
            sq_base=("sq_base", "sum"),
            abs_new=("abs_new", "sum"),
            sq_new=("sq_new", "sum"),
        )
    elif lens == PUBLICATION_LENS:
        grouped = paired.groupby("doi_base", sort=False).agg(
            y_sum=("y_base", "mean"),
            y2_sum=("y2", "mean"),
            abs_base=("abs_base", "mean"),
            sq_base=("sq_base", "mean"),
            abs_new=("abs_new", "mean"),
            sq_new=("sq_new", "mean"),
        )
        grouped.insert(0, "mass", 1.0)
    else:
        raise ValueError(f"Unknown evaluation lens: {lens}")
    return grouped[
        ["mass", "y_sum", "y2_sum", "abs_base", "sq_base", "abs_new", "sq_new"]
    ].to_numpy(dtype=float)


def paired_effects_from_totals(totals: np.ndarray) -> tuple[float, float, float]:
    mass, y_sum, y2_sum, abs_base, sq_base, abs_new, sq_new = totals
    sst = y2_sum - y_sum * y_sum / mass
    r2_base = 1.0 - sq_base / sst if sst > 0 else np.nan
    r2_new = 1.0 - sq_new / sst if sst > 0 else np.nan
    mae_base = abs_base / mass
    mae_new = abs_new / mass
    rmse_base = math.sqrt(sq_base / mass)
    rmse_new = math.sqrt(sq_new / mass)
    return (
        float(r2_new - r2_base),
        float(mae_new / mae_base - 1.0),
        float(rmse_new / rmse_base - 1.0),
    )


def paired_comparison_table(
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    config: ModelConfig,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    indexed = metrics.set_index(
        ["scheme", "training_weighting", "evaluation_lens", "model", "target"]
    )
    for scheme in SCHEME_ORDER:
        for weighting in [TEMPERED, FULL]:
            for lens in LENS_ORDER:
                for model in ["Elastic Net", "Random Forest"]:
                    for target in TARGET_ORDER:
                        baseline = predictions.loc[
                            predictions["scheme"].eq(scheme)
                            & predictions["training_weighting"].eq(UNWEIGHTED)
                            & predictions["model"].eq(model)
                            & predictions["target"].eq(target)
                        ]
                        candidate = predictions.loc[
                            predictions["scheme"].eq(scheme)
                            & predictions["training_weighting"].eq(weighting)
                            & predictions["model"].eq(model)
                            & predictions["target"].eq(target)
                        ]
                        stats = paired_cluster_stats(baseline, candidate, lens)
                        rng = np.random.default_rng(
                            config.seed
                            + sum(map(ord, scheme + weighting + lens + model + target))
                            + 404
                        )
                        group_count = len(stats)
                        boot = np.empty((config.bootstrap_replicates, 3), dtype=float)
                        for replicate in range(config.bootstrap_replicates):
                            totals = stats[
                                rng.integers(0, group_count, size=group_count)
                            ].sum(axis=0)
                            boot[replicate] = paired_effects_from_totals(totals)

                        base_metric = indexed.loc[(scheme, UNWEIGHTED, lens, model, target)]
                        new_metric = indexed.loc[(scheme, weighting, lens, model, target)]
                        effects = paired_effects_from_totals(stats.sum(axis=0))
                        row: dict[str, object] = {
                            "scheme": scheme,
                            "training_weighting": weighting,
                            "evaluation_lens": lens,
                            "model": model,
                            "target": target,
                            "unit": TARGETS[target][1],
                            "unweighted_R2": float(base_metric["R2"]),
                            "weighted_R2": float(new_metric["R2"]),
                            "delta_R2_weighted_minus_unweighted": effects[0],
                            "unweighted_MAE": float(base_metric["MAE"]),
                            "weighted_MAE": float(new_metric["MAE"]),
                            "MAE_change_fraction_weighted_vs_unweighted": effects[1],
                            "MAE_reduction_fraction": -effects[1],
                            "unweighted_RMSE": float(base_metric["RMSE"]),
                            "weighted_RMSE": float(new_metric["RMSE"]),
                            "RMSE_change_fraction_weighted_vs_unweighted": effects[2],
                        }
                        names = ["delta_R2", "MAE_change_fraction", "RMSE_change_fraction"]
                        for index, name in enumerate(names):
                            low, high = np.nanpercentile(boot[:, index], [2.5, 97.5])
                            row[f"{name}_CI_low"] = float(low)
                            row[f"{name}_CI_high"] = float(high)
                        rows.append(row)
    return pd.DataFrame(rows)


def doi_size_table(split_manifest: pd.DataFrame) -> pd.DataFrame:
    stats = split_manifest.groupby("doi_norm", sort=False).agg(
        records=("Ref_ID", "size"), publication_year=("publication_year", "first")
    )
    stats["DOI_size_bin"] = pd.cut(
        stats["records"],
        bins=[0, 1, 3, 9, 19, np.inf],
        labels=["1", "2–3", "4–9", "10–19", "≥20"],
    )
    rows: list[dict[str, object]] = []
    for label, frame in stats.groupby("DOI_size_bin", observed=True, sort=False):
        rows.append(
            {
                "DOI_size_bin": str(label),
                "DOI_groups": int(len(frame)),
                "records": int(frame["records"].sum()),
                "fraction_DOI_groups": float(len(frame) / len(stats)),
                "fraction_records": float(frame["records"].sum() / stats["records"].sum()),
                "median_records_per_DOI": float(frame["records"].median()),
                "max_records_per_DOI": int(frame["records"].max()),
            }
        )
    return pd.DataFrame(rows)


def performance_by_doi_size(predictions: pd.DataFrame) -> pd.DataFrame:
    full_sizes = predictions[["Ref_ID", "doi_norm"]].drop_duplicates().groupby(
        "doi_norm"
    )["Ref_ID"].size()
    work = predictions.copy()
    work["DOI_records"] = work["doi_norm"].map(full_sizes)
    work["DOI_size_bin"] = pd.cut(
        work["DOI_records"],
        bins=[0, 1, 3, 9, 19, np.inf],
        labels=["1", "2–3", "4–9", "10–19", "≥20"],
    )
    rows: list[dict[str, object]] = []
    for keys, frame in work.groupby(
        ["scheme", "training_weighting", "model", "target", "DOI_size_bin"],
        observed=True,
        sort=False,
    ):
        scheme, weighting, model, target, size_bin = keys
        for lens in LENS_ORDER:
            values = metric_values(
                frame["y_true"].to_numpy(),
                frame["y_pred"].to_numpy(),
                evaluation_weights(frame, lens),
            )
            rows.append(
                {
                    "scheme": scheme,
                    "training_weighting": weighting,
                    "evaluation_lens": lens,
                    "model": model,
                    "target": target,
                    "DOI_size_bin": str(size_bin),
                    "records": int(len(frame)),
                    "DOI_groups": int(frame["doi_norm"].nunique()),
                    **values,
                }
            )
    return pd.DataFrame(rows)


def error_cluster_stats(frame: pd.DataFrame, lens: str) -> np.ndarray:
    work = frame[["doi_norm", "y_true", "y_pred"]].copy()
    work["error"] = work["y_pred"] - work["y_true"]
    work["abs_error"] = work["error"].abs()
    work["sq_error"] = work["error"] ** 2
    if lens == DEVICE_LENS:
        grouped = work.groupby("doi_norm", sort=False).agg(
            mass=("y_true", "size"),
            y_sum=("y_true", "sum"),
            pred_sum=("y_pred", "sum"),
            error_sum=("error", "sum"),
            abs_sum=("abs_error", "sum"),
            sq_sum=("sq_error", "sum"),
        )
    elif lens == PUBLICATION_LENS:
        grouped = work.groupby("doi_norm", sort=False).agg(
            y_sum=("y_true", "mean"),
            pred_sum=("y_pred", "mean"),
            error_sum=("error", "mean"),
            abs_sum=("abs_error", "mean"),
            sq_sum=("sq_error", "mean"),
        )
        grouped.insert(0, "mass", 1.0)
    else:
        raise ValueError(f"Unknown evaluation lens: {lens}")
    return grouped[
        ["mass", "y_sum", "pred_sum", "error_sum", "abs_sum", "sq_sum"]
    ].to_numpy(dtype=float)


def error_metrics_from_totals(totals: np.ndarray) -> tuple[float, ...]:
    mass, y_sum, pred_sum, error_sum, abs_sum, sq_sum = totals
    return (
        float(y_sum / mass),
        float(pred_sum / mass),
        float(error_sum / mass),
        float(abs_sum / mass),
        float(math.sqrt(sq_sum / mass)),
    )


def chronological_pce_calibration(
    predictions: pd.DataFrame, config: ModelConfig
) -> pd.DataFrame:
    work = predictions.loc[
        predictions["scheme"].eq(CHRONO_SCHEME) & predictions["target"].eq("PCE")
    ].copy()
    work["measured_PCE_bin"] = pd.cut(
        work["y_true"],
        bins=[-np.inf, 5, 10, 15, 20, np.inf],
        labels=["0–5", "5–10", "10–15", "15–20", "≥20"],
        right=False,
    )
    rows: list[dict[str, object]] = []
    for keys, frame in work.groupby(
        ["training_weighting", "model", "measured_PCE_bin"],
        observed=True,
        sort=False,
    ):
        weighting, model, pce_bin = keys
        for lens in LENS_ORDER:
            stats = error_cluster_stats(frame, lens)
            point = error_metrics_from_totals(stats.sum(axis=0))
            rng = np.random.default_rng(
                config.seed
                + sum(map(ord, weighting + model + str(pce_bin) + lens))
                + 707
            )
            group_count = len(stats)
            boot = np.empty((config.bootstrap_replicates, 5), dtype=float)
            for replicate in range(config.bootstrap_replicates):
                totals = stats[
                    rng.integers(0, group_count, size=group_count)
                ].sum(axis=0)
                boot[replicate] = error_metrics_from_totals(totals)
            row: dict[str, object] = {
                "training_weighting": weighting,
                "model": model,
                "evaluation_lens": lens,
                "measured_PCE_bin": str(pce_bin),
                "records": int(len(frame)),
                "DOI_groups": int(frame["doi_norm"].nunique()),
                "measured_PCE_mean": point[0],
                "predicted_PCE_mean": point[1],
                "mean_bias_predicted_minus_measured": point[2],
                "MAE": point[3],
                "RMSE": point[4],
            }
            names = [
                "measured_PCE_mean",
                "predicted_PCE_mean",
                "mean_bias_predicted_minus_measured",
                "MAE",
                "RMSE",
            ]
            for index, name in enumerate(names):
                low, high = np.nanpercentile(boot[:, index], [2.5, 97.5])
                row[f"{name}_CI_low"] = float(low)
                row[f"{name}_CI_high"] = float(high)
            rows.append(row)
    return pd.DataFrame(rows)


def paired_high_efficiency_stats(
    baseline: pd.DataFrame, candidate: pd.DataFrame, lens: str
) -> np.ndarray:
    left = baseline[["Ref_ID", "doi_norm", "y_true", "y_pred"]].rename(
        columns={"doi_norm": "doi_base", "y_true": "y_base", "y_pred": "pred_base"}
    )
    right = candidate[["Ref_ID", "doi_norm", "y_true", "y_pred"]].rename(
        columns={"doi_norm": "doi_new", "y_true": "y_new", "y_pred": "pred_new"}
    )
    paired = left.merge(right, on="Ref_ID", how="inner", validate="one_to_one")
    if len(paired) != len(left) or len(paired) != len(right):
        raise AssertionError("High-efficiency paired predictions do not align")
    if not paired["doi_base"].eq(paired["doi_new"]).all():
        raise AssertionError("High-efficiency paired DOI labels disagree")
    if not np.allclose(paired["y_base"], paired["y_new"], atol=1e-12, rtol=0):
        raise AssertionError("High-efficiency paired targets disagree")
    paired["error_base"] = paired["pred_base"] - paired["y_base"]
    paired["abs_base"] = paired["error_base"].abs()
    paired["sq_base"] = paired["error_base"] ** 2
    paired["error_new"] = paired["pred_new"] - paired["y_base"]
    paired["abs_new"] = paired["error_new"].abs()
    paired["sq_new"] = paired["error_new"] ** 2
    if lens == DEVICE_LENS:
        grouped = paired.groupby("doi_base", sort=False).agg(
            mass=("y_base", "size"),
            error_base=("error_base", "sum"),
            abs_base=("abs_base", "sum"),
            sq_base=("sq_base", "sum"),
            error_new=("error_new", "sum"),
            abs_new=("abs_new", "sum"),
            sq_new=("sq_new", "sum"),
        )
    elif lens == PUBLICATION_LENS:
        grouped = paired.groupby("doi_base", sort=False).agg(
            error_base=("error_base", "mean"),
            abs_base=("abs_base", "mean"),
            sq_base=("sq_base", "mean"),
            error_new=("error_new", "mean"),
            abs_new=("abs_new", "mean"),
            sq_new=("sq_new", "mean"),
        )
        grouped.insert(0, "mass", 1.0)
    else:
        raise ValueError(f"Unknown evaluation lens: {lens}")
    return grouped[
        [
            "mass",
            "error_base",
            "abs_base",
            "sq_base",
            "error_new",
            "abs_new",
            "sq_new",
        ]
    ].to_numpy(dtype=float)


def high_efficiency_effects_from_totals(totals: np.ndarray) -> tuple[float, ...]:
    mass, error_base, abs_base, sq_base, error_new, abs_new, sq_new = totals
    bias_base = error_base / mass
    bias_new = error_new / mass
    mae_base = abs_base / mass
    mae_new = abs_new / mass
    rmse_base = math.sqrt(sq_base / mass)
    rmse_new = math.sqrt(sq_new / mass)
    return (
        float(bias_base),
        float(bias_new),
        float(bias_new - bias_base),
        float(mae_base),
        float(mae_new),
        float(mae_new / mae_base - 1.0),
        float(rmse_base),
        float(rmse_new),
        float(rmse_new / rmse_base - 1.0),
    )


def high_efficiency_paired_comparison(
    predictions: pd.DataFrame, config: ModelConfig
) -> pd.DataFrame:
    work = predictions.loc[
        predictions["scheme"].eq(CHRONO_SCHEME)
        & predictions["target"].eq("PCE")
        & predictions["y_true"].ge(20.0)
    ].copy()
    rows: list[dict[str, object]] = []
    for weighting in [TEMPERED, FULL]:
        for model in ["Elastic Net", "Random Forest"]:
            baseline = work.loc[
                work["training_weighting"].eq(UNWEIGHTED)
                & work["model"].eq(model)
            ]
            candidate = work.loc[
                work["training_weighting"].eq(weighting)
                & work["model"].eq(model)
            ]
            for lens in LENS_ORDER:
                stats = paired_high_efficiency_stats(baseline, candidate, lens)
                point = high_efficiency_effects_from_totals(stats.sum(axis=0))
                rng = np.random.default_rng(
                    config.seed + sum(map(ord, weighting + model + lens)) + 909
                )
                group_count = len(stats)
                boot = np.empty((config.bootstrap_replicates, 9), dtype=float)
                for replicate in range(config.bootstrap_replicates):
                    totals = stats[
                        rng.integers(0, group_count, size=group_count)
                    ].sum(axis=0)
                    boot[replicate] = high_efficiency_effects_from_totals(totals)
                row: dict[str, object] = {
                    "scheme": CHRONO_SCHEME,
                    "subset": "measured PCE >= 20%",
                    "training_weighting": weighting,
                    "evaluation_lens": lens,
                    "model": model,
                    "records": int(len(baseline)),
                    "DOI_groups": int(baseline["doi_norm"].nunique()),
                    "unweighted_mean_bias": point[0],
                    "weighted_mean_bias": point[1],
                    "delta_bias_weighted_minus_unweighted": point[2],
                    "unweighted_MAE": point[3],
                    "weighted_MAE": point[4],
                    "MAE_change_fraction_weighted_vs_unweighted": point[5],
                    "MAE_reduction_fraction": -point[5],
                    "unweighted_RMSE": point[6],
                    "weighted_RMSE": point[7],
                    "RMSE_change_fraction_weighted_vs_unweighted": point[8],
                }
                names = [
                    "unweighted_mean_bias",
                    "weighted_mean_bias",
                    "delta_bias",
                    "unweighted_MAE",
                    "weighted_MAE",
                    "MAE_change_fraction",
                    "unweighted_RMSE",
                    "weighted_RMSE",
                    "RMSE_change_fraction",
                ]
                for index, name in enumerate(names):
                    low, high = np.nanpercentile(boot[:, index], [2.5, 97.5])
                    row[f"{name}_CI_low"] = float(low)
                    row[f"{name}_CI_high"] = float(high)
                rows.append(row)
    return pd.DataFrame(rows)


def plot_figure4(
    metrics: pd.DataFrame,
    paired: pd.DataFrame,
    calibration: pd.DataFrame,
    size_distribution: pd.DataFrame,
    output_dir: Path,
) -> None:
    sns.set_theme(style="whitegrid", context="paper")
    colors = {UNWEIGHTED: "#657A8E", TEMPERED: "#E28E2C", FULL: "#C43C39"}
    labels = {UNWEIGHTED: "Unweighted", TEMPERED: r"$1/\sqrt{n_{DOI}}$", FULL: r"$1/n_{DOI}$"}
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 8.8), constrained_layout=True)

    size_order = ["1", "2–3", "4–9", "10–19", "≥20"]
    size_plot = size_distribution.set_index("DOI_size_bin").reindex(size_order)
    bars = axes[0, 0].bar(
        size_order,
        size_plot["DOI_groups"],
        color="#5B7FA3",
        edgecolor="white",
        linewidth=0.8,
    )
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_ylim(80, 10000)
    axes[0, 0].set_xlabel("Records per DOI")
    axes[0, 0].set_ylabel("Number of DOI groups (log scale)")
    axes[0, 0].set_title("(a) Publication-size distribution", loc="left", fontweight="bold")
    for bar, value, record_share in zip(
        bars,
        size_plot["DOI_groups"],
        size_plot["fraction_records"],
        strict=True,
    ):
        axes[0, 0].text(
            bar.get_x() + bar.get_width() / 2,
            value * 1.12,
            f"{int(value):,}\n({record_share:.0%} rows)",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    rows = [(target, model) for target in TARGET_ORDER for model in ["Elastic Net", "Random Forest"]]
    row_labels = [f"{target} · {model}" for target, model in rows]
    grouped_metrics = metrics.loc[
        metrics["scheme"].eq(GROUPED_SCHEME)
        & metrics["evaluation_lens"].eq(PUBLICATION_LENS)
    ].set_index(["training_weighting", "model", "target"])
    heat = pd.DataFrame(
        [
            [grouped_metrics.loc[(weighting, model, target), "R2"] for weighting in WEIGHT_ORDER]
            for target, model in rows
        ],
        index=row_labels,
        columns=[labels[item] for item in WEIGHT_ORDER],
    )
    sns.heatmap(
        heat,
        annot=True,
        fmt=".3f",
        cmap="RdYlBu",
        center=0,
        vmin=min(-0.2, float(np.nanmin(heat.to_numpy()))),
        vmax=max(0.6, float(np.nanmax(heat.to_numpy()))),
        linewidths=0.6,
        linecolor="white",
        cbar_kws={"label": "Publication-balanced $R^2$"},
        ax=axes[0, 1],
    )
    axes[0, 1].set_xlabel("Training weighting")
    axes[0, 1].set_ylabel("Target and model")
    axes[0, 1].set_title(
        "(b) DOI-grouped publication-balanced performance", loc="left", fontweight="bold"
    )

    trade = paired.loc[
        paired["scheme"].eq(GROUPED_SCHEME)
        & paired["evaluation_lens"].eq(PUBLICATION_LENS)
        & paired["model"].eq("Random Forest")
    ].copy()
    markers = {TEMPERED: "o", FULL: "s"}
    offsets = {"PCE": (4, 4), "Voc": (4, -10), "Jsc": (4, 4), "FF": (4, -10)}
    for weighting in [TEMPERED, FULL]:
        subset = trade.loc[trade["training_weighting"].eq(weighting)].set_index("target").reindex(TARGET_ORDER)
        x = subset["delta_R2_weighted_minus_unweighted"].to_numpy(dtype=float)
        y = 100.0 * subset["MAE_change_fraction_weighted_vs_unweighted"].to_numpy(dtype=float)
        xlow = subset["delta_R2_CI_low"].to_numpy(dtype=float)
        xhigh = subset["delta_R2_CI_high"].to_numpy(dtype=float)
        ylow = 100.0 * subset["MAE_change_fraction_CI_low"].to_numpy(dtype=float)
        yhigh = 100.0 * subset["MAE_change_fraction_CI_high"].to_numpy(dtype=float)
        axes[1, 0].errorbar(
            x,
            y,
            xerr=np.vstack([x - xlow, xhigh - x]),
            yerr=np.vstack([y - ylow, yhigh - y]),
            fmt=markers[weighting],
            color=colors[weighting],
            label=labels[weighting],
            markersize=6,
            capsize=2,
            lw=0.9,
        )
        for target, xx, yy in zip(TARGET_ORDER, x, y, strict=True):
            axes[1, 0].annotate(
                target,
                (xx, yy),
                xytext=offsets[target],
                textcoords="offset points",
                fontsize=8,
            )
    axes[1, 0].axvline(0, color="#2F3A4A", lw=0.9, ls="--")
    axes[1, 0].axhline(0, color="#2F3A4A", lw=0.9, ls="--")
    axes[1, 0].set_xlabel(r"$\Delta R^2$ (weighted − unweighted)")
    axes[1, 0].set_ylabel("MAE change (%)")
    axes[1, 0].set_title(
        "(c) Random Forest weighting trade-off", loc="left", fontweight="bold"
    )
    axes[1, 0].legend(frameon=False, title="")

    cal = calibration.loc[
        calibration["model"].eq("Random Forest")
        & calibration["evaluation_lens"].eq(DEVICE_LENS)
    ]
    pce_bins = ["0–5", "5–10", "10–15", "15–20", "≥20"]
    for weighting in WEIGHT_ORDER:
        subset = cal.loc[cal["training_weighting"].eq(weighting)].set_index(
            "measured_PCE_bin"
        ).reindex(pce_bins)
        high_bias = float(subset.loc["≥20", "mean_bias_predicted_minus_measured"])
        axes[1, 1].plot(
            subset["measured_PCE_mean"],
            subset["predicted_PCE_mean"],
            marker="o",
            lw=1.7,
            color=colors[weighting],
            label=f"{labels[weighting]} (≥20 bias {high_bias:+.2f} pp)",
        )
    axes[1, 1].plot([0, 25], [0, 25], ls="--", color="#2F3A4A", lw=1.1, label="Ideal")
    axes[1, 1].set_xlim(0, 25)
    axes[1, 1].set_ylim(0, 25)
    axes[1, 1].set_xlabel("Measured PCE (%)")
    axes[1, 1].set_ylabel("Mean predicted PCE (%)")
    axes[1, 1].set_title(
        "(d) Chronological Random Forest PCE calibration", loc="left", fontweight="bold"
    )
    axes[1, 1].legend(frameon=False, fontsize=8, loc="upper left")

    for axis in axes.flat:
        axis.tick_params(labelsize=8.5)
    fig.suptitle(
        "DOI-balanced training and publication-level generalization",
        fontsize=14,
        fontweight="bold",
    )
    for suffix in ["png", "pdf", "svg"]:
        kwargs = {"dpi": 600} if suffix == "png" else {}
        fig.savefig(output_dir / f"Figure4_DOI_balanced_weighting.{suffix}", bbox_inches="tight", **kwargs)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    config = ModelConfig()
    if args.quick:
        config = ModelConfig(
            row_random_folds=2,
            grouped_folds=2,
            bootstrap_replicates=50,
            token_min_df=20,
            token_max_features=800,
            rf_estimators=20,
            elastic_max_iter=2000,
        )

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
            "doi_raw": raw["Ref_DOI_number"].astype("string"),
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
    doi_labels_match = (
        split_manifest["doi_norm"]
        .astype("string")
        .eq(metadata["doi_norm"].astype("string"))
        .all()
    )
    if not doi_labels_match:
        raise AssertionError("Normalized DOI labels differ from frozen split manifest")

    grouped_folds = sorted(split_manifest["grouped_fold"].unique())
    if args.quick:
        grouped_folds = grouped_folds[:2]

    frozen = pd.read_csv(args.baseline_results_dir / "baseline_predictions.csv.gz")
    frozen = frozen.loc[frozen["scheme"].isin(SCHEME_ORDER)].copy()
    if args.quick:
        quick_fold_labels = {f"fold_{int(value)}" for value in grouped_folds}
        frozen = frozen.loc[
            frozen["scheme"].eq(CHRONO_SCHEME)
            | (
                frozen["scheme"].eq(GROUPED_SCHEME)
                & frozen["fold"].isin(quick_fold_labels)
            )
        ].copy()
    frozen["training_weighting"] = UNWEIGHTED
    grouped_test_records = int(
        split_manifest["grouped_fold"].isin(grouped_folds).sum()
    )
    expected_frozen = grouped_test_records * len(MODEL_ORDER) * len(TARGET_ORDER) + int(
        split_manifest["chronological_role"].eq("test_2019_onward").sum()
    ) * len(MODEL_ORDER) * len(TARGET_ORDER)
    if len(frozen) != expected_frozen:
        raise AssertionError(
            f"Frozen prediction count mismatch: expected {expected_frozen}, observed {len(frozen)}"
        )

    weighted_predictions: list[pd.DataFrame] = []
    diagnostics: list[dict[str, object]] = []
    for fold_number in grouped_folds:
        test_index = np.flatnonzero(split_manifest["grouped_fold"].eq(fold_number).to_numpy())
        train_index = np.flatnonzero(split_manifest["grouped_fold"].ne(fold_number).to_numpy())
        print(
            f"[DOI-grouped fold {int(fold_number)}/{len(grouped_folds)}] "
            f"train={len(train_index):,}, test={len(test_index):,}",
            flush=True,
        )
        fit_weight_variants_for_partition(
            features,
            targets,
            metadata,
            train_index,
            test_index,
            numeric_features,
            config,
            GROUPED_SCHEME,
            f"fold_{int(fold_number)}",
            weighted_predictions,
            diagnostics,
        )

    chrono_train = np.flatnonzero(
        split_manifest["chronological_role"].eq("train_through_2018").to_numpy()
    )
    chrono_test = np.flatnonzero(
        split_manifest["chronological_role"].eq("test_2019_onward").to_numpy()
    )
    print(
        f"[chronological] train={len(chrono_train):,}, test={len(chrono_test):,}",
        flush=True,
    )
    fit_weight_variants_for_partition(
        features,
        targets,
        metadata,
        chrono_train,
        chrono_test,
        numeric_features,
        config,
        CHRONO_SCHEME,
        "holdout_2019_onward",
        weighted_predictions,
        diagnostics,
    )

    weighted = pd.concat(weighted_predictions, ignore_index=True)
    common_columns = [
        "Ref_ID",
        "doi_norm",
        "publication_year",
        "scheme",
        "fold",
        "training_weighting",
        "model",
        "target",
        "y_true",
        "y_pred",
    ]
    predictions = pd.concat(
        [frozen[common_columns], weighted[common_columns]], ignore_index=True
    )
    predictions["residual"] = predictions["y_pred"] - predictions["y_true"]

    metrics = summarize_predictions(predictions, config)
    paired = paired_comparison_table(predictions, metrics, config)
    by_size = performance_by_doi_size(predictions)
    calibration = chronological_pce_calibration(predictions, config)
    high_efficiency = high_efficiency_paired_comparison(predictions, config)
    size_distribution = doi_size_table(split_manifest)

    metrics.to_csv(args.output_dir / "publication_weighting_metrics.csv", index=False)
    paired.to_csv(
        args.output_dir / "publication_weighting_paired_comparison.csv", index=False
    )
    by_size.to_csv(args.output_dir / "weighting_performance_by_DOI_size.csv", index=False)
    calibration.to_csv(
        args.output_dir / "chronological_PCE_weighting_calibration.csv", index=False
    )
    high_efficiency.to_csv(
        args.output_dir
        / "chronological_PCE_high_efficiency_paired_comparison.csv",
        index=False,
    )
    size_distribution.to_csv(args.output_dir / "DOI_size_distribution.csv", index=False)
    predictions.to_csv(
        args.output_dir / "publication_weighting_predictions.csv.gz",
        index=False,
        compression={"method": "gzip", "compresslevel": 6, "mtime": 0},
    )
    pd.DataFrame(diagnostics).to_csv(
        args.output_dir / "training_weight_diagnostics.csv", index=False
    )
    if not args.quick:
        plot_figure4(metrics, paired, calibration, size_distribution, args.output_dir)

    manifest = {
        "status": "completed",
        "runtime_seconds": float(time.perf_counter() - started),
        "software": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "inputs": {
            "raw_path": str(args.raw.resolve()),
            "raw_sha256": sha256(args.raw),
            "cohort_path": str(args.cohort.resolve()),
            "cohort_sha256": sha256(args.cohort),
            "baseline_results_dir": str(args.baseline_results_dir.resolve()),
        },
        "config": asdict(config),
        "weighting_definitions": {
            UNWEIGHTED: "w_i = 1 (frozen baseline)",
            TEMPERED: "w_i proportional to n_DOI^(-1/2), normalized to mean 1 in training",
            FULL: "w_i proportional to n_DOI^(-1), normalized to mean 1 in training",
        },
        "evaluation_definitions": {
            DEVICE_LENS: "each device record receives equal evaluation weight",
            PUBLICATION_LENS: "each DOI receives equal total evaluation weight",
        },
        "frozen_unweighted_predictions_reused": True,
        "records": int(len(metadata)),
        "DOI_groups": int(metadata["doi_norm"].nunique()),
        "prediction_rows": int(len(predictions)),
        "models": MODEL_ORDER,
        "targets": TARGET_ORDER,
        "schemes": SCHEME_ORDER,
    }
    (args.output_dir / "weighting_run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": "completed",
                "runtime_seconds": manifest["runtime_seconds"],
                "prediction_rows": manifest["prediction_rows"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
