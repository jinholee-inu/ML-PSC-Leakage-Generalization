#!/usr/bin/env python3
"""CatBoost robustness benchmark and DOI-balanced temporal-recency analysis.

The analysis preserves the audited 33,175-record cohort, the frozen DOI-grouped
five-fold assignments, and the 2019--2021 chronological holdout. DOI, year, and
photovoltaic targets are never model inputs. CatBoost hyperparameters are tuned
only inside each outer training partition. Temporal-recency strength is chosen
using historical rolling-origin pseudo-futures (2016, 2017, and 2018) without
accessing the 2019--2021 holdout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from dataclasses import asdict
from pathlib import Path

import catboost
from catboost import CatBoostRegressor, Pool
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from scipy import sparse
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold


SCRIPT_DIR = Path(__file__).resolve().parent
BASELINE_CODE_DIR = SCRIPT_DIR / "baseline-code"
if not (BASELINE_CODE_DIR / "psc_baseline_validation.py").exists():
    BASELINE_CODE_DIR = (
        SCRIPT_DIR.parent
        / "recovered-analysis"
        / "baseline"
        / "PSC_baseline_validation"
    )
sys.path.insert(0, str(BASELINE_CODE_DIR))

from psc_baseline_validation import (  # noqa: E402
    RAW_REQUIRED,
    TARGETS,
    ModelConfig,
    build_features,
    make_preprocessor,
    normalize_doi,
)


GROUPED_SCHEME = "DOI-grouped 5-fold"
CHRONO_SCHEME = "Chronological >2018"
FULL_WEIGHTING = "Full 1/n_DOI"
TARGET_ORDER = list(TARGETS)
LENSES = ["Device-level", "Publication-balanced"]

RF_FULL = "Random Forest | Full DOI"
CB_FULL = "CatBoost | Full DOI"
RF_RECENCY = "Random Forest | Full DOI + recency"
CB_RECENCY = "CatBoost | Full DOI + recency"

LAMBDA_GRID = [0.0, 0.025, 0.05, 0.10, 0.15, 0.20, 0.30]
ROLLING_YEARS = [2016, 2017, 2018]

CB_CANDIDATES = [
    {
        "candidate": "depth6_lr0.05_l2-10",
        "depth": 6,
        "learning_rate": 0.05,
        "l2_leaf_reg": 10.0,
        "iterations": 700,
    },
    {
        "candidate": "depth8_lr0.04_l2-10",
        "depth": 8,
        "learning_rate": 0.04,
        "l2_leaf_reg": 10.0,
        "iterations": 700,
    },
    {
        "candidate": "depth6_lr0.10_l2-20",
        "depth": 6,
        "learning_rate": 0.10,
        "l2_leaf_reg": 20.0,
        "iterations": 450,
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", required=True, type=Path)
    parser.add_argument("--cohort", required=True, type=Path)
    parser.add_argument("--split-manifest", required=True, type=Path)
    parser.add_argument("--frozen-weighting-predictions", required=True, type=Path)
    parser.add_argument("--baseline-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_training_weights(
    dois: pd.Series,
    years: pd.Series,
    recency_lambda: float,
) -> np.ndarray:
    counts = dois.value_counts()
    publication_balance = 1.0 / dois.map(counts).to_numpy(dtype=float)
    max_year = int(years.max())
    recency = np.exp(
        -float(recency_lambda) * (max_year - years.to_numpy(dtype=float))
    )
    raw = publication_balance * recency
    weights = raw / raw.mean()
    if not np.isfinite(weights).all() or np.any(weights <= 0):
        raise AssertionError("Training weights must be positive and finite")
    return weights


def target_location_scale(
    y_train: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    mean = np.average(y_train, axis=0, weights=weights)
    variance = np.average((y_train - mean) ** 2, axis=0, weights=weights)
    std = np.sqrt(np.maximum(variance, 0.0))
    std = np.where(std > 0, std, 1.0)
    return mean, std


def evaluation_weights(dois: pd.Series, lens: str) -> np.ndarray:
    if lens == "Device-level":
        return np.ones(len(dois), dtype=float)
    if lens == "Publication-balanced":
        counts = dois.value_counts()
        weights = 1.0 / dois.map(counts).to_numpy(dtype=float)
        return weights / weights.mean()
    raise ValueError(lens)


def weighted_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray
) -> tuple[float, float, float]:
    mean = float(np.average(y_true, weights=weights))
    residual = y_pred - y_true
    sse = float(np.sum(weights * residual**2))
    sst = float(np.sum(weights * (y_true - mean) ** 2))
    r2 = 1.0 - sse / sst if sst > 0 else np.nan
    mae = float(np.average(np.abs(residual), weights=weights))
    rmse = float(np.sqrt(np.average(residual**2, weights=weights)))
    return r2, mae, rmse


def publication_balanced_standardized_mae(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    dois: pd.Series,
    train_std: np.ndarray,
) -> float:
    weights = evaluation_weights(dois.reset_index(drop=True), "Publication-balanced")
    scores = [
        np.average(np.abs(y_pred[:, j] - y_true[:, j]), weights=weights)
        / train_std[j]
        for j in range(y_true.shape[1])
    ]
    return float(np.mean(scores))


def prepare_matrices(
    features: pd.DataFrame,
    train_index: np.ndarray,
    test_index: np.ndarray,
    numeric_features: list[str],
    config: ModelConfig,
):
    processor = make_preprocessor(config, numeric_features)
    x_train = processor.fit_transform(features.iloc[train_index])
    x_test = processor.transform(features.iloc[test_index])
    if sparse.issparse(x_train):
        x_train = x_train.tocsr().astype(np.float32)
        x_test = x_test.tocsr().astype(np.float32)
    else:
        x_train = np.asarray(x_train, dtype=np.float32)
        x_test = np.asarray(x_test, dtype=np.float32)
    return x_train, x_test, int(x_train.shape[1])


def catboost_model(
    candidate: dict[str, object],
    seed: int,
    iterations: int | None = None,
) -> CatBoostRegressor:
    return CatBoostRegressor(
        loss_function="MultiRMSE",
        eval_metric="MultiRMSE",
        iterations=int(iterations or candidate["iterations"]),
        depth=int(candidate["depth"]),
        learning_rate=float(candidate["learning_rate"]),
        l2_leaf_reg=float(candidate["l2_leaf_reg"]),
        bootstrap_type="Bernoulli",
        subsample=0.80,
        rsm=0.50,
        random_strength=1.0,
        random_seed=int(seed),
        thread_count=-1,
        allow_writing_files=False,
        verbose=False,
    )


def select_catboost_config(
    context: str,
    features: pd.DataFrame,
    targets: pd.DataFrame,
    metadata: pd.DataFrame,
    outer_train_index: np.ndarray,
    numeric_features: list[str],
    config: ModelConfig,
    candidates: list[dict[str, object]],
    inner_folds: int,
    selection_rows: list[dict[str, object]],
) -> tuple[dict[str, object], int]:
    groups = metadata.iloc[outer_train_index]["doi_norm"].to_numpy()
    splitter = GroupKFold(n_splits=inner_folds)
    split_pairs = list(splitter.split(outer_train_index, groups=groups))
    score_store = {str(item["candidate"]): [] for item in candidates}
    iter_store = {str(item["candidate"]): [] for item in candidates}

    for inner_number, (relative_train, relative_valid) in enumerate(split_pairs, 1):
        train_index = outer_train_index[relative_train]
        valid_index = outer_train_index[relative_valid]
        x_train, x_valid, encoded_features = prepare_matrices(
            features, train_index, valid_index, numeric_features, config
        )
        y_train = targets.iloc[train_index].to_numpy(dtype=float)
        y_valid = targets.iloc[valid_index].to_numpy(dtype=float)
        train_meta = metadata.iloc[train_index]
        valid_meta = metadata.iloc[valid_index]
        weights = make_training_weights(
            train_meta["doi_norm"].reset_index(drop=True),
            train_meta["publication_year"].reset_index(drop=True),
            0.0,
        )
        y_mean, y_std = target_location_scale(y_train, weights)
        y_train_scaled = (y_train - y_mean) / y_std
        y_valid_scaled = (y_valid - y_mean) / y_std
        valid_weights = evaluation_weights(
            valid_meta["doi_norm"].reset_index(drop=True),
            "Publication-balanced",
        )
        train_pool = Pool(x_train, label=y_train_scaled, weight=weights)
        valid_pool = Pool(x_valid, label=y_valid_scaled, weight=valid_weights)

        for candidate_number, candidate in enumerate(candidates):
            seed = config.seed + inner_number * 100 + candidate_number
            model = catboost_model(candidate, seed)
            started = time.perf_counter()
            model.fit(
                train_pool,
                eval_set=valid_pool,
                use_best_model=True,
                early_stopping_rounds=60,
            )
            pred = model.predict(x_valid) * y_std + y_mean
            score = publication_balanced_standardized_mae(
                y_valid,
                pred,
                valid_meta["doi_norm"],
                y_std,
            )
            best_iteration = int(model.get_best_iteration()) + 1
            if best_iteration <= 0:
                best_iteration = int(candidate["iterations"])
            name = str(candidate["candidate"])
            score_store[name].append(score)
            iter_store[name].append(best_iteration)
            selection_rows.append(
                {
                    "context": context,
                    "inner_fold": inner_number,
                    "candidate": name,
                    "score_mean_standardized_publication_balanced_MAE": score,
                    "best_iteration": best_iteration,
                    "train_records": len(train_index),
                    "validation_records": len(valid_index),
                    "train_DOI": train_meta["doi_norm"].nunique(),
                    "validation_DOI": valid_meta["doi_norm"].nunique(),
                    "encoded_features": encoded_features,
                    "fit_seconds": time.perf_counter() - started,
                }
            )

    mean_scores = {name: float(np.mean(values)) for name, values in score_store.items()}
    selected_name = min(mean_scores, key=mean_scores.get)
    selected = next(item for item in candidates if item["candidate"] == selected_name)
    selected_iterations = int(np.median(iter_store[selected_name]))
    selected_iterations = max(50, min(selected_iterations, int(selected["iterations"])))
    for row in selection_rows:
        if row["context"] == context:
            row["candidate_mean_score"] = mean_scores[str(row["candidate"])]
            row["selected_candidate"] = selected_name
            row["selected_final_iterations"] = selected_iterations
    print(
        f"[{context}] selected {selected_name}, iterations={selected_iterations}, "
        f"score={mean_scores[selected_name]:.5f}",
        flush=True,
    )
    return dict(selected), selected_iterations


def fit_catboost_partition(
    features: pd.DataFrame,
    targets: pd.DataFrame,
    metadata: pd.DataFrame,
    train_index: np.ndarray,
    test_index: np.ndarray,
    numeric_features: list[str],
    config: ModelConfig,
    candidate: dict[str, object],
    iterations: int,
    recency_lambda: float,
    seed_offset: int,
) -> tuple[np.ndarray, dict[str, object]]:
    started = time.perf_counter()
    x_train, x_test, encoded_features = prepare_matrices(
        features, train_index, test_index, numeric_features, config
    )
    y_train = targets.iloc[train_index].to_numpy(dtype=float)
    train_meta = metadata.iloc[train_index]
    weights = make_training_weights(
        train_meta["doi_norm"].reset_index(drop=True),
        train_meta["publication_year"].reset_index(drop=True),
        recency_lambda,
    )
    y_mean, y_std = target_location_scale(y_train, weights)
    y_scaled = (y_train - y_mean) / y_std
    model = catboost_model(candidate, config.seed + seed_offset, iterations)
    model.fit(Pool(x_train, label=y_scaled, weight=weights))
    prediction = model.predict(x_test) * y_std + y_mean
    counts = train_meta["doi_norm"].value_counts()
    total_per_doi = pd.DataFrame(
        {
            "doi": train_meta["doi_norm"].to_numpy(),
            "weight": weights,
        }
    ).groupby("doi")["weight"].sum()
    diagnostics = {
        "encoded_features": encoded_features,
        "train_records": len(train_index),
        "test_records": len(test_index),
        "train_DOI": len(counts),
        "test_DOI": metadata.iloc[test_index]["doi_norm"].nunique(),
        "recency_lambda": recency_lambda,
        "weight_min": float(weights.min()),
        "weight_max": float(weights.max()),
        "weight_effective_N": float(weights.sum() ** 2 / np.square(weights).sum()),
        "DOI_total_weight_CV": float(
            total_per_doi.std(ddof=0) / total_per_doi.mean()
        ),
        "fit_seconds": time.perf_counter() - started,
    }
    return prediction, diagnostics


def fit_rf_partition(
    features: pd.DataFrame,
    targets: pd.DataFrame,
    metadata: pd.DataFrame,
    train_index: np.ndarray,
    test_index: np.ndarray,
    numeric_features: list[str],
    config: ModelConfig,
    recency_lambda: float,
    seed_offset: int,
) -> tuple[np.ndarray, dict[str, object]]:
    started = time.perf_counter()
    x_train, x_test, encoded_features = prepare_matrices(
        features, train_index, test_index, numeric_features, config
    )
    y_train = targets.iloc[train_index].to_numpy(dtype=float)
    train_meta = metadata.iloc[train_index]
    weights = make_training_weights(
        train_meta["doi_norm"].reset_index(drop=True),
        train_meta["publication_year"].reset_index(drop=True),
        recency_lambda,
    )
    y_mean, y_std = target_location_scale(y_train, weights)
    y_scaled = (y_train - y_mean) / y_std
    model = RandomForestRegressor(
        n_estimators=config.rf_estimators,
        max_features=config.rf_max_features,
        min_samples_leaf=config.rf_min_samples_leaf,
        max_samples=config.rf_max_samples,
        bootstrap=True,
        random_state=config.seed + seed_offset,
        n_jobs=-1,
    )
    model.fit(x_train, y_scaled, sample_weight=weights)
    prediction = model.predict(x_test) * y_std + y_mean
    return prediction, {
        "encoded_features": encoded_features,
        "train_records": len(train_index),
        "test_records": len(test_index),
        "train_DOI": train_meta["doi_norm"].nunique(),
        "test_DOI": metadata.iloc[test_index]["doi_norm"].nunique(),
        "recency_lambda": recency_lambda,
        "weight_min": float(weights.min()),
        "weight_max": float(weights.max()),
        "weight_effective_N": float(weights.sum() ** 2 / np.square(weights).sum()),
        "fit_seconds": time.perf_counter() - started,
    }


def append_multioutput_predictions(
    storage: list[pd.DataFrame],
    metadata: pd.DataFrame,
    targets: pd.DataFrame,
    test_index: np.ndarray,
    predictions: np.ndarray,
    scheme: str,
    fold: str,
    condition: str,
    model: str,
    weighting: str,
    recency_lambda: float,
) -> None:
    test_meta = metadata.iloc[test_index]
    y_true = targets.iloc[test_index].to_numpy(dtype=float)
    for j, target in enumerate(TARGET_ORDER):
        storage.append(
            pd.DataFrame(
                {
                    "Ref_ID": test_meta["Ref_ID"].to_numpy(),
                    "doi_norm": test_meta["doi_norm"].to_numpy(),
                    "publication_year": test_meta["publication_year"].to_numpy(),
                    "scheme": scheme,
                    "fold": fold,
                    "condition": condition,
                    "model": model,
                    "training_weighting": weighting,
                    "recency_lambda": recency_lambda,
                    "target": target,
                    "y_true": y_true[:, j],
                    "y_pred": predictions[:, j],
                }
            )
        )


def metric_bootstrap(
    frame: pd.DataFrame,
    lens: str,
    replicates: int,
    seed: int,
) -> np.ndarray:
    work = frame[["doi_norm", "y_true", "y_pred"]].copy()
    work["y2"] = work["y_true"] ** 2
    work["abs_error"] = (work["y_pred"] - work["y_true"]).abs()
    work["sq_error"] = (work["y_pred"] - work["y_true"]) ** 2
    grouped = work.groupby("doi_norm", sort=False).agg(
        n=("y_true", "size"),
        y_sum=("y_true", "sum"),
        y2_sum=("y2", "sum"),
        abs_sum=("abs_error", "sum"),
        sq_sum=("sq_error", "sum"),
    )
    stats = grouped.to_numpy(dtype=float)
    if lens == "Publication-balanced":
        stats[:, 1:] /= stats[:, [0]]
        stats[:, 0] = 1.0
    rng = np.random.default_rng(seed)
    output = np.empty((replicates, 3), dtype=float)
    for i in range(replicates):
        total = stats[rng.integers(0, len(stats), len(stats))].sum(axis=0)
        n, y_sum, y2_sum, abs_sum, sq_sum = total
        sst = y2_sum - y_sum**2 / n
        output[i, 0] = 1.0 - sq_sum / sst if sst > 0 else np.nan
        output[i, 1] = abs_sum / n
        output[i, 2] = math.sqrt(sq_sum / n)
    return output


def metric_table(predictions: pd.DataFrame, replicates: int, seed: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    group_columns = ["scheme", "condition", "model", "training_weighting", "recency_lambda", "target"]
    for keys, frame in predictions.groupby(group_columns, sort=False, dropna=False):
        base = dict(zip(group_columns, keys, strict=True))
        for lens in LENSES:
            weights = evaluation_weights(frame["doi_norm"].reset_index(drop=True), lens)
            point = weighted_metrics(
                frame["y_true"].to_numpy(dtype=float),
                frame["y_pred"].to_numpy(dtype=float),
                weights,
            )
            boot = metric_bootstrap(
                frame,
                lens,
                replicates,
                seed + sum(map(ord, str(keys) + lens)),
            )
            row = {
                **base,
                "evaluation_lens": lens,
                "records": len(frame),
                "DOI_groups": frame["doi_norm"].nunique(),
            }
            for j, metric in enumerate(["R2", "MAE", "RMSE"]):
                low, high = np.nanpercentile(boot[:, j], [2.5, 97.5])
                row[metric] = point[j]
                row[f"{metric}_CI_low"] = float(low)
                row[f"{metric}_CI_high"] = float(high)
            rows.append(row)
    return pd.DataFrame(rows)


def paired_bootstrap_comparison(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    lens: str,
    replicates: int,
    seed: int,
) -> tuple[dict[str, float], np.ndarray]:
    left = reference[["Ref_ID", "doi_norm", "y_true", "y_pred"]].rename(
        columns={"doi_norm": "doi_a", "y_true": "y_a", "y_pred": "pred_a"}
    )
    right = candidate[["Ref_ID", "doi_norm", "y_true", "y_pred"]].rename(
        columns={"doi_norm": "doi_b", "y_true": "y_b", "y_pred": "pred_b"}
    )
    paired = left.merge(right, on="Ref_ID", validate="one_to_one")
    if len(paired) != len(left) or len(paired) != len(right):
        raise AssertionError("Paired predictions do not cover identical records")
    if not paired["doi_a"].eq(paired["doi_b"]).all():
        raise AssertionError("DOI mismatch in paired predictions")
    if not np.allclose(paired["y_a"], paired["y_b"], rtol=0, atol=1e-12):
        raise AssertionError("Target mismatch in paired predictions")
    paired["y2"] = paired["y_a"] ** 2
    paired["abs_a"] = (paired["pred_a"] - paired["y_a"]).abs()
    paired["sq_a"] = (paired["pred_a"] - paired["y_a"]) ** 2
    paired["abs_b"] = (paired["pred_b"] - paired["y_a"]).abs()
    paired["sq_b"] = (paired["pred_b"] - paired["y_a"]) ** 2
    grouped = paired.groupby("doi_a", sort=False).agg(
        n=("y_a", "size"),
        y_sum=("y_a", "sum"),
        y2_sum=("y2", "sum"),
        abs_a=("abs_a", "sum"),
        sq_a=("sq_a", "sum"),
        abs_b=("abs_b", "sum"),
        sq_b=("sq_b", "sum"),
    )
    stats = grouped.to_numpy(dtype=float)
    if lens == "Publication-balanced":
        stats[:, 1:] /= stats[:, [0]]
        stats[:, 0] = 1.0

    def calculate(total: np.ndarray) -> np.ndarray:
        n, y_sum, y2_sum, abs_a, sq_a, abs_b, sq_b = total
        sst = y2_sum - y_sum**2 / n
        r2_a = 1.0 - sq_a / sst if sst > 0 else np.nan
        r2_b = 1.0 - sq_b / sst if sst > 0 else np.nan
        mae_a = abs_a / n
        mae_b = abs_b / n
        rmse_a = math.sqrt(sq_a / n)
        rmse_b = math.sqrt(sq_b / n)
        return np.array(
            [
                r2_a,
                r2_b,
                r2_b - r2_a,
                mae_a,
                mae_b,
                100.0 * (mae_b / mae_a - 1.0),
                rmse_a,
                rmse_b,
                100.0 * (rmse_b / rmse_a - 1.0),
            ]
        )

    point = calculate(stats.sum(axis=0))
    rng = np.random.default_rng(seed)
    boot = np.empty((replicates, len(point)), dtype=float)
    for i in range(replicates):
        boot[i] = calculate(
            stats[rng.integers(0, len(stats), len(stats))].sum(axis=0)
        )
    names = [
        "reference_R2",
        "candidate_R2",
        "delta_R2_candidate_minus_reference",
        "reference_MAE",
        "candidate_MAE",
        "MAE_change_percent_candidate_vs_reference",
        "reference_RMSE",
        "candidate_RMSE",
        "RMSE_change_percent_candidate_vs_reference",
    ]
    return dict(zip(names, point, strict=True)), boot


def paired_comparison_table(
    predictions: pd.DataFrame,
    replicates: int,
    seed: int,
) -> pd.DataFrame:
    comparisons = [
        (GROUPED_SCHEME, RF_FULL, CB_FULL),
        (CHRONO_SCHEME, RF_FULL, CB_FULL),
        (CHRONO_SCHEME, RF_FULL, RF_RECENCY),
        (CHRONO_SCHEME, CB_FULL, CB_RECENCY),
        (CHRONO_SCHEME, RF_FULL, CB_RECENCY),
    ]
    rows: list[dict[str, object]] = []
    for scheme, reference_condition, candidate_condition in comparisons:
        for target in TARGET_ORDER:
            ref = predictions.loc[
                predictions["scheme"].eq(scheme)
                & predictions["condition"].eq(reference_condition)
                & predictions["target"].eq(target)
            ]
            cand = predictions.loc[
                predictions["scheme"].eq(scheme)
                & predictions["condition"].eq(candidate_condition)
                & predictions["target"].eq(target)
            ]
            if ref.empty or cand.empty:
                continue
            for lens in LENSES:
                point, boot = paired_bootstrap_comparison(
                    ref,
                    cand,
                    lens,
                    replicates,
                    seed + sum(map(ord, scheme + reference_condition + candidate_condition + target + lens)),
                )
                row = {
                    "scheme": scheme,
                    "reference_condition": reference_condition,
                    "candidate_condition": candidate_condition,
                    "target": target,
                    "evaluation_lens": lens,
                    "records": len(ref),
                    "DOI_groups": ref["doi_norm"].nunique(),
                    **point,
                }
                for j, name in enumerate(point):
                    low, high = np.nanpercentile(boot[:, j], [2.5, 97.5])
                    row[f"{name}_CI_low"] = float(low)
                    row[f"{name}_CI_high"] = float(high)
                rows.append(row)
    return pd.DataFrame(rows)


def rolling_result_row(
    model: str,
    validation_year: int,
    recency_lambda: float,
    metadata: pd.DataFrame,
    targets: pd.DataFrame,
    valid_index: np.ndarray,
    prediction: np.ndarray,
) -> dict[str, object]:
    y_true = targets.iloc[valid_index]["PCE"].to_numpy(dtype=float)
    y_pred = prediction[:, TARGET_ORDER.index("PCE")]
    dois = metadata.iloc[valid_index]["doi_norm"].reset_index(drop=True)
    pub_weights = evaluation_weights(dois, "Publication-balanced")
    device_weights = np.ones(len(y_true), dtype=float)
    pub_metrics = weighted_metrics(y_true, y_pred, pub_weights)
    device_metrics = weighted_metrics(y_true, y_pred, device_weights)
    high = y_true >= 20.0
    return {
        "model": model,
        "validation_year": validation_year,
        "recency_lambda": recency_lambda,
        "half_life_years": np.inf if recency_lambda == 0 else math.log(2) / recency_lambda,
        "train_through_year": validation_year - 1,
        "validation_records": len(valid_index),
        "validation_DOI": dois.nunique(),
        "publication_balanced_PCE_R2": pub_metrics[0],
        "publication_balanced_PCE_MAE": pub_metrics[1],
        "publication_balanced_PCE_RMSE": pub_metrics[2],
        "device_level_PCE_R2": device_metrics[0],
        "device_level_PCE_MAE": device_metrics[1],
        "high_PCE_records": int(high.sum()),
        "high_PCE_mean_bias": float(np.mean(y_pred[high] - y_true[high])) if high.any() else np.nan,
        "high_PCE_MAE": float(np.mean(np.abs(y_pred[high] - y_true[high]))) if high.any() else np.nan,
    }


def select_lambda_one_se(rolling: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model, frame in rolling.groupby("model", sort=False):
        summary = frame.groupby("recency_lambda").agg(
            mean_publication_balanced_PCE_MAE=("publication_balanced_PCE_MAE", "mean"),
            sd_across_pseudo_futures=("publication_balanced_PCE_MAE", "std"),
            pseudo_future_count=("validation_year", "nunique"),
            mean_high_PCE_bias=("high_PCE_mean_bias", "mean"),
        ).reset_index()
        summary["se_across_pseudo_futures"] = (
            summary["sd_across_pseudo_futures"]
            / np.sqrt(summary["pseudo_future_count"])
        )
        raw_best_index = summary["mean_publication_balanced_PCE_MAE"].idxmin()
        raw_best = summary.loc[raw_best_index]
        threshold = (
            raw_best["mean_publication_balanced_PCE_MAE"]
            + raw_best["se_across_pseudo_futures"]
        )
        eligible = summary.loc[
            summary["mean_publication_balanced_PCE_MAE"] <= threshold + 1e-12
        ]
        selected_lambda = float(eligible["recency_lambda"].min())
        for _, row in summary.iterrows():
            rows.append(
                {
                    "model": model,
                    **row.to_dict(),
                    "raw_minimum_lambda": float(raw_best["recency_lambda"]),
                    "one_SE_threshold": float(threshold),
                    "selected_lambda_one_SE": selected_lambda,
                    "selected": bool(float(row["recency_lambda"]) == selected_lambda),
                }
            )
    return pd.DataFrame(rows)


def high_pce_table(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    chrono = predictions.loc[
        predictions["scheme"].eq(CHRONO_SCHEME)
        & predictions["target"].eq("PCE")
        & predictions["y_true"].ge(20.0)
    ]
    for condition, frame in chrono.groupby("condition", sort=False):
        rows.append(
            {
                "condition": condition,
                "records": len(frame),
                "DOI_groups": frame["doi_norm"].nunique(),
                "measured_PCE_mean": frame["y_true"].mean(),
                "predicted_PCE_mean": frame["y_pred"].mean(),
                "mean_bias_predicted_minus_measured": (
                    frame["y_pred"] - frame["y_true"]
                ).mean(),
                "MAE": (frame["y_pred"] - frame["y_true"]).abs().mean(),
                "RMSE": np.sqrt(np.mean((frame["y_pred"] - frame["y_true"]) ** 2)),
            }
        )
    return pd.DataFrame(rows)


def calibration_table(predictions: pd.DataFrame) -> pd.DataFrame:
    bins = [-np.inf, 5, 10, 15, 20, np.inf]
    labels = ["0–5", "5–10", "10–15", "15–20", "≥20"]
    work = predictions.loc[
        predictions["scheme"].eq(CHRONO_SCHEME)
        & predictions["target"].eq("PCE")
    ].copy()
    work["measured_PCE_bin"] = pd.cut(work["y_true"], bins=bins, labels=labels)
    return (
        work.groupby(["condition", "measured_PCE_bin"], observed=False)
        .agg(
            records=("Ref_ID", "size"),
            DOI_groups=("doi_norm", "nunique"),
            measured_PCE_mean=("y_true", "mean"),
            predicted_PCE_mean=("y_pred", "mean"),
        )
        .reset_index()
        .assign(
            mean_bias_predicted_minus_measured=lambda d: d["predicted_PCE_mean"]
            - d["measured_PCE_mean"]
        )
    )


def plot_summary(
    rolling_summary: pd.DataFrame,
    metrics: pd.DataFrame,
    comparisons: pd.DataFrame,
    calibration: pd.DataFrame,
    output_dir: Path,
) -> None:
    colors = {
        RF_FULL: "#456A8A",
        CB_FULL: "#E28E2C",
        RF_RECENCY: "#4FAF8B",
        CB_RECENCY: "#C43C39",
    }
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.6), constrained_layout=True)

    for model, frame in rolling_summary.groupby("model", sort=False):
        label = "Random Forest" if model == "Random Forest" else "CatBoost"
        axes[0, 0].plot(
            frame["recency_lambda"],
            frame["mean_publication_balanced_PCE_MAE"],
            marker="o",
            label=label,
        )
        selected = frame.loc[frame["selected"]]
        axes[0, 0].scatter(
            selected["recency_lambda"],
            selected["mean_publication_balanced_PCE_MAE"],
            s=90,
            facecolors="none",
            edgecolors="black",
            linewidths=1.2,
        )
    axes[0, 0].set_xlabel(r"Recency strength $\lambda$ (year$^{-1}$)")
    axes[0, 0].set_ylabel("Mean rolling-origin publication-balanced PCE MAE")
    axes[0, 0].set_title("(a) Historical-only recency selection", loc="left", fontweight="bold")
    axes[0, 0].legend(frameon=False)

    chrono_pce = metrics.loc[
        metrics["scheme"].eq(CHRONO_SCHEME)
        & metrics["target"].eq("PCE")
        & metrics["evaluation_lens"].eq("Device-level")
    ].copy()
    order = [RF_FULL, CB_FULL, RF_RECENCY, CB_RECENCY]
    chrono_pce = chrono_pce.set_index("condition").reindex(order)
    x = np.arange(len(order))
    axes[0, 1].bar(x, chrono_pce["R2"], color=[colors[c] for c in order])
    axes[0, 1].set_xticks(x, ["RF\nDOI", "CB\nDOI", "RF\nrecency", "CB\nrecency"])
    axes[0, 1].set_ylabel(r"Chronological PCE $R^2$")
    axes[0, 1].set_title("(b) Independent future performance", loc="left", fontweight="bold")
    for xx, value in zip(x, chrono_pce["R2"], strict=True):
        axes[0, 1].text(xx, value + 0.01, f"{value:.3f}", ha="center", fontsize=8)

    comp = comparisons.loc[
        comparisons["scheme"].eq(CHRONO_SCHEME)
        & comparisons["evaluation_lens"].eq("Publication-balanced")
        & comparisons["reference_condition"].eq(RF_FULL)
        & comparisons["candidate_condition"].isin([CB_FULL, RF_RECENCY, CB_RECENCY])
    ].copy()
    for condition, frame in comp.groupby("candidate_condition", sort=False):
        frame = frame.set_index("target").reindex(TARGET_ORDER)
        axes[1, 0].plot(
            TARGET_ORDER,
            frame["MAE_change_percent_candidate_vs_reference"],
            marker="o",
            label=condition.replace(" | Full DOI", "").replace("Random Forest", "RF").replace("CatBoost", "CB"),
            color=colors[condition],
        )
    axes[1, 0].axhline(0, color="black", ls="--", lw=0.9)
    axes[1, 0].set_ylabel("Publication-balanced MAE change vs RF full DOI (%)")
    axes[1, 0].set_title("(c) Target-wise trade-off", loc="left", fontweight="bold")
    axes[1, 0].legend(frameon=False, fontsize=8)

    for condition in order:
        frame = calibration.loc[calibration["condition"].eq(condition)]
        axes[1, 1].plot(
            frame["measured_PCE_mean"],
            frame["predicted_PCE_mean"],
            marker="o",
            label=condition.replace("Random Forest", "RF").replace("CatBoost", "CB"),
            color=colors[condition],
        )
    axes[1, 1].plot([0, 25], [0, 25], color="black", ls="--", lw=0.9)
    axes[1, 1].set_xlim(0, 25)
    axes[1, 1].set_ylim(0, 25)
    axes[1, 1].set_xlabel("Measured PCE (%)")
    axes[1, 1].set_ylabel("Mean predicted PCE (%)")
    axes[1, 1].set_title("(d) Chronological PCE calibration", loc="left", fontweight="bold")
    axes[1, 1].legend(frameon=False, fontsize=7)

    fig.suptitle("CatBoost robustness and DOI-balanced temporal-recency weighting", fontweight="bold")
    for suffix in ["png", "pdf", "svg"]:
        kwargs = {"dpi": 600} if suffix == "png" else {}
        fig.savefig(output_dir / f"FigureS_CatBoost_Recency.{suffix}", bbox_inches="tight", **kwargs)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    config = ModelConfig()
    candidates = [dict(item) for item in CB_CANDIDATES]
    lambda_grid = list(LAMBDA_GRID)
    rolling_years = list(ROLLING_YEARS)
    inner_folds = 2
    bootstrap_replicates = config.bootstrap_replicates
    grouped_folds_to_run: list[int] | None = None
    if args.quick:
        config = ModelConfig(
            grouped_folds=2,
            seed=config.seed,
            bootstrap_replicates=50,
            token_min_df=20,
            token_max_features=800,
            rf_estimators=20,
            elastic_max_iter=2000,
        )
        candidates = [{**candidates[0], "iterations": 80}]
        lambda_grid = [0.0, 0.10]
        rolling_years = [2017, 2018]
        bootstrap_replicates = 50
        grouped_folds_to_run = [1, 2]

    baseline_manifest = json.loads(args.baseline_manifest.read_text(encoding="utf-8"))
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
    targets["FF"] *= 100.0
    features, numeric_features = build_features(raw)
    forbidden = {"Ref_ID", "doi_norm", "publication_year", *TARGET_ORDER}
    if forbidden.intersection(features.columns):
        raise AssertionError("Forbidden identifiers or targets entered model features")

    split_manifest = pd.read_csv(args.split_manifest)
    if not split_manifest["Ref_ID"].equals(metadata["Ref_ID"]):
        raise AssertionError("Frozen split manifest does not align to cohort")
    if not split_manifest["doi_norm"].astype("string").eq(metadata["doi_norm"].astype("string")).all():
        raise AssertionError("DOI labels differ from frozen split manifest")
    if len(metadata) != 33175 or metadata["doi_norm"].nunique() != 6368:
        raise AssertionError("Frozen cohort size or DOI count changed")

    selection_rows: list[dict[str, object]] = []
    diagnostic_rows: list[dict[str, object]] = []
    generated_predictions: list[pd.DataFrame] = []

    grouped_folds = sorted(int(x) for x in split_manifest["grouped_fold"].unique())
    if grouped_folds_to_run is not None:
        grouped_folds = grouped_folds_to_run
    for fold in grouped_folds:
        test_index = np.flatnonzero(split_manifest["grouped_fold"].eq(fold).to_numpy())
        train_index = np.flatnonzero(split_manifest["grouped_fold"].ne(fold).to_numpy())
        train_doi = set(metadata.iloc[train_index]["doi_norm"])
        test_doi = set(metadata.iloc[test_index]["doi_norm"])
        if train_doi.intersection(test_doi):
            raise AssertionError(f"DOI leakage in grouped fold {fold}")
        print(f"[outer grouped fold {fold}/{len(grouped_folds)}] tuning CatBoost", flush=True)
        candidate, iterations = select_catboost_config(
            f"outer_grouped_fold_{fold}",
            features,
            targets,
            metadata,
            train_index,
            numeric_features,
            config,
            candidates,
            inner_folds,
            selection_rows,
        )
        prediction, diagnostics = fit_catboost_partition(
            features,
            targets,
            metadata,
            train_index,
            test_index,
            numeric_features,
            config,
            candidate,
            iterations,
            0.0,
            fold,
        )
        diagnostic_rows.append(
            {
                "context": f"outer_grouped_fold_{fold}",
                "condition": CB_FULL,
                "candidate": candidate["candidate"],
                "iterations": iterations,
                **diagnostics,
            }
        )
        append_multioutput_predictions(
            generated_predictions,
            metadata,
            targets,
            test_index,
            prediction,
            GROUPED_SCHEME,
            f"fold_{fold}",
            CB_FULL,
            "CatBoost",
            FULL_WEIGHTING,
            0.0,
        )

    chrono_train = np.flatnonzero(
        split_manifest["chronological_role"].eq("train_through_2018").to_numpy()
    )
    chrono_test = np.flatnonzero(
        split_manifest["chronological_role"].eq("test_2019_onward").to_numpy()
    )
    if set(metadata.iloc[chrono_train]["doi_norm"]).intersection(
        set(metadata.iloc[chrono_test]["doi_norm"])
    ):
        raise AssertionError("DOI leakage in chronological split")

    print("[chronological training] tuning CatBoost on <=2018 only", flush=True)
    chrono_candidate, chrono_iterations = select_catboost_config(
        "chronological_train_through_2018",
        features,
        targets,
        metadata,
        chrono_train,
        numeric_features,
        config,
        candidates,
        inner_folds,
        selection_rows,
    )
    cb_full_prediction, diagnostics = fit_catboost_partition(
        features,
        targets,
        metadata,
        chrono_train,
        chrono_test,
        numeric_features,
        config,
        chrono_candidate,
        chrono_iterations,
        0.0,
        2019,
    )
    diagnostic_rows.append(
        {
            "context": "chronological_train_through_2018",
            "condition": CB_FULL,
            "candidate": chrono_candidate["candidate"],
            "iterations": chrono_iterations,
            **diagnostics,
        }
    )
    append_multioutput_predictions(
        generated_predictions,
        metadata,
        targets,
        chrono_test,
        cb_full_prediction,
        CHRONO_SCHEME,
        "holdout_2019_onward",
        CB_FULL,
        "CatBoost",
        FULL_WEIGHTING,
        0.0,
    )

    rolling_rows: list[dict[str, object]] = []
    for validation_year in rolling_years:
        train_index = np.flatnonzero(metadata["publication_year"].lt(validation_year).to_numpy())
        valid_index = np.flatnonzero(metadata["publication_year"].eq(validation_year).to_numpy())
        if not len(valid_index):
            raise AssertionError(f"No records in rolling validation year {validation_year}")
        if set(metadata.iloc[train_index]["doi_norm"]).intersection(
            set(metadata.iloc[valid_index]["doi_norm"])
        ):
            raise AssertionError(f"DOI leakage in rolling year {validation_year}")
        print(f"[rolling origin {validation_year}] tuning historical CatBoost", flush=True)
        rolling_candidate, rolling_iterations = select_catboost_config(
            f"rolling_train_before_{validation_year}",
            features,
            targets,
            metadata,
            train_index,
            numeric_features,
            config,
            candidates,
            inner_folds,
            selection_rows,
        )
        for recency_lambda in lambda_grid:
            print(
                f"[rolling {validation_year}] lambda={recency_lambda:.3f} RF + CatBoost",
                flush=True,
            )
            rf_prediction, rf_diag = fit_rf_partition(
                features,
                targets,
                metadata,
                train_index,
                valid_index,
                numeric_features,
                config,
                recency_lambda,
                validation_year,
            )
            rolling_rows.append(
                rolling_result_row(
                    "Random Forest",
                    validation_year,
                    recency_lambda,
                    metadata,
                    targets,
                    valid_index,
                    rf_prediction,
                )
            )
            cb_prediction, cb_diag = fit_catboost_partition(
                features,
                targets,
                metadata,
                train_index,
                valid_index,
                numeric_features,
                config,
                rolling_candidate,
                rolling_iterations,
                recency_lambda,
                validation_year,
            )
            rolling_rows.append(
                rolling_result_row(
                    "CatBoost",
                    validation_year,
                    recency_lambda,
                    metadata,
                    targets,
                    valid_index,
                    cb_prediction,
                )
            )
            diagnostic_rows.extend(
                [
                    {
                        "context": f"rolling_{validation_year}",
                        "condition": "Random Forest rolling selection",
                        "candidate": "frozen RF config",
                        "iterations": config.rf_estimators,
                        **rf_diag,
                    },
                    {
                        "context": f"rolling_{validation_year}",
                        "condition": "CatBoost rolling selection",
                        "candidate": rolling_candidate["candidate"],
                        "iterations": rolling_iterations,
                        **cb_diag,
                    },
                ]
            )

    rolling = pd.DataFrame(rolling_rows)
    rolling_summary = select_lambda_one_se(rolling)
    selected_rf_lambda = float(
        rolling_summary.loc[
            rolling_summary["model"].eq("Random Forest")
            & rolling_summary["selected"]
        , "selected_lambda_one_SE"].iloc[0]
    )
    selected_cb_lambda = float(
        rolling_summary.loc[
            rolling_summary["model"].eq("CatBoost")
            & rolling_summary["selected"]
        , "selected_lambda_one_SE"].iloc[0]
    )
    print(
        f"[historical-only selection] RF lambda={selected_rf_lambda:.3f}; "
        f"CatBoost lambda={selected_cb_lambda:.3f}",
        flush=True,
    )

    rf_recency_prediction, diagnostics = fit_rf_partition(
        features,
        targets,
        metadata,
        chrono_train,
        chrono_test,
        numeric_features,
        config,
        selected_rf_lambda,
        2019,
    )
    diagnostic_rows.append(
        {
            "context": "chronological_final",
            "condition": RF_RECENCY,
            "candidate": "frozen RF config",
            "iterations": config.rf_estimators,
            **diagnostics,
        }
    )
    append_multioutput_predictions(
        generated_predictions,
        metadata,
        targets,
        chrono_test,
        rf_recency_prediction,
        CHRONO_SCHEME,
        "holdout_2019_onward",
        RF_RECENCY,
        "Random Forest",
        "Full DOI + historical recency",
        selected_rf_lambda,
    )

    cb_recency_prediction, diagnostics = fit_catboost_partition(
        features,
        targets,
        metadata,
        chrono_train,
        chrono_test,
        numeric_features,
        config,
        chrono_candidate,
        chrono_iterations,
        selected_cb_lambda,
        2019,
    )
    diagnostic_rows.append(
        {
            "context": "chronological_final",
            "condition": CB_RECENCY,
            "candidate": chrono_candidate["candidate"],
            "iterations": chrono_iterations,
            **diagnostics,
        }
    )
    append_multioutput_predictions(
        generated_predictions,
        metadata,
        targets,
        chrono_test,
        cb_recency_prediction,
        CHRONO_SCHEME,
        "holdout_2019_onward",
        CB_RECENCY,
        "CatBoost",
        "Full DOI + historical recency",
        selected_cb_lambda,
    )

    generated = pd.concat(generated_predictions, ignore_index=True)
    frozen = pd.read_csv(args.frozen_weighting_predictions)
    frozen = frozen.loc[
        frozen["training_weighting"].eq(FULL_WEIGHTING)
        & frozen["model"].eq("Random Forest")
        & frozen["scheme"].isin([GROUPED_SCHEME, CHRONO_SCHEME])
    ].copy()
    if grouped_folds_to_run is not None:
        frozen = frozen.loc[
            frozen["scheme"].eq(CHRONO_SCHEME)
            | frozen["fold"].isin({f"fold_{x}" for x in grouped_folds})
        ].copy()
    frozen["condition"] = RF_FULL
    frozen["recency_lambda"] = 0.0
    frozen = frozen[
        [
            "Ref_ID",
            "doi_norm",
            "publication_year",
            "scheme",
            "fold",
            "condition",
            "model",
            "training_weighting",
            "recency_lambda",
            "target",
            "y_true",
            "y_pred",
        ]
    ]
    predictions = pd.concat([frozen, generated], ignore_index=True)
    if predictions.duplicated(["Ref_ID", "scheme", "condition", "target"]).any():
        raise AssertionError("Duplicate prediction keys detected")
    if not np.isfinite(predictions[["y_true", "y_pred"]].to_numpy()).all():
        raise AssertionError("Non-finite prediction values detected")

    metrics = metric_table(predictions, bootstrap_replicates, config.seed)
    comparisons = paired_comparison_table(predictions, bootstrap_replicates, config.seed)
    high_pce = high_pce_table(predictions)
    calibration = calibration_table(predictions)
    selection = pd.DataFrame(selection_rows)
    diagnostics = pd.DataFrame(diagnostic_rows)

    predictions["residual"] = predictions["y_pred"] - predictions["y_true"]
    predictions.to_csv(args.output_dir / "catboost_recency_predictions.csv.gz", index=False)
    selection.to_csv(args.output_dir / "catboost_model_selection.csv", index=False)
    rolling.to_csv(args.output_dir / "rolling_origin_recency_lambda_results.csv", index=False)
    rolling_summary.to_csv(args.output_dir / "rolling_origin_recency_lambda_selection.csv", index=False)
    metrics.to_csv(args.output_dir / "catboost_recency_metrics.csv", index=False)
    comparisons.to_csv(args.output_dir / "catboost_recency_paired_comparison.csv", index=False)
    high_pce.to_csv(args.output_dir / "chronological_high_PCE_comparison.csv", index=False)
    calibration.to_csv(args.output_dir / "chronological_PCE_catboost_recency_calibration.csv", index=False)
    diagnostics.to_csv(args.output_dir / "catboost_recency_fit_diagnostics.csv", index=False)
    plot_summary(rolling_summary, metrics, comparisons, calibration, args.output_dir)

    verification = {
        "status": "passed",
        "records": int(len(metadata)),
        "DOI_groups": int(metadata["doi_norm"].nunique()),
        "raw_sha256_matches_frozen": sha256(args.raw) == baseline_manifest["inputs"]["raw_sha256"],
        "cohort_sha256_matches_frozen": sha256(args.cohort) == baseline_manifest["inputs"]["cohort_sha256"],
        "forbidden_features_absent": not bool(forbidden.intersection(features.columns)),
        "duplicate_prediction_keys": int(
            predictions.duplicated(["Ref_ID", "scheme", "condition", "target"]).sum()
        ),
        "nonfinite_prediction_values": int(
            (~np.isfinite(predictions[["y_true", "y_pred"]].to_numpy())).sum()
        ),
        "grouped_DOI_overlap": 0,
        "chronological_DOI_overlap": 0,
        "selected_RF_lambda": selected_rf_lambda,
        "selected_CatBoost_lambda": selected_cb_lambda,
        "rolling_years": rolling_years,
        "2019_2021_holdout_used_for_lambda_selection": False,
        "prediction_rows": int(len(predictions)),
    }
    (args.output_dir / "verification_report.json").write_text(
        json.dumps(verification, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    manifest = {
        "status": "completed",
        "runtime_seconds": time.perf_counter() - started,
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "catboost": catboost.__version__,
        },
        "inputs": {
            "raw": str(args.raw.resolve()),
            "raw_sha256": sha256(args.raw),
            "cohort": str(args.cohort.resolve()),
            "cohort_sha256": sha256(args.cohort),
            "split_manifest": str(args.split_manifest.resolve()),
            "split_manifest_sha256": sha256(args.split_manifest),
            "frozen_weighting_predictions": str(args.frozen_weighting_predictions.resolve()),
            "frozen_weighting_predictions_sha256": sha256(args.frozen_weighting_predictions),
        },
        "config": asdict(config),
        "catboost_candidates": candidates,
        "catboost_inner_group_folds": inner_folds,
        "recency_weight": "w_i proportional to n_DOI^(-1) * exp[-lambda * (Tmax - Ti)], normalized to mean 1 in each training partition",
        "lambda_grid": lambda_grid,
        "rolling_origin": [
            {"train": f"year < {year}", "validate": f"year = {year}"}
            for year in rolling_years
        ],
        "lambda_selection_rule": "smallest lambda within one standard error of the minimum mean publication-balanced PCE MAE across rolling pseudo-futures",
        "selected_RF_lambda": selected_rf_lambda,
        "selected_CatBoost_lambda": selected_cb_lambda,
        "final_chronological_CatBoost_candidate": chrono_candidate,
        "final_chronological_CatBoost_iterations": chrono_iterations,
        "bootstrap_replicates": bootstrap_replicates,
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(verification, indent=2), flush=True)


if __name__ == "__main__":
    main()
