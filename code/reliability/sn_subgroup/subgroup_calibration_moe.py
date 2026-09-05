#!/usr/bin/env python3
"""Leakage-controlled subgroup calibration and mixture-of-experts analysis.

The analysis targets two sparse absorber domains identified before modelling:
Sn-only B sites and mixed Pb-Sn B sites.  The 2019--2021 outcomes are held out
until every calibrator, expert hyperparameter, and convex mixture weight has
been selected using publication-disjoint predictions from data through 2018.
The frozen full-DOI-balanced chronological Random Forest remains the reference.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import platform
import sys
import time
import zlib
from dataclasses import asdict, replace
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


TARGETS = ["PCE", "Voc", "Jsc", "FF"]
TARGET_UNITS = {
    "PCE": "%-point",
    "Voc": "V",
    "Jsc": "mA cm$^{-2}$",
    "FF": "%-point",
}
DOMAINS = {
    "Sn-only": "Sn (no Pb)",
    "Mixed Pb-Sn": "Pb+Sn",
}
GLOBAL_METHOD = "Frozen global"
METHODS = [
    GLOBAL_METHOD,
    "Subgroup calibrator",
    "Domain expert",
    "Convex mixture",
    "Development-selected policy",
]
BOOTSTRAP_REPLICATES = 1000
SEED = 20260829
CHRONO_SCHEME = "Chronological >2018"
FULL_WEIGHTING = "Full 1/n_DOI"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", required=True, type=Path)
    parser.add_argument("--cohort", required=True, type=Path)
    parser.add_argument("--assignments", required=True, type=Path)
    parser.add_argument("--composition-predictions", required=True, type=Path)
    parser.add_argument("--split-manifest", required=True, type=Path)
    parser.add_argument("--weighting-predictions", required=True, type=Path)
    parser.add_argument("--baseline-code-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_seed(*values: object) -> int:
    payload = "|".join(map(str, values)).encode("utf-8")
    return SEED + int(zlib.crc32(payload) % 100_000)


def publication_weights(dois: pd.Series) -> np.ndarray:
    counts = dois.value_counts()
    weights = dois.map(counts).to_numpy(dtype=float) ** -1.0
    return weights / weights.mean()


def weighted_location_scale(
    targets: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    mean = np.average(targets, axis=0, weights=weights)
    variance = np.average((targets - mean) ** 2, axis=0, weights=weights)
    std = np.sqrt(np.maximum(variance, 0.0))
    return mean, np.where(std > 0, std, 1.0)


def as_matrix(value: object) -> object:
    if sparse.issparse(value):
        return value.tocsr().astype(np.float32)
    return np.asarray(value, dtype=np.float32)


def fit_predict_forest(
    features: pd.DataFrame,
    targets: pd.DataFrame,
    metadata: pd.DataFrame,
    train_index: np.ndarray,
    test_index: np.ndarray,
    numeric_features: list[str],
    config: object,
    make_preprocessor: object,
    random_state: int,
) -> np.ndarray:
    processor = make_preprocessor(config, numeric_features)
    x_train = as_matrix(processor.fit_transform(features.iloc[train_index]))
    x_test = as_matrix(processor.transform(features.iloc[test_index]))
    y_train = targets.iloc[train_index].to_numpy(dtype=float)
    weights = publication_weights(metadata.iloc[train_index]["doi_norm"])
    y_mean, y_std = weighted_location_scale(y_train, weights)
    y_scaled = (y_train - y_mean) / y_std
    forest = RandomForestRegressor(
        n_estimators=config.rf_estimators,
        max_features=config.rf_max_features,
        min_samples_leaf=config.rf_min_samples_leaf,
        max_samples=config.rf_max_samples,
        bootstrap=True,
        random_state=random_state,
        n_jobs=-1,
    )
    forest.fit(x_train, y_scaled, sample_weight=weights)
    prediction = forest.predict(x_test) * y_std + y_mean
    del processor, x_train, x_test, forest
    gc.collect()
    return prediction


def metric_values(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    dois: np.ndarray,
    evaluation: str,
    multiplicity: dict[str, int] | None = None,
) -> dict[str, float]:
    frame = pd.DataFrame(
        {"y": y_true, "p": y_pred, "doi": dois.astype(str)}
    )
    if multiplicity is None:
        cluster_multiplier = np.ones(len(frame), dtype=float)
    else:
        cluster_multiplier = frame["doi"].map(multiplicity).fillna(0).to_numpy(float)
    if evaluation == "Publication-balanced":
        counts = frame["doi"].value_counts()
        base = frame["doi"].map(counts).to_numpy(float) ** -1.0
    elif evaluation == "Device-level":
        base = np.ones(len(frame), dtype=float)
    else:
        raise ValueError(evaluation)
    weights = base * cluster_multiplier
    valid = np.isfinite(frame["y"]) & np.isfinite(frame["p"]) & (weights > 0)
    y = frame.loc[valid, "y"].to_numpy(float)
    p = frame.loc[valid, "p"].to_numpy(float)
    w = weights[valid]
    mean_y = np.average(y, weights=w)
    residual = p - y
    mae = np.average(np.abs(residual), weights=w)
    mse = np.average(residual**2, weights=w)
    sst = np.sum(w * (y - mean_y) ** 2)
    sse = np.sum(w * residual**2)
    r2 = 1.0 - sse / sst if sst > 0 else np.nan
    return {
        "R2": float(r2),
        "MAE": float(mae),
        "RMSE": float(math.sqrt(mse)),
        "bias": float(np.average(residual, weights=w)),
    }


def cluster_bootstrap(
    frame: pd.DataFrame,
    evaluation: str,
    replicates: int,
    seed: int,
) -> pd.DataFrame:
    groups = frame["doi_norm"].astype(str).unique()
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float]] = []
    y = frame["y_true"].to_numpy(float)
    p = frame["y_pred"].to_numpy(float)
    d = frame["doi_norm"].astype(str).to_numpy()
    for replicate in range(replicates):
        sampled = rng.choice(groups, size=len(groups), replace=True)
        unique, count = np.unique(sampled, return_counts=True)
        mult = dict(zip(unique.tolist(), count.tolist()))
        values = metric_values(y, p, d, evaluation, mult)
        values["replicate"] = replicate
        rows.append(values)
    return pd.DataFrame(rows)


def paired_cluster_bootstrap(
    method: pd.DataFrame,
    reference: pd.DataFrame,
    evaluation: str,
    replicates: int,
    seed: int,
) -> pd.DataFrame:
    left = method[["Ref_ID", "doi_norm", "y_true", "y_pred"]].rename(
        columns={"y_pred": "method_pred"}
    )
    right = reference[["Ref_ID", "doi_norm", "y_true", "y_pred"]].rename(
        columns={"y_pred": "reference_pred", "y_true": "reference_y"}
    )
    paired = left.merge(right, on=["Ref_ID", "doi_norm"], validate="one_to_one")
    if not np.allclose(paired["y_true"], paired["reference_y"], atol=0, rtol=0):
        raise AssertionError("Paired outcomes differ")
    groups = paired["doi_norm"].astype(str).unique()
    rng = np.random.default_rng(seed)
    output: list[dict[str, float]] = []
    y = paired["y_true"].to_numpy(float)
    d = paired["doi_norm"].astype(str).to_numpy()
    pm = paired["method_pred"].to_numpy(float)
    pr = paired["reference_pred"].to_numpy(float)
    for replicate in range(replicates):
        sampled = rng.choice(groups, size=len(groups), replace=True)
        unique, count = np.unique(sampled, return_counts=True)
        mult = dict(zip(unique.tolist(), count.tolist()))
        mv = metric_values(y, pm, d, evaluation, mult)
        rv = metric_values(y, pr, d, evaluation, mult)
        output.append(
            {
                "replicate": replicate,
                "delta_R2": mv["R2"] - rv["R2"],
                "delta_MAE": mv["MAE"] - rv["MAE"],
                "MAE_change_percent": 100.0 * (mv["MAE"] / rv["MAE"] - 1.0),
                "delta_RMSE": mv["RMSE"] - rv["RMSE"],
                "delta_absolute_bias": abs(mv["bias"]) - abs(rv["bias"]),
            }
        )
    return pd.DataFrame(output)


def per_doi_mae(
    y_true: np.ndarray, y_pred: np.ndarray, dois: np.ndarray
) -> pd.Series:
    frame = pd.DataFrame(
        {"doi": dois.astype(str), "ae": np.abs(y_pred - y_true)}
    )
    return frame.groupby("doi", sort=False)["ae"].mean()


def choose_one_se(
    candidate_errors: dict[str, pd.Series], simplicity_order: list[str]
) -> tuple[str, pd.DataFrame]:
    summary = []
    for name, errors in candidate_errors.items():
        summary.append(
            {
                "candidate": name,
                "mean_DOI_MAE": float(errors.mean()),
                "SE_DOI_MAE": float(errors.std(ddof=1) / math.sqrt(len(errors))),
                "n_DOI": int(len(errors)),
            }
        )
    table = pd.DataFrame(summary)
    best_row = table.loc[table["mean_DOI_MAE"].idxmin()]
    threshold = float(best_row["mean_DOI_MAE"] + best_row["SE_DOI_MAE"])
    eligible = set(table.loc[table["mean_DOI_MAE"] <= threshold, "candidate"])
    selected = next(name for name in simplicity_order if name in eligible)
    table["minimum_mean_plus_one_SE"] = threshold
    table["within_one_SE"] = table["candidate"].isin(eligible)
    table["selected"] = table["candidate"].eq(selected)
    return selected, table


def fit_calibrator(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    weights: np.ndarray,
    kind: str,
    ridge: float | None = None,
) -> dict[str, float | str]:
    if kind == "identity":
        return {
            "kind": kind,
            "ridge": None,
            "x_center": 0.0,
            "x_scale": 1.0,
            "y_scale": 1.0,
            "delta_intercept": 0.0,
            "delta_slope": 0.0,
        }
    x_center = float(np.average(y_pred, weights=weights))
    x_scale = float(np.sqrt(np.average((y_pred - x_center) ** 2, weights=weights)))
    y_scale = float(np.sqrt(np.average((y_true - np.average(y_true, weights=weights)) ** 2, weights=weights)))
    x_scale = x_scale if x_scale > 1e-12 else 1.0
    y_scale = y_scale if y_scale > 1e-12 else 1.0
    residual = (y_true - y_pred) / y_scale
    if kind == "bias":
        delta_intercept = float(np.average(residual, weights=weights))
        delta_slope = 0.0
    elif kind == "affine":
        z = (y_pred - x_center) / x_scale
        design = np.column_stack([np.ones(len(z)), z])
        w = weights / weights.mean()
        penalty = float(ridge if ridge is not None else 0.0)
        lhs = design.T @ (w[:, None] * design) / w.sum()
        lhs += penalty * np.eye(2)
        rhs = design.T @ (w * residual) / w.sum()
        beta = np.linalg.solve(lhs, rhs)
        delta_intercept, delta_slope = map(float, beta)
    else:
        raise ValueError(kind)
    return {
        "kind": kind,
        "ridge": float(ridge) if ridge is not None else None,
        "x_center": x_center,
        "x_scale": x_scale,
        "y_scale": y_scale,
        "delta_intercept": delta_intercept,
        "delta_slope": delta_slope,
    }


def apply_calibrator(y_pred: np.ndarray, params: dict[str, object]) -> np.ndarray:
    z = (y_pred - float(params["x_center"])) / float(params["x_scale"])
    correction = float(params["y_scale"]) * (
        float(params["delta_intercept"]) + float(params["delta_slope"]) * z
    )
    return y_pred + correction


def calibrator_candidate_specifications() -> list[tuple[str, str, float | None]]:
    values = [("Identity", "identity", None), ("Bias-only", "bias", None)]
    for ridge in [10.0, 1.0, 0.1, 0.01, 0.001, 0.0]:
        values.append((f"Affine ridge={ridge:g}", "affine", ridge))
    return values


def calibration_crossfit(
    frame: pd.DataFrame,
) -> tuple[str, np.ndarray, dict[str, object], pd.DataFrame]:
    specifications = calibrator_candidate_specifications()
    candidate_predictions: dict[str, np.ndarray] = {}
    candidate_errors: dict[str, pd.Series] = {}
    y = frame["y_true"].to_numpy(float)
    x = frame["global_pred"].to_numpy(float)
    dois = frame["doi_norm"].astype(str).to_numpy()
    folds = frame["development_fold"].to_numpy(int)
    for name, kind, ridge in specifications:
        prediction = np.empty(len(frame), dtype=float)
        for fold in sorted(np.unique(folds)):
            train = folds != fold
            test = folds == fold
            params = fit_calibrator(
                y[train], x[train], publication_weights(pd.Series(dois[train])), kind, ridge
            )
            prediction[test] = apply_calibrator(x[test], params)
        candidate_predictions[name] = prediction
        candidate_errors[name] = per_doi_mae(y, prediction, dois)
    simplicity = [
        "Identity",
        "Bias-only",
        "Affine ridge=10",
        "Affine ridge=1",
        "Affine ridge=0.1",
        "Affine ridge=0.01",
        "Affine ridge=0.001",
        "Affine ridge=0",
    ]
    selected, selection = choose_one_se(candidate_errors, simplicity)
    selected_spec = next(spec for spec in specifications if spec[0] == selected)
    final_params = fit_calibrator(
        y,
        x,
        publication_weights(frame["doi_norm"]),
        selected_spec[1],
        selected_spec[2],
    )
    final_params["candidate"] = selected
    return selected, candidate_predictions[selected], final_params, selection


def expert_candidate_configs(base: object, quick: bool) -> dict[str, object]:
    trees = 60 if quick else 200
    return {
        "Expert conservative": replace(
            base,
            ohe_min_frequency=5,
            ohe_max_categories=128,
            token_min_df=5,
            token_max_features=1000,
            rf_estimators=trees,
            rf_max_features=0.35,
            rf_min_samples_leaf=8,
            rf_max_samples=0.85,
        ),
        "Expert moderate": replace(
            base,
            ohe_min_frequency=3,
            ohe_max_categories=128,
            token_min_df=3,
            token_max_features=1250,
            rf_estimators=trees,
            rf_max_features=0.50,
            rf_min_samples_leaf=4,
            rf_max_samples=0.85,
        ),
        "Expert frozen spec": replace(base, rf_estimators=trees),
        "Expert flexible": replace(
            base,
            ohe_min_frequency=2,
            ohe_max_categories=192,
            token_min_df=2,
            token_max_features=1500,
            rf_estimators=trees,
            rf_max_features=0.50,
            rf_min_samples_leaf=2,
            rf_max_samples=0.85,
        ),
    }


def select_expert(
    domain: str,
    domain_hist_index: np.ndarray,
    features: pd.DataFrame,
    targets: pd.DataFrame,
    metadata: pd.DataFrame,
    numeric_features: list[str],
    configs: dict[str, object],
    make_preprocessor: object,
) -> tuple[str, np.ndarray, pd.DataFrame]:
    folds = metadata.iloc[domain_hist_index]["development_fold"].to_numpy(int)
    candidate_predictions: dict[str, np.ndarray] = {}
    candidate_composite_errors: dict[str, pd.Series] = {}
    y = targets.iloc[domain_hist_index].to_numpy(float)
    dois = metadata.iloc[domain_hist_index]["doi_norm"].astype(str).to_numpy()
    scales = np.array(
        [
            max(np.quantile(y[:, j], 0.75) - np.quantile(y[:, j], 0.25), 1e-8)
            for j in range(len(TARGETS))
        ]
    )
    for candidate_index, (name, config) in enumerate(configs.items()):
        prediction = np.empty_like(y, dtype=float)
        for fold in sorted(np.unique(folds)):
            train_local = np.flatnonzero(folds != fold)
            test_local = np.flatnonzero(folds == fold)
            train_global = domain_hist_index[train_local]
            test_global = domain_hist_index[test_local]
            prediction[test_local] = fit_predict_forest(
                features,
                targets,
                metadata,
                train_global,
                test_global,
                numeric_features,
                config,
                make_preprocessor,
                stable_seed(domain, name, fold, candidate_index),
            )
        candidate_predictions[name] = prediction
        normalized_ae = np.abs(prediction - y) / scales
        error_frame = pd.DataFrame(normalized_ae, columns=TARGETS)
        error_frame["doi"] = dois
        candidate_composite_errors[name] = error_frame.groupby("doi")[TARGETS].mean().mean(axis=1)
    simplicity = [
        "Expert conservative",
        "Expert moderate",
        "Expert frozen spec",
        "Expert flexible",
    ]
    selected, selection = choose_one_se(candidate_composite_errors, simplicity)
    selection["selection_loss"] = "Mean DOI-balanced MAE / historical target IQR across four targets"
    return selected, candidate_predictions[selected], selection


def select_blend_alpha(
    y_true: np.ndarray,
    global_pred: np.ndarray,
    expert_pred: np.ndarray,
    dois: np.ndarray,
) -> tuple[float, np.ndarray, pd.DataFrame]:
    errors: dict[str, pd.Series] = {}
    predictions: dict[str, np.ndarray] = {}
    alphas = np.round(np.linspace(0.0, 1.0, 21), 2)
    for alpha in alphas:
        name = f"alpha={alpha:.2f}"
        pred = (1.0 - alpha) * global_pred + alpha * expert_pred
        predictions[name] = pred
        errors[name] = per_doi_mae(y_true, pred, dois)
    simplicity = [f"alpha={value:.2f}" for value in alphas]
    selected, selection = choose_one_se(errors, simplicity)
    alpha = float(selected.split("=")[1])
    selection["alpha"] = selection["candidate"].str.split("=").str[1].astype(float)
    return alpha, predictions[selected], selection


def select_development_policy(
    y_true: np.ndarray,
    predictions: dict[str, np.ndarray],
    dois: np.ndarray,
) -> tuple[str, pd.DataFrame]:
    errors = {name: per_doi_mae(y_true, pred, dois) for name, pred in predictions.items()}
    simplicity = [GLOBAL_METHOD, "Subgroup calibrator", "Convex mixture", "Domain expert"]
    return choose_one_se(errors, simplicity)


def metric_summary(predictions: pd.DataFrame, replicates: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    group_columns = ["domain", "target", "method"]
    for keys, group in predictions.groupby(group_columns, sort=False):
        domain, target, method = keys
        for evaluation in ["Device-level", "Publication-balanced"]:
            point = metric_values(
                group["y_true"].to_numpy(float),
                group["y_pred"].to_numpy(float),
                group["doi_norm"].astype(str).to_numpy(),
                evaluation,
            )
            boot = cluster_bootstrap(
                group,
                evaluation,
                replicates,
                stable_seed("metric", domain, target, method, evaluation),
            )
            row: dict[str, object] = {
                "domain": domain,
                "target": target,
                "method": method,
                "evaluation": evaluation,
                "records": int(group["Ref_ID"].nunique()),
                "DOI_groups": int(group["doi_norm"].nunique()),
            }
            for metric in ["R2", "MAE", "RMSE", "bias"]:
                row[metric] = point[metric]
                row[f"{metric}_CI_low"] = float(np.nanquantile(boot[metric], 0.025))
                row[f"{metric}_CI_high"] = float(np.nanquantile(boot[metric], 0.975))
            rows.append(row)
    return pd.DataFrame(rows)


def paired_summary(predictions: pd.DataFrame, replicates: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (domain, target), block in predictions.groupby(["domain", "target"], sort=False):
        reference = block.loc[block["method"].eq(GLOBAL_METHOD)]
        for method in METHODS[1:]:
            candidate = block.loc[block["method"].eq(method)]
            for evaluation in ["Device-level", "Publication-balanced"]:
                point_method = metric_values(
                    candidate["y_true"].to_numpy(float),
                    candidate["y_pred"].to_numpy(float),
                    candidate["doi_norm"].astype(str).to_numpy(),
                    evaluation,
                )
                point_ref = metric_values(
                    reference["y_true"].to_numpy(float),
                    reference["y_pred"].to_numpy(float),
                    reference["doi_norm"].astype(str).to_numpy(),
                    evaluation,
                )
                point = {
                    "delta_R2": point_method["R2"] - point_ref["R2"],
                    "delta_MAE": point_method["MAE"] - point_ref["MAE"],
                    "MAE_change_percent": 100.0 * (point_method["MAE"] / point_ref["MAE"] - 1.0),
                    "delta_RMSE": point_method["RMSE"] - point_ref["RMSE"],
                    "delta_absolute_bias": abs(point_method["bias"]) - abs(point_ref["bias"]),
                }
                boot = paired_cluster_bootstrap(
                    candidate,
                    reference,
                    evaluation,
                    replicates,
                    stable_seed("paired", domain, target, method, evaluation),
                )
                row: dict[str, object] = {
                    "domain": domain,
                    "target": target,
                    "method": method,
                    "reference": GLOBAL_METHOD,
                    "evaluation": evaluation,
                    "records": int(candidate["Ref_ID"].nunique()),
                    "DOI_groups": int(candidate["doi_norm"].nunique()),
                }
                for metric, value in point.items():
                    row[metric] = value
                    row[f"{metric}_CI_low"] = float(np.nanquantile(boot[metric], 0.025))
                    row[f"{metric}_CI_high"] = float(np.nanquantile(boot[metric], 0.975))
                rows.append(row)
    return pd.DataFrame(rows)


def weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    order = np.argsort(values, kind="mergesort")
    x = values[order]
    w = weights[order]
    cumulative = np.cumsum(w) - 0.5 * w
    cumulative /= w.sum()
    return float(np.interp(quantile, cumulative, x))


def calibration_bins(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    selected = predictions.loc[
        predictions["target"].eq("PCE")
        & predictions["method"].isin([GLOBAL_METHOD, "Development-selected policy"])
    ].copy()
    for (domain, method), group in selected.groupby(["domain", "method"], sort=False):
        weights = publication_weights(group["doi_norm"])
        quantiles = [weighted_quantile(group["y_pred"].to_numpy(float), weights, q) for q in np.linspace(0, 1, 5)]
        quantiles[0] = -np.inf
        quantiles[-1] = np.inf
        bins = pd.cut(group["y_pred"], bins=np.unique(quantiles), include_lowest=True, duplicates="drop")
        group = group.assign(calibration_bin=bins)
        for bin_index, (_interval, subset) in enumerate(group.groupby("calibration_bin", observed=True), start=1):
            w = publication_weights(subset["doi_norm"])
            rows.append(
                {
                    "domain": domain,
                    "method": method,
                    "bin": bin_index,
                    "records": len(subset),
                    "DOI_groups": subset["doi_norm"].nunique(),
                    "mean_predicted_PCE": float(np.average(subset["y_pred"], weights=w)),
                    "mean_measured_PCE": float(np.average(subset["y_true"], weights=w)),
                }
            )
    return pd.DataFrame(rows)


def upper_tail_summary(
    predictions: pd.DataFrame,
    historical_targets: pd.DataFrame,
    historical_metadata: pd.DataFrame,
) -> pd.DataFrame:
    thresholds: dict[str, float] = {}
    for domain in DOMAINS:
        mask = historical_metadata["domain"].eq(domain)
        values = historical_targets.loc[mask, "PCE"].to_numpy(float)
        weights = publication_weights(historical_metadata.loc[mask, "doi_norm"])
        thresholds[domain] = weighted_quantile(values, weights, 0.75)
    rows: list[dict[str, object]] = []
    pce = predictions.loc[predictions["target"].eq("PCE")]
    for (domain, method), group in pce.groupby(["domain", "method"], sort=False):
        definitions = [
            ("Historical publication-balanced PCE Q75", thresholds[domain]),
            ("Conventional PCE >=20%", 20.0),
        ]
        for label, threshold in definitions:
            subset = group.loc[group["y_true"] >= threshold]
            if len(subset):
                values = metric_values(
                    subset["y_true"].to_numpy(float),
                    subset["y_pred"].to_numpy(float),
                    subset["doi_norm"].astype(str).to_numpy(),
                    "Publication-balanced",
                )
            else:
                values = {"R2": np.nan, "MAE": np.nan, "RMSE": np.nan, "bias": np.nan}
            rows.append(
                {
                    "domain": domain,
                    "method": method,
                    "subset": label,
                    "threshold_PCE": threshold,
                    "records": len(subset),
                    "DOI_groups": subset["doi_norm"].nunique(),
                    **values,
                    "inferential_status": "descriptive only" if subset["doi_norm"].nunique() < 20 else "bootstrap-eligible",
                }
            )
    return pd.DataFrame(rows)


def robustness_strata(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    definitions = {
        "All": pd.Series(True, index=predictions.index),
        "Formula seen historically": ~predictions["formula_unseen_historical"].fillna(False),
        "Formula unseen historically": predictions["formula_unseen_historical"].fillna(False),
        "Feature OOD <=95th percentile": predictions["feature_ood_percentile"].le(0.95),
        "Feature OOD >95th percentile": predictions["feature_ood_percentile"].gt(0.95),
    }
    for stratum, mask in definitions.items():
        for (domain, target, method), group in predictions.loc[mask].groupby(
            ["domain", "target", "method"], sort=False
        ):
            if not len(group):
                continue
            values = metric_values(
                group["y_true"].to_numpy(float),
                group["y_pred"].to_numpy(float),
                group["doi_norm"].astype(str).to_numpy(),
                "Publication-balanced",
            )
            rows.append(
                {
                    "stratum": stratum,
                    "domain": domain,
                    "target": target,
                    "method": method,
                    "records": group["Ref_ID"].nunique(),
                    "DOI_groups": group["doi_norm"].nunique(),
                    **values,
                    "inferential_status": "descriptive only" if group["doi_norm"].nunique() < 20 else "supported",
                }
            )
    return pd.DataFrame(rows)


def make_figure(
    support: pd.DataFrame,
    development: pd.DataFrame,
    paired: pd.DataFrame,
    bins: pd.DataFrame,
    output_dir: Path,
) -> None:
    sns.set_theme(style="whitegrid", context="paper")
    colors = {
        "Subgroup calibrator": "#0072B2",
        "Domain expert": "#D55E00",
        "Convex mixture": "#009E73",
        "Development-selected policy": "#7A3E9D",
    }
    fig = plt.figure(figsize=(12.0, 9.2))
    grid = fig.add_gridspec(2, 2, hspace=0.36, wspace=0.27)

    ax = fig.add_subplot(grid[0, 0])
    doi_support = support.copy()
    x = np.arange(len(DOMAINS))
    width = 0.34
    for idx, period in enumerate(["Historical <=2018", "Future 2019-2021"]):
        values = [
            int(doi_support.loc[(doi_support["domain"] == domain) & (doi_support["period"] == period), "DOI_groups"].iloc[0])
            for domain in DOMAINS
        ]
        bars = ax.bar(
            x + (idx - 0.5) * width,
            values,
            width,
            label=period,
            color=["#4C78A8", "#F58518"][idx],
        )
        for bar, domain in zip(bars, DOMAINS):
            records = int(doi_support.loc[(doi_support["domain"] == domain) & (doi_support["period"] == period), "records"].iloc[0])
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2, f"{records} rec.", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x, DOMAINS)
    ax.set_ylabel("Independent DOI groups")
    ax.set_title("a  Sparse subgroup support", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=8)

    ax = fig.add_subplot(grid[0, 1])
    methods = ["Subgroup calibrator", "Domain expert", "Convex mixture", "Development-selected policy"]
    pivot = development.pivot(index="domain_target", columns="method", values="MAE_ratio_to_global").reindex(
        [f"{domain} | {target}" for domain in DOMAINS for target in TARGETS]
    )[methods]
    sns.heatmap(
        pivot,
        ax=ax,
        cmap="vlag",
        center=1.0,
        vmin=0.75,
        vmax=1.25,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        cbar_kws={"label": "Historical OOF MAE / global MAE", "shrink": 0.75},
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("b  Historical DOI-disjoint development", loc="left", fontweight="bold")
    ax.tick_params(axis="x", rotation=25, labelsize=7)
    ax.tick_params(axis="y", labelsize=7)

    ax = fig.add_subplot(grid[1, 0])
    plot = paired.loc[
        paired["evaluation"].eq("Publication-balanced")
        & paired["method"].isin(colors)
    ].copy()
    plot["label"] = plot["domain"] + " | " + plot["target"]
    order = [f"{domain} | {target}" for domain in DOMAINS for target in TARGETS]
    ybase = np.arange(len(order))
    offsets = np.linspace(-0.24, 0.24, len(colors))
    for offset, (method, color) in zip(offsets, colors.items()):
        part = plot.loc[plot["method"].eq(method)].set_index("label").reindex(order)
        xval = part["MAE_change_percent"].to_numpy(float)
        low = part["MAE_change_percent_CI_low"].to_numpy(float)
        high = part["MAE_change_percent_CI_high"].to_numpy(float)
        ax.errorbar(
            xval,
            ybase + offset,
            xerr=np.vstack([xval - low, high - xval]),
            fmt="o",
            ms=4,
            capsize=2,
            color=color,
            label=method,
            alpha=0.9,
        )
    ax.axvline(0, color="black", lw=1)
    ax.set_yticks(ybase, order, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Future publication-balanced MAE change vs global (%)")
    ax.set_title("c  Independent chronological test", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=7, ncol=2, loc="lower right")

    ax = fig.add_subplot(grid[1, 1])
    domain_colors = {"Sn-only": "#56B4E9", "Mixed Pb-Sn": "#CC79A7"}
    marker_map = {GLOBAL_METHOD: "o", "Development-selected policy": "s"}
    linestyle_map = {GLOBAL_METHOD: "--", "Development-selected policy": "-"}
    extrema = []
    for (domain, method), part in bins.groupby(["domain", "method"], sort=False):
        part = part.sort_values("mean_predicted_PCE")
        label = f"{domain}: {'global' if method == GLOBAL_METHOD else 'selected'}"
        ax.plot(
            part["mean_predicted_PCE"],
            part["mean_measured_PCE"],
            marker=marker_map[method],
            linestyle=linestyle_map[method],
            color=domain_colors[domain],
            label=label,
            lw=1.5,
            ms=4,
        )
        extrema.extend(part["mean_predicted_PCE"].tolist())
        extrema.extend(part["mean_measured_PCE"].tolist())
    lo, hi = min(extrema), max(extrema)
    margin = 0.05 * (hi - lo)
    ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin], color="0.35", lw=1, ls=":")
    ax.set_xlim(lo - margin, hi + margin)
    ax.set_ylim(lo - margin, hi + margin)
    ax.set_xlabel("Mean predicted PCE (%)")
    ax.set_ylabel("Mean measured PCE (%)")
    ax.set_title("d  Publication-balanced PCE calibration", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=7)

    fig.suptitle(
        "Subgroup calibration and shrinkage mixture-of-experts for Sn-containing PSCs",
        fontsize=13,
        fontweight="bold",
        y=0.99,
    )
    fig.subplots_adjust(top=0.94, bottom=0.08, left=0.09, right=0.98)
    for extension in ["png", "pdf", "svg"]:
        fig.savefig(output_dir / f"Figure9_subgroup_calibration_moe.{extension}", dpi=600 if extension == "png" else None, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.baseline_code_dir.resolve()))
    from psc_baseline_validation import (  # type: ignore
        RAW_REQUIRED,
        TARGETS as BASELINE_TARGETS,
        ModelConfig,
        build_features,
        make_preprocessor,
        normalize_doi,
    )

    config = ModelConfig()
    if list(BASELINE_TARGETS) != TARGETS:
        raise AssertionError("Target order differs from frozen baseline")

    print("Loading frozen cohort, assignments, and raw features", flush=True)
    cohort = pd.read_csv(args.cohort, low_memory=False)
    cohort["doi_norm"] = normalize_doi(cohort["Ref_DOI_number"])
    cohort["publication_year"] = pd.to_datetime(cohort["Ref_publication_date"], errors="coerce").dt.year.astype(int)
    raw = pd.read_csv(args.raw, usecols=RAW_REQUIRED, low_memory=False)
    joined = cohort[["Ref_ID", "doi_norm", "publication_year"]].merge(
        raw, on="Ref_ID", how="left", validate="one_to_one"
    )
    assignments = pd.read_csv(args.assignments, low_memory=False)
    split = pd.read_csv(args.split_manifest, usecols=["Ref_ID", "grouped_fold"])
    joined = joined.merge(
        assignments[
            [
                "Ref_ID",
                "b_site_pattern",
                "formula_unseen_historical",
                "Sn_fraction_among_Pb_Sn",
            ]
        ],
        on="Ref_ID",
        validate="one_to_one",
    ).merge(split, on="Ref_ID", validate="one_to_one")
    joined["domain"] = joined["b_site_pattern"].map({value: key for key, value in DOMAINS.items()})
    joined["development_fold"] = joined["grouped_fold"].astype(int)
    if joined["Ref_ID"].duplicated().any() or len(joined) != 33175:
        raise AssertionError("Cohort join changed row identity")
    features, numeric_features = build_features(joined)
    targets = pd.DataFrame(
        {target: pd.to_numeric(joined[column], errors="raise") for target, (column, _unit) in BASELINE_TARGETS.items()},
        index=joined.index,
    )
    targets["FF"] = targets["FF"] * 100.0
    metadata = joined[["Ref_ID", "doi_norm", "publication_year", "domain", "development_fold"]].copy()

    historical = joined["publication_year"].le(config.temporal_cutoff_year).to_numpy()
    future = ~historical
    if set(joined.loc[historical, "doi_norm"]) & set(joined.loc[future, "doi_norm"]):
        raise AssertionError("Historical and future DOI overlap")

    support_rows = []
    for domain in DOMAINS:
        for period, mask in [("Historical <=2018", historical), ("Future 2019-2021", future)]:
            subset = joined.loc[mask & joined["domain"].eq(domain)]
            support_rows.append(
                {
                    "domain": domain,
                    "period": period,
                    "records": len(subset),
                    "DOI_groups": subset["doi_norm"].nunique(),
                    "year_min": subset["publication_year"].min(),
                    "year_max": subset["publication_year"].max(),
                }
            )
    support = pd.DataFrame(support_rows)

    print("Generating historical-only DOI-disjoint global OOF predictions", flush=True)
    hist_index = np.flatnonzero(historical)
    hist_folds = metadata.iloc[hist_index]["development_fold"].to_numpy(int)
    global_oof = np.full((len(joined), len(TARGETS)), np.nan, dtype=float)
    global_config = replace(config, rf_estimators=40 if args.quick else config.rf_estimators)
    for fold in sorted(np.unique(hist_folds)):
        train_index = hist_index[hist_folds != fold]
        test_index = hist_index[hist_folds == fold]
        global_oof[test_index] = fit_predict_forest(
            features,
            targets,
            metadata,
            train_index,
            test_index,
            numeric_features,
            global_config,
            make_preprocessor,
            stable_seed("global-development", fold),
        )
        print(f"  historical global fold {fold}/5 complete", flush=True)
    if not np.isfinite(global_oof[hist_index]).all():
        raise AssertionError("Historical global OOF predictions incomplete")

    print("Loading frozen future predictions", flush=True)
    archive = pd.read_csv(args.weighting_predictions, low_memory=False)
    archive = archive.loc[
        archive["scheme"].eq(CHRONO_SCHEME)
        & archive["training_weighting"].eq(FULL_WEIGHTING)
        & archive["model"].eq("Random Forest")
    ].copy()
    archive_pred = archive.pivot(index="Ref_ID", columns="target", values="y_pred").reindex(columns=TARGETS)
    archive_true = archive.pivot(index="Ref_ID", columns="target", values="y_true").reindex(columns=TARGETS)
    future_ids = joined.loc[future, "Ref_ID"]
    if archive_pred.reindex(future_ids).isna().any().any():
        raise AssertionError("Frozen chronological archive is incomplete")
    frozen_future = archive_pred.reindex(future_ids).to_numpy(float)
    y_future = targets.loc[future].to_numpy(float)
    archive_difference = float(np.max(np.abs(archive_true.reindex(future_ids).to_numpy(float) - y_future)))
    if archive_difference > 1e-12:
        raise AssertionError("Archived future outcomes differ from cohort")

    composition_predictions = pd.read_csv(args.composition_predictions, low_memory=False)
    ood = composition_predictions.loc[
        composition_predictions["scheme"].eq(CHRONO_SCHEME),
        ["Ref_ID", "target", "feature_ood_percentile", "model_ood_percentile"],
    ].drop_duplicates(["Ref_ID", "target"])

    expert_configs = expert_candidate_configs(config, args.quick)
    development_rows: list[dict[str, object]] = []
    calibrator_rows: list[dict[str, object]] = []
    blend_rows: list[dict[str, object]] = []
    policy_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    domain_manifest: dict[str, object] = {}

    for domain in DOMAINS:
        print(f"Selecting subgroup models for {domain}", flush=True)
        domain_hist_index = np.flatnonzero(historical & joined["domain"].eq(domain).to_numpy())
        domain_future_index = np.flatnonzero(future & joined["domain"].eq(domain).to_numpy())
        if metadata.iloc[domain_hist_index]["doi_norm"].nunique() < 20:
            raise AssertionError(f"Insufficient development DOI for {domain}")
        selected_expert, expert_oof, expert_selection = select_expert(
            domain,
            domain_hist_index,
            features,
            targets,
            metadata,
            numeric_features,
            expert_configs,
            make_preprocessor,
        )
        expert_selection.insert(0, "domain", domain)
        expert_selection.insert(1, "selection_component", "expert hyperparameter")
        development_rows.extend(expert_selection.to_dict("records"))
        print(f"  selected expert: {selected_expert}", flush=True)

        final_expert_prediction = fit_predict_forest(
            features,
            targets,
            metadata,
            domain_hist_index,
            domain_future_index,
            numeric_features,
            expert_configs[selected_expert],
            make_preprocessor,
            stable_seed(domain, "final-expert"),
        )

        global_hist_domain = global_oof[domain_hist_index]
        global_future_domain = archive_pred.reindex(joined.iloc[domain_future_index]["Ref_ID"]).to_numpy(float)
        y_hist_domain = targets.iloc[domain_hist_index].to_numpy(float)
        y_future_domain = targets.iloc[domain_future_index].to_numpy(float)
        hist_meta = metadata.iloc[domain_hist_index].reset_index(drop=True)
        fut_meta = metadata.iloc[domain_future_index].reset_index(drop=True)

        calibrated_oof = np.empty_like(y_hist_domain)
        calibrated_future = np.empty_like(y_future_domain)
        mixture_oof = np.empty_like(y_hist_domain)
        mixture_future = np.empty_like(y_future_domain)
        selected_policy_oof = np.empty_like(y_hist_domain)
        selected_policy_future = np.empty_like(y_future_domain)
        selected_policy_methods: dict[str, str] = {}
        selected_alphas: dict[str, float] = {}
        calibration_params: dict[str, object] = {}

        for target_index, target in enumerate(TARGETS):
            calibration_frame = pd.DataFrame(
                {
                    "doi_norm": hist_meta["doi_norm"],
                    "development_fold": hist_meta["development_fold"],
                    "y_true": y_hist_domain[:, target_index],
                    "global_pred": global_hist_domain[:, target_index],
                }
            )
            selected_calibrator, cal_oof, cal_params, cal_selection = calibration_crossfit(calibration_frame)
            calibrated_oof[:, target_index] = cal_oof
            calibrated_future[:, target_index] = apply_calibrator(
                global_future_domain[:, target_index], cal_params
            )
            calibration_params[target] = cal_params
            cal_selection.insert(0, "domain", domain)
            cal_selection.insert(1, "target", target)
            calibrator_rows.extend(cal_selection.to_dict("records"))

            alpha, blend_oof, blend_selection = select_blend_alpha(
                y_hist_domain[:, target_index],
                global_hist_domain[:, target_index],
                expert_oof[:, target_index],
                hist_meta["doi_norm"].astype(str).to_numpy(),
            )
            mixture_oof[:, target_index] = blend_oof
            mixture_future[:, target_index] = (
                (1.0 - alpha) * global_future_domain[:, target_index]
                + alpha * final_expert_prediction[:, target_index]
            )
            selected_alphas[target] = alpha
            blend_selection.insert(0, "domain", domain)
            blend_selection.insert(1, "target", target)
            blend_rows.extend(blend_selection.to_dict("records"))

            development_predictions = {
                GLOBAL_METHOD: global_hist_domain[:, target_index],
                "Subgroup calibrator": calibrated_oof[:, target_index],
                "Domain expert": expert_oof[:, target_index],
                "Convex mixture": mixture_oof[:, target_index],
            }
            selected_policy, policy_selection = select_development_policy(
                y_hist_domain[:, target_index],
                development_predictions,
                hist_meta["doi_norm"].astype(str).to_numpy(),
            )
            selected_policy_methods[target] = selected_policy
            selected_policy_oof[:, target_index] = development_predictions[selected_policy]
            future_map = {
                GLOBAL_METHOD: global_future_domain[:, target_index],
                "Subgroup calibrator": calibrated_future[:, target_index],
                "Domain expert": final_expert_prediction[:, target_index],
                "Convex mixture": mixture_future[:, target_index],
            }
            selected_policy_future[:, target_index] = future_map[selected_policy]
            policy_selection.insert(0, "domain", domain)
            policy_selection.insert(1, "target", target)
            policy_rows.extend(policy_selection.to_dict("records"))
            print(
                f"  {target}: calibrator={selected_calibrator}, alpha={alpha:.2f}, policy={selected_policy}",
                flush=True,
            )

        development_method_predictions = {
            GLOBAL_METHOD: global_hist_domain,
            "Subgroup calibrator": calibrated_oof,
            "Domain expert": expert_oof,
            "Convex mixture": mixture_oof,
            "Development-selected policy": selected_policy_oof,
        }
        for target_index, target in enumerate(TARGETS):
            global_mae = per_doi_mae(
                y_hist_domain[:, target_index],
                global_hist_domain[:, target_index],
                hist_meta["doi_norm"].astype(str).to_numpy(),
            ).mean()
            for method, prediction in development_method_predictions.items():
                mae = per_doi_mae(
                    y_hist_domain[:, target_index],
                    prediction[:, target_index],
                    hist_meta["doi_norm"].astype(str).to_numpy(),
                ).mean()
                development_rows.append(
                    {
                        "domain": domain,
                        "selection_component": "method performance",
                        "target": target,
                        "method": method,
                        "historical_OOF_publication_balanced_MAE": float(mae),
                        "MAE_ratio_to_global": float(mae / global_mae),
                        "domain_target": f"{domain} | {target}",
                    }
                )

        future_method_predictions = {
            GLOBAL_METHOD: global_future_domain,
            "Subgroup calibrator": calibrated_future,
            "Domain expert": final_expert_prediction,
            "Convex mixture": mixture_future,
            "Development-selected policy": selected_policy_future,
        }
        future_extra = joined.iloc[domain_future_index][
            ["Ref_ID", "doi_norm", "publication_year", "formula_unseen_historical", "Sn_fraction_among_Pb_Sn"]
        ].reset_index(drop=True)
        for method, method_prediction in future_method_predictions.items():
            for target_index, target in enumerate(TARGETS):
                frame = future_extra.copy()
                frame["domain"] = domain
                frame["target"] = target
                frame["method"] = method
                frame["y_true"] = y_future_domain[:, target_index]
                frame["y_pred"] = method_prediction[:, target_index]
                frame["residual"] = frame["y_pred"] - frame["y_true"]
                frame = frame.merge(
                    ood.loc[ood["target"].eq(target)],
                    on=["Ref_ID", "target"],
                    how="left",
                    validate="one_to_one",
                )
                prediction_frames.append(frame)

        domain_manifest[domain] = {
            "historical_records": int(len(domain_hist_index)),
            "historical_DOI": int(metadata.iloc[domain_hist_index]["doi_norm"].nunique()),
            "future_records": int(len(domain_future_index)),
            "future_DOI": int(metadata.iloc[domain_future_index]["doi_norm"].nunique()),
            "selected_expert": selected_expert,
            "selected_expert_config": asdict(expert_configs[selected_expert]),
            "calibration_parameters": calibration_params,
            "selected_blend_alpha": selected_alphas,
            "selected_policy_method": selected_policy_methods,
        }

    predictions = pd.concat(prediction_frames, ignore_index=True)
    expected_rows = sum(
        int(support.loc[(support["domain"] == domain) & (support["period"] == "Future 2019-2021"), "records"].iloc[0])
        for domain in DOMAINS
    ) * len(TARGETS) * len(METHODS)
    if len(predictions) != expected_rows:
        raise AssertionError("Prediction archive has unexpected size")
    if predictions.duplicated(["Ref_ID", "target", "method"]).any():
        raise AssertionError("Duplicate prediction key")

    print("Computing 1,000-replicate paired DOI bootstrap", flush=True)
    metric_table = metric_summary(predictions, BOOTSTRAP_REPLICATES if not args.quick else 100)
    paired_table = paired_summary(predictions, BOOTSTRAP_REPLICATES if not args.quick else 100)
    calibration_table = calibration_bins(predictions)
    historical_domain_metadata = metadata.loc[historical].copy()
    upper_tail = upper_tail_summary(
        predictions,
        targets.loc[historical].reset_index(drop=True),
        historical_domain_metadata.reset_index(drop=True),
    )
    robustness = robustness_strata(predictions)

    calibrator_table = pd.DataFrame(calibrator_rows)
    blend_table = pd.DataFrame(blend_rows)
    policy_table = pd.DataFrame(policy_rows)
    development_table = pd.DataFrame(development_rows)
    method_development = development_table.loc[development_table["selection_component"].eq("method performance")].copy()

    support.to_csv(args.output_dir / "subgroup_support.csv", index=False)
    development_table.to_csv(args.output_dir / "subgroup_development_selection.csv", index=False)
    calibrator_table.to_csv(args.output_dir / "subgroup_calibrator_selection.csv", index=False)
    blend_table.to_csv(args.output_dir / "subgroup_mixture_weight_selection.csv", index=False)
    policy_table.to_csv(args.output_dir / "subgroup_policy_selection.csv", index=False)
    predictions.to_csv(args.output_dir / "subgroup_future_predictions.csv.gz", index=False, compression="gzip")
    metric_table.to_csv(args.output_dir / "subgroup_future_metrics.csv", index=False)
    paired_table.to_csv(args.output_dir / "subgroup_paired_comparisons.csv", index=False)
    calibration_table.to_csv(args.output_dir / "subgroup_PCE_calibration_bins.csv", index=False)
    upper_tail.to_csv(args.output_dir / "subgroup_PCE_upper_tail.csv", index=False)
    robustness.to_csv(args.output_dir / "subgroup_robustness_strata.csv", index=False)
    make_figure(support, method_development, paired_table, calibration_table, args.output_dir)

    manifest = {
        "analysis": "Sn-only and mixed Pb-Sn subgroup calibration / mixture-of-experts",
        "created_utc": pd.Timestamp.utcnow().isoformat(),
        "temporal_design": {
            "development": "publication-disjoint OOF predictions from records published through 2018",
            "final_test": "frozen chronological 2019-2021 holdout evaluated once",
            "future_outcomes_used_for_tuning": False,
            "DOI_overlap": 0,
        },
        "bootstrap_replicates": BOOTSTRAP_REPLICATES if not args.quick else 100,
        "seed": SEED,
        "global_model": "Frozen full 1/n_DOI weighted multi-output Random Forest",
        "global_config": asdict(global_config),
        "selection_rule": "DOI-level one-standard-error rule, favoring stronger shrinkage / lower complexity",
        "domains": domain_manifest,
        "archive_outcome_max_abs_difference": archive_difference,
        "prediction_rows": int(len(predictions)),
        "duplicate_prediction_keys": int(predictions.duplicated(["Ref_ID", "target", "method"]).sum()),
        "input_sha256": {
            "raw": sha256(args.raw),
            "cohort": sha256(args.cohort),
            "assignments": sha256(args.assignments),
            "composition_predictions": sha256(args.composition_predictions),
            "split_manifest": sha256(args.split_manifest),
            "weighting_predictions": sha256(args.weighting_predictions),
        },
        "software": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    (args.output_dir / "subgroup_calibration_moe_run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Completed in {manifest['elapsed_seconds']:.1f} seconds", flush=True)


if __name__ == "__main__":
    main()
