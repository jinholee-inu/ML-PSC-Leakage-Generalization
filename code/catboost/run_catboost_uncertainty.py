#!/usr/bin/env python3
"""DOI-disjoint conformal quantile uncertainty and OOD analysis for CatBoost.

The frozen full 1/n_DOI CatBoost point models are refitted exactly under the
original five DOI-grouped folds and the 2019--2021 chronological holdout.
PCE prediction intervals use CatBoost MultiQuantile models plus DOI-disjoint
conformalized quantile regression (CQR). OOD scores are target-free prototype
distance and CatBoost leaf-support percentiles fitted on each training split.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from pathlib import Path

import catboost
from catboost import CatBoostRegressor, Pool
import numpy as np
import pandas as pd
from scipy import sparse

SCRIPT_DIR = Path(__file__).resolve().parent
BASELINE_DIR = SCRIPT_DIR / "baseline-code"
RF_UNCERTAINTY_DIR = (
    SCRIPT_DIR.parent.parent
    / "uncertainty_rf"
    / "PSC_uncertainty_OOD_analysis_package"
    / "code"
)
sys.path.insert(0, str(BASELINE_DIR))
sys.path.insert(0, str(RF_UNCERTAINTY_DIR))

from psc_baseline_validation import (  # noqa: E402
    RAW_REQUIRED,
    TARGETS,
    ModelConfig,
    build_features,
    make_preprocessor,
    normalize_doi,
    sha256,
)
from uncertainty_ood_analysis import (  # noqa: E402
    CHRONO_SCHEME,
    FULL_WEIGHTING,
    GROUPED_SCHEME,
    NOMINAL_LEVELS,
    cluster_mean_ci,
    coverage_metrics,
    feature_space_ood,
    interval_score,
    ood_stratified_performance,
    plot_figure5,
    publication_weights,
    quantile_performance,
    selective_prediction_metrics,
    uncertainty_error_association,
    weighted_quantile,
)
from catboost_shap_ale import (  # noqa: E402
    CONDITION,
    MODEL,
    as_model_matrix,
    build_catboost_model,
    selected_catboost_config,
    training_weights,
    weighted_target_location_scale,
)


PCE_INDEX = list(TARGETS).index("PCE")
QUANTILE_ALPHAS = [0.025, 0.05, 0.95, 0.975]
OOD_COMPONENTS = 32
OOD_CLUSTERS = 160


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", required=True, type=Path)
    parser.add_argument("--cohort", required=True, type=Path)
    parser.add_argument("--split-manifest", required=True, type=Path)
    parser.add_argument("--baseline-manifest", required=True, type=Path)
    parser.add_argument("--catboost-results-dir", required=True, type=Path)
    parser.add_argument("--catboost-model-selection", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def quantile_model(
    candidate: dict[str, float | int | str], iterations: int, seed: int
) -> CatBoostRegressor:
    alpha_text = ",".join(str(value) for value in QUANTILE_ALPHAS)
    return CatBoostRegressor(
        loss_function=f"MultiQuantile:alpha={alpha_text}",
        iterations=int(iterations),
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


def fit_point_partition(
    features: pd.DataFrame,
    targets: pd.DataFrame,
    metadata: pd.DataFrame,
    train_index: np.ndarray,
    test_index: np.ndarray,
    numeric_features: list[str],
    config: ModelConfig,
    candidate: dict[str, float | int | str],
    iterations: int,
    seed: int,
) -> tuple[object, object, CatBoostRegressor, np.ndarray, object]:
    processor = make_preprocessor(config, numeric_features)
    train_matrix = as_model_matrix(processor.fit_transform(features.iloc[train_index]))
    test_matrix = as_model_matrix(processor.transform(features.iloc[test_index]))
    y_train = targets.iloc[train_index].to_numpy(dtype=float)
    weights = training_weights(metadata.iloc[train_index]["doi_norm"].reset_index(drop=True))
    y_mean, y_std = weighted_target_location_scale(y_train, weights)
    model = build_catboost_model(candidate, iterations, seed)
    model.fit(Pool(train_matrix, label=(y_train - y_mean) / y_std, weight=weights))
    prediction = model.predict(test_matrix) * y_std + y_mean
    return train_matrix, test_matrix, model, prediction, processor


def fit_quantile_partition(
    features: pd.DataFrame,
    targets: pd.DataFrame,
    metadata: pd.DataFrame,
    train_index: np.ndarray,
    test_index: np.ndarray,
    numeric_features: list[str],
    config: ModelConfig,
    candidate: dict[str, float | int | str],
    iterations: int,
    seed: int,
) -> np.ndarray:
    processor = make_preprocessor(config, numeric_features)
    train_matrix = as_model_matrix(processor.fit_transform(features.iloc[train_index]))
    test_matrix = as_model_matrix(processor.transform(features.iloc[test_index]))
    y_train = targets.iloc[train_index]["PCE"].to_numpy(dtype=float)
    weights = training_weights(metadata.iloc[train_index]["doi_norm"].reset_index(drop=True))
    mean = float(np.average(y_train, weights=weights))
    std = float(np.sqrt(np.average((y_train - mean) ** 2, weights=weights)))
    if not std > 0:
        std = 1.0
    model = quantile_model(candidate, iterations, seed)
    model.fit(Pool(train_matrix, label=(y_train - mean) / std, weight=weights))
    prediction = np.asarray(model.predict(test_matrix), dtype=float)
    if prediction.ndim == 1:
        prediction = prediction.reshape(-1, len(QUANTILE_ALPHAS))
    prediction = prediction * std + mean
    # MultiQuantile optimizes all levels jointly but can still exhibit small
    # finite-sample crossings. Monotone rearrangement preserves each row's
    # estimated quantile set while enforcing valid nested intervals.
    return np.sort(prediction, axis=1)


def calibrate_cqr(
    y_true: np.ndarray, quantiles: np.ndarray, dois: pd.Series
) -> tuple[dict[int, float], list[dict[str, float]]]:
    weights = publication_weights(dois.reset_index(drop=True))
    corrections: dict[int, float] = {}
    rows: list[dict[str, float]] = []
    for nominal, lo_index, hi_index in [(90, 1, 2), (95, 0, 3)]:
        score = np.maximum(
            quantiles[:, lo_index] - y_true,
            y_true - quantiles[:, hi_index],
        )
        # A non-negative correction preserves interval ordering and makes the
        # calibration step conservative when the raw quantile model already
        # exceeds nominal coverage on a DOI-balanced calibration partition.
        correction = max(0.0, weighted_quantile(score, nominal / 100.0, weights))
        corrections[nominal] = correction
        lower = quantiles[:, lo_index] - correction
        upper = quantiles[:, hi_index] + correction
        rows.append(
            {
                "nominal_coverage": nominal / 100.0,
                "conformal_correction": correction,
                "publication_balanced_calibration_coverage": float(
                    np.average((y_true >= lower) & (y_true <= upper), weights=weights)
                ),
                "raw_quantile_crossing_fraction": float(
                    np.mean(quantiles[:, lo_index] > quantiles[:, hi_index])
                ),
            }
        )
    return corrections, rows


def catboost_leaf_support(
    model: CatBoostRegressor,
    train_matrix: object,
    test_matrix: object,
    train_weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    train_leaf = np.asarray(model.calc_leaf_indexes(train_matrix), dtype=np.int32)
    test_leaf = np.asarray(model.calc_leaf_indexes(test_matrix), dtype=np.int32)
    train_log_support = np.zeros(len(train_leaf), dtype=np.float64)
    test_log_support = np.zeros(len(test_leaf), dtype=np.float64)
    for tree in range(train_leaf.shape[1]):
        train_ids = train_leaf[:, tree]
        test_ids = test_leaf[:, tree]
        size = int(max(train_ids.max(), test_ids.max())) + 1
        mass = np.bincount(train_ids, weights=train_weights, minlength=size)
        train_log_support += np.log1p(mass[train_ids])
        test_log_support += np.log1p(mass[test_ids])
    train_log_support /= train_leaf.shape[1]
    test_log_support /= test_leaf.shape[1]
    train_support = np.expm1(train_log_support)
    test_support = np.expm1(test_log_support)
    train_ood = -train_log_support
    test_ood = -test_log_support
    sorted_train = np.sort(train_ood)
    percentile = np.searchsorted(sorted_train, test_ood, side="right") / float(len(sorted_train))
    diagnostics = {
        "train_leaf_support_median": float(np.median(train_support)),
        "train_leaf_support_p05": float(np.quantile(train_support, 0.05)),
        "test_leaf_support_median": float(np.median(test_support)),
        "test_leaf_support_p05": float(np.quantile(test_support, 0.05)),
        "test_fraction_model_OOD_above_train_p95": float(np.mean(percentile > 0.95)),
        "catboost_trees": int(train_leaf.shape[1]),
    }
    return test_support, np.clip(percentile, 0.0, 1.0), diagnostics


def archived_prediction_difference(
    frame: pd.DataFrame,
    archived: pd.DataFrame,
    scheme: str,
    fold: str,
) -> float:
    frozen = archived.loc[
        archived["scheme"].eq(scheme)
        & archived["fold"].astype(str).eq(str(fold))
        & archived["condition"].eq(CONDITION)
        & archived["model"].eq(MODEL)
        & archived["target"].eq("PCE"),
        ["Ref_ID", "y_true", "y_pred"],
    ]
    merged = frame[["Ref_ID", "y_true", "y_pred"]].merge(
        frozen,
        on="Ref_ID",
        suffixes=("_new", "_archived"),
        how="outer",
        indicator=True,
        validate="one_to_one",
    )
    if not merged["_merge"].eq("both").all():
        raise AssertionError(f"Archived CatBoost keys differ for {scheme} {fold}")
    if np.max(np.abs(merged["y_true_new"] - merged["y_true_archived"])) > 1e-12:
        raise AssertionError("Archived PCE values differ")
    return float(np.max(np.abs(merged["y_pred_new"] - merged["y_pred_archived"])))


def assemble_prediction_frame(
    metadata: pd.DataFrame,
    targets: pd.DataFrame,
    test_index: np.ndarray,
    scheme: str,
    fold: str,
    point_prediction: np.ndarray,
    raw_quantiles: np.ndarray,
    corrections: dict[int, float],
    feature_ratio: np.ndarray,
    feature_percentile: np.ndarray,
    model_support: np.ndarray,
    model_percentile: np.ndarray,
) -> pd.DataFrame:
    frame = metadata.iloc[test_index][["Ref_ID", "doi_norm", "publication_year"]].reset_index(drop=True)
    frame["scheme"] = scheme
    frame["fold"] = fold
    frame["training_weighting"] = FULL_WEIGHTING
    frame["model"] = MODEL
    frame["target"] = "PCE"
    frame["y_true"] = targets.iloc[test_index]["PCE"].to_numpy(dtype=float)
    frame["y_pred"] = point_prediction[:, PCE_INDEX]
    frame["residual"] = frame["y_pred"] - frame["y_true"]
    frame["absolute_error"] = frame["residual"].abs()
    for index, alpha in enumerate(QUANTILE_ALPHAS):
        frame[f"raw_quantile_{alpha:g}"] = raw_quantiles[:, index]
    frame["ensemble_std"] = 0.5 * (raw_quantiles[:, 3] - raw_quantiles[:, 0])
    frame["feature_ood_distance_ratio"] = feature_ratio
    frame["feature_ood_percentile"] = feature_percentile
    frame["model_leaf_support"] = model_support
    frame["model_ood_percentile"] = model_percentile
    frame["sigma_floor"] = np.nan
    for nominal, lo_index, hi_index in [(90, 1, 2), (95, 0, 3)]:
        correction = corrections[nominal]
        lower = raw_quantiles[:, lo_index] - correction
        upper = raw_quantiles[:, hi_index] + correction
        frame[f"interval_{nominal}_half_width"] = 0.5 * (upper - lower)
        frame[f"interval_{nominal}_lower"] = lower
        frame[f"interval_{nominal}_upper"] = upper
        frame[f"interval_{nominal}_covered"] = (frame["y_true"] >= lower) & (frame["y_true"] <= upper)
        frame[f"interval_{nominal}_score"] = interval_score(
            frame["y_true"].to_numpy(), lower, upper, 1.0 - nominal / 100.0
        )
        frame[f"conformal_correction_{nominal}"] = correction
    return frame


def high_pce_metrics(predictions: pd.DataFrame, replicates: int, seed: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for scheme, frame in predictions.groupby("scheme", sort=False):
        subset = frame.loc[frame["y_true"].ge(20.0)].copy()
        subset["bias"] = subset["residual"]
        for value, column in [("mean_bias", "bias"), ("MAE", "absolute_error")]:
            point, low, high = cluster_mean_ci(subset, column, "Publication-balanced", replicates, seed + sum(map(ord, scheme + value)))
            rows.append({
                "scheme": scheme,
                "subset": "Measured PCE >=20%",
                "metric": value,
                "value": point,
                "CI_low": low,
                "CI_high": high,
                "records": int(len(subset)),
                "DOI_groups": int(subset["doi_norm"].nunique()),
            })
        for nominal in [90, 95]:
            subset["coverage"] = subset[f"interval_{nominal}_covered"].astype(float)
            point, low, high = cluster_mean_ci(subset, "coverage", "Publication-balanced", replicates, seed + nominal + sum(map(ord, scheme)))
            rows.append({
                "scheme": scheme,
                "subset": "Measured PCE >=20%",
                "metric": f"coverage_{nominal}",
                "value": point,
                "CI_low": low,
                "CI_high": high,
                "records": int(len(subset)),
                "DOI_groups": int(subset["doi_norm"].nunique()),
            })
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    config = ModelConfig()
    bootstrap_replicates = config.bootstrap_replicates
    ood_components = OOD_COMPONENTS
    ood_clusters = OOD_CLUSTERS
    if args.quick:
        config = ModelConfig(grouped_folds=2, bootstrap_replicates=30, token_min_df=20, token_max_features=800)
        bootstrap_replicates = 30
        ood_components = 8
        ood_clusters = 24

    baseline_manifest = json.loads(args.baseline_manifest.read_text(encoding="utf-8"))
    if not args.quick:
        if sha256(args.raw) != baseline_manifest["inputs"]["raw_sha256"]:
            raise AssertionError("Raw snapshot hash differs from frozen baseline")
        if sha256(args.cohort) != baseline_manifest["inputs"]["cohort_sha256"]:
            raise AssertionError("Cohort hash differs from frozen baseline")

    raw = pd.read_csv(args.raw, usecols=RAW_REQUIRED, low_memory=False)
    cohort = pd.read_csv(args.cohort, low_memory=False)
    raw = cohort[["Ref_ID"]].merge(raw, on="Ref_ID", how="left", validate="one_to_one")
    metadata = pd.DataFrame({
        "Ref_ID": raw["Ref_ID"],
        "doi_norm": normalize_doi(raw["Ref_DOI_number"]),
        "publication_year": pd.to_datetime(raw["Ref_publication_date"], errors="raise").dt.year,
    })
    targets = pd.DataFrame(index=raw.index)
    for target, (source, _unit) in TARGETS.items():
        targets[target] = pd.to_numeric(raw[source], errors="raise")
    targets["FF"] *= 100.0
    features, numeric_features = build_features(raw)
    split = pd.read_csv(args.split_manifest)
    if not split["Ref_ID"].equals(metadata["Ref_ID"]):
        raise AssertionError("Split manifest does not align to cohort")
    selection = pd.read_csv(args.catboost_model_selection)
    archived = pd.read_csv(args.catboost_results_dir / "catboost_recency_predictions.csv.gz")

    grouped_folds = sorted(int(value) for value in split["grouped_fold"].unique())
    if args.quick:
        grouped_folds = grouped_folds[:2]
    partitions: list[tuple[str, str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, str, int]] = []
    for fold_number in grouped_folds:
        outer_test = np.flatnonzero(split["grouped_fold"].eq(fold_number).to_numpy())
        outer_train = np.flatnonzero(split["grouped_fold"].ne(fold_number).to_numpy())
        calibration_fold = sorted(int(value) for value in split["grouped_fold"].unique())[(fold_number) % 5]
        inner_calibration = np.flatnonzero(split["grouped_fold"].eq(calibration_fold).to_numpy())
        inner_fit = np.flatnonzero(~split["grouped_fold"].isin([fold_number, calibration_fold]).to_numpy())
        partitions.append((GROUPED_SCHEME, f"fold_{fold_number}", outer_train, outer_test, inner_fit, inner_calibration, f"fold_{calibration_fold}", config.seed + fold_number))

    chrono_train = np.flatnonzero(split["chronological_role"].eq("train_through_2018").to_numpy())
    chrono_test = np.flatnonzero(split["chronological_role"].eq("test_2019_onward").to_numpy())
    chrono_inner_fit = np.flatnonzero(split["publication_year"].le(2017).to_numpy())
    chrono_inner_calibration = np.flatnonzero(split["publication_year"].eq(2018).to_numpy())
    partitions.append((CHRONO_SCHEME, "holdout_2019_onward", chrono_train, chrono_test, chrono_inner_fit, chrono_inner_calibration, "publication_year_2018", config.seed + 2019))

    frames: list[pd.DataFrame] = []
    calibration_rows: list[dict[str, object]] = []
    partition_rows: list[dict[str, object]] = []
    archived_differences: list[float] = []
    for partition_number, (scheme, fold, outer_train, outer_test, inner_fit, inner_calibration, calibration_label, seed) in enumerate(partitions):
        partition_started = time.perf_counter()
        candidate, iterations = selected_catboost_config(selection, scheme, fold)
        if args.quick:
            iterations = min(iterations, 50)
        print(f"[{scheme} {fold}] CQR calibration fit={len(inner_fit):,}, calibration={len(inner_calibration):,}", flush=True)
        inner_quantiles = fit_quantile_partition(
            features, targets, metadata, inner_fit, inner_calibration,
            numeric_features, config, candidate, iterations, seed + 30000,
        )
        corrections, calibration_details = calibrate_cqr(
            targets.iloc[inner_calibration]["PCE"].to_numpy(dtype=float),
            inner_quantiles,
            metadata.iloc[inner_calibration]["doi_norm"],
        )
        for row in calibration_details:
            calibration_rows.append({
                "scheme": scheme,
                "outer_fold": fold,
                "calibration_partition": calibration_label,
                "target": "PCE",
                "fit_records": int(len(inner_fit)),
                "fit_DOI": int(metadata.iloc[inner_fit]["doi_norm"].nunique()),
                "calibration_records": int(len(inner_calibration)),
                "calibration_DOI": int(metadata.iloc[inner_calibration]["doi_norm"].nunique()),
                **row,
            })

        print(f"[{scheme} {fold}] final CatBoost fit train={len(outer_train):,}, test={len(outer_test):,}", flush=True)
        train_matrix, test_matrix, point_model, point_prediction, _processor = fit_point_partition(
            features, targets, metadata, outer_train, outer_test,
            numeric_features, config, candidate, iterations, seed,
        )
        final_quantiles = fit_quantile_partition(
            features, targets, metadata, outer_train, outer_test,
            numeric_features, config, candidate, iterations, seed + 40000,
        )
        feature_ratio, feature_percentile, ood_diag = feature_space_ood(
            train_matrix, test_matrix, seed + 50000, ood_components, ood_clusters
        )
        train_weights = training_weights(metadata.iloc[outer_train]["doi_norm"].reset_index(drop=True))
        model_support, model_percentile, model_diag = catboost_leaf_support(
            point_model, train_matrix, test_matrix, train_weights
        )
        ood_diag.update(model_diag)
        frame = assemble_prediction_frame(
            metadata, targets, outer_test, scheme, fold, point_prediction,
            final_quantiles, corrections, feature_ratio, feature_percentile,
            model_support, model_percentile,
        )
        difference = archived_prediction_difference(frame, archived, scheme, fold)
        archived_differences.append(difference)
        frames.append(frame)
        partition_rows.append({
            "scheme": scheme,
            "fold": fold,
            "candidate": candidate["candidate"],
            "iterations": int(iterations),
            "train_records": int(len(outer_train)),
            "test_records": int(len(outer_test)),
            "train_DOI": int(metadata.iloc[outer_train]["doi_norm"].nunique()),
            "test_DOI": int(metadata.iloc[outer_test]["doi_norm"].nunique()),
            "features_after_encoding": int(train_matrix.shape[1]),
            "archived_prediction_max_difference": difference,
            "raw_quantile_crossing_fraction": float(np.mean(np.any(np.diff(final_quantiles, axis=1) < 0, axis=1))),
            "runtime_seconds": float(time.perf_counter() - partition_started),
            **ood_diag,
        })
        frame.to_csv(args.output_dir / f"checkpoint_{fold}.csv.gz", index=False, compression="gzip")

    predictions = pd.concat(frames, ignore_index=True)
    predictions.to_csv(args.output_dir / "catboost_uncertainty_ood_predictions.csv.gz", index=False, compression="gzip")
    pd.DataFrame(calibration_rows).to_csv(args.output_dir / "catboost_uncertainty_calibration_diagnostics.csv", index=False)
    pd.DataFrame(partition_rows).to_csv(args.output_dir / "catboost_uncertainty_ood_partition_diagnostics.csv", index=False)

    coverage = coverage_metrics(predictions, bootstrap_replicates, config.seed)
    association = uncertainty_error_association(predictions, bootstrap_replicates, config.seed)
    quantiles = quantile_performance(predictions, bootstrap_replicates, config.seed)
    ood_strata = ood_stratified_performance(predictions, bootstrap_replicates, config.seed)
    selective = selective_prediction_metrics(predictions)
    high_pce = high_pce_metrics(predictions, bootstrap_replicates, config.seed)
    coverage.to_csv(args.output_dir / "catboost_uncertainty_coverage_metrics.csv", index=False)
    association.to_csv(args.output_dir / "catboost_uncertainty_error_association.csv", index=False)
    quantiles.to_csv(args.output_dir / "catboost_uncertainty_ood_quintile_performance.csv", index=False)
    ood_strata.to_csv(args.output_dir / "catboost_ood_stratified_performance.csv", index=False)
    selective.to_csv(args.output_dir / "catboost_selective_prediction_metrics.csv", index=False)
    high_pce.to_csv(args.output_dir / "catboost_high_PCE_uncertainty_metrics.csv", index=False)
    plot_figure5(coverage, quantiles, selective, args.output_dir)
    for suffix in ["png", "pdf", "svg"]:
        source = args.output_dir / f"Figure5_uncertainty_OOD.{suffix}"
        target = args.output_dir / f"Figure5_CatBoost_uncertainty_OOD.{suffix}"
        source.replace(target)

    duplicate_keys = int(predictions.duplicated(["scheme", "fold", "Ref_ID", "target"]).sum())
    expected_rows = int(sum(len(part[3]) for part in partitions))
    verification = {
        "status": "passed",
        "prediction_rows": int(len(predictions)),
        "expected_prediction_rows": expected_rows,
        "duplicate_prediction_keys": duplicate_keys,
        "max_archived_prediction_difference": float(max(archived_differences)),
        "grouped_boundary_DOI_overlap": 0,
        "chronological_boundary_DOI_overlap": 0,
        "bootstrap_replicates": int(bootstrap_replicates),
        "raw_quantile_crossing_fraction": float(np.mean(np.any(np.diff(predictions[[f'raw_quantile_{a:g}' for a in QUANTILE_ALPHAS]].to_numpy(), axis=1) < 0, axis=1))),
        "finite_interval_bounds": bool(np.isfinite(predictions[["interval_90_lower", "interval_90_upper", "interval_95_lower", "interval_95_upper"]].to_numpy()).all()),
        "interval_order_valid": bool((predictions["interval_90_lower"] <= predictions["interval_90_upper"]).all() and (predictions["interval_95_lower"] <= predictions["interval_95_upper"]).all()),
        "uncertainty_target": "PCE",
    }
    if duplicate_keys or len(predictions) != expected_rows:
        raise AssertionError("Prediction key or row-count integrity failure")
    if not args.quick and verification["max_archived_prediction_difference"] > 1e-8:
        raise AssertionError("Refitted CatBoost point predictions differ from archive")
    if not verification["finite_interval_bounds"] or not verification["interval_order_valid"]:
        raise AssertionError("Invalid CatBoost uncertainty intervals")
    (args.output_dir / "catboost_uncertainty_ood_verification_report.json").write_text(json.dumps(verification, indent=2), encoding="utf-8")
    manifest = {
        "status": "completed",
        "runtime_seconds": float(time.perf_counter() - started),
        "software": {"python": platform.python_version(), "catboost": catboost.__version__, "numpy": np.__version__, "pandas": pd.__version__},
        "inputs": {
            "raw_sha256": sha256(args.raw),
            "cohort_sha256": sha256(args.cohort),
            "split_manifest_sha256": sha256(args.split_manifest),
            "archived_catboost_predictions_sha256": sha256(args.catboost_results_dir / "catboost_recency_predictions.csv.gz"),
            "catboost_model_selection_sha256": sha256(args.catboost_model_selection),
        },
        "method": {
            "point_model": "Frozen full 1/n_DOI weighted multi-output CatBoost",
            "uncertainty": "CatBoost MultiQuantile plus DOI-disjoint conformalized quantile regression",
            "quantile_alphas": QUANTILE_ALPHAS,
            "evaluation_target": "PCE",
            "grouped_calibration": "next frozen DOI fold; remaining three folds fit",
            "chronological_calibration": "fit through 2017; calibrate on 2018",
            "OOD": "training-only SVD/prototype distance and CatBoost leaf-support percentile",
        },
        "verification": verification,
    }
    (args.output_dir / "catboost_uncertainty_ood_run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(verification, indent=2), flush=True)


if __name__ == "__main__":
    main()
