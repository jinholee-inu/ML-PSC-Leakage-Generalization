#!/usr/bin/env python3
"""Leakage-controlled baseline models for literature PSC performance.

This script joins the audited final cohort to the frozen 410-column public
Perovskite Database snapshot, constructs a target-free device/process feature
table, and evaluates three model baselines under three validation schemes:

1. Five-fold row-wise random out-of-fold validation (publication leakage benchmark).
2. Five-fold DOI-grouped out-of-fold validation.
3. A chronological holdout trained through 2018 and tested from 2019 onward.

The nonlinear model is a multi-output random forest trained on standardized
PCE, Voc, Jsc, and FF targets. The Elastic Net is fitted independently for each
target. All learned preprocessing is fitted within the corresponding training
partition. Publication year and DOI are used only for splitting and auditing,
never as model inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import re
import sys
import time
import warnings
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGETS = {
    "PCE": ("JV_default_PCE", "%"),
    "Voc": ("JV_default_Voc", "V"),
    "Jsc": ("JV_default_Jsc", "mA cm^-2"),
    "FF": ("JV_default_FF", "percentage points"),
}

RAW_REQUIRED = [
    "Ref_ID",
    "Ref_DOI_number",
    "Ref_publication_date",
    "Cell_architecture",
    "Perovskite_composition_short_form",
    "Perovskite_composition_a_ions",
    "Perovskite_composition_a_ions_coefficients",
    "Perovskite_composition_b_ions",
    "Perovskite_composition_b_ions_coefficients",
    "Perovskite_composition_c_ions",
    "Perovskite_composition_c_ions_coefficients",
    "Perovskite_composition_inorganic",
    "Perovskite_composition_leadfree",
    "Perovskite_band_gap",
    "Perovskite_thickness",
    "Perovskite_dimension_0D",
    "Perovskite_dimension_2D",
    "Perovskite_dimension_2D3D_mixture",
    "Perovskite_dimension_3D",
    "Perovskite_dimension_3D_with_2D_capping_layer",
    "Substrate_stack_sequence",
    "ETL_stack_sequence",
    "HTL_stack_sequence",
    "Backcontact_stack_sequence",
    "Perovskite_deposition_procedure",
    "Perovskite_deposition_synthesis_atmosphere",
    "Perovskite_deposition_thermal_annealing_temperature",
    "Perovskite_deposition_thermal_annealing_time",
    "Perovskite_deposition_thermal_annealing_atmosphere",
    "Perovskite_deposition_substrate_temperature",
    "Perovskite_deposition_solvents",
    "Perovskite_additives_compounds",
    "Perovskite_deposition_quenching_media",
    "Cell_area_measured",
    *[column for column, _unit in TARGETS.values()],
]

CATEGORICAL_FEATURES = [
    "architecture",
    "absorber_short_form",
    "substrate_stack",
    "etl_stack",
    "htl_stack",
    "backcontact_stack",
    "absorber_deposition",
    "absorber_atmosphere",
    "annealing_atmosphere",
    "quenching_medium",
    "solvent_system",
]

COMMON_IONS = {
    "A": ["MA", "FA", "Cs", "Rb", "K", "GUA", "EA", "DMA", "PEA", "BA"],
    "B": ["Pb", "Sn", "Ge", "Bi", "Sb", "Cu"],
    "X": ["I", "Br", "Cl", "F"],
}

BOOL_COLUMNS = {
    "is_inorganic": "Perovskite_composition_inorganic",
    "is_leadfree": "Perovskite_composition_leadfree",
    "dimension_0D": "Perovskite_dimension_0D",
    "dimension_2D": "Perovskite_dimension_2D",
    "dimension_2D3D": "Perovskite_dimension_2D3D_mixture",
    "dimension_3D": "Perovskite_dimension_3D",
    "dimension_3D_2Dcap": "Perovskite_dimension_3D_with_2D_capping_layer",
}

SITE_COLUMNS = {
    "A": (
        "Perovskite_composition_a_ions",
        "Perovskite_composition_a_ions_coefficients",
    ),
    "B": (
        "Perovskite_composition_b_ions",
        "Perovskite_composition_b_ions_coefficients",
    ),
    "X": (
        "Perovskite_composition_c_ions",
        "Perovskite_composition_c_ions_coefficients",
    ),
}


@dataclass(frozen=True)
class ModelConfig:
    row_random_folds: int = 5
    grouped_folds: int = 5
    temporal_cutoff_year: int = 2018
    seed: int = 20260826
    bootstrap_replicates: int = 1000
    ohe_min_frequency: int = 10
    ohe_max_categories: int = 256
    token_min_df: int = 10
    token_max_features: int = 2000
    elastic_alpha: float = 1e-2
    elastic_l1_ratio: float = 0.10
    elastic_max_iter: int = 10000
    elastic_tolerance: float = 1e-3
    rf_estimators: int = 120
    rf_max_features: float = 0.35
    rf_min_samples_leaf: int = 2
    rf_max_samples: float = 0.80


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", required=True, type=Path)
    parser.add_argument("--cohort", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--quick", action="store_true", help="Fast smoke-test settings")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_doi(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .str.strip()
        .str.lower()
        .str.replace(r"^https?://(dx\.)?doi\.org/", "", regex=True)
        .str.replace(r"^doi:\s*", "", regex=True)
        .str.replace(r"[\s\.;,]+$", "", regex=True)
    )


def clean_text(value: object, missing: str = "unknown") -> str:
    if pd.isna(value):
        return missing
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "not reported"}:
        return missing
    return re.sub(r"\s+", " ", text)


def parse_bool(value: object) -> float:
    if pd.isna(value):
        return np.nan
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return 1.0
    if text in {"false", "0", "no"}:
        return 0.0
    return np.nan


def numbers(value: object) -> list[float]:
    if pd.isna(value):
        return []
    text = str(value).strip()
    if text.lower() in {"", "unknown", "nan", "none"}:
        return []
    found = re.findall(r"(?<![A-Za-z])[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
    output: list[float] = []
    for item in found:
        try:
            number = float(item)
        except ValueError:
            continue
        if math.isfinite(number):
            output.append(number)
    return output


def numeric_summary(value: object) -> tuple[float, float, float, float]:
    vals = numbers(value)
    if not vals:
        return np.nan, np.nan, np.nan, 0.0
    arr = np.asarray(vals, dtype=float)
    return float(arr[0]), float(arr[-1]), float(arr.mean()), float(len(arr))


def split_semicolon(value: object) -> list[str]:
    if pd.isna(value):
        return []
    text = str(value).strip()
    if text.lower() in {"", "unknown", "nan", "none", "not reported"}:
        return []
    return [item.strip() for item in text.split(";") if item.strip()]


def composition_fractions(
    ion_value: object, coefficient_value: object, common: list[str]
) -> tuple[dict[str, float], list[str]]:
    ions = split_semicolon(ion_value)
    coeff_text = split_semicolon(coefficient_value)
    coeffs: list[float] = []
    for item in coeff_text:
        vals = numbers(item)
        coeffs.append(vals[0] if vals else np.nan)
    result = {ion: np.nan for ion in common}
    result["other"] = np.nan
    result["parsed"] = 0.0
    if not ions or len(ions) != len(coeffs) or not all(np.isfinite(coeffs)):
        return result, ions
    total = float(np.sum(coeffs))
    if total <= 0:
        return result, ions
    result = {ion: 0.0 for ion in common}
    result["other"] = 0.0
    result["parsed"] = 1.0
    common_lookup = {ion.lower(): ion for ion in common}
    for ion, coefficient in zip(ions, coeffs):
        key = common_lookup.get(ion.strip().lower())
        if key is None:
            result["other"] += float(coefficient) / total
        else:
            result[key] += float(coefficient) / total
    return result, ions


def material_parts(value: object) -> list[str]:
    if pd.isna(value):
        return []
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "unknown", "not reported"}:
        return []
    return [part.strip() for part in re.split(r"\s*(?:\||;|>>)+\s*", text) if part.strip()]


def token(value: object) -> str:
    text = clean_text(value).lower()
    text = text.replace("+", "plus").replace("-", "_")
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return text or "unknown"


def prefixed_tokens(prefix: str, values: list[str]) -> list[str]:
    return [f"{prefix}__{token(value)}" for value in values if token(value)]


def build_features(raw: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    features = pd.DataFrame(index=raw.index)

    categorical_map = {
        "architecture": "Cell_architecture",
        "absorber_short_form": "Perovskite_composition_short_form",
        "substrate_stack": "Substrate_stack_sequence",
        "etl_stack": "ETL_stack_sequence",
        "htl_stack": "HTL_stack_sequence",
        "backcontact_stack": "Backcontact_stack_sequence",
        "absorber_deposition": "Perovskite_deposition_procedure",
        "absorber_atmosphere": "Perovskite_deposition_synthesis_atmosphere",
        "annealing_atmosphere": "Perovskite_deposition_thermal_annealing_atmosphere",
        "quenching_medium": "Perovskite_deposition_quenching_media",
        "solvent_system": "Perovskite_deposition_solvents",
    }
    for output, source in categorical_map.items():
        features[output] = raw[source].map(clean_text).astype("string")

    for output, source in BOOL_COLUMNS.items():
        features[output] = raw[source].map(parse_bool).astype(float)

    bandgap = raw["Perovskite_band_gap"].map(numbers)
    features["bandgap_eV"] = bandgap.map(lambda vals: np.mean(vals) if vals else np.nan)
    thickness = raw["Perovskite_thickness"].map(numeric_summary)
    features[["thickness_first_nm", "thickness_last_nm", "thickness_mean_nm", "thickness_steps"]] = pd.DataFrame(
        thickness.tolist(), index=raw.index
    )
    anneal_temperature = raw["Perovskite_deposition_thermal_annealing_temperature"].map(numeric_summary)
    features[["anneal_temp_first_C", "anneal_temp_last_C", "anneal_temp_mean_C", "anneal_temp_steps"]] = pd.DataFrame(
        anneal_temperature.tolist(), index=raw.index
    )
    anneal_time = raw["Perovskite_deposition_thermal_annealing_time"].map(numeric_summary)
    features[["anneal_time_first_min", "anneal_time_last_min", "anneal_time_mean_min", "anneal_time_steps"]] = pd.DataFrame(
        anneal_time.tolist(), index=raw.index
    )
    substrate_temperature = raw["Perovskite_deposition_substrate_temperature"].map(numeric_summary)
    features[["substrate_temp_first_C", "substrate_temp_last_C", "substrate_temp_mean_C", "substrate_temp_steps"]] = pd.DataFrame(
        substrate_temperature.tolist(), index=raw.index
    )
    area = pd.to_numeric(raw["Cell_area_measured"], errors="coerce")
    area = area.where(area > 0)
    features["cell_area_cm2"] = area
    features["log10_cell_area_cm2"] = np.log10(area)

    token_rows: list[str] = []
    ion_numeric_columns: list[str] = []
    per_row_compositions: dict[str, list[dict[str, float]]] = {site: [] for site in SITE_COLUMNS}
    per_row_ions: dict[str, list[list[str]]] = {site: [] for site in SITE_COLUMNS}
    for site, (ion_column, coefficient_column) in SITE_COLUMNS.items():
        for ion_value, coefficient_value in zip(raw[ion_column], raw[coefficient_column]):
            fractions, ions = composition_fractions(
                ion_value, coefficient_value, COMMON_IONS[site]
            )
            per_row_compositions[site].append(fractions)
            per_row_ions[site].append(ions)
        site_frame = pd.DataFrame(per_row_compositions[site], index=raw.index)
        site_frame = site_frame.rename(columns=lambda name: f"comp_{site}_{name}_fraction")
        for column in site_frame.columns:
            features[column] = site_frame[column].astype(float)
            ion_numeric_columns.append(column)

    token_source_map = {
        "sub": "Substrate_stack_sequence",
        "etl": "ETL_stack_sequence",
        "htl": "HTL_stack_sequence",
        "back": "Backcontact_stack_sequence",
        "solv": "Perovskite_deposition_solvents",
        "add": "Perovskite_additives_compounds",
        "quench": "Perovskite_deposition_quenching_media",
    }
    for row_position, (_index, row) in enumerate(raw.iterrows()):
        pieces: list[str] = []
        for site in ["A", "B", "X"]:
            pieces.extend(prefixed_tokens(site.lower(), per_row_ions[site][row_position]))
        for prefix, source in token_source_map.items():
            pieces.extend(prefixed_tokens(prefix, material_parts(row[source])))
        token_rows.append(" ".join(pieces) if pieces else "no_material_tokens")
    features["role_tokens"] = pd.Series(token_rows, index=raw.index, dtype="string")

    count_map = {
        "substrate_layer_count": "Substrate_stack_sequence",
        "etl_layer_count": "ETL_stack_sequence",
        "htl_layer_count": "HTL_stack_sequence",
        "backcontact_layer_count": "Backcontact_stack_sequence",
        "solvent_count": "Perovskite_deposition_solvents",
        "additive_count": "Perovskite_additives_compounds",
        "quench_count": "Perovskite_deposition_quenching_media",
        "deposition_step_count": "Perovskite_deposition_procedure",
    }
    for output, source in count_map.items():
        features[output] = raw[source].map(lambda value: float(len(material_parts(value))))

    numeric_features = [
        column
        for column in features.columns
        if column not in CATEGORICAL_FEATURES and column != "role_tokens"
    ]
    assert not set(TARGETS).intersection(features.columns)
    return features, numeric_features


def make_preprocessor(config: ModelConfig, numeric_features: list[str]) -> ColumnTransformer:
    numeric = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler(with_mean=False)),
        ]
    )
    categorical = OneHotEncoder(
        handle_unknown="infrequent_if_exist",
        min_frequency=config.ohe_min_frequency,
        max_categories=config.ohe_max_categories,
        sparse_output=True,
        dtype=np.float32,
    )
    tokens = CountVectorizer(
        binary=True,
        lowercase=False,
        min_df=config.token_min_df,
        max_features=config.token_max_features,
        token_pattern=r"(?u)\b[a-z][a-z0-9_]{1,}\b",
        dtype=np.float32,
    )
    return ColumnTransformer(
        [
            ("numeric", numeric, numeric_features),
            ("categorical", categorical, CATEGORICAL_FEATURES),
            ("materials", tokens, "role_tokens"),
        ],
        remainder="drop",
        sparse_threshold=1.0,
        verbose_feature_names_out=True,
    )


def metric_values(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(math.sqrt(mean_squared_error(y_true, y_pred))),
    }


def group_bootstrap_samples(
    frame: pd.DataFrame, replicates: int, seed: int
) -> np.ndarray:
    work = frame[["doi_norm", "y_true", "y_pred"]].copy()
    work["abs_error"] = (work["y_true"] - work["y_pred"]).abs()
    work["sq_error"] = (work["y_true"] - work["y_pred"]) ** 2
    work["y2"] = work["y_true"] ** 2
    grouped = work.groupby("doi_norm", sort=False).agg(
        n=("y_true", "size"),
        y_sum=("y_true", "sum"),
        y2_sum=("y2", "sum"),
        abs_error_sum=("abs_error", "sum"),
        sq_error_sum=("sq_error", "sum"),
    )
    stats = grouped.to_numpy(dtype=float)
    group_count = len(stats)
    rng = np.random.default_rng(seed)
    boot = np.empty((replicates, 3), dtype=float)
    for replicate in range(replicates):
        sample = stats[rng.integers(0, group_count, size=group_count)]
        totals = sample.sum(axis=0)
        n, y_sum, y2_sum, abs_sum, sq_sum = totals
        sst = y2_sum - (y_sum * y_sum / n)
        boot[replicate, 0] = 1.0 - sq_sum / sst if sst > 0 else np.nan
        boot[replicate, 1] = abs_sum / n
        boot[replicate, 2] = math.sqrt(sq_sum / n)
    return boot


def group_bootstrap_ci(
    frame: pd.DataFrame, replicates: int, seed: int
) -> dict[str, tuple[float, float]]:
    boot = group_bootstrap_samples(frame, replicates=replicates, seed=seed)
    output: dict[str, tuple[float, float]] = {}
    for index, metric in enumerate(["R2", "MAE", "RMSE"]):
        low, high = np.nanpercentile(boot[:, index], [2.5, 97.5])
        output[metric] = (float(low), float(high))
    return output


def generalization_gap_table(
    predictions: pd.DataFrame,
    summary: pd.DataFrame,
    config: ModelConfig,
    grouped_scheme: str = "DOI-grouped 5-fold",
    chronological_scheme: str = "Chronological >2018",
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model in ["Elastic Net", "Random Forest"]:
        for target in TARGETS:
            grouped_prediction = predictions.loc[
                predictions["scheme"].eq(grouped_scheme)
                & predictions["model"].eq(model)
                & predictions["target"].eq(target)
            ]
            chronological_prediction = predictions.loc[
                predictions["scheme"].eq(chronological_scheme)
                & predictions["model"].eq(model)
                & predictions["target"].eq(target)
            ]
            grouped_metric = summary.loc[
                summary["scheme"].eq(grouped_scheme)
                & summary["model"].eq(model)
                & summary["target"].eq(target)
            ].iloc[0]
            chronological_metric = summary.loc[
                summary["scheme"].eq(chronological_scheme)
                & summary["model"].eq(model)
                & summary["target"].eq(target)
            ].iloc[0]
            base_seed = config.seed + sum(map(ord, model + target))
            grouped_boot = group_bootstrap_samples(
                grouped_prediction,
                replicates=config.bootstrap_replicates,
                seed=base_seed + 101,
            )
            chronological_boot = group_bootstrap_samples(
                chronological_prediction,
                replicates=config.bootstrap_replicates,
                seed=base_seed + 202,
            )
            delta_r2_boot = chronological_boot[:, 0] - grouped_boot[:, 0]
            mae_ratio_boot = chronological_boot[:, 1] / grouped_boot[:, 1]
            delta_low, delta_high = np.nanpercentile(delta_r2_boot, [2.5, 97.5])
            ratio_low, ratio_high = np.nanpercentile(mae_ratio_boot, [2.5, 97.5])
            rows.append(
                {
                    "model": model,
                    "target": target,
                    "grouped_R2": grouped_metric["R2"],
                    "grouped_MAE": grouped_metric["MAE"],
                    "grouped_RMSE": grouped_metric["RMSE"],
                    "chronological_R2": chronological_metric["R2"],
                    "chronological_MAE": chronological_metric["MAE"],
                    "chronological_RMSE": chronological_metric["RMSE"],
                    "delta_R2_chrono_minus_grouped": chronological_metric["R2"]
                    - grouped_metric["R2"],
                    "delta_R2_CI_low": float(delta_low),
                    "delta_R2_CI_high": float(delta_high),
                    "MAE_ratio_chrono_over_grouped": chronological_metric["MAE"]
                    / grouped_metric["MAE"],
                    "MAE_ratio_CI_low": float(ratio_low),
                    "MAE_ratio_CI_high": float(ratio_high),
                }
            )
    return pd.DataFrame(rows)


def paired_publication_bootstrap(
    row_random: pd.DataFrame,
    doi_grouped: pd.DataFrame,
    replicates: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """Paired DOI-cluster bootstrap for random-versus-grouped predictions."""
    left = row_random[["Ref_ID", "doi_norm", "y_true", "y_pred"]].rename(
        columns={"doi_norm": "doi_random", "y_true": "y_random", "y_pred": "pred_random"}
    )
    right = doi_grouped[["Ref_ID", "doi_norm", "y_true", "y_pred"]].rename(
        columns={"doi_norm": "doi_grouped", "y_true": "y_grouped", "y_pred": "pred_grouped"}
    )
    paired = left.merge(right, on="Ref_ID", how="inner", validate="one_to_one")
    if len(paired) != len(left) or len(paired) != len(right):
        raise AssertionError("Random and DOI-grouped predictions do not cover identical records")
    if not paired["doi_random"].eq(paired["doi_grouped"]).all():
        raise AssertionError("DOI labels disagree between paired validation predictions")
    if not np.allclose(paired["y_random"], paired["y_grouped"], rtol=0, atol=1e-12):
        raise AssertionError("Target values disagree between paired validation predictions")

    paired["y2"] = paired["y_random"] ** 2
    paired["abs_random"] = (paired["pred_random"] - paired["y_random"]).abs()
    paired["sq_random"] = (paired["pred_random"] - paired["y_random"]) ** 2
    paired["abs_grouped"] = (paired["pred_grouped"] - paired["y_random"]).abs()
    paired["sq_grouped"] = (paired["pred_grouped"] - paired["y_random"]) ** 2
    grouped = paired.groupby("doi_random", sort=False).agg(
        n=("y_random", "size"),
        y_sum=("y_random", "sum"),
        y2_sum=("y2", "sum"),
        abs_random=("abs_random", "sum"),
        sq_random=("sq_random", "sum"),
        abs_grouped=("abs_grouped", "sum"),
        sq_grouped=("sq_grouped", "sum"),
    )
    stats = grouped.to_numpy(dtype=float)
    group_count = len(stats)
    rng = np.random.default_rng(seed)
    output = {
        "delta_R2_random_minus_grouped": np.empty(replicates, dtype=float),
        "MAE_reduction_fraction": np.empty(replicates, dtype=float),
        "RMSE_reduction_fraction": np.empty(replicates, dtype=float),
    }
    for replicate in range(replicates):
        totals = stats[rng.integers(0, group_count, size=group_count)].sum(axis=0)
        n, y_sum, y2_sum, abs_random, sq_random, abs_grouped, sq_grouped = totals
        sst = y2_sum - y_sum * y_sum / n
        r2_random = 1.0 - sq_random / sst if sst > 0 else np.nan
        r2_grouped = 1.0 - sq_grouped / sst if sst > 0 else np.nan
        mae_random = abs_random / n
        mae_grouped = abs_grouped / n
        rmse_random = math.sqrt(sq_random / n)
        rmse_grouped = math.sqrt(sq_grouped / n)
        output["delta_R2_random_minus_grouped"][replicate] = r2_random - r2_grouped
        output["MAE_reduction_fraction"][replicate] = 1.0 - mae_random / mae_grouped
        output["RMSE_reduction_fraction"][replicate] = 1.0 - rmse_random / rmse_grouped
    return output


def publication_leakage_inflation_table(
    predictions: pd.DataFrame,
    summary: pd.DataFrame,
    config: ModelConfig,
    row_random_scheme: str = "Row-wise random 5-fold",
    grouped_scheme: str = "DOI-grouped 5-fold",
) -> pd.DataFrame:
    """Quantify apparent performance inflation caused by row-wise splitting."""
    rows: list[dict[str, object]] = []
    for model in ["Elastic Net", "Random Forest"]:
        for target in TARGETS:
            row_prediction = predictions.loc[
                predictions["scheme"].eq(row_random_scheme)
                & predictions["model"].eq(model)
                & predictions["target"].eq(target)
            ]
            grouped_prediction = predictions.loc[
                predictions["scheme"].eq(grouped_scheme)
                & predictions["model"].eq(model)
                & predictions["target"].eq(target)
            ]
            row_metric = summary.loc[
                summary["scheme"].eq(row_random_scheme)
                & summary["model"].eq(model)
                & summary["target"].eq(target)
            ].iloc[0]
            grouped_metric = summary.loc[
                summary["scheme"].eq(grouped_scheme)
                & summary["model"].eq(model)
                & summary["target"].eq(target)
            ].iloc[0]
            boot = paired_publication_bootstrap(
                row_prediction,
                grouped_prediction,
                replicates=config.bootstrap_replicates,
                seed=config.seed + sum(map(ord, model + target)) + 303,
            )
            delta_low, delta_high = np.nanpercentile(
                boot["delta_R2_random_minus_grouped"], [2.5, 97.5]
            )
            mae_low, mae_high = np.nanpercentile(
                boot["MAE_reduction_fraction"], [2.5, 97.5]
            )
            rmse_low, rmse_high = np.nanpercentile(
                boot["RMSE_reduction_fraction"], [2.5, 97.5]
            )
            rows.append(
                {
                    "model": model,
                    "target": target,
                    "unit": TARGETS[target][1],
                    "row_random_R2": row_metric["R2"],
                    "doi_grouped_R2": grouped_metric["R2"],
                    "delta_R2_random_minus_grouped": row_metric["R2"] - grouped_metric["R2"],
                    "delta_R2_CI_low": float(delta_low),
                    "delta_R2_CI_high": float(delta_high),
                    "row_random_MAE": row_metric["MAE"],
                    "doi_grouped_MAE": grouped_metric["MAE"],
                    "MAE_reduction_fraction": 1.0 - row_metric["MAE"] / grouped_metric["MAE"],
                    "MAE_reduction_CI_low": float(mae_low),
                    "MAE_reduction_CI_high": float(mae_high),
                    "row_random_RMSE": row_metric["RMSE"],
                    "doi_grouped_RMSE": grouped_metric["RMSE"],
                    "RMSE_reduction_fraction": 1.0 - row_metric["RMSE"] / grouped_metric["RMSE"],
                    "RMSE_reduction_CI_low": float(rmse_low),
                    "RMSE_reduction_CI_high": float(rmse_high),
                }
            )
    return pd.DataFrame(rows)


def row_random_leakage_tables(
    split_manifest: pd.DataFrame,
    predictions: pd.DataFrame,
    row_random_scheme: str = "Row-wise random 5-fold",
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    """Describe DOI fragmentation and row-random performance by leakage exposure."""
    doi_stats = split_manifest.groupby("doi_norm", sort=False).agg(
        records=("Ref_ID", "size"),
        random_folds=("row_random_fold", "nunique"),
    )
    doi_stats["fragmented_across_random_folds"] = doi_stats["random_folds"].gt(1)
    fragmented = doi_stats["fragmented_across_random_folds"]
    exposed_records = int(doi_stats.loc[fragmented, "records"].sum())
    leakage_summary: dict[str, object] = {
        "row_random_scheme": row_random_scheme,
        "records": int(len(split_manifest)),
        "DOI_groups": int(len(doi_stats)),
        "DOI_groups_fragmented_across_folds": int(fragmented.sum()),
        "fraction_DOI_groups_fragmented": float(fragmented.mean()),
        "records_with_same_DOI_in_training": exposed_records,
        "fraction_records_with_same_DOI_in_training": float(exposed_records / len(split_manifest)),
        "singleton_DOI_groups": int(doi_stats["records"].eq(1).sum()),
    }

    size_bins = pd.cut(
        doi_stats["records"],
        bins=[0, 1, 3, 9, 19, np.inf],
        labels=["1", "2–3", "4–9", "10–19", "≥20"],
        right=True,
    )
    size_frame = doi_stats.assign(DOI_size_bin=size_bins).reset_index()
    size_rows: list[dict[str, object]] = []
    for label, group in size_frame.groupby("DOI_size_bin", observed=True, sort=False):
        group_fragmented = group["fragmented_across_random_folds"]
        records_in_fragmented = int(group.loc[group_fragmented, "records"].sum())
        size_rows.append(
            {
                "DOI_size_bin": str(label),
                "DOI_groups": int(len(group)),
                "records": int(group["records"].sum()),
                "fragmented_DOI_groups": int(group_fragmented.sum()),
                "fraction_DOI_groups_fragmented": float(group_fragmented.mean()),
                "records_with_same_DOI_in_training": records_in_fragmented,
                "fraction_records_with_same_DOI_in_training": float(
                    records_in_fragmented / group["records"].sum()
                ),
            }
        )
    by_size = pd.DataFrame(size_rows)

    exposure = split_manifest[["Ref_ID", "doi_norm", "row_random_DOI_seen_in_train"]]
    random_predictions = predictions.loc[predictions["scheme"].eq(row_random_scheme)].merge(
        exposure, on=["Ref_ID", "doi_norm"], how="left", validate="many_to_one"
    )
    strata_rows: list[dict[str, object]] = []
    for (model, target, seen), group in random_predictions.groupby(
        ["model", "target", "row_random_DOI_seen_in_train"], sort=False
    ):
        point = metric_values(group["y_true"].to_numpy(), group["y_pred"].to_numpy())
        strata_rows.append(
            {
                "model": model,
                "target": target,
                "same_DOI_seen_in_training": bool(seen),
                "records": int(len(group)),
                "DOI_groups": int(group["doi_norm"].nunique()),
                **point,
            }
        )
    strata = pd.DataFrame(strata_rows)
    return leakage_summary, by_size, strata


def append_predictions(
    storage: list[pd.DataFrame],
    metadata: pd.DataFrame,
    scheme: str,
    fold: str,
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
                "model": model,
                "target": target,
                "y_true": y_true,
                "y_pred": y_pred,
            }
        )
    )


def fit_partition(
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
    keep_feature_importance: bool = False,
) -> tuple[pd.DataFrame | None, int]:
    start = time.perf_counter()
    processor = make_preprocessor(config, numeric_features)
    train_matrix = processor.fit_transform(features.iloc[train_index])
    test_matrix = processor.transform(features.iloc[test_index])
    if sparse.issparse(train_matrix):
        train_matrix = train_matrix.tocsr().astype(np.float32)
        test_matrix = test_matrix.tocsr().astype(np.float32)
    else:
        train_matrix = np.asarray(train_matrix, dtype=np.float32)
        test_matrix = np.asarray(test_matrix, dtype=np.float32)
    # Coordinate-descent Elastic Net requires float64 here. With float32 sparse
    # matrices, the dual-gap calculation can stall despite stable predictions.
    linear_train_matrix = train_matrix.astype(np.float64)
    linear_test_matrix = test_matrix.astype(np.float64)

    y_train = targets.iloc[train_index].to_numpy(dtype=float)
    y_test = targets.iloc[test_index].to_numpy(dtype=float)
    y_mean = y_train.mean(axis=0)
    y_std = y_train.std(axis=0, ddof=0)
    y_std = np.where(y_std > 0, y_std, 1.0)
    y_train_scaled = (y_train - y_mean) / y_std

    for target_index, target in enumerate(TARGETS):
        dummy_pred = np.full(len(test_index), y_mean[target_index], dtype=float)
        append_predictions(
            prediction_storage,
            metadata.iloc[test_index],
            scheme,
            fold,
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
            elastic.fit(linear_train_matrix, y_train_scaled[:, target_index])
        pred = elastic.predict(linear_test_matrix) * y_std[target_index] + y_mean[target_index]
        append_predictions(
            prediction_storage,
            metadata.iloc[test_index],
            scheme,
            fold,
            "Elastic Net",
            target,
            y_test[:, target_index],
            pred,
        )
        diagnostics.append(
            {
                "scheme": scheme,
                "fold": fold,
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
    forest.fit(train_matrix, y_train_scaled)
    forest_pred = forest.predict(test_matrix) * y_std + y_mean
    for target_index, target in enumerate(TARGETS):
        append_predictions(
            prediction_storage,
            metadata.iloc[test_index],
            scheme,
            fold,
            "Random Forest",
            target,
            y_test[:, target_index],
            forest_pred[:, target_index],
        )

    importance = None
    if keep_feature_importance:
        names = processor.get_feature_names_out()
        importance = (
            pd.DataFrame(
                {
                    "feature": names,
                    "importance": forest.feature_importances_,
                }
            )
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
        importance["rank"] = np.arange(1, len(importance) + 1)

    diagnostics.append(
        {
            "scheme": scheme,
            "fold": fold,
            "model": "preprocessor_and_random_forest",
            "target": "all",
            "train_records": int(len(train_index)),
            "test_records": int(len(test_index)),
            "train_DOI": int(metadata.iloc[train_index]["doi_norm"].nunique()),
            "test_DOI": int(metadata.iloc[test_index]["doi_norm"].nunique()),
            "features_after_encoding": int(train_matrix.shape[1]),
            "fit_seconds": float(time.perf_counter() - start),
        }
    )
    return importance, int(train_matrix.shape[1])


def summarize_predictions(
    predictions: pd.DataFrame, config: ModelConfig
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    for (scheme, model, target), group in predictions.groupby(
        ["scheme", "model", "target"], sort=False
    ):
        point = metric_values(group["y_true"].to_numpy(), group["y_pred"].to_numpy())
        ci = group_bootstrap_ci(
            group,
            replicates=config.bootstrap_replicates,
            seed=config.seed + sum(map(ord, scheme + model + target)),
        )
        row: dict[str, object] = {
            "scheme": scheme,
            "model": model,
            "target": target,
            "unit": TARGETS[target][1],
            "records": int(len(group)),
            "DOI_groups": int(group["doi_norm"].nunique()),
        }
        for metric in ["R2", "MAE", "RMSE"]:
            row[metric] = point[metric]
            row[f"{metric}_CI_low"] = ci[metric][0]
            row[f"{metric}_CI_high"] = ci[metric][1]
        summary_rows.append(row)
        for fold, fold_group in group.groupby("fold", sort=False):
            fold_point = metric_values(
                fold_group["y_true"].to_numpy(), fold_group["y_pred"].to_numpy()
            )
            fold_rows.append(
                {
                    "scheme": scheme,
                    "fold": fold,
                    "model": model,
                    "target": target,
                    "records": int(len(fold_group)),
                    "DOI_groups": int(fold_group["doi_norm"].nunique()),
                    **fold_point,
                }
            )
    summary = pd.DataFrame(summary_rows)
    fold_metrics = pd.DataFrame(fold_rows)

    dummy_mae = summary.loc[
        summary["model"].eq("Dummy mean"), ["scheme", "target", "MAE"]
    ].rename(columns={"MAE": "dummy_MAE"})
    summary = summary.merge(dummy_mae, on=["scheme", "target"], how="left")
    summary["MAE_skill_vs_dummy"] = 1.0 - summary["MAE"] / summary["dummy_MAE"]
    return summary, fold_metrics


def plot_results(
    summary: pd.DataFrame,
    predictions: pd.DataFrame,
    generalization: pd.DataFrame,
    inflation: pd.DataFrame,
    output_dir: Path,
) -> None:
    sns.set_theme(style="whitegrid", context="paper")
    model_order = ["Elastic Net", "Random Forest"]
    target_order = list(TARGETS)
    scheme_order = ["Row-wise random 5-fold", "DOI-grouped 5-fold", "Chronological >2018"]
    scheme_labels = {
        "Row-wise random 5-fold": "Row-random",
        "DOI-grouped 5-fold": "DOI-grouped",
        "Chronological >2018": "Chronological",
    }
    scheme_colors = {
        "Row-wise random 5-fold": "#5B7FA3",
        "DOI-grouped 5-fold": "#E28E2C",
        "Chronological >2018": "#C43C39",
    }
    model_colors = {"Elastic Net": "#E28E2C", "Random Forest": "#C43C39"}

    fig, axes = plt.subplots(2, 2, figsize=(11.6, 8.6), constrained_layout=True)
    ordered_rows = [(model, target) for target in target_order for model in model_order]
    row_labels = [f"{target} · {model}" for model, target in ordered_rows]

    def metric_matrix(metric: str) -> pd.DataFrame:
        indexed = summary.set_index(["scheme", "model", "target"])
        values = []
        for model, target in ordered_rows:
            values.append(
                [indexed.loc[(scheme, model, target), metric] for scheme in scheme_order]
            )
        return pd.DataFrame(
            values,
            index=row_labels,
            columns=[scheme_labels[item] for item in scheme_order],
        )

    r2_matrix = metric_matrix("R2")
    sns.heatmap(
        r2_matrix,
        annot=True,
        fmt=".2f",
        cmap="RdYlBu",
        center=0.0,
        vmin=min(-0.2, float(np.nanmin(r2_matrix.to_numpy()))),
        vmax=0.8,
        linewidths=0.6,
        linecolor="white",
        cbar_kws={"label": "$R^2$"},
        ax=axes[0, 0],
    )
    axes[0, 0].set_title("(a) Predictive performance", loc="left", fontweight="bold")
    axes[0, 0].set_xlabel("Validation scheme")
    axes[0, 0].set_ylabel("Target and model")

    skill_matrix = metric_matrix("MAE_skill_vs_dummy")
    sns.heatmap(
        skill_matrix,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        vmin=0.0,
        vmax=max(0.5, float(np.nanmax(skill_matrix.to_numpy()))),
        linewidths=0.6,
        linecolor="white",
        cbar_kws={"label": "MAE skill vs dummy"},
        ax=axes[0, 1],
    )
    axes[0, 1].set_title("(b) Error reduction versus mean baseline", loc="left", fontweight="bold")
    axes[0, 1].set_xlabel("Validation scheme")
    axes[0, 1].set_ylabel("")

    x_positions = np.arange(len(target_order), dtype=float)
    bar_width = 0.36
    for model_index, model in enumerate(["Elastic Net", "Random Forest"]):
        model_gap = inflation.loc[inflation["model"].eq(model)].set_index("target").reindex(target_order)
        values = model_gap["delta_R2_random_minus_grouped"].to_numpy(dtype=float)
        lows = model_gap["delta_R2_CI_low"].to_numpy(dtype=float)
        highs = model_gap["delta_R2_CI_high"].to_numpy(dtype=float)
        centers = x_positions + (model_index - 0.5) * bar_width
        axes[1, 0].bar(
            centers,
            values,
            width=bar_width,
            color=model_colors[model],
            label=model,
            alpha=0.92,
        )
        axes[1, 0].errorbar(
            centers,
            values,
            yerr=np.vstack([values - lows, highs - values]),
            fmt="none",
            ecolor="#2F3A4A",
            elinewidth=0.9,
            capsize=2.5,
        )
    axes[1, 0].axhline(0, color="black", lw=0.9)
    axes[1, 0].set_title("(c) Apparent gain from row-wise splitting", loc="left", fontweight="bold")
    axes[1, 0].set_xlabel("Prediction target")
    axes[1, 0].set_ylabel(r"$\Delta R^2$ (row-random − DOI-grouped)")
    axes[1, 0].set_xticks(x_positions, target_order)
    axes[1, 0].legend(title="", frameon=False, loc="best")

    bin_edges = [-np.inf, 5, 10, 15, 20, np.inf]
    bin_labels = ["0–5", "5–10", "10–15", "15–20", "≥20"]
    for scheme in scheme_order:
        pce = predictions.loc[
            predictions["scheme"].eq(scheme)
            & predictions["model"].eq("Random Forest")
            & predictions["target"].eq("PCE")
        ].copy()
        pce["PCE_bin"] = pd.cut(pce["y_true"], bins=bin_edges, labels=bin_labels)
        calibration = pce.groupby("PCE_bin", observed=True).agg(
            measured=("y_true", "mean"),
            predicted=("y_pred", "mean"),
            records=("Ref_ID", "size"),
        )
        axes[1, 1].plot(
            calibration["measured"],
            calibration["predicted"],
            marker="o",
            markersize=5,
            lw=1.7,
            color=scheme_colors[scheme],
            label=scheme_labels[scheme],
        )
    axes[1, 1].plot([0, 25], [0, 25], ls="--", color="#2F3A4A", lw=1.1, label="Ideal")
    axes[1, 1].set_xlim(0, 25)
    axes[1, 1].set_ylim(0, 25)
    axes[1, 1].set_xlabel("Measured PCE (%)")
    axes[1, 1].set_ylabel("Mean predicted PCE (%)")
    axes[1, 1].set_title("(d) Random Forest PCE calibration", loc="left", fontweight="bold")
    axes[1, 1].legend(frameon=False, loc="upper left")

    for axis in axes.flat:
        axis.tick_params(labelsize=8.5)
    fig.suptitle(
        "Publication leakage and temporal generalization in PSC prediction",
        fontsize=14,
        fontweight="bold",
    )
    for suffix in ["png", "pdf", "svg"]:
        path = output_dir / f"Figure3_baseline_validation.{suffix}"
        # The vector PDF/SVG are the publication masters. A 300 dpi PNG keeps
        # raster export reliable across constrained rendering environments.
        kwargs = {"dpi": 300} if suffix == "png" else {}
        fig.savefig(path, bbox_inches="tight", **kwargs)
    plt.close(fig)


def format_metric(value: float, low: float, high: float, decimals: int) -> str:
    return f"{value:.{decimals}f} [{low:.{decimals}f}, {high:.{decimals}f}]"


def write_markdown_tables(summary: pd.DataFrame, output_dir: Path) -> None:
    rows: list[str] = []
    rows.append("| Validation | Model | Target | R² [95% CI] | MAE [95% CI] | RMSE [95% CI] |")
    rows.append("|---|---|---|---:|---:|---:|")
    order = {"Dummy mean": 0, "Elastic Net": 1, "Random Forest": 2}
    ordered = summary.assign(model_order=summary["model"].map(order)).sort_values(
        ["scheme", "model_order", "target"]
    )
    for _, row in ordered.iterrows():
        rows.append(
            "| {scheme} | {model} | {target} | {r2} | {mae} {unit} | {rmse} {unit} |".format(
                scheme=row["scheme"],
                model=row["model"],
                target=row["target"],
                r2=format_metric(row["R2"], row["R2_CI_low"], row["R2_CI_high"], 3),
                mae=format_metric(row["MAE"], row["MAE_CI_low"], row["MAE_CI_high"], 3),
                rmse=format_metric(row["RMSE"], row["RMSE_CI_low"], row["RMSE_CI_high"], 3),
                unit=row["unit"],
            )
        )
    (output_dir / "baseline_metrics_table.md").write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
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
    started = time.perf_counter()

    raw = pd.read_csv(args.raw, usecols=RAW_REQUIRED, low_memory=False)
    cohort = pd.read_csv(args.cohort, low_memory=False)
    missing = sorted(set(RAW_REQUIRED) - set(raw.columns))
    if missing:
        raise ValueError(f"Missing raw columns: {missing}")
    if not cohort["Ref_ID"].is_unique:
        raise ValueError("Audited cohort Ref_ID must be unique")
    raw = cohort[["Ref_ID"]].merge(raw, on="Ref_ID", how="left", validate="one_to_one")
    if raw["Ref_DOI_number"].isna().any():
        raise ValueError("Some final-cohort records did not match the raw snapshot")

    metadata = pd.DataFrame(
        {
            "Ref_ID": raw["Ref_ID"],
            "doi_raw": raw["Ref_DOI_number"].astype("string"),
            "doi_norm": normalize_doi(raw["Ref_DOI_number"]),
            "publication_year": pd.to_datetime(raw["Ref_publication_date"], errors="raise").dt.year,
        }
    )
    if metadata["doi_norm"].isna().any() or metadata["doi_norm"].eq("").any():
        raise ValueError("DOI normalization produced empty groups")
    doi_years = metadata.groupby("doi_norm")["publication_year"].nunique()
    if (doi_years > 1).any():
        raise ValueError("A normalized DOI has records assigned to multiple publication years")

    targets = pd.DataFrame(index=raw.index)
    for target, (source, _unit) in TARGETS.items():
        targets[target] = pd.to_numeric(raw[source], errors="raise")
    targets["FF"] = targets["FF"] * 100.0
    if targets.isna().any().any():
        raise ValueError("Targets contain missing values")

    features, numeric_features = build_features(raw)
    forbidden = {
        "Ref_ID",
        "Ref_DOI_number",
        "Ref_publication_date",
        "publication_year",
        *[source for source, _unit in TARGETS.values()],
    }
    overlap = forbidden.intersection(features.columns)
    if overlap:
        raise AssertionError(f"Forbidden target/metadata features present: {sorted(overlap)}")

    prediction_storage: list[pd.DataFrame] = []
    diagnostics: list[dict[str, object]] = []
    split_manifest = metadata.copy()

    row_random_scheme = (
        "Row-wise random 5-fold" if not args.quick else "Row-wise random 2-fold quick"
    )
    row_splitter = KFold(
        n_splits=config.row_random_folds, shuffle=True, random_state=config.seed
    )
    split_manifest["row_random_fold"] = -1
    for fold_number, (train_index, test_index) in enumerate(
        row_splitter.split(features, targets), start=1
    ):
        split_manifest.loc[test_index, "row_random_fold"] = fold_number
        print(
            f"[row-random fold {fold_number}/{config.row_random_folds}] "
            f"train={len(train_index):,}, test={len(test_index):,}",
            flush=True,
        )
        fit_partition(
            features,
            targets,
            metadata,
            train_index,
            test_index,
            numeric_features,
            config,
            scheme=row_random_scheme,
            fold=f"fold_{fold_number}",
            prediction_storage=prediction_storage,
            diagnostics=diagnostics,
        )
    if (split_manifest["row_random_fold"] < 1).any():
        raise AssertionError("Some records lack row-random-fold assignment")
    random_fold_counts = split_manifest.groupby("doi_norm")["row_random_fold"].transform("nunique")
    split_manifest["row_random_DOI_seen_in_train"] = random_fold_counts.gt(1)

    group_splitter = GroupKFold(
        n_splits=config.grouped_folds, shuffle=True, random_state=config.seed
    )
    split_manifest["grouped_fold"] = -1
    for fold_number, (train_index, test_index) in enumerate(
        group_splitter.split(features, targets, groups=metadata["doi_norm"]), start=1
    ):
        train_doi = set(metadata.iloc[train_index]["doi_norm"])
        test_doi = set(metadata.iloc[test_index]["doi_norm"])
        if train_doi.intersection(test_doi):
            raise AssertionError("DOI leakage detected in grouped validation")
        split_manifest.loc[test_index, "grouped_fold"] = fold_number
        print(
            f"[grouped fold {fold_number}/{config.grouped_folds}] "
            f"train={len(train_index):,}, test={len(test_index):,}",
            flush=True,
        )
        fit_partition(
            features,
            targets,
            metadata,
            train_index,
            test_index,
            numeric_features,
            config,
            scheme="DOI-grouped 5-fold" if not args.quick else "DOI-grouped 2-fold quick",
            fold=f"fold_{fold_number}",
            prediction_storage=prediction_storage,
            diagnostics=diagnostics,
        )
    if (split_manifest["grouped_fold"] < 1).any():
        raise AssertionError("Some records lack grouped-fold assignment")

    chronological_train = np.flatnonzero(
        metadata["publication_year"].le(config.temporal_cutoff_year).to_numpy()
    )
    chronological_test = np.flatnonzero(
        metadata["publication_year"].gt(config.temporal_cutoff_year).to_numpy()
    )
    if set(metadata.iloc[chronological_train]["doi_norm"]).intersection(
        set(metadata.iloc[chronological_test]["doi_norm"])
    ):
        raise AssertionError("DOI leakage detected in chronological validation")
    split_manifest["chronological_role"] = np.where(
        split_manifest["publication_year"].le(config.temporal_cutoff_year),
        "train_through_2018",
        "test_2019_onward",
    )
    print(
        f"[chronological] train={len(chronological_train):,}, "
        f"test={len(chronological_test):,}",
        flush=True,
    )
    importance, encoded_feature_count = fit_partition(
        features,
        targets,
        metadata,
        chronological_train,
        chronological_test,
        numeric_features,
        config,
        scheme="Chronological >2018",
        fold="holdout_2019_onward",
        prediction_storage=prediction_storage,
        diagnostics=diagnostics,
        keep_feature_importance=True,
    )

    predictions = pd.concat(prediction_storage, ignore_index=True)
    predictions["residual"] = predictions["y_pred"] - predictions["y_true"]
    summary, fold_metrics = summarize_predictions(predictions, config)
    model_rows = summary[summary["model"].ne("Dummy mean")]
    grouped_name = "DOI-grouped 5-fold" if not args.quick else "DOI-grouped 2-fold quick"
    grouped = model_rows.loc[
        model_rows["scheme"].eq(grouped_name),
        ["model", "target", "R2", "MAE", "RMSE"],
    ].rename(columns={"R2": "grouped_R2", "MAE": "grouped_MAE", "RMSE": "grouped_RMSE"})
    chronological = model_rows.loc[
        model_rows["scheme"].eq("Chronological >2018"),
        ["model", "target", "R2", "MAE", "RMSE"],
    ].rename(
        columns={
            "R2": "chronological_R2",
            "MAE": "chronological_MAE",
            "RMSE": "chronological_RMSE",
        }
    )
    generalization = generalization_gap_table(
        predictions, summary, config, grouped_scheme=grouped_name
    )
    inflation = publication_leakage_inflation_table(
        predictions,
        summary,
        config,
        row_random_scheme=row_random_scheme,
        grouped_scheme=grouped_name,
    )
    leakage_summary, leakage_by_size, leakage_strata = row_random_leakage_tables(
        split_manifest,
        predictions,
        row_random_scheme=row_random_scheme,
    )

    summary.to_csv(args.output_dir / "baseline_metrics_summary.csv", index=False)
    fold_metrics.to_csv(args.output_dir / "baseline_fold_metrics.csv", index=False)
    generalization.to_csv(args.output_dir / "baseline_generalization_gap.csv", index=False)
    inflation.to_csv(args.output_dir / "publication_leakage_inflation.csv", index=False)
    leakage_by_size.to_csv(args.output_dir / "row_random_leakage_by_DOI_size.csv", index=False)
    leakage_strata.to_csv(args.output_dir / "row_random_leakage_strata_metrics.csv", index=False)
    (args.output_dir / "row_random_leakage_summary.json").write_text(
        json.dumps(leakage_summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    predictions.to_csv(
        args.output_dir / "baseline_predictions.csv.gz",
        index=False,
        compression={"method": "gzip", "compresslevel": 6, "mtime": 0},
    )
    split_manifest.to_csv(args.output_dir / "split_manifest.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(args.output_dir / "fit_diagnostics.csv", index=False)
    if importance is not None:
        importance.to_csv(
            args.output_dir / "chronological_random_forest_feature_importance.csv",
            index=False,
        )
    write_markdown_tables(summary, args.output_dir)
    if not args.quick:
        plot_results(summary, predictions, generalization, inflation, args.output_dir)

    alias_counts = metadata.groupby("doi_norm")["doi_raw"].nunique()
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
        },
        "config": asdict(config),
        "cohort": {
            "records": int(len(metadata)),
            "raw_DOI_strings": int(metadata["doi_raw"].nunique()),
            "normalized_DOI_groups": int(metadata["doi_norm"].nunique()),
            "DOI_groups_with_case_or_format_aliases": int((alias_counts > 1).sum()),
            "publication_year_min": int(metadata["publication_year"].min()),
            "publication_year_max": int(metadata["publication_year"].max()),
        },
        "row_random_split": {
            "folds": config.row_random_folds,
            "records": int(len(metadata)),
            "DOI_groups": int(metadata["doi_norm"].nunique()),
            "DOI_groups_fragmented_across_folds": leakage_summary[
                "DOI_groups_fragmented_across_folds"
            ],
            "records_with_same_DOI_in_training": leakage_summary[
                "records_with_same_DOI_in_training"
            ],
            "fraction_records_with_same_DOI_in_training": leakage_summary[
                "fraction_records_with_same_DOI_in_training"
            ],
        },
        "chronological_split": {
            "train_records": int(len(chronological_train)),
            "train_DOI_groups": int(metadata.iloc[chronological_train]["doi_norm"].nunique()),
            "test_records": int(len(chronological_test)),
            "test_DOI_groups": int(metadata.iloc[chronological_test]["doi_norm"].nunique()),
            "test_years": sorted(metadata.iloc[chronological_test]["publication_year"].unique().tolist()),
        },
        "features": {
            "raw_numeric_features": numeric_features,
            "raw_categorical_features": CATEGORICAL_FEATURES,
            "token_feature": "role_tokens",
            "encoded_feature_count_chronological": encoded_feature_count,
            "publication_year_in_model": False,
            "DOI_in_model": False,
            "target_metrics_in_model": False,
        },
        "models": {
            "Dummy mean": "training-partition mean for each target",
            "Elastic Net": "independent target-standardized sparse linear models",
            "Random Forest": "joint four-target forest on standardized targets",
        },
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "completed", "runtime_seconds": manifest["runtime_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
