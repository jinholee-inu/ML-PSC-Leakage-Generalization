#!/usr/bin/env python3
"""Leakage-controlled hierarchical CatBoost attribution within PSC feature families.

This analysis refits the frozen full 1/n_DOI weighted multi-output CatBoost model
under the DOI-grouped and chronological partitions.  It decomposes the five
most important PCE parent families from the archived grouped-SHAP analysis into
scientifically interpretable material and process subgroups.

The implementation follows a nested (Owen-style) permutation path.  Parent
families enter in exactly the same Monte-Carlo order and against exactly the
same training-DOI background records as the archived feature-family analysis.
When a selected parent family enters, its child subgroups enter in a random
within-family order.  Child contributions therefore telescope exactly to the
archived parent-family contribution for every record and permutation.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time
import warnings
from collections import OrderedDict
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

# Joblib worker processes otherwise repeat a non-numerical sklearn configuration
# warning once per tree and prediction batch.  The warning does not affect model
# values; suppress it in both the parent process and spawned workers.
os.environ.setdefault(
    "PYTHONWARNINGS", "ignore::UserWarning:sklearn.utils.parallel"
)
warnings.filterwarnings(
    "ignore",
    message=r"`sklearn\.utils\.parallel\.delayed` should be used.*",
    category=UserWarning,
)


SCRIPT_DIR = Path(__file__).resolve().parent
BASELINE_CODE_DIR = SCRIPT_DIR / "baseline-code"
sys.path.insert(0, str(BASELINE_CODE_DIR))
sys.path.insert(0, str(SCRIPT_DIR))

from psc_baseline_validation import (  # noqa: E402
    RAW_REQUIRED,
    TARGETS,
    ModelConfig,
    build_features,
    make_preprocessor,
    normalize_doi,
    sha256,
)
from catboost_shap_ale import (  # noqa: E402
    CHRONO_SCHEME,
    CONDITION,
    FAMILY_ORDER,
    FULL_WEIGHTING,
    GROUPED_SCHEME,
    MODEL,
    archived_prediction_difference,
    as_model_matrix,
    choose_one_record_per_doi,
    family_columns,
    family_for_feature,
    model_predict,
    build_catboost_model,
    selected_catboost_config,
    training_weights,
    weighted_target_location_scale,
)


SUPPORTED_FAMILIES = {
    "Absorber composition",
    "Solvent/additive/quench",
    "Hole-transport layer",
    "Electron-transport layer",
    "Thermal processing",
}
ROLE_TOP_K = {
    "solv": 10,
    "add": 12,
    "quench": 10,
    "htl": 15,
    "etl": 18,
}
ROLE_LABEL = {
    "solv": "Solvent material",
    "add": "Additive",
    "quench": "Quench material",
    "htl": "HTL material",
    "etl": "ETL material",
}
FAMILY_COLOR = {
    "Absorber composition": "#3569A8",
    "Solvent/additive/quench": "#C96A3D",
    "Hole-transport layer": "#6B52A3",
    "Electron-transport layer": "#2A8C82",
    "Thermal processing": "#A77A21",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", required=True, type=Path)
    parser.add_argument("--cohort", required=True, type=Path)
    parser.add_argument("--baseline-results-dir", required=True, type=Path)
    parser.add_argument("--weighting-results-dir", required=True, type=Path)
    parser.add_argument("--parent-results-dir", required=True, type=Path)
    parser.add_argument("--catboost-model-selection", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def raw_numeric_name(encoded_name: str) -> str | None:
    if not encoded_name.startswith("numeric__"):
        return None
    raw = encoded_name[len("numeric__") :]
    return raw.replace("missingindicator_", "")


def pretty_token(token: str) -> str:
    special = {
        "ma": "MA",
        "fa": "FA",
        "cs": "Cs",
        "rb": "Rb",
        "dmf": "DMF",
        "dmso": "DMSO",
        "ipa": "IPA",
        "gbl": "GBL",
        "nmp": "NMP",
        "h2o": "H2O",
        "n2": "N2",
        "ar": "Ar",
        "cl": "Cl (database token)",
        "spiro_meotad": "Spiro-OMeTAD",
        "pedot_pss": "PEDOT:PSS",
        "nio_c": "NiO compact",
        "nio_np": "NiO nanoparticles",
        "nio_mp": "NiO mesoporous",
        "ptaa": "PTAA",
        "p3ht": "P3HT",
        "cuscn": "CuSCN",
        "moo3": "MoO3",
        "nimglio": "NiMgLiO",
        "p3ct_na": "P3CT-Na",
        "polytpd": "poly-TPD",
        "cui": "CuI",
        "graphene_oxide": "graphene oxide",
        "tio2_c": "TiO2 compact",
        "tio2_mp": "TiO2 mesoporous",
        "tio2_np": "TiO2 nanoparticles",
        "tio2_nw": "TiO2 nanowires",
        "pcbm_60": "PCBM-60",
        "pcbm_70": "PCBM-70",
        "bcp": "BCP",
        "c60": "C60",
        "sno2_c": "SnO2 compact",
        "sno2_np": "SnO2 nanoparticles",
        "zno_c": "ZnO compact",
        "zno_np": "ZnO nanoparticles",
        "zno_nw": "ZnO nanowires",
        "lif": "LiF",
        "zro2_mp": "ZrO2 mesoporous",
        "al2o3_mp": "Al2O3 mesoporous",
        "bphen": "BPhen",
        "bis_c60": "bis-C60",
        "pei": "PEI",
        "diethyl_ether": "diethyl ether",
        "ethyl_acetate": "ethyl acetate",
        "ethyl_ether": "ethyl ether",
        "2_butanol": "2-butanol",
        "trifluorotoluene": "trifluorotoluene",
        "acetonitrile": "acetonitrile",
        "dimethylacetamide": "dimethylacetamide",
        "5_avai": "5-AVAI",
        "pb_scn_2": "Pb(SCN)2",
        "snf2": "SnF2",
        "nh4cl": "NH4Cl",
        "peai": "PEAI",
    }
    if token in special:
        return special[token]
    return token.replace("_", " ")


def token_doi_prevalence(
    matrix: object,
    names: np.ndarray,
    train_dois: pd.Series,
) -> tuple[dict[str, list[str]], pd.DataFrame]:
    """Select a stable common-material vocabulary using historical training DOI only."""
    rows: list[dict[str, object]] = []
    selected: dict[str, list[str]] = {}
    doi_codes, _ = pd.factorize(train_dois.astype(str), sort=True)
    for role, top_k in ROLE_TOP_K.items():
        prefix = f"materials__{role}__"
        indices = np.flatnonzero(np.char.startswith(names.astype(str), prefix))
        role_rows: list[dict[str, object]] = []
        for column in indices:
            feature = str(names[column])
            token = feature[len(prefix) :]
            if sparse.issparse(matrix):
                present_rows = matrix[:, column].nonzero()[0]
            else:
                present_rows = np.flatnonzero(np.asarray(matrix)[:, column] != 0)
            record_count = int(len(present_rows))
            doi_count = int(len(np.unique(doi_codes[present_rows])))
            role_rows.append(
                {
                    "role": role,
                    "role_label": ROLE_LABEL[role],
                    "token": token,
                    "display_label": pretty_token(token),
                    "encoded_feature": feature,
                    "historical_train_record_prevalence": record_count,
                    "historical_train_DOI_prevalence": doi_count,
                }
            )
        role_rows.sort(
            key=lambda row: (
                -int(row["historical_train_DOI_prevalence"]),
                -int(row["historical_train_record_prevalence"]),
                str(row["token"]),
            )
        )
        chosen = [str(row["token"]) for row in role_rows[:top_k]]
        selected[role] = chosen
        for rank, row in enumerate(role_rows, start=1):
            row["DOI_prevalence_rank"] = rank
            row["selected_as_individual_subgroup"] = str(row["token"]) in chosen
            rows.append(row)
    return selected, pd.DataFrame(rows)


def absorber_subgroup(name: str) -> tuple[str, str]:
    if name.startswith("categorical__absorber_short_form"):
        return "Exact absorber formula category", "category"
    if name.startswith("materials__"):
        token = name[len("materials__") :]
        role, material = token.split("__", 1)
        site_map = {
            "a": {
                "ma": "MA",
                "fa": "FA",
                "cs": "Cs",
                "rb": "Rb",
                "k": "K",
                "gu": "GUA",
                "ea": "EA",
                "dma": "DMA",
                "pea": "PEA",
                "ba": "BA",
            },
            "b": {"pb": "Pb", "sn": "Sn", "ge": "Ge", "bi": "Bi", "sb": "Sb", "cu": "Cu"},
            "x": {"i": "I", "br": "Br", "cl": "Cl", "f": "F"},
        }
        if role not in site_map:
            raise KeyError(name)
        if material in site_map[role]:
            return f"{role.upper()}-site: {site_map[role][material]}", "composition"
        return f"{role.upper()}-site: Other/rare chemistry", "composition"
    raw = raw_numeric_name(name)
    if raw is None:
        raise KeyError(name)
    flag_map = {
        "is_inorganic": ("All-inorganic flag", "class flag"),
        "is_leadfree": ("Lead-free flag", "class flag"),
        "dimension_0D": ("Dimensionality: 0D", "class flag"),
        "dimension_2D": ("Dimensionality: 2D", "class flag"),
        "dimension_2D3D": ("Dimensionality: 2D/3D", "class flag"),
        "dimension_3D": ("Dimensionality: 3D", "class flag"),
        "dimension_3D_2Dcap": ("Dimensionality: 3D/2D cap", "class flag"),
    }
    if raw in flag_map:
        return flag_map[raw]
    if raw.startswith("comp_"):
        _, site, component, _fraction = raw.split("_", 3)
        if component == "parsed":
            return "Composition parsing coverage", "data completeness"
        if component == "other":
            return f"{site}-site: Other/rare chemistry", "composition"
        component_map = {"GUA": "GUA"}
        return f"{site}-site: {component_map.get(component, component)}", "composition"
    raise KeyError(name)


def subgroup_for_feature(
    name: str,
    family: str,
    common_tokens: dict[str, list[str]],
) -> tuple[str, str]:
    if family == "Absorber composition":
        return absorber_subgroup(name)

    if family in {"Hole-transport layer", "Electron-transport layer"}:
        role = "htl" if family.startswith("Hole") else "etl"
        short = "HTL" if role == "htl" else "ETL"
        if name.startswith(f"categorical__{role}_stack"):
            return f"{short} stack category", "category"
        raw = raw_numeric_name(name)
        if raw == f"{role}_layer_count":
            return f"{short} layer count", "count"
        prefix = f"materials__{role}__"
        if name.startswith(prefix):
            token = name[len(prefix) :]
            if token in common_tokens[role]:
                return f"{short} material: {pretty_token(token)}", "material"
            return f"Other {short} material tokens", "other material"
        raise KeyError(name)

    if family == "Solvent/additive/quench":
        if name.startswith("categorical__solvent_system"):
            return "Solvent-system category", "category"
        if name.startswith("categorical__quenching_medium"):
            return "Quenching-medium category", "category"
        raw = raw_numeric_name(name)
        count_map = {
            "solvent_count": "Solvent count",
            "additive_count": "Additive count",
            "quench_count": "Quench-medium count",
        }
        if raw in count_map:
            return count_map[raw], "count"
        if name.startswith("materials__"):
            token = name[len("materials__") :]
            role, material = token.split("__", 1)
            if role not in {"solv", "add", "quench"}:
                raise KeyError(name)
            if material in common_tokens[role]:
                return f"{ROLE_LABEL[role]}: {pretty_token(material)}", "material"
            return f"Other {ROLE_LABEL[role].lower()} tokens", "other material"
        raise KeyError(name)

    if family == "Thermal processing":
        if name.startswith("categorical__annealing_atmosphere"):
            return "Annealing atmosphere", "process"
        raw = raw_numeric_name(name)
        if raw is None:
            raise KeyError(name)
        if raw.startswith("anneal_temp_"):
            return "Annealing temperature", "process"
        if raw.startswith("anneal_time_"):
            return "Annealing time", "process"
        if raw.startswith("substrate_temp_"):
            return "Substrate temperature", "process"
        raise KeyError(name)
    raise KeyError(f"Unsupported selected family: {family}")


def make_subgroups(
    feature_names: np.ndarray,
    parent_columns: dict[str, np.ndarray],
    selected_families: list[str],
    common_tokens: dict[str, list[str]],
) -> tuple[
    dict[str, OrderedDict[str, np.ndarray]],
    dict[tuple[str, str], str],
]:
    output: dict[str, OrderedDict[str, np.ndarray]] = {}
    subgroup_class: dict[tuple[str, str], str] = {}
    for family in selected_families:
        temporary: OrderedDict[str, list[int]] = OrderedDict()
        for column in parent_columns[family]:
            name = str(feature_names[column])
            subgroup, class_name = subgroup_for_feature(name, family, common_tokens)
            temporary.setdefault(subgroup, []).append(int(column))
            subgroup_class[(family, subgroup)] = class_name
        output[family] = OrderedDict(
            (subgroup, np.asarray(columns, dtype=int))
            for subgroup, columns in temporary.items()
        )
        assigned = np.concatenate(list(output[family].values()))
        expected = np.sort(parent_columns[family])
        if not np.array_equal(np.sort(assigned), expected):
            raise AssertionError(f"Child subgroups do not partition {family}")
        if len(np.unique(assigned)) != len(assigned):
            raise AssertionError(f"Duplicated encoded columns within {family}")
    return output, subgroup_class


def hierarchical_monte_carlo_shap(
    forest: CatBoostRegressor,
    explain_matrix: object,
    background_matrix: object,
    parent_columns: dict[str, np.ndarray],
    subgroups: dict[str, OrderedDict[str, np.ndarray]],
    y_mean: np.ndarray,
    y_std: np.ndarray,
    permutations: int,
    seed: int,
) -> dict[str, object]:
    """Nested Monte-Carlo Shapley/Owen attribution with exact parent telescoping."""
    outer_rng = np.random.default_rng(seed)
    inner_rng = np.random.default_rng(seed + 1_000_003)
    x = explain_matrix.toarray() if sparse.issparse(explain_matrix) else np.asarray(explain_matrix)
    background = (
        background_matrix.toarray()
        if sparse.issparse(background_matrix)
        else np.asarray(background_matrix)
    )
    x = np.asarray(x, dtype=np.float32)
    background = np.asarray(background, dtype=np.float32)
    n_rows, n_features = x.shape
    n_targets = len(TARGETS)

    flat_keys: list[tuple[str, str]] = []
    for family in FAMILY_ORDER:
        if family in subgroups:
            flat_keys.extend((family, subgroup) for subgroup in subgroups[family])
    key_index = {key: index for index, key in enumerate(flat_keys)}
    selected_count = len(flat_keys)
    transitions = sum(
        len(subgroups[family]) if family in subgroups else 1 for family in FAMILY_ORDER
    )
    child_sum = np.zeros((n_rows, selected_count, n_targets), dtype=np.float64)
    child_square_sum = np.zeros_like(child_sum)
    parent_sum = np.zeros((n_rows, len(FAMILY_ORDER), n_targets), dtype=np.float64)
    base_sum = np.zeros((n_rows, n_targets), dtype=np.float64)
    full_prediction = model_predict(forest, x, y_mean, y_std)
    terminal_difference = 0.0

    for replicate in range(permutations):
        background_index = outer_rng.integers(0, len(background), size=n_rows)
        outer_orders = np.vstack(
            [outer_rng.permutation(len(FAMILY_ORDER)) for _ in range(n_rows)]
        )
        states = np.empty((n_rows, transitions + 1, n_features), dtype=np.float32)
        child_labels = np.full((n_rows, transitions), -1, dtype=np.int16)
        parent_labels = np.empty((n_rows, transitions), dtype=np.int8)
        states[:, 0, :] = background[background_index]

        for row in range(n_rows):
            current = states[row, 0].copy()
            transition = 0
            for parent_index in outer_orders[row]:
                family = FAMILY_ORDER[int(parent_index)]
                if family in subgroups:
                    subgroup_names = list(subgroups[family])
                    child_order = inner_rng.permutation(len(subgroup_names))
                    for child_position in child_order:
                        subgroup = subgroup_names[int(child_position)]
                        columns = subgroups[family][subgroup]
                        current[columns] = x[row, columns]
                        states[row, transition + 1] = current
                        child_labels[row, transition] = key_index[(family, subgroup)]
                        parent_labels[row, transition] = int(parent_index)
                        transition += 1
                else:
                    columns = parent_columns[family]
                    current[columns] = x[row, columns]
                    states[row, transition + 1] = current
                    parent_labels[row, transition] = int(parent_index)
                    transition += 1
            if transition != transitions:
                raise AssertionError("Unexpected hierarchical path length")

        predictions = model_predict(
            forest,
            states.reshape(n_rows * (transitions + 1), n_features),
            y_mean,
            y_std,
        ).reshape(n_rows, transitions + 1, n_targets)
        deltas = np.diff(predictions, axis=1)
        base_sum += predictions[:, 0, :]
        terminal_difference = max(
            terminal_difference,
            float(np.max(np.abs(predictions[:, -1, :] - full_prediction))),
        )
        for parent_index in range(len(FAMILY_ORDER)):
            positions = np.argwhere(parent_labels == parent_index)
            values = deltas[positions[:, 0], positions[:, 1], :]
            np.add.at(parent_sum[:, parent_index, :], positions[:, 0], values)
        for child_index in range(selected_count):
            positions = np.argwhere(child_labels == child_index)
            values = deltas[positions[:, 0], positions[:, 1], :]
            # Every selected child enters exactly once along every row path.
            if len(positions) != n_rows or len(np.unique(positions[:, 0])) != n_rows:
                raise AssertionError("A child subgroup did not enter exactly once per row")
            ordered = np.empty((n_rows, n_targets), dtype=np.float64)
            ordered[positions[:, 0]] = values
            child_sum[:, child_index, :] += ordered
            child_square_sum[:, child_index, :] += ordered**2
        if replicate == 0 or (replicate + 1) % 8 == 0:
            print(
                f"    hierarchical permutations {replicate + 1}/{permutations}",
                flush=True,
            )

    child_values = child_sum / float(permutations)
    parent_values = parent_sum / float(permutations)
    base_values = base_sum / float(permutations)
    if permutations > 1:
        variance = np.maximum(
            (child_square_sum - permutations * child_values**2) / (permutations - 1),
            0.0,
        )
        child_mcse = np.sqrt(variance / permutations)
    else:
        child_mcse = np.full_like(child_values, np.nan)
    additivity = full_prediction - (base_values + parent_values.sum(axis=1))
    child_parent_difference = 0.0
    for family in subgroups:
        child_indices = [key_index[(family, subgroup)] for subgroup in subgroups[family]]
        parent_index = FAMILY_ORDER.index(family)
        difference = child_values[:, child_indices, :].sum(axis=1) - parent_values[:, parent_index, :]
        child_parent_difference = max(
            child_parent_difference,
            float(np.max(np.abs(difference))),
        )
    return {
        "keys": flat_keys,
        "values": child_values,
        "mcse": child_mcse,
        "parent_values": parent_values,
        "base_values": base_values,
        "predictions": full_prediction,
        "transitions": transitions,
        "max_additivity_residual": float(np.max(np.abs(additivity))),
        "max_child_parent_difference": child_parent_difference,
        "max_terminal_prediction_difference": terminal_difference,
    }


def bootstrap_importance(
    local: pd.DataFrame,
    replicates: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fold_rows: list[dict[str, object]] = []
    for keys, group in local.groupby(
        ["scheme", "fold", "target", "parent_family", "subgroup", "subgroup_class"],
        sort=False,
    ):
        scheme, fold, target, family, subgroup, subgroup_class = keys
        parent = group.groupby("doi_norm", sort=False)["parent_shap_value"].first()
        fold_rows.append(
            {
                "scheme": scheme,
                "fold": fold,
                "target": target,
                "parent_family": family,
                "subgroup": subgroup,
                "subgroup_class": subgroup_class,
                "explained_DOI": int(group["doi_norm"].nunique()),
                "mean_abs_hierarchical_SHAP": float(group["abs_shap_value"].mean()),
                "mean_signed_hierarchical_SHAP": float(group["shap_value"].mean()),
                "positive_fraction": float((group["shap_value"] > 0).mean()),
                "parent_mean_abs_SHAP": float(parent.abs().mean()),
            }
        )
    fold_frame = pd.DataFrame(fold_rows)

    summary_rows: list[dict[str, object]] = []
    rng = np.random.default_rng(seed)
    for (scheme, target, family), group in local.groupby(
        ["scheme", "target", "parent_family"], sort=False
    ):
        value_pivot = group.pivot(index="doi_norm", columns="subgroup", values="shap_value")
        mcse_pivot = group.pivot(index="doi_norm", columns="subgroup", values="shap_mcse")
        meta = (
            group[["subgroup", "subgroup_class", "encoded_column_count"]]
            .drop_duplicates("subgroup")
            .set_index("subgroup")
        )
        values = value_pivot.to_numpy(dtype=float)
        point = np.mean(np.abs(values), axis=0)
        boot = np.empty((replicates, values.shape[1]), dtype=float)
        for replicate in range(replicates):
            sample = rng.integers(0, len(values), size=len(values))
            boot[replicate] = np.mean(np.abs(values[sample]), axis=0)
        lower, upper = np.quantile(boot, [0.025, 0.975], axis=0)
        parent = group.groupby("doi_norm", sort=False)["parent_shap_value"].first()
        parent_mean_abs = float(parent.abs().mean())
        attribution_mass = float(point.sum())
        for index, subgroup in enumerate(value_pivot.columns):
            series = value_pivot[subgroup]
            summary_rows.append(
                {
                    "scheme": scheme,
                    "target": target,
                    "parent_family": family,
                    "subgroup": subgroup,
                    "subgroup_class": meta.loc[subgroup, "subgroup_class"],
                    "encoded_column_count": int(meta.loc[subgroup, "encoded_column_count"]),
                    "explained_DOI": int(len(value_pivot)),
                    "mean_abs_hierarchical_SHAP": float(point[index]),
                    "mean_abs_CI_low": float(lower[index]),
                    "mean_abs_CI_high": float(upper[index]),
                    "mean_signed_hierarchical_SHAP": float(series.mean()),
                    "positive_fraction": float((series > 0).mean()),
                    "median_local_MCSE": float(mcse_pivot[subgroup].median()),
                    "parent_mean_abs_SHAP": parent_mean_abs,
                    "attribution_mass_percent": float(100.0 * point[index] / attribution_mass),
                    "mean_abs_to_parent_ratio": float(point[index] / parent_mean_abs)
                    if parent_mean_abs > 0
                    else math.nan,
                }
            )
    summary = pd.DataFrame(summary_rows)
    summary["rank_within_family"] = summary.groupby(
        ["scheme", "target", "parent_family"]
    )["mean_abs_hierarchical_SHAP"].rank(method="min", ascending=False)
    return fold_frame, summary


def component_class_summary(summary: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        summary.groupby(
            ["scheme", "target", "parent_family", "subgroup_class"],
            as_index=False,
        )["mean_abs_hierarchical_SHAP"]
        .sum()
    )
    grouped["class_attribution_mass_percent"] = 100.0 * grouped[
        "mean_abs_hierarchical_SHAP"
    ] / grouped.groupby(["scheme", "target", "parent_family"])[
        "mean_abs_hierarchical_SHAP"
    ].transform("sum")
    return grouped


def rank_stability(
    fold_frame: pd.DataFrame,
    summary: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for target in TARGETS:
        for family in summary["parent_family"].unique():
            grouped = summary.loc[
                summary["scheme"].eq(GROUPED_SCHEME)
                & summary["target"].eq(target)
                & summary["parent_family"].eq(family)
            ].set_index("subgroup")
            chrono = summary.loc[
                summary["scheme"].eq(CHRONO_SCHEME)
                & summary["target"].eq(target)
                & summary["parent_family"].eq(family)
            ].set_index("subgroup")
            shared = sorted(set(grouped.index) & set(chrono.index))
            rho = spearmanr(
                grouped.loc[shared, "mean_abs_hierarchical_SHAP"],
                chrono.loc[shared, "mean_abs_hierarchical_SHAP"],
            ).statistic
            top_n = min(5, len(shared))
            top_grouped = set(
                grouped.loc[shared].nlargest(top_n, "mean_abs_hierarchical_SHAP").index
            )
            top_chrono = set(
                chrono.loc[shared].nlargest(top_n, "mean_abs_hierarchical_SHAP").index
            )
            fold_subset = fold_frame.loc[
                fold_frame["scheme"].eq(GROUPED_SCHEME)
                & fold_frame["target"].eq(target)
                & fold_frame["parent_family"].eq(family)
            ]
            fold_pivot = fold_subset.pivot(
                index="subgroup", columns="fold", values="mean_abs_hierarchical_SHAP"
            )
            pairwise: list[float] = []
            columns = list(fold_pivot.columns)
            for left in range(len(columns)):
                for right in range(left + 1, len(columns)):
                    pairwise.append(
                        float(
                            spearmanr(
                                fold_pivot[columns[left]], fold_pivot[columns[right]]
                            ).statistic
                        )
                    )
            rows.append(
                {
                    "target": target,
                    "parent_family": family,
                    "shared_subgroups": len(shared),
                    "grouped_vs_chronological_spearman_rho": float(rho),
                    "top_n_compared": int(top_n),
                    "top5_overlap": int(len(top_grouped & top_chrono)),
                    "grouped_fold_pairwise_spearman_median": float(np.nanmedian(pairwise)),
                    "grouped_fold_pairwise_spearman_min": float(np.nanmin(pairwise)),
                }
            )
    return pd.DataFrame(rows)


def short_label(label: str) -> str:
    replacements = {
        "Exact absorber formula category": "Exact formula category",
        "Composition parsing coverage": "Parsing coverage",
        "Solvent-system category": "Solvent-system category",
        "Quenching-medium category": "Quenching category",
        "Other solvent material tokens": "Other solvent tokens",
        "Other additive tokens": "Other additive tokens",
        "Other quench material tokens": "Other quench tokens",
        "Other HTL material tokens": "Other HTL tokens",
        "Other ETL material tokens": "Other ETL tokens",
    }
    return replacements.get(label, label)


def plot_bar_with_ci(ax, frame: pd.DataFrame, color: str, title: str, top_n: int) -> None:
    show = frame.nlargest(top_n, "mean_abs_hierarchical_SHAP").sort_values(
        "mean_abs_hierarchical_SHAP"
    )
    y = np.arange(len(show))
    values = show["mean_abs_hierarchical_SHAP"].to_numpy()
    lower = values - show["mean_abs_CI_low"].to_numpy()
    upper = show["mean_abs_CI_high"].to_numpy() - values
    ax.barh(y, values, color=color, alpha=0.88, edgecolor="white", linewidth=0.6)
    ax.errorbar(values, y, xerr=np.vstack([lower, upper]), fmt="none", ecolor="#333333", capsize=2, lw=0.9)
    ax.set_yticks(y, [short_label(value) for value in show["subgroup"]])
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_xlabel("Mean |hierarchical SHAP| (PCE %-point)")
    ax.grid(axis="x", color="#D9D9D9", linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    sns.despine(ax=ax)


def make_figure(summary: pd.DataFrame, output_dir: Path) -> None:
    sns.set_theme(style="white", context="paper")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 9.2,
            "axes.labelsize": 8.2,
            "xtick.labelsize": 7.3,
            "ytick.labelsize": 7.3,
            "legend.fontsize": 7.3,
        }
    )
    chrono = summary.loc[
        summary["scheme"].eq(CHRONO_SCHEME) & summary["target"].eq("PCE")
    ].copy()
    grouped = summary.loc[
        summary["scheme"].eq(GROUPED_SCHEME) & summary["target"].eq("PCE")
    ].copy()
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.4), constrained_layout=False)
    plt.subplots_adjust(left=0.14, right=0.98, top=0.93, bottom=0.09, wspace=0.48, hspace=0.46)

    plot_bar_with_ci(
        axes[0, 0],
        chrono.loc[chrono["parent_family"].eq("Absorber composition")],
        FAMILY_COLOR["Absorber composition"],
        "(a) Absorber composition",
        10,
    )
    plot_bar_with_ci(
        axes[0, 1],
        chrono.loc[chrono["parent_family"].eq("Solvent/additive/quench")],
        FAMILY_COLOR["Solvent/additive/quench"],
        "(b) Formulation and quenching",
        10,
    )

    transport = chrono.loc[
        chrono["parent_family"].isin(
            ["Hole-transport layer", "Electron-transport layer"]
        )
        & chrono["subgroup_class"].isin(["material", "other material"])
    ].copy()
    transport = pd.concat(
        [
            frame.nlargest(6, "mean_abs_hierarchical_SHAP")
            for _family, frame in transport.groupby("parent_family", sort=False)
        ],
        ignore_index=True,
    )
    transport["display"] = transport.apply(
        lambda row: ("HTL · " if row["parent_family"].startswith("Hole") else "ETL · ")
        + short_label(row["subgroup"].split(": ", 1)[-1]),
        axis=1,
    )
    transport = transport.sort_values("mean_abs_hierarchical_SHAP")
    colors = [FAMILY_COLOR[value] for value in transport["parent_family"]]
    y = np.arange(len(transport))
    values = transport["mean_abs_hierarchical_SHAP"].to_numpy()
    lower = values - transport["mean_abs_CI_low"].to_numpy()
    upper = transport["mean_abs_CI_high"].to_numpy() - values
    axes[1, 0].barh(y, values, color=colors, alpha=0.88, edgecolor="white", linewidth=0.6)
    axes[1, 0].errorbar(values, y, xerr=np.vstack([lower, upper]), fmt="none", ecolor="#333333", capsize=2, lw=0.9)
    axes[1, 0].set_yticks(y, transport["display"])
    axes[1, 0].set_title("(c) Transport-layer materials", loc="left", fontweight="bold")
    axes[1, 0].set_xlabel("Mean |hierarchical SHAP| (PCE %-point)")
    axes[1, 0].grid(axis="x", color="#D9D9D9", linewidth=0.6, alpha=0.7)
    axes[1, 0].set_axisbelow(True)
    sns.despine(ax=axes[1, 0])

    thermal_c = chrono.loc[chrono["parent_family"].eq("Thermal processing")].set_index("subgroup")
    thermal_g = grouped.loc[grouped["parent_family"].eq("Thermal processing")].set_index("subgroup")
    labels = thermal_c.sort_values("mean_abs_hierarchical_SHAP").index.tolist()
    y = np.arange(len(labels))
    height = 0.34
    for offset, frame, label, color in [
        (-height / 2, thermal_g, "DOI-grouped", "#8996A8"),
        (height / 2, thermal_c, "Chronological", FAMILY_COLOR["Thermal processing"]),
    ]:
        values = frame.loc[labels, "mean_abs_hierarchical_SHAP"].to_numpy()
        lower = values - frame.loc[labels, "mean_abs_CI_low"].to_numpy()
        upper = frame.loc[labels, "mean_abs_CI_high"].to_numpy() - values
        axes[1, 1].barh(y + offset, values, height=height, color=color, label=label, alpha=0.9)
        axes[1, 1].errorbar(values, y + offset, xerr=np.vstack([lower, upper]), fmt="none", ecolor="#333333", capsize=2, lw=0.9)
    axes[1, 1].set_yticks(y, [short_label(value) for value in labels])
    axes[1, 1].set_title("(d) Thermal-process stability", loc="left", fontweight="bold")
    axes[1, 1].set_xlabel("Mean |hierarchical SHAP| (PCE %-point)")
    axes[1, 1].legend(frameon=False, loc="lower right")
    axes[1, 1].grid(axis="x", color="#D9D9D9", linewidth=0.6, alpha=0.7)
    axes[1, 1].set_axisbelow(True)
    sns.despine(ax=axes[1, 1])

    fig.suptitle(
        "Leakage-controlled within-family attribution for PCE",
        x=0.14,
        ha="left",
        fontsize=12,
        fontweight="bold",
    )
    fig.text(
        0.14,
        0.025,
        "Held-out records are DOI-balanced (one record per DOI). Error bars: DOI-cluster bootstrap 95% CI (1,000 replicates).",
        fontsize=7.2,
        color="#444444",
    )
    for suffix in ["png", "pdf", "svg"]:
        kwargs = {"dpi": 600} if suffix == "png" else {}
        fig.savefig(
            output_dir / f"Figure7_CatBoost_within_family_attribution.{suffix}",
            bbox_inches="tight",
            facecolor="white",
            **kwargs,
        )
    plt.close(fig)


def write_markdown_report(
    output_dir: Path,
    summary: pd.DataFrame,
    stability: pd.DataFrame,
    selected_families: list[str],
    verification: dict[str, object],
) -> None:
    archived_parent_text = (
        f"{verification['max_archived_parent_SHAP_difference']:.3e}"
        if verification["max_archived_parent_SHAP_difference"] is not None
        else "not evaluated in quick mode"
    )
    archived_prediction_text = (
        f"{verification['max_archived_prediction_difference']:.3e}"
        if verification["max_archived_prediction_difference"] is not None
        else "not evaluated in quick mode"
    )
    chrono = summary.loc[
        summary["scheme"].eq(CHRONO_SCHEME) & summary["target"].eq("PCE")
    ]
    lines = [
        "# Leakage-controlled within-family attribution report",
        "",
        "## Scope and design",
        "",
        "This follow-up analysis decomposes five prespecified major parent feature families from the frozen chronological PCE explanation into individual material and processing subgroups. The selected full `1/n_DOI` CatBoost model, frozen cohort, feature pipeline, DOI-grouped folds, and 2019–2021 holdout were unchanged.",
        "",
        "Nested permutation paths preserve the original parent-family order. Within a selected parent family, child subgroups enter in a random order. Therefore the signed child contributions telescope exactly to the archived parent contribution; absolute child attribution mass is reported only as a within-family allocation and is not interpreted as a causal effect.",
        "",
        "Selected parent families: " + "; ".join(selected_families) + ".",
        "",
        "## Chronological PCE results",
        "",
    ]
    for family in selected_families:
        frame = chrono.loc[chrono["parent_family"].eq(family)].nlargest(
            5, "mean_abs_hierarchical_SHAP"
        )
        lines.append(f"### {family}")
        lines.append("")
        for row in frame.itertuples():
            lines.append(
                f"- {row.subgroup}: mean |hierarchical SHAP| = {row.mean_abs_hierarchical_SHAP:.3f} PCE %-point "
                f"(95% CI {row.mean_abs_CI_low:.3f}–{row.mean_abs_CI_high:.3f}); "
                f"{row.attribution_mass_percent:.1f}% of within-family absolute attribution mass."
            )
        lines.append("")
    lines.extend(
        [
            "## Stability and interpretation",
            "",
        ]
    )
    for row in stability.loc[stability["target"].eq("PCE")].itertuples():
        lines.append(
            f"- {row.parent_family}: grouped-versus-chronological Spearman rho = "
            f"{row.grouped_vs_chronological_spearman_rho:.3f}; top-{row.top_n_compared} overlap = "
            f"{row.top5_overlap}/{row.top_n_compared}; "
            f"median pairwise grouped-fold rho = {row.grouped_fold_pairwise_spearman_median:.3f}."
        )
    lines.extend(
        [
            "",
            "The estimates are model-based associations within the support of the historical database. Correlated and redundant encodings (for example formula categories and site fractions) can redistribute attribution within a parent family. Material tokens identify database encodings, not controlled interventions; rare tokens are pooled. These results should guide hypothesis generation and subgroup validation rather than causal claims.",
            "",
            "## Verification",
            "",
            f"- Status: {verification['status']}",
            f"- Explained DOI groups: {verification['explained_DOI_groups']}",
            f"- Held-out DOI-partition instances: {verification['heldout_DOI_partition_instances']}",
            f"- Hierarchical local rows: {verification['local_rows']}",
            f"- Duplicate local keys: {verification['duplicate_local_keys']}",
            f"- Maximum child-to-parent telescoping difference: {verification['max_child_parent_difference']:.3e}",
            f"- Maximum difference from archived parent SHAP: {archived_parent_text}",
            f"- Maximum archived prediction difference: {archived_prediction_text}",
            f"- DOI boundary overlap: grouped {verification['grouped_boundary_DOI_overlap']}, chronological {verification['chronological_boundary_DOI_overlap']}",
            f"- Bootstrap replicates: {verification['bootstrap_replicates']}",
            f"- Nested permutations per record: {verification['hierarchical_permutations']}",
            "",
        ]
    )
    (output_dir / "PSC_CatBoost_within_family_attribution_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    config = ModelConfig()
    explain_per_grouped_fold = 150
    explain_chronological = 300
    background_dois = 400
    permutations = 48
    bootstrap_replicates = 1000
    if args.quick:
        config = ModelConfig(
            grouped_folds=2,
            bootstrap_replicates=30,
            token_min_df=20,
            token_max_features=800,
            rf_estimators=20,
        )
        explain_per_grouped_fold = 20
        explain_chronological = 30
        background_dois = 50
        permutations = 3
        bootstrap_replicates = 30

    baseline_manifest = json.loads(
        (args.baseline_results_dir / "run_manifest.json").read_text(encoding="utf-8")
    )
    if not args.quick:
        if sha256(args.raw) != baseline_manifest["inputs"]["raw_sha256"]:
            raise AssertionError("Raw snapshot hash differs from frozen baseline")
        if sha256(args.cohort) != baseline_manifest["inputs"]["cohort_sha256"]:
            raise AssertionError("Cohort hash differs from frozen baseline")

    parent_summary = pd.read_csv(args.parent_results_dir / "shap_family_importance_summary.csv")
    chronological_parent = parent_summary.loc[
        parent_summary["scheme"].eq(CHRONO_SCHEME)
        & parent_summary["target"].eq("PCE")
        & parent_summary["family"].isin(SUPPORTED_FAMILIES)
    ].sort_values("rank")
    selected_families = chronological_parent["family"].tolist()
    if set(selected_families) != SUPPORTED_FAMILIES:
        raise AssertionError("CatBoost parent results do not contain every prespecified supported family")

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
    archived_parent = pd.read_csv(args.parent_results_dir / "shap_local_values.csv.gz")
    selection = pd.read_csv(args.catboost_model_selection)

    chrono_train = np.flatnonzero(
        split_manifest["chronological_role"].eq("train_through_2018").to_numpy()
    )
    historical_processor = make_preprocessor(config, numeric_features)
    historical_matrix = as_model_matrix(
        historical_processor.fit_transform(features.iloc[chrono_train])
    )
    historical_names = np.asarray(historical_processor.get_feature_names_out(), dtype=str)
    common_tokens, vocabulary = token_doi_prevalence(
        historical_matrix,
        historical_names,
        metadata.iloc[chrono_train]["doi_norm"].reset_index(drop=True),
    )
    vocabulary.to_csv(args.output_dir / "historical_training_material_vocabulary.csv", index=False)
    del historical_matrix, historical_processor

    partitions: list[tuple[str, str, np.ndarray, np.ndarray, int]] = []
    grouped_folds = sorted(split_manifest["grouped_fold"].unique())
    if args.quick:
        grouped_folds = grouped_folds[:2]
    for fold_number in grouped_folds:
        test = np.flatnonzero(split_manifest["grouped_fold"].eq(fold_number).to_numpy())
        train = np.flatnonzero(split_manifest["grouped_fold"].ne(fold_number).to_numpy())
        partitions.append(
            (
                GROUPED_SCHEME,
                f"fold_{int(fold_number)}",
                train,
                test,
                config.seed + int(fold_number),
            )
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
    parent_rows: list[dict[str, object]] = []
    diagnostics: list[dict[str, object]] = []
    max_archived_prediction_difference = 0.0
    max_child_parent_difference = 0.0
    max_additivity_residual = 0.0
    max_terminal_difference = 0.0
    max_archived_parent_difference = 0.0
    grouped_overlap = 0
    chrono_overlap = 0

    for partition_number, (scheme, fold, train_index, test_index, random_state) in enumerate(partitions):
        partition_started = time.perf_counter()
        explain_max = explain_chronological if scheme == CHRONO_SCHEME else explain_per_grouped_fold
        explain_index = choose_one_record_per_doi(
            test_index, metadata, explain_max, config.seed + 41000 + partition_number
        )
        background_index = choose_one_record_per_doi(
            train_index, metadata, background_dois, config.seed + 42000 + partition_number
        )
        train_doi = set(metadata.iloc[train_index]["doi_norm"])
        test_doi = set(metadata.iloc[test_index]["doi_norm"])
        overlap = len(train_doi & test_doi)
        if scheme == GROUPED_SCHEME:
            grouped_overlap = max(grouped_overlap, overlap)
        else:
            chrono_overlap = max(chrono_overlap, overlap)
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
        candidate, iterations = selected_catboost_config(selection, scheme, fold)
        if args.quick:
            iterations = min(iterations, 60)
        forest = build_catboost_model(candidate, iterations, random_state)
        forest.fit(Pool(train_matrix, label=(y_train - y_mean) / y_std, weight=weights))
        full_prediction = model_predict(forest, test_matrix, y_mean, y_std)
        if not args.quick:
            archived_difference = archived_prediction_difference(
                archived, scheme, fold, metadata, test_index, full_prediction
            )
            max_archived_prediction_difference = max(
                max_archived_prediction_difference, archived_difference
            )

        encoded_names = np.asarray(processor.get_feature_names_out(), dtype=str)
        parents = family_columns(encoded_names)
        subgroups, subgroup_classes = make_subgroups(
            encoded_names, parents, selected_families, common_tokens
        )
        result = hierarchical_monte_carlo_shap(
            forest,
            explain_matrix,
            background_matrix,
            parents,
            subgroups,
            y_mean,
            y_std,
            permutations,
            config.seed + 43000 + partition_number,
        )
        max_child_parent_difference = max(
            max_child_parent_difference, float(result["max_child_parent_difference"])
        )
        max_additivity_residual = max(
            max_additivity_residual, float(result["max_additivity_residual"])
        )
        max_terminal_difference = max(
            max_terminal_difference, float(result["max_terminal_prediction_difference"])
        )
        explain_targets = targets.iloc[explain_index].to_numpy(dtype=float)
        keys = list(result["keys"])
        values = np.asarray(result["values"])
        mcse = np.asarray(result["mcse"])
        parent_values = np.asarray(result["parent_values"])
        predictions = np.asarray(result["predictions"])
        base_values = np.asarray(result["base_values"])

        for row_position, original_index in enumerate(explain_index):
            for target_index, target in enumerate(TARGETS):
                for parent_index, family in enumerate(FAMILY_ORDER):
                    parent_rows.append(
                        {
                            "Ref_ID": metadata.iloc[original_index]["Ref_ID"],
                            "scheme": scheme,
                            "fold": fold,
                            "target": target,
                            "family": family,
                            "parent_shap_value": float(
                                parent_values[row_position, parent_index, target_index]
                            ),
                        }
                    )
                for child_index, (family, subgroup) in enumerate(keys):
                    value = float(values[row_position, child_index, target_index])
                    local_rows.append(
                        {
                            "Ref_ID": metadata.iloc[original_index]["Ref_ID"],
                            "doi_norm": metadata.iloc[original_index]["doi_norm"],
                            "publication_year": int(
                                metadata.iloc[original_index]["publication_year"]
                            ),
                            "scheme": scheme,
                            "fold": fold,
                            "target": target,
                            "parent_family": family,
                            "subgroup": subgroup,
                            "subgroup_class": subgroup_classes[(family, subgroup)],
                            "encoded_column_count": int(len(subgroups[family][subgroup])),
                            "shap_value": value,
                            "abs_shap_value": abs(value),
                            "shap_mcse": float(mcse[row_position, child_index, target_index]),
                            "parent_shap_value": float(
                                parent_values[
                                    row_position, FAMILY_ORDER.index(family), target_index
                                ]
                            ),
                            "base_value": float(base_values[row_position, target_index]),
                            "y_pred": float(predictions[row_position, target_index]),
                            "y_true": float(explain_targets[row_position, target_index]),
                        }
                    )

        if not args.quick:
            current_parent = pd.DataFrame(parent_rows).loc[
                lambda frame: frame["scheme"].eq(scheme) & frame["fold"].eq(fold)
            ]
            frozen_parent = archived_parent.loc[
                archived_parent["scheme"].eq(scheme)
                & archived_parent["fold"].astype(str).eq(str(fold)),
                ["Ref_ID", "scheme", "fold", "target", "family", "shap_value"],
            ]
            merged = current_parent.merge(
                frozen_parent,
                on=["Ref_ID", "scheme", "fold", "target", "family"],
                validate="one_to_one",
                how="left",
            )
            if merged["shap_value"].isna().any():
                raise AssertionError("Archived parent SHAP alignment failed")
            max_archived_parent_difference = max(
                max_archived_parent_difference,
                float(
                    np.max(
                        np.abs(merged["parent_shap_value"] - merged["shap_value"])
                    )
                ),
            )

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
                "selected_parent_families": int(len(selected_families)),
                "child_subgroups": int(len(keys)),
                "path_transitions": int(result["transitions"]),
                "hierarchical_permutations": int(permutations),
                "catboost_candidate": candidate["candidate"],
                "catboost_iterations": int(iterations),
                "archived_prediction_max_difference": float(
                    archived_difference if not args.quick else math.nan
                ),
                "archived_parent_SHAP_max_difference": float(
                    max_archived_parent_difference if not args.quick else math.nan
                ),
                "child_parent_max_difference": float(result["max_child_parent_difference"]),
                "additivity_max_residual": float(result["max_additivity_residual"]),
                "terminal_prediction_max_difference": float(
                    result["max_terminal_prediction_difference"]
                ),
                "runtime_seconds": float(time.perf_counter() - partition_started),
            }
        )
        print(
            f"[{scheme} {fold}] complete in {diagnostics[-1]['runtime_seconds']:.1f}s; "
            f"child groups={len(keys)}",
            flush=True,
        )

    local = pd.DataFrame(local_rows)
    parent_frame = pd.DataFrame(parent_rows)
    duplicate_keys = int(
        local.duplicated(
            ["Ref_ID", "scheme", "fold", "target", "parent_family", "subgroup"]
        ).sum()
    )
    if duplicate_keys:
        raise AssertionError("Duplicated hierarchical local attribution keys")
    fold_importance, summary = bootstrap_importance(
        local, bootstrap_replicates, config.seed + 62000
    )
    class_summary = component_class_summary(summary)
    stability = rank_stability(fold_importance, summary)

    local.to_csv(
        args.output_dir / "hierarchical_attribution_local_values.csv.gz",
        index=False,
        compression="gzip",
    )
    parent_frame.to_csv(
        args.output_dir / "hierarchical_parent_values.csv.gz",
        index=False,
        compression="gzip",
    )
    fold_importance.to_csv(
        args.output_dir / "within_family_importance_by_fold.csv", index=False
    )
    summary.to_csv(args.output_dir / "within_family_importance_summary.csv", index=False)
    class_summary.to_csv(
        args.output_dir / "within_family_component_class_summary.csv", index=False
    )
    stability.to_csv(args.output_dir / "within_family_rank_stability.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(
        args.output_dir / "hierarchical_attribution_diagnostics.csv", index=False
    )

    explained_dois = int(local["doi_norm"].nunique())
    heldout_doi_partition_instances = int(
        local[["scheme", "fold", "doi_norm"]].drop_duplicates().shape[0]
    )
    verification = {
        "status": "passed",
        "explained_DOI_groups": explained_dois,
        "heldout_DOI_partition_instances": heldout_doi_partition_instances,
        "local_rows": int(len(local)),
        "duplicate_local_keys": duplicate_keys,
        "selected_parent_families": selected_families,
        "max_child_parent_difference": max_child_parent_difference,
        "max_archived_parent_SHAP_difference": max_archived_parent_difference
        if not args.quick
        else None,
        "max_archived_prediction_difference": max_archived_prediction_difference
        if not args.quick
        else None,
        "max_hierarchical_additivity_residual": max_additivity_residual,
        "max_terminal_path_prediction_difference": max_terminal_difference,
        "median_local_MCSE": float(local["shap_mcse"].median()),
        "p95_local_MCSE": float(local["shap_mcse"].quantile(0.95)),
        "finite_local_values": bool(
            np.isfinite(local[["shap_value", "shap_mcse", "parent_shap_value"]]).all().all()
        ),
        "grouped_boundary_DOI_overlap": grouped_overlap,
        "chronological_boundary_DOI_overlap": chrono_overlap,
        "historical_vocabulary_selection": "DOI prevalence in train_through_2018 only",
        "bootstrap_replicates": bootstrap_replicates,
        "hierarchical_permutations": permutations,
    }
    thresholds_pass = (
        verification["finite_local_values"]
        and duplicate_keys == 0
        and grouped_overlap == 0
        and chrono_overlap == 0
        and max_child_parent_difference < 1e-10
        and max_additivity_residual < 1e-10
        and max_terminal_difference < 1e-10
    )
    if not args.quick:
        thresholds_pass = thresholds_pass and (
            max_archived_parent_difference < 1e-8
            and max_archived_prediction_difference < 1e-8
        )
    if not thresholds_pass:
        verification["status"] = "failed"
        raise AssertionError(f"Verification thresholds failed: {verification}")
    (args.output_dir / "within_family_verification_report.json").write_text(
        json.dumps(verification, indent=2), encoding="utf-8"
    )
    make_figure(summary, args.output_dir)
    write_markdown_report(
        args.output_dir, summary, stability, selected_families, verification
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
            "catboost": catboost.__version__,
        },
        "inputs": {
            "raw_sha256": sha256(args.raw),
            "cohort_sha256": sha256(args.cohort),
            "split_manifest_sha256": sha256(
                args.baseline_results_dir / "split_manifest.csv"
            ),
            "parent_local_SHAP_sha256": sha256(
                args.parent_results_dir / "shap_local_values.csv.gz"
            ),
            "analysis_code_sha256": sha256(Path(__file__)),
        },
        "model": {
            "training_weighting": FULL_WEIGHTING,
            "model": MODEL,
            "selection_source": str(args.catboost_model_selection.resolve()),
        },
        "hierarchical_attribution": {
            "method": "nested interventional Monte-Carlo Shapley/Owen paths",
            "selected_parent_families": selected_families,
            "permutations_per_record": permutations,
            "background": "one record per training DOI, randomly sampled",
            "evaluation": "one held-out record per DOI",
            "material_vocabulary": "top role-specific tokens by DOI prevalence in <=2018 training only; remaining tokens pooled",
            "role_top_k": ROLE_TOP_K,
        },
        "runtime_note": "A repeated non-numerical sklearn parallel-configuration warning was suppressed for reproducible reruns; numerical equivalence is assessed by frozen-prediction and parent-SHAP checks.",
        "verification": verification,
    }
    (args.output_dir / "within_family_run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(
        f"Completed hierarchical attribution in {manifest['runtime_seconds']:.1f}s; "
        f"local rows={len(local):,}",
        flush=True,
    )


if __name__ == "__main__":
    main()
