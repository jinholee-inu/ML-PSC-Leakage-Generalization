#!/usr/bin/env python3
"""Composition-domain reliability audit for the frozen PSC predictor.

This script does not train or alter a predictive model. It joins the archived
full 1/n_DOI weighted Random-Forest predictions and uncertainty/OOD diagnostics
to the audited PSC composition fields, then evaluates FA/MA/Cs and Pb/Sn
domains on the exact same 2019--2021 records under DOI-grouped and
chronological validation.
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
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn


GROUPED_SCHEME = "DOI-grouped 5-fold"
CHRONO_SCHEME = "Chronological >2018"
TARGETS = ["PCE", "Voc", "Jsc", "FF"]
TARGET_UNITS = {
    "PCE": "percentage point",
    "Voc": "V",
    "Jsc": "mA cm^-2",
    "FF": "percentage point",
}

A_ORDER = [
    "MA (no FA/Cs)",
    "FA (no MA/Cs)",
    "Cs (no FA/MA)",
    "FA+MA",
    "FA+Cs",
    "MA+Cs",
    "FA+MA+Cs",
    "Other/unknown",
]
B_ORDER = ["Pb (no Sn)", "Sn (no Pb)", "Pb+Sn", "Other/unknown"]

OOD_THRESHOLD = 0.95
MIN_DESCRIPTIVE_DOI = 5
MIN_INFERENTIAL_DOI = 20
BOOTSTRAP_REPLICATES = 1000
SEED = 20260829

COMPOSITION_COLUMNS = [
    "Ref_ID",
    "Ref_DOI_number",
    "Ref_publication_date",
    "Perovskite_composition_a_ions",
    "Perovskite_composition_a_ions_coefficients",
    "Perovskite_composition_b_ions",
    "Perovskite_composition_b_ions_coefficients",
    "Perovskite_composition_short_form",
    "Perovskite_composition_inorganic",
    "Perovskite_composition_leadfree",
]

METRIC_NAMES = [
    "mean_measured",
    "mean_predicted",
    "MAE",
    "bias",
    "RMSE",
    "R2",
    "target_SD",
    "MAE_over_target_SD",
    "mean_feature_OOD_percentile",
    "high_feature_OOD_fraction",
    "mean_model_OOD_percentile",
    "high_model_OOD_fraction",
    "formula_unseen_fraction",
    "coverage_90",
    "coverage_95",
    "interval_90_mean_width",
    "interval_95_mean_width",
]

COMPARISON_METRICS = [
    "MAE",
    "bias",
    "RMSE",
    "R2",
    "mean_feature_OOD_percentile",
    "high_feature_OOD_fraction",
    "coverage_90",
    "interval_90_mean_width",
]


@dataclass(frozen=True)
class DomainSpec:
    name: str
    column: str
    order: tuple[str, ...] | None = None


DOMAIN_SPECS = [
    DomainSpec("A-site pattern", "a_site_pattern", tuple(A_ORDER)),
    DomainSpec("B-site pattern", "b_site_pattern", tuple(B_ORDER)),
    DomainSpec("A x B domain", "composition_domain", None),
    DomainSpec("FA presence", "FA_status", ("FA absent", "FA present")),
    DomainSpec("MA presence", "MA_status", ("MA absent", "MA present")),
    DomainSpec("Cs presence", "Cs_status", ("Cs absent", "Cs present")),
    DomainSpec("Pb presence", "Pb_status", ("Pb absent", "Pb present")),
    DomainSpec("Sn presence", "Sn_status", ("Sn absent", "Sn present")),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", required=True, type=Path)
    parser.add_argument("--cohort", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--bootstrap", type=int, default=BOOTSTRAP_REPLICATES)
    parser.add_argument("--seed", type=int, default=SEED)
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


def numbers(value: object) -> list[float]:
    if pd.isna(value):
        return []
    found = re.findall(
        r"(?<![A-Za-z])[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(value)
    )
    output: list[float] = []
    for item in found:
        try:
            number = float(item)
        except ValueError:
            continue
        if math.isfinite(number):
            output.append(number)
    return output


def split_layered(value: object) -> list[str]:
    """Split semicolon-delimited ions and pipe-delimited absorber layers."""
    if pd.isna(value):
        return []
    text = str(value).strip()
    if text.lower() in {"", "unknown", "nan", "none", "not reported"}:
        return []
    return [part.strip() for part in re.split(r"\s*(?:;|\|)\s*", text) if part.strip()]


def ion_presence(value: object) -> set[str]:
    return {item.upper() for item in split_layered(value)}


def robust_fractions(
    ions_value: object, coefficients_value: object, requested: Iterable[str]
) -> tuple[dict[str, float], bool]:
    ions = split_layered(ions_value)
    coefficient_parts = split_layered(coefficients_value)
    coeffs: list[float] = []
    for item in coefficient_parts:
        vals = numbers(item)
        coeffs.append(vals[0] if vals else np.nan)
    keys = list(requested)
    result = {key: np.nan for key in keys}
    result["other"] = np.nan
    if not ions or len(ions) != len(coeffs) or not all(np.isfinite(coeffs)):
        return result, False
    total = float(np.sum(coeffs))
    if total <= 0:
        return result, False
    result = {key: 0.0 for key in keys}
    result["other"] = 0.0
    lookup = {key.lower(): key for key in keys}
    for ion, coefficient in zip(ions, coeffs):
        key = lookup.get(ion.strip().lower())
        if key is None:
            result["other"] += float(coefficient) / total
        else:
            result[key] += float(coefficient) / total
    return result, True


def a_pattern(tokens: set[str]) -> str:
    fa, ma, cs = "FA" in tokens, "MA" in tokens, "CS" in tokens
    if fa and ma and cs:
        return "FA+MA+Cs"
    if fa and ma:
        return "FA+MA"
    if fa and cs:
        return "FA+Cs"
    if ma and cs:
        return "MA+Cs"
    if fa:
        return "FA (no MA/Cs)"
    if ma:
        return "MA (no FA/Cs)"
    if cs:
        return "Cs (no FA/MA)"
    return "Other/unknown"


def b_pattern(tokens: set[str]) -> str:
    pb, sn = "PB" in tokens, "SN" in tokens
    if pb and sn:
        return "Pb+Sn"
    if pb:
        return "Pb (no Sn)"
    if sn:
        return "Sn (no Pb)"
    return "Other/unknown"


def build_composition_metadata(raw: pd.DataFrame, cohort: pd.DataFrame) -> pd.DataFrame:
    if raw["Ref_ID"].duplicated().any():
        raise ValueError("Raw Ref_ID is not unique.")
    frame = cohort[["Ref_ID"]].merge(
        raw[COMPOSITION_COLUMNS], on="Ref_ID", how="left", validate="one_to_one"
    )
    if frame["Ref_DOI_number"].isna().any():
        raise ValueError("Composition join left missing DOI metadata.")
    frame["doi_norm"] = normalize_doi(frame["Ref_DOI_number"])
    frame["publication_year"] = pd.to_datetime(
        frame["Ref_publication_date"], errors="coerce"
    ).dt.year
    if frame["publication_year"].isna().any():
        raise ValueError("Publication year could not be parsed.")

    a_tokens = frame["Perovskite_composition_a_ions"].map(ion_presence)
    b_tokens = frame["Perovskite_composition_b_ions"].map(ion_presence)
    frame["FA_present"] = a_tokens.map(lambda x: "FA" in x)
    frame["MA_present"] = a_tokens.map(lambda x: "MA" in x)
    frame["Cs_present"] = a_tokens.map(lambda x: "CS" in x)
    frame["Pb_present"] = b_tokens.map(lambda x: "PB" in x)
    frame["Sn_present"] = b_tokens.map(lambda x: "SN" in x)
    frame["a_site_pattern"] = a_tokens.map(a_pattern)
    frame["b_site_pattern"] = b_tokens.map(b_pattern)
    frame["composition_domain"] = (
        frame["a_site_pattern"] + " / " + frame["b_site_pattern"]
    )
    for ion in ["FA", "MA", "Cs", "Pb", "Sn"]:
        frame[f"{ion}_status"] = np.where(
            frame[f"{ion}_present"], f"{ion} present", f"{ion} absent"
        )

    a_fraction_rows = []
    b_fraction_rows = []
    for row in frame.itertuples(index=False):
        a_values, a_ok = robust_fractions(
            row.Perovskite_composition_a_ions,
            row.Perovskite_composition_a_ions_coefficients,
            ["FA", "MA", "Cs"],
        )
        a_values["parsed"] = a_ok
        a_fraction_rows.append(a_values)
        b_values, b_ok = robust_fractions(
            row.Perovskite_composition_b_ions,
            row.Perovskite_composition_b_ions_coefficients,
            ["Pb", "Sn"],
        )
        b_values["parsed"] = b_ok
        b_fraction_rows.append(b_values)
    a_frac = pd.DataFrame(a_fraction_rows, index=frame.index).rename(
        columns={
            "FA": "A_FA_fraction",
            "MA": "A_MA_fraction",
            "Cs": "A_Cs_fraction",
            "other": "A_other_fraction",
            "parsed": "A_fraction_parsed",
        }
    )
    b_frac = pd.DataFrame(b_fraction_rows, index=frame.index).rename(
        columns={
            "Pb": "B_Pb_fraction",
            "Sn": "B_Sn_fraction",
            "other": "B_other_fraction",
            "parsed": "B_fraction_parsed",
        }
    )
    frame = pd.concat([frame, a_frac, b_frac], axis=1)
    denominator = frame["B_Pb_fraction"] + frame["B_Sn_fraction"]
    frame["Sn_fraction_among_Pb_Sn"] = np.where(
        denominator > 0, frame["B_Sn_fraction"] / denominator, np.nan
    )

    frame["absorber_short_form_clean"] = frame[
        "Perovskite_composition_short_form"
    ].map(clean_text)
    historical_formulas = set(
        frame.loc[
            frame["publication_year"].le(2018), "absorber_short_form_clean"
        ].tolist()
    )
    frame["formula_seen_historical"] = frame[
        "absorber_short_form_clean"
    ].isin(historical_formulas)
    frame["formula_unseen_historical"] = ~frame["formula_seen_historical"]
    return frame


def domain_values(frame: pd.DataFrame, spec: DomainSpec) -> list[str]:
    present = frame[spec.column].dropna().astype(str).unique().tolist()
    if spec.order is None:
        return sorted(present)
    ordered = [item for item in spec.order if item in present]
    return ordered + sorted(set(present).difference(ordered))


def publication_balanced_mean(frame: pd.DataFrame, column: str) -> float:
    if frame.empty:
        return np.nan
    return float(frame.groupby("doi_norm", sort=False)[column].mean().mean())


def support_table(metadata: pd.DataFrame) -> pd.DataFrame:
    train = metadata.loc[metadata["publication_year"].le(2018)].copy()
    future = metadata.loc[metadata["publication_year"].ge(2019)].copy()
    rows: list[dict[str, object]] = []
    for spec in DOMAIN_SPECS:
        for domain in domain_values(metadata, spec):
            tr = train.loc[train[spec.column].eq(domain)]
            te = future.loc[future[spec.column].eq(domain)]
            if tr.empty and te.empty:
                continue
            historical_doi = int(tr["doi_norm"].nunique())
            future_doi = int(te["doi_norm"].nunique())
            if historical_doi >= 100:
                support_class = "established (>=100 historical DOI)"
            elif historical_doi >= 10:
                support_class = "limited (10-99 historical DOI)"
            else:
                support_class = "sparse (<10 historical DOI)"
            rows.append(
                {
                    "domain_type": spec.name,
                    "domain": domain,
                    "historical_records": int(len(tr)),
                    "historical_DOI": historical_doi,
                    "future_records": int(len(te)),
                    "future_DOI": future_doi,
                    "historical_record_share": float(len(tr) / len(train)),
                    "future_record_share": float(len(te) / len(future)),
                    "historical_DOI_share": float(
                        historical_doi / train["doi_norm"].nunique()
                    ),
                    "future_DOI_share": float(
                        future_doi / future["doi_norm"].nunique()
                    ),
                    "future_formula_unseen_device_fraction": float(
                        te["formula_unseen_historical"].mean()
                    )
                    if len(te)
                    else np.nan,
                    "future_formula_unseen_DOI_balanced_fraction": publication_balanced_mean(
                        te, "formula_unseen_historical"
                    ),
                    "support_class": support_class,
                    "descriptive_eligible": future_doi >= MIN_DESCRIPTIVE_DOI,
                    "inferential_eligible": future_doi >= MIN_INFERENTIAL_DOI,
                }
            )
    return pd.DataFrame(rows)


def aggregate_doi(frame: pd.DataFrame) -> pd.DataFrame:
    work = pd.DataFrame(
        {
            "doi_norm": frame["doi_norm"],
            "measured": frame["y_true"].astype(float),
            "predicted": frame["y_pred"].astype(float),
            "measured_sq": frame["y_true"].astype(float) ** 2,
            "absolute_error": (frame["y_pred"] - frame["y_true"]).abs(),
            "residual": frame["y_pred"] - frame["y_true"],
            "squared_error": (frame["y_pred"] - frame["y_true"]) ** 2,
            "feature_ood": frame["feature_ood_percentile"].astype(float),
            "high_feature_ood": frame["feature_ood_percentile"].ge(OOD_THRESHOLD),
            "model_ood": frame["model_ood_percentile"].astype(float),
            "high_model_ood": frame["model_ood_percentile"].ge(OOD_THRESHOLD),
            "formula_unseen": frame["formula_unseen_historical"].astype(float),
            "coverage_90": frame["interval_90_covered"].astype(float),
            "coverage_95": frame["interval_95_covered"].astype(float),
            "width_90": 2.0 * frame["interval_90_half_width"].astype(float),
            "width_95": 2.0 * frame["interval_95_half_width"].astype(float),
        }
    )
    return work.groupby("doi_norm", sort=False).mean()


def metric_bundle(
    frame: pd.DataFrame,
    all_dois: pd.Index,
    bootstrap_counts: np.ndarray,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    aggregate = aggregate_doi(frame)
    position = pd.Series(np.arange(len(all_dois)), index=all_dois)
    indices = position.loc[aggregate.index].to_numpy(dtype=int)
    mask = np.zeros(len(all_dois), dtype=float)
    mask[indices] = 1.0
    denominator = bootstrap_counts @ mask

    def values(column: str) -> tuple[float, np.ndarray]:
        vector = np.zeros(len(all_dois), dtype=float)
        vector[indices] = aggregate[column].to_numpy(dtype=float)
        point = float(aggregate[column].mean())
        with np.errstate(invalid="ignore", divide="ignore"):
            boot = (bootstrap_counts @ vector) / denominator
        boot[denominator <= 0] = np.nan
        return point, boot

    y, y_boot = values("measured")
    yp, yp_boot = values("predicted")
    y2, y2_boot = values("measured_sq")
    mae, mae_boot = values("absolute_error")
    bias, bias_boot = values("residual")
    mse, mse_boot = values("squared_error")
    feature_ood, feature_ood_boot = values("feature_ood")
    high_feature, high_feature_boot = values("high_feature_ood")
    model_ood, model_ood_boot = values("model_ood")
    high_model, high_model_boot = values("high_model_ood")
    unseen, unseen_boot = values("formula_unseen")
    coverage90, coverage90_boot = values("coverage_90")
    coverage95, coverage95_boot = values("coverage_95")
    width90, width90_boot = values("width_90")
    width95, width95_boot = values("width_95")

    variance = max(y2 - y * y, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        variance_boot = y2_boot - y_boot * y_boot
        r2_boot = 1.0 - mse_boot / variance_boot
        target_sd_boot = np.sqrt(np.maximum(variance_boot, 0.0))
        mae_sd_boot = mae_boot / target_sd_boot
    r2 = 1.0 - mse / variance if variance > 0 else np.nan
    target_sd = math.sqrt(variance)
    mae_sd = mae / target_sd if target_sd > 0 else np.nan
    points = {
        "mean_measured": y,
        "mean_predicted": yp,
        "MAE": mae,
        "bias": bias,
        "RMSE": math.sqrt(mse),
        "R2": r2,
        "target_SD": target_sd,
        "MAE_over_target_SD": mae_sd,
        "mean_feature_OOD_percentile": feature_ood,
        "high_feature_OOD_fraction": high_feature,
        "mean_model_OOD_percentile": model_ood,
        "high_model_OOD_fraction": high_model,
        "formula_unseen_fraction": unseen,
        "coverage_90": coverage90,
        "coverage_95": coverage95,
        "interval_90_mean_width": width90,
        "interval_95_mean_width": width95,
    }
    boots = {
        "mean_measured": y_boot,
        "mean_predicted": yp_boot,
        "MAE": mae_boot,
        "bias": bias_boot,
        "RMSE": np.sqrt(mse_boot),
        "R2": r2_boot,
        "target_SD": target_sd_boot,
        "MAE_over_target_SD": mae_sd_boot,
        "mean_feature_OOD_percentile": feature_ood_boot,
        "high_feature_OOD_fraction": high_feature_boot,
        "mean_model_OOD_percentile": model_ood_boot,
        "high_model_OOD_fraction": high_model_boot,
        "formula_unseen_fraction": unseen_boot,
        "coverage_90": coverage90_boot,
        "coverage_95": coverage95_boot,
        "interval_90_mean_width": width90_boot,
        "interval_95_mean_width": width95_boot,
    }
    return points, boots


def ci(values: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return np.nan, np.nan
    return tuple(np.quantile(finite, [0.025, 0.975]).tolist())


def performance_tables(
    predictions: pd.DataFrame,
    bootstrap_counts: np.ndarray,
    all_dois: pd.Index,
) -> tuple[pd.DataFrame, dict[tuple[str, str, str, str], dict[str, np.ndarray]]]:
    rows: list[dict[str, object]] = []
    cache: dict[tuple[str, str, str, str], dict[str, np.ndarray]] = {}
    for scheme in [GROUPED_SCHEME, CHRONO_SCHEME]:
        for target in TARGETS:
            base = predictions.loc[
                predictions["scheme"].eq(scheme) & predictions["target"].eq(target)
            ]
            for spec in DOMAIN_SPECS:
                for domain in domain_values(base, spec):
                    group = base.loc[base[spec.column].eq(domain)]
                    doi_count = int(group["doi_norm"].nunique())
                    if not doi_count:
                        continue
                    points, boots = metric_bundle(group, all_dois, bootstrap_counts)
                    record: dict[str, object] = {
                        "scheme": scheme,
                        "target": target,
                        "unit": TARGET_UNITS[target],
                        "domain_type": spec.name,
                        "domain": domain,
                        "records": int(len(group)),
                        "DOI": doi_count,
                        "descriptive_eligible": doi_count >= MIN_DESCRIPTIVE_DOI,
                        "inferential_eligible": doi_count >= MIN_INFERENTIAL_DOI,
                    }
                    for metric in METRIC_NAMES:
                        record[metric] = points[metric]
                        low, high = ci(boots[metric]) if doi_count >= MIN_DESCRIPTIVE_DOI else (np.nan, np.nan)
                        record[f"{metric}_CI_low"] = low
                        record[f"{metric}_CI_high"] = high
                    rows.append(record)
                    cache[(scheme, target, spec.name, domain)] = boots
    return pd.DataFrame(rows), cache


def comparisons_table(
    performance: pd.DataFrame,
    cache: dict[tuple[str, str, str, str], dict[str, np.ndarray]],
) -> pd.DataFrame:
    lookup = performance.set_index(["scheme", "target", "domain_type", "domain"])
    rows: list[dict[str, object]] = []

    def add_comparison(
        comparison_type: str,
        target: str,
        domain_type: str,
        domain: str,
        scheme_a: str,
        domain_a: str,
        scheme_b: str,
        domain_b: str,
        label: str,
    ) -> None:
        key_a = (scheme_a, target, domain_type, domain_a)
        key_b = (scheme_b, target, domain_type, domain_b)
        if key_a not in cache or key_b not in cache:
            return
        row_a = lookup.loc[key_a]
        row_b = lookup.loc[key_b]
        for metric in COMPARISON_METRICS:
            point = float(row_a[metric] - row_b[metric])
            with np.errstate(invalid="ignore"):
                boot = cache[key_a][metric] - cache[key_b][metric]
            low, high = ci(boot)
            rows.append(
                {
                    "comparison_type": comparison_type,
                    "target": target,
                    "unit": TARGET_UNITS[target],
                    "domain_type": domain_type,
                    "domain": domain,
                    "comparison": label,
                    "metric": metric,
                    "estimate": point,
                    "CI_low": low,
                    "CI_high": high,
                    "CI_excludes_zero": bool(np.isfinite(low) and np.isfinite(high) and (low > 0 or high < 0)),
                    "domain_DOI": int(row_a["DOI"]),
                    "reference_DOI": int(row_b["DOI"]),
                    "inferential_eligible": bool(
                        row_a["DOI"] >= MIN_INFERENTIAL_DOI
                        and row_b["DOI"] >= MIN_INFERENTIAL_DOI
                    ),
                }
            )

    for target in TARGETS:
        for domain_type in performance["domain_type"].unique():
            domains = performance.loc[
                performance["target"].eq(target)
                & performance["domain_type"].eq(domain_type),
                "domain",
            ].unique()
            for domain in domains:
                add_comparison(
                    "Chronological minus DOI-grouped on identical future records",
                    target,
                    domain_type,
                    domain,
                    CHRONO_SCHEME,
                    domain,
                    GROUPED_SCHEME,
                    domain,
                    "chronological - DOI-grouped",
                )

        for domain in A_ORDER:
            if domain == "MA (no FA/Cs)":
                continue
            add_comparison(
                "Within chronological cohort",
                target,
                "A-site pattern",
                domain,
                CHRONO_SCHEME,
                domain,
                CHRONO_SCHEME,
                "MA (no FA/Cs)",
                f"{domain} - MA (no FA/Cs)",
            )
        for domain in B_ORDER:
            if domain == "Pb (no Sn)":
                continue
            add_comparison(
                "Within chronological cohort",
                target,
                "B-site pattern",
                domain,
                CHRONO_SCHEME,
                domain,
                CHRONO_SCHEME,
                "Pb (no Sn)",
                f"{domain} - Pb (no Sn)",
            )
        for ion in ["FA", "MA", "Cs", "Pb", "Sn"]:
            add_comparison(
                "Presence contrast within chronological cohort",
                target,
                f"{ion} presence",
                f"{ion} present",
                CHRONO_SCHEME,
                f"{ion} present",
                CHRONO_SCHEME,
                f"{ion} absent",
                f"{ion} present - absent",
            )
    return pd.DataFrame(rows)


def high_efficiency_table(
    predictions: pd.DataFrame,
    bootstrap_counts: np.ndarray,
    all_dois: pd.Index,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    high = predictions.loc[
        predictions["target"].eq("PCE") & predictions["y_true"].ge(20.0)
    ]
    for scheme in [GROUPED_SCHEME, CHRONO_SCHEME]:
        base = high.loc[high["scheme"].eq(scheme)]
        for spec in DOMAIN_SPECS[:3]:
            for domain in domain_values(base, spec):
                group = base.loc[base[spec.column].eq(domain)]
                doi_count = int(group["doi_norm"].nunique())
                if not doi_count:
                    continue
                points, boots = metric_bundle(group, all_dois, bootstrap_counts)
                record = {
                    "scheme": scheme,
                    "domain_type": spec.name,
                    "domain": domain,
                    "records": int(len(group)),
                    "DOI": doi_count,
                    "descriptive_eligible": doi_count >= MIN_DESCRIPTIVE_DOI,
                    "inferential_eligible": doi_count >= MIN_INFERENTIAL_DOI,
                }
                for metric in ["mean_measured", "mean_predicted", "MAE", "bias", "RMSE"]:
                    record[metric] = points[metric]
                    low, high_ci = ci(boots[metric]) if doi_count >= MIN_DESCRIPTIVE_DOI else (np.nan, np.nan)
                    record[f"{metric}_CI_low"] = low
                    record[f"{metric}_CI_high"] = high_ci
                rows.append(record)
    return pd.DataFrame(rows)


def draw_figure(
    performance: pd.DataFrame,
    support: pd.DataFrame,
    output_dir: Path,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.2,
            "axes.titlesize": 11,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 9.8))
    fig.suptitle(
        "Composition-domain reliability of the DOI-balanced PSC predictor",
        x=0.06,
        y=0.985,
        ha="left",
        fontsize=15,
        fontweight="bold",
    )

    a_domains = [item for item in A_ORDER if item != "Other/unknown"]
    chrono_a = performance.loc[
        performance["scheme"].eq(CHRONO_SCHEME)
        & performance["target"].eq("PCE")
        & performance["domain_type"].eq("A-site pattern")
        & performance["domain"].isin(a_domains)
    ].set_index("domain").loc[a_domains]
    support_a = support.loc[
        support["domain_type"].eq("A-site pattern")
        & support["domain"].isin(a_domains)
    ].set_index("domain").loc[a_domains]
    ax = axes[0, 0]
    y = np.arange(len(a_domains))
    x = 100 * chrono_a["high_feature_OOD_fraction"].to_numpy()
    low = 100 * chrono_a["high_feature_OOD_fraction_CI_low"].to_numpy()
    high = 100 * chrono_a["high_feature_OOD_fraction_CI_high"].to_numpy()
    sizes = 55 + 260 * np.sqrt(
        support_a["future_DOI"].to_numpy() / support_a["future_DOI"].max()
    )
    color_value = np.log10(np.maximum(support_a["historical_DOI"].to_numpy(), 1))
    scatter = ax.scatter(
        x,
        y,
        s=sizes,
        c=color_value,
        cmap="viridis",
        edgecolor="white",
        linewidth=0.8,
        zorder=3,
    )
    ax.errorbar(x, y, xerr=[x - low, high - x], fmt="none", ecolor="#4B5563", capsize=2.5, zorder=2)
    ax.set_yticks(y, a_domains)
    ax.invert_yaxis()
    ax.set_xlabel("DOI-balanced future weight above historical 95th OOD percentile (%)")
    ax.set_title("(a) A-site support and feature-space OOD", loc="left", fontweight="bold")
    ax.grid(axis="x", color="#E5E7EB", linewidth=0.7)
    cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("log10 historical DOI support")
    ax.set_xlim(0, 75)
    for yi, n in zip(y, support_a["future_DOI"]):
        ax.text(73.5, yi, f"n={int(n)} DOI", ha="right", va="center", fontsize=7.3, color="#374151")

    ax = axes[0, 1]
    offsets = {GROUPED_SCHEME: -0.12, CHRONO_SCHEME: 0.12}
    colors = {GROUPED_SCHEME: "#8A9AAF", CHRONO_SCHEME: "#D97745"}
    labels = {GROUPED_SCHEME: "DOI-grouped", CHRONO_SCHEME: "Chronological"}
    for scheme in [GROUPED_SCHEME, CHRONO_SCHEME]:
        data = performance.loc[
            performance["scheme"].eq(scheme)
            & performance["target"].eq("PCE")
            & performance["domain_type"].eq("A-site pattern")
        ].set_index("domain").loc[a_domains]
        xv = data["MAE"].to_numpy()
        lo = data["MAE_CI_low"].to_numpy()
        hi = data["MAE_CI_high"].to_numpy()
        ax.errorbar(
            xv,
            y + offsets[scheme],
            xerr=[xv - lo, hi - xv],
            fmt="o",
            color=colors[scheme],
            ecolor=colors[scheme],
            capsize=2.5,
            markersize=5,
            label=labels[scheme],
        )
    ax.set_yticks(y, a_domains)
    ax.invert_yaxis()
    ax.set_xlabel("Publication-balanced PCE MAE (percentage point)")
    ax.set_title("(b) Temporal performance loss by A-site pattern", loc="left", fontweight="bold")
    ax.grid(axis="x", color="#E5E7EB", linewidth=0.7)
    ax.legend(frameon=False, loc="lower right")

    b_domains = [item for item in B_ORDER if item != "Other/unknown"]
    ax = axes[1, 0]
    yb = np.arange(len(b_domains))
    for scheme in [GROUPED_SCHEME, CHRONO_SCHEME]:
        data = performance.loc[
            performance["scheme"].eq(scheme)
            & performance["target"].eq("PCE")
            & performance["domain_type"].eq("B-site pattern")
        ].set_index("domain").loc[b_domains]
        xv = 100 * data["high_feature_OOD_fraction"].to_numpy()
        lo = 100 * data["high_feature_OOD_fraction_CI_low"].to_numpy()
        hi = 100 * data["high_feature_OOD_fraction_CI_high"].to_numpy()
        ax.errorbar(
            xv,
            yb + offsets[scheme],
            xerr=[xv - lo, hi - xv],
            fmt="o",
            color=colors[scheme],
            ecolor=colors[scheme],
            capsize=2.5,
            markersize=5.5,
            label=labels[scheme],
        )
    ax.set_yticks(yb, b_domains)
    ax.invert_yaxis()
    ax.set_xlabel("DOI-balanced future weight above partition 95th OOD percentile (%)")
    ax.set_title("(c) B-site OOD increases under temporal transfer", loc="left", fontweight="bold")
    ax.grid(axis="x", color="#E5E7EB", linewidth=0.7)
    ax.legend(frameon=False, loc="upper right")

    ax = axes[1, 1]
    heat = performance.loc[
        performance["scheme"].eq(CHRONO_SCHEME)
        & performance["domain_type"].eq("B-site pattern")
        & performance["domain"].isin(b_domains)
    ].pivot(index="domain", columns="target", values="R2").reindex(index=b_domains, columns=TARGETS)
    image = ax.imshow(heat.to_numpy(), cmap="RdBu", vmin=-0.55, vmax=0.75, aspect="auto")
    ax.set_xticks(np.arange(len(TARGETS)), ["PCE", r"$V_{OC}$", r"$J_{SC}$", "FF"])
    ax.set_yticks(np.arange(len(b_domains)), b_domains)
    ax.set_title("(d) Chronological within-domain $R^2$", loc="left", fontweight="bold")
    for i in range(len(b_domains)):
        for j in range(len(TARGETS)):
            value = heat.iloc[i, j]
            color = "white" if abs(value) > 0.33 else "#111827"
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", color=color, fontweight="bold")
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label(r"Publication-balanced $R^2$")
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.text(
        0.06,
        0.015,
        "The future cohort contains the same 2019-2021 DOI groups in both validation schemes. "
        "Error bars: DOI-cluster bootstrap 95% CI (1,000 replicates). OOD is target-free. "
        "Bubble area in panel (a) scales with future DOI count.",
        fontsize=8,
        color="#4B5563",
    )
    fig.tight_layout(rect=[0.04, 0.045, 0.99, 0.95], h_pad=2.2, w_pad=2.0)
    for suffix, dpi in [("png", 600), ("pdf", None), ("svg", None)]:
        path = output_dir / f"Figure8_composition_domain_reliability.{suffix}"
        kwargs = {"bbox_inches": "tight"}
        if dpi is not None:
            kwargs["dpi"] = dpi
        fig.savefig(path, **kwargs)
    plt.close(fig)


def selected_row(
    performance: pd.DataFrame,
    target: str,
    domain_type: str,
    domain: str,
    scheme: str = CHRONO_SCHEME,
) -> pd.Series:
    row = performance.loc[
        performance["scheme"].eq(scheme)
        & performance["target"].eq(target)
        & performance["domain_type"].eq(domain_type)
        & performance["domain"].eq(domain)
    ]
    if row.empty:
        raise KeyError((scheme, target, domain_type, domain))
    return row.iloc[0]


def report_text(
    performance: pd.DataFrame,
    support: pd.DataFrame,
    high_efficiency: pd.DataFrame,
    verification: dict[str, object],
) -> str:
    cs = selected_row(performance, "PCE", "A-site pattern", "Cs (no FA/MA)")
    triple = selected_row(performance, "PCE", "A-site pattern", "FA+MA+Cs")
    pb_sn = selected_row(performance, "PCE", "B-site pattern", "Pb+Sn")
    pb_sn_j = selected_row(performance, "Jsc", "B-site pattern", "Pb+Sn")
    pb_sn_ff = selected_row(performance, "FF", "B-site pattern", "Pb+Sn")
    sn = selected_row(performance, "PCE", "B-site pattern", "Sn (no Pb)")
    sn_v = selected_row(performance, "Voc", "B-site pattern", "Sn (no Pb)")
    cross = selected_row(
        performance,
        "PCE",
        "A x B domain",
        "FA+MA+Cs / Pb+Sn",
    )
    support_lookup = support.set_index(["domain_type", "domain"])
    pb_sn_sup = support_lookup.loc[("B-site pattern", "Pb+Sn")]
    sn_sup = support_lookup.loc[("B-site pattern", "Sn (no Pb)")]
    cross_sup = support_lookup.loc[("A x B domain", "FA+MA+Cs / Pb+Sn")]

    he = high_efficiency.loc[
        high_efficiency["scheme"].eq(CHRONO_SCHEME)
        & high_efficiency["domain_type"].eq("A-site pattern")
        & high_efficiency["inferential_eligible"]
    ].sort_values("bias")
    he_sentence = ""
    if len(he):
        worst = he.iloc[0]
        he_sentence = (
            f"Among A-site patterns with at least {MIN_INFERENTIAL_DOI} high-efficiency DOI groups, "
            f"the largest underprediction occurred for **{worst.domain}** "
            f"(bias {worst.bias:.3f} percentage point; MAE {worst.MAE:.3f})."
        )

    return f"""# Composition-domain OOD and prediction-error audit

## Scope

This post-hoc audit retained the frozen 33,175-record cohort, normalized DOI groups, full `1/n_DOI` weighted Random Forest, validation partitions, prediction intervals, and target-free OOD scores. No model was retrained. The analysis joins archived predictions to the database A- and B-site ion fields and evaluates the same 2019-2021 records under DOI-grouped and chronological validation.

A-site domains are presence patterns among FA, MA, and Cs; B-site domains are presence patterns among Pb and Sn. The word `no` in a label refers only to absence of the other named target ions. Other minor or low-dimensional A/B ions may co-occur. Ions separated by the database layer delimiter `|` are included in the presence audit.

## Main results

- **Cs without FA/MA was far from historical feature support.** Its chronological high-OOD fraction was **{100*cs.high_feature_OOD_fraction:.1f}%** (95% CI {100*cs.high_feature_OOD_fraction_CI_low:.1f}-{100*cs.high_feature_OOD_fraction_CI_high:.1f}%), with PCE MAE **{cs.MAE:.3f}** and mean bias **{cs.bias:.3f}** percentage point. This domain contained {int(cs.DOI)} future DOI groups.
- **FA+MA+Cs was comparatively supported despite its newer chemistry.** Its high-OOD fraction was **{100*triple.high_feature_OOD_fraction:.1f}%**, PCE MAE **{triple.MAE:.3f}**, and bias **{triple.bias:.3f}** percentage point across {int(triple.DOI)} future DOI groups.
- **Mixed Pb-Sn was the clearest calibrated-performance failure.** Only {int(pb_sn_sup.historical_DOI)} historical and {int(pb_sn_sup.future_DOI)} future DOI groups were available, and **{100*pb_sn_sup.future_formula_unseen_DOI_balanced_fraction:.1f}%** of future exact-formula exposure was unseen historically. Chronological PCE MAE was **{pb_sn.MAE:.3f}**, bias **{pb_sn.bias:.3f}**, and within-domain R2 **{pb_sn.R2:.3f}**. The associated Jsc MAE/bias were **{pb_sn_j.MAE:.3f}/{pb_sn_j.bias:.3f} mA cm^-2**, while FF bias was **{pb_sn_ff.bias:.3f}** percentage point and R2 was **{pb_sn_ff.R2:.3f}**.
- **Sn without Pb had high OOD but deceptively modest absolute PCE error.** The high-OOD fraction was **{100*sn.high_feature_OOD_fraction:.1f}%**, with {int(sn_sup.historical_DOI)} historical and {int(sn_sup.future_DOI)} future DOI groups and **{100*sn_sup.future_formula_unseen_DOI_balanced_fraction:.1f}%** unseen-formula exposure. PCE MAE was only **{sn.MAE:.3f}** because the domain mean PCE was **{sn.mean_measured:.3f}%**; normalized reliability remained weak, including Voc R2 **{sn_v.R2:.3f}**.
- **The sparsest high-impact cell was FA+MA+Cs / Pb+Sn.** It had only {int(cross_sup.historical_DOI)} historical and {int(cross_sup.future_DOI)} future DOI groups, **{100*cross_sup.future_formula_unseen_DOI_balanced_fraction:.1f}%** unseen exact formulas, PCE MAE **{cross.MAE:.3f}**, and bias **{cross.bias:.3f}** percentage point. This is descriptive because the future-domain DOI count is below {MIN_INFERENTIAL_DOI}.
- {he_sentence}

## Interpretation

The results separate two distinct failure modes. Cs-only and Sn-only devices are frequently target-free OOD because their feature combinations occupy sparse regions of the historical manifold. Mixed Pb-Sn devices show an additional calibration problem: absolute and variance-normalized errors are poor, especially for Jsc and FF, and chronological underprediction exceeds that observed under DOI-grouped validation on the same future records. A low absolute MAE in a low-PCE subgroup must therefore not be interpreted as reliable transfer.

For prospective reporting, predictions for `Pb+Sn`, `Sn (no Pb)`, and `Cs (no FA/MA)` should carry an explicit composition-domain flag, OOD percentile, historical DOI support count, and interval. The mixed `FA+MA+Cs / Pb+Sn` cell should be treated as unsupported rather than used for materials ranking.

## Statistical definitions

- Device rows were weighted so that every DOI contributed equal total evaluation weight within a domain.
- High OOD denotes a feature-prototype distance above the corresponding training-partition 95th percentile.
- Confidence intervals use a paired global DOI-cluster bootstrap with {verification['bootstrap_replicates']:,} replicates. The same resampled future DOI groups were used for DOI-grouped-versus-chronological contrasts.
- Domains with fewer than {MIN_DESCRIPTIVE_DOI} future DOI groups receive descriptive point estimates only; domains with fewer than {MIN_INFERENTIAL_DOI} are not used for primary inferential claims.
- R2 was calculated globally after assigning equal total evaluation weight to each DOI; it was not averaged over per-DOI R2 values.

## Interpretation boundaries

The analysis is post-hoc and composition-domain associations are not causal material effects. Presence patterns do not distinguish all stoichiometries or all low-dimensional cations. Exact-formula novelty reflects the database short-form category and can be sensitive to nomenclature. The legacy data snapshot contains relatively few Sn-only and mixed Pb-Sn DOI groups, so negative or near-zero within-domain R2 values are more informative than fine ranking among sparse subgroups.

## Integrity checks

- Composition assignments: {verification['composition_assignment_rows']:,} records, {verification['normalized_DOI_groups']:,} normalized DOI groups
- Future paired prediction rows: {verification['future_paired_prediction_rows']:,}
- Duplicate prediction keys: {verification['duplicate_prediction_keys']}
- Missing composition joins: {verification['missing_composition_joins']}
- Maximum y_true difference between paired validation schemes: {verification['max_paired_y_true_difference']:.3e}
- OOD range: {verification['feature_OOD_range'][0]:.3f} to {verification['feature_OOD_range'][1]:.3f}
- Bootstrap replicates: {verification['bootstrap_replicates']:,}
- Verification status: {verification['status']}
"""


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    raw = pd.read_csv(args.raw, usecols=COMPOSITION_COLUMNS, low_memory=False)
    cohort = pd.read_csv(args.cohort, low_memory=False)
    metadata = build_composition_metadata(raw, cohort)
    support = support_table(metadata)

    predictions = pd.read_csv(args.predictions, low_memory=False)
    expected_condition = (
        predictions["training_weighting"].eq("Full 1/n_DOI")
        & predictions["model"].eq("Random Forest")
    )
    if not expected_condition.all():
        raise ValueError("Prediction archive contains an unexpected model or weighting.")
    predictions = predictions.loc[
        predictions["publication_year"].ge(2019)
        & predictions["scheme"].isin([GROUPED_SCHEME, CHRONO_SCHEME])
    ].copy()
    predictions = predictions.merge(
        metadata.drop(columns=["Ref_DOI_number", "Ref_publication_date"]),
        on="Ref_ID",
        how="left",
        suffixes=("", "_composition"),
        validate="many_to_one",
    )
    if predictions["a_site_pattern"].isna().any():
        raise ValueError("Prediction-to-composition join was incomplete.")
    if not predictions["doi_norm"].eq(predictions["doi_norm_composition"]).all():
        raise ValueError("Normalized DOI changed during the composition join.")
    predictions = predictions.drop(columns=["doi_norm_composition", "publication_year_composition"])

    future_dois = pd.Index(
        sorted(metadata.loc[metadata["publication_year"].ge(2019), "doi_norm"].unique())
    )
    rng = np.random.default_rng(args.seed)
    bootstrap_counts = rng.multinomial(
        len(future_dois),
        np.full(len(future_dois), 1.0 / len(future_dois)),
        size=args.bootstrap,
    ).astype(np.int16)

    performance, cache = performance_tables(predictions, bootstrap_counts, future_dois)
    comparisons = comparisons_table(performance, cache)
    high_efficiency = high_efficiency_table(predictions, bootstrap_counts, future_dois)

    assignment_columns = [
        "Ref_ID",
        "doi_norm",
        "publication_year",
        "Perovskite_composition_short_form",
        "Perovskite_composition_a_ions",
        "Perovskite_composition_a_ions_coefficients",
        "Perovskite_composition_b_ions",
        "Perovskite_composition_b_ions_coefficients",
        "a_site_pattern",
        "b_site_pattern",
        "composition_domain",
        "FA_present",
        "MA_present",
        "Cs_present",
        "Pb_present",
        "Sn_present",
        "A_FA_fraction",
        "A_MA_fraction",
        "A_Cs_fraction",
        "A_other_fraction",
        "A_fraction_parsed",
        "B_Pb_fraction",
        "B_Sn_fraction",
        "B_other_fraction",
        "B_fraction_parsed",
        "Sn_fraction_among_Pb_Sn",
        "absorber_short_form_clean",
        "formula_seen_historical",
        "formula_unseen_historical",
    ]
    metadata[assignment_columns].to_csv(
        args.output_dir / "composition_domain_assignments.csv.gz", index=False
    )
    prediction_columns = [
        "Ref_ID",
        "doi_norm",
        "publication_year",
        "scheme",
        "fold",
        "target",
        "y_true",
        "y_pred",
        "residual",
        "absolute_error",
        "feature_ood_distance_ratio",
        "feature_ood_percentile",
        "model_leaf_support",
        "model_ood_percentile",
        "interval_90_half_width",
        "interval_90_covered",
        "interval_95_half_width",
        "interval_95_covered",
        "a_site_pattern",
        "b_site_pattern",
        "composition_domain",
        "FA_present",
        "MA_present",
        "Cs_present",
        "Pb_present",
        "Sn_present",
        "A_FA_fraction",
        "A_MA_fraction",
        "A_Cs_fraction",
        "B_Pb_fraction",
        "B_Sn_fraction",
        "Sn_fraction_among_Pb_Sn",
        "formula_unseen_historical",
    ]
    predictions[prediction_columns].to_csv(
        args.output_dir / "composition_domain_predictions.csv.gz", index=False
    )
    support.to_csv(args.output_dir / "composition_domain_support.csv", index=False)
    performance.to_csv(
        args.output_dir / "composition_domain_performance.csv", index=False
    )
    comparisons.to_csv(
        args.output_dir / "composition_domain_comparisons.csv", index=False
    )
    high_efficiency.to_csv(
        args.output_dir / "composition_domain_high_efficiency_PCE.csv", index=False
    )

    draw_figure(performance, support, args.output_dir)

    grouped_keys = predictions.loc[predictions["scheme"].eq(GROUPED_SCHEME), ["Ref_ID", "target", "y_true"]]
    chrono_keys = predictions.loc[predictions["scheme"].eq(CHRONO_SCHEME), ["Ref_ID", "target", "y_true"]]
    paired = grouped_keys.merge(
        chrono_keys, on=["Ref_ID", "target"], suffixes=("_grouped", "_chrono"), validate="one_to_one"
    )
    duplicate_keys = int(
        predictions.duplicated(["Ref_ID", "scheme", "target"]).sum()
    )
    ci_columns = [col for col in performance.columns if col.endswith("_CI_low")]
    ci_order_valid = True
    for low_col in ci_columns:
        high_col = low_col.replace("_CI_low", "_CI_high")
        valid = performance[low_col].isna() | performance[high_col].isna() | performance[low_col].le(performance[high_col])
        ci_order_valid = ci_order_valid and bool(valid.all())
    expected_future_rows = 9116 * len(TARGETS) * 2
    verification = {
        "status": "passed",
        "composition_assignment_rows": int(len(metadata)),
        "normalized_DOI_groups": int(metadata["doi_norm"].nunique()),
        "historical_records": int(metadata["publication_year"].le(2018).sum()),
        "future_records": int(metadata["publication_year"].ge(2019).sum()),
        "future_DOI": int(len(future_dois)),
        "future_paired_prediction_rows": int(len(predictions)),
        "expected_future_paired_prediction_rows": int(expected_future_rows),
        "duplicate_prediction_keys": duplicate_keys,
        "missing_composition_joins": int(predictions["a_site_pattern"].isna().sum()),
        "max_paired_y_true_difference": float(
            np.max(np.abs(paired["y_true_grouped"] - paired["y_true_chrono"]))
        ),
        "feature_OOD_range": [
            float(predictions["feature_ood_percentile"].min()),
            float(predictions["feature_ood_percentile"].max()),
        ],
        "bootstrap_replicates": int(args.bootstrap),
        "bootstrap_count_shape": list(bootstrap_counts.shape),
        "bootstrap_row_sums_valid": bool(
            np.all(bootstrap_counts.sum(axis=1) == len(future_dois))
        ),
        "performance_CI_order_valid": bool(ci_order_valid),
        "A_fraction_parsed_fraction": float(metadata["A_fraction_parsed"].mean()),
        "B_fraction_parsed_fraction": float(metadata["B_fraction_parsed"].mean()),
        "figure_files_present": all(
            (args.output_dir / f"Figure8_composition_domain_reliability.{suffix}").exists()
            for suffix in ["png", "pdf", "svg"]
        ),
    }
    required_truths = [
        len(metadata) == 33175,
        metadata["doi_norm"].nunique() == 6368,
        int(metadata["publication_year"].le(2018).sum()) == 24059,
        int(metadata["publication_year"].ge(2019).sum()) == 9116,
        len(future_dois) == 1735,
        len(predictions) == expected_future_rows,
        duplicate_keys == 0,
        verification["max_paired_y_true_difference"] <= 1e-12,
        verification["bootstrap_row_sums_valid"],
        verification["performance_CI_order_valid"],
        verification["figure_files_present"],
    ]
    if not all(required_truths):
        verification["status"] = "failed"
    with (args.output_dir / "composition_domain_verification_report.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(verification, handle, indent=2)

    report = report_text(performance, support, high_efficiency, verification)
    (args.output_dir / "PSC_composition_domain_audit_report.md").write_text(
        report, encoding="utf-8"
    )

    manifest = {
        "status": verification["status"],
        "runtime_seconds": time.perf_counter() - started,
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
            "predictions_path": str(args.predictions.resolve()),
            "predictions_sha256": sha256(args.predictions),
        },
        "design": {
            "model_retrained": False,
            "future_cohort": "2019-2021",
            "paired_schemes": [GROUPED_SCHEME, CHRONO_SCHEME],
            "primary_evaluation": "publication-balanced",
            "OOD_threshold": OOD_THRESHOLD,
            "bootstrap_replicates": int(args.bootstrap),
            "seed": int(args.seed),
            "minimum_descriptive_DOI": MIN_DESCRIPTIVE_DOI,
            "minimum_inferential_DOI": MIN_INFERENTIAL_DOI,
            "A_site_definition": "presence pattern among FA, MA, and Cs across semicolon- and pipe-delimited layers",
            "B_site_definition": "presence pattern among Pb and Sn across semicolon- and pipe-delimited layers",
        },
        "verification": verification,
    }
    with (args.output_dir / "composition_domain_run_manifest.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(manifest, handle, indent=2)

    if verification["status"] != "passed":
        raise RuntimeError("Composition-domain verification failed.")
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    main()
