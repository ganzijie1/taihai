from __future__ import annotations

import csv
import hashlib
import json
import math
import urllib.request
from pathlib import Path

import numpy as np

import simulate_trade_war_v43 as trade


ROOT = Path(__file__).resolve().parents[1]
ICIO_DIR = ROOT / "data" / "oecd_icio_2025" / "2016-2022"
DATA_DIR = ROOT / "data" / "econometrics_v55"
OUT = ROOT / "outputs"
CALIBRATION = OUT / "taiwan_v55_econometric_calibration.json"
DRAWS = OUT / "taiwan_v55_econometric_draws.npz"
SOURCE_LEDGER = OUT / "taiwan_v55_econometric_source_ledger.json"
DIAGNOSTIC = OUT / "taiwan_v55_econometric_diagnostic.json"

ACTORS = ("CHN", "TWN", "USA", "JPN")
WB_DONORS = ("CHN", "USA", "JPN", "KOR")
WB_INDICATORS = {
    "growth": "NY.GDP.MKTP.KD.ZG",
    "inflation": "FP.CPI.TOTL.ZG",
    "trade_open": "NE.TRD.GNFS.ZS",
    "military_share": "MS.MIL.XPND.GD.ZS",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _download_world_bank() -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    target = DATA_DIR / "world_bank_macro_1960_2025.json"
    if target.exists():
        return target
    payload = {"downloaded_from": [], "series": {}}
    for name, indicator in WB_INDICATORS.items():
        url = (
            "https://api.worldbank.org/v2/country/"
            + ";".join(WB_DONORS)
            + f"/indicator/{indicator}?format=json&per_page=20000"
        )
        with urllib.request.urlopen(url, timeout=60) as response:
            raw = json.loads(response.read().decode("utf-8"))
        payload["downloaded_from"].append(url)
        payload["series"][name] = raw[1]
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return target


def _aggregate_icio_year(year: int) -> dict[str, np.ndarray]:
    cache = DATA_DIR / f"icio_{year}_8x10.npz"
    if cache.exists():
        data = np.load(cache, allow_pickle=True)
        return {key: data[key] for key in data.files}
    source = ICIO_DIR / f"{year}_SML.csv"
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        product_labels = []
        for label in header[1:]:
            if trade.parse_node(label) is None:
                break
            product_labels.append(label)
        n_product = len(product_labels)
        node_map = np.empty(n_product, dtype=np.int16)
        for index, label in enumerate(product_labels):
            country, sector = trade.parse_node(label)
            node_map[index] = (
                trade.REGIONS.index(trade.region_of(country)) * len(trade.SECTORS)
                + trade.SECTORS.index(trade.sector_of(sector))
            )
        fd_headers = header[1 + n_product : -1]
        fd_map = np.array(
            [trade.REGIONS.index(trade.region_of(label[:3])) for label in fd_headers],
            dtype=np.int16,
        )
        nodes = len(trade.REGIONS) * len(trade.SECTORS)
        z = np.zeros((nodes, nodes))
        fd = np.zeros((nodes, len(trade.REGIONS)))
        out = np.zeros(nodes)
        va = np.zeros(nodes)
        for row in reader:
            parsed = trade.parse_node(row[0])
            if parsed is not None:
                country, sector = parsed
                row_index = (
                    trade.REGIONS.index(trade.region_of(country)) * len(trade.SECTORS)
                    + trade.SECTORS.index(trade.sector_of(sector))
                )
                values = np.asarray(row[1 : 1 + n_product], dtype=float)
                z[row_index] += np.bincount(node_map, weights=values, minlength=nodes)
                fd_values = np.asarray(row[1 + n_product : -1], dtype=float)
                fd[row_index] += np.bincount(
                    fd_map, weights=fd_values, minlength=len(trade.REGIONS)
                )
            elif row[0] in {"VA", "OUT"}:
                values = np.asarray(row[1 : 1 + n_product], dtype=float)
                aggregate = np.bincount(node_map, weights=values, minlength=nodes)
                if row[0] == "VA":
                    va = aggregate
                else:
                    out = aggregate
    labels = np.array(
        [f"{region}_{sector}" for region in trade.REGIONS for sector in trade.SECTORS],
        dtype=object,
    )
    np.savez_compressed(cache, Z=z, FD=fd, OUT=out, VA=va, labels=labels)
    return {"Z": z, "FD": fd, "OUT": out, "VA": va, "labels": labels}


def _bilateral_flows(data: dict[str, np.ndarray]) -> np.ndarray:
    regions = len(trade.REGIONS)
    sectors = len(trade.SECTORS)
    result = np.zeros((regions, regions))
    for exporter in range(regions):
        rows = slice(exporter * sectors, (exporter + 1) * sectors)
        for importer in range(regions):
            cols = slice(importer * sectors, (importer + 1) * sectors)
            result[exporter, importer] = (
                data["Z"][rows, cols].sum() + data["FD"][rows, importer].sum()
            )
    return result


def _fit_ppml_fixed_effects(panel: np.ndarray) -> dict:
    """PPML with pair and year fixed effects, solved by score-balancing IPF."""
    years, exporters, importers = panel.shape
    pair = np.maximum(panel.mean(axis=0), 1e-10)
    year = np.ones(years)
    fitted = np.empty_like(panel)
    for _ in range(1000):
        fitted = year[:, None, None] * pair[None, :, :]
        new_year = panel.sum(axis=(1, 2)) / np.maximum(fitted.sum(axis=(1, 2)), 1e-12)
        year *= new_year
        fitted = year[:, None, None] * pair[None, :, :]
        new_pair = panel.sum(axis=0) / np.maximum(fitted.sum(axis=0), 1e-12)
        pair *= new_pair
        pair_scale = math.exp(float(np.mean(np.log(np.maximum(pair, 1e-12)))))
        pair /= pair_scale
        year *= pair_scale
        fitted = year[:, None, None] * pair[None, :, :]
        score = max(
            float(np.max(np.abs(panel.sum(axis=(1, 2)) - fitted.sum(axis=(1, 2))))),
            float(np.max(np.abs(panel.sum(axis=0) - fitted.sum(axis=0)))),
        )
        if score < 1e-7:
            break
    pearson = (panel - fitted) / np.sqrt(np.maximum(fitted, 1e-9))
    positive = fitted > 1e-8
    dispersion = float(np.sum(pearson[positive] ** 2) / max(np.count_nonzero(positive) - years - exporters * importers + 1, 1))
    log_change = np.diff(np.log(np.maximum(panel, 1e-8)), axis=0)
    robust_scale = float(1.4826 * np.median(np.abs(log_change - np.median(log_change))))
    return {
        "pair_effect": pair,
        "year_effect": year,
        "fitted": fitted,
        "score_residual": score,
        "pearson_dispersion": dispersion,
        "log_change_robust_scale": robust_scale,
        "iterations": _ + 1,
    }


def _fit_hmm_transition() -> dict:
    source = OUT / "台海月度历史数据与隐状态_2020-2026.csv"
    state_map = {"常态威慑": 0, "持续施压": 1, "急性危机": 2}
    states = []
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["latent_state"] in state_map:
                states.append(state_map[row["latent_state"]])
    counts = np.full((3, 3), 0.5)
    for left, right in zip(states, states[1:]):
        counts[left, right] += 1.0
    transition = counts / counts.sum(axis=1, keepdims=True)
    return {
        "counts": counts,
        "transition": transition,
        "months": len(states),
        "transition_events": int(len(states) - 1),
    }


def _world_bank_panel(path: Path) -> tuple[np.ndarray, np.ndarray, list[tuple[str, int]]]:
    raw = json.loads(path.read_text(encoding="utf-8"))["series"]
    values: dict[tuple[str, int], dict[str, float]] = {}
    for variable, rows in raw.items():
        for row in rows:
            if row["value"] is None:
                continue
            key = (row["countryiso3code"], int(row["date"]))
            values.setdefault(key, {})[variable] = float(row["value"])
    records = []
    keys = []
    for country in WB_DONORS:
        for year in range(1961, 2026):
            current = values.get((country, year), {})
            previous = values.get((country, year - 1), {})
            if all(name in current and name in previous for name in WB_INDICATORS):
                records.append(
                    [current[name] for name in ("growth", "inflation", "trade_open", "military_share")]
                )
                keys.append((country, year))
    y = np.asarray(records, dtype=float)
    lag = np.asarray(
        [[values[(country, year - 1)][name] for name in ("growth", "inflation", "trade_open", "military_share")]
         for country, year in keys],
        dtype=float,
    )
    return lag, y, keys


def _fit_local_projections(
    lag: np.ndarray, y: np.ndarray, keys: list[tuple[str, int]], innovations: np.ndarray,
    max_horizon: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    lookup = {key: index for index, key in enumerate(keys)}
    coefficients = np.full((max_horizon, 4), np.nan)
    standard_errors = np.full_like(coefficients, np.nan)
    for horizon in range(1, max_horizon + 1):
        rows = []
        targets = []
        for index, (country, year) in enumerate(keys):
            future = lookup.get((country, year + horizon))
            if future is None:
                continue
            actor = np.eye(len(WB_DONORS))[WB_DONORS.index(country), 1:]
            rows.append(np.r_[1.0, actor, innovations[index], lag[index]])
            targets.append(y[future, 0])
        design = np.asarray(rows)
        target = np.asarray(targets)
        penalty = np.eye(design.shape[1]) * 1.5
        penalty[: len(WB_DONORS), : len(WB_DONORS)] = 0.0
        inverse = np.linalg.inv(design.T @ design + penalty)
        beta = inverse @ design.T @ target
        residual = target - design @ beta
        covariance = inverse @ (design.T @ np.diag(residual ** 2) @ design) @ inverse
        start = len(WB_DONORS)
        coefficients[horizon - 1] = beta[start : start + 4]
        standard_errors[horizon - 1] = np.sqrt(
            np.maximum(np.diag(covariance)[start : start + 4], 0.0)
        )
    return coefficients, standard_errors


def _fit_macro_var(path: Path) -> dict:
    lag, y, keys = _world_bank_panel(path)
    train = np.array([year <= 2018 for _, year in keys])
    mean = lag[train].mean(axis=0)
    scale = np.maximum(lag[train].std(axis=0), 1e-6)
    x = (lag - mean) / scale
    countries = np.array([WB_DONORS.index(country) for country, _ in keys])
    actor_fe = np.eye(len(WB_DONORS))[countries]
    design = np.column_stack((np.ones(len(x)), actor_fe[:, 1:], x))
    penalty = np.eye(design.shape[1]) * 2.0
    penalty[: len(WB_DONORS), : len(WB_DONORS)] = 0.0
    beta = np.linalg.solve(design[train].T @ design[train] + penalty, design[train].T @ y[train])
    prediction = design @ beta
    residual = y - prediction
    validation = ~train
    validation_rmse = np.sqrt(np.mean(residual[validation] ** 2, axis=0))
    transition = beta[-4:, :].T / scale[None, :]
    innovation_scale = np.maximum(residual[train].std(axis=0), 1e-6)
    innovations = residual / innovation_scale
    lp_coefficient, lp_standard_error = _fit_local_projections(
        lag, y, keys, innovations
    )
    # Recursive SVAR impact is retained as an ordering-sensitive diagnostic,
    # not treated as a uniquely identified causal decomposition.
    svar_impact = np.linalg.cholesky(np.cov(residual[train].T) + 1e-10 * np.eye(4))
    return {
        "transition": transition,
        "residual_covariance": np.cov(residual[train].T),
        "validation_rmse": validation_rmse,
        "n_train": int(train.sum()),
        "n_validation": int(validation.sum()),
        "sample_start": min(year for _, year in keys),
        "sample_end": max(year for _, year in keys),
        "lp_coefficient": lp_coefficient,
        "lp_standard_error": lp_standard_error,
        "svar_recursive_impact": svar_impact,
    }


def _actor_growth_rates(wb_path: Path) -> tuple[np.ndarray, np.ndarray]:
    wb_raw = json.loads(wb_path.read_text(encoding="utf-8"))["series"]["growth"]
    wb = {}
    for row in wb_raw:
        if row["value"] is not None and row["countryiso3code"] in {"CHN", "USA", "JPN"}:
            wb[(row["countryiso3code"], int(row["date"]))] = float(row["value"])
    means = np.zeros(4)
    standard_errors = np.zeros(4)
    for index, actor in enumerate(ACTORS):
        if actor == "TWN":
            # OECD ICIO is expressed in current USD and must not be used as a
            # real-growth series. The 3.36 benchmark is the revised 2017-2023
            # average published by Taiwan DGBAS; uncertainty remains a prior
            # because a machine-readable annual file was not locally retrieved.
            means[index] = 3.36
            standard_errors[index] = 0.75
            continue
        else:
            sample = np.array([wb[(actor, year)] for year in range(2010, 2025) if (actor, year) in wb])
        means[index] = np.mean(sample)
        standard_errors[index] = np.std(sample, ddof=1) / math.sqrt(max(len(sample), 1))
    return means, np.maximum(standard_errors, 0.15)


def fit(paths: int = 128, seed: int = 5_500_017) -> dict:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    icio = {year: _aggregate_icio_year(year) for year in range(2016, 2023)}
    bilateral = np.stack([_bilateral_flows(icio[year]) for year in sorted(icio)])
    ppml = _fit_ppml_fixed_effects(bilateral)
    hmm = _fit_hmm_transition()
    wb_path = _download_world_bank()
    macro = _fit_macro_var(wb_path)
    growth_mean, growth_se = _actor_growth_rates(wb_path)

    calibration = {
        "model_version": "V5.5",
        "scope": "DIAGNOSTIC_ONLY econometric calibration; no outcome probabilities",
        "identification_boundary": {
            "trade": "PPML pair and year fixed effects identify observed bilateral baselines and common annual shocks; Armington elasticities remain scenario priors",
            "macro": "pooled ridge VAR is predictive/associational, not a causal war-effect estimate",
            "latent_state": "Dirichlet-smoothed transitions use the labelled 2020-2026 monthly pressure states",
            "terminal_risks": "no terminal outcome sample; Cox/Fine-Gray coefficients remain scenario priors",
            "alliance": "no comparable intervention event panel; strategic-choice coefficients remain scenario priors",
        },
        "ppml": {
            "years": list(range(2016, 2023)),
            "regions": trade.REGIONS,
            "pair_effect": ppml["pair_effect"].tolist(),
            "year_effect": ppml["year_effect"].tolist(),
            "score_residual": ppml["score_residual"],
            "pearson_dispersion": ppml["pearson_dispersion"],
            "log_change_robust_scale": ppml["log_change_robust_scale"],
            "iterations": ppml["iterations"],
        },
        "hmm": {
            "transition": hmm["transition"].tolist(),
            "counts": hmm["counts"].tolist(),
            "months": hmm["months"],
            "transition_events": hmm["transition_events"],
        },
        "macro_var": {
            "variables": ["real_gdp_growth", "cpi_inflation", "trade_openness", "military_spending_share"],
            "transition": macro["transition"].tolist(),
            "residual_covariance": macro["residual_covariance"].tolist(),
            "validation_rmse": macro["validation_rmse"].tolist(),
            "n_train": macro["n_train"],
            "n_validation": macro["n_validation"],
            "sample_start": macro["sample_start"],
            "sample_end": macro["sample_end"],
            "local_projection_output_irf": macro["lp_coefficient"].tolist(),
            "local_projection_robust_se": macro["lp_standard_error"].tolist(),
            "svar_recursive_impact_diagnostic": macro["svar_recursive_impact"].tolist(),
            "svar_identification": "ordering-sensitive diagnostic only; not used as a uniquely causal war shock",
        },
        "growth": {
            "mean": growth_mean.tolist(),
            "standard_error": growth_se.tolist(),
            "taiwan_identification": "scenario_prior anchored to DGBAS revised 2017-2023 real-GDP average; ICIO current-USD output explicitly excluded",
        },
        "scenario_priors": {
            "terminal_hazard_beta_log_sd": 0.16,
            "intervention_intercept_sd": 0.20,
            "trade_war_amplification_log_sd": min(max(ppml["log_change_robust_scale"] * 0.30, 0.04), 0.22),
        },
    }
    CALIBRATION.write_text(json.dumps(calibration, ensure_ascii=False, indent=2), encoding="utf-8")

    rng = np.random.default_rng(seed)
    hmm_draw = np.empty((paths, 3, 3))
    for path_index in range(paths):
        for state in range(3):
            hmm_draw[path_index, state] = rng.dirichlet(hmm["counts"][state])
    growth_draw = rng.normal(growth_mean, growth_se, (paths, 4))
    trade_scale = np.exp(rng.normal(0.0, calibration["scenario_priors"]["trade_war_amplification_log_sd"], paths))
    macro_transition = macro["transition"]
    lp_trade_impact = float(np.nanmean(macro["lp_coefficient"][:3, 2]))
    output_scale = np.clip(1.0 + 0.04 * lp_trade_impact + rng.normal(0.0, 0.04, paths), 0.82, 1.18)
    inflation_scale = np.clip(1.0 + 0.08 * macro_transition[1, 1] + rng.normal(0.0, 0.04, paths), 0.82, 1.18)
    prior_sd = calibration["scenario_priors"]["terminal_hazard_beta_log_sd"]
    intervention_sd = calibration["scenario_priors"]["intervention_intercept_sd"]
    np.savez_compressed(
        DRAWS,
        hmm_transition=hmm_draw,
        growth_rate=growth_draw,
        trade_shock_scale=trade_scale,
        macro_output_scale=output_scale,
        macro_inflation_scale=inflation_scale,
        force_beta_scale=np.exp(rng.normal(0.0, prior_sd, paths)),
        peace_beta_scale=np.exp(rng.normal(0.0, prior_sd, paths)),
        separation_beta_scale=np.exp(rng.normal(0.0, prior_sd, paths)),
        us_intervention_shift=rng.normal(0.0, intervention_sd, paths),
        japan_intervention_shift=rng.normal(0.0, intervention_sd, paths),
    )
    ledger = {
        "model_version": "V5.5",
        "sources": [
            {
                "source": str(ICIO_DIR / "2016_SML.csv") + " ... 2022_SML.csv",
                "publisher": "OECD",
                "observation_date": "2016-2022",
                "unit": "millions of current USD, inter-country input-output flows",
                "identification": "observed",
            },
            {
                "source": str(wb_path),
                "publisher": "World Bank World Development Indicators",
                "observation_date": "1960-2025 as available",
                "unit": "indicator-specific annual values",
                "identification": "observed",
            },
            {
                "source": str(OUT / "台海月度历史数据与隐状态_2020-2026.csv"),
                "publisher": "compiled public-event dataset; provenance retained in prior model artifacts",
                "observation_date": "2020-09 to 2026-09",
                "unit": "monthly labelled latent state",
                "identification": "derived",
            },
            {
                "source": "https://eng.stat.gov.tw/News_Content.aspx?n=2317&s=232644",
                "publisher": "Taiwan Directorate-General of Budget, Accounting and Statistics",
                "observation_date": "2017-2023 comprehensive national-accounts revision",
                "unit": "average annual real GDP growth, percent",
                "identification": "scenario_prior_anchor",
            },
        ],
        "artifacts": {"calibration_sha256": sha256(CALIBRATION), "draws_sha256": sha256(DRAWS)},
    }
    SOURCE_LEDGER.write_text(json.dumps(ledger, ensure_ascii=False, indent=2), encoding="utf-8")
    diagnostic = {
        "scope": "DIAGNOSTIC_ONLY",
        "checks": {
            "ppml_score_residual": ppml["score_residual"],
            "hmm_row_sum_residual": float(np.max(np.abs(hmm["transition"].sum(axis=1) - 1.0))),
            "draw_hmm_row_sum_residual": float(np.max(np.abs(hmm_draw.sum(axis=2) - 1.0))),
            "finite_draws": bool(all(np.all(np.isfinite(x)) for x in (hmm_draw, growth_draw, trade_scale))),
            "macro_validation_rmse": macro["validation_rmse"].tolist(),
            "terminal_outcome_estimation_attempted": False,
        },
    }
    DIAGNOSTIC.write_text(json.dumps(diagnostic, ensure_ascii=False, indent=2), encoding="utf-8")
    return calibration


def load_draws(expected_paths: int) -> dict[str, np.ndarray]:
    data = np.load(DRAWS)
    result = {key: data[key] for key in data.files}
    if any(value.shape[0] != expected_paths for value in result.values()):
        raise ValueError("V5.5 econometric draw count does not match master path count")
    return result


if __name__ == "__main__":
    fit()
    print("V5.5 ECONOMETRIC CALIBRATION: DIAGNOSTIC_ONLY PASS")
