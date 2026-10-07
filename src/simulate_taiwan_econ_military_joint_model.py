from __future__ import annotations

import csv
import json
import math
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "work"
OUT = ROOT / "outputs"
YEARS = np.arange(2026, 2101)
ACTORS = ("china", "united_states", "japan", "taiwan")
N = 30_000
SEED = 20261003

# Estimated from the existing 2020-09 to 2026-09 ADIZ pressure series.
MONTHLY_PRESSURE_TRANSITION = np.array([
    [0.8475992633, 0.1152668430, 0.0371338937],
    [0.0142083939, 0.9443913347, 0.0414002714],
    [0.0233899982, 0.0803093399, 0.8963006619],
])
ANNUAL_PRESSURE_TRANSITION = np.linalg.matrix_power(MONTHLY_PRESSURE_TRANSITION, 12)

# 2024 nominal GDP, USD trillion. USA/China/Japan are World Bank values;
# Taiwan is the DGBAS statistical-yearbook value converted to USD.
GDP_2024 = np.array([18.7297, 29.2980, 4.1900, 0.8357])

# 2025 and 2026 real-growth anchors. Taiwan's 2026 anchor is the midpoint
# between the April-2026 IMF WEO (5.2) and the later DGBAS forecast (7.71).
GROWTH_2025 = np.array([5.0, 2.1, 1.2, 8.68])
GROWTH_2026 = np.array([4.4, 2.3, 0.8, 6.455])

# Transparent conversion priors, sampled with uncertainty in the simulation.
# They are not claims that one dollar buys identical capability in each actor.
CONVERSION_PRIOR = np.array([0.83, 1.00, 0.88, 0.76])
THEATER_SHARE_PRIOR = np.array([0.72, 0.23, 0.60, 0.96])
STOCK_DEPRECIATION = np.array([0.045, 0.040, 0.045, 0.050])


@dataclass(frozen=True)
class Scenario:
    name: str
    china_growth: float = 0.0
    us_growth: float = 0.0
    japan_growth: float = 0.0
    taiwan_growth: float = 0.0
    china_burden: float = 0.0
    allied_burden: float = 0.0
    alliance_credibility: float = 0.0
    interdependence: float = 0.0
    political_pressure: float = 0.0
    decoupling: float = 0.0


SCENARIOS = (
    Scenario("central"),
    Scenario("growth_rebalancing_and_stabilization", china_growth=0.15, us_growth=0.10,
             taiwan_growth=0.15, alliance_credibility=0.05, interdependence=0.30,
             political_pressure=-0.25),
    Scenario("china_high_growth_rearmament", china_growth=0.65, china_burden=0.35,
             political_pressure=0.20),
    Scenario("china_structural_slowdown", china_growth=-0.85, china_burden=-0.10,
             political_pressure=0.15),
    Scenario("us_japan_taiwan_coordination", us_growth=0.10, japan_growth=0.15,
             taiwan_growth=0.10, allied_burden=0.40, alliance_credibility=0.55),
    Scenario("technology_trade_decoupling", china_growth=-0.35, taiwan_growth=-0.25,
             interdependence=-0.55, political_pressure=0.30, decoupling=0.60),
    Scenario("closing_window_pressure", china_growth=-0.25, allied_burden=0.35,
             alliance_credibility=0.40, political_pressure=0.65, decoupling=0.25),
)


def xlsx_rows(path: Path, sheet_number: int) -> list[dict[str, str]]:
    ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    with zipfile.ZipFile(path) as archive:
        shared_root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
        shared = ["".join(t.text or "" for t in item.iter(ns + "t"))
                  for item in shared_root.findall(ns + "si")]
        root = ET.fromstring(archive.read(f"xl/worksheets/sheet{sheet_number}.xml"))
    rows = []
    for row in root.iter(ns + "row"):
        values: dict[str, str] = {}
        for cell in row.findall(ns + "c"):
            column = re.match(r"[A-Z]+", cell.attrib["r"]).group()
            value_node = cell.find(ns + "v")
            value = "" if value_node is None else value_node.text or ""
            if cell.attrib.get("t") == "s" and value:
                value = shared[int(value)]
            values[column] = value
        rows.append(values)
    return rows


def sipri_series(sheet_number: int) -> dict[str, dict[int, float]]:
    path = WORK / "SIPRI-Milex-data-1949-2025_v1.2.xlsx"
    rows = xlsx_rows(path, sheet_number)
    header = next(row for row in rows if row.get("A") == "Country")
    years = {column: int(value) for column, value in header.items() if value.isdigit()}
    names = {
        "China": "china",
        "United States of America": "united_states",
        "Japan": "japan",
        "Taiwan": "taiwan",
    }
    result = {actor: {} for actor in ACTORS}
    for row in rows:
        actor = names.get(row.get("A", ""))
        if not actor:
            continue
        for column, year in years.items():
            try:
                result[actor][year] = float(row[column])
            except (KeyError, ValueError):
                pass
    return result


def world_bank_growth() -> dict[str, list[float]]:
    payload = json.loads((WORK / "worldbank_growth.json").read_text(encoding="utf-8"))
    data = {"china": [], "united_states": [], "japan": []}
    mapping = {"CHN": "china", "USA": "united_states", "JPN": "japan"}
    for row in payload[1]:
        if row["value"] is not None:
            data[mapping[row["countryiso3code"]]].append((int(row["date"]), float(row["value"])))
    return {key: [value for _, value in sorted(values)] for key, values in data.items()}


# DGBAS/IMF historical real-growth series. The 2025 observation is the
# February-2026 preliminary DGBAS estimate.
TAIWAN_GROWTH = [
    5.80, -1.65, 5.26, 3.67, 6.19, 4.70, 5.44, 5.98, 0.73, -1.61,
    10.25, 3.67, 2.22, 2.48, 4.72, 1.47, 2.17, 3.31, 2.75, 3.06,
    3.39, 6.62, 2.59, 1.28, 5.30, 8.68,
]


def ar1_calibration(series: list[float]) -> dict[str, float]:
    values = np.clip(np.asarray(series, dtype=float), -4.0, 10.0)
    x = np.column_stack([np.ones(len(values) - 1), values[:-1]])
    y = values[1:]
    ridge = np.diag([0.0, 1.5])
    coefficients = np.linalg.solve(x.T @ x + ridge, x.T @ y)
    residual = y - x @ coefficients
    rho = float(np.clip(coefficients[1], -0.2, 0.75))
    mean = float(coefficients[0] / max(1.0 - rho, 0.2))
    return {"historical_mean": float(values.mean()), "rho": rho,
            "innovation_sd": float(residual.std(ddof=2)), "ar_mean": mean}


def long_run_growth(actor: int, year: int, scenario: Scenario) -> float:
    # Demographic and convergence-consistent central paths after the WEO horizon.
    anchors = {
        0: ((2026, 4.4), (2035, 3.4), (2050, 2.3), (2075, 1.7), (2100, 1.4)),
        1: ((2026, 2.3), (2035, 2.0), (2050, 1.8), (2075, 1.7), (2100, 1.6)),
        2: ((2026, 0.8), (2035, 0.9), (2050, 0.7), (2075, 0.6), (2100, 0.5)),
        3: ((2026, 6.455), (2030, 3.0), (2035, 2.5), (2050, 1.9), (2075, 1.4), (2100, 1.1)),
    }
    offsets = (scenario.china_growth, scenario.us_growth, scenario.japan_growth, scenario.taiwan_growth)
    points = anchors[actor]
    for (y0, g0), (y1, g1) in zip(points[:-1], points[1:]):
        if y0 <= year <= y1:
            weight = (year - y0) / (y1 - y0)
            return (1.0 - weight) * g0 + weight * g1 + offsets[actor]
    return points[-1][1] + offsets[actor]


def defense_burden_path(actor: int, year: int, initial: np.ndarray, scenario: Scenario) -> np.ndarray:
    # Policy targets gradually fade into a bounded long-run burden.
    targets = np.array([1.55, 3.05, 2.00, 3.32])
    targets[0] += scenario.china_burden
    targets[1:] += scenario.allied_burden
    convergence = 1.0 - math.exp(-(year - 2025) / 6.0)
    burden = initial[:, actor] * (1.0 - convergence) + targets[actor] * convergence
    return np.clip(burden, 0.8, 5.5)


def sample_categorical(rng: np.random.Generator, probabilities: np.ndarray) -> np.ndarray:
    draws = rng.random(probabilities.shape[0])
    return (draws[:, None] > np.cumsum(probabilities, axis=1)).sum(axis=1)


def first_event_year(events: np.ndarray, event_code: int) -> np.ndarray:
    hit = events == event_code
    any_hit = hit.any(axis=1)
    index = np.argmax(hit, axis=1)
    return np.where(any_hit, YEARS[index], -1)


def share_by(years: np.ndarray, year: int) -> float:
    return float(np.mean((years > 0) & (years <= year)))


def simulate(scenario: Scenario, seed: int, military_spending: dict[str, dict[int, float]],
             burden_history: dict[str, dict[int, float]], calibrations: dict[str, dict[str, float]]) -> dict:
    rng = np.random.default_rng(seed)
    steps = len(YEARS)

    gdp = np.tile(GDP_2024 * (1.0 + GROWTH_2025 / 100.0), (N, 1))
    growth = np.tile(GROWTH_2026, (N, 1))

    spend_2025 = np.array([military_spending[a][2025] for a in ACTORS]) / 1000.0
    stock_seed = []
    for actor in ACTORS:
        values = military_spending[actor]
        stock = 0.0
        for year in range(2000, 2026):
            stock = (1.0 - STOCK_DEPRECIATION[ACTORS.index(actor)]) * stock + values.get(year, values[min(values)])
        stock_seed.append(stock / 1000.0)
    stock = np.tile(np.array(stock_seed), (N, 1))

    conversion = rng.lognormal(np.log(CONVERSION_PRIOR), 0.11, (N, 4))
    theater_share = np.clip(rng.normal(THEATER_SHARE_PRIOR, [0.07, 0.06, 0.08, 0.03], (N, 4)), 0.08, 1.0)
    initial_burden_values = []
    for actor in ACTORS:
        raw = burden_history[actor].get(2025)
        if raw is None or not np.isfinite(raw):
            raw = 100.0 * spend_2025[ACTORS.index(actor)] / gdp[0, ACTORS.index(actor)]
        initial_burden_values.append(raw)
    initial_burden = rng.normal(initial_burden_values, [0.08, 0.12, 0.10, 0.12], (N, 4))

    pressure = rng.choice(3, N, p=[0.10, 0.62, 0.28]).astype(np.int8)
    alive = np.ones(N, dtype=bool)
    events = np.zeros((N, steps), dtype=np.int8)  # 1 force, 2 peace, 3 separation

    # Economic linkage starts near the observed Taiwan export-share scale but
    # includes investment and supply-chain exposure, so it is normalized.
    linkage = np.clip(rng.normal(0.34, 0.05, N), 0.15, 0.55)
    identity_distance = np.clip(rng.normal(0.72, 0.05, N), 0.50, 0.90)
    credibility = np.clip(rng.normal(0.58 + scenario.alliance_credibility, 0.10, N), 0.15, 0.95)

    snapshots: dict[int, dict[str, np.ndarray]] = {}
    previous_balance = np.ones(N)
    common_noise = np.zeros(N)

    for j, year in enumerate(YEARS):
        innovations = np.empty((N, 4))
        for i, actor in enumerate(ACTORS):
            sigma = calibrations[actor]["innovation_sd"]
            innovations[:, i] = rng.normal(0.0, min(max(sigma, 0.65), 2.0), N)
            rho = calibrations[actor]["rho"]
            target = long_run_growth(i, year, scenario)
            growth[:, i] = target + rho * (growth[:, i] - target) + 0.40 * innovations[:, i]
        common_noise = 0.45 * common_noise + rng.normal(0.0, 0.55, N)
        growth += common_noise[:, None] * np.array([0.55, 0.45, 0.60, 0.70])
        if scenario.decoupling:
            growth[:, [0, 3]] -= scenario.decoupling * np.array([0.18, 0.25])
        growth = np.clip(growth, -8.0, 10.0)
        gdp *= 1.0 + growth / 100.0

        burden = np.column_stack([defense_burden_path(i, year, initial_burden, scenario) for i in range(4)])
        # GDP is in USD trillion and the accumulated stock is in USD billion.
        annual_spend = gdp * 1000.0 * burden / 100.0
        stock = stock * (1.0 - STOCK_DEPRECIATION) + annual_spend

        # Technology, conversion and distance/logistics convert stock into
        # theater-available capability. The exponent prevents money from being
        # treated as linearly interchangeable with capability.
        tech_trend = np.array([
            1.0 + 0.0045 * (year - 2025),
            1.0 + 0.0035 * (year - 2025),
            1.0 + 0.0038 * (year - 2025),
            1.0 + 0.0040 * (year - 2025),
        ])
        capability = np.power(np.maximum(stock, 1e-6), 0.78) * conversion * theater_share * tech_trend

        p_us = 1.0 / (1.0 + np.exp(-(-0.35 + 2.0 * credibility + 0.35 * pressure
                                      - 0.70 * scenario.interdependence)))
        p_jp = 1.0 / (1.0 + np.exp(-(-0.85 + 1.65 * credibility + 0.45 * pressure
                                      + 0.35 * p_us)))
        opposition = capability[:, 3] + p_us * capability[:, 1] + p_jp * capability[:, 2]
        complementarity = 0.10 * np.sqrt(np.maximum(capability[:, 3] * (p_us * capability[:, 1] + p_jp * capability[:, 2]), 0))
        opposition += complementarity
        balance = capability[:, 0] / np.maximum(opposition, 1e-6)
        bilateral_balance = capability[:, 0] / np.maximum(capability[:, 3], 1e-6)
        closing_window = np.maximum(previous_balance - balance, 0.0)
        previous_balance = balance.copy()

        # Pressure transition preserves the empirically estimated reversible
        # HMM, then tilts its rows using capability and political covariates.
        base_probs = ANNUAL_PRESSURE_TRANSITION[pressure]
        tilt = 0.24 * np.log(np.maximum(balance, 0.15)) + 0.35 * closing_window
        tilt += scenario.political_pressure + 0.30 * scenario.decoupling
        logits = np.log(np.maximum(base_probs, 1e-12))
        logits[:, 0] -= tilt
        logits[:, 2] += tilt
        logits -= logits.max(axis=1, keepdims=True)
        probs = np.exp(logits)
        probs /= probs.sum(axis=1, keepdims=True)
        pressure = sample_categorical(rng, probs).astype(np.int8)

        linkage += 0.025 * ((0.28 + 0.12 * scenario.interdependence) - linkage)
        linkage -= 0.0015 * pressure + 0.0030 * scenario.decoupling
        linkage = np.clip(linkage, 0.04, 0.55)
        identity_distance = np.clip(identity_distance + 0.0012 + 0.0010 * pressure, 0.50, 0.98)

        # Cause-specific competing risks. Coefficients are deliberately broad
        # priors because there is no observed terminal-event sample to identify
        # a Taiwan-specific force-onset regression.
        feasibility = np.log(np.maximum(balance, 0.15))
        deterrence = np.log1p(np.maximum(opposition / np.maximum(capability[:, 0], 1e-6), 0))
        econ_stress = np.maximum(2.4 - growth[:, 0], 0.0)
        log_force = -5.85 + np.choose(pressure, [0.0, 0.55, 1.25])
        log_force += 0.55 * feasibility - 0.50 * deterrence + 1.10 * closing_window
        log_force += 0.40 * identity_distance - 0.55 * linkage + 0.10 * econ_stress
        log_force += scenario.political_pressure

        log_peace = -6.25 - 0.55 * pressure + 0.90 * linkage + 0.35 * scenario.interdependence
        log_peace += 0.20 * deterrence - 0.45 * identity_distance
        log_separation = -6.45 + 0.35 * pressure + 0.65 * identity_distance - 0.35 * linkage

        hazards = np.column_stack([np.exp(log_force), np.exp(log_peace), np.exp(log_separation)])
        total = hazards.sum(axis=1)
        event_probability = 1.0 - np.exp(-total)
        occur = alive & (rng.random(N) < event_probability)
        if np.any(occur):
            chosen = sample_categorical(rng, hazards[occur] / total[occur, None]) + 1
            events[occur, j] = chosen
            alive[occur] = False

        if year in {2030, 2035, 2040, 2050, 2075, 2100}:
            snapshots[year] = {
                "gdp": gdp.copy(), "spend": annual_spend.copy(), "capability": capability.copy(),
                "balance": balance.copy(), "bilateral_balance": bilateral_balance.copy(),
                "p_us": p_us.copy(), "p_jp": p_jp.copy(),
                "linkage": linkage.copy(), "pressure": pressure.copy(), "alive": alive.copy(),
            }

    force_year = first_event_year(events, 1)
    peace_year = first_event_year(events, 2)
    separation_year = first_event_year(events, 3)
    terminal_year = np.maximum.reduce([force_year, peace_year, separation_year])
    windows = ((2026, 2030), (2031, 2035), (2036, 2040), (2041, 2050),
               (2051, 2060), (2061, 2070), (2071, 2080), (2081, 2090), (2091, 2100))
    window_rates = []
    for start, end in windows:
        at_risk = (terminal_year < 0) | (terminal_year >= start)
        force_count = np.sum((force_year >= start) & (force_year <= end))
        annualized = force_count / max(np.sum(at_risk), 1) / (end - start + 1)
        window_rates.append({
            "window": f"{start}-{end}",
            "force_incidence_all_paths": float(force_count / N),
            "annualized_force_hazard_given_survival": float(annualized),
        })
    results = []
    for year, snap in snapshots.items():
        row = {
            "year": int(year),
            "force": share_by(force_year, year),
            "peace": share_by(peace_year, year),
            "separation": share_by(separation_year, year),
            "status_quo": float(np.mean((force_year < 0) & (peace_year < 0) & (separation_year < 0))) if year == 2100 else float(np.mean(snap["alive"])),
            "china_us_gdp_ratio": float(np.median(snap["gdp"][:, 0] / snap["gdp"][:, 1])),
            "china_taiwan_theater_balance": float(np.median(snap["bilateral_balance"])),
            "china_allied_theater_balance": float(np.median(snap["balance"])),
            "balance_p10": float(np.quantile(snap["balance"], 0.10)),
            "balance_p90": float(np.quantile(snap["balance"], 0.90)),
            "us_support_availability_score": float(np.median(snap["p_us"])),
            "japan_support_availability_score": float(np.median(snap["p_jp"])),
            "economic_linkage": float(np.median(snap["linkage"])),
            "latent_crisis_given_survival": float(np.mean(snap["pressure"][snap["alive"]] == 2)) if np.any(snap["alive"]) else 0.0,
        }
        for i, actor in enumerate(ACTORS):
            row[f"{actor}_gdp_index"] = float(np.median(snap["gdp"][:, i] / GDP_2024[i]))
            row[f"{actor}_military_stock_index"] = float(np.median(snap["capability"][:, i] / snapshots[2030]["capability"][:, i])) if year >= 2030 else 1.0
        results.append(row)
    peak_window = max(window_rates, key=lambda x: x["annualized_force_hazard_given_survival"])["window"]
    return {"scenario": scenario.name, "years": results,
            "force_risk_windows": window_rates, "peak_force_risk_window": peak_window,
            "force_year_median_conditional": None if np.all(force_year < 0) else float(np.median(force_year[force_year > 0]))}


def main() -> None:
    OUT.mkdir(exist_ok=True)
    military_spending = sipri_series(5)  # constant 2024 USD million
    burden_history = sipri_series(7)     # percentage of GDP
    growth_data = world_bank_growth()
    growth_data["taiwan"] = TAIWAN_GROWTH
    calibrations = {actor: ar1_calibration(growth_data[actor]) for actor in ACTORS}

    simulations = [simulate(s, SEED + i * 1009, military_spending, burden_history, calibrations)
                   for i, s in enumerate(SCENARIOS)]
    payload = {
        "model": "four-economy stochastic growth + defense-capital accumulation + theater conversion + alliance-entry game + reversible HMM pressure + cause-specific competing risks",
        "paths_per_scenario": N,
        "period": [int(YEARS[0]), int(YEARS[-1])],
        "seed": SEED,
        "calibration": {
            "growth_ar1": calibrations,
            "sipri_2025_constant_2024_usd_billion": {a: military_spending[a][2025] / 1000 for a in ACTORS},
            "annual_pressure_transition": ANNUAL_PRESSURE_TRANSITION.tolist(),
        },
        "identification_warning": "Economic and military-stock paths are data-calibrated. Theater conversion, alliance entry, and terminal force/peace/separation coefficients are uncertain structural priors because Taiwan has no repeated terminal-event sample.",
        "results": simulations,
    }
    (OUT / "台海经济军力联合模型_仿真摘要.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    rows = []
    for result in simulations:
        for row in result["years"]:
            rows.append({"scenario": result["scenario"], **row})
    with (OUT / "台海经济军力联合模型_长期预测.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    central = next(r for r in simulations if r["scenario"] == "central")
    print(json.dumps({"calibration": payload["calibration"], "central": central,
                      "scenario_2050_2100": [
                          {"scenario": r["scenario"],
                           "2050": next(x for x in r["years"] if x["year"] == 2050),
                           "2100": next(x for x in r["years"] if x["year"] == 2100)}
                          for r in simulations]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
