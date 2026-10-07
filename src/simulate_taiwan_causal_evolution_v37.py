import csv
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
SEED = 20261006
PATHS = 18000
START_YEAR = 2026
END_YEAR = 2100
CHECKPOINTS = (2030, 2035, 2040, 2050, 2075, 2100)

CHINA_STRATEGIES = ("status_quo", "gray_pressure", "blockade", "limited_strike", "invasion")
ALLIANCE_STRATEGIES = ("noncombat", "limited", "delayed_full", "timely_full")
TAIWAN_STRATEGIES = ("restraint", "resilience", "legal_separation")


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -35.0, 35.0)))


def normalize_simplex(x):
    x = np.maximum(x, 1e-9)
    return x / x.sum(axis=1, keepdims=True)


def ridge_fit(x, y, penalty=2.0):
    design = np.column_stack((np.ones(len(x)), x))
    gram = design.T @ design
    regularizer = np.eye(gram.shape[0]) * penalty
    regularizer[0, 0] = 0.0
    return np.linalg.solve(gram + regularizer, design.T @ y)


def ridge_predict(x, beta):
    return beta[0] + x @ beta[1:]


def logistic_ridge(x, y, penalty=2.0, iterations=80):
    design = np.column_stack((np.ones(len(x)), x))
    beta = np.zeros(design.shape[1])
    regularizer = np.eye(design.shape[1]) * penalty
    regularizer[0, 0] = 0.0
    for _ in range(iterations):
        p = sigmoid(design @ beta)
        w = np.maximum(p * (1.0 - p), 1e-5)
        hessian = design.T @ (design * w[:, None]) + regularizer
        gradient = design.T @ (p - y) + regularizer @ beta
        step = np.linalg.solve(hessian, gradient)
        beta -= step
        if np.max(np.abs(step)) < 1e-8:
            break
    return beta


def load_history_module():
    path = ROOT / "work" / "taiwan_dynamic_game_hmm.py"
    spec = importlib.util.spec_from_file_location("taiwan_history", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def causal_local_effect(seed=SEED):
    """Cross-fitted AIPW estimate of coded political-shock effect on log ADIZ activity."""
    module = load_history_module()
    rows = module.build_rows()
    records = []
    for t in range(1, len(rows)):
        year, month = map(int, rows[t]["month"].split("-"))
        records.append({
            "year": year,
            "treatment": rows[t]["shock"],
            "outcome": rows[t]["log_adiz"],
            "features": [
                rows[t - 1]["log_adiz"], rows[t]["election"],
                rows[t]["defense_growth"], rows[t]["export_share_cn_hk"],
                math.sin(2.0 * math.pi * month / 12.0),
                math.cos(2.0 * math.pi * month / 12.0), t / (len(rows) - 1),
            ],
        })
    x = np.asarray([r["features"] for r in records], dtype=float)
    t = np.asarray([r["treatment"] for r in records], dtype=float)
    y = np.asarray([r["outcome"] for r in records], dtype=float)
    years = np.asarray([r["year"] for r in records])
    x = (x - x.mean(axis=0)) / np.where(x.std(axis=0) > 1e-8, x.std(axis=0), 1.0)

    propensity = np.zeros(len(y))
    m0 = np.zeros(len(y))
    m1 = np.zeros(len(y))
    for held_year in np.unique(years):
        train = years != held_year
        test = ~train
        p_beta = logistic_ridge(x[train], t[train], penalty=4.0)
        propensity[test] = sigmoid(p_beta[0] + x[test] @ p_beta[1:])
        augmented = np.column_stack((x[train], t[train], x[train] * t[train, None]))
        outcome_beta = ridge_fit(augmented, y[train], penalty=4.0)
        zero = np.column_stack((x[test], np.zeros(test.sum()), np.zeros_like(x[test])))
        one = np.column_stack((x[test], np.ones(test.sum()), x[test]))
        m0[test] = ridge_predict(zero, outcome_beta)
        m1[test] = ridge_predict(one, outcome_beta)
    propensity = np.clip(propensity, 0.025, 0.975)
    pseudo = m1 - m0 + t * (y - m1) / propensity - (1.0 - t) * (y - m0) / (1.0 - propensity)
    estimate = float(pseudo.mean())

    rng = np.random.default_rng(seed)
    unique_years = np.unique(years)
    boot = []
    for _ in range(3000):
        sampled = rng.choice(unique_years, size=len(unique_years), replace=True)
        indices = np.concatenate([np.flatnonzero(years == yr) for yr in sampled])
        boot.append(float(pseudo[indices].mean()))

    treated_outcome = y[t == 1]
    untreated_outcome = y[t == 0]
    naive = float(treated_outcome.mean() - untreated_outcome.mean())
    return {
        "estimand": "conditional local average effect of pre-coded political-shock months on log(1+monthly ADIZ sorties)",
        "method": "leave-one-year-out cross-fitted augmented inverse-probability weighting",
        "months": int(len(y)),
        "treated_months": int(t.sum()),
        "estimate_log_points": estimate,
        "multiplicative_ratio": float(math.exp(estimate)),
        "year_block_bootstrap_p10_p50_p90": [float(v) for v in np.quantile(boot, [0.10, 0.50, 0.90])],
        "naive_difference_log_points": naive,
        "propensity_min_median_max": [float(propensity.min()), float(np.median(propensity)), float(propensity.max())],
        "identification": "Interpret causally only under consistency, conditional exchangeability and overlap. Six treated months make the interval fragile; the estimate calibrates a local pressure channel, not war probability.",
    }


def load_reference_paths():
    joint = json.loads((OUT / "台海经济军力联合模型_仿真摘要.json").read_text(encoding="utf-8"))
    central = next(row for row in joint["results"] if row["scenario"] == "central")
    years = np.asarray([r["year"] for r in central["years"]], dtype=float)
    fields = {}
    for key in ("china_taiwan_theater_balance", "china_allied_theater_balance", "economic_linkage"):
        values = np.asarray([r[key] for r in central["years"]], dtype=float)
        fields[key] = {year: float(np.interp(year, years, values)) for year in range(START_YEAR, END_YEAR + 1)}
    return fields


def load_campaign_table():
    path = OUT / "台海V3.6_GIS走廊多层备件搜索与军事经济_仿真摘要.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload["scenario_envelopes"]
    table = {}
    for case in ("timely_full", "limited", "delayed", "taiwan_alone"):
        case_rows = sorted((r for r in rows if r["case"] == case), key=lambda r: r["conflict_year"])
        xs = np.asarray([r["conflict_year"] for r in case_rows], dtype=float)
        table[case] = {}
        for year in range(START_YEAR, END_YEAR + 1):
            table[case][year] = {
                key: float(np.interp(year, xs, [r[key] for r in case_rows]))
                for key in ("broad_control_min", "broad_control_central", "broad_control_max")
            }
    return table


SCENARIOS = {
    "central": {"alliance_payoff": 0.0, "revision": 1.0, "shock": 1.0, "force_scale": 1.0},
    "weak_alliance_deterrence": {"alliance_payoff": -0.65, "revision": 1.0, "shock": 1.0, "force_scale": 1.0},
    "strong_alliance_deterrence": {"alliance_payoff": 0.65, "revision": 1.0, "shock": 1.0, "force_scale": 1.0},
    "low_evolutionary_inertia": {"alliance_payoff": 0.0, "revision": 1.7, "shock": 1.0, "force_scale": 1.0},
    "high_evolutionary_inertia": {"alliance_payoff": 0.0, "revision": 0.45, "shock": 1.0, "force_scale": 1.0},
    "no_local_shock_channel": {"alliance_payoff": 0.0, "revision": 1.0, "shock": 0.0, "force_scale": 1.0},
}


def campaign_vectors(campaign, year):
    # Alliance strategy order is noncombat, limited, delayed full, timely full.
    ordered = ("taiwan_alone", "limited", "delayed", "timely_full")
    central = np.asarray([campaign[c][year]["broad_control_central"] for c in ordered])
    low = np.asarray([campaign[c][year]["broad_control_min"] for c in ordered])
    high = np.asarray([campaign[c][year]["broad_control_max"] for c in ordered])
    return low, central, high


def evolve_simplex(x, payoff, revision, mutation):
    average = np.sum(x * payoff, axis=1, keepdims=True)
    next_x = x + revision * x * (payoff - average) + mutation * (1.0 / x.shape[1] - x)
    return normalize_simplex(next_x)


def summarize_distribution(values):
    return [float(v) for v in np.quantile(values, [0.10, 0.50, 0.90])]


def simulate_scenario(name, cfg, causal, reference, campaign, seed):
    rng = np.random.default_rng(seed)
    n = PATHS
    x_china = np.tile([0.34, 0.37, 0.12, 0.09, 0.08], (n, 1)).astype(float)
    x_alliance = np.tile([0.32, 0.29, 0.10, 0.29], (n, 1)).astype(float)
    x_taiwan = np.tile([0.31, 0.58, 0.11], (n, 1)).astype(float)
    x_china = normalize_simplex(x_china * rng.lognormal(0, 0.10, x_china.shape))
    x_alliance = normalize_simplex(x_alliance * rng.lognormal(0, 0.12, x_alliance.shape))
    x_taiwan = normalize_simplex(x_taiwan * rng.lognormal(0, 0.09, x_taiwan.shape))

    live = np.ones(n, dtype=bool)
    terminal = np.full(n, -1, dtype=np.int8)  # 0 force, 1 peace, 2 separation
    terminal_year = np.full(n, -1, dtype=np.int16)
    broad_control = np.zeros(n, dtype=bool)
    alliance_at_force = np.full(n, -1, dtype=np.int8)
    local_effect = causal["estimate_log_points"] * cfg["shock"]
    baseline_shock_probability = min(0.22, 6.0 / causal["months"] * 1.6)
    snapshots = []

    for year in range(START_YEAR, END_YEAR + 1):
        bilateral = reference["china_taiwan_theater_balance"][year]
        old_allied_balance = reference["china_allied_theater_balance"][year]
        linkage = reference["economic_linkage"][year]
        progress = (year - START_YEAR) / (END_YEAR - START_YEAR)
        identity_distance = np.clip(0.48 + 0.22 * progress + rng.normal(0, 0.035, n), 0.25, 0.90)
        closing_window = np.clip(0.30 + 0.45 * progress + rng.normal(0, 0.06, n), 0.0, 1.0)
        crisis_jump = rng.random(n) < baseline_shock_probability
        pressure_impulse = crisis_jump * local_effect

        low_campaign, central_campaign, high_campaign = campaign_vectors(campaign, year)
        expected_success = np.sum(x_alliance * central_campaign[None, :], axis=1)
        direct_alliance = x_alliance[:, 1] + x_alliance[:, 2] + x_alliance[:, 3]
        timely = x_alliance[:, 3]
        china_aggression = x_china[:, 2] + x_china[:, 3] + x_china[:, 4]
        taiwan_separation = x_taiwan[:, 2]
        taiwan_resilience = x_taiwan[:, 1]
        sanction_cost = np.clip(0.50 + 0.65 * direct_alliance + 0.35 * linkage, 0.0, 1.5)
        feasibility = np.tanh(math.log(max(bilateral, 1e-4)) / 2.5) - 0.70 * timely

        common_china = rng.normal(0, 0.08, (n, 1))
        payoff_china = np.column_stack((
            0.56 + 0.48 * linkage - 0.22 * identity_distance,
            0.48 + 0.42 * identity_distance - 0.24 * linkage - 0.18 * direct_alliance + 0.20 * pressure_impulse,
            0.10 + 0.68 * feasibility + 0.32 * identity_distance - 0.60 * sanction_cost + 0.18 * closing_window,
            -0.02 + 0.82 * feasibility + 0.42 * identity_distance - 0.76 * sanction_cost + 0.36 * closing_window,
            -0.36 + 2.10 * (expected_success - 0.40) + 0.58 * identity_distance
            + 0.62 * closing_window - 0.72 * sanction_cost + 0.22 * pressure_impulse,
        )) + common_china + rng.normal(0, 0.045, (n, 5))

        threat = np.clip(0.15 + 0.75 * china_aggression + 0.35 * pressure_impulse, 0.0, 1.5)
        nuclear_cost = np.clip(0.40 + 0.28 * progress + rng.normal(0, 0.05, n), 0.1, 0.9)
        alliance_shift = cfg["alliance_payoff"]
        payoff_alliance = np.column_stack((
            0.48 - 0.70 * threat,
            0.30 + 0.70 * threat - 0.22 * nuclear_cost + 0.22 * alliance_shift,
            0.12 + 0.92 * threat - 0.34 * nuclear_cost + 0.52 * alliance_shift,
            -0.02 + 1.14 * threat - 0.52 * nuclear_cost + 0.75 * alliance_shift,
        )) + rng.normal(0, 0.055, (n, 4))

        protection = x_alliance[:, 1] + 0.78 * x_alliance[:, 2] + 1.15 * x_alliance[:, 3]
        payoff_taiwan = np.column_stack((
            0.40 + 0.56 * linkage - 0.36 * threat + 0.16 * protection,
            0.42 + 0.58 * threat + 0.35 * protection - 0.22 * linkage,
            -0.12 + 0.92 * identity_distance + 0.42 * protection - 0.96 * threat,
        )) + rng.normal(0, 0.05, (n, 3))

        rate = 0.055 * cfg["revision"]
        x_china = evolve_simplex(x_china, payoff_china, rate, 0.005)
        x_alliance = evolve_simplex(x_alliance, payoff_alliance, rate, 0.006)
        x_taiwan = evolve_simplex(x_taiwan, payoff_taiwan, rate, 0.005)

        # Cause-specific hazards are separated from conditional campaign success.
        log_force = (
            -5.72 + 2.05 * x_china[:, 4] + 0.92 * x_china[:, 3] + 0.45 * x_china[:, 2]
            + 0.46 * feasibility + 0.44 * closing_window + 0.34 * pressure_impulse
            - 0.48 * direct_alliance - 0.26 * linkage + math.log(cfg["force_scale"])
        )
        log_peace = -6.15 + 1.55 * x_china[:, 0] + 1.15 * x_taiwan[:, 0] + 0.82 * linkage - 0.70 * threat
        log_separation = -6.28 + 2.05 * taiwan_separation + 0.90 * identity_distance + 0.42 * protection - 0.72 * threat
        hazards = np.column_stack((np.exp(log_force), np.exp(log_peace), np.exp(log_separation)))
        hazards = np.clip(hazards, 0.0, 0.12)
        total_hazard = hazards.sum(axis=1)
        exit_probability = 1.0 - np.exp(-total_hazard)
        draws = rng.random(n)
        exits = live & (draws < exit_probability)
        if np.any(exits):
            conditional = hazards[exits] / total_hazard[exits, None]
            cause_draw = rng.random(exits.sum())
            causes = (cause_draw[:, None] > np.cumsum(conditional, axis=1)).sum(axis=1).astype(np.int8)
            indices = np.flatnonzero(exits)
            terminal[indices] = causes
            terminal_year[indices] = year
            force_indices = indices[causes == 0]
            if len(force_indices):
                probs = x_alliance[force_indices]
                alliance_draw = rng.random(len(force_indices))
                regimes = (alliance_draw[:, None] > np.cumsum(probs, axis=1)).sum(axis=1)
                regimes = np.minimum(regimes, 3)
                alliance_at_force[force_indices] = regimes
                structural = rng.triangular(-1.0, 0.0, 1.0, len(force_indices))
                success_prob = central_campaign[regimes]
                left = structural < 0
                success_prob[left] += (-structural[left]) * (low_campaign[regimes[left]] - success_prob[left])
                success_prob[~left] += structural[~left] * (high_campaign[regimes[~left]] - success_prob[~left])
                broad_control[force_indices] = rng.random(len(force_indices)) < success_prob
            live[indices] = False

        if year in CHECKPOINTS:
            force = terminal == 0
            peace = terminal == 1
            separation = terminal == 2
            snapshot = {
                "year": year,
                "force_onset": float(force.mean()),
                "peaceful_settlement": float(peace.mean()),
                "separation": float(separation.mean()),
                "status_quo": float(live.mean()),
                "broad_control_after_force": float(broad_control.mean()),
                "conditional_broad_control_given_force": float(broad_control.sum() / max(force.sum(), 1)),
                "china_strategy_mean": dict(zip(CHINA_STRATEGIES, x_china[live].mean(axis=0) if live.any() else x_china.mean(axis=0))),
                "alliance_strategy_mean": dict(zip(ALLIANCE_STRATEGIES, x_alliance[live].mean(axis=0) if live.any() else x_alliance.mean(axis=0))),
                "taiwan_strategy_mean": dict(zip(TAIWAN_STRATEGIES, x_taiwan[live].mean(axis=0) if live.any() else x_taiwan.mean(axis=0))),
                "invasion_share_p10_p50_p90": summarize_distribution(x_china[live, 4] if live.any() else x_china[:, 4]),
                "timely_full_share_p10_p50_p90": summarize_distribution(x_alliance[live, 3] if live.any() else x_alliance[:, 3]),
                "v36_old_allied_balance_reference": old_allied_balance,
            }
            total = snapshot["force_onset"] + snapshot["peaceful_settlement"] + snapshot["separation"] + snapshot["status_quo"]
            assert abs(total - 1.0) < 1e-12
            snapshots.append(snapshot)

    force_indices = terminal == 0
    regime_distribution = {
        ALLIANCE_STRATEGIES[i]: float(np.mean(alliance_at_force[force_indices] == i)) if force_indices.any() else 0.0
        for i in range(4)
    }
    return {
        "scenario": name,
        "years": snapshots,
        "force_year_p10_p50_p90_conditional": summarize_distribution(terminal_year[force_indices]) if force_indices.any() else None,
        "alliance_regime_at_force": regime_distribution,
    }


def validate(results):
    failures = []
    for scenario in results:
        previous = {key: -1.0 for key in ("force_onset", "peaceful_settlement", "separation", "broad_control_after_force")}
        for row in scenario["years"]:
            for key in previous:
                if row[key] + 1e-12 < previous[key]:
                    failures.append(f"{scenario['scenario']} {key} is not monotone at {row['year']}")
                previous[key] = row[key]
            for group in ("china_strategy_mean", "alliance_strategy_mean", "taiwan_strategy_mean"):
                if abs(sum(row[group].values()) - 1.0) > 1e-8:
                    failures.append(f"{scenario['scenario']} {group} does not sum to one")
    central = next(r for r in results if r["scenario"] == "central")
    weak = next(r for r in results if r["scenario"] == "weak_alliance_deterrence")
    strong = next(r for r in results if r["scenario"] == "strong_alliance_deterrence")
    if not (weak["years"][-1]["broad_control_after_force"] > central["years"][-1]["broad_control_after_force"] > strong["years"][-1]["broad_control_after_force"]):
        failures.append("alliance-deterrence counterfactual ordering failed")
    if failures:
        raise AssertionError("; ".join(failures))
    return "TAIWAN_CAUSAL_EVOLUTION_V37_VERIFICATION: PASS"


def write_csv(results):
    path = OUT / "台海V3.7长期策略与终局.csv"
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "scenario", "year", "force_onset", "peaceful_settlement", "separation", "status_quo",
            "broad_control_after_force", "conditional_broad_control_given_force", "china_invasion_share",
            "alliance_timely_full_share", "taiwan_legal_separation_share",
        ])
        for scenario in results:
            for row in scenario["years"]:
                writer.writerow([
                    scenario["scenario"], row["year"], row["force_onset"], row["peaceful_settlement"],
                    row["separation"], row["status_quo"], row["broad_control_after_force"],
                    row["conditional_broad_control_given_force"], row["china_strategy_mean"]["invasion"],
                    row["alliance_strategy_mean"]["timely_full"], row["taiwan_strategy_mean"]["legal_separation"],
                ])
    return path


def write_causal_csv(causal):
    path = OUT / "台海V3.7因果局部效应.csv"
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["estimand", "estimate_log_points", "multiplicative_ratio", "bootstrap_p10", "bootstrap_p50", "bootstrap_p90", "treated_months", "months"])
        writer.writerow([
            causal["estimand"], causal["estimate_log_points"], causal["multiplicative_ratio"],
            *causal["year_block_bootstrap_p10_p50_p90"], causal["treated_months"], causal["months"],
        ])
    return path


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    OUT.mkdir(parents=True, exist_ok=True)
    causal = causal_local_effect()
    reference = load_reference_paths()
    campaign = load_campaign_table()
    results = []
    for offset, (name, cfg) in enumerate(SCENARIOS.items()):
        results.append(simulate_scenario(name, cfg, causal, reference, campaign, SEED + 1000 * offset))
    verification = validate(results)
    payload = {
        "model": "V3.7 causal local mechanism + three-population evolutionary game + cause-specific competing risks + V3.6 conditional campaign",
        "period": [START_YEAR, END_YEAR],
        "paths_per_scenario": PATHS,
        "seed": SEED,
        "causal_local_effect": causal,
        "identification_partition": {
            "estimated": "political-shock month to local pressure response, under stated causal assumptions",
            "evolved": "China, alliance and Taiwan mixed-strategy distributions",
            "structural_prior": "terminal cause-specific baseline hazards and payoff coefficients",
            "conditional_campaign": "V3.6 four-regime campaign envelopes after force onset",
        },
        "results": results,
        "verification": verification,
    }
    json_path = OUT / "台海V3.7因果_演化博弈_竞争风险_联合仿真摘要.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    csv_path = write_csv(results)
    causal_csv_path = write_causal_csv(causal)
    print(json.dumps({
        "causal": causal,
        "central": next(r for r in results if r["scenario"] == "central"),
        "verification": verification,
        "json": str(json_path),
        "csv": str(csv_path),
        "causal_csv": str(causal_csv_path),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
