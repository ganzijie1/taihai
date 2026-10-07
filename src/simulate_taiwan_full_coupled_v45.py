from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

import simulate_dual_circulation_v44 as trade
import simulate_taiwan_multidomain_v42 as battle
from simulate_four_party_financial_v45 import FourPartyFinancialSystem, PUBLIC_CALIBRATION, SOURCE_URLS


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
OUT_CSV = OUT / "台海V4.5全模块同路径耦合_条件广泛控制率.csv"
OUT_JSON = OUT / "台海V4.5全模块同路径耦合_仿真摘要.json"
OUT_FINANCE = OUT / "台海V4.5四方金融状态_完整摘要.json"
YEAR = 2030


def q(values):
    return [float(x) for x in np.quantile(values, (0.10, 0.50, 0.90))]


def load_json_named(pattern: str):
    matches = list(OUT.glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one {pattern}, found {len(matches)}")
    return json.loads(matches[0].read_text(encoding="utf-8"))


def interpolate_quantiles(rng, quantile_triplet, paths):
    lo, med, hi = quantile_triplet
    sigma = max((hi - lo) / 2.563, 1e-4)
    return np.clip(med + sigma * rng.normal(0.0, 1.0, paths), 0.001, 1.50)


def prewar_state(paths: int, seed: int):
    rng = np.random.default_rng(seed)
    opinion = load_json_named("*V3.9*摘要.json")
    row = next(x for x in opinion["results"] if x["year"] == YEAR)
    direct = row["direct_war_support_p10_p50_p90"]
    actor_support = np.column_stack((
        interpolate_quantiles(rng, direct["china"], paths),
        interpolate_quantiles(rng, direct["taiwan"], paths),
        interpolate_quantiles(rng, direct["united_states"], paths),
        interpolate_quantiles(rng, direct["japan"], paths),
    ))
    readiness = interpolate_quantiles(
        rng, row["taiwan_prewar_readiness_multiplier_p10_p50_p90"], paths
    )
    arms_support_us = interpolate_quantiles(
        rng, row["arms_or_indirect_support_p10_p50_p90"]["united_states"], paths
    )
    arms_support_japan = interpolate_quantiles(
        rng, row["arms_or_indirect_support_p10_p50_p90"]["japan"], paths
    )

    leader = load_json_named("*领导突变*摘要.json")
    baseline = next(x for x in leader["results"] if x["scenario"] == "baseline_resilience")
    execution = baseline["leader_process"]["execution"]
    entered = baseline["entered_leader_chain"]
    common = rng.random(paths) < entered
    actor_scale = np.array([1.0, 1.25, 0.55, 0.70])[None, :]
    leader_event = common[:, None] & (rng.random((paths, 4)) < execution * actor_scale / max(entered, 1e-6))
    severity = rng.uniform(0.30, 0.65, (paths, 4))
    continuity = np.where(leader_event, 1.0 - severity, 1.0)
    return {
        "actor_support": actor_support,
        "taiwan_readiness": readiness,
        "arms_support_us": arms_support_us,
        "arms_support_japan": arms_support_japan,
        "leader_continuity": continuity,
        "metadata": {
            "opinion_source": "V3.9 path-distribution summary for 2030",
            "leader_source": "baseline-resilience finite leader-jump process",
            "leader_chain_entry_probability": entered,
            "leader_execution_probability": execution,
        },
    }


def intervention_weights(paths: int, seed: int):
    rng = np.random.default_rng(seed)
    opinion = load_json_named("*V3.9*摘要.json")
    row = next(x for x in opinion["results"] if x["year"] == YEAR)
    old = row["alliance_case_probability_mean"]
    mean = np.array([
        old["timely_full"],
        max(old["limited"] - 0.061, 0.02),
        old["delayed"],
        0.061,
        old["taiwan_alone"],
    ])
    mean /= mean.sum()
    concentration = 85.0
    draws = rng.dirichlet(mean * concentration, paths)
    return draws, dict(zip(battle.CASES, mean.tolist()))


def force_onset_paths(paths: int, seed: int):
    rng = np.random.default_rng(seed)
    evolution = load_json_named("*V3.7*联合仿真摘要.json")
    central = next(x for x in evolution["results"] if x["scenario"] == "central")
    year = next(x for x in central["years"] if x["year"] == YEAR)
    mean = year["force_onset"]
    logits = math.log(mean / (1.0 - mean)) + rng.normal(0.0, 0.25, paths)
    return 1.0 / (1.0 + np.exp(-logits)), year


def run(paths: int = 600, months: int = 360):
    battle.PATHS = paths
    battle.MONTHS = months
    battle.HORIZON_MONTHS = tuple(y * 12 for y in battle.HORIZON_YEARS if y * 12 <= months)
    base, v40_rows, terrain, sipri = battle.load_inputs()
    scenarios = list(trade.v43.SCENARIOS)
    profiles = {
        name: trade.load_coupling_profile(
            name, paths, months, battle.SEED + 700_000 + i * 10_007
        )
        for i, name in enumerate(scenarios)
    }
    prewar = prewar_state(paths, battle.SEED + 810_001)
    prewar_for_battle = {k: v for k, v in prewar.items() if k != "metadata"}
    conditional_paths = {}
    rows = []
    financial_summaries = {}
    reference_differences = []
    baseline_name = next(name for name, code in trade.v43.SCENARIOS.items() if code == "baseline")

    for case_index, case in enumerate(battle.CASES):
        cell_seed = battle.SEED + YEAR * 101 + case_index * 1009
        prior_rng = np.random.default_rng(cell_seed)
        z_standard = prior_rng.normal(0.0, 1.0, paths)
        reference, _ = battle.simulate_mechanisms(
            np.random.default_rng(cell_seed), YEAR, case, "central", terrain, sipri
        )
        zero_shock, _ = battle.simulate_mechanisms(
            np.random.default_rng(cell_seed), YEAR, case, "central", terrain, sipri,
            economy_profile=profiles[baseline_name],
        )
        for horizon in battle.HORIZON_MONTHS:
            reference_differences.append(float(np.max(np.abs(
                reference[horizon]["score"] - zero_shock[horizon]["score"]
            ))))
        conditional_paths[case] = {}
        for scenario_index, name in enumerate(scenarios):
            finance = FourPartyFinancialSystem(
                paths, months,
                battle.SEED + 1_500_000 + case_index * 100_003 + scenario_index * 10_009,
                case, trade.v43.SCENARIOS[name],
            )
            snapshots, _ = battle.simulate_mechanisms(
                np.random.default_rng(cell_seed), YEAR, case, "central", terrain, sipri,
                economy_profile=profiles[name], financial_system=finance,
                prewar_state=prewar_for_battle,
            )
            financial_summaries[f"{case}|{trade.v43.SCENARIOS[name]}"] = finance.summary()
            conditional_paths[case][name] = {}
            previous = np.zeros(paths)
            for horizon in battle.HORIZON_MONTHS:
                prior_log_odds = battle.make_prior_paths(
                    z_standard, base[(YEAR, case)], v40_rows[(YEAR, case)], horizon
                )
                probability_paths = battle.sigmoid(prior_log_odds + snapshots[horizon]["score"])
                probability_paths = np.maximum(probability_paths, previous)
                previous = probability_paths
                conditional_paths[case][name][horizon] = probability_paths
                summary = battle.summarize_probability(probability_paths)
                rows.append({
                    "conflict_year": YEAR,
                    "horizon_years": horizon // 12,
                    "case": case,
                    "trade_scenario": trade.v43.SCENARIOS[name],
                    "probability": summary["probability"],
                    "p10": summary["path_probability_p10_p50_p90"][0],
                    "p50": summary["path_probability_p10_p50_p90"][1],
                    "p90": summary["path_probability_p10_p50_p90"][2],
                    "cvar99": summary["cvar99_path_probability"],
                    "bridgehead_share": snapshots[horizon]["diagnostics"]["mechanical_bridgehead_share"],
                    "majority_control_share": snapshots[horizon]["diagnostics"]["mechanical_majority_control_share"],
                })
    weights, mean_weights = intervention_weights(paths, battle.SEED + 920_003)
    force_paths, force_row = force_onset_paths(paths, battle.SEED + 930_007)
    aggregate = []
    for name in scenarios:
        for horizon in battle.HORIZON_MONTHS:
            matrix = np.column_stack([
                conditional_paths[case][name][horizon] for case in battle.CASES
            ])
            weighted = np.sum(weights * matrix, axis=1)
            unconditional = force_paths * weighted
            aggregate.append({
                "trade_scenario": trade.v43.SCENARIOS[name],
                "horizon_years": horizon // 12,
                "weighted_conditional_p10_p50_p90": q(weighted),
                "weighted_conditional_mean": float(np.mean(weighted)),
                "force_onset_by_2030_p10_p50_p90": q(force_paths),
                "unconditional_broad_control_p10_p50_p90": q(unconditional),
                "unconditional_broad_control_mean": float(np.mean(unconditional)),
            })

    errors = []
    if max(reference_differences, default=0.0) > 1e-12:
        errors.append("zero-shock regression failed")
    for row in rows:
        if not 0.0 <= row["probability"] <= 1.0:
            errors.append("conditional probability outside [0,1]")
    for case in battle.CASES:
        for name in scenarios:
            series = [conditional_paths[case][name][h] for h in battle.HORIZON_MONTHS]
            if any(np.any(b + 1e-12 < a) for a, b in zip(series, series[1:])):
                errors.append(f"non-monotone cumulative probability: {case}|{name}")
    if not np.allclose(weights.sum(axis=1), 1.0, atol=1e-12):
        errors.append("intervention weights do not sum to one")
    for finance in financial_summaries.values():
        for horizon, actor_rows in finance.items():
            if horizon == "cumulative":
                continue
            for actor, metrics in actor_rows.items():
                for key, values in metrics.items():
                    if isinstance(values, list) and not np.all(np.isfinite(values)):
                        errors.append(f"non-finite financial metric: {horizon}|{actor}|{key}")
    if errors:
        raise RuntimeError("; ".join(sorted(set(errors))))

    verification = {
        "status": "PASS",
        "paths": paths,
        "months": months,
        "master_clock": "monthly after conflict onset; annual competing-risk calibration before onset",
        "same_path_coupling": True,
        "conditional_cases": list(battle.CASES),
        "trade_scenarios": {name: trade.v43.SCENARIOS[name] for name in scenarios},
        "intervention_weight_means": mean_weights,
        "prewar_metadata": prewar["metadata"],
        "force_onset_2030_source_row": force_row,
        "zero_shock_reference_max_abs_difference": max(reference_differences, default=0.0),
        "warning": (
            "Conditional campaign probabilities assume conflict onset. Unconditional values multiply the "
            "pathwise intervention-weighted campaign result by the V3.7 force-onset distribution. "
            "Unobserved wartime behavioral coefficients remain scenario priors, not measured facts."
        ),
    }
    return rows, aggregate, financial_summaries, verification


def file_hash(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def write(rows, aggregate, financial_summaries, verification):
    OUT.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    OUT_FINANCE.write_text(json.dumps({
        "public_calibration": PUBLIC_CALIBRATION,
        "source_urls": SOURCE_URLS,
        "results": financial_summaries,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    verification["result_sha256"] = file_hash(OUT_CSV)
    verification["financial_summary_sha256"] = file_hash(OUT_FINANCE)
    OUT_JSON.write_text(json.dumps({
        "verification": verification,
        "aggregate": aggregate,
        "conditional_results": rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    rows, aggregate, finance, verification = run()
    write(rows, aggregate, finance, verification)
    print("TAIWAN_FULL_COUPLED_V45_VERIFICATION: PASS")
    print(json.dumps(verification, ensure_ascii=False))
    for row in aggregate:
        if row["horizon_years"] in (15, 30):
            print(row["trade_scenario"], row["horizon_years"], row["weighted_conditional_mean"])


if __name__ == "__main__":
    main()
