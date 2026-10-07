from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

import simulate_dual_circulation_v44 as trade
import simulate_taiwan_full_coupled_v45 as v45
import simulate_taiwan_multidomain_v42 as battle
from simulate_four_party_financial_v45 import FourPartyFinancialSystem, PUBLIC_CALIBRATION, SOURCE_URLS
from wartime_social_contagion import WartimeSocialContagionSystem


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
OUT_CSV = OUT / "台海V4.6传播耦合_条件广泛控制率.csv"
OUT_JSON = OUT / "台海V4.6传播耦合_仿真摘要.json"
OUT_SOCIAL = OUT / "台海V4.6传播耦合_社会状态.json"
YEAR = 2030


def _hash(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def run(paths=600, months=360, scenario_names=None, include_ablation=True, social_features=None):
    battle.PATHS = paths
    battle.MONTHS = months
    battle.HORIZON_MONTHS = tuple(y * 12 for y in battle.HORIZON_YEARS if y * 12 <= months)
    base, v40_rows, terrain, sipri = battle.load_inputs()
    all_scenarios = list(trade.v43.SCENARIOS)
    scenarios = all_scenarios if scenario_names is None else scenario_names
    profiles = {
        name: trade.load_coupling_profile(
            name, paths, months, battle.SEED + 700_000 + all_scenarios.index(name) * 10_007
        )
        for name in scenarios
    }
    prewar = v45.prewar_state(paths, battle.SEED + 810_001)
    prewar_battle = {k: val for k, val in prewar.items() if k != "metadata"}
    baseline_name = next(name for name, code in trade.v43.SCENARIOS.items() if code == "baseline")
    conditional = {}
    rows = []
    social_summaries = {}
    ablation = []

    for case_index, case in enumerate(battle.CASES):
        cell_seed = battle.SEED + YEAR * 101 + case_index * 1009
        z = np.random.default_rng(cell_seed).normal(0.0, 1.0, paths)
        conditional[case] = {}
        for scenario_index, name in enumerate(scenarios):
            finance_seed = battle.SEED + 1_500_000 + case_index * 100_003 + all_scenarios.index(name) * 10_009
            social_seed = battle.SEED + 2_100_000 + case_index * 100_019 + all_scenarios.index(name) * 10_037
            finance = FourPartyFinancialSystem(
                paths, months, finance_seed, case, trade.v43.SCENARIOS[name]
            )
            social = WartimeSocialContagionSystem(
                paths, social_seed, prewar_battle["actor_support"], features=social_features
            )
            snapshots, _ = battle.simulate_mechanisms(
                np.random.default_rng(cell_seed), YEAR, case, "central", terrain, sipri,
                economy_profile=profiles[name], financial_system=finance,
                prewar_state=prewar_battle, social_system=social,
            )
            key = f"{case}|{trade.v43.SCENARIOS[name]}"
            social_summaries[key] = social.summary()
            conditional[case][name] = {}
            previous = np.zeros(paths)
            for horizon in battle.HORIZON_MONTHS:
                prior = battle.make_prior_paths(z, base[(YEAR, case)], v40_rows[(YEAR, case)], horizon)
                probability = battle.sigmoid(prior + snapshots[horizon]["score"])
                probability = np.maximum(probability, previous)
                previous = probability
                conditional[case][name][horizon] = probability
                summary = battle.summarize_probability(probability)
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

            if include_ablation and name == baseline_name:
                legacy_finance = FourPartyFinancialSystem(
                    paths, months, finance_seed, case, trade.v43.SCENARIOS[name]
                )
                legacy_snapshots, _ = battle.simulate_mechanisms(
                    np.random.default_rng(cell_seed), YEAR, case, "central", terrain, sipri,
                    economy_profile=profiles[name], financial_system=legacy_finance,
                    prewar_state=prewar_battle, social_system=None,
                )
                old_previous = np.zeros(paths)
                for horizon in battle.HORIZON_MONTHS:
                    prior = battle.make_prior_paths(z, base[(YEAR, case)], v40_rows[(YEAR, case)], horizon)
                    legacy_probability = np.maximum(
                        battle.sigmoid(prior + legacy_snapshots[horizon]["score"]), old_previous
                    )
                    old_previous = legacy_probability
                    current = conditional[case][name][horizon]
                    ablation.append({
                        "case": case,
                        "horizon_years": horizon // 12,
                        "with_hybrid_social_mean": float(np.mean(current)),
                        "legacy_graphon_mean": float(np.mean(legacy_probability)),
                        "difference_percentage_points": float(100.0 * np.mean(current - legacy_probability)),
                    })

    weights, mean_weights = v45.intervention_weights(paths, battle.SEED + 920_003)
    force_paths, force_row = v45.force_onset_paths(paths, battle.SEED + 930_007)
    aggregate = []
    for name in scenarios:
        for horizon in battle.HORIZON_MONTHS:
            matrix = np.column_stack([conditional[case][name][horizon] for case in battle.CASES])
            weighted = np.sum(weights * matrix, axis=1)
            unconditional = force_paths * weighted
            aggregate.append({
                "trade_scenario": trade.v43.SCENARIOS[name],
                "horizon_years": horizon // 12,
                "weighted_conditional_p10_p50_p90": v45.q(weighted),
                "weighted_conditional_mean": float(np.mean(weighted)),
                "force_onset_by_2030_p10_p50_p90": v45.q(force_paths),
                "unconditional_broad_control_p10_p50_p90": v45.q(unconditional),
                "unconditional_broad_control_mean": float(np.mean(unconditional)),
            })

    errors = []
    if any(not 0.0 <= row["probability"] <= 1.0 for row in rows):
        errors.append("conditional probability outside [0,1]")
    for case in battle.CASES:
        for name in scenarios:
            series = [conditional[case][name][h] for h in battle.HORIZON_MONTHS]
            if any(np.any(b + 1e-12 < a) for a, b in zip(series, series[1:])):
                errors.append(f"non-monotone cumulative probability: {case}|{name}")
    if not np.allclose(weights.sum(axis=1), 1.0, atol=1e-12):
        errors.append("intervention weights do not sum to one")
    max_social_error = max(
        (entry["max_population_conservation_error"] for entry in social_summaries.values()),
        default=0.0,
    )
    if max_social_error > 1e-10:
        errors.append("social population conservation failed")
    if errors:
        raise RuntimeError("; ".join(sorted(set(errors))))

    verification = {
        "status": "PASS",
        "version": "V4.6",
        "paths": paths,
        "months": months,
        "master_clock": "monthly campaign/finance/trade with four weekly social substeps",
        "same_path_coupling": True,
        "social_model": "age-stratified competing contagion + marked Hawkes + complex threshold + compound Levy marks",
        "social_parameter_status": "structural scenario priors; not empirically calibrated wartime coefficients",
        "social_population_conservation_max_error": max_social_error,
        "conditional_cases": list(battle.CASES),
        "trade_scenarios": {name: trade.v43.SCENARIOS[name] for name in scenarios},
        "intervention_weight_means": mean_weights,
        "prewar_metadata": prewar["metadata"],
        "force_onset_2030_source_row": force_row,
        "warning": (
            "Conditional probabilities assume conflict onset. Unconditional values multiply the pathwise "
            "intervention-weighted campaign outcome by the V3.7 force-onset distribution. Social transmission "
            "coefficients are transparent scenario priors and should not be read as measured forecasts."
        ),
    }
    return rows, aggregate, social_summaries, ablation, verification


def write(rows, aggregate, social_summaries, ablation, verification):
    OUT.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    OUT_SOCIAL.write_text(json.dumps(social_summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    verification["result_sha256"] = _hash(OUT_CSV)
    verification["social_state_sha256"] = _hash(OUT_SOCIAL)
    OUT_JSON.write_text(json.dumps({
        "verification": verification,
        "aggregate": aggregate,
        "conditional_results": rows,
        "baseline_social_ablation": ablation,
        "public_financial_calibration": PUBLIC_CALIBRATION,
        "financial_source_urls": SOURCE_URLS,
    }, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    rows, aggregate, social, ablation, verification = run()
    write(rows, aggregate, social, ablation, verification)
    print("TAIWAN_FULL_COUPLED_V46_VERIFICATION: PASS")
    print(json.dumps(verification, ensure_ascii=False))
    for row in aggregate:
        if row["horizon_years"] in (5, 15, 30):
            print(json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()
