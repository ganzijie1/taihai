from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

import simulate_dual_circulation_v44 as trade
import simulate_taiwan_full_coupled_v45 as full
import simulate_taiwan_multidomain_v42 as battle
from simulate_four_party_financial_v45 import FourPartyFinancialSystem


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "台海V4.5模块消融.json"


CONFIGS = {
    "all_modules": {},
    "without_dynamic_alliance": {"module_flags": {"stackelberg_alliance": False}},
    "without_optimal_transport": {"module_flags": {"optimal_transport": False}},
    "without_graphon_social": {"module_flags": {"graphon_social": False}},
    "without_principal_agent": {"module_flags": {"principal_agent": False}},
    "without_finance": {"finance": False},
    "without_trade": {"trade": False},
    "without_prewar_state": {"prewar": False},
}


def run(paths: int = 120, months: int = 360):
    battle.PATHS = paths
    battle.MONTHS = months
    battle.HORIZON_MONTHS = tuple(y * 12 for y in battle.HORIZON_YEARS if y * 12 <= months)
    base, v40_rows, terrain, sipri = battle.load_inputs()
    scenario_name = next(name for name, code in trade.v43.SCENARIOS.items() if code == "combined")
    baseline_name = next(name for name, code in trade.v43.SCENARIOS.items() if code == "baseline")
    profiles = {
        scenario_name: trade.load_coupling_profile(scenario_name, paths, months, battle.SEED + 710_003),
        baseline_name: trade.load_coupling_profile(baseline_name, paths, months, battle.SEED + 710_003),
    }
    prewar = full.prewar_state(paths, battle.SEED + 810_001)
    prewar_state = {key: value for key, value in prewar.items() if key != "metadata"}
    weights, mean_weights = full.intervention_weights(paths, battle.SEED + 920_003)
    results = {}

    for config_index, (config_name, config) in enumerate(CONFIGS.items()):
        case_paths = {}
        diagnostics = {}
        for case_index, case in enumerate(battle.CASES):
            cell_seed = battle.SEED + full.YEAR * 101 + case_index * 1009
            prior_rng = np.random.default_rng(cell_seed)
            z_standard = prior_rng.normal(0.0, 1.0, paths)
            profile_name = scenario_name if config.get("trade", True) else baseline_name
            finance = None
            if config.get("finance", True):
                finance = FourPartyFinancialSystem(
                    paths,
                    months,
                    battle.SEED + 1_600_000 + case_index * 100_003,
                    case,
                    trade.v43.SCENARIOS[scenario_name],
                )
            snapshots, _ = battle.simulate_mechanisms(
                np.random.default_rng(cell_seed),
                full.YEAR,
                case,
                "central",
                terrain,
                sipri,
                economy_profile=profiles[profile_name],
                financial_system=finance,
                prewar_state=prewar_state if config.get("prewar", True) else None,
                module_flags=config.get("module_flags"),
            )
            previous = np.zeros(paths)
            case_paths[case] = {}
            for horizon in battle.HORIZON_MONTHS:
                prior_log_odds = battle.make_prior_paths(
                    z_standard, base[(full.YEAR, case)], v40_rows[(full.YEAR, case)], horizon
                )
                probability = battle.sigmoid(prior_log_odds + snapshots[horizon]["score"])
                probability = np.maximum(probability, previous)
                previous = probability
                case_paths[case][horizon] = probability
            diagnostics[case] = snapshots[max(battle.HORIZON_MONTHS)]["diagnostics"]

        horizons = {}
        for horizon in battle.HORIZON_MONTHS:
            matrix = np.column_stack([case_paths[case][horizon] for case in battle.CASES])
            weighted = np.sum(weights * matrix, axis=1)
            horizons[str(horizon // 12)] = {
                "conditional_mean": float(np.mean(weighted)),
                "p10_p50_p90": full.q(weighted),
            }
        results[config_name] = {"horizons_years": horizons, "terminal_diagnostics": diagnostics}

    baseline = results["all_modules"]["horizons_years"]
    for name, row in results.items():
        row["delta_from_all_modules"] = {
            year: row["horizons_years"][year]["conditional_mean"] - baseline[year]["conditional_mean"]
            for year in baseline
        }
    return {
        "verification": {
            "status": "PASS",
            "paths": paths,
            "months": months,
            "scenario": "combined",
            "same_random_numbers": True,
            "intervention_weight_means": mean_weights,
            "interpretation": "Ablations are structural sensitivity tests, not causal treatment effects.",
        },
        "results": results,
    }


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    payload = run()
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("TAIWAN_V45_MODULE_ABLATION: PASS")
    for name, result in payload["results"].items():
        print(name, result["delta_from_all_modules"])


if __name__ == "__main__":
    main()
