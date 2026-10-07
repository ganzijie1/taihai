from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from continuous_multipop_mfg_v47 import ContinuousMultiPopulationMFG
from dynamic_macro_equilibrium_v47 import CoupledMacroEconomicSystem
from dynamic_mechanism_design_v48 import RobustDynamicMechanismDesigner
from four_party_energy_network_v49 import FourPartyEnergyNetwork
from operational_constraints_v47 import DynamicGeospatialPNTSystem, DynamicOperationalConstraintSystem
from rolling_differential_game_v47 import RollingHorizonDifferentialGame
from simulate_four_party_financial_v45 import FourPartyFinancialSystem
from spatial_control_network_v47 import SpatialControlNetwork
from wartime_social_contagion import WartimeSocialContagionSystem
import simulate_taiwan_multidomain_v42 as battle


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "taiwan_v48_mechanism_ablation.json"
PATHS = 24
MONTHS = 120
SEED = 771_901
SCENARIO = "central"
CASE = "limited"


def run(disabled: str | None, energy_enabled: bool = False) -> dict:
    battle.PATHS = PATHS
    battle.MONTHS = MONTHS
    battle.HORIZON_MONTHS = (60, 120)
    _, _, terrain, sipri = battle.load_inputs()
    support = np.repeat(np.array([[0.62, 0.58, 0.53, 0.55]]), PATHS, axis=0)
    prewar = {
        "actor_support": support,
        "taiwan_readiness": np.ones(PATHS),
        "leader_continuity": np.full((PATHS, 4), 0.88),
        "economic_capital": np.ones((PATHS, 4)),
        "military_capital": np.ones((PATHS, 4)),
    }
    finance = FourPartyFinancialSystem(PATHS, MONTHS, SEED + 1, CASE, "baseline")
    social = WartimeSocialContagionSystem(PATHS, SEED + 2, support)
    mfg = ContinuousMultiPopulationMFG(PATHS, SEED + 3, support)
    macro = CoupledMacroEconomicSystem(PATHS, SEED + 4, "baseline")
    operations = DynamicOperationalConstraintSystem(PATHS, SEED + 5, terrain)
    pnt = DynamicGeospatialPNTSystem(PATHS, terrain)
    strategy = RollingHorizonDifferentialGame(PATHS, SEED + 6)
    spatial = SpatialControlNetwork(PATHS, SEED + 7, terrain)
    flags = {disabled: False} if disabled is not None else None
    mechanism = RobustDynamicMechanismDesigner(PATHS, SEED + 8, flags)
    energy = FourPartyEnergyNetwork(PATHS, SEED + 9) if energy_enabled else None
    snapshots, _ = battle.simulate_mechanisms(
        np.random.default_rng(SEED), 2030, CASE, SCENARIO, terrain, sipri,
        financial_system=finance, prewar_state=prewar, social_system=social,
        macro_system=macro, mfg_system=mfg, operations_system=operations,
        pnt_system=pnt, strategy_system=strategy, spatial_system=spatial,
        mechanism_system=mechanism,
        energy_system=energy,
    )
    snap = snapshots[MONTHS]
    diagnostic = snap["diagnostics"]
    return {
        "disabled": disabled or "none",
        "broad_control_weighted": float(np.average(
            snap["broad_control_paths"], weights=snap["importance_weight_paths"]
        )),
        "land_control_median": diagnostic["terminal_land_control_p10_p50_p90"][1],
        "population_control_median": diagnostic["population_control_p10_p50_p90"][1],
        "organized_defense_median": diagnostic["organized_defense_p10_p50_p90"][1],
        "china_min_stock_median": diagnostic["china_min_stock_p10_p50_p90"][1],
        "defender_min_stock_median": diagnostic["defender_min_stock_p10_p50_p90"][1],
        "mechanism_diagnostics": mechanism.diagnostics(),
        "energy_diagnostics": energy.diagnostics() if energy is not None else None,
    }


def main() -> None:
    audit = json.loads(
        (ROOT / "outputs" / "taiwan_v48_static_coupling_audit.json").read_text(encoding="utf-8")
    )
    if audit.get("status") != "PASS":
        raise RuntimeError("V4.8 ablation blocked by static audit")
    rows = [run(None)]
    for module in RobustDynamicMechanismDesigner.MODULES:
        rows.append(run(module))
    baseline = rows[0]
    for row in rows[1:]:
        row["delta_vs_all_on"] = {
            key: row[key] - baseline[key]
            for key in (
                "broad_control_weighted", "land_control_median",
                "population_control_median", "organized_defense_median",
                "china_min_stock_median", "defender_min_stock_median",
            )
        }
    changed = {
        row["disabled"]: any(abs(value) > 1e-10 for value in row["delta_vs_all_on"].values())
        for row in rows[1:]
    }
    if not all(changed.values()):
        raise RuntimeError(f"mechanism ablation has inactive campaign channel: {changed}")
    OUT.write_text(json.dumps({
        "model_version": "V4.8",
        "classification": "dynamic validation, not the formal probability estimate",
        "paths": PATHS,
        "months": MONTHS,
        "common_random_numbers": True,
        "all_modules_changed_campaign_state": changed,
        "runs": rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print("TAIWAN_V48_MECHANISM_ABLATION: PASS")


if __name__ == "__main__":
    main()
