from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from continuous_multipop_mfg_v47 import ContinuousMultiPopulationMFG
from dynamic_macro_equilibrium_v47 import CoupledMacroEconomicSystem
from dynamic_mechanism_design_v48 import RobustDynamicMechanismDesigner
from four_party_energy_network_v49 import FourPartyEnergyNetwork
from geometric_multiscale_v50 import GeometricMultiscaleCoupler
from operational_constraints_v47 import DynamicGeospatialPNTSystem, DynamicOperationalConstraintSystem
from rolling_differential_game_v47 import RollingHorizonDifferentialGame
from simulate_four_party_financial_v45 import FourPartyFinancialSystem
from spatial_control_network_v47 import SpatialControlNetwork
from wartime_social_contagion import WartimeSocialContagionSystem
import simulate_taiwan_multidomain_v42 as battle


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "taiwan_v50_geometry_ablation.json"
PATHS = 16
MONTHS = 60
SEED = 880_501
CASE = "limited"
COMPONENTS = ("fisher_rao", "uot_finsler", "spd_pnt", "spatial_finsler", "hodge_port", "multiscale")


def run(disabled: str | None) -> dict:
    all_off = disabled == "all_geometry"
    enabled = {name: not all_off and disabled != name for name in COMPONENTS}
    battle.PATHS = PATHS
    battle.MONTHS = MONTHS
    battle.HORIZON_MONTHS = (60,)
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
    mfg = ContinuousMultiPopulationMFG(
        PATHS, SEED + 3, support, geometry_enabled=enabled["fisher_rao"]
    )
    macro = CoupledMacroEconomicSystem(PATHS, SEED + 4, "baseline")
    operations = DynamicOperationalConstraintSystem(
        PATHS, SEED + 5, terrain, geometry_enabled=enabled["uot_finsler"]
    )
    pnt = DynamicGeospatialPNTSystem(
        PATHS, terrain, geometry_enabled=enabled["spd_pnt"]
    )
    strategy = RollingHorizonDifferentialGame(PATHS, SEED + 6)
    spatial = SpatialControlNetwork(
        PATHS, SEED + 7, terrain, geometry_enabled=enabled["spatial_finsler"]
    )
    mechanism = RobustDynamicMechanismDesigner(PATHS, SEED + 8)
    energy = FourPartyEnergyNetwork(
        PATHS, SEED + 9, geometry_enabled=enabled["hodge_port"]
    )
    geometry = GeometricMultiscaleCoupler(PATHS) if enabled["multiscale"] else None
    snapshots, _ = battle.simulate_mechanisms(
        np.random.default_rng(SEED), 2030, CASE, "central", terrain, sipri,
        financial_system=finance, prewar_state=prewar, social_system=social,
        macro_system=macro, mfg_system=mfg, operations_system=operations,
        pnt_system=pnt, strategy_system=strategy, spatial_system=spatial,
        mechanism_system=mechanism, energy_system=energy, geometry_system=geometry,
    )
    snap = snapshots[MONTHS]
    diagnostic = snap["diagnostics"]
    return {
        "disabled": disabled or "none",
        "land_control_median": diagnostic["terminal_land_control_p10_p50_p90"][1],
        "population_control_median": diagnostic["population_control_p10_p50_p90"][1],
        "organized_defense_median": diagnostic["organized_defense_p10_p50_p90"][1],
        "china_min_stock_median": diagnostic["china_min_stock_p10_p50_p90"][1],
        "defender_min_stock_median": diagnostic["defender_min_stock_p10_p50_p90"][1],
        "mfg_support_mean": float(np.mean(mfg.density * mfg.x[None, None, None, :]) * mfg.dx),
        "pnt_integrity_mean": float(np.mean(pnt.integrity)),
        "energy_availability_mean": float(np.mean(energy.current().availability)),
        "macro_gdp_mean": float(np.mean(macro.current().gdp_factor)),
        "geometry_consistency_mean": (
            float(np.mean(geometry.current().consistency)) if geometry is not None else 1.0
        ),
    }


def main() -> None:
    audit = json.loads(
        (ROOT / "outputs" / "taiwan_v50_static_coupling_audit.json").read_text(encoding="utf-8")
    )
    if audit.get("status") != "PASS":
        raise RuntimeError("V5.0 geometry ablation blocked by static audit")
    rows = [run(None)] + [run(name) for name in COMPONENTS] + [run("all_geometry")]
    baseline = rows[0]
    metrics = tuple(key for key in baseline if key not in {"disabled"})
    for row in rows[1:]:
        row["delta_vs_all_on"] = {key: row[key] - baseline[key] for key in metrics}
    changed = {
        row["disabled"]: any(abs(value) > 1.0e-10 for value in row["delta_vs_all_on"].values())
        for row in rows[1:]
    }
    if not all(changed.values()):
        raise RuntimeError(f"geometry ablation has inactive downstream channel: {changed}")
    OUT.write_text(json.dumps({
        "model_version": "V5.0",
        "classification": "same-seed dynamic validation, not a formal probability estimate",
        "paths": PATHS,
        "months": MONTHS,
        "common_random_numbers": True,
        "all_components_changed_same_path_state": changed,
        "runs": rows,
        "status": "PASS",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print("TAIWAN_V50_GEOMETRY_ABLATION: PASS")


if __name__ == "__main__":
    main()
