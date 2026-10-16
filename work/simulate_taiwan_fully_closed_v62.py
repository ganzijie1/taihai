from __future__ import annotations

import csv
import hashlib
import json
import os
from functools import partial
from pathlib import Path

import numpy as np

import simulate_taiwan_fully_closed_v61 as v61
from continuous_multipop_mfg_v62 import ContinuousMultiPopulationMFGV62
from sectoral_economy_v62 import CoupledSectoralEconomicSystemV62


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
TRACE = OUT / "taiwan_v62_traceability.json"
AUDIT = OUT / "taiwan_v62_static_coupling_audit.json"
FREEZE = OUT / "taiwan_v62_frozen_specification.json"
OUT_CSV = OUT / "taiwan_v62_results.csv"
OUT_JSON = OUT / "taiwan_v62_simulation.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def validate_v62_diagnostics(diagnostics: dict) -> None:
    v61.validate_sector_diagnostics(diagnostics)
    errors = []
    for cell, item in diagnostics.items():
        sector = item["macro"].get("three_sector_physical_economy", {})
        structural = sector.get("v62_structural_transformation")
        mfg = item.get("mfg", {})
        transport = (mfg.get("partial_observation") or {}).get(
            "v62_dynamic_transport"
        )
        joint = mfg.get("v62_joint_state_action_wfr")
        if structural is None:
            errors.append(f"{cell}: missing V6.2 structural transformation")
        elif not structural.get("employment_and_value_added_separated"):
            errors.append(f"{cell}: employment/value-added not separated")
        if transport is None:
            errors.append(f"{cell}: missing dynamic causal OT")
        elif transport.get("future_information_used"):
            errors.append(f"{cell}: noncausal information detected")
        if joint is None:
            errors.append(f"{cell}: missing joint state-action WFR")
        if transport and transport["max_bridge_transition_residual"] > 1.0e-10:
            errors.append(f"{cell}: bridge transition is not stochastic")
        if joint and joint["max_joint_action_mass_residual"] > 1.0e-12:
            errors.append(f"{cell}: joint action mass bound")
    if errors:
        raise RuntimeError("V6.2 runtime validation failed: " + "; ".join(errors))


def run(paths: int = 128, months: int = 360,
        scenario_names: list[str] | None = None,
        case_names: list[str] | None = None, *,
        sector_flags: dict | None = None,
        stochastic_geometry_factory=None,
        transport_symplectic_factory=None,
        rare_event_factory=None):
    draws = v61.load_frozen_draws(paths)
    macro_factory = (
        CoupledSectoralEconomicSystemV62
        if not sector_flags else partial(
            CoupledSectoralEconomicSystemV62, **sector_flags
        )
    )
    core = v61.v60.v59.v58.core
    original_mfg = core.ContinuousMultiPopulationMFG
    core.ContinuousMultiPopulationMFG = ContinuousMultiPopulationMFGV62
    try:
        result = core.run(
            paths=paths,
            months=months,
            scenario_names=scenario_names,
            case_names=case_names,
            mechanism_enabled=True,
            energy_enabled=True,
            geometry_enabled=True,
            hybrid_enabled=True,
            topology_enabled=True,
            mfg_config=v61.v60.MFG_CONFIG,
            geometry_config=v61.v60.GEOMETRY_CONFIG,
            macro_factory=macro_factory,
            political_factory=v61.v60.v59.v58.WartimeCollectiveActionRegimeSystem,
            econometric_draws=draws,
            stochastic_geometry_factory=stochastic_geometry_factory,
            transport_symplectic_factory=transport_symplectic_factory,
            rare_event_factory=rare_event_factory,
            traceability_path=TRACE,
            coupling_audit_path=AUDIT,
        )
    finally:
        core.ContinuousMultiPopulationMFG = original_mfg
    validate_v62_diagnostics(result[2])
    return result


def write(rows, aggregate, diagnostics, verification) -> None:
    with OUT_CSV.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    verification["model_version"] = "V6.2"
    verification["runtime_validation"] = "TAIWAN_V62_RUNTIME_VALIDATION: PASS"
    verification["result_sha256"] = sha256(OUT_CSV)
    verification["v62_upgrade"] = {
        "employment_value_added_separation": True,
        "unbalanced_optimal_transport_labor": True,
        "nonhomothetic_structural_demand": True,
        "kaldor_verdoorn_and_baumol": True,
        "dynamic_pathwise_fgw": True,
        "causal_lagged_schrodinger_bridge": True,
        "online_ot_dro": True,
        "joint_state_action_wfr": True,
        "same_path_bidirectional_feedback": True,
        "no_posthoc_probability_adjustment": True,
        "frozen_specification_sha256": sha256(FREEZE),
    }
    OUT_JSON.write_text(json.dumps({
        "verification": verification,
        "aggregate": aggregate,
        "conditional": rows,
        "module_diagnostics": diagnostics,
    }, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    if os.environ.get("TAIWAN_ALLOW_COMPLETE_RUN") != "1":
        raise RuntimeError(
            "V6.2 outcome simulation blocked until the complete-model gate passes"
        )
    frozen = json.loads(FREEZE.read_text(encoding="utf-8"))
    if frozen.get("status") != "FROZEN":
        raise RuntimeError("V6.2 frozen specification is missing or invalid")
    paths = int(os.environ.get("TAIWAN_V62_PATHS", "128"))
    months = int(os.environ.get("TAIWAN_V62_MONTHS", "360"))
    rows, aggregate, diagnostics, verification = run(paths=paths, months=months)
    write(rows, aggregate, diagnostics, verification)
    print("TAIWAN_V62_COMPLETE_RUN: PASS")


if __name__ == "__main__":
    main()
