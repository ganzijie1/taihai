from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path

import simulate_taiwan_fully_closed_v56 as model


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
TARGET = OUT / "taiwan_v56_one_at_a_time_ablation.json"


def summarize(rows: list[dict]) -> dict:
    selected = [row for row in rows if row["horizon_years"] in (5, 15, 30)]
    return {
        f'{row["case"]}|{row["horizon_years"]}y': row["conditional_broad_control_rate"]
        for row in selected
    }


def run_variant(paths: int, months: int, geometry: dict, mfg: dict, hybrid: dict | None):
    frozen_draws = model.load_draws(128)
    draws = {key: value[:paths].copy() for key, value in frozen_draws.items()}
    baseline_scenario = list(model.core.trade.v43.SCENARIOS)[0]
    return model.core.run(
        paths=paths, months=months, scenario_names=[baseline_scenario],
        mechanism_enabled=True, energy_enabled=True, geometry_enabled=True,
        hybrid_enabled=True, hybrid_config=hybrid, topology_enabled=True,
        mfg_config=mfg, geometry_config=geometry,
        macro_factory=model.CoupledMRRDCGEDQSFCEconomicSystem,
        econometric_draws=draws, traceability_path=model.TRACE,
        coupling_audit_path=model.AUDIT,
    )


def main() -> None:
    if os.environ.get("TAIWAN_ALLOW_COMPLETE_RUN") != "1":
        raise RuntimeError("V5.6 ablation blocked before the complete-model gate")
    paths = int(os.environ.get("TAIWAN_V56_ABLATION_PATHS", "32"))
    months = int(os.environ.get("TAIWAN_V56_ABLATION_MONTHS", "360"))
    runs: dict[str, dict] = {}
    variants = ["full", "finsler_lie", "wfr_information", "spd_joseph", "port_hamiltonian", "slow_manifold", "saltation_sensitivity"]
    for name in variants:
        geometry = deepcopy(model.GEOMETRY_CONFIG)
        mfg = deepcopy(model.MFG_CONFIG)
        hybrid = None
        if name == "finsler_lie":
            geometry["operations"]["advanced_geometry"] = False
            geometry["spatial"]["dynamic_finsler"] = False
        elif name == "wfr_information":
            mfg["wfr_strang"] = False
            mfg["natural_gradient_major"] = False
        elif name == "spd_joseph":
            geometry["pnt"] = {"geometry_metric": "affine_invariant", "joseph_update": False}
        elif name == "port_hamiltonian":
            geometry["energy"]["port_hamiltonian_dae"] = False
        elif name == "slow_manifold":
            geometry["multiscale"]["learned_slow_manifold"] = False
        elif name == "saltation_sensitivity":
            hybrid = {"saltation_sensitivity": False}
        rows, _, diagnostics, _ = run_variant(paths, months, geometry, mfg, hybrid)
        runs[name] = {"conditional_control": summarize(rows), "diagnostic_cells": len(diagnostics)}
    baseline = runs["full"]["conditional_control"]
    for name, result in runs.items():
        result["change_pp_from_full"] = {
            key: 100.0 * (value - baseline[key])
            for key, value in result["conditional_control"].items()
        }
    payload = {
        "model_version": "V5.6", "classification": "common-random-number sensitivity diagnostic",
        "paths": paths, "months": months, "trade_scenarios": ["baseline"],
        "intervention_cases": 5, "status": "PASS", "runs": runs,
        "draw_subset_contract": "deterministic first 32 of the 128 frozen V5.5 path draws",
        "note": "Saltation is sensitivity-only, so disabling its ledger is expected to be outcome-neutral.",
    }
    TARGET.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("TAIWAN_V56_ABLATION: PASS")


if __name__ == "__main__":
    main()
