from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path

import simulate_taiwan_fully_closed_v64 as v64
from continuous_multipop_mfg_v66 import ContinuousMultiPopulationMFGV66
from transport_symplectic_v66 import CoupledTransportSymplecticSystemV66


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
TRACE = OUT / "taiwan_v66_traceability.json"
AUDIT = OUT / "taiwan_v66_static_coupling_audit.json"
FREEZE = OUT / "taiwan_v66_frozen_specification.json"
OUT_CSV = OUT / "taiwan_v66_results.csv"
OUT_JSON = OUT / "taiwan_v66_simulation.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def validate_v66(diagnostics: dict, months: int) -> None:
    errors: list[str] = []
    for cell, item in diagnostics.items():
        transport = item.get("transport_symplectic")
        if transport is None:
            errors.append(f"{cell}: missing transport/symplectic module")
        else:
            limits = {
                "max_causal_clock_violation": 0.0,
                "max_mass_balance_residual": 1.0e-10,
                "max_shared_capacity_violation": 1.0e-12,
                "max_nonnegativity_violation": 1.0e-12,
                "max_wfr_mass_residual": 1.0e-12,
                "max_port_power_residual": 1.0e-10,
                "max_symplectic_defect": 1.0e-12,
                "max_so3_orthogonality_residual": 1.0e-10,
                "max_so3_determinant_residual": 1.0e-10,
                "max_jump_account_residual": 1.0e-12,
            }
            if transport["months"] != months:
                errors.append(f"{cell}: transport clock mismatch")
            if transport["dynamic_flow_months"] <= 0:
                errors.append(f"{cell}: capacity transport inactive")
            if transport["future_information_used"]:
                errors.append(f"{cell}: transport uses future information")
            for key, limit in limits.items():
                if transport[key] > limit:
                    errors.append(f"{cell}: {key}={transport[key]:.3g}")
        partial = item["mfg"].get("partial_observation") or {}
        deep = partial.get("v66_deep_causal_transport")
        if deep is None:
            errors.append(f"{cell}: missing deep causal OT")
        else:
            if deep["future_information_used"]:
                errors.append(f"{cell}: deep OT uses future information")
            if deep["multilayer_refresh_count"] <= 0:
                errors.append(f"{cell}: multilayer FGW inactive")
            for key in (
                "max_path_kernel_row_residual",
                "max_temporal_marginal_residual",
                "max_partial_mass_residual",
            ):
                if deep[key] > 1.0e-10:
                    errors.append(f"{cell}: {key}={deep[key]:.3g}")
    if errors:
        raise RuntimeError("V6.6 runtime validation failed: " + "; ".join(errors))


def run(paths: int = 128, months: int = 360,
        scenario_names: list[str] | None = None,
        case_names: list[str] | None = None, **kwargs):
    old_paths = (v64.TRACE, v64.AUDIT, v64.FREEZE)
    old_mfg = v64.v63.ContinuousMultiPopulationMFGV63
    v64.TRACE, v64.AUDIT, v64.FREEZE = TRACE, AUDIT, FREEZE
    v64.v63.ContinuousMultiPopulationMFGV63 = ContinuousMultiPopulationMFGV66
    try:
        result = v64.run(
            paths=paths, months=months,
            scenario_names=scenario_names, case_names=case_names,
            transport_symplectic_factory=CoupledTransportSymplecticSystemV66,
            **kwargs,
        )
    finally:
        v64.TRACE, v64.AUDIT, v64.FREEZE = old_paths
        v64.v63.ContinuousMultiPopulationMFGV63 = old_mfg
    validate_v66(result[2], months)
    return result


def write(rows, aggregate, diagnostics, verification) -> None:
    with OUT_CSV.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    verification["model_version"] = "V6.6-OT-SYM"
    verification["runtime_validation"] = "TAIWAN_V66_RUNTIME_VALIDATION: PASS"
    verification["result_sha256"] = sha256(OUT_CSV)
    verification["v66_upgrade"] = {
        "path_conditioned_causal_schrodinger_bridge": True,
        "multilayer_partial_fgw": True,
        "multi_commodity_capacity_dynamic_ot": True,
        "random_geometry_wfr_feedback": True,
        "stormer_verlet_symplectic_core": True,
        "so3_lie_group_update": True,
        "port_hamiltonian_dissipative_shell": True,
        "deds_jump_account": True,
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
        raise RuntimeError("V6.6 outcome simulation blocked until the complete-model gate passes")
    frozen = json.loads(FREEZE.read_text(encoding="utf-8"))
    if frozen.get("status") != "FROZEN":
        raise RuntimeError("V6.6 frozen specification is missing or invalid")
    result = run(
        paths=int(os.environ.get("TAIWAN_V66_PATHS", "128")),
        months=int(os.environ.get("TAIWAN_V66_MONTHS", "360")),
    )
    write(*result)
    print("TAIWAN_V66_COMPLETE_RUN: PASS")


if __name__ == "__main__":
    main()
