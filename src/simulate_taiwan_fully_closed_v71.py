from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path

import simulate_taiwan_fully_closed_v70 as v70
from combinatorial_network_v71 import CombinatorialOperationalNetworkV71


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
TRACE = OUT / "taiwan_v71_traceability.json"
AUDIT = OUT / "taiwan_v71_static_coupling_audit.json"
FREEZE = OUT / "taiwan_v71_frozen_specification.json"
OUT_CSV = OUT / "taiwan_v71_results.csv"
OUT_JSON = OUT / "taiwan_v71_simulation.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def validate_v71(diagnostics: dict, months: int) -> None:
    errors: list[str] = []
    for cell, item in diagnostics.items():
        network = item.get("operations", {}).get("combinatorial_network_v71")
        if network is None:
            errors.append(f"{cell}: missing combinatorial operational network")
            continue
        if network["months"] != months:
            errors.append(f"{cell}: combinatorial network clock mismatch")
        gates = {
            "capacity": network["maximum_capacity_residual"] <= 1.0e-10,
            "stock": network["maximum_stock_residual"] <= 1.0e-9,
            "matching": network["maximum_matching_residual"] <= 1.0e-10,
            "maxflow-mincut": network["maximum_maxflow_mincut_residual"] <= 1.0e-10,
            "cascade": network["maximum_cascade_residual"] <= 1.0e-7,
            "monotonicity": network["maximum_monotonicity_violation"] <= 1.0e-12,
            "pipeline": network["maximum_pipeline_negative_mass"] <= 1.0e-12,
            "protection": network["protection_budget_violations"] == 0,
            "connectivity": network["minimum_edge_connectivity"] >= 0,
            "cascade convergence": network["cascade_nonconvergence_months"] == 0,
            "no posthoc": not network["terminal_probability_adjustment"],
        }
        errors.extend(f"{cell}: {name}" for name, passed in gates.items() if not passed)
    if errors:
        raise RuntimeError("V7.1 runtime validation failed: " + "; ".join(errors))


def run(
    paths: int = 128, months: int = 360,
    scenario_names: list[str] | None = None,
    case_names: list[str] | None = None, *,
    combinatorial_network_enabled: bool = True,
    **kwargs,
):
    core = v70.v69.core
    old_paths = (v70.TRACE, v70.AUDIT, v70.FREEZE)
    old_operations = core.DynamicOperationalConstraintSystem
    v70.TRACE, v70.AUDIT, v70.FREEZE = TRACE, AUDIT, FREEZE
    if combinatorial_network_enabled:
        core.DynamicOperationalConstraintSystem = CombinatorialOperationalNetworkV71
    try:
        result = v70.run(
            paths=paths, months=months, scenario_names=scenario_names,
            case_names=case_names, **kwargs,
        )
    finally:
        core.DynamicOperationalConstraintSystem = old_operations
        v70.TRACE, v70.AUDIT, v70.FREEZE = old_paths
    if combinatorial_network_enabled:
        validate_v71(result[2], months)
    return result


def write(rows, aggregate, diagnostics, verification) -> None:
    with OUT_CSV.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    verification["model_version"] = "V7.1-COMBINATORIAL-NETWORK"
    verification["runtime_validation"] = "TAIWAN_V71_RUNTIME_VALIDATION: PASS"
    verification["result_sha256"] = sha256(OUT_CSV)
    verification["v71_upgrade"] = {
        "time_expanded_shipment_pipeline": True,
        "shared_capacity_multicommodity_flow": True,
        "exact_series_parallel_maxflow_mincut": True,
        "menger_survivability": True,
        "fractional_b_matching_repairs": True,
        "partition_matroid_protection": True,
        "hypergraph_production_complementarity": True,
        "cascade_fixed_point": True,
        "same_path_pre_transition_feedback": True,
        "posthoc_probability_adjustment": False,
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
        raise RuntimeError("V7.1 outcome simulation blocked until the complete-model gate passes")
    frozen = json.loads(FREEZE.read_text(encoding="utf-8"))
    if frozen.get("status") != "FROZEN":
        raise RuntimeError("V7.1 frozen specification is missing or invalid")
    result = run(
        paths=int(os.environ.get("TAIWAN_V71_PATHS", "128")),
        months=int(os.environ.get("TAIWAN_V71_MONTHS", "360")),
    )
    write(*result)
    print("TAIWAN_V71_COMPLETE_RUN: PASS")


if __name__ == "__main__":
    main()
