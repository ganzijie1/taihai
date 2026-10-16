from __future__ import annotations

import json
from pathlib import Path

import simulate_taiwan_fully_closed_v71 as v71
from combinatorial_network_v71 import CombinatorialOperationalNetworkV71
from operational_constraints_v47 import DynamicOperationalConstraintSystem


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
TARGET = OUT / "taiwan_v71_code_coupling_audit.json"


def main() -> None:
    diagnostic = json.loads(
        (OUT / "taiwan_v71_combinatorial_network_diagnostic.json").read_text(
            encoding="utf-8"
        )
    )
    trace = json.loads(
        (OUT / "taiwan_v71_traceability.json").read_text(encoding="utf-8")
    )
    source = (ROOT / "work" / "simulate_taiwan_fully_closed_v71.py").read_text(
        encoding="utf-8"
    )
    required_methods = (
        "_advance_pipeline", "_route_capacity", "_plan_routes",
        "_select_protection", "_cascade_capacities",
        "_allocate_multicommodity_flows", "_allocate_repairs", "update",
    )
    modules = {item["module_id"]: item for item in trace["modules"]}
    checks = {
        "inherits_unique_inventory_owner": issubclass(
            CombinatorialOperationalNetworkV71,
            DynamicOperationalConstraintSystem,
        ),
        "all_solver_methods_present": all(
            callable(getattr(CombinatorialOperationalNetworkV71, name, None))
            for name in required_methods
        ),
        "active_wrapper_swap": (
            "core.DynamicOperationalConstraintSystem = CombinatorialOperationalNetworkV71"
            in source
        ),
        "wrapper_restores_class": (
            "core.DynamicOperationalConstraintSystem = old_operations" in source
        ),
        "runtime_gate_called": "validate_v71(result[2], months)" in source,
        "diagnostic_passed": diagnostic.get("status") == "PASS",
        "trace_gate_passed": trace.get("gate_status") == "PASS",
        "four_v71_modules_active": all(
            modules.get(name, {}).get("status") == "active"
            for name in (
                "v71_time_expanded_multicommodity_network",
                "v71_cut_connectivity_and_cascade",
                "v71_matching_protection_and_recovery",
                "v71_hypergraph_complementarity",
            )
        ),
        "no_probability_postprocessing": (
            "terminal_probability_adjustment" in source
            and diagnostic["checks"]["no_terminal_probability_adjustment"]
        ),
    }
    payload = {
        "model_version": "V7.1-COMBINATORIAL-NETWORK",
        "classification": "STATIC_COUPLING_AUDIT; no outcome probabilities",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "blocking_gaps": [name for name, passed in checks.items() if not passed],
    }
    TARGET.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"TAIWAN_V71_CODE_COUPLING_AUDIT: {payload['status']}")
    if payload["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
