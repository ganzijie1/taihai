from __future__ import annotations

import json
from pathlib import Path

from algebraic_discrete_network_v72 import AlgebraicDiscreteOperationalNetworkV72


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
TARGET = OUT / "taiwan_v72_code_coupling_audit.json"
HEADING_MAP = OUT / "taiwan_v72_heading_module_map.json"


def main() -> None:
    diagnostic = json.loads(
        (OUT / "taiwan_v72_algebraic_discrete_diagnostic.json").read_text(encoding="utf-8")
    )
    stress = json.loads(
        (OUT / "taiwan_v72_operational_stress.json").read_text(encoding="utf-8")
    )
    trace = json.loads((OUT / "taiwan_v72_traceability.json").read_text(encoding="utf-8"))
    heading_map = json.loads(HEADING_MAP.read_text(encoding="utf-8"))
    wrapper = (ROOT / "work" / "simulate_taiwan_fully_closed_v72.py").read_text(
        encoding="utf-8"
    )
    required = (
        "_algebraic_metrics", "_nonbacktracking_radius", "_select_protection",
        "_cascade_capacities", "_minimum_cut", "_temporal_metrics",
        "_allocate_multicommodity_flows", "update", "diagnostics",
    )
    module_ids = {item["module_id"] for item in trace["modules"]}
    checks = {
        "all_solver_methods_present": all(
            callable(getattr(AlgebraicDiscreteOperationalNetworkV72, name, None))
            for name in required
        ),
        "active_wrapper_swap": (
            "core.DynamicOperationalConstraintSystem = AlgebraicDiscreteOperationalNetworkV72"
            in wrapper
        ),
        "wrapper_restores_class": "core.DynamicOperationalConstraintSystem = old_operations" in wrapper,
        "runtime_gate_called": "validate_v72(result[2], months)" in wrapper,
        "complete_model_gate_enforced": "complete_model_gate()" in wrapper,
        "diagnostic_passed": diagnostic.get("status") == "PASS",
        "long_horizon_operational_stress_passed": stress.get("status") == "PASS",
        "all_v72_modules_traced": all(name in module_ids for name in (
            "v72_resistance_tree_spectrum", "v72_signed_coarsened_graph",
            "v72_temporal_survivable_network", "v72_matroid_conflict_scheduler",
            "v72_treewidth_solver_selection",
        )),
        "heading_level_bidirectional_traceability_passed": (
            heading_map.get("status") == "PASS"
            and heading_map.get("coverage_percent") == 100.0
            and not heading_map.get("active_modules_without_headings")
        ),
        "complete_manifest_gate_passed": (
            trace.get("gate_status") == "PASS"
            and trace.get("coverage_percent") == 100.0
            and not trace.get("blocking_gaps")
        ),
        "no_probability_postprocessing": diagnostic["checks"]["no_probability_postprocessing"],
    }
    payload = {
        "model_version": "V7.2-ALGEBRAIC-DISCRETE-GRAPH",
        "classification": "STATIC_COUPLING_AUDIT; no outcome probabilities",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "blocking_gaps": [name for name, passed in checks.items() if not passed],
    }
    TARGET.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"TAIWAN_V72_CODE_COUPLING_AUDIT: {payload['status']}")
    if payload["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
