from __future__ import annotations

import json
from pathlib import Path

from finsler_network_v74 import FinslerOperationalNetworkV74


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
TARGET = OUT / "taiwan_v74_code_coupling_audit.json"


def main() -> None:
    diagnostic = json.loads((OUT / "taiwan_v74_finsler_diagnostic.json").read_text(encoding="utf-8"))
    stress = json.loads((OUT / "taiwan_v74_finsler_stress.json").read_text(encoding="utf-8"))
    trace = json.loads((OUT / "taiwan_v74_traceability.json").read_text(encoding="utf-8"))
    heading_map = json.loads((OUT / "taiwan_v74_heading_module_map.json").read_text(encoding="utf-8"))
    wrapper = (ROOT / "work" / "simulate_taiwan_fully_closed_v74.py").read_text(encoding="utf-8")
    base = (ROOT / "work" / "simulate_taiwan_multidomain_v42.py").read_text(encoding="utf-8")
    mfg = (ROOT / "work" / "continuous_multipop_mfg_v47.py").read_text(encoding="utf-8")
    required = (
        "_randers_pair", "_build_finsler_metric", "_shortest_arrival",
        "_path_geometry", "_allocate_multicommodity_flows", "update", "diagnostics",
    )
    expected = {
        "v74_randers_eikonal",
        "v74_finsler_hard_flow_queue",
        "v74_finsler_mfg_feedback",
        "v74_finsler_validation",
    }
    module_ids = {item["module_id"] for item in trace["modules"]}
    checks = {
        "all_solver_methods_present": all(callable(getattr(FinslerOperationalNetworkV74, name, None)) for name in required),
        "wrapper_swaps_v74_class": "v73.ContinuousGraphOperationalNetworkV73 = FinslerOperationalNetworkV74" in wrapper,
        "wrapper_restores_class": "v73.ContinuousGraphOperationalNetworkV73 = old_class" in wrapper,
        "runtime_gate_called": "validate_v74(result[2], months)" in wrapper,
        "complete_model_gate_enforced": "complete_model_gate()" in wrapper,
        "operations_export_same_path_cost": "last_mfg_mobility_cost" in (ROOT / "work" / "finsler_network_v74.py").read_text(encoding="utf-8"),
        "base_passes_cost_to_mfg": (
            "mobility_cost=(" in base
            and 'getattr(operations_system, "last_mfg_mobility_cost", None)' in base
        ),
        "mfg_consumes_control_cost": "effective_control_cost" in mfg and "mobility_cost" in mfg,
        "diagnostic_passed": diagnostic.get("status") == "PASS",
        "long_horizon_stress_passed": stress.get("status") == "PASS",
        "all_v74_modules_traced": expected <= module_ids,
        "heading_traceability": heading_map.get("status") == "PASS" and heading_map.get("coverage_percent") == 100.0,
        "complete_manifest_gate": trace.get("gate_status") == "PASS" and not trace.get("blocking_gaps"),
        "single_inventory_owner": diagnostic["checks"]["single_inventory_owner"],
        "no_probability_postprocessing": diagnostic["checks"]["no_probability_postprocessing"],
    }
    payload = {
        "model_version": "V7.4-FINSLER-CLOSED-LOOP",
        "classification": "STATIC_COUPLING_AUDIT; no terminal probabilities",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "blocking_gaps": [name for name, passed in checks.items() if not passed],
    }
    TARGET.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"TAIWAN_V74_CODE_COUPLING_AUDIT: {payload['status']}")
    if payload["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
