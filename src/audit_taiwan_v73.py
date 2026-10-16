from __future__ import annotations

import json
from pathlib import Path

from continuous_graph_network_v73 import ContinuousGraphOperationalNetworkV73


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
TARGET = OUT / "taiwan_v73_code_coupling_audit.json"


def main() -> None:
    diagnostic = json.loads((OUT / "taiwan_v73_continuous_graph_diagnostic.json").read_text(encoding="utf-8"))
    stress = json.loads((OUT / "taiwan_v73_continuous_graph_stress.json").read_text(encoding="utf-8"))
    trace = json.loads((OUT / "taiwan_v73_traceability.json").read_text(encoding="utf-8"))
    heading_map = json.loads((OUT / "taiwan_v73_heading_module_map.json").read_text(encoding="utf-8"))
    wrapper = (ROOT / "work" / "simulate_taiwan_fully_closed_v73.py").read_text(encoding="utf-8")
    required = (
        "_finite_volume_step", "_update_metric_graph", "_update_enterprise_graphon",
        "_cascade_capacities", "_temporal_metrics", "update", "diagnostics",
    )
    module_ids = {item["module_id"] for item in trace["modules"]}
    expected = {
        "v73_metric_graph_transport", "v73_telegraph_control",
        "v73_enterprise_financial_graphon", "v73_sparse_continuum_validation",
    }
    checks = {
        "all_solver_methods_present": all(callable(getattr(ContinuousGraphOperationalNetworkV73, n, None)) for n in required),
        "active_wrapper_swap": "v72.AlgebraicDiscreteOperationalNetworkV72 = ContinuousGraphOperationalNetworkV73" in wrapper,
        "wrapper_restores_class": "v72.AlgebraicDiscreteOperationalNetworkV72 = old_class" in wrapper,
        "runtime_gate_called": "validate_v73(result[2], months)" in wrapper,
        "complete_model_gate_enforced": "complete_model_gate()" in wrapper,
        "diagnostic_passed": diagnostic.get("status") == "PASS",
        "long_horizon_stress_passed": stress.get("status") == "PASS",
        "all_v73_modules_traced": expected <= module_ids,
        "heading_bidirectional_traceability": (
            heading_map.get("status") == "PASS"
            and heading_map.get("coverage_percent") == 100.0
            and not heading_map.get("active_modules_without_headings")
        ),
        "complete_manifest_gate": trace.get("gate_status") == "PASS" and not trace.get("blocking_gaps"),
        "single_inventory_owner": diagnostic["checks"]["physical_inventory_has_single_owner"],
        "no_probability_postprocessing": diagnostic["checks"]["no_probability_postprocessing"],
    }
    payload = {
        "model_version": "V7.3-CONTINUOUS-GRAPH",
        "classification": "STATIC_COUPLING_AUDIT; no terminal probabilities",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "blocking_gaps": [name for name, passed in checks.items() if not passed],
    }
    TARGET.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"TAIWAN_V73_CODE_COUPLING_AUDIT: {payload['status']}")
    if payload["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
