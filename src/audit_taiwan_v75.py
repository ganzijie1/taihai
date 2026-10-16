from __future__ import annotations

import json
from pathlib import Path

from hypergraph_mfg_v75 import DirectedMultilayerHypergraphMFGV75
from quantitative_financial_instability_v75 import QuantitativeFinancialInstabilityV75


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
TARGET = OUT / "taiwan_v75_code_coupling_audit.json"


def main() -> None:
    diagnostic = json.loads(
        (OUT / "taiwan_v75_diagnostic_only.json").read_text(encoding="utf-8")
    )
    wrapper = (ROOT / "work" / "simulate_taiwan_fully_closed_v75.py").read_text(
        encoding="utf-8"
    )
    finance = (ROOT / "work" / "quantitative_financial_instability_v75.py").read_text(
        encoding="utf-8"
    )
    hypergraph = (ROOT / "work" / "hypergraph_mfg_v75.py").read_text(
        encoding="utf-8"
    )
    finance_methods = (
        "_update_minsky_states",
        "_eisenberg_noe_clearing",
        "_update_regime_and_intensity",
        "update",
        "summary",
    )
    hypergraph_methods = (
        "_build_directed_hypergraph",
        "_effective_graphon",
        "_hypergraph_fields",
        "_multilayer_fields",
        "diagnostics",
    )
    checks = {
        "diagnostic_only_passed": diagnostic.get("status") == "PASS",
        "finance_methods_present": all(
            callable(getattr(QuantitativeFinancialInstabilityV75, name, None))
            for name in finance_methods
        ),
        "hypergraph_methods_present": all(
            callable(getattr(DirectedMultilayerHypergraphMFGV75, name, None))
            for name in hypergraph_methods
        ),
        "wrapper_replaces_mfg_on_active_chain": (
            "v66.ContinuousMultiPopulationMFGV66 = DirectedMultilayerHypergraphMFGV75"
            in wrapper
        ),
        "wrapper_replaces_finance_on_active_chain": (
            "v69.MartingaleOTFinancialSystemV69 = QuantitativeFinancialInstabilityV75"
            in wrapper
        ),
        "wrapper_restores_both_classes": (
            "v66.ContinuousMultiPopulationMFGV66 = old_mfg" in wrapper
            and "v69.MartingaleOTFinancialSystemV69 = old_finance" in wrapper
        ),
        "complete_model_gate_precedes_run": (
            wrapper.index("complete_model_gate()") < wrapper.index("v74.run(")
        ),
        "runtime_validation_called": "validate_v75(result[2], months)" in wrapper,
        "finance_changes_current_factors": (
            "adjusted = FinancialFactors(" in finance
            and "self.last_factors = adjusted" in finance
            and "return adjusted" in finance
        ),
        "finance_has_no_terminal_adjustment": (
            '"terminal_probability_adjustment": False' in finance
        ),
        "hypergraph_inside_fixed_point_operator": (
            "def _multilayer_fields(" in hypergraph
            and "super()._multilayer_fields(" in hypergraph
            and "self._hypergraph_fields(" in hypergraph
        ),
        "hypergraph_is_not_fixed_clique_proxy": (
            "edge_dispersion" in hypergraph
            and "major_gate" in hypergraph
            and "nonlinear_action" in hypergraph
        ),
        "finite_rollback_includes_hyperedges": (
            "def _finite_network_fields(" in hypergraph
            and "sampled_state = (1.0 - share) * pair_state + share * hyper_state"
            in hypergraph
        ),
        "hard_capacity_ownership_preserved": (
            '"hard_capacity_owner": "specialized operational network solvers"'
            in hypergraph
        ),
        "no_outcome_artifact_created": not (
            OUT / "taiwan_v75_results.csv"
        ).exists() and not (OUT / "taiwan_v75_simulation.json").exists(),
    }
    payload = {
        "model_version": "V7.5-HYPERGRAPH-MINSKY-FINANCE",
        "classification": "STATIC_COUPLING_AUDIT; no outcome simulation",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "blocking_gaps": [name for name, passed in checks.items() if not passed],
    }
    TARGET.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"TAIWAN_V75_CODE_COUPLING_AUDIT: {payload['status']}")
    if payload["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
