from __future__ import annotations

import json
import math
from pathlib import Path
import statistics

from simulate_taiwan_multidomain_v42 import CASES


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
TARGET = OUT / "taiwan_v75_result_audit.json"
SUMMARY = OUT / "taiwan_v75_result_summary.json"


def main() -> None:
    result = json.loads((OUT / "taiwan_v75_simulation.json").read_text(encoding="utf-8"))
    rows = result["conditional"]
    probability_columns = (
        "conditional_broad_control_rate",
        "kernel_uncertainty_low_rate",
        "kernel_uncertainty_mid_rate",
        "kernel_uncertainty_high_rate",
        "mechanical_majority_control_share",
        "status_quo_settlement_share",
        "unification_settlement_share",
    )
    finite = all(math.isfinite(float(row[key])) for row in rows for key in probability_columns)
    bounded = all(-1.0e-12 <= float(row[key]) <= 1.0 + 1.0e-12 for row in rows for key in probability_columns)
    diagnostic_errors = []
    finance_diags, hypergraph_diags = [], []
    for cell, item in result["module_diagnostics"].items():
        finance = item["finance"]["v75_quantitative_financial_instability"]
        hypergraph = item["mfg"]["v75_directed_multilayer_hypergraph_mfg"]
        finance_diags.append(finance)
        hypergraph_diags.append(hypergraph)
        gates = (
            finance["months"] == 360
            and finance["max_financing_mass_residual"] <= 1.0e-12
            and finance["max_regime_mass_residual"] <= 1.0e-12
            and finance["max_clearing_residual"] <= 1.0e-8
            and finance["max_negative_payment"] <= 1.0e-12
            and hypergraph["enabled"]
            and hypergraph["weight_updates"] == 360
            and hypergraph["field_evaluations"] > 0
            and hypergraph["max_incidence_residual"] <= 1.0e-12
            and hypergraph["max_higher_order_contribution"] > 0.0
            and hypergraph["max_clique_rollback_error"] > 0.0
            and item["mfg"].get("nonconverged_months", 1) == 0
            and item["mfg"].get("max_fixed_point_residual", math.inf) <= 5.0e-4
            and not finance["terminal_probability_adjustment"]
            and not hypergraph["terminal_probability_adjustment"]
        )
        if not gates:
            diagnostic_errors.append(cell)

    trace = json.loads((OUT / "taiwan_v75_traceability.json").read_text(encoding="utf-8"))
    static_audit = json.loads((OUT / "taiwan_v75_static_coupling_audit.json").read_text(encoding="utf-8"))
    freeze = json.loads((OUT / "taiwan_v75_frozen_specification.json").read_text(encoding="utf-8"))
    case_horizon = []
    for case in CASES:
        for horizon in (5, 10, 15, 20, 30):
            selected = [row for row in rows if row["case"] == case and row["horizon_years"] == horizon]
            values = [float(row["conditional_broad_control_rate"]) for row in selected]
            case_horizon.append({
                "case": case,
                "horizon_years": horizon,
                "equal_weight_trade_scenario_mean": statistics.mean(values),
                "minimum": min(values),
                "maximum": max(values),
            })
    intervention = result["verification"]["intervention_probability_means"]
    intervention_weighted = []
    for horizon in (5, 10, 15, 20, 30):
        values = {item["case"]: item["equal_weight_trade_scenario_mean"] for item in case_horizon if item["horizon_years"] == horizon}
        intervention_weighted.append({
            "horizon_years": horizon,
            "intervention_weighted_equal_trade_scenario_mean": sum(float(intervention[case]) * values[case] for case in CASES),
        })

    checks = {
        "conditional_row_count": len(rows) == 150,
        "diagnostic_cell_count": len(result["module_diagnostics"]) == 30,
        "all_probability_outputs_finite": finite,
        "all_probability_outputs_bounded": bounded,
        "all_v75_runtime_gates": not diagnostic_errors,
        "complete_model_gate": trace.get("gate_status") == "PASS" and not trace.get("blocking_gaps") and static_audit.get("simulation_authorized") is True and freeze.get("status") == "FROZEN",
        "runtime_validation": result["verification"].get("runtime_validation") == "TAIWAN_V75_RUNTIME_VALIDATION: PASS",
        "checkpoint_contract": result["verification"].get("checkpoint_contract", {}).get("cell_count") == 30,
    }
    summary = {
        "model_version": "V7.5-HYPERGRAPH-MINSKY-FINANCE",
        "interpretation": "registered trade scenarios are equally weighted sensitivity cases, not empirically estimated scenario probabilities",
        "case_horizon_conditional_broad_control": case_horizon,
        "intervention_probability_means": intervention,
        "intervention_weighted_conditional_broad_control": intervention_weighted,
        "long_term_competing_risk_checkpoints": result["verification"]["long_term_checkpoints"],
        "v75_runtime_envelope": {
            "maximum_financing_mass_residual": max(item["max_financing_mass_residual"] for item in finance_diags),
            "maximum_clearing_residual": max(item["max_clearing_residual"] for item in finance_diags),
            "maximum_regime_mass_residual": max(item["max_regime_mass_residual"] for item in finance_diags),
            "maximum_higher_order_contribution": max(item["max_higher_order_contribution"] for item in hypergraph_diags),
            "minimum_clique_rollback_error": min(item["max_clique_rollback_error"] for item in hypergraph_diags),
        },
    }
    SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    payload = {
        "model_version": "V7.5-HYPERGRAPH-MINSKY-FINANCE",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "diagnostic_errors": diagnostic_errors,
        "result_sha256": result["verification"]["result_sha256"],
    }
    TARGET.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"TAIWAN_V75_RESULT_AUDIT: {payload['status']}")
    if payload["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
