from __future__ import annotations

import json
import math
from pathlib import Path
import statistics

from simulate_taiwan_multidomain_v42 import CASES


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"


def main() -> None:
    result = json.loads((OUT / "taiwan_v76_simulation.json").read_text(encoding="utf-8"))
    rows = result["conditional"]
    columns = (
        "conditional_broad_control_rate", "kernel_uncertainty_low_rate",
        "kernel_uncertainty_mid_rate", "kernel_uncertainty_high_rate",
        "mechanical_majority_control_share", "status_quo_settlement_share",
        "unification_settlement_share",
    )
    finite = all(math.isfinite(float(row[key])) for row in rows for key in columns)
    bounded = all(-1e-12 <= float(row[key]) <= 1.0 + 1e-12 for row in rows for key in columns)
    errors, signed_diags, tda_diags = [], [], []
    for cell, item in result["module_diagnostics"].items():
        signed = item["mfg"]["v76_signed_potential_scf"]
        tda = item["mfg"]["v76_online_tda"]
        signed_diags.append(signed)
        tda_diags.append(tda)
        gates = (
            signed["signed_graphon_updates"] == 360
            and signed["relation_memory_updates"] == 360
            and signed["minimum_zero_boundary_formation_rate"] > 0.0
            and not signed["atomized_boundary_absorbing"]
            and 0.0 <= signed["minimum_bridge_index"] <= signed["maximum_bridge_index"] <= 1.0
            and signed["maximum_signed_field_correction"] > 0.0
            and signed["maximum_decomposition_residual"] <= 1e-12
            and signed["maximum_pair_nonpotential_ratio"] > 0.0
            and signed["maximum_hypergraph_nonpotential_ratio"] > 0.0
            and signed["minimum_signed_laplacian_eigenvalue"] >= -1e-8
            and signed["scf_calls"] > 0
            and signed["reduced_branch_evaluations"] == 30
            and not signed["repelling_rule_used"]
            and not signed["terminal_probability_adjustment"]
            and item["mfg"].get("nonconverged_months", 1) == 0
            and item["mfg"].get("max_fixed_point_residual", math.inf) <= 5e-4
            and tda["updates"] == 120
            and tda["hold_months"] == 240
            and tda["maximum_feedback_shock"] > 0.0
            and not tda["terminal_probability_adjustment"]
        )
        if not gates:
            errors.append(cell)
    trace = json.loads((OUT / "taiwan_v76_traceability.json").read_text(encoding="utf-8"))
    static = json.loads((OUT / "taiwan_v76_static_coupling_audit.json").read_text(encoding="utf-8"))
    freeze = json.loads((OUT / "taiwan_v76_frozen_specification.json").read_text(encoding="utf-8"))
    checks = {
        "conditional_row_count": len(rows) == 150,
        "diagnostic_cell_count": len(result["module_diagnostics"]) == 30,
        "probabilities_finite": finite,
        "probabilities_bounded": bounded,
        "runtime_gates": not errors,
        "complete_model_gate": trace.get("gate_status") == "PASS" and not trace.get("blocking_gaps") and static.get("simulation_authorized") is True and freeze.get("status") == "FROZEN",
        "runtime_validation": result["verification"].get("runtime_validation") == "TAIWAN_V76_RUNTIME_VALIDATION: PASS",
        "checkpoint_contract": result["verification"].get("checkpoint_contract", {}).get("cell_count") == 30,
    }
    case_horizon = []
    for case in CASES:
        for horizon in (5, 10, 15, 20, 30):
            values = [float(row["conditional_broad_control_rate"]) for row in rows if row["case"] == case and row["horizon_years"] == horizon]
            case_horizon.append({"case": case, "horizon_years": horizon, "equal_weight_trade_scenario_mean": statistics.mean(values), "minimum": min(values), "maximum": max(values)})
    summary = {
        "model_version": "V7.6-SIGNED-POTENTIAL-SCF",
        "interpretation": "registered trade scenarios are equal-weight sensitivity cases, not estimated scenario probabilities",
        "case_horizon_conditional_broad_control": case_horizon,
        "intervention_probability_means": result["verification"]["intervention_probability_means"],
        "long_term_competing_risk_checkpoints": result["verification"]["long_term_checkpoints"],
        "runtime_envelope": {
            "maximum_signed_field_correction": max(x["maximum_signed_field_correction"] for x in signed_diags),
            "maximum_structural_frustration": max(x["maximum_structural_frustration"] for x in signed_diags),
            "maximum_pair_nonpotential_ratio": max(x["maximum_pair_nonpotential_ratio"] for x in signed_diags),
            "maximum_hypergraph_nonpotential_ratio": max(x["maximum_hypergraph_nonpotential_ratio"] for x in signed_diags),
            "maximum_reduced_branch_count": max(x["maximum_reduced_branch_count"] for x in signed_diags),
            "maximum_reduced_branch_spectral_radius": max(x["maximum_reduced_branch_spectral_radius"] for x in signed_diags),
            "minimum_bridge_index": min(x["minimum_bridge_index"] for x in signed_diags),
            "maximum_bridge_index": max(x["maximum_bridge_index"] for x in signed_diags),
            "maximum_nonautonomous_growth_rate": max(x["maximum_nonautonomous_growth_rate"] for x in signed_diags),
            "maximum_tda_feedback_shock": max(x["maximum_feedback_shock"] for x in tda_diags),
            "maximum_beta0_persistence_area": max(x["maximum_beta0_persistence_area"] for x in tda_diags),
            "maximum_mapper_components": max(x["maximum_mapper_components"] for x in tda_diags),
            "maximum_bipersistence_rank": max(x["maximum_bipersistence_rank"] for x in tda_diags),
        },
    }
    (OUT / "taiwan_v76_result_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    payload = {"model_version": "V7.6-SIGNED-POTENTIAL-SCF", "status": "PASS" if all(checks.values()) else "FAIL", "checks": checks, "diagnostic_errors": errors, "result_sha256": result["verification"]["result_sha256"]}
    (OUT / "taiwan_v76_result_audit.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"TAIWAN_V76_RESULT_AUDIT: {payload['status']}")
    if payload["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
