from __future__ import annotations

import json
import math
from pathlib import Path
import statistics

from simulate_taiwan_multidomain_v42 import CASES


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"


def main() -> None:
    result = json.loads((OUT / "taiwan_v78_simulation.json").read_text(encoding="utf-8"))
    rows = result["conditional"]
    columns = (
        "conditional_broad_control_rate", "kernel_uncertainty_low_rate",
        "kernel_uncertainty_mid_rate", "kernel_uncertainty_high_rate",
        "mechanical_majority_control_share", "status_quo_settlement_share",
        "unification_settlement_share",
    )
    finite = all(math.isfinite(float(row[key])) for row in rows for key in columns)
    bounded = all(-1.0e-12 <= float(row[key]) <= 1.0 + 1.0e-12 for row in rows for key in columns)
    errors: list[str] = []
    envelope = {
        "vi_gap": [], "mirror_correction": [], "nonpotential_norm": [],
        "monotonicity_margin": [], "weighted_adjoint": [],
    }
    for cell, item in result["module_diagnostics"].items():
        mfg = item["mfg"]
        mixed = mfg["v78_mixed_potential_nonpotential"]
        ablations = mixed["maximum_ablation_field_deviation"]
        gates = (
            mixed["potential_proximal_updates"] >= 360
            and mixed["mirror_prox_evaluations"] >= 360
            and mixed["maximum_weighted_adjoint_residual"] <= 1.0e-12
            and mixed["maximum_potential_decomposition_residual"] <= 1.0e-12
            and mixed["maximum_nonpotential_operator_norm"] > 0.0
            and mixed["maximum_mirror_prox_predictor_step"] > 0.0
            and mixed["maximum_mirror_prox_correction"] > 0.0
            and math.isfinite(mixed["maximum_mixed_vi_gap"])
            and mixed["ablation_evaluations"] == 360
            and set(ablations) == {
                "no_graph_mfg", "block_gmfg", "low_rank_graphon", "finite_graph"
            }
            and all(value > 0.0 for value in ablations.values())
            and not mixed["terminal_probability_adjustment"]
            and mfg.get("nonconverged_months", 1) == 0
            and mfg.get("max_fixed_point_residual", math.inf) <= 5.0e-4
        )
        if not gates:
            errors.append(cell)
        envelope["vi_gap"].append(mixed["maximum_mixed_vi_gap"])
        envelope["mirror_correction"].append(mixed["maximum_mirror_prox_correction"])
        envelope["nonpotential_norm"].append(mixed["maximum_nonpotential_operator_norm"])
        envelope["monotonicity_margin"].append(mixed["minimum_monotonicity_margin"])
        envelope["weighted_adjoint"].append(mixed["maximum_weighted_adjoint_residual"])
    trace = json.loads((OUT / "taiwan_v78_traceability.json").read_text(encoding="utf-8"))
    static = json.loads((OUT / "taiwan_v78_static_coupling_audit.json").read_text(encoding="utf-8"))
    freeze = json.loads((OUT / "taiwan_v78_frozen_specification.json").read_text(encoding="utf-8"))
    checks = {
        "conditional_row_count": len(rows) == 150,
        "diagnostic_cell_count": len(result["module_diagnostics"]) == 30,
        "probabilities_finite": finite,
        "probabilities_bounded": bounded,
        "runtime_gates": not errors,
        "complete_model_gate": (
            trace.get("gate_status") == "PASS"
            and not trace.get("blocking_gaps")
            and static.get("simulation_authorized") is True
            and freeze.get("status") == "FROZEN"
        ),
        "runtime_validation": result["verification"].get("runtime_validation")
        == "TAIWAN_V78_RUNTIME_VALIDATION: PASS",
        "checkpoint_contract": result["verification"].get(
            "checkpoint_contract", {}
        ).get("cell_count") == 30,
    }
    case_horizon = []
    for case in CASES:
        for horizon in (5, 10, 15, 20, 30):
            values = [
                float(row["conditional_broad_control_rate"])
                for row in rows
                if row["case"] == case and row["horizon_years"] == horizon
            ]
            case_horizon.append({
                "case": case,
                "horizon_years": horizon,
                "equal_weight_trade_scenario_mean": statistics.mean(values),
                "minimum": min(values),
                "maximum": max(values),
            })
    summary = {
        "model_version": "V7.8-MIXED-POTENTIAL-NONPOTENTIAL-MFG",
        "interpretation": "registered trade scenarios are equal-weight sensitivity cases, not estimated scenario probabilities",
        "case_horizon_conditional_broad_control": case_horizon,
        "intervention_probability_means": result["verification"]["intervention_probability_means"],
        "long_term_competing_risk_checkpoints": result["verification"]["long_term_checkpoints"],
        "runtime_envelope": {
            "maximum_mixed_vi_gap": max(envelope["vi_gap"]),
            "maximum_mirror_prox_correction": max(envelope["mirror_correction"]),
            "maximum_nonpotential_operator_norm": max(envelope["nonpotential_norm"]),
            "minimum_monotonicity_margin": min(envelope["monotonicity_margin"]),
            "maximum_weighted_adjoint_residual": max(envelope["weighted_adjoint"]),
        },
    }
    (OUT / "taiwan_v78_result_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    payload = {
        "model_version": "V7.8-MIXED-POTENTIAL-NONPOTENTIAL-MFG",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "diagnostic_errors": errors,
        "result_sha256": result["verification"]["result_sha256"],
    }
    (OUT / "taiwan_v78_result_audit.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"TAIWAN_V78_RESULT_AUDIT: {payload['status']}")
    if payload["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
