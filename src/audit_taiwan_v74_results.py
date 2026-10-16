from __future__ import annotations

import json
import math
from pathlib import Path
import statistics

from simulate_taiwan_multidomain_v42 import CASES


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
TARGET = OUT / "taiwan_v74_result_audit.json"
SUMMARY = OUT / "taiwan_v74_result_summary.json"


def main() -> None:
    result = json.loads((OUT / "taiwan_v74_simulation.json").read_text(encoding="utf-8"))
    rows = result["conditional"]
    probability_columns = (
        "conditional_broad_control_rate", "kernel_uncertainty_low_rate",
        "kernel_uncertainty_mid_rate", "kernel_uncertainty_high_rate",
        "mechanical_majority_control_share", "status_quo_settlement_share",
        "unification_settlement_share",
    )
    finite = all(math.isfinite(float(row[key])) for row in rows for key in probability_columns)
    bounded = all(-1.0e-12 <= float(row[key]) <= 1.0 + 1.0e-12 for row in rows for key in probability_columns)
    diagnostic_errors = []
    finsler_diags = []
    for cell, item in result["module_diagnostics"].items():
        finsler = item["operations"]["finsler_network_v74"]
        inherited = item["operations"]["combinatorial_network_v71"]
        mfg = item["mfg"]
        finsler_diags.append(finsler)
        gates = (
            finsler["months"] == 360
            and finsler["maximum_randers_one_form_norm"] < 1.0
            and finsler["minimum_finsler_cost"] > 0.0
            and finsler["maximum_positive_homogeneity_residual"] <= 1.0e-12
            and finsler["maximum_eikonal_residual"] <= 1.0e-10
            and inherited["maximum_capacity_residual"] <= 1.0e-10
            and inherited["maximum_stock_residual"] <= 1.0e-9
            and mfg.get("finsler_mobility_cost_active") is True
            and not finsler["continuous_geometry_owns_physical_inventory"]
            and not finsler["terminal_probability_adjustment"]
        )
        if not gates:
            diagnostic_errors.append(cell)

    keys = ("case", "trade_scenario", "horizon_years")
    old = json.loads((OUT / "taiwan_v73_simulation.json").read_text(encoding="utf-8"))
    trace = json.loads((OUT / "taiwan_v74_traceability.json").read_text(encoding="utf-8"))
    static_audit = json.loads((OUT / "taiwan_v74_static_coupling_audit.json").read_text(encoding="utf-8"))
    freeze = json.loads((OUT / "taiwan_v74_frozen_specification.json").read_text(encoding="utf-8"))
    old_rows = {tuple(row[key] for key in keys): row for row in old["conditional"]}
    differences = [
        float(row["conditional_broad_control_rate"])
        - float(old_rows[tuple(row[key] for key in keys)]["conditional_broad_control_rate"])
        for row in rows
    ]
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
        values = {
            item["case"]: item["equal_weight_trade_scenario_mean"]
            for item in case_horizon if item["horizon_years"] == horizon
        }
        intervention_weighted.append({
            "horizon_years": horizon,
            "intervention_weighted_equal_trade_scenario_mean": sum(
                float(intervention[case]) * values[case] for case in CASES
            ),
        })

    checks = {
        "conditional_row_count": len(rows) == 150,
        "diagnostic_cell_count": len(result["module_diagnostics"]) == 30,
        "all_probability_outputs_finite": finite,
        "all_probability_outputs_bounded": bounded,
        "all_finsler_runtime_gates": not diagnostic_errors,
        "complete_model_gate": (
            trace.get("gate_status") == "PASS"
            and not trace.get("blocking_gaps")
            and static_audit.get("simulation_authorized") is True
            and freeze.get("status") == "FROZEN"
        ),
        "runtime_validation": result["verification"].get("runtime_validation") == "TAIWAN_V74_RUNTIME_VALIDATION: PASS",
    }
    summary = {
        "model_version": "V7.4-FINSLER-CLOSED-LOOP",
        "interpretation": "registered trade scenarios are equally weighted sensitivity cases, not empirically estimated scenario probabilities",
        "case_horizon_conditional_broad_control": case_horizon,
        "intervention_probability_means": intervention,
        "intervention_weighted_conditional_broad_control": intervention_weighted,
        "long_term_competing_risk_checkpoints": result["verification"]["long_term_checkpoints"],
        "finsler_runtime_envelope": {
            "maximum_randers_one_form_norm": max(item["maximum_randers_one_form_norm"] for item in finsler_diags),
            "minimum_finsler_cost": min(item["minimum_finsler_cost"] for item in finsler_diags),
            "maximum_eikonal_residual": max(item["maximum_eikonal_residual"] for item in finsler_diags),
            "minimum_path_reliability": min(item["minimum_path_reliability"] for item in finsler_diags),
        },
        "matched_v73_v74_ablation": {
            "matched_row_count": len(differences),
            "mean_change_v74_minus_v73": statistics.mean(differences),
            "minimum_change": min(differences),
            "maximum_change": max(differences),
        },
    }
    SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    payload = {
        "model_version": "V7.4-FINSLER-CLOSED-LOOP",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "diagnostic_errors": diagnostic_errors,
        "result_sha256": result["verification"]["result_sha256"],
    }
    TARGET.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"TAIWAN_V74_RESULT_AUDIT: {payload['status']}")
    if payload["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
