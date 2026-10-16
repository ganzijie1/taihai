from __future__ import annotations

import json
import math
from pathlib import Path
import statistics

from simulate_taiwan_multidomain_v42 import CASES


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"


def main() -> None:
    result = json.loads((OUT / "taiwan_v77_simulation.json").read_text(encoding="utf-8"))
    rows = result["conditional"]
    columns = (
        "conditional_broad_control_rate", "kernel_uncertainty_low_rate",
        "kernel_uncertainty_mid_rate", "kernel_uncertainty_high_rate",
        "mechanical_majority_control_share", "status_quo_settlement_share",
        "unification_settlement_share",
    )
    finite = all(math.isfinite(float(row[key])) for row in rows for key in columns)
    bounded = all(-1e-12 <= float(row[key]) <= 1.0 + 1e-12 for row in rows for key in columns)
    errors: list[str] = []
    envelopes = {
        "ib_residual": [], "fgw_residual": [], "ollivier_residual": [],
        "hodge_residual": [], "oscillator_order_min": [], "tda_h1": [],
        "ces_effect": [],
    }
    for cell, item in result["module_diagnostics"].items():
        mfg = item["mfg"]
        belief = mfg["partial_observation"]["v77_information_bottleneck_active_observation"]
        network = mfg["network_geometry"]["v77_complete_network_geometry"]
        oscillator = mfg["v77_coupled_oscillator"]
        tda = mfg["v76_online_tda"]
        ces = item["macro"]["cge_extension_diagnostics"]["v77_nested_ces"]
        gates = (
            belief["information_bottleneck_updates"] == 360
            and belief["active_observation_updates"] == 360
            and belief["maximum_ib_fixed_point_residual"] <= 1e-7
            and belief["maximum_attention_kkt_residual"] <= 1e-10
            and network["fgw_updates"] == 360
            and network["directed_ollivier_updates"] == 360
            and network["network_hodge_updates"] == 360
            and network["maximum_fgw_marginal_residual"] <= 2e-6
            and network["maximum_ollivier_marginal_residual"] <= 2e-4
            and network["maximum_hodge_reconstruction_residual"] <= 1e-10
            and oscillator["updates"] == 360
            and oscillator["maximum_graphon_kernel_change"] > 0.0
            and tda["updates"] == 30
            and tda["hold_months"] == 330
            and tda["maximum_h0_barcode_total_lifetime"] > 0.0
            and tda["maximum_bipersistence_image_rank"] > 0
            and ces["updates"] > 0
            and ces["maximum_cobb_douglas_limit_residual"] <= 2e-6
            and mfg.get("nonconverged_months", 1) == 0
            and mfg.get("max_fixed_point_residual", math.inf) <= 5e-4
        )
        if not gates:
            errors.append(cell)
        envelopes["ib_residual"].append(belief["maximum_ib_fixed_point_residual"])
        envelopes["fgw_residual"].append(network["maximum_fgw_marginal_residual"])
        envelopes["ollivier_residual"].append(network["maximum_ollivier_marginal_residual"])
        envelopes["hodge_residual"].append(network["maximum_hodge_reconstruction_residual"])
        envelopes["oscillator_order_min"].append(oscillator["minimum_global_order"])
        envelopes["tda_h1"].append(tda["maximum_vietoris_rips_h1_betti"])
        envelopes["ces_effect"].append(ces["maximum_capacity_change_from_baseline"])
    trace = json.loads((OUT / "taiwan_v77_traceability.json").read_text(encoding="utf-8"))
    static = json.loads((OUT / "taiwan_v77_static_coupling_audit.json").read_text(encoding="utf-8"))
    freeze = json.loads((OUT / "taiwan_v77_frozen_specification.json").read_text(encoding="utf-8"))
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
        "runtime_validation": result["verification"].get("runtime_validation") == "TAIWAN_V77_RUNTIME_VALIDATION: PASS",
        "checkpoint_contract": result["verification"].get("checkpoint_contract", {}).get("cell_count") == 30,
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
        "model_version": "V7.7-COMPLETE-THEORY-CLOSURE",
        "interpretation": "registered trade scenarios are equal-weight sensitivity cases, not estimated scenario probabilities",
        "case_horizon_conditional_broad_control": case_horizon,
        "intervention_probability_means": result["verification"]["intervention_probability_means"],
        "long_term_competing_risk_checkpoints": result["verification"]["long_term_checkpoints"],
        "runtime_envelope": {
            "maximum_information_bottleneck_residual": max(envelopes["ib_residual"]),
            "maximum_fgw_marginal_residual": max(envelopes["fgw_residual"]),
            "maximum_ollivier_marginal_residual": max(envelopes["ollivier_residual"]),
            "maximum_hodge_reconstruction_residual": max(envelopes["hodge_residual"]),
            "minimum_oscillator_order": min(envelopes["oscillator_order_min"]),
            "maximum_tda_h1": max(envelopes["tda_h1"]),
            "maximum_nested_ces_capacity_effect": max(envelopes["ces_effect"]),
        },
    }
    (OUT / "taiwan_v77_result_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    payload = {
        "model_version": "V7.7-COMPLETE-THEORY-CLOSURE",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "diagnostic_errors": errors,
        "result_sha256": result["verification"]["result_sha256"],
    }
    (OUT / "taiwan_v77_result_audit.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"TAIWAN_V77_RESULT_AUDIT: {payload['status']}")
    if payload["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
