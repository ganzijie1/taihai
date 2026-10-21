from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import statistics

from simulate_dual_circulation_v44 import v43
from simulate_taiwan_multidomain_v42 import CASES
import simulate_taiwan_v80_checkpointed as checkpoints


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
RESULT = OUT / "taiwan_v80_simulation.json"
CSV = OUT / "taiwan_v80_results.csv"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main() -> None:
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    rows = result["conditional"]
    columns = (
        "conditional_broad_control_rate",
        "kernel_uncertainty_low_rate",
        "kernel_uncertainty_mid_rate",
        "kernel_uncertainty_high_rate",
        "mechanical_majority_control_share",
        "status_quo_settlement_share",
        "unification_settlement_share",
    )
    finite = all(
        math.isfinite(float(row[key])) for row in rows for key in columns
    )
    bounded = all(
        -1.0e-12 <= float(row[key]) <= 1.0 + 1.0e-12
        for row in rows for key in columns
    )
    monotone_errors: list[str] = []
    for trade in v43.SCENARIOS:
        for case in CASES:
            selected = sorted(
                (
                    int(row["horizon_years"]),
                    float(row["conditional_broad_control_rate"]),
                )
                for row in rows
                if row["trade_scenario"] == trade and row["case"] == case
            )
            if any(right + 1.0e-12 < left for (_, left), (_, right) in zip(selected, selected[1:])):
                monotone_errors.append(f"{trade}|{case}")

    runtime_errors: list[str] = []
    envelope = {
        "mfg_residual": [], "field_residual": [], "memory_radius": [],
        "dirac_power": [], "mechanics_constraint": [], "balance_residual": [],
        "binder_scale_spread": [], "msrjd_action": [],
    }
    for cell, item in result["module_diagnostics"].items():
        mfg = item.get("mfg", {})
        field = mfg.get("v80_non_equilibrium_statistical_field")
        mechanics = mfg.get("v80_analytical_mechanics")
        evidence = mfg.get("v80_soc_evidence")
        gates = field is not None and mechanics is not None and evidence is not None
        if gates:
            gates = (
                field["updates"] >= 360
                and field["feedback_updates"] >= 360
                and field["dmft"]["fit_updates"] > 0
                and field["dmft"]["maximum_memory_spectral_radius"] < 1.0
                and field["dmft"]["innovation_covariance_minimum_eigenvalue"] >= -1.0e-10
                and field["cavity_equilibrium_resolve_updates"] >= 30
                and mechanics["updates"] >= 360
                and mechanics["maximum_constraint_residual"] <= 1.0e-10
                and mechanics["maximum_dirac_power_residual"] <= 1.0e-10
                and evidence["claim_boundary"] == "quasi-critical cascade, not confirmed SOC"
                and not field["terminal_probability_adjustment"]
                and not mechanics["terminal_probability_adjustment"]
                and not evidence["terminal_probability_adjustment"]
                and mfg.get("nonconverged_months", 1) == 0
                and mfg.get("max_fixed_point_residual", math.inf) <= 5.0e-4
            )
        if not gates:
            runtime_errors.append(cell)
            continue
        envelope["mfg_residual"].append(mfg["max_fixed_point_residual"])
        envelope["field_residual"].append(field["maximum_inner_fixed_point_residual"])
        envelope["memory_radius"].append(field["dmft"]["maximum_memory_spectral_radius"])
        envelope["dirac_power"].append(mechanics["maximum_dirac_power_residual"])
        envelope["mechanics_constraint"].append(mechanics["maximum_constraint_residual"])
        envelope["balance_residual"].append(mechanics["maximum_discrete_energy_balance_residual"])
        envelope["binder_scale_spread"].append(field["maximum_binder_scale_spread"])
        envelope["msrjd_action"].append(field["maximum_msrjd_action_density"])

    invalid_checkpoints: list[str] = []
    for i, scenario in enumerate(checkpoints.SCENARIOS):
        for j, case in enumerate(CASES):
            path = checkpoints.checkpoint_path(i, j)
            if not path.exists():
                invalid_checkpoints.append(f"{i}:{j}:missing")
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not checkpoints.valid(payload, 128, 360, scenario, case):
                invalid_checkpoints.append(f"{i}:{j}:invalid")

    trace = json.loads((OUT / "taiwan_v80_traceability.json").read_text(encoding="utf-8"))
    static = json.loads((OUT / "taiwan_v80_static_coupling_audit.json").read_text(encoding="utf-8"))
    freeze = json.loads((OUT / "taiwan_v80_frozen_specification.json").read_text(encoding="utf-8"))
    intervention = result["verification"]["intervention_probability_means"]
    checks = {
        "conditional_row_count": len(rows) == 150,
        "diagnostic_cell_count": len(result["module_diagnostics"]) == 30,
        "checkpoint_count_and_hashes": not invalid_checkpoints,
        "probabilities_finite": finite,
        "probabilities_bounded": bounded,
        "cumulative_broad_control_monotone": not monotone_errors,
        "intervention_simplex": abs(sum(intervention.values()) - 1.0) <= 1.0e-10,
        "runtime_gates": not runtime_errors,
        "complete_model_gate": (
            trace.get("gate_status") == "PASS"
            and not trace.get("blocking_gaps")
            and static.get("simulation_authorized") is True
            and freeze.get("status") == "FROZEN"
        ),
        "runtime_validation": result["verification"].get("runtime_validation")
        == "TAIWAN_V80_RUNTIME_VALIDATION: PASS",
        "checkpoint_contract": result["verification"].get(
            "checkpoint_contract", {}
        ).get("cell_count") == 30,
        "csv_hash": result["verification"].get("result_sha256") == sha256(CSV),
        "no_posthoc_adjustment": not result["verification"]["v80_upgrade"][
            "posthoc_probability_adjustment"
        ],
    }

    case_horizon: list[dict] = []
    mixture_by_horizon: list[dict] = []
    for horizon in (5, 10, 15, 20, 30):
        case_means: dict[str, float] = {}
        for case in CASES:
            selected = [
                row for row in rows
                if row["case"] == case and row["horizon_years"] == horizon
            ]
            broad = [float(row["conditional_broad_control_rate"]) for row in selected]
            mechanical = [float(row["mechanical_majority_control_share"]) for row in selected]
            unification = [float(row["unification_settlement_share"]) for row in selected]
            status_quo = [float(row["status_quo_settlement_share"]) for row in selected]
            case_means[case] = statistics.mean(broad)
            case_horizon.append({
                "case": case,
                "horizon_years": horizon,
                "equal_weight_trade_scenario_mean": statistics.mean(broad),
                "minimum": min(broad),
                "maximum": max(broad),
                "mechanical_majority_mean": statistics.mean(mechanical),
                "unification_settlement_mean": statistics.mean(unification),
                "status_quo_settlement_mean": statistics.mean(status_quo),
            })
        mixture_by_horizon.append({
            "horizon_years": horizon,
            "factorized_mean_sensitivity_indicator": sum(
                intervention[case] * case_means[case] for case in CASES
            ),
        })

    summary = {
        "model_version": "V8.0-STATISTICAL-FIELD-ANALYTICAL-MECHANICS",
        "conditioning": (
            "broad-control rates are conditional on conflict and the named intervention regime"
        ),
        "interpretation": (
            "registered trade scenarios are equal-weight sensitivity cases, not estimated scenario probabilities"
        ),
        "case_horizon_conditional_broad_control": case_horizon,
        "intervention_probability_means": intervention,
        "intervention_mixture": mixture_by_horizon,
        "intervention_mixture_boundary": (
            "mean intervention shares multiplied by case-level conditional means; "
            "not a pathwise joint estimate and not an unconditional probability"
        ),
        "force_onset_mean": result["verification"]["force_onset_mean"],
        "long_term_competing_risk_checkpoints": result["verification"][
            "long_term_checkpoints"
        ],
        "runtime_envelope": {
            "maximum_mfg_fixed_point_residual": max(envelope["mfg_residual"]),
            "maximum_field_fixed_point_residual": max(envelope["field_residual"]),
            "maximum_dmft_memory_radius": max(envelope["memory_radius"]),
            "maximum_dirac_power_residual": max(envelope["dirac_power"]),
            "maximum_mechanics_constraint_residual": max(envelope["mechanics_constraint"]),
            "maximum_discrete_energy_balance_residual": max(envelope["balance_residual"]),
            "maximum_binder_scale_spread": max(envelope["binder_scale_spread"]),
            "maximum_msrjd_action_density": max(envelope["msrjd_action"]),
        },
        "result_csv_sha256": sha256(CSV),
        "result_json_sha256": sha256(RESULT),
        "publication_validation": {
            "status": "BLOCKED",
            "gaps": [
                "V8.0 statistical-field outcome ablation not yet run",
                "V8.0 analytical-mechanics outcome ablation not yet run",
                "checkpoint aggregate rows omit case labels and are excluded from interpretation",
            ],
        },
    }
    (OUT / "taiwan_v80_result_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    payload = {
        "model_version": summary["model_version"],
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "runtime_errors": runtime_errors,
        "monotonicity_errors": monotone_errors,
        "invalid_checkpoints": invalid_checkpoints,
        "result_csv_sha256": sha256(CSV),
        "result_json_sha256": sha256(RESULT),
        "artifact_integrity_status": "PASS" if all(checks.values()) else "FAIL",
        "publication_gate": "BLOCKED",
        "publication_gaps": [
            "V8.0 statistical-field outcome ablation not yet run",
            "V8.0 analytical-mechanics outcome ablation not yet run",
            "checkpoint aggregate schema does not retain case labels",
        ],
    }
    (OUT / "taiwan_v80_result_audit.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"TAIWAN_V80_RESULT_AUDIT: {payload['status']}")
    if payload["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
