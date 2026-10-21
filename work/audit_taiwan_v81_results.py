from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import statistics

from simulate_dual_circulation_v44 import v43
from simulate_taiwan_multidomain_v42 import CASES
import simulate_taiwan_v81_checkpointed as checkpoints


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
RESULT = OUT / "taiwan_v81_simulation.json"
CSV = OUT / "taiwan_v81_results.csv"
PARENT = OUT / "taiwan_v80_simulation.json"
INHERITED_ABLATION_AUDIT = OUT / "taiwan_v81_inherited_ablation_audit.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def row_key(row: dict) -> tuple[str, str, int]:
    return row["trade_scenario"], row["case"], int(row["horizon_years"])


def main() -> None:
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    parent = json.loads(PARENT.read_text(encoding="utf-8"))
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
    finite = all(math.isfinite(float(row[key])) for row in rows for key in columns)
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
            if any(
                right + 1.0e-12 < left
                for (_, left), (_, right) in zip(selected, selected[1:])
            ):
                monotone_errors.append(f"{trade}|{case}")

    runtime_errors: list[str] = []
    envelope = {
        "simplex": [],
        "spd_eigenvalue": [],
        "spd_condition": [],
        "normal_residual": [],
        "support_shift": [],
        "separatrix": [],
        "chart_transitions": [],
    }
    for cell, item in result["module_diagnostics"].items():
        mfg = item.get("mfg", {})
        manifold = mfg.get("v81_differential_manifold")
        gates = manifold is not None
        if gates:
            gates = (
                manifold["updates"] >= 360
                and manifold["feedback_updates"] >= 360
                and manifold["simplex_updates"] >= 360
                and manifold["spd_updates"] >= 360
                and manifold["slow_manifold_updates"] >= 360
                and manifold["maximum_probability_simplex_residual"] <= 1.0e-12
                and manifold["minimum_spd_eigenvalue"] > 0.0
                and manifold["maximum_spd_condition_number"] < 1.0e10
                and manifold["minimum_attractor_separatrix_margin"] >= 0.0
                and manifold["maximum_support_output_shift"] > 0.0
                and not manifold["terminal_probability_adjustment"]
                and mfg.get("nonconverged_months", 1) == 0
                and mfg.get("max_fixed_point_residual", math.inf) <= 5.0e-4
            )
        if not gates:
            runtime_errors.append(cell)
            continue
        envelope["simplex"].append(manifold["maximum_probability_simplex_residual"])
        envelope["spd_eigenvalue"].append(manifold["minimum_spd_eigenvalue"])
        envelope["spd_condition"].append(manifold["maximum_spd_condition_number"])
        envelope["normal_residual"].append(manifold["maximum_slow_normal_residual"])
        envelope["support_shift"].append(manifold["maximum_support_output_shift"])
        envelope["separatrix"].append(manifold["minimum_attractor_separatrix_margin"])
        envelope["chart_transitions"].append(manifold["hybrid_chart_transition_count"])

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

    trace = json.loads((OUT / "taiwan_v81_traceability.json").read_text(encoding="utf-8"))
    static = json.loads((OUT / "taiwan_v81_static_coupling_audit.json").read_text(encoding="utf-8"))
    freeze = json.loads((OUT / "taiwan_v81_frozen_specification.json").read_text(encoding="utf-8"))
    intervention = result["verification"]["intervention_probability_means"]
    aggregate_labeled = all(
        "trade_scenario" in row and "case" in row for row in result["aggregate"]
    )

    current_by_key = {row_key(row): row for row in rows}
    parent_by_key = {row_key(row): row for row in parent["conditional"]}
    common = sorted(set(current_by_key) & set(parent_by_key))
    deltas = [
        float(current_by_key[key]["conditional_broad_control_rate"])
        - float(parent_by_key[key]["conditional_broad_control_rate"])
        for key in common
    ]
    manifold_ablation = {
        "baseline": "V8.1 complete model",
        "ablation": "V8.0 same seeds and paths; differential-manifold layer absent",
        "matched_cells": len(common),
        "mean_difference": statistics.mean(deltas),
        "mean_absolute_difference": statistics.mean(map(abs, deltas)),
        "minimum_difference": min(deltas),
        "maximum_difference": max(deltas),
        "interpretation": (
            "paired structural ablation, not a confidence interval or causal estimate"
        ),
    }
    (OUT / "taiwan_v81_manifold_outcome_ablation.json").write_text(
        json.dumps(manifold_ablation, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    checks = {
        "conditional_row_count": len(rows) == 150,
        "aggregate_row_count_and_labels": len(result["aggregate"]) == 150 and aggregate_labeled,
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
        == "TAIWAN_V81_RUNTIME_VALIDATION: PASS",
        "checkpoint_contract": result["verification"].get(
            "checkpoint_contract", {}
        ).get("cell_count") == 30,
        "csv_hash": result["verification"].get("result_sha256") == sha256(CSV),
        "paired_manifold_ablation": len(common) == 150,
        "no_posthoc_adjustment": not result["verification"]["v81_upgrade"][
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
            case_means[case] = statistics.mean(broad)
            case_horizon.append({
                "case": case,
                "horizon_years": horizon,
                "equal_weight_trade_scenario_mean": statistics.mean(broad),
                "minimum": min(broad),
                "maximum": max(broad),
            })
        mixture_by_horizon.append({
            "horizon_years": horizon,
            "factorized_mean_sensitivity_indicator": sum(
                intervention[case] * case_means[case] for case in CASES
            ),
        })

    inherited_ablation = None
    if INHERITED_ABLATION_AUDIT.exists():
        candidate = json.loads(
            INHERITED_ABLATION_AUDIT.read_text(encoding="utf-8")
        )
        if (
            candidate.get("status") == "PASS"
            and candidate.get("baseline_result_json_sha256") == sha256(RESULT)
            and candidate.get("baseline_result_csv_sha256") == sha256(CSV)
        ):
            inherited_ablation = candidate
    publication_gaps = [] if inherited_ablation else [
        "V8.0 statistical-field one-at-a-time outcome ablation not yet run",
        "V8.0 analytical-mechanics one-at-a-time outcome ablation not yet run",
    ]
    publication_status = "PASS" if not publication_gaps else "BLOCKED"
    summary = {
        "model_version": "V8.1-DIFFERENTIAL-MANIFOLD-CONSTRAINTS",
        "conditioning": (
            "broad-control rates are conditional on conflict and the named intervention regime"
        ),
        "case_horizon_conditional_broad_control": case_horizon,
        "intervention_probability_means": intervention,
        "intervention_mixture": mixture_by_horizon,
        "force_onset_mean": result["verification"]["force_onset_mean"],
        "long_term_competing_risk_checkpoints": result["verification"][
            "long_term_checkpoints"
        ],
        "manifold_outcome_ablation": manifold_ablation,
        "runtime_envelope": {
            "maximum_probability_simplex_residual": max(envelope["simplex"]),
            "minimum_spd_eigenvalue": min(envelope["spd_eigenvalue"]),
            "maximum_spd_condition_number": max(envelope["spd_condition"]),
            "maximum_slow_normal_residual": max(envelope["normal_residual"]),
            "maximum_support_output_shift": max(envelope["support_shift"]),
            "minimum_separatrix_margin": min(envelope["separatrix"]),
            "minimum_chart_transition_count": min(envelope["chart_transitions"]),
        },
        "result_csv_sha256": sha256(CSV),
        "result_json_sha256": sha256(RESULT),
        "inherited_v80_ablation_audit": inherited_ablation,
        "publication_validation": {
            "status": publication_status,
            "gaps": publication_gaps,
        },
    }
    (OUT / "taiwan_v81_result_summary.json").write_text(
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
        "publication_gate": publication_status,
        "publication_gaps": publication_gaps,
    }
    (OUT / "taiwan_v81_result_audit.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"TAIWAN_V81_RESULT_AUDIT: {payload['status']}")
    if payload["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
