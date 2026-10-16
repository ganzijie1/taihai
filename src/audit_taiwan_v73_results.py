from __future__ import annotations

import json
import math
from pathlib import Path
import statistics

from simulate_dual_circulation_v44 import v43
from simulate_taiwan_multidomain_v42 import CASES
import simulate_taiwan_fully_closed_v73 as model


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
TARGET = OUT / "taiwan_v73_result_audit.json"
SUMMARY = OUT / "taiwan_v73_result_summary.json"


def main() -> None:
    result = json.loads((OUT / "taiwan_v73_simulation.json").read_text(encoding="utf-8"))
    checkpoints = sorted((OUT / "taiwan_v73_scenario_checkpoints").glob("*.json"))
    expected = {(s, c) for s in tuple(v43.SCENARIOS) for c in tuple(CASES)}
    observed = set()
    hashes = set()
    for path in checkpoints:
        meta = json.loads(path.read_text(encoding="utf-8"))["checkpoint"]
        observed.add((meta["scenario"], meta["case"]))
        hashes.add(meta["freeze_sha256"])
    rows = result["conditional"]
    probability_columns = (
        "conditional_broad_control_rate", "kernel_uncertainty_low_rate",
        "kernel_uncertainty_mid_rate", "kernel_uncertainty_high_rate",
        "mechanical_majority_control_share", "status_quo_settlement_share",
        "unification_settlement_share",
    )
    finite = all(
        math.isfinite(float(row[key]))
        for row in rows for key in probability_columns
    )
    bounded = all(
        -1.0e-12 <= float(row[key]) <= 1.0 + 1.0e-12
        for row in rows for key in probability_columns
    )
    diagnostic_errors = []
    for cell, item in result["module_diagnostics"].items():
        graph = item["operations"]["continuous_graph_network_v73"]
        inherited = item["operations"]["combinatorial_network_v71"]
        if not (
            graph["months"] == 360
            and graph["maximum_metric_balance_residual"] <= 1.0e-10
            and graph["maximum_cfl"] < 1.0
            and graph["maximum_signal_bound_violation"] <= 1.0e-12
            and graph["maximum_control_bound_violation"] <= 1.0e-12
            and graph["maximum_finite_graphon_rollback_error"] <= 0.10
            and inherited["maximum_stock_residual"] <= 1.0e-9
            and not graph["terminal_probability_adjustment"]
        ):
            diagnostic_errors.append(cell)
    checks = {
        "checkpoint_count": len(checkpoints) == 30,
        "scenario_case_cartesian_product": observed == expected,
        "single_current_freeze_hash": hashes == {model.sha256(model.FREEZE)},
        "conditional_row_count": len(rows) == 150,
        "diagnostic_cell_count": len(result["module_diagnostics"]) == 30,
        "all_probability_outputs_finite": finite,
        "all_probability_outputs_bounded": bounded,
        "all_continuous_graph_runtime_gates": not diagnostic_errors,
        "merged_checkpoint_contract_cell_count": result["verification"]["checkpoint_contract"]["cell_count"] == 30,
        "functional_analysis_is_per_checkpoint_template": result["verification"]["functional_analysis"]["cell_count"] == 1,
        "no_probability_postprocessing": not result["verification"]["checkpoint_contract"]["posthoc_probability_adjustment"],
    }

    case_horizon = []
    for case in CASES:
        for horizon in (5, 10, 15, 20, 30):
            selected = [r for r in rows if r["case"] == case and r["horizon_years"] == horizon]
            case_horizon.append({
                "case": case, "horizon_years": horizon,
                "equal_weight_trade_scenario_mean": sum(float(r["conditional_broad_control_rate"]) for r in selected) / len(selected),
                "minimum": min(float(r["conditional_broad_control_rate"]) for r in selected),
                "maximum": max(float(r["conditional_broad_control_rate"]) for r in selected),
            })
    intervention = result["verification"]["intervention_probability_means"]
    expected_by_horizon = []
    for horizon in (5, 10, 15, 20, 30):
        values = {x["case"]: x["equal_weight_trade_scenario_mean"] for x in case_horizon if x["horizon_years"] == horizon}
        expected_by_horizon.append({
            "horizon_years": horizon,
            "intervention_weighted_equal_trade_scenario_mean": sum(intervention[c] * values[c] for c in CASES),
        })
    graph_diags = [item["operations"]["continuous_graph_network_v73"] for item in result["module_diagnostics"].values()]
    matched_differences = []
    matched_cells = []
    v72_dir = OUT / "taiwan_v72_scenario_checkpoints"
    for old_path in sorted(v72_dir.glob("*.json")):
        new_path = OUT / "taiwan_v73_scenario_checkpoints" / old_path.name
        if not new_path.exists():
            continue
        old = json.loads(old_path.read_text(encoding="utf-8"))
        new = json.loads(new_path.read_text(encoding="utf-8"))
        old_rows = {
            (r["case"], r["trade_scenario"], r["horizon_years"]): r
            for r in old["rows"]
        }
        cell_differences = [
            float(r["conditional_broad_control_rate"])
            - float(old_rows[(r["case"], r["trade_scenario"], r["horizon_years"])]["conditional_broad_control_rate"])
            for r in new["rows"]
        ]
        matched_differences.extend(cell_differences)
        matched_cells.append(old_path.stem)
    summary = {
        "model_version": "V7.3-CONTINUOUS-GRAPH",
        "interpretation": "registered trade scenarios are equally weighted sensitivity cases, not empirically estimated scenario probabilities",
        "case_horizon_conditional_broad_control": case_horizon,
        "intervention_probability_means": intervention,
        "intervention_weighted_conditional_broad_control": expected_by_horizon,
        "long_term_competing_risk_checkpoints": result["verification"]["long_term_checkpoints"],
        "continuous_graph_runtime_envelope": {
            "maximum_mass_balance_residual": max(d["maximum_metric_balance_residual"] for d in graph_diags),
            "maximum_cfl": max(d["maximum_cfl"] for d in graph_diags),
            "maximum_finite_graphon_rollback_error": max(d["maximum_finite_graphon_rollback_error"] for d in graph_diags),
            "minimum_capacity_gate": min(d["minimum_continuous_capacity_gate"] for d in graph_diags),
            "minimum_enterprise_gate": min(d["minimum_enterprise_gate"] for d in graph_diags),
        },
        "matched_v72_v73_ablation": {
            "scope_warning": "V7.2 has only a partial 12-cell checkpoint set; this is a matched diagnostic subset, not a full-version estimate",
            "matched_cells": matched_cells,
            "matched_row_count": len(matched_differences),
            "mean_change_v73_minus_v72": statistics.mean(matched_differences),
            "minimum_change": min(matched_differences),
            "maximum_change": max(matched_differences),
        },
    }
    SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    payload = {
        "model_version": "V7.3-CONTINUOUS-GRAPH",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "diagnostic_errors": diagnostic_errors,
        "result_sha256": result["verification"]["result_sha256"],
    }
    TARGET.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"TAIWAN_V73_RESULT_AUDIT: {payload['status']}")
    if payload["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
