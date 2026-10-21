from __future__ import annotations

import hashlib
import json
import math
import statistics
from pathlib import Path

from simulate_taiwan_multidomain_v42 import CASES
import simulate_taiwan_v83_checkpointed as runs


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
RUN_ROOT = OUT / "taiwan_v83_runs"
AUDIT = OUT / "taiwan_v83_result_audit.json"
SUMMARY = OUT / "taiwan_v83_result_summary.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def row_key(row: dict) -> tuple[str, str, int]:
    return row["trade_scenario"], row["case"], int(row["horizon_years"])


def main() -> None:
    plan = json.loads(runs.PLAN.read_text(encoding="utf-8"))
    paths, months = int(plan["paths"]), int(plan["months"])
    plan_hash = sha256(runs.PLAN)
    missing, invalid_checkpoints, results = [], [], {}
    for variant in runs.VARIANTS:
        result_path = RUN_ROOT / variant / "simulation.json"
        csv_path = RUN_ROOT / variant / "results.csv"
        if not result_path.exists() or not csv_path.exists():
            missing.append(variant)
            continue
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        if payload.get("verification", {}).get("result_sha256") != sha256(csv_path):
            missing.append(f"{variant}:hash")
            continue
        results[variant] = payload
        for i, scenario in enumerate(runs.SCENARIOS):
            for j, case in enumerate(CASES):
                path = runs.checkpoint_path(variant, i, j)
                if not path.exists():
                    invalid_checkpoints.append(f"{variant}:{i}:{j}:missing")
                    continue
                checkpoint = json.loads(path.read_text(encoding="utf-8"))
                if not runs.valid(checkpoint, variant, paths, months, scenario, case, plan_hash):
                    invalid_checkpoints.append(f"{variant}:{i}:{j}:invalid")
    checks = {
        "all_variants_present": not missing,
        "all_270_checkpoints_valid": not invalid_checkpoints,
    }
    comparisons, runtime_errors = {}, []
    if "baseline" in results:
        baseline = results["baseline"]
        base_rows = baseline.get("conditional", [])
        finite_columns = (
            "conditional_broad_control_rate", "kernel_uncertainty_low_rate",
            "kernel_uncertainty_mid_rate", "kernel_uncertainty_high_rate",
            "mechanical_majority_control_share", "status_quo_settlement_share",
            "unification_settlement_share",
        )
        checks.update({
            "baseline_conditional_rows": len(base_rows) == 150,
            "baseline_aggregate_rows": len(baseline.get("aggregate", [])) == 150,
            "baseline_diagnostic_cells": len(baseline.get("module_diagnostics", {})) == 30,
            "baseline_probabilities_finite": all(
                math.isfinite(float(row[key])) for row in base_rows for key in finite_columns
            ),
            "baseline_probabilities_bounded": all(
                -1.0e-12 <= float(row[key]) <= 1.0 + 1.0e-12
                for row in base_rows for key in finite_columns
            ),
        })
        for cell, item in baseline.get("module_diagnostics", {}).items():
            particle = item.get("mfg", {}).get("v83_particle_methods")
            trade = item.get("macro", {}).get("cge_extension_diagnostics", {}).get(
                "v83_structural_international_economics"
            )
            if not particle or not trade:
                runtime_errors.append(cell)
                continue
            ok = (
                particle["months"] >= months
                and particle["maximum_weight_normalization_residual"] <= 1.0e-12
                and particle["minimum_effective_sample_size"] > 0.0
                and trade["months"] >= months
                and trade["maximum_source_share_residual"] <= 1.0e-12
                and trade["maximum_external_liquidity_bound_violation"] <= 1.0e-12
                and not particle["terminal_probability_adjustment"]
                and not trade["terminal_probability_adjustment"]
            )
            if not ok:
                runtime_errors.append(cell)
        checks["v83_runtime_invariants"] = not runtime_errors
        base_index = {row_key(row): row for row in base_rows}
        for variant, payload in results.items():
            if variant == "baseline":
                continue
            variant_index = {row_key(row): row for row in payload.get("conditional", [])}
            common = sorted(set(base_index) & set(variant_index))
            deltas = [
                float(base_index[key]["conditional_broad_control_rate"])
                - float(variant_index[key]["conditional_broad_control_rate"])
                for key in common
            ]
            comparisons[variant] = {
                "matched_rows": len(common),
                "mean_baseline_minus_variant": statistics.mean(deltas) if deltas else None,
                "mean_absolute_difference": statistics.mean(map(abs, deltas)) if deltas else None,
                "minimum_difference": min(deltas) if deltas else None,
                "maximum_difference": max(deltas) if deltas else None,
                "interpretation": "paired structural sensitivity, not a causal estimate",
            }
            checks[f"{variant}_matched_rows"] = len(common) == 150
        convergence = [
            comparisons.get("particle_n32", {}).get("mean_absolute_difference"),
            comparisons.get("particle_n128", {}).get("mean_absolute_difference"),
        ]
        checks["particle_count_convergence_reported"] = all(x is not None for x in convergence)
    status = "PASS" if checks and all(checks.values()) else "FAIL"
    publication_gaps = [] if status == "PASS" else (
        [f"missing or invalid variant: {item}" for item in missing]
        + invalid_checkpoints[:20] + runtime_errors[:20]
    )
    SUMMARY.write_text(json.dumps({
        "model_version": plan["model_version"],
        "conditioning": "broad-control rates are conditional on conflict and intervention regime",
        "ablation_comparisons": comparisons,
        "publication_validation": {
            "status": "PASS" if status == "PASS" else "BLOCKED",
            "gaps": publication_gaps,
        },
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    AUDIT.write_text(json.dumps({
        "model_version": plan["model_version"], "status": status,
        "checks": checks, "missing_variants": missing,
        "invalid_checkpoints": invalid_checkpoints,
        "runtime_errors": runtime_errors,
        "publication_gate": "PASS" if status == "PASS" else "BLOCKED",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"TAIWAN_V83_RESULT_AUDIT: {status}")
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
