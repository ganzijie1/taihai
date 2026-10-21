from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
ABLATION_ROOT = OUT / "taiwan_v81_inherited_ablations"
PLAN = OUT / "taiwan_v81_inherited_ablation_plan.json"
BASELINE = OUT / "taiwan_v81_simulation.json"
AUDIT = OUT / "taiwan_v81_inherited_ablation_audit.json"
SUMMARY = OUT / "taiwan_v81_inherited_ablation_summary.json"
ABLATIONS = ("no_statistical_field", "no_analytical_mechanics")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def row_key(row: dict) -> tuple[str, str, int]:
    return row["trade_scenario"], row["case"], int(row["horizon_years"])


def monotone(rows: list[dict]) -> bool:
    grouped: dict[tuple[str, str], list[tuple[int, float]]] = {}
    for row in rows:
        grouped.setdefault((row["trade_scenario"], row["case"]), []).append((
            int(row["horizon_years"]),
            float(row["conditional_broad_control_rate"]),
        ))
    return all(
        all(right + 1.0e-12 >= left for (_, left), (_, right) in zip(
            sorted(values), sorted(values)[1:]
        ))
        for values in grouped.values()
    )


def main() -> None:
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    baseline_rows = {row_key(row): row for row in baseline["conditional"]}
    columns = (
        "conditional_broad_control_rate",
        "kernel_uncertainty_low_rate",
        "kernel_uncertainty_mid_rate",
        "kernel_uncertainty_high_rate",
        "mechanical_majority_control_share",
        "status_quo_settlement_share",
        "unification_settlement_share",
    )
    checks: dict[str, bool] = {
        "plan_frozen": plan.get("status") == "FROZEN",
        "baseline_json_hash": plan["baseline"]["result_json_sha256"] == sha256(BASELINE),
        "baseline_csv_hash": plan["baseline"]["result_csv_sha256"] == sha256(
            OUT / "taiwan_v81_results.csv"
        ),
    }
    summaries = {}
    for ablation in ABLATIONS:
        result_path = ABLATION_ROOT / ablation / "simulation.json"
        csv_path = ABLATION_ROOT / ablation / "results.csv"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        rows = result["conditional"]
        current = {row_key(row): row for row in rows}
        diagnostics = result["module_diagnostics"]
        finite = all(
            math.isfinite(float(row[column])) for row in rows for column in columns
        )
        bounded = all(
            -1.0e-12 <= float(row[column]) <= 1.0 + 1.0e-12
            for row in rows for column in columns
        )
        diagnostic_switches = []
        for item in diagnostics.values():
            mfg = item.get("mfg", {})
            field = mfg.get("v80_non_equilibrium_statistical_field", {})
            mechanics = mfg.get("v80_analytical_mechanics", {})
            if ablation == "no_statistical_field":
                diagnostic_switches.append(
                    field.get("enabled") is False
                    and field.get("updates") == 0
                    and mechanics.get("enabled") is True
                    and mechanics.get("updates", 0) >= 360
                )
            else:
                diagnostic_switches.append(
                    mechanics.get("enabled") is False
                    and mechanics.get("updates") == 0
                    and field.get("enabled") is True
                    and field.get("updates", 0) >= 360
                )
        common = sorted(set(baseline_rows) & set(current))
        deltas = [
            float(baseline_rows[key]["conditional_broad_control_rate"])
            - float(current[key]["conditional_broad_control_rate"])
            for key in common
        ]
        checkpoint_count = len(list((ABLATION_ROOT / ablation / "checkpoints").glob("*.json")))
        contract = result["verification"].get("ablation_contract", {})
        local_checks = {
            "rows": len(rows) == 150,
            "aggregate_labels": len(result["aggregate"]) == 150 and all(
                "trade_scenario" in row and "case" in row
                for row in result["aggregate"]
            ),
            "diagnostics": len(diagnostics) == 30,
            "checkpoints": checkpoint_count == 30,
            "finite": finite,
            "bounded": bounded,
            "monotone": monotone(rows),
            "paired_keys": len(common) == 150,
            "switch_isolation": all(diagnostic_switches),
            "runtime_validation": result["verification"].get("runtime_validation")
            == "TAIWAN_V81_ABLATION_RUNTIME_VALIDATION: PASS",
            "contract": contract.get("ablation") == ablation
            and contract.get("cell_count") == 30
            and contract.get("same_paths_seeds_and_clock") is True
            and contract.get("posthoc_probability_adjustment") is False,
            "csv_hash": result["verification"].get("result_sha256") == sha256(csv_path),
        }
        checks.update({f"{ablation}:{key}": value for key, value in local_checks.items()})
        summaries[ablation] = {
            "disabled_only": plan["ablations"][ablation]["disabled_only"],
            "matched_cells": len(common),
            "baseline_minus_ablation_mean": statistics.mean(deltas),
            "baseline_minus_ablation_mean_absolute": statistics.mean(map(abs, deltas)),
            "baseline_minus_ablation_minimum": min(deltas),
            "baseline_minus_ablation_maximum": max(deltas),
            "interpretation": "paired structural sensitivity, not a confidence interval or causal estimate",
            "result_json_sha256": sha256(result_path),
            "result_csv_sha256": sha256(csv_path),
        }
    status = "PASS" if all(checks.values()) else "FAIL"
    summary = {
        "status": status,
        "model_version": "V8.1-DIFFERENTIAL-MANIFOLD-CONSTRAINTS",
        "design": "one module disabled at a time; all other modules unchanged",
        "ablations": summaries,
        "publication_effect": "clears the two inherited V8.0 ablation gaps only when status is PASS",
    }
    SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    AUDIT.write_text(json.dumps({
        "status": status,
        "checks": checks,
        "plan_sha256": sha256(PLAN),
        "baseline_result_json_sha256": sha256(BASELINE),
        "baseline_result_csv_sha256": sha256(OUT / "taiwan_v81_results.csv"),
        "ablation_summary_sha256": sha256(SUMMARY),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"TAIWAN_V81_INHERITED_ABLATION_AUDIT: {status}")
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
