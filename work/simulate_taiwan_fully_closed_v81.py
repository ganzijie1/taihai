from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path

import simulate_taiwan_fully_closed_v80 as v80
from differential_manifold_mfg_v81 import DifferentialManifoldMFGV81


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
TRACE = OUT / "taiwan_v81_traceability.json"
AUDIT = OUT / "taiwan_v81_static_coupling_audit.json"
FREEZE = OUT / "taiwan_v81_frozen_specification.json"
OUT_CSV = OUT / "taiwan_v81_results.csv"
OUT_JSON = OUT / "taiwan_v81_simulation.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def complete_model_gate() -> tuple[bool, list[str]]:
    if not TRACE.exists() or not AUDIT.exists() or not FREEZE.exists():
        return False, ["V8.1 traceability, audit or frozen specification missing"]
    trace = json.loads(TRACE.read_text(encoding="utf-8"))
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    gaps = list(trace.get("blocking_gaps", []))
    passed = (
        trace.get("gate_status") == "PASS"
        and audit.get("simulation_authorized") is True
        and freeze.get("status") == "FROZEN"
        and not gaps
    )
    return passed, gaps


def validate_v81(diagnostics: dict, months: int) -> None:
    v80.validate_v80(diagnostics, months)
    errors: list[str] = []
    for cell, item in diagnostics.items():
        manifold = item.get("mfg", {}).get("v81_differential_manifold")
        if manifold is None:
            errors.append(f"{cell}: missing V8.1 differential manifold")
            continue
        checks = {
            "manifold clock": manifold["updates"] >= months,
            "feedback clock": manifold["feedback_updates"] >= months,
            "simplex clock": (
                not manifold["simplex_enabled"]
                or manifold["simplex_updates"] >= months
            ),
            "simplex residual": manifold[
                "maximum_probability_simplex_residual"
            ] <= 1.0e-12,
            "positive covariance": manifold["minimum_spd_eigenvalue"] > 0.0,
            "finite covariance condition": manifold[
                "maximum_spd_condition_number"
            ] < 1.0e10,
            "slow clock": (
                not manifold["slow_manifold_enabled"]
                or manifold["slow_manifold_updates"] >= months
            ),
            "active support retraction": (
                not manifold["simplex_enabled"]
                or manifold["maximum_support_output_shift"] > 0.0
            ),
            "finite separatrix": (
                not manifold["basin_diagnostics_enabled"]
                or manifold["minimum_attractor_separatrix_margin"] >= 0.0
            ),
            "no posthoc": not manifold["terminal_probability_adjustment"],
        }
        errors.extend(
            f"{cell}: {name}" for name, passed in checks.items() if not passed
        )
    if errors:
        raise RuntimeError("V8.1 runtime validation failed: " + "; ".join(errors))


def run(paths: int = 128, months: int = 360, **kwargs):
    gate, gaps = complete_model_gate()
    if not gate:
        raise RuntimeError(
            "V8.1 complete-model gate failed: "
            + "; ".join(gaps or ["gate not passed"])
        )
    old_class = v80.StatisticalFieldMFGV80
    old_paths = (v80.TRACE, v80.AUDIT, v80.FREEZE)
    v80.StatisticalFieldMFGV80 = DifferentialManifoldMFGV81
    v80.TRACE, v80.AUDIT, v80.FREEZE = TRACE, AUDIT, FREEZE
    try:
        result = v80.run(paths=paths, months=months, **kwargs)
    finally:
        v80.StatisticalFieldMFGV80 = old_class
        v80.TRACE, v80.AUDIT, v80.FREEZE = old_paths
    validate_v81(result[2], months)
    return result


def write(rows, aggregate, diagnostics, verification) -> None:
    with OUT_CSV.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    verification["model_version"] = "V8.1-DIFFERENTIAL-MANIFOLD-CONSTRAINTS"
    verification["runtime_validation"] = "TAIWAN_V81_RUNTIME_VALIDATION: PASS"
    verification["result_sha256"] = sha256(OUT_CSV)
    verification["v81_upgrade"] = {
        "product_state_manifold": True,
        "fisher_rao_probability_retraction": True,
        "spd_innovation_covariance": True,
        "hybrid_chart_hysteresis": True,
        "slow_manifold_residual": True,
        "jump_fast_exit": True,
        "attractor_basin_diagnostic": True,
        "physical_state_owners_preserved": True,
        "posthoc_probability_adjustment": False,
        "frozen_specification_sha256": sha256(FREEZE),
    }
    OUT_JSON.write_text(json.dumps({
        "verification": verification,
        "aggregate": aggregate,
        "conditional": rows,
        "module_diagnostics": diagnostics,
    }, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    if os.environ.get("TAIWAN_ALLOW_COMPLETE_RUN") != "1":
        raise RuntimeError("V8.1 outcome simulation requires complete-run authorization")
    result = run(
        paths=int(os.environ.get("TAIWAN_V81_PATHS", "128")),
        months=int(os.environ.get("TAIWAN_V81_MONTHS", "360")),
    )
    write(*result)
    print("TAIWAN_V81_COMPLETE_RUN: PASS")


if __name__ == "__main__":
    main()
