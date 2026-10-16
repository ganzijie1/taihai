from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np

import simulate_taiwan_fully_closed_v73 as v73
from finsler_network_v74 import FinslerOperationalNetworkV74


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
TRACE = OUT / "taiwan_v74_traceability.json"
AUDIT = OUT / "taiwan_v74_static_coupling_audit.json"
FREEZE = OUT / "taiwan_v74_frozen_specification.json"
OUT_CSV = OUT / "taiwan_v74_results.csv"
OUT_JSON = OUT / "taiwan_v74_simulation.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def complete_model_gate() -> tuple[bool, list[str]]:
    if not TRACE.exists() or not AUDIT.exists() or not FREEZE.exists():
        return False, ["V7.4 traceability, audit or frozen specification missing"]
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


def validate_v74(diagnostics: dict, months: int) -> None:
    errors: list[str] = []
    for cell, item in diagnostics.items():
        finsler = item.get("operations", {}).get("finsler_network_v74")
        mfg = item.get("mfg", {})
        inherited = item.get("operations", {}).get("combinatorial_network_v71")
        if finsler is None or inherited is None:
            errors.append(f"{cell}: missing V7.4 Finsler network")
            continue
        gates = {
            "clock": finsler["months"] == months,
            "positive cost": finsler["minimum_finsler_cost"] > 0.0,
            "strong convexity": finsler["maximum_randers_one_form_norm"] < 1.0,
            "direction asymmetry": finsler["maximum_direction_asymmetry"] > 0.0,
            "zero drift": finsler["maximum_zero_drift_error"] <= 1.0e-12,
            "homogeneity": finsler["maximum_positive_homogeneity_residual"] <= 1.0e-12,
            "eikonal": finsler["maximum_eikonal_residual"] <= 1.0e-10,
            "capacity": inherited["maximum_capacity_residual"] <= 1.0e-10,
            "stock": inherited["maximum_stock_residual"] <= 1.0e-9,
            "MFG cost active": mfg.get("finsler_mobility_cost_active") is True,
            "MFG cost finite": np.isfinite(mfg.get("mean_last_mobility_cost", np.nan)),
            "single owner": not finsler["continuous_geometry_owns_physical_inventory"],
            "no posthoc": not finsler["terminal_probability_adjustment"],
        }
        errors.extend(f"{cell}: {name}" for name, passed in gates.items() if not passed)
    if errors:
        raise RuntimeError("V7.4 runtime validation failed: " + "; ".join(errors))


def run(paths: int = 128, months: int = 360, **kwargs):
    gate, gaps = complete_model_gate()
    if not gate:
        raise RuntimeError("V7.4 complete-model gate failed: " + "; ".join(gaps or ["gate not passed"]))
    old_class = v73.ContinuousGraphOperationalNetworkV73
    old_paths = (v73.TRACE, v73.AUDIT, v73.FREEZE)
    v73.ContinuousGraphOperationalNetworkV73 = FinslerOperationalNetworkV74
    v73.TRACE, v73.AUDIT, v73.FREEZE = TRACE, AUDIT, FREEZE
    try:
        result = v73.run(paths=paths, months=months, **kwargs)
    finally:
        v73.ContinuousGraphOperationalNetworkV73 = old_class
        v73.TRACE, v73.AUDIT, v73.FREEZE = old_paths
    validate_v74(result[2], months)
    return result


def write(rows, aggregate, diagnostics, verification) -> None:
    with OUT_CSV.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    verification["model_version"] = "V7.4-FINSLER-CLOSED-LOOP"
    verification["runtime_validation"] = "TAIWAN_V74_RUNTIME_VALIDATION: PASS"
    verification["result_sha256"] = sha256(OUT_CSV)
    OUT_JSON.write_text(json.dumps({
        "verification": verification,
        "aggregate": aggregate,
        "conditional": rows,
        "module_diagnostics": diagnostics,
    }, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    if os.environ.get("TAIWAN_ALLOW_COMPLETE_RUN") != "1":
        raise RuntimeError("V7.4 outcome simulation requires explicit complete-run authorization")
    result = run(
        paths=int(os.environ.get("TAIWAN_V74_PATHS", "128")),
        months=int(os.environ.get("TAIWAN_V74_MONTHS", "360")),
    )
    write(*result)
    print("TAIWAN_V74_COMPLETE_RUN: PASS")


if __name__ == "__main__":
    main()
