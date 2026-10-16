from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path

import simulate_taiwan_fully_closed_v72 as v72
from continuous_graph_network_v73 import ContinuousGraphOperationalNetworkV73


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
TRACE = OUT / "taiwan_v73_traceability.json"
AUDIT = OUT / "taiwan_v73_static_coupling_audit.json"
FREEZE = OUT / "taiwan_v73_frozen_specification.json"
OUT_CSV = OUT / "taiwan_v73_results.csv"
OUT_JSON = OUT / "taiwan_v73_simulation.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def complete_model_gate() -> tuple[bool, list[str]]:
    if not TRACE.exists() or not AUDIT.exists() or not FREEZE.exists():
        return False, ["V7.3 traceability, audit or frozen specification missing"]
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


def validate_v73(diagnostics: dict, months: int) -> None:
    errors: list[str] = []
    for cell, item in diagnostics.items():
        graph = item.get("operations", {}).get("continuous_graph_network_v73")
        inherited = item.get("operations", {}).get("combinatorial_network_v71")
        if graph is None or inherited is None:
            errors.append(f"{cell}: missing V7.3 continuous graph")
            continue
        gates = {
            "clock": graph["months"] == months,
            "mass balance": graph["maximum_metric_balance_residual"] <= 1.0e-10,
            "positivity": graph["maximum_negative_density"] <= 1.0e-12,
            "CFL": graph["maximum_cfl"] < 1.0,
            "signal bounds": graph["maximum_signal_bound_violation"] <= 1.0e-12,
            "control bounds": graph["maximum_control_bound_violation"] <= 1.0e-12,
            "finite rollback": graph["maximum_finite_graphon_rollback_error"] <= 0.10,
            "capacity": inherited["maximum_capacity_residual"] <= 1.0e-10,
            "stock": inherited["maximum_stock_residual"] <= 1.0e-9,
            "inventory ownership": not graph["continuous_state_owns_physical_inventory"],
            "no posthoc": not graph["terminal_probability_adjustment"],
        }
        errors.extend(f"{cell}: {name}" for name, passed in gates.items() if not passed)
    if errors:
        raise RuntimeError("V7.3 runtime validation failed: " + "; ".join(errors))


def run(paths: int = 128, months: int = 360, **kwargs):
    gate, gaps = complete_model_gate()
    if not gate:
        raise RuntimeError("V7.3 complete-model gate failed: " + "; ".join(gaps or ["gate not passed"]))
    old_class = v72.AlgebraicDiscreteOperationalNetworkV72
    old_paths = (v72.TRACE, v72.AUDIT, v72.FREEZE)
    v72.AlgebraicDiscreteOperationalNetworkV72 = ContinuousGraphOperationalNetworkV73
    v72.TRACE, v72.AUDIT, v72.FREEZE = TRACE, AUDIT, FREEZE
    try:
        result = v72.run(paths=paths, months=months, **kwargs)
    finally:
        v72.AlgebraicDiscreteOperationalNetworkV72 = old_class
        v72.TRACE, v72.AUDIT, v72.FREEZE = old_paths
    validate_v73(result[2], months)
    return result


def write(rows, aggregate, diagnostics, verification) -> None:
    with OUT_CSV.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    verification["model_version"] = "V7.3-CONTINUOUS-GRAPH"
    verification["runtime_validation"] = "TAIWAN_V73_RUNTIME_VALIDATION: PASS"
    verification["result_sha256"] = sha256(OUT_CSV)
    OUT_JSON.write_text(json.dumps({
        "verification": verification,
        "aggregate": aggregate,
        "conditional": rows,
        "module_diagnostics": diagnostics,
    }, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    if os.environ.get("TAIWAN_ALLOW_COMPLETE_RUN") != "1":
        raise RuntimeError("V7.3 outcome simulation requires explicit complete-run authorization")
    result = run(
        paths=int(os.environ.get("TAIWAN_V73_PATHS", "128")),
        months=int(os.environ.get("TAIWAN_V73_MONTHS", "360")),
    )
    write(*result)
    print("TAIWAN_V73_COMPLETE_RUN: PASS")


if __name__ == "__main__":
    main()
