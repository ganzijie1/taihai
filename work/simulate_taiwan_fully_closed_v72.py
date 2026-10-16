from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path

import simulate_taiwan_fully_closed_v71 as v71
from algebraic_discrete_network_v72 import AlgebraicDiscreteOperationalNetworkV72


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
TRACE = OUT / "taiwan_v72_traceability.json"
AUDIT = OUT / "taiwan_v72_static_coupling_audit.json"
FREEZE = OUT / "taiwan_v72_frozen_specification.json"
OUT_CSV = OUT / "taiwan_v72_results.csv"
OUT_JSON = OUT / "taiwan_v72_simulation.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def complete_model_gate() -> tuple[bool, list[str]]:
    if not TRACE.exists() or not AUDIT.exists() or not FREEZE.exists():
        return False, ["V7.2 traceability, audit or frozen specification missing"]
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


def validate_v72(diagnostics: dict, months: int) -> None:
    errors: list[str] = []
    for cell, item in diagnostics.items():
        graph = item.get("operations", {}).get("algebraic_discrete_network_v72")
        inherited = item.get("operations", {}).get("combinatorial_network_v71")
        if graph is None or inherited is None:
            errors.append(f"{cell}: missing V7.2 operational graph")
            continue
        gates = {
            "clock": graph["months"] == months,
            "multilayer graph": graph["nodes"] == 7 and graph["directed_edges"] == 12,
            "capacity": inherited["maximum_capacity_residual"] <= 1.0e-10,
            "stock": inherited["maximum_stock_residual"] <= 1.0e-9,
            "cascade": inherited["maximum_cascade_residual"] <= 1.0e-7,
            "cascade convergence": inherited["cascade_nonconvergence_months"] == 0,
            "interlacing": graph["maximum_interlacing_violation"] <= 1.0e-9,
            "partition matroid": graph["maximum_protection_partition_violation"] == 0,
            "conflict graph": graph["protection_conflict_violations"] == 0,
            "treewidth branch": graph["exact_treewidth"] <= 4,
            "no posthoc": not graph["terminal_probability_adjustment"],
        }
        errors.extend(f"{cell}: {name}" for name, passed in gates.items() if not passed)
    if errors:
        raise RuntimeError("V7.2 runtime validation failed: " + "; ".join(errors))


def run(paths: int = 128, months: int = 360, **kwargs):
    gate, gaps = complete_model_gate()
    if not gate:
        raise RuntimeError(
            "V7.2 outcome simulation blocked by the complete-model gate: "
            + "; ".join(gaps or ["gate not passed"])
        )
    core = v71.v70.v69.core
    old_paths = (v71.TRACE, v71.AUDIT, v71.FREEZE)
    old_operations = core.DynamicOperationalConstraintSystem
    v71.TRACE, v71.AUDIT, v71.FREEZE = TRACE, AUDIT, FREEZE
    core.DynamicOperationalConstraintSystem = AlgebraicDiscreteOperationalNetworkV72
    try:
        result = v71.run(
            paths=paths, months=months, combinatorial_network_enabled=False, **kwargs
        )
    finally:
        core.DynamicOperationalConstraintSystem = old_operations
        v71.TRACE, v71.AUDIT, v71.FREEZE = old_paths
    validate_v72(result[2], months)
    return result


def write(rows, aggregate, diagnostics, verification) -> None:
    with OUT_CSV.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    verification["model_version"] = "V7.2-ALGEBRAIC-DISCRETE-GRAPH"
    verification["runtime_validation"] = "TAIWAN_V72_RUNTIME_VALIDATION: PASS"
    verification["result_sha256"] = sha256(OUT_CSV)
    OUT_JSON.write_text(json.dumps({
        "verification": verification,
        "aggregate": aggregate,
        "conditional": rows,
        "module_diagnostics": diagnostics,
    }, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    if os.environ.get("TAIWAN_ALLOW_COMPLETE_RUN") != "1":
        raise RuntimeError("V7.2 outcome simulation requires explicit complete-run authorization")
    result = run(
        paths=int(os.environ.get("TAIWAN_V72_PATHS", "128")),
        months=int(os.environ.get("TAIWAN_V72_MONTHS", "360")),
    )
    write(*result)
    print("TAIWAN_V72_COMPLETE_RUN: PASS")


if __name__ == "__main__":
    main()
