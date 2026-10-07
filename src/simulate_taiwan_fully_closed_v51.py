from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path

import simulate_taiwan_fully_closed_v47 as core


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
TRACE = OUT / "taiwan_v51_traceability.json"
AUDIT = OUT / "taiwan_v51_static_coupling_audit.json"
OUT_CSV = OUT / "taiwan_v51_hybrid_results.csv"
OUT_JSON = OUT / "taiwan_v51_hybrid_simulation.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def run(
    paths: int = 128, months: int = 360,
    scenario_names: list[str] | None = None,
    case_names: list[str] | None = None,
    hybrid_config: dict | None = None,
    topology_enabled: bool = True,
):
    return core.run(
        paths=paths, months=months,
        scenario_names=scenario_names, case_names=case_names,
        mechanism_enabled=True, energy_enabled=True, geometry_enabled=True,
        hybrid_enabled=True, hybrid_config=hybrid_config,
        topology_enabled=topology_enabled,
        traceability_path=TRACE, coupling_audit_path=AUDIT,
    )


def write(rows, aggregate, diagnostics, verification) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    verification["result_sha256"] = sha256(OUT_CSV)
    verification["model_version"] = "V5.1"
    verification["hybrid_filippov_impulse"] = {
        "filippov_surfaces": ["logistics", "financial", "mobilization", "command"],
        "hysteresis": "active_same_path",
        "minimum_dwell_months": 2,
        "impulse_ledger": "unique_same_path_reset",
        "event_generators": "inherited Cox/DEDS/Hawkes/state-triggered; no duplicate draw",
    }
    verification["symbolic_topological_dynamics"] = {
        "symbolic_cells": 16,
        "partition": "four Filippov regime indicators",
        "pathwise_transition_graph": "active_same_path",
        "morse_scc_diagnostics": "active",
        "bounded_feedback": "command/financial/logistics/mobilization/panic",
    }
    OUT_JSON.write_text(json.dumps({
        "verification": verification,
        "aggregate": aggregate,
        "conditional": rows,
        "module_diagnostics": diagnostics,
    }, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    if os.environ.get("TAIWAN_ALLOW_COMPLETE_RUN") != "1":
        raise RuntimeError("V5.1 outcome simulation blocked until the complete-model gate passes")
    paths = int(os.environ.get("TAIWAN_V51_PATHS", "128"))
    months = int(os.environ.get("TAIWAN_V51_MONTHS", "360"))
    rows, aggregate, diagnostics, verification = run(paths=paths, months=months)
    write(rows, aggregate, diagnostics, verification)
    print("TAIWAN_V51_COMPLETE_RUN: PASS")


if __name__ == "__main__":
    main()
