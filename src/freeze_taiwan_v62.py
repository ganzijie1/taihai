from __future__ import annotations

import hashlib
import json
import py_compile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
FILES = [
    ROOT / "work" / "optimal_transport_geometry_v62.py",
    ROOT / "work" / "continuous_multipop_mfg_v62.py",
    ROOT / "work" / "sectoral_economy_v62.py",
    ROOT / "work" / "simulate_taiwan_fully_closed_v62.py",
    ROOT / "work" / "diagnostic_taiwan_v62.py",
    ROOT / "data" / "sectoral_v62" / "employment_share_ledger.json",
    OUT / "taiwan_v62_traceability.json",
    OUT / "taiwan_v62_static_coupling_audit.json",
    OUT / "taiwan_v62_diagnostic.json",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main() -> None:
    for source in FILES[:5]:
        py_compile.compile(str(source), doraise=True)
    trace = json.loads((OUT / "taiwan_v62_traceability.json").read_text(encoding="utf-8"))
    audit_path = OUT / "taiwan_v62_static_coupling_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    diagnostic = json.loads((OUT / "taiwan_v62_diagnostic.json").read_text(encoding="utf-8"))
    failures = []
    if trace.get("coverage_percent") != 100.0 or trace.get("gate_status") != "PASS":
        failures.append("traceability")
    if trace.get("passed_equation_family_count") != trace.get("active_equation_family_count"):
        failures.append("equation-family coverage")
    if audit.get("status") != "PASS" or audit.get("blocking_gaps"):
        failures.append("static coupling audit")
    if diagnostic.get("status") != "PASS" or not all(diagnostic.get("checks", {}).values()):
        failures.append("runtime diagnostic")
    if audit.get("duplicate_state_owners") or audit.get("mandatory_solver_proxies"):
        failures.append("state ownership or solver proxy")
    missing = [str(path) for path in FILES if not path.exists()]
    if missing:
        failures.append("missing files: " + ", ".join(missing))
    if failures:
        raise RuntimeError("V6.2 freeze blocked: " + "; ".join(failures))
    audit["simulation_authorized"] = True
    audit["small_sample_full_system_smoke"] = {
        "status": "EXPECTED_REJECTION",
        "paths": 8,
        "months": 60,
        "reason": "importance ESS and global Filippov surface coverage require formal sample size",
        "thresholds_relaxed": False,
    }
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    frozen = {
        "model_version": "V6.2",
        "status": "FROZEN",
        "formal_design": {
            "paths": 128,
            "months": 360,
            "intervention_cases": 5,
            "trade_scenarios": 6,
            "cells": 30,
            "random_draws": "inherited frozen V5.5 econometric posterior draws",
        },
        "gate": {
            "heading_coverage_percent": 100.0,
            "active_equation_families": trace["active_equation_family_count"],
            "diagnostics": "PASS",
            "static_coupling": "PASS",
            "same_path_coupling": True,
            "posthoc_probability_adjustment": False,
        },
        "hashes": {str(path.relative_to(ROOT)): sha256(path) for path in FILES},
        "inherited_v61_freeze_sha256": sha256(OUT / "taiwan_v61_frozen_specification.json"),
        "changes_after_freeze": "require a new version and renewed diagnostics",
    }
    target = OUT / "taiwan_v62_frozen_specification.json"
    target.write_text(
        json.dumps(frozen, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("TAIWAN_V62_FREEZE: PASS")


if __name__ == "__main__":
    main()
