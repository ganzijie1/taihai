from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np

import simulate_taiwan_fully_closed_v78 as v78
from information_criticality_mfg_v79 import InformationCriticalityMFGV79


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
TRACE = OUT / "taiwan_v79_traceability.json"
AUDIT = OUT / "taiwan_v79_static_coupling_audit.json"
FREEZE = OUT / "taiwan_v79_frozen_specification.json"
OUT_CSV = OUT / "taiwan_v79_results.csv"
OUT_JSON = OUT / "taiwan_v79_simulation.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def complete_model_gate() -> tuple[bool, list[str]]:
    if not TRACE.exists() or not AUDIT.exists() or not FREEZE.exists():
        return False, ["V7.9 traceability, audit or frozen specification missing"]
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


def validate_v79(diagnostics: dict, months: int) -> None:
    v78.validate_v78(diagnostics, months)
    errors: list[str] = []
    for cell, item in diagnostics.items():
        mfg = item.get("mfg", {})
        information = mfg.get("v79_information_flow")
        cascade = mfg.get("v79_quasicritical_cascade")
        scale = mfg.get("v79_renormalization_scale_audit")
        belief = mfg.get("partial_observation", {}).get(
            "v79_correlated_rate_distortion_information"
        )
        if None in (information, cascade, scale, belief):
            errors.append(f"{cell}: missing V7.9 diagnostics")
            continue
        checks = {
            "information clock": information["updates"] >= months,
            "information lag": "next month" in information["causal_clock"],
            "cascade clock": cascade["updates"] >= months,
            "cascade accounting": cascade[
                "maximum_fast_mass_balance_residual"
            ] <= 1.0e-10,
            "cascade feedback": cascade["feedback_updates"] >= months,
            "scale clock": scale["updates"] >= months,
            "scale conservation": scale[
                "maximum_scale_conservation_residual"
            ] <= 1.0e-10,
            "belief clock": belief["updates"] >= months,
            "belief budget": belief["maximum_budget_residual"] <= 1.0e-10,
            "no posthoc": not information["terminal_probability_adjustment"]
            and not cascade["terminal_probability_adjustment"]
            and not scale["terminal_probability_adjustment"]
            and not belief["terminal_probability_adjustment"],
        }
        errors.extend(f"{cell}: {name}" for name, passed in checks.items() if not passed)
    if errors:
        raise RuntimeError("V7.9 runtime validation failed: " + "; ".join(errors))


def run(paths: int = 128, months: int = 360, **kwargs):
    gate, gaps = complete_model_gate()
    if not gate:
        raise RuntimeError(
            "V7.9 complete-model gate failed: "
            + "; ".join(gaps or ["gate not passed"])
        )
    old_class = v78.MixedPotentialNonpotentialMFGV78
    old_paths = (v78.TRACE, v78.AUDIT, v78.FREEZE)
    v78.MixedPotentialNonpotentialMFGV78 = InformationCriticalityMFGV79
    v78.TRACE, v78.AUDIT, v78.FREEZE = TRACE, AUDIT, FREEZE
    try:
        result = v78.run(paths=paths, months=months, **kwargs)
    finally:
        v78.MixedPotentialNonpotentialMFGV78 = old_class
        v78.TRACE, v78.AUDIT, v78.FREEZE = old_paths
    validate_v79(result[2], months)
    return result


def write(rows, aggregate, diagnostics, verification) -> None:
    with OUT_CSV.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    verification["model_version"] = "V7.9-INFORMATION-CRITICALITY-SCALE"
    verification["runtime_validation"] = "TAIWAN_V79_RUNTIME_VALIDATION: PASS"
    verification["result_sha256"] = sha256(OUT_CSV)
    verification["v79_upgrade"] = {
        "correlated_source_effective_sample": True,
        "conditional_transfer_entropy": True,
        "asymmetric_rate_distortion": True,
        "quasicritical_deds_cascade": True,
        "renormalization_scale_audit": True,
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
        raise RuntimeError("V7.9 outcome simulation requires complete-run authorization")
    result = run(
        paths=int(os.environ.get("TAIWAN_V79_PATHS", "128")),
        months=int(os.environ.get("TAIWAN_V79_MONTHS", "360")),
    )
    write(*result)
    print("TAIWAN_V79_COMPLETE_RUN: PASS")


if __name__ == "__main__":
    main()
