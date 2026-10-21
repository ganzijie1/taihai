from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path

import simulate_taiwan_fully_closed_v78 as v78
from statistical_field_mfg_v80 import StatisticalFieldMFGV80


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
TRACE = OUT / "taiwan_v80_traceability.json"
AUDIT = OUT / "taiwan_v80_static_coupling_audit.json"
FREEZE = OUT / "taiwan_v80_frozen_specification.json"
OUT_CSV = OUT / "taiwan_v80_results.csv"
OUT_JSON = OUT / "taiwan_v80_simulation.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def complete_model_gate() -> tuple[bool, list[str]]:
    if not TRACE.exists() or not AUDIT.exists() or not FREEZE.exists():
        return False, ["V8.0 traceability, audit or frozen specification missing"]
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


def validate_v80(diagnostics: dict, months: int) -> None:
    v78.validate_v78(diagnostics, months)
    errors: list[str] = []
    for cell, item in diagnostics.items():
        mfg = item.get("mfg", {})
        field = mfg.get("v80_non_equilibrium_statistical_field")
        mechanics = mfg.get("v80_analytical_mechanics")
        evidence = mfg.get("v80_soc_evidence")
        if None in (field, mechanics, evidence):
            errors.append(f"{cell}: missing V8.0 diagnostics")
            continue
        checks = {
            "field clock": not field.get("enabled", True)
            or field["updates"] >= months,
            "field feedback clock": not field.get("enabled", True)
            or field["feedback_updates"] >= months,
            "dmft fit": not field.get("enabled", True)
            or field["dmft"]["fit_updates"] > 0,
            "dmft stability": not field.get("enabled", True)
            or field["dmft"]["maximum_memory_spectral_radius"] < 1.0,
            "cavity clock": not field.get("enabled", True)
            or field["cavity_equilibrium_resolve_updates"] >= months // 12,
            "mechanics clock": not mechanics.get("enabled", True)
            or mechanics["updates"] >= months,
            "mechanics constraint": not mechanics.get("enabled", True)
            or mechanics["maximum_constraint_residual"] <= 1.0e-10,
            "dirac zero power": not mechanics.get("enabled", True)
            or mechanics["maximum_dirac_power_residual"] <= 1.0e-10,
            "claim boundary": evidence["claim_boundary"]
            == "quasi-critical cascade, not confirmed SOC",
            "no posthoc": not field["terminal_probability_adjustment"]
            and not mechanics["terminal_probability_adjustment"]
            and not evidence["terminal_probability_adjustment"],
        }
        errors.extend(
            f"{cell}: {name}" for name, passed in checks.items() if not passed
        )
    if errors:
        raise RuntimeError("V8.0 runtime validation failed: " + "; ".join(errors))


def run(paths: int = 128, months: int = 360, **kwargs):
    gate, gaps = complete_model_gate()
    if not gate:
        raise RuntimeError(
            "V8.0 complete-model gate failed: "
            + "; ".join(gaps or ["gate not passed"])
        )
    old_class = v78.MixedPotentialNonpotentialMFGV78
    old_paths = (v78.TRACE, v78.AUDIT, v78.FREEZE)
    v78.MixedPotentialNonpotentialMFGV78 = StatisticalFieldMFGV80
    v78.TRACE, v78.AUDIT, v78.FREEZE = TRACE, AUDIT, FREEZE
    try:
        result = v78.run(paths=paths, months=months, **kwargs)
    finally:
        v78.MixedPotentialNonpotentialMFGV78 = old_class
        v78.TRACE, v78.AUDIT, v78.FREEZE = old_paths
    validate_v80(result[2], months)
    return result


def write(rows, aggregate, diagnostics, verification) -> None:
    with OUT_CSV.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    verification["model_version"] = (
        "V8.0-STATISTICAL-FIELD-ANALYTICAL-MECHANICS"
    )
    verification["runtime_validation"] = "TAIWAN_V80_RUNTIME_VALIDATION: PASS"
    verification["result_sha256"] = sha256(OUT_CSV)
    verification["v80_upgrade"] = {
        "non_equilibrium_statistical_field": True,
        "self_consistent_dmft_memory": True,
        "soc_evidence_boundary": True,
        "discrete_lagrange_dalembert": True,
        "dirac_generic_split": True,
        "noether_pmp_audits": True,
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
        raise RuntimeError("V8.0 outcome simulation requires complete-run authorization")
    result = run(
        paths=int(os.environ.get("TAIWAN_V80_PATHS", "128")),
        months=int(os.environ.get("TAIWAN_V80_MONTHS", "360")),
    )
    write(*result)
    print("TAIWAN_V80_COMPLETE_RUN: PASS")


if __name__ == "__main__":
    main()
