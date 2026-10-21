from __future__ import annotations

import csv
import hashlib
import json
import os
from functools import partial
from pathlib import Path

import dynamic_macro_equilibrium_v47 as macro_core
import simulate_taiwan_fully_closed_v82 as v82
from particle_international_economics_v83 import (
    ParticleValidatedMFGV83,
    StructuralInternationalCGEV83,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
TRACE = OUT / "taiwan_v83_traceability.json"
AUDIT = OUT / "taiwan_v83_static_coupling_audit.json"
FREEZE = OUT / "taiwan_v83_frozen_specification.json"
OUT_CSV = OUT / "taiwan_v83_results.csv"
OUT_JSON = OUT / "taiwan_v83_simulation.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def complete_model_gate() -> tuple[bool, list[str]]:
    if not TRACE.exists() or not AUDIT.exists() or not FREEZE.exists():
        return False, ["V8.3 traceability, audit or frozen specification missing"]
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


def validate_v83(diagnostics: dict, months: int) -> None:
    v82.validate_v82(diagnostics, months)
    errors: list[str] = []
    for cell, item in diagnostics.items():
        particle = item.get("mfg", {}).get("v83_particle_methods")
        trade = (
            item.get("macro", {}).get("cge_extension_diagnostics", {})
            .get("v83_structural_international_economics")
        )
        if particle is None:
            errors.append(f"{cell}: missing V8.3 particle diagnostics")
        else:
            checks = {
                "particle clock": particle["months"] >= months,
                "particle normalization": particle["maximum_weight_normalization_residual"] <= 1.0e-12,
                "particle ESS": particle["minimum_effective_sample_size"] > 0.0,
                "particle validation": particle["finite_particle_validation_checks"] >= months // 12,
                "particle no posthoc": not particle["terminal_probability_adjustment"],
                "AMS estimator only": particle["ams_backend_implemented"] and not particle["ams_changes_outcomes"],
            }
            errors.extend(f"{cell}: {name}" for name, ok in checks.items() if not ok)
        if trade is None:
            errors.append(f"{cell}: missing V8.3 international economics diagnostics")
        else:
            checks = {
                "trade clock": trade["months"] >= months,
                "source shares": trade["maximum_source_share_residual"] <= 1.0e-12,
                "external liquidity bounds": trade["maximum_external_liquidity_bound_violation"] <= 1.0e-12,
                "firm survival": trade["minimum_firm_active_share"] > 0.0,
                "structural coupling": trade["maximum_structural_shock_shift"] > 0.0,
                "trade no posthoc": not trade["terminal_probability_adjustment"],
            }
            errors.extend(f"{cell}: {name}" for name, ok in checks.items() if not ok)
    if errors:
        raise RuntimeError("V8.3 runtime validation failed: " + "; ".join(errors))


def run(paths: int = 128, months: int = 360, *,
        particle_intent_enabled: bool = True,
        intent_particles: int | None = None,
        structural_gravity_enabled: bool = True,
        melitz_enabled: bool = True,
        external_liquidity_enabled: bool = True, **kwargs):
    gate, gaps = complete_model_gate()
    if not gate:
        raise RuntimeError(
            "V8.3 complete-model gate failed: "
            + "; ".join(gaps or ["gate not passed"])
        )
    old_cge = macro_core.DynamicMultiRegionalCGE
    v77 = v82.v81.v80.v78.v77
    old_nested_cge = v77.NestedCESCGEV77
    old_mfg = v82.v81.DifferentialManifoldMFGV81
    old_paths = (v82.TRACE, v82.AUDIT, v82.FREEZE)
    cge_factory = partial(
        StructuralInternationalCGEV83,
        structural_gravity_enabled=structural_gravity_enabled,
        melitz_enabled=melitz_enabled,
        external_liquidity_enabled=external_liquidity_enabled,
    )
    mfg_factory = partial(
        ParticleValidatedMFGV83,
        particle_intent_enabled=particle_intent_enabled,
        intent_particles=intent_particles,
    )
    macro_core.DynamicMultiRegionalCGE = cge_factory
    v77.NestedCESCGEV77 = cge_factory
    v82.v81.DifferentialManifoldMFGV81 = mfg_factory
    v82.TRACE, v82.AUDIT, v82.FREEZE = TRACE, AUDIT, FREEZE
    try:
        result = v82.run(paths=paths, months=months, **kwargs)
    finally:
        macro_core.DynamicMultiRegionalCGE = old_cge
        v77.NestedCESCGEV77 = old_nested_cge
        v82.v81.DifferentialManifoldMFGV81 = old_mfg
        v82.TRACE, v82.AUDIT, v82.FREEZE = old_paths
    validate_v83(result[2], months)
    return result


def write(rows, aggregate, diagnostics, verification) -> None:
    with OUT_CSV.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    verification["model_version"] = "V8.3-PARTICLE-INTERNATIONAL-ECONOMICS"
    verification["runtime_validation"] = "TAIWAN_V83_RUNTIME_VALIDATION: PASS"
    verification["result_sha256"] = sha256(OUT_CSV)
    verification["v83_upgrade"] = {
        "sequential_monte_carlo_intent": True,
        "finite_particle_mfg_validation": True,
        "ams_estimator_backend": True,
        "structural_gravity_ek_melitz": True,
        "balance_of_payments_liquidity": True,
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
        raise RuntimeError("V8.3 outcome simulation requires complete-run authorization")
    result = run(
        paths=int(os.environ.get("TAIWAN_V83_PATHS", "128")),
        months=int(os.environ.get("TAIWAN_V83_MONTHS", "360")),
    )
    write(*result)
    print("TAIWAN_V83_COMPLETE_RUN: PASS")


if __name__ == "__main__":
    main()
