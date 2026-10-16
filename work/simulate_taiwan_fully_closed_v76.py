from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np

import continuous_multipop_mfg_v47 as mfg_base
import simulate_taiwan_fully_closed_v75 as v75
from scf_signed_potential_mfg_v76 import (
    AdaptiveTrustRegionSCFAnderson,
    SignedPotentialHypergraphMFGV76,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
TRACE = OUT / "taiwan_v76_traceability.json"
AUDIT = OUT / "taiwan_v76_static_coupling_audit.json"
FREEZE = OUT / "taiwan_v76_frozen_specification.json"
OUT_CSV = OUT / "taiwan_v76_results.csv"
OUT_JSON = OUT / "taiwan_v76_simulation.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def complete_model_gate() -> tuple[bool, list[str]]:
    if not TRACE.exists() or not AUDIT.exists() or not FREEZE.exists():
        return False, ["V7.6 traceability, audit or frozen specification missing"]
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


def validate_v76(diagnostics: dict, months: int) -> None:
    errors = []
    for cell, item in diagnostics.items():
        signed = item.get("mfg", {}).get("v76_signed_potential_scf")
        tda = item.get("mfg", {}).get("v76_online_tda")
        if signed is None:
            errors.append(f"{cell}: missing V7.6 signed-potential SCF closure")
            continue
        if tda is None:
            errors.append(f"{cell}: missing V7.6 online TDA closure")
            continue
        checks = {
            "monthly signed update": signed["signed_graphon_updates"] == months,
            "monthly relation memory": signed["relation_memory_updates"] == months,
            "nonabsorbing atomization boundary": signed["minimum_zero_boundary_formation_rate"] > 0.0 and not signed["atomized_boundary_absorbing"],
            "bridge bounds": 0.0 <= signed["minimum_bridge_index"] <= signed["maximum_bridge_index"] <= 1.0,
            "nonautonomous spectrum": np.isfinite(signed["maximum_nonautonomous_growth_rate"]),
            "signed field active": signed["maximum_signed_field_correction"] > 0.0,
            "exact decomposition": signed["maximum_decomposition_residual"] <= 1.0e-12,
            "pair circulation": signed["maximum_pair_nonpotential_ratio"] > 0.0,
            "hypergraph circulation": signed["maximum_hypergraph_nonpotential_ratio"] > 0.0,
            "signed Laplacian": signed["minimum_signed_laplacian_eigenvalue"] >= -1.0e-8,
            "SCF trust active": signed["scf_calls"] > 0,
            "branch clock": signed["reduced_branch_evaluations"] == months // 12,
            "bounded opposing": not signed["repelling_rule_used"],
            "no posthoc": not signed["terminal_probability_adjustment"],
            "TDA clock": tda["updates"] == months // tda["evaluation_interval_months"],
            "TDA held months": tda["hold_months"] == months - tda["updates"],
            "TDA persistence": tda["maximum_beta0_persistence_area"] >= 0.0,
            "TDA bipersistence": tda["maximum_bipersistence_rank"] >= 0.0,
            "TDA directed flags": tda["maximum_directed_flag_triangles"] >= 0.0,
            "TDA feedback": tda["maximum_feedback_shock"] > 0.0,
            "no posthoc TDA": not tda["terminal_probability_adjustment"],
        }
        errors.extend(f"{cell}: {name}" for name, passed in checks.items() if not passed)
        base = item.get("mfg", {})
        if base.get("nonconverged_months", 1) != 0:
            errors.append(f"{cell}: continuous MFG did not converge")
        if base.get("max_fixed_point_residual", np.inf) > 5.0e-4:
            errors.append(f"{cell}: MFG residual exceeded tolerance")
    if errors:
        raise RuntimeError("V7.6 runtime validation failed: " + "; ".join(errors))


def run(paths: int = 128, months: int = 360, **kwargs):
    gate, gaps = complete_model_gate()
    if not gate:
        raise RuntimeError("V7.6 complete-model gate failed: " + "; ".join(gaps or ["gate not passed"]))
    old_class = v75.DirectedMultilayerHypergraphMFGV75
    old_accelerator = mfg_base.RegularizedRiemannianAnderson
    old_paths = (v75.TRACE, v75.AUDIT, v75.FREEZE)
    v75.DirectedMultilayerHypergraphMFGV75 = SignedPotentialHypergraphMFGV76
    mfg_base.RegularizedRiemannianAnderson = AdaptiveTrustRegionSCFAnderson
    v75.TRACE, v75.AUDIT, v75.FREEZE = TRACE, AUDIT, FREEZE
    try:
        result = v75.run(paths=paths, months=months, **kwargs)
    finally:
        v75.DirectedMultilayerHypergraphMFGV75 = old_class
        mfg_base.RegularizedRiemannianAnderson = old_accelerator
        v75.TRACE, v75.AUDIT, v75.FREEZE = old_paths
    validate_v76(result[2], months)
    return result


def write(rows, aggregate, diagnostics, verification) -> None:
    with OUT_CSV.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    verification["model_version"] = "V7.6-SIGNED-POTENTIAL-SCF"
    verification["runtime_validation"] = "TAIWAN_V76_RUNTIME_VALIDATION: PASS"
    verification["result_sha256"] = sha256(OUT_CSV)
    verification["v76_upgrade"] = {
        "signed_graphon_opposing_dynamics": True,
        "magnitude_sign_separated": True,
        "potential_circulation_decomposition": True,
        "adaptive_scf_anderson_trust_region": True,
        "reduced_multistart_branch_diagnostic": True,
        "time_dependent_positive_negative_graphon": True,
        "nonabsorbing_atomization_boundary_test": True,
        "nonautonomous_spectral_growth": True,
        "online_tda_lagged_feedback": True,
        "mapper_bipersistence_zigzag_directed_flag": True,
        "repelling_dynamics": False,
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
        raise RuntimeError("V7.6 outcome simulation requires explicit complete-run authorization")
    result = run(
        paths=int(os.environ.get("TAIWAN_V76_PATHS", "128")),
        months=int(os.environ.get("TAIWAN_V76_MONTHS", "360")),
    )
    write(*result)
    print("TAIWAN_V76_COMPLETE_RUN: PASS")


if __name__ == "__main__":
    main()
