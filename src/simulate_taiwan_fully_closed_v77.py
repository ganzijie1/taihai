from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np

import dynamic_macro_equilibrium_v47 as macro_base
import simulate_taiwan_fully_closed_v76 as v76
from complete_theory_mfg_v77 import CompleteTheoryMFGV77
from nested_ces_cge_v77 import NestedCESCGEV77


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
TRACE = OUT / "taiwan_v77_traceability.json"
AUDIT = OUT / "taiwan_v77_static_coupling_audit.json"
FREEZE = OUT / "taiwan_v77_frozen_specification.json"
OUT_CSV = OUT / "taiwan_v77_results.csv"
OUT_JSON = OUT / "taiwan_v77_simulation.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def complete_model_gate() -> tuple[bool, list[str]]:
    if not TRACE.exists() or not AUDIT.exists() or not FREEZE.exists():
        return False, ["V7.7 traceability, audit or frozen specification missing"]
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


def validate_v77(diagnostics: dict, months: int) -> None:
    errors: list[str] = []
    for cell, item in diagnostics.items():
        mfg = item.get("mfg", {})
        belief = (mfg.get("partial_observation") or {}).get(
            "v77_information_bottleneck_active_observation"
        )
        network = (mfg.get("network_geometry") or {}).get(
            "v77_complete_network_geometry"
        )
        oscillator = mfg.get("v77_coupled_oscillator")
        tda = mfg.get("v76_online_tda")
        macro_extension = (item.get("macro") or {}).get("cge_extension_diagnostics")
        ces = (macro_extension or {}).get("v77_nested_ces")
        for name, value in (
            ("information bottleneck", belief),
            ("complete network geometry", network),
            ("coupled oscillator", oscillator),
            ("complete TDA", tda),
            ("nested CES", ces),
        ):
            if value is None:
                errors.append(f"{cell}: missing {name}")
        if belief is not None:
            if belief["information_bottleneck_updates"] != months:
                errors.append(f"{cell}: information bottleneck clock")
            if belief["maximum_ib_fixed_point_residual"] > 1.0e-7:
                errors.append(f"{cell}: information bottleneck convergence")
            if belief["maximum_attention_kkt_residual"] > 1.0e-10:
                errors.append(f"{cell}: attention KKT residual")
        if network is not None:
            if network["fgw_updates"] != months:
                errors.append(f"{cell}: FGW clock")
            if network["maximum_fgw_marginal_residual"] > 2.0e-6:
                errors.append(f"{cell}: FGW marginal residual")
            if network["maximum_ollivier_marginal_residual"] > 2.0e-4:
                errors.append(f"{cell}: Ollivier transport residual")
            if network["maximum_hodge_reconstruction_residual"] > 1.0e-10:
                errors.append(f"{cell}: Hodge reconstruction")
        if oscillator is not None:
            if oscillator["updates"] != months:
                errors.append(f"{cell}: oscillator clock")
            if oscillator["maximum_graphon_kernel_change"] <= 0.0:
                errors.append(f"{cell}: oscillator not coupled")
        if tda is not None:
            expected = months // tda["evaluation_interval_months"]
            if tda["updates"] != expected:
                errors.append(f"{cell}: exact TDA clock")
            for key in (
                "maximum_h0_barcode_total_lifetime",
                "maximum_vietoris_rips_h1_betti",
                "maximum_bipersistence_image_rank",
                "maximum_mapper_vertices",
            ):
                if key not in tda:
                    errors.append(f"{cell}: missing TDA {key}")
        if ces is not None:
            if ces["updates"] <= 0:
                errors.append(f"{cell}: nested CES inactive")
            if ces["maximum_cobb_douglas_limit_residual"] > 2.0e-6:
                errors.append(f"{cell}: CES Cobb-Douglas limit")
    if errors:
        raise RuntimeError("V7.7 runtime validation failed: " + "; ".join(errors))


def run(paths: int = 128, months: int = 360, **kwargs):
    gate, gaps = complete_model_gate()
    if not gate:
        raise RuntimeError("V7.7 complete-model gate failed: " + "; ".join(gaps or ["gate not passed"]))
    old_mfg = v76.SignedPotentialHypergraphMFGV76
    old_cge = macro_base.DynamicMultiRegionalCGE
    old_paths = (v76.TRACE, v76.AUDIT, v76.FREEZE)
    v76.SignedPotentialHypergraphMFGV76 = CompleteTheoryMFGV77
    macro_base.DynamicMultiRegionalCGE = NestedCESCGEV77
    v76.TRACE, v76.AUDIT, v76.FREEZE = TRACE, AUDIT, FREEZE
    try:
        result = v76.run(paths=paths, months=months, **kwargs)
    finally:
        v76.SignedPotentialHypergraphMFGV76 = old_mfg
        macro_base.DynamicMultiRegionalCGE = old_cge
        v76.TRACE, v76.AUDIT, v76.FREEZE = old_paths
    validate_v77(result[2], months)
    return result


def write(rows, aggregate, diagnostics, verification) -> None:
    with OUT_CSV.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    verification["model_version"] = "V7.7-COMPLETE-THEORY-CLOSURE"
    verification["runtime_validation"] = "TAIWAN_V77_RUNTIME_VALIDATION: PASS"
    verification["result_sha256"] = sha256(OUT_CSV)
    verification["v77_upgrade"] = {
        "information_bottleneck": True,
        "active_observation": True,
        "directed_ollivier": True,
        "network_hodge": True,
        "fgw_proximal": True,
        "exact_finite_complex_tda": True,
        "nested_ces": True,
        "coupled_oscillator": True,
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
        raise RuntimeError("V7.7 outcome simulation requires complete-run authorization")
    result = run(
        paths=int(os.environ.get("TAIWAN_V77_PATHS", "128")),
        months=int(os.environ.get("TAIWAN_V77_MONTHS", "360")),
    )
    write(*result)
    print("TAIWAN_V77_COMPLETE_RUN: PASS")


if __name__ == "__main__":
    main()
