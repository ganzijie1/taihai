from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np

import simulate_taiwan_fully_closed_v77 as v77
from mixed_potential_nonpotential_mfg_v78 import MixedPotentialNonpotentialMFGV78


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
TRACE = OUT / "taiwan_v78_traceability.json"
AUDIT = OUT / "taiwan_v78_static_coupling_audit.json"
FREEZE = OUT / "taiwan_v78_frozen_specification.json"
OUT_CSV = OUT / "taiwan_v78_results.csv"
OUT_JSON = OUT / "taiwan_v78_simulation.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def complete_model_gate() -> tuple[bool, list[str]]:
    if not TRACE.exists() or not AUDIT.exists() or not FREEZE.exists():
        return False, ["V7.8 traceability, audit or frozen specification missing"]
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


def validate_v78(diagnostics: dict, months: int) -> None:
    v77.validate_v77(diagnostics, months)
    errors: list[str] = []
    for cell, item in diagnostics.items():
        mixed = item.get("mfg", {}).get("v78_mixed_potential_nonpotential")
        if mixed is None:
            errors.append(f"{cell}: missing V7.8 mixed MFG")
            continue
        ablations = mixed.get("maximum_ablation_field_deviation", {})
        checks = {
            "potential clock": mixed["potential_proximal_updates"] >= months,
            "mirror-prox clock": mixed["mirror_prox_evaluations"] >= months,
            "weighted adjoint": mixed["maximum_weighted_adjoint_residual"] <= 1.0e-12,
            "potential split": mixed["maximum_potential_decomposition_residual"] <= 1.0e-12,
            "directional operator active": mixed["maximum_nonpotential_operator_norm"] > 0.0,
            "predictor active": mixed["maximum_mirror_prox_predictor_step"] > 0.0,
            "corrector active": mixed["maximum_mirror_prox_correction"] > 0.0,
            "finite VI gap": np.isfinite(mixed["maximum_mixed_vi_gap"]),
            "monthly ablation": mixed["ablation_evaluations"] == months,
            "four ablations": set(ablations) == {
                "no_graph_mfg", "block_gmfg", "low_rank_graphon", "finite_graph"
            } and all(value > 0.0 for value in ablations.values()),
            "no posthoc": not mixed["terminal_probability_adjustment"],
        }
        errors.extend(f"{cell}: {name}" for name, passed in checks.items() if not passed)
    if errors:
        raise RuntimeError("V7.8 runtime validation failed: " + "; ".join(errors))


def run(paths: int = 128, months: int = 360, **kwargs):
    gate, gaps = complete_model_gate()
    if not gate:
        raise RuntimeError(
            "V7.8 complete-model gate failed: "
            + "; ".join(gaps or ["gate not passed"])
        )
    old_class = v77.CompleteTheoryMFGV77
    old_paths = (v77.TRACE, v77.AUDIT, v77.FREEZE)
    v77.CompleteTheoryMFGV77 = MixedPotentialNonpotentialMFGV78
    v77.TRACE, v77.AUDIT, v77.FREEZE = TRACE, AUDIT, FREEZE
    try:
        result = v77.run(paths=paths, months=months, **kwargs)
    finally:
        v77.CompleteTheoryMFGV77 = old_class
        v77.TRACE, v77.AUDIT, v77.FREEZE = old_paths
    validate_v78(result[2], months)
    return result


def write(rows, aggregate, diagnostics, verification) -> None:
    with OUT_CSV.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    verification["model_version"] = "V7.8-MIXED-POTENTIAL-NONPOTENTIAL-MFG"
    verification["runtime_validation"] = "TAIWAN_V78_RUNTIME_VALIDATION: PASS"
    verification["result_sha256"] = sha256(OUT_CSV)
    verification["v78_upgrade"] = {
        "weighted_potential_split": True,
        "same_iteration_mirror_prox": True,
        "directed_hypergraph_nonpotential_block": True,
        "fisher_rao_proximal_feasibility": True,
        "vi_gap_and_exploitability": True,
        "four_graph_ablation_families": True,
        "single_mfg_state_owner": True,
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
        raise RuntimeError("V7.8 outcome simulation requires complete-run authorization")
    result = run(
        paths=int(os.environ.get("TAIWAN_V78_PATHS", "128")),
        months=int(os.environ.get("TAIWAN_V78_MONTHS", "360")),
    )
    write(*result)
    print("TAIWAN_V78_COMPLETE_RUN: PASS")


if __name__ == "__main__":
    main()
