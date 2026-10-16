from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np

import simulate_taiwan_fully_closed_v74 as v74
from hypergraph_mfg_v75 import DirectedMultilayerHypergraphMFGV75
from quantitative_financial_instability_v75 import QuantitativeFinancialInstabilityV75


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
TRACE = OUT / "taiwan_v75_traceability.json"
AUDIT = OUT / "taiwan_v75_static_coupling_audit.json"
FREEZE = OUT / "taiwan_v75_frozen_specification.json"
OUT_CSV = OUT / "taiwan_v75_results.csv"
OUT_JSON = OUT / "taiwan_v75_simulation.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _module_chain():
    v73 = v74.v73
    v72 = v73.v72
    v71 = v72.v71
    v70 = v71.v70
    v69 = v70.v69
    v68 = v69.v68
    v67 = v68.v67
    v66 = v67.v66
    return v69, v66


def complete_model_gate() -> tuple[bool, list[str]]:
    if not TRACE.exists() or not AUDIT.exists() or not FREEZE.exists():
        return False, ["V7.5 traceability, audit or frozen specification missing"]
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


def validate_v75(diagnostics: dict, months: int) -> None:
    errors: list[str] = []
    for cell, item in diagnostics.items():
        finance = item.get("finance", {}).get(
            "v75_quantitative_financial_instability"
        )
        hypergraph = item.get("mfg", {}).get(
            "v75_directed_multilayer_hypergraph_mfg"
        )
        if finance is None:
            errors.append(f"{cell}: missing V7.5 financial-instability closure")
        else:
            checks = {
                "finance clock": finance["months"] == months,
                "financing mass": finance["max_financing_mass_residual"] <= 1.0e-12,
                "regime mass": finance["max_regime_mass_residual"] <= 1.0e-12,
                "clearing": finance["max_clearing_residual"] <= 1.0e-8,
                "nonnegative clearing": finance["max_negative_payment"] <= 1.0e-12,
                "no posthoc finance": not finance["terminal_probability_adjustment"],
            }
            errors.extend(
                f"{cell}: {name}" for name, passed in checks.items() if not passed
            )
        if hypergraph is None:
            errors.append(f"{cell}: missing directed multilayer hypergraph MFG")
        else:
            checks = {
                "hypergraph enabled": hypergraph["enabled"],
                "higher-order operator evaluated": hypergraph["field_evaluations"] > 0,
                "monthly hyperedge update": hypergraph["weight_updates"] == months,
                "incidence": hypergraph["max_incidence_residual"] <= 1.0e-12,
                "non-pairwise contribution": hypergraph["max_higher_order_contribution"] > 0.0,
                "not clique equivalent": hypergraph["max_clique_rollback_error"] > 0.0,
                "no posthoc MFG": not hypergraph["terminal_probability_adjustment"],
            }
            errors.extend(
                f"{cell}: {name}" for name, passed in checks.items() if not passed
            )
        base_mfg = item.get("mfg", {})
        if base_mfg.get("nonconverged_months", 1) != 0:
            errors.append(f"{cell}: inherited continuous MFG did not converge")
        if base_mfg.get("max_fixed_point_residual", np.inf) > 5.0e-4:
            errors.append(f"{cell}: inherited MFG residual exceeded tolerance")
    if errors:
        raise RuntimeError("V7.5 runtime validation failed: " + "; ".join(errors))


def run(paths: int = 128, months: int = 360, **kwargs):
    gate, gaps = complete_model_gate()
    if not gate:
        raise RuntimeError(
            "V7.5 complete-model gate failed: "
            + "; ".join(gaps or ["gate not passed"])
        )
    v69, v66 = _module_chain()
    old_mfg = v66.ContinuousMultiPopulationMFGV66
    old_finance = v69.MartingaleOTFinancialSystemV69
    old_paths = (v74.TRACE, v74.AUDIT, v74.FREEZE)
    v66.ContinuousMultiPopulationMFGV66 = DirectedMultilayerHypergraphMFGV75
    v69.MartingaleOTFinancialSystemV69 = QuantitativeFinancialInstabilityV75
    v74.TRACE, v74.AUDIT, v74.FREEZE = TRACE, AUDIT, FREEZE
    try:
        result = v74.run(paths=paths, months=months, **kwargs)
    finally:
        v66.ContinuousMultiPopulationMFGV66 = old_mfg
        v69.MartingaleOTFinancialSystemV69 = old_finance
        v74.TRACE, v74.AUDIT, v74.FREEZE = old_paths
    validate_v75(result[2], months)
    return result


def write(rows, aggregate, diagnostics, verification) -> None:
    with OUT_CSV.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    verification["model_version"] = "V7.5-HYPERGRAPH-MINSKY-FINANCE"
    verification["runtime_validation"] = "TAIWAN_V75_RUNTIME_VALIDATION: PASS"
    verification["result_sha256"] = sha256(OUT_CSV)
    verification["v75_upgrade"] = {
        "directed_multilayer_hypergraph_gmfg": True,
        "pairwise_graphon_retained": True,
        "major_player_hyperedge_gate": True,
        "minsky_financing_state_process": True,
        "eisenberg_noe_clearing_and_fire_sale": True,
        "jcir_hawkes_regime_switching_tail_risk": True,
        "same_path_current_step_feedback": True,
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
        raise RuntimeError(
            "V7.5 outcome simulation requires explicit complete-run authorization"
        )
    result = run(
        paths=int(os.environ.get("TAIWAN_V75_PATHS", "128")),
        months=int(os.environ.get("TAIWAN_V75_MONTHS", "360")),
    )
    write(*result)
    print("TAIWAN_V75_COMPLETE_RUN: PASS")


if __name__ == "__main__":
    main()
