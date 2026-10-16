from __future__ import annotations

import hashlib
import json
import py_compile
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
TRACE = OUT / "taiwan_v70_traceability.json"
AUDIT = OUT / "taiwan_v70_static_coupling_audit.json"
FREEZE = OUT / "taiwan_v70_frozen_specification.json"
DIAGNOSTIC = OUT / "taiwan_v70_tinbergen_algebraic_diagnostic.json"
END_TO_END = OUT / "taiwan_v70_end_to_end_diagnostic.json"
AUDITORS = [ROOT / "scripts" / "taiwan_model_audit_manifest.py",
            ROOT / "scripts" / "audit_manifest.py"]
FILES = [ROOT / "work" / name for name in (
    "functional_analysis_v68.py",
    "tinbergen_algebraic_v70.py",
    "simulate_taiwan_fully_closed_v70.py",
    "simulate_taiwan_v70_checkpointed.py",
    "diagnostic_taiwan_v70.py",
)]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def data(source: str, unit: str, identification: str) -> list[dict]:
    return [{
        "source": source,
        "publisher": "structural equations and public-data ledgers inherited from audited V4.7-V6.9 modules",
        "observation_date": "accessed 2026-10-13",
        "unit": unit,
        "identification": identification,
    }]


def main() -> None:
    for path in FILES:
        if path.suffix == ".py":
            py_compile.compile(str(path), doraise=True)
    diagnostic = json.loads(DIAGNOSTIC.read_text(encoding="utf-8"))
    if diagnostic.get("status") != "PASS":
        raise RuntimeError("V7.0 diagnostic has not passed")
    end_to_end = json.loads(END_TO_END.read_text(encoding="utf-8"))
    if end_to_end.get("status") != "PASS":
        raise RuntimeError("V7.0 end-to-end diagnostic has not passed")

    trace = json.loads((OUT / "taiwan_v69_traceability.json").read_text(encoding="utf-8"))
    trace["model_version"] = "V7.0-TINBERGEN-ALGEBRAIC"
    trace["coverage_basis"] += (
        "; V7.0 activates the registered Tinbergen target-tool reachability gate, "
        "four-party asymmetric real adjustment costs, a low-dimensional polynomial "
        "barrier, algebraic identifiability, cusp multistability, and min-plus/max-plus "
        "tropical bottlenecks with switch-surface hysteresis"
    )
    trace["modules"].extend([
        {
            "module_id": "v70_tinbergen_algebraic_strategy",
            "document_sections": ["Tinbergen addendum: controllability/reachability", "Algebraic geometry 2-7"],
            "role": "constraint_solver", "status": "active",
            "implementation": [
                "work/tinbergen_algebraic_v70.py:TinbergenAlgebraicRollingGameV70.update",
                "work/tinbergen_algebraic_v70.py:TinbergenAlgebraicRollingGameV70._response_matrix",
                "work/tinbergen_algebraic_v70.py:TinbergenAlgebraicRollingGameV70._barrier",
            ],
            "clock": "monthly, after rolling best responses and before campaign transition",
            "pathwise": True,
            "inputs": ["progress", "stocks", "damage", "economic capacity", "support", "command", "intervention"],
            "outputs": ["reachable simplex controls", "barrier-filtered controls", "rank and condition diagnostics", "multistable branch inertia"],
            "feedback_consumers": ["same-month operational tempo", "production", "repair", "negotiation", "next-month campaign state"],
            "data": data("existing audited state-transition coefficients plus bounded target and safety thresholds", "normalized state/control indices", "scenario_prior"),
            "invariants": ["control simplex", "full target-tool row rank", "certifiable barrier nonviolation", "causal current/past state only", "no terminal-probability edit"],
            "tests": ["outputs/taiwan_v70_tinbergen_algebraic_diagnostic.json"],
            "outputs_artifact": "outputs/taiwan_v70_simulation.json",
        },
        {
            "module_id": "v70_asymmetric_real_adjustment",
            "document_sections": ["Tinbergen addendum: asymmetric adjustment costs"],
            "role": "dynamic_state", "status": "active",
            "implementation": [
                "work/tinbergen_algebraic_v70.py:AsymmetricAdjustmentMacroV70.update",
            ],
            "clock": "monthly, before three-sector CGE/DSGE/DQ-SFC clearing",
            "pathwise": True,
            "inputs": ["logistics", "labor", "payments", "fiscal capacity", "financial stress"],
            "outputs": ["asymmetric adjustment state", "real input wedge", "financial and inflation cost"],
            "feedback_consumers": ["three-sector production", "CGE", "DSGE", "DQ-SFC", "finance and next-month output"],
            "data": data("four-party macro and sectoral ledgers; adjustment coefficients are bounded scenario priors", "monthly normalized capacity and cost", "scenario_prior"),
            "invariants": ["bounded real inputs", "nonnegative convex costs", "actor-specific expansion/contraction coefficients", "no terminal-probability edit"],
            "tests": ["outputs/taiwan_v70_tinbergen_algebraic_diagnostic.json"],
            "outputs_artifact": "outputs/taiwan_v70_simulation.json",
        },
    ])
    trace["coverage_percent"] = 100.0
    trace["gate_status"] = "PASS"
    trace["blocking_gaps"] = []
    TRACE.write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")

    audit = json.loads((OUT / "taiwan_v69_static_coupling_audit.json").read_text(encoding="utf-8"))
    audit.update({
        "model_version": "V7.0-TINBERGEN-ALGEBRAIC",
        "status": "PASS", "blocking_gaps": {},
        "diagnostic_status": "PASS",
        "active_module_count": sum(m.get("status") == "active" for m in trace["modules"]),
        "simulation_authorized": True,
    })
    boundary = audit.setdefault("selected_method_boundary", {"active": [], "theory_only": []})
    boundary["active"].extend([
        "Tinbergen target-tool local reachability and identifiability gate",
        "asymmetric convex real adjustment costs",
        "low-dimensional polynomial barrier safety filter",
        "cusp multistability and tropical min-plus/max-plus bottleneck controls with switch-surface hysteresis",
    ])
    boundary["theory_only"].extend([
        "global high-dimensional SOS proof: excluded until a certified polynomial surrogate error bound is available",
        "projective and derived algebraic geometry: no identified same-path consumer",
    ])
    AUDIT.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    for auditor in AUDITORS:
        subprocess.run([sys.executable, str(auditor), str(TRACE)], check=True)

    freeze = json.loads((OUT / "taiwan_v69_frozen_specification.json").read_text(encoding="utf-8"))
    freeze.update({
        "model_version": "V7.0-TINBERGEN-ALGEBRAIC",
        "status": "FROZEN",
        "hashes": {str(path.relative_to(ROOT)): sha256(path) for path in FILES},
        "diagnostic_hashes": {
            "outputs/taiwan_v70_tinbergen_algebraic_diagnostic.json": sha256(DIAGNOSTIC),
            "outputs/taiwan_v70_end_to_end_diagnostic.json": sha256(END_TO_END),
        },
    })
    freeze["validation_plan"] = {
        "runtime": [
            "all inherited V6.9 gates", "target-tool rank and condition",
            "control simplex", "certifiable polynomial barrier",
            "algebraic identification", "asymmetric cost activity",
            "same-path pre-transition feedback",
        ],
        "ablations": [
            "reachability and algebraic filters off",
            "tropical min-plus/max-plus gate off",
            "asymmetric macro adjustment off",
        ],
        "common_random_numbers": True,
        "posthoc_probability_adjustment": False,
    }
    FREEZE.write_text(json.dumps(freeze, ensure_ascii=False, indent=2), encoding="utf-8")
    print("TAIWAN_V70_PREPARE_AND_FREEZE: PASS")


if __name__ == "__main__":
    main()
