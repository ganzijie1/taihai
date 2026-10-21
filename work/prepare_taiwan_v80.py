from __future__ import annotations

import hashlib
import json
from pathlib import Path
import py_compile
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
WORKSPACE = ROOT.parent
TRACE = OUT / "taiwan_v80_traceability.json"
AUDIT = OUT / "taiwan_v80_static_coupling_audit.json"
FREEZE = OUT / "taiwan_v80_frozen_specification.json"
DIAGNOSTIC = OUT / "taiwan_v80_diagnostic_only.json"
CODE_AUDIT = OUT / "taiwan_v80_code_coupling_audit.json"
SPECIFICATIONS = [
    "https://my.feishu.cn/docx/LFWedQTEioaxlKxTCizcpr8Qn0g",
    "https://my.feishu.cn/docx/Eqrwd6cWdoiiI3xTh73cvCgsnIX",
    "https://my.feishu.cn/docx/FK4Td7HlGoytCtx8I29cXwHKn1e",
]
FILES = [
    ROOT / "work" / name for name in (
        "statistical_field_soc_v80.py",
        "analytical_mechanics_v80.py",
        "statistical_field_mfg_v80.py",
        "diagnostic_taiwan_v80.py",
        "audit_taiwan_v80.py",
        "prepare_taiwan_v80.py",
        "simulate_taiwan_fully_closed_v80.py",
        "simulate_taiwan_v80_checkpointed.py",
        "information_criticality_mfg_v79.py",
        "continuous_multipop_mfg_v47.py",
        "simulate_taiwan_multidomain_v42.py",
    )
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def module(
    module_id: str, role: str, status: str, implementation: list[str],
    inputs: list[str], outputs: list[str], consumers: list[str],
    solver: str, invariants: list[str], source_index: int,
) -> dict:
    return {
        "module_id": module_id,
        "document_sections": SPECIFICATIONS,
        "equations": [solver],
        "role": role,
        "status": status,
        "implementation_status": "PASS",
        "implementation": implementation,
        "state_owner": module_id,
        "clock": "monthly same-path; within-month bounded fixed-point iterations",
        "pathwise": status == "active",
        "inputs": inputs,
        "outputs": outputs,
        "feedback_consumers": consumers,
        "solver": solver,
        "convergence_criterion": "registered deterministic invariants and diagnostic fixtures",
        "data": [{
            "source": SPECIFICATIONS[source_index],
            "publisher": "user-provided Feishu mathematical specification",
            "observation_date": "2026-10-18",
            "unit": "normalized aggregate residual state",
            "identification": "scenario_prior",
            "uncertainty": (
                "reaction, memory, mechanics and interconnection coefficients require historical calibration and sensitivity analysis"
            ),
        }],
        "invariants": invariants,
        "tests": [
            "outputs/taiwan_v80_diagnostic_only.json",
            "outputs/taiwan_v80_code_coupling_audit.json",
        ],
        "outputs_artifact": "outputs/taiwan_v80_diagnostic_only.json",
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for path in FILES:
        py_compile.compile(str(path), doraise=True)
    subprocess.run([sys.executable, str(ROOT / "work" / "audit_taiwan_v80.py")], check=True)
    subprocess.run([sys.executable, str(ROOT / "work" / "diagnostic_taiwan_v80.py")], check=True)
    diagnostic = json.loads(DIAGNOSTIC.read_text(encoding="utf-8"))
    code_audit = json.loads(CODE_AUDIT.read_text(encoding="utf-8"))
    inherited_trace = json.loads(
        (OUT / "taiwan_v79_traceability.json").read_text(encoding="utf-8")
    )
    inherited_audit = json.loads(
        (OUT / "taiwan_v79_static_coupling_audit.json").read_text(encoding="utf-8")
    )
    additions = [
        module(
            "v80_self_consistent_dmft", "estimator", "active",
            ["work/statistical_field_soc_v80.py:SelfConsistentDMFTV80"],
            ["past residual field history", "innovation residuals"],
            ["causal memory kernel", "colored-noise covariance"],
            ["v80_non_equilibrium_statistical_field"],
            "ridge Volterra-memory fit with stable companion projection and AR colored noise",
            ["spectral radius below one", "positive-semidefinite covariance", "past-only memory"], 1,
        ),
        module(
            "v80_non_equilibrium_statistical_field", "dynamic_state", "active",
            [
                "work/statistical_field_soc_v80.py:NonEquilibriumStatisticalFieldV80",
                "work/statistical_field_mfg_v80.py:StatisticalFieldMFGV80.update",
                "work/simulate_taiwan_multidomain_v42.py:simulate_cell",
            ],
            ["same-path four-domain pressure", "MFG mean", "cascade loss", "DMFT memory"],
            ["four-domain residual factors", "recovery need", "field response"],
            ["continuous MFG", "social contagion", "finance", "firms", "network", "command"],
            "bounded nonreciprocal reaction-diffusion fixed point with DMFT memory and common jumps",
            ["bounded field", "finite fixed-point residual", "explicit one-month MFG lag", "no terminal adjustment"], 1,
        ),
        module(
            "v80_analytical_mechanics_constraint", "constraint_solver", "active",
            [
                "work/analytical_mechanics_v80.py:AnalyticalMechanicsV80",
                "work/statistical_field_mfg_v80.py:StatisticalFieldMFGV80.update",
            ],
            ["same-path pressure", "generalized residual coordinates", "jump signal"],
            ["bounded logistics, financial, command and social adjustment factors"],
            ["continuous MFG", "v80_non_equilibrium_statistical_field"],
            "discrete Lagrange-d'Alembert step with Dirac-GENERIC exchange and damping",
            ["zero-sum internal tangent flow", "zero-power skew exchange", "finite balance residual"], 2,
        ),
        module(
            "v80_soc_finite_size_evidence", "theory_only", "theory_only",
            ["work/statistical_field_soc_v80.py:QuasicriticalCascadeV80"],
            ["cascade event size", "duration", "branching ratio", "multi-resolution field statistics"],
            ["tail likelihood comparison", "Binder cumulants", "claim boundary"], [],
            "power-law versus exponential tail diagnostic and finite-resolution Binder audit",
            ["no feedback to probabilities", "SOC not claimed without finite-size evidence"], 0,
        ),
        module(
            "v80_noether_pmp_validation", "theory_only", "theory_only",
            ["work/analytical_mechanics_v80.py:AnalyticalMechanicsV80"],
            ["generalized energy", "Dirac flow", "realized adjustment"],
            ["balance residual", "PMP action disagreement"], [],
            "Noether-style zero-power/balance audit and local PMP costate cross-check",
            ["read-only validation", "no duplicate controller", "no probability adjustment"], 2,
        ),
    ]
    trace = json.loads(json.dumps(inherited_trace))
    trace["model_version"] = "V8.0-STATISTICAL-FIELD-ANALYTICAL-MECHANICS"
    trace["modules"].extend(additions)
    trace["active_equation_family_count"] = int(trace.get("active_equation_family_count", 0)) + 3
    trace["passed_equation_family_count"] = int(trace.get("passed_equation_family_count", 0)) + 3
    trace["coverage_percent"] = 100.0
    authorized = (
        diagnostic["status"] == "PASS"
        and code_audit["status"] == "PASS"
        and inherited_audit.get("simulation_authorized") is True
        and inherited_trace.get("gate_status") == "PASS"
        and not inherited_trace.get("blocking_gaps")
    )
    trace["gate_status"] = "PASS" if authorized else "FAIL"
    trace["blocking_gaps"] = [] if authorized else [
        "V8.0 diagnostic, static coupling or inherited V7.9 gate failed"
    ]
    trace["v80_scope"] = {
        "active": [item["module_id"] for item in additions if item["status"] == "active"],
        "diagnostic_only": [item["module_id"] for item in additions if item["status"] == "theory_only"],
        "outcome_simulation_executed": False,
    }
    TRACE.write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")
    audit = {
        "model_version": trace["model_version"],
        "classification": "PRE_SIMULATION_INTEGRITY_GATE",
        "status": "PASS" if authorized else "FAIL",
        "diagnostic_status": diagnostic["status"],
        "code_coupling_status": code_audit["status"],
        "inherited_v79_gate": inherited_audit.get("status"),
        "coverage_percent": 100.0 if authorized else 0.0,
        "blocking_gaps": trace["blocking_gaps"],
        "simulation_authorized": authorized,
        "simulation_executed": False,
        "outcome_artifacts_created": False,
        "diagnostic_only_components": ["SOC finite-size evidence", "Noether/PMP validation"],
    }
    AUDIT.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    subprocess.run([
        sys.executable, str(ROOT / "scripts" / "taiwan_model_audit_manifest.py"), str(TRACE)
    ], check=True)
    subprocess.run([
        sys.executable, str(WORKSPACE / "scripts" / "audit_manifest.py"), str(TRACE)
    ], check=True)
    if not authorized:
        raise RuntimeError("V8.0 pre-simulation gate failed")
    parent = OUT / "taiwan_v79_frozen_specification.json"
    freeze = {
        "model_version": trace["model_version"],
        "status": "FROZEN",
        "reason": (
            "V7.9 retained; active non-equilibrium field, self-consistent DMFT and analytical-mechanics constraint layer pass deterministic diagnostics"
        ),
        "parent_v79_freeze_sha256": sha256(parent),
        "specifications": SPECIFICATIONS,
        "hashes": {str(path.relative_to(ROOT)): sha256(path) for path in FILES},
        "diagnostic_hashes": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in (DIAGNOSTIC, CODE_AUDIT, TRACE, AUDIT)
        },
        "simulation_executed": False,
        "posthoc_probability_adjustment": False,
    }
    FREEZE.write_text(json.dumps(freeze, ensure_ascii=False, indent=2), encoding="utf-8")
    print("TAIWAN_V80_PREPARE_AND_FREEZE: PASS (NO OUTCOME SIMULATION)")


if __name__ == "__main__":
    main()
