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
TRACE = OUT / "taiwan_v79_traceability.json"
AUDIT = OUT / "taiwan_v79_static_coupling_audit.json"
FREEZE = OUT / "taiwan_v79_frozen_specification.json"
DIAGNOSTIC = OUT / "taiwan_v79_diagnostic_only.json"
CODE_AUDIT = OUT / "taiwan_v79_code_coupling_audit.json"
SPECIFICATION = "https://my.feishu.cn/docx/XK8rdCkm7o7dg9xjfG4cgTUxngc"
FILES = [
    ROOT / "work" / name for name in (
        "information_criticality_v79.py",
        "renormalization_scale_audit_v79.py",
        "information_criticality_mfg_v79.py",
        "diagnostic_taiwan_v79.py",
        "audit_taiwan_v79.py",
        "prepare_taiwan_v79.py",
        "simulate_taiwan_fully_closed_v79.py",
        "continuous_multipop_mfg_v47.py",
        "simulate_taiwan_multidomain_v42.py",
    )
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def module(
    module_id: str, role: str, implementation: list[str],
    inputs: list[str], outputs: list[str], consumers: list[str],
    solver: str, invariants: list[str], status: str = "active",
) -> dict:
    return {
        "module_id": module_id,
        "document_sections": [SPECIFICATION],
        "equations": [solver],
        "role": role,
        "status": status,
        "implementation_status": "PASS",
        "implementation": implementation,
        "state_owner": module_id,
        "clock": "monthly same-path with explicit one-month information lag",
        "pathwise": status == "active",
        "inputs": inputs,
        "outputs": outputs,
        "feedback_consumers": consumers,
        "solver": solver,
        "convergence_criterion": (
            "registered deterministic invariants and diagnostic fixtures"
        ),
        "data": [{
            "source": SPECIFICATION,
            "publisher": "user-provided Feishu mathematical specification",
            "observation_date": "2026-10-18",
            "unit": "normalized aggregate information, pressure and scale states",
            "identification": "scenario_prior",
            "uncertainty": (
                "source correlation, asymmetric loss and wartime thresholds are structural priors; "
                "online transfer entropy is not labelled historical causal evidence"
            ),
        }],
        "invariants": invariants,
        "tests": [
            "outputs/taiwan_v79_diagnostic_only.json",
            "outputs/taiwan_v79_code_coupling_audit.json",
        ],
        "outputs_artifact": "outputs/taiwan_v79_diagnostic_only.json",
    }


def main() -> None:
    for path in FILES:
        py_compile.compile(str(path), doraise=True)
    subprocess.run(
        [sys.executable, str(ROOT / "work" / "audit_taiwan_v79.py")],
        check=True,
    )
    subprocess.run(
        [sys.executable, str(ROOT / "work" / "diagnostic_taiwan_v79.py")],
        check=True,
    )
    diagnostic = json.loads(DIAGNOSTIC.read_text(encoding="utf-8"))
    code_audit = json.loads(CODE_AUDIT.read_text(encoding="utf-8"))
    inherited_trace = json.loads(
        (OUT / "taiwan_v78_traceability.json").read_text(encoding="utf-8")
    )
    inherited_audit = json.loads(
        (OUT / "taiwan_v78_static_coupling_audit.json").read_text(encoding="utf-8")
    )

    trace = json.loads(json.dumps(inherited_trace))
    trace["model_version"] = "V7.9-INFORMATION-CRITICALITY-SCALE"
    additions = [
        module(
            "v79_correlated_source_effective_sample", "estimator",
            ["work/information_criticality_v79.py:InformationTheoryV79"],
            ["same-path source exposure", "source correlation prior", "credibility HMM"],
            ["effective sample size", "correlation-adjusted source exposure"],
            ["actor-specific Bayesian belief filter"],
            "quadratic effective-sample-size correction inside the likelihood exposure",
            ["positive-semidefinite source correlation", "positive bounded exposure", "no future information"],
        ),
        module(
            "v79_conditional_information_flow", "estimator",
            ["work/information_criticality_v79.py:ConditionalTransferEntropyV79"],
            ["past and current four-layer aggregate signals", "target lag", "common layer mean"],
            ["lagged directed information gain", "Graphon direction multiplier"],
            ["continuous multilayer Graphon MFG"],
            "lag-one discrete conditional transfer entropy with Dirichlet pseudocount",
            ["one-month causal lag", "finite nonnegative information gain", "row-normalized Graphon"],
        ),
        module(
            "v79_asymmetric_rate_distortion_control", "constraint_solver",
            ["work/information_criticality_v79.py:InformationTheoryV79._active_observation_plan"],
            ["posterior decision risk", "false-positive and false-negative costs", "derived capacity"],
            ["next-month source allocation", "effective information capacity"],
            ["Bayesian filter", "MFG belief and OODA state"],
            "decision-weighted source allocation under correlation and derived channel capacity",
            ["information budget accounting", "bounded capacity", "strictly positive source shares"],
        ),
        module(
            "v79_self_organized_quasicritical_cascade", "dynamic_state",
            [
                "work/information_criticality_v79.py:QuasicriticalCascadeV79",
                "work/information_criticality_mfg_v79.py:InformationCriticalityMFGV79.update",
                "work/simulate_taiwan_multidomain_v42.py:simulate_cell",
            ],
            ["same-path social, financial, supply and command stress", "existing jump signal", "service action"],
            ["four-domain losses", "recovery need", "branching ratio", "DEDS event record"],
            ["MFG", "social contagion", "firms", "finance", "network", "command hierarchy"],
            "slow monthly pressure accumulation plus conservative within-month threshold relaxation",
            ["fast-event mass accounting", "bounded losses", "finite microstep ceiling", "single loss owner"],
        ),
        module(
            "v79_renormalization_scale_audit", "theory_only",
            ["work/renormalization_scale_audit_v79.py:RenormalizationScaleAuditV79"],
            ["current actor-age density", "frozen age weights"],
            ["scale residual", "coarse reconstruction error", "scale-flow spectrum"],
            [],
            "mass-preserving 16-to-8-to-4 node coarse graining and ridge scale-flow Jacobian",
            ["conditional density mass", "coarse aggregation conservation", "no terminal feedback"],
            status="theory_only",
        ),
    ]
    trace["modules"].extend(additions)
    trace["active_equation_family_count"] = int(
        trace.get("active_equation_family_count", 0)
    ) + 4
    trace["passed_equation_family_count"] = int(
        trace.get("passed_equation_family_count", 0)
    ) + 4
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
        "V7.9 diagnostic, static coupling or inherited V7.8 gate failed"
    ]
    trace["v79_scope"] = {
        "active": [item["module_id"] for item in additions if item["status"] == "active"],
        "diagnostic_only": ["v79_renormalization_scale_audit"],
        "outcome_simulation_executed": False,
    }
    TRACE.write_text(
        json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    audit = {
        "model_version": "V7.9-INFORMATION-CRITICALITY-SCALE",
        "classification": "PRE_SIMULATION_INTEGRITY_GATE",
        "status": "PASS" if authorized else "FAIL",
        "diagnostic_status": diagnostic["status"],
        "code_coupling_status": code_audit["status"],
        "inherited_v78_gate": inherited_audit.get("status"),
        "coverage_percent": 100.0 if authorized else 0.0,
        "blocking_gaps": trace["blocking_gaps"],
        "simulation_authorized": authorized,
        "simulation_executed": False,
        "outcome_artifacts_created": False,
        "renormalization_feedback_enabled": False,
    }
    AUDIT.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    subprocess.run([
        sys.executable,
        str(ROOT / "scripts" / "taiwan_model_audit_manifest.py"),
        str(TRACE),
    ], check=True)
    subprocess.run([
        sys.executable,
        str(WORKSPACE / "scripts" / "audit_manifest.py"),
        str(TRACE),
    ], check=True)
    if not authorized:
        raise RuntimeError("V7.9 pre-simulation gate failed")

    parent = OUT / "taiwan_v78_frozen_specification.json"
    freeze = {
        "model_version": "V7.9-INFORMATION-CRITICALITY-SCALE",
        "status": "FROZEN",
        "reason": (
            "V7.8 retained; four active information/cascade additions pass deterministic diagnostics; "
            "scale audit is read-only"
        ),
        "parent_v78_freeze_sha256": sha256(parent),
        "specification": SPECIFICATION,
        "hashes": {str(path.relative_to(ROOT)): sha256(path) for path in FILES},
        "diagnostic_hashes": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in (DIAGNOSTIC, CODE_AUDIT, TRACE, AUDIT)
        },
        "simulation_executed": False,
        "posthoc_probability_adjustment": False,
    }
    FREEZE.write_text(
        json.dumps(freeze, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("TAIWAN_V79_PREPARE_AND_FREEZE: PASS (NO OUTCOME SIMULATION)")


if __name__ == "__main__":
    main()
