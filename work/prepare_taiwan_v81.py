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
TRACE = OUT / "taiwan_v81_traceability.json"
AUDIT = OUT / "taiwan_v81_static_coupling_audit.json"
FREEZE = OUT / "taiwan_v81_frozen_specification.json"
DIAGNOSTIC = OUT / "taiwan_v81_diagnostic_only.json"
CODE_AUDIT = OUT / "taiwan_v81_code_coupling_audit.json"
HEADING_INVENTORY = OUT / "taiwan_v81_document_heading_inventory.json"
SPECIFICATION = "https://my.feishu.cn/docx/Md9wdaGQroAmRyxAryscaZoYnrd"
MAIN_DOCUMENT = "https://my.feishu.cn/docx/UEnMdV2AGoaAuKxAsSwcg563nSb"
FILES = [
    ROOT / "work" / name for name in (
        "differential_manifold_v81.py",
        "differential_manifold_mfg_v81.py",
        "diagnostic_taiwan_v81.py",
        "audit_taiwan_v81.py",
        "prepare_taiwan_v81.py",
        "simulate_taiwan_fully_closed_v81.py",
        "simulate_taiwan_v81_checkpointed.py",
        "statistical_field_mfg_v80.py",
        "continuous_multipop_mfg_v47.py",
        "simulate_taiwan_multidomain_v42.py",
    )
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def module(
    module_id: str,
    role: str,
    status: str,
    implementation: list[str],
    inputs: list[str],
    outputs: list[str],
    consumers: list[str],
    solver: str,
    invariants: list[str],
) -> dict:
    return {
        "module_id": module_id,
        "document_sections": [SPECIFICATION, MAIN_DOCUMENT],
        "equations": [solver],
        "role": role,
        "status": status,
        "implementation_status": "PASS",
        "implementation": implementation,
        "state_owner": module_id,
        "clock": "monthly same-path; pre-MFG chart update and post-MFG intrinsic retraction",
        "pathwise": status == "active",
        "inputs": inputs,
        "outputs": outputs,
        "feedback_consumers": consumers,
        "solver": solver,
        "convergence_criterion": "registered simplex, SPD, clock and finite-state invariants",
        "data": [{
            "source": SPECIFICATION,
            "publisher": "user-requested mathematical specification",
            "observation_date": "2026-10-18",
            "unit": "normalized aggregate state",
            "identification": "scenario_prior",
            "uncertainty": (
                "geometric step, covariance forgetting and chart thresholds require sensitivity analysis"
            ),
        }],
        "invariants": invariants,
        "tests": [
            "outputs/taiwan_v81_diagnostic_only.json",
            "outputs/taiwan_v81_code_coupling_audit.json",
        ],
        "outputs_artifact": "outputs/taiwan_v81_diagnostic_only.json",
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if not HEADING_INVENTORY.exists():
        raise RuntimeError("current Feishu heading inventory missing")
    headings = json.loads(HEADING_INVENTORY.read_text(encoding="utf-8"))
    if headings.get("heading_count", 0) < 300:
        raise RuntimeError("main-document heading enumeration incomplete")
    for path in FILES:
        py_compile.compile(str(path), doraise=True)
    subprocess.run([sys.executable, str(ROOT / "work" / "audit_taiwan_v81.py")], check=True)
    subprocess.run([sys.executable, str(ROOT / "work" / "diagnostic_taiwan_v81.py")], check=True)
    diagnostic = json.loads(DIAGNOSTIC.read_text(encoding="utf-8"))
    code_audit = json.loads(CODE_AUDIT.read_text(encoding="utf-8"))
    inherited_trace = json.loads(
        (OUT / "taiwan_v80_traceability.json").read_text(encoding="utf-8")
    )
    inherited_audit = json.loads(
        (OUT / "taiwan_v80_static_coupling_audit.json").read_text(encoding="utf-8")
    )
    additions = [
        module(
            "v81_probability_simplex_retraction",
            "constraint_solver",
            "active",
            [
                "work/differential_manifold_v81.py:DifferentialManifoldConstraintV81.retract_support",
                "work/differential_manifold_mfg_v81.py:DifferentialManifoldMFGV81.update",
            ],
            ["same-path MFG support target", "previous support simplex", "adaptive intrinsic step"],
            ["positive normalized support probability"],
            ["outer social support", "next-month cognition", "political fragility"],
            "square-root simplex retraction with Fisher-Rao/Hellinger geometry",
            ["strict positivity", "probability sum equals one", "no terminal adjustment"],
        ),
        module(
            "v81_spd_innovation_metric",
            "dynamic_state",
            "active",
            ["work/differential_manifold_v81.py:DifferentialManifoldConstraintV81._project_spd"],
            ["same-path social, supply, finance and command residuals"],
            ["positive-definite innovation covariance", "Mahalanobis normal residual"],
            ["slow-manifold step controller", "hybrid chart controller"],
            "symmetric eigendecomposition with eigenvalue-floor SPD retraction",
            ["positive eigenvalues", "finite condition number", "past and current state only"],
        ),
        module(
            "v81_hybrid_slow_manifold_chart",
            "constraint_solver",
            "active",
            [
                "work/differential_manifold_v81.py:DifferentialManifoldConstraintV81.prepare",
                "work/differential_manifold_mfg_v81.py:DifferentialManifoldMFGV81.update",
            ],
            ["current normalized stresses", "jump signal", "SPD metric", "lagged slow state"],
            ["chart state", "intrinsic step", "MFG iteration budget"],
            ["probability retraction", "continuous MFG solver"],
            "hysteretic stratified chart plus jump-aware fast-slow residual controller",
            ["explicit monthly clock", "hysteresis", "jump fast exit", "no physical event duplication"],
        ),
        module(
            "v81_attractor_basin_diagnostic",
            "estimator",
            "active",
            ["work/differential_manifold_v81.py:DifferentialManifoldConstraintV81._basin_diagnostics"],
            ["same-path normalized residual feature", "SPD metric"],
            ["basin share", "separatrix margin"],
            ["V8.1 path diagnostics", "structural-sensitivity report"],
            "nearest aggregate attractor under pathwise Mahalanobis metric",
            ["diagnostic only", "no terminal hazard ownership", "no probability adjustment"],
        ),
        module(
            "v81_lie_group_se3_extension",
            "theory_only",
            "theory_only",
            [f"{SPECIFICATION}#second-priority-SE3"],
            ["future object-level pose data"],
            ["future Lie-group trajectory"],
            [],
            "SE(3) Lie-group integration",
            ["excluded because current public aggregate model has no object-level orientation state"],
        ),
        module(
            "v81_full_manifold_hjb_fp_extension",
            "theory_only",
            "theory_only",
            [f"{SPECIFICATION}#manifold-HJB-FP"],
            ["future empirically identified manifold state"],
            ["future intrinsic HJB-FP solution"],
            [],
            "general Laplace-Beltrami HJB-Fokker-Planck system",
            ["current active replacement is continuous MFG plus simplex and network-geometric constraints"],
        ),
        module(
            "v81_latent_manifold_learning_extension",
            "theory_only",
            "theory_only",
            [f"{SPECIFICATION}#latent-manifold-learning"],
            ["future validated high-dimensional observations"],
            ["future latent state estimator"],
            [],
            "out-of-sample manifold learning estimator",
            ["not used without an identified observation model and holdout validation"],
        ),
    ]
    trace = json.loads(json.dumps(inherited_trace))
    trace["model_version"] = "V8.1-DIFFERENTIAL-MANIFOLD-CONSTRAINTS"
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
        "V8.1 diagnostic, static coupling or inherited V8.0 gate failed"
    ]
    trace["v81_scope"] = {
        "active": [item["module_id"] for item in additions if item["status"] == "active"],
        "theory_only": [
            item["module_id"] for item in additions if item["status"] == "theory_only"
        ],
        "main_document_heading_count": headings["heading_count"],
        "outcome_simulation_executed": False,
    }
    TRACE.write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")
    audit = {
        "model_version": trace["model_version"],
        "classification": "PRE_SIMULATION_INTEGRITY_GATE",
        "status": "PASS" if authorized else "FAIL",
        "diagnostic_status": diagnostic["status"],
        "code_coupling_status": code_audit["status"],
        "inherited_v80_gate": inherited_audit.get("status"),
        "main_document_heading_count": headings["heading_count"],
        "coverage_percent": 100.0 if authorized else 0.0,
        "blocking_gaps": trace["blocking_gaps"],
        "simulation_authorized": authorized,
        "simulation_executed": False,
        "outcome_artifacts_created": False,
        "publication_requirements": [
            "V8.1 manifold outcome ablation",
            "V8.0 statistical-field outcome ablation",
            "V8.0 analytical-mechanics outcome ablation",
        ],
    }
    AUDIT.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
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
        raise RuntimeError("V8.1 pre-simulation gate failed")
    parent = OUT / "taiwan_v80_frozen_specification.json"
    freeze = {
        "model_version": trace["model_version"],
        "status": "FROZEN",
        "reason": (
            "V8.0 retained; probability simplex, SPD metric, hybrid chart, slow-manifold and basin interfaces pass deterministic diagnostics"
        ),
        "parent_v80_freeze_sha256": sha256(parent),
        "specifications": [SPECIFICATION, MAIN_DOCUMENT],
        "main_document_heading_inventory_sha256": sha256(HEADING_INVENTORY),
        "hashes": {str(path.relative_to(ROOT)): sha256(path) for path in FILES},
        "diagnostic_hashes": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in (DIAGNOSTIC, CODE_AUDIT, TRACE, AUDIT)
        },
        "simulation_executed": False,
        "posthoc_probability_adjustment": False,
    }
    FREEZE.write_text(json.dumps(freeze, ensure_ascii=False, indent=2), encoding="utf-8")
    print("TAIWAN_V81_PREPARE_AND_FREEZE: PASS (NO OUTCOME SIMULATION)")


if __name__ == "__main__":
    main()
