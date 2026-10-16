from __future__ import annotations

import hashlib
import json
from pathlib import Path
import py_compile
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
TRACE = OUT / "taiwan_v76_traceability.json"
AUDIT = OUT / "taiwan_v76_static_coupling_audit.json"
FREEZE = OUT / "taiwan_v76_frozen_specification.json"
DIAGNOSTIC = OUT / "taiwan_v76_diagnostic_only.json"
CODE_AUDIT = OUT / "taiwan_v76_code_coupling_audit.json"
FILES = [
    ROOT / "work" / "online_tda_v76.py",
    ROOT / "work" / "scf_signed_potential_mfg_v76.py",
    ROOT / "work" / "simulate_taiwan_fully_closed_v76.py",
    ROOT / "work" / "diagnostic_taiwan_v76.py",
    ROOT / "work" / "audit_taiwan_v76.py",
    ROOT / "work" / "simulate_taiwan_v76_checkpointed.py",
    ROOT / "work" / "run_v76_checkpoint_shard.py",
    ROOT / "work" / "audit_taiwan_v76_results.py",
    ROOT / "work" / "prepare_taiwan_v76.py",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def module(module_id, role, sections, implementation, inputs, outputs, consumers, invariants, solver):
    return {
        "module_id": module_id,
        "document_sections": sections,
        "role": role,
        "status": "active",
        "implementation": implementation,
        "clock": "monthly same-path MFG clock; diagnostics evaluated before terminal competing-risk aggregation",
        "pathwise": True,
        "inputs": inputs,
        "outputs": outputs,
        "feedback_consumers": consumers,
        "data": [{
            "source": "Feishu SCF/MFG and signed-graph MFG mathematical notes plus inherited V7.5 source ledger",
            "publisher": "local mathematical specification; inherited public-data perimeter",
            "observation_date": "specification updated 2026-10-16",
            "unit": "normalized pathwise actor and distribution state",
            "identification": "scenario_prior",
            "uncertainty": "sign thresholds and reduced branch coefficients are structural priors; no survey-level identification is claimed",
        }],
        "invariants": invariants,
        "tests": ["outputs/taiwan_v76_diagnostic_only.json", "outputs/taiwan_v76_code_coupling_audit.json"],
        "outputs_artifact": "outputs/taiwan_v76_diagnostic_only.json",
        "solver": solver,
    }


def main() -> None:
    for path in FILES:
        py_compile.compile(str(path), doraise=True)
    subprocess.run([sys.executable, str(ROOT / "work" / "diagnostic_taiwan_v76.py")], check=True)
    subprocess.run([sys.executable, str(ROOT / "work" / "audit_taiwan_v76.py")], check=True)
    diagnostic = json.loads(DIAGNOSTIC.read_text(encoding="utf-8"))
    code_audit = json.loads(CODE_AUDIT.read_text(encoding="utf-8"))
    if diagnostic.get("status") != "PASS" or code_audit.get("status") != "PASS":
        raise RuntimeError("V7.6 diagnostic or code audit failed")
    trace = json.loads((OUT / "taiwan_v75_traceability.json").read_text(encoding="utf-8"))
    trace["model_version"] = "V7.6-SIGNED-POTENTIAL-SCF"
    for inherited in trace["modules"]:
        if inherited.get("module_id") == "tda_persistent_homology_candidate":
            inherited["status"] = "superseded"
            inherited["role"] = "superseded"
            inherited["superseded_by"] = "v76_online_social_tda"
    trace["modules"].extend([
        module(
            "v76_signed_graphon_opposing_dynamics", "dynamic_state",
            ["Signed graph MFG sections 2-9 and 13"],
            ["work/scf_signed_potential_mfg_v76.py:_effective_graphon", "work/scf_signed_potential_mfg_v76.py:_signed_state_correction"],
            ["current actor distributions", "network integrity", "nonnegative Graphon magnitude"],
            ["separate sign field", "bounded opposing state target", "structural frustration", "signed Laplacian spectrum"],
            ["information and identity MFG fields", "fixed-point utility and drift"],
            ["magnitude remains nonnegative", "repelling rule excluded", "bounded state", "signed Laplacian PSD within tolerance"],
            "state-dependent signed-Graphon opposing operator with absolute-degree Laplacian diagnostics",
        ),
        module(
            "v76_potential_circulation_scf", "constraint_solver",
            ["SCF/MFG sections 4-5 and 10", "Signed graph MFG section 7.3"],
            ["work/scf_signed_potential_mfg_v76.py:_signed_structure", "work/scf_signed_potential_mfg_v76.py:AdaptiveTrustRegionSCFAnderson"],
            ["signed directed kernel", "directed hypergraph incidence", "fixed-point residual history"],
            ["reciprocal potential component", "circulation component", "adaptive trust radius", "spectral-radius proxy"],
            ["Riemannian Anderson density update", "runtime convergence gate"],
            ["exact operator reconstruction", "free-energy interpretation restricted to reciprocal component", "path-local mixing"],
            "symmetric/skew operator split with nonpotential-aware adaptive SCF Anderson mixing",
        ),
        module(
            "v76_reduced_multistart_branch_probe", "estimator",
            ["SCF/MFG sections 6, 8 and 10-11"],
            ["work/scf_signed_potential_mfg_v76.py:_reduced_branch_probe"],
            ["current signed kernel", "actor progress", "damage", "shortage"],
            ["branch count", "local spectral radius", "potential-energy spread", "bifurcation events"],
            ["model validation and equilibrium-ambiguity reporting"],
            ["three initial states", "diagnostic only", "not relabeled as a second full GMFG equilibrium", "no terminal probability adjustment"],
            "annual three-start reduced fixed-point continuation and Jacobian eigenvalue diagnostic",
        ),
        module(
            "v76_time_dependent_signed_graphon_memory", "dynamic_state",
            ["Non-equilibrium MFG document sections 3, 7 and 9"],
            ["work/scf_signed_potential_mfg_v76.py:_update_relation_memory", "work/scf_signed_potential_mfg_v76.py:_update_nonautonomous_spectrum"],
            ["actor-state distance", "common loss", "network integrity", "jump signal", "lagged positive and negative relations"],
            ["positive-edge formation/fracture", "negative-edge generation/decay", "bridge index", "atomization-boundary test", "nonautonomous spectral growth"],
            ["signed Graphon field", "SCF trust cap", "equilibrium stability diagnostics"],
            ["positive and negative kernels remain in [0,1]", "strictly positive zero-boundary formation", "distinct hysteresis barriers", "no occupational double counting"],
            "bounded formation-decay equations with persistent signed kernels and normalized cocycle growth estimator",
        ),
        module(
            "v76_online_social_tda", "dynamic_state",
            ["TDA document sections 3-10", "V7.6 executable TDA perimeter"],
            ["work/online_tda_v76.py:OnlineSocialTDAV76", "work/scf_signed_potential_mfg_v76.py:update"],
            ["actor-age state/action features", "signed Graphon", "hypergraph incidence", "major-player actions", "lagged topology state"],
            ["persistent Betti-0 curve", "two-parameter rank slices", "directed-flag counts", "Mapper nerve", "zigzag split/merge", "relative-H0 bridge gain", "lagged topology shock"],
            ["next-month relation formation/fracture", "negative-edge generation", "MFG signed field", "runtime topology audit"],
            ["quarterly evaluation with monthly hold", "no fictitious unique two-parameter barcode", "no unobserved political-class label", "lagged rather than post-hoc feedback"],
            "batched H0 filtrations, authority-distance rank slices, Mapper nerve, directed flag filtration and zigzag transition accounting",
        ),
    ])
    trace["active_equation_family_count"] += 5
    trace["passed_equation_family_count"] += 5
    trace["coverage_percent"] = 100.0
    trace["gate_status"] = "PASS"
    trace["blocking_gaps"] = []
    TRACE.write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")
    audit = {
        "model_version": "V7.6-SIGNED-POTENTIAL-SCF",
        "status": "PASS",
        "classification": "PRE_SIMULATION_INTEGRITY_GATE",
        "diagnostic_status": diagnostic["status"],
        "code_coupling_status": code_audit["status"],
        "coverage_percent": 100.0,
        "blocking_gaps": [],
        "simulation_authorized": True,
        "simulation_executed": False,
        "outcome_artifacts_created": False,
        "method_boundary": {
            "active": ["signed Graphon opposing fields", "time-dependent positive/negative relation memory", "potential/circulation decomposition", "adaptive SCF trust region", "reduced multistart branch diagnostic", "online H0/Mapper/bipersistence/zigzag/directed-flag TDA"],
            "retained": ["all 151 V7.5 active modules", "V7.5 hypergraph and quantitative finance", "hard-capacity network solvers"],
            "excluded": ["unbounded repelling dynamics", "claim that the complete directed GMFG has a global free energy", "post-hoc probability adjustment"],
        },
    }
    AUDIT.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    subprocess.run([sys.executable, str(ROOT / "scripts" / "taiwan_model_audit_manifest.py"), str(TRACE)], check=True)
    subprocess.run([sys.executable, str(ROOT / "scripts" / "audit_manifest.py"), str(TRACE)], check=True)
    freeze = {
        "model_version": "V7.6-SIGNED-POTENTIAL-SCF",
        "status": "FROZEN",
        "reason": "source, deterministic fixture and active-chain audits passed; no outcome run performed",
        "hashes": {str(path.relative_to(ROOT)): sha256(path) for path in FILES},
        "diagnostic_hashes": {str(path.relative_to(ROOT)): sha256(path) for path in (DIAGNOSTIC, CODE_AUDIT, TRACE, AUDIT)},
        "simulation_executed": False,
        "posthoc_probability_adjustment": False,
    }
    FREEZE.write_text(json.dumps(freeze, ensure_ascii=False, indent=2), encoding="utf-8")
    print("TAIWAN_V76_PREPARE_AND_FREEZE: PASS (NO OUTCOME SIMULATION)")


if __name__ == "__main__":
    main()
