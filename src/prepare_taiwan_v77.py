from __future__ import annotations

import hashlib
import json
from pathlib import Path
import py_compile
import subprocess
import sys

import geomloss
import numba
import numpy as np
import ot
import torch

from theory_resolution_v77 import RESOLUTION


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
TRACE = OUT / "taiwan_v77_traceability.json"
AUDIT = OUT / "taiwan_v77_static_coupling_audit.json"
FREEZE = OUT / "taiwan_v77_frozen_specification.json"
DIAGNOSTIC = OUT / "taiwan_v77_diagnostic_only.json"
CODE_AUDIT = OUT / "taiwan_v77_code_coupling_audit.json"
RESOLUTION_FILE = OUT / "taiwan_v77_theory_only_resolution.json"
FILES = [
    ROOT / "work" / name for name in (
        "information_geometry_v58.py",
        "dynamic_macro_equilibrium_v47.py",
        "complete_information_geometry_v77.py",
        "complete_network_geometry_v77.py",
        "complete_online_tda_v77.py",
        "coupled_oscillator_v77.py",
        "nested_ces_cge_v77.py",
        "complete_theory_mfg_v77.py",
        "simulate_taiwan_fully_closed_v77.py",
        "diagnostic_taiwan_v77.py",
        "audit_taiwan_v77.py",
        "theory_resolution_v77.py",
        "prepare_taiwan_v77.py",
        "simulate_taiwan_v77_checkpointed.py",
        "run_v77_checkpoint_shard.py",
        "audit_taiwan_v77_results.py",
    )
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def active_module(module_id, role, sections, implementation, equations,
                  inputs, outputs, consumers, solver, convergence, invariants):
    return {
        "module_id": module_id,
        "document_sections": sections,
        "equations": equations,
        "role": role,
        "status": "active",
        "implementation_status": "PASS",
        "implementation": implementation,
        "state_owner": module_id,
        "clock": "monthly same-path master clock unless an explicit annual TDA multirate clock is named",
        "pathwise": True,
        "inputs": inputs,
        "outputs": outputs,
        "feedback_consumers": consumers,
        "solver": solver,
        "convergence_criterion": convergence,
        "data": [{
            "source": "inherited V7.6 public-data ledger plus DsBJdUOGno4h1JxrFAYc3QOlnsd and resolved mathematical specifications",
            "publisher": "public-data model ledger and user-provided mathematical specifications",
            "observation_date": "specification resolved 2026-10-16",
            "unit": "normalized same-path state unless inherited physical units apply",
            "identification": "scenario_prior",
            "uncertainty": "structural coefficients remain in the frozen sensitivity ledger; no classified precision is claimed",
        }],
        "invariants": invariants,
        "tests": [
            "outputs/taiwan_v77_diagnostic_only.json",
            "outputs/taiwan_v77_code_coupling_audit.json",
        ],
        "outputs_artifact": "outputs/taiwan_v77_diagnostic_only.json",
    }


def main() -> None:
    for path in FILES:
        py_compile.compile(str(path), doraise=True)
    subprocess.run([sys.executable, str(ROOT / "work" / "diagnostic_taiwan_v77.py")], check=True)
    subprocess.run([sys.executable, str(ROOT / "work" / "audit_taiwan_v77.py")], check=True)
    diagnostic = json.loads(DIAGNOSTIC.read_text(encoding="utf-8"))
    code_audit = json.loads(CODE_AUDIT.read_text(encoding="utf-8"))
    resolution = json.loads(RESOLUTION_FILE.read_text(encoding="utf-8"))
    if diagnostic["status"] != "PASS" or code_audit["status"] != "PASS":
        raise RuntimeError("V7.7 diagnostic or code audit failed")

    trace = json.loads((OUT / "taiwan_v76_traceability.json").read_text(encoding="utf-8"))
    trace["model_version"] = "V7.7-COMPLETE-THEORY-CLOSURE"
    for inherited in trace["modules"]:
        module_id = inherited.get("module_id")
        decision = RESOLUTION.get(module_id)
        if decision and decision["disposition"] in {
            "superseded_by_active", "superseded_by_identified_estimator",
            "superseded_duplicate_exclusion", "split_and_resolved",
        }:
            inherited["role"] = "superseded"
            inherited["status"] = "superseded"
            inherited["implementation_status"] = "SUPERSEDED"
            inherited["superseded_by"] = decision.get("replacement")
        elif decision:
            inherited["implementation_status"] = "EXCLUDED_WITH_EVIDENCE"
            inherited["exclusion_reason"] = decision["reason"]
            inherited["useful_quantitative_mechanism"] = False
        if module_id == "v76_online_social_tda":
            inherited["role"] = "superseded"
            inherited["status"] = "superseded"
            inherited["implementation_status"] = "SUPERSEDED"
            inherited["superseded_by"] = "v77_complete_online_tda"

    trace["modules"].extend([
        active_module(
            "v77_information_bottleneck", "dynamic_state",
            ["Information geometry: information bottleneck"],
            ["work/complete_information_geometry_v77.py:_information_bottleneck"],
            ["discrete Blahut-Arimoto information-bottleneck equations"],
            ["same-path posterior", "source reliability channel"],
            ["three-state compressed posterior", "information rate", "distortion"],
            ["next-month belief filter", "MFG progress perception", "social actions"],
            "Numba-parallel log-domain Blahut-Arimoto fixed point on the nine-point belief grid",
            "active-support and joint fixed-point residual <= 1e-7, excluded mass <= 1e-7, and objective increase <= 2e-10",
            ["posterior simplex", "numerically unidentified rows masked at p(x)<1e-7 with total excluded mass <=1e-7", "causal current/past clock", "no terminal adjustment"],
        ),
        active_module(
            "v77_active_observation", "constraint_solver",
            ["Information geometry: active observation"],
            ["work/complete_information_geometry_v77.py:_active_observation_plan"],
            ["entropy-reduction objective with KL attention cost and budget simplex"],
            ["posterior variance", "source reliability", "lagged information action"],
            ["pathwise source allocation", "next-month source exposure"],
            ["belief observation likelihood", "information bottleneck"],
            "closed-form entropic KKT solution",
            "attention KKT residual <= 1e-10 and exact budget adding-up",
            ["one-month causal lag", "nonnegative allocation", "budget conservation"],
        ),
        active_module(
            "v77_fgw_ollivier_hodge", "constraint_solver",
            ["Network geometry NG 6, NG 8 and NG 11"],
            ["work/complete_network_geometry_v77.py:CompleteNetworkGeometryV77"],
            ["entropic FGW proximal fixed point", "directed Ollivier transport curvature", "network Hodge projection"],
            ["prior and target Graphons", "signed directed transition", "current actor stress"],
            ["aligned kernel", "curvature", "gradient/cycle flow", "bottleneck risk"],
            ["current MFG intensity", "network memory", "social propagation"],
            "pathwise FGW outer iteration, vectorized log-domain Sinkhorn with POT plan oracle, incidence-Laplacian Hodge projection",
            "FGW marginal <= 2e-6; Ollivier marginal <= 2e-4; Hodge residual <= 1e-10",
            ["row stochasticity", "transport marginals", "exact Hodge reconstruction"],
        ),
        active_module(
            "v77_complete_online_tda", "dynamic_state",
            ["TDA sections 3-10", "Network geometry NG 9-10"],
            ["work/complete_online_tda_v77.py:CompleteOnlineTDAV77"],
            ["Kruskal H0 barcode", "GF(2) clique and directed-flag H1", "Mapper nerve", "two-parameter image-rank invariant", "consecutive H0 union zigzag", "hypergraph simplicial-closure homology"],
            ["actor-age state/action cloud", "signed Graphon", "dynamic hyperedge weights", "major actions"],
            ["pathwise topology state", "lagged topology shock", "barcodes and rank invariants"],
            ["next-month relation formation/fracture", "signed MFG field"],
            "exact finite-complex algorithms on 16 nodes with an annual multirate clock",
            "finite-complex boundary identities and deterministic replay",
            ["no fictitious two-parameter barcode", "held state remains pathwise", "no post-hoc terminal adjustment"],
        ),
        active_module(
            "v77_nested_ces_production", "dynamic_state",
            ["Three-sector framework section 3 alternative CES nesting"],
            ["work/nested_ces_cge_v77.py:NestedCESCGEV77"],
            ["capital-labor CES nested inside value-added-logistics CES"],
            ["installed capital", "skill-adjusted labor", "logistics", "productivity", "war shock"],
            ["capacity by region-sector", "substitution response"],
            ["CGE market clearing", "DSGE", "DQSFC inventories and finance"],
            "closed-form nested CES inside every CGE equilibrium evaluation",
            "Cobb-Douglas limit residual <= 2e-6; positive bounded capacity",
            ["unit baseline", "positive output", "single capacity owner"],
        ),
        active_module(
            "v77_coupled_oscillator", "dynamic_state",
            ["DsBJdUOGno4h1JxrFAYc3QOlnsd sections 6-9 and 12"],
            ["work/coupled_oscillator_v77.py:CoupledOscillatorV77", "work/complete_theory_mfg_v77.py:_effective_graphon"],
            ["inertial signed directed Sakaguchi-Kuramoto equation", "order parameters", "master-stability transverse exponent"],
            ["signed Graphon", "damage", "shortage", "network", "jumps", "common noise"],
            ["phase compatibility", "global/layer order", "desynchronization", "stability proxy"],
            ["current MFG kernel/intensity", "next-month relation memory"],
            "four-substep semi-implicit inertial phase integrator with circular wrapping",
            "finite phase/frequency, bounded compatibility, deterministic seed replay",
            ["phase on circle", "bounded intensity", "explicit interpretation boundary"],
        ),
    ])
    trace["active_equation_family_count"] += 6
    trace["passed_equation_family_count"] += 6
    remaining_theory = [
        module for module in trace["modules"]
        if module.get("role") == "theory_only"
    ]
    unjustified = [
        module["module_id"] for module in remaining_theory
        if RESOLUTION.get(module["module_id"], {}).get("useful") is not False
    ]
    trace["coverage_percent"] = 100.0 if not unjustified else 0.0
    trace["gate_status"] = "PASS" if not unjustified else "FAIL"
    trace["blocking_gaps"] = unjustified
    trace["theory_only_resolution_artifact"] = str(RESOLUTION_FILE.relative_to(ROOT))
    TRACE.write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")

    audit = {
        "model_version": "V7.7-COMPLETE-THEORY-CLOSURE",
        "status": "PASS" if not unjustified else "FAIL",
        "classification": "PRE_SIMULATION_INTEGRITY_GATE",
        "diagnostic_status": diagnostic["status"],
        "code_coupling_status": code_audit["status"],
        "theory_only_total_reviewed": resolution["counts"]["total"],
        "useful_theory_items_resolved": resolution["counts"]["useful"],
        "remaining_theory_only": [module["module_id"] for module in remaining_theory],
        "unjustified_theory_only": unjustified,
        "coverage_percent": trace["coverage_percent"],
        "blocking_gaps": unjustified,
        "simulation_authorized": not unjustified,
        "simulation_executed": False,
        "outcome_artifacts_created": False,
        "method_boundary": {
            "active": [
                "information bottleneck", "active observation", "FGW proximal network",
                "directed Ollivier curvature", "network Hodge", "exact finite-complex TDA",
                "nested CES production", "inertial signed coupled oscillator",
            ],
            "excluded_with_evidence": [module["module_id"] for module in remaining_theory],
            "forbidden": ["fabricated instruments or microdata", "post-hoc probability adjustment", "duplicate state ownership"],
        },
    }
    AUDIT.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    subprocess.run([sys.executable, str(ROOT / "scripts" / "taiwan_model_audit_manifest.py"), str(TRACE)], check=True)
    subprocess.run([sys.executable, str(ROOT / "scripts" / "audit_manifest.py"), str(TRACE)], check=True)
    if unjustified:
        raise RuntimeError("unjustified theory-only items: " + ", ".join(unjustified))
    freeze = {
        "model_version": "V7.7-COMPLETE-THEORY-CLOSURE",
        "status": "FROZEN",
        "reason": "all useful theory-only mechanisms resolved into active implementations or existing active owners; evidence-based exclusions retained",
        "hashes": {str(path.relative_to(ROOT)): sha256(path) for path in FILES},
        "solver_dependencies": {
            "numpy": np.__version__,
            "numba": numba.__version__,
            "pot": ot.__version__,
            "geomloss": getattr(geomloss, "__version__", "unknown"),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
        },
        "diagnostic_hashes": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in (DIAGNOSTIC, CODE_AUDIT, RESOLUTION_FILE, TRACE, AUDIT)
        },
        "simulation_executed": False,
        "posthoc_probability_adjustment": False,
    }
    FREEZE.write_text(json.dumps(freeze, ensure_ascii=False, indent=2), encoding="utf-8")
    print("TAIWAN_V77_PREPARE_AND_FREEZE: PASS (NO OUTCOME SIMULATION)")


if __name__ == "__main__":
    main()
