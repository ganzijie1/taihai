from __future__ import annotations

import hashlib
import json
from pathlib import Path
import py_compile
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
TRACE = OUT / "taiwan_v75_traceability.json"
AUDIT = OUT / "taiwan_v75_static_coupling_audit.json"
FREEZE = OUT / "taiwan_v75_frozen_specification.json"
DIAGNOSTIC = OUT / "taiwan_v75_diagnostic_only.json"
CODE_AUDIT = OUT / "taiwan_v75_code_coupling_audit.json"
FILES = [
    ROOT / "work" / "quantitative_financial_instability_v75.py",
    ROOT / "work" / "hypergraph_mfg_v75.py",
    ROOT / "work" / "simulate_taiwan_fully_closed_v75.py",
    ROOT / "work" / "diagnostic_taiwan_v75.py",
    ROOT / "work" / "audit_taiwan_v75.py",
    ROOT / "work" / "simulate_taiwan_v75_checkpointed.py",
    ROOT / "work" / "run_v75_checkpoint_shard.py",
    ROOT / "work" / "audit_taiwan_v75_results.py",
    ROOT / "work" / "prepare_taiwan_v75.py",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def module(
    module_id: str,
    role: str,
    sections: list[str],
    implementation: list[str],
    inputs: list[str],
    outputs: list[str],
    consumers: list[str],
    invariants: list[str],
    solver: str,
) -> dict:
    return {
        "module_id": module_id,
        "document_sections": sections,
        "role": role,
        "status": "active",
        "implementation": implementation,
        "clock": "monthly same-path master event clock; current financial and hypergraph fields enter before social-political and terminal transitions",
        "pathwise": True,
        "inputs": inputs,
        "outputs": outputs,
        "feedback_consumers": consumers,
        "data": [{
            "source": "outputs/量化金融模型在台海多系统演化仿真中的应用.md and inherited V7.4 public-data source ledger",
            "publisher": "local mathematical specification plus BIS, central-bank, fiscal, trade and national-account sources already recorded in inherited ledgers",
            "observation_date": "specification updated 2026-10-16",
            "unit": "normalized monthly actor/path state; public aggregate perimeter",
            "identification": "scenario_prior",
            "uncertainty": "wartime transition, hyperedge and tail coefficients are priors pending calibration; no institution-level or classified precision is claimed",
        }],
        "invariants": invariants,
        "tests": [
            "outputs/taiwan_v75_diagnostic_only.json",
            "outputs/taiwan_v75_code_coupling_audit.json",
        ],
        "outputs_artifact": "outputs/taiwan_v75_diagnostic_only.json",
        "solver": solver,
    }


def main() -> None:
    for path in FILES:
        py_compile.compile(str(path), doraise=True)
    subprocess.run(
        [sys.executable, str(ROOT / "work" / "diagnostic_taiwan_v75.py")],
        check=True,
    )
    subprocess.run(
        [sys.executable, str(ROOT / "work" / "audit_taiwan_v75.py")],
        check=True,
    )
    diagnostic = json.loads(DIAGNOSTIC.read_text(encoding="utf-8"))
    code_audit = json.loads(CODE_AUDIT.read_text(encoding="utf-8"))
    if diagnostic.get("status") != "PASS" or code_audit.get("status") != "PASS":
        raise RuntimeError("V7.5 diagnostic or static code audit failed")

    trace = json.loads(
        (OUT / "taiwan_v74_traceability.json").read_text(encoding="utf-8")
    )
    trace["model_version"] = "V7.5-HYPERGRAPH-MINSKY-FINANCE"
    trace["modules"].extend([
        module(
            "v75_minsky_financing_states",
            "dynamic_state",
            ["Quantitative finance document sections 5.1-5.2"],
            ["work/quantitative_financial_instability_v75.py:_update_minsky_states"],
            ["cash flow", "debt service", "distress", "shortage", "policy backstop"],
            ["hedge/speculative/Ponzi shares", "DSCR", "refinancing fragility"],
            ["credit supply", "asset sale", "DSGE risk premium", "CGE capital cost", "MFG financial stress"],
            ["state shares nonnegative", "state shares sum to one", "no fabricated observed classification"],
            "continuous-time-inspired finite-volume flow among three financing states",
        ),
        module(
            "v75_financial_network_clearing",
            "constraint_solver",
            ["Quantitative finance document sections 2 and 5.3"],
            ["work/quantitative_financial_instability_v75.py:_eisenberg_noe_clearing"],
            ["external assets", "aggregate obligations", "four-party liability kernel"],
            ["greatest clearing payment", "payment fraction", "unpaid obligation"],
            ["credit", "payments", "fire sale", "production", "fiscal capacity"],
            ["limited liability", "payment bounded by obligation", "clearing fixed-point residual"],
            "damped Eisenberg-Noe greatest-clearing fixed point",
        ),
        module(
            "v75_liquidity_fire_sale",
            "dynamic_state",
            ["Quantitative finance document sections 3-4 and 5.3"],
            ["work/quantitative_financial_instability_v75.py:update"],
            ["debt service", "cash flow", "credit", "liquidity buffer", "collateral haircut"],
            ["liquidity gap", "asset sale", "endogenous price impact"],
            ["bank capital", "collateral capacity", "credit intensity", "real production"],
            ["nonnegative sale", "bounded price", "single financial balance-sheet owner"],
            "lagged pressure with current-step collateral-constrained fire-sale feedback",
        ),
        module(
            "v75_jcir_hawkes_regime_tail",
            "dynamic_state",
            ["Quantitative finance document sections 6-9"],
            ["work/quantitative_financial_instability_v75.py:_update_regime_and_intensity"],
            ["financial stress", "clearing shortfall", "bank event", "Minsky shares"],
            ["positive credit intensity", "Hawkes excitation", "three-regime variance probabilities"],
            ["credit", "risk wedge", "asset prices", "event hazards"],
            ["positive intensity", "regime probabilities sum to one", "bounded excitation"],
            "JCIR Euler transition with bounded Hawkes excitation and conservative regime probability flows",
        ),
        module(
            "v75_term_fx_cvar_dro",
            "dynamic_state",
            ["Quantitative finance document sections 10-13"],
            ["work/quantitative_financial_instability_v75.py:update"],
            ["policy and sovereign rates", "external balance", "capital access", "local variance", "tail intensity"],
            ["term spread", "FX pressure", "investment option wedge", "CVaR-DRO wedge"],
            ["investment", "production", "credit", "fiscal aid", "inflation"],
            ["path-local risk measure", "bounded wedge", "no cross-path future information", "no terminal rescaling"],
            "affine stress mapping with parametric path-local CVaR and ambiguity surcharge",
        ),
        module(
            "v75_directed_multilayer_hypergraph_operator",
            "constraint_solver",
            ["Graphon GMFG document sections 3-7", "V7.5 hypergraph upgrade"],
            ["work/hypergraph_mfg_v75.py:_build_directed_hypergraph", "work/hypergraph_mfg_v75.py:_hypergraph_fields"],
            ["actor-age state distribution", "joint action distribution", "four dynamic layers", "pathwise hyperedge weights"],
            ["higher-order state field", "higher-order action field", "hypergraph major field"],
            ["HJB utility", "FP drift", "joint action response", "social contagion"],
            ["positive incidence coverage", "bounded field", "pathwise frozen uncertainty", "not a fixed clique proxy"],
            "directed incidence contraction with nonlinear tail aggregation, dispersion penalty and state-action synergy",
        ),
        module(
            "v75_hypergraph_major_minor_fixed_point",
            "constraint_solver",
            ["Graphon GMFG document sections 8-12", "V7.5 hypergraph upgrade"],
            ["work/hypergraph_mfg_v75.py:_effective_graphon", "work/hypergraph_mfg_v75.py:_multilayer_fields"],
            ["major policies", "minor state-action measures", "damage", "shortage", "network", "common jumps"],
            ["major-gated hyperedge weights", "hybrid Graphon-hypergraph equilibrium field"],
            ["continuous MFG HJB-FP fixed point", "politics", "mobilization", "command", "next-month network"],
            ["pairwise Graphon retained", "higher-order contribution nonzero", "hard flow ownership unchanged", "fixed-point convergence required"],
            "hybrid pairwise Graphon and directed-hypergraph operator evaluated inside every major-minor fixed-point iteration",
        ),
        module(
            "v75_pre_simulation_validation",
            "constraint_solver",
            ["V7.5 implementation and integrity gate"],
            ["work/diagnostic_taiwan_v75.py", "work/audit_taiwan_v75.py", "work/simulate_taiwan_fully_closed_v75.py:validate_v75"],
            ["source code", "deterministic fixtures", "inherited V7.4 manifest"],
            ["compile status", "mass and clearing residuals", "coupling audit", "runtime gate"],
            ["complete-model simulation gate"],
            ["no outcome simulation", "no result CSV", "no scenario probability", "class replacement restored after run"],
            "deterministic fixture tests plus static active-chain inspection",
        ),
    ])
    trace["active_equation_family_count"] += 8
    trace["passed_equation_family_count"] += 8
    trace["coverage_percent"] = 100.0
    trace["gate_status"] = "PASS"
    trace["blocking_gaps"] = []
    trace["v75_document_coverage"] = {
        "inherited_v74_heading_map": "outputs/taiwan_v74_heading_module_map.json",
        "quantitative_finance_specification": "outputs/量化金融模型在台海多系统演化仿真中的应用.md",
        "hypergraph_upgrade": "outputs/台海V7.5_量化金融与超图MFG升级实现说明.md",
        "coverage_rule": "all inherited active methods preserved; eight new executable equation families appended",
    }
    TRACE.write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")

    audit = {
        "model_version": "V7.5-HYPERGRAPH-MINSKY-FINANCE",
        "status": "PASS",
        "classification": "PRE_SIMULATION_INTEGRITY_GATE",
        "diagnostic_status": diagnostic["status"],
        "code_coupling_status": code_audit["status"],
        "inherited_v74_traceability": "PASS",
        "coverage_percent": 100.0,
        "blocking_gaps": [],
        "simulation_authorized": True,
        "simulation_executed": False,
        "outcome_artifacts_created": False,
        "selected_method_boundary": {
            "active": [
                "Minsky financing state flows",
                "aggregate financial clearing and fire sale",
                "JCIR-Hawkes-regime tail risk",
                "term/FX/CVaR-DRO feedback",
                "directed multilayer hypergraph GMFG",
            ],
            "retained": [
                "all V7.4 modules",
                "pairwise Graphon layers",
                "specialized hard-capacity network flow",
                "CGE, DSGE, HANK and continuous HJB-FP solvers",
            ],
            "excluded": [
                "institution-level exposure claims",
                "classified operational networks",
                "post-hoc probability adjustment",
                "outcome simulation in this task",
            ],
        },
    }
    AUDIT.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")

    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "taiwan_model_audit_manifest.py"), str(TRACE)],
        check=True,
    )
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "audit_manifest.py"), str(TRACE)],
        check=True,
    )
    freeze = {
        "model_version": "V7.5-HYPERGRAPH-MINSKY-FINANCE",
        "status": "FROZEN",
        "reason": "source, deterministic fixture and active-chain audits passed; no outcome run performed",
        "hashes": {str(path.relative_to(ROOT)): sha256(path) for path in FILES},
        "diagnostic_hashes": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in (DIAGNOSTIC, CODE_AUDIT, TRACE, AUDIT)
        },
        "simulation_executed": False,
        "posthoc_probability_adjustment": False,
    }
    FREEZE.write_text(json.dumps(freeze, ensure_ascii=False, indent=2), encoding="utf-8")
    print("TAIWAN_V75_PREPARE_AND_FREEZE: PASS (NO OUTCOME SIMULATION)")


if __name__ == "__main__":
    main()
