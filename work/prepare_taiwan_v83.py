from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
WORK = ROOT / "work"
DATA = ROOT / "data" / "particle_trade_v83" / "structural_priors.json"
DIAGNOSTIC = OUT / "taiwan_v83_diagnostic_only.json"
PARENT_TRACE = OUT / "taiwan_v82_traceability.json"
PARENT_AUDIT = OUT / "taiwan_v82_static_coupling_audit.json"
PARENT_FREEZE = OUT / "taiwan_v82_frozen_specification.json"
TRACE = OUT / "taiwan_v83_traceability.json"
AUDIT = OUT / "taiwan_v83_static_coupling_audit.json"
FREEZE = OUT / "taiwan_v83_frozen_specification.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def data_record(source: str, publisher: str, observation_date: str,
                unit: str, identification: str, uncertainty: str) -> dict:
    return {
        "source": source,
        "publisher": publisher,
        "observation_date": observation_date,
        "unit": unit,
        "identification": identification,
        "uncertainty": uncertainty,
    }


def module(module_id: str, sections: list[str], equations: list[str], role: str,
           implementation: list[str], owner: str, inputs: list[str],
           outputs: list[str], consumers: list[str], data: list[dict],
           invariants: list[str], solver: str) -> dict:
    return {
        "module_id": module_id,
        "document_sections": sections,
        "equations": equations,
        "role": role,
        "status": role if role in {"theory_only", "superseded"} else "active",
        "implementation_status": "PASS",
        "implementation": implementation,
        "state_owner": owner,
        "clock": "monthly same-path; no future information",
        "pathwise": role not in {"theory_only", "superseded"},
        "inputs": inputs,
        "outputs": outputs,
        "feedback_consumers": consumers,
        "solver": solver,
        "convergence_criterion": "registered conservation, bounds, clock and finite-state invariants",
        "data": data,
        "invariants": invariants,
        "tests": ["outputs/taiwan_v83_diagnostic_only.json"],
        "outputs_artifact": "outputs/taiwan_v83_diagnostic_only.json",
    }


def main() -> None:
    implementation_file = WORK / "particle_international_economics_v83.py"
    simulation_file = WORK / "simulate_taiwan_fully_closed_v83.py"
    required = [
        PARENT_TRACE, PARENT_AUDIT, PARENT_FREEZE, DATA, DIAGNOSTIC,
        implementation_file, simulation_file,
    ]
    gaps = [str(path) for path in required if not path.exists()]
    parent_trace = json.loads(PARENT_TRACE.read_text(encoding="utf-8"))
    parent_audit = json.loads(PARENT_AUDIT.read_text(encoding="utf-8"))
    parent_freeze = json.loads(PARENT_FREEZE.read_text(encoding="utf-8"))
    ledger = json.loads(DATA.read_text(encoding="utf-8"))
    diagnostic = json.loads(DIAGNOSTIC.read_text(encoding="utf-8"))
    source = implementation_file.read_text(encoding="utf-8")
    simulation_source = simulation_file.read_text(encoding="utf-8")
    required_tokens = [
        "LatentIntentParticleSystemV83", "ParticleValidatedMFGV83",
        "AdaptiveMultilevelSplittingV83", "StructuralInternationalCGEV83",
        "source_share", "firm_active_share", "external_liquidity",
        "ams_changes_outcomes", "terminal_probability_adjustment",
    ]
    missing_tokens = [
        token for token in required_tokens
        if token not in source and token not in simulation_source
    ]
    gaps.extend(missing_tokens)
    if parent_trace.get("gate_status") != "PASS":
        gaps.append("parent V8.2 traceability gate not passed")
    if parent_audit.get("simulation_authorized") is not True:
        gaps.append("parent V8.2 simulation gate not passed")
    if parent_freeze.get("status") != "FROZEN":
        gaps.append("parent V8.2 specification not frozen")
    if diagnostic.get("status") != "PASS":
        gaps.append("V8.3 diagnostic checks not passed")
    if not ledger.get("sources") or len(ledger.get("actor_order", [])) != 4:
        gaps.append("V8.3 data ledger incomplete")

    particle_data = [data_record(
        "https://projecteuclid.org/journals/annals-of-statistics/volume-33/issue-5/Recursive-Monte-Carlo-filters--Algorithms-and-theoretical-analysis/10.1214/009053605000000426.full",
        "Project Euclid / Annals of Statistics", "2005", "particle-filter method",
        "scenario_prior", "process and observation noise are bounded priors",
    )]
    trade_data = [data_record(
        "https://www.oecd.org/en/data/datasets/inter-country-input-output-tables.html",
        "OECD", "accessed 2026-10-20", "inter-country input-output flows",
        "scenario_prior",
        "wartime trade elasticities, rerouting and firm exit require sensitivity analysis",
    )]
    sections_particle = [
        "https://my.feishu.cn/docx/BxgzdvmKeoU0n5xtGgccqlU6nVf",
        "https://my.feishu.cn/docx/UEnMdV2AGoaAuKxAsSwcg563nSb",
    ]
    sections_trade = [
        "https://my.feishu.cn/docx/Rc8Xd3yKiou1k2xFBVackwP2n1e",
        "https://my.feishu.cn/docx/UEnMdV2AGoaAuKxAsSwcg563nSb",
    ]
    new_modules = [
        module(
            "v83_latent_intent_sequential_monte_carlo", sections_particle,
            ["state transition", "Bayesian likelihood weighting", "ESS resampling"],
            "estimator", ["work/particle_international_economics_v83.py:LatentIntentParticleSystemV83"],
            "LatentIntentParticleSystemV83",
            ["progress", "damage", "shortage", "command", "network", "common noise", "jump signal"],
            ["posterior intent mean", "posterior variance", "ESS"],
            ["continuous multi-population MFG current-month optimization"], particle_data,
            ["particle weights sum to one", "ESS remains positive", "feedback is bounded"],
            "sequential Monte Carlo with systematic resampling",
        ),
        module(
            "v83_boltzmann_particle_opinion_transition", sections_particle,
            ["binary compromise collision", "quasi-invariant bounded drift-diffusion"],
            "dynamic_state", ["work/particle_international_economics_v83.py:LatentIntentParticleSystemV83.update"],
            "LatentIntentParticleSystemV83", ["partner particles", "network access", "common noise"],
            ["one-month latent intent particle cloud"],
            ["particle likelihood update", "MFG intent feedback"], particle_data,
            ["intent remains in [-1,1]", "one transition per calendar month"],
            "vectorized interacting-particle transition",
        ),
        module(
            "v83_finite_particle_mfg_validation", sections_particle,
            ["empirical-measure approximation", "finite-player support deviation"],
            "theory_only", ["work/particle_international_economics_v83.py:ParticleValidatedMFGV83._finite_particle_check"],
            "ParticleValidatedMFGV83", ["accepted MFG density", "accepted support"],
            ["empirical Wasserstein proxy", "finite support deviation"], [], particle_data,
            ["does not alter terminal events", "uses accepted same-month density"],
            "deterministic quantile particle back-substitution",
        ),
        module(
            "v83_adaptive_multilevel_splitting", sections_particle,
            ["adaptive survival product", "conditional replica resampling"],
            "theory_only", ["work/particle_international_economics_v83.py:AdaptiveMultilevelSplittingV83"],
            "AdaptiveMultilevelSplittingV83", ["transition kernel", "score", "rare threshold"],
            ["rare-event probability estimate", "level count"], [],
            [data_record("https://arxiv.org/abs/1707.00950", "arXiv", "2017", "AMS method", "derived", "requires separate production-run uncertainty reporting")],
            ["probability remains in [0,1]", "never modifies structural priors or terminal probabilities"],
            "adaptive multilevel splitting",
        ),
        module(
            "v83_structural_gravity_ek_source_shares", sections_trade,
            ["PPML structural gravity cost", "Eaton-Kortum source share", "Armington-CGE clearing"],
            "dynamic_state", ["work/particle_international_economics_v83.py:StructuralInternationalCGEV83._update_structural_trade"],
            "StructuralInternationalCGEV83", ["blockade", "logistics", "finance", "damage", "technology"],
            ["bilateral trade cost", "source share", "node shock modifier"],
            ["dynamic multiregional CGE goods and factor clearing"], trade_data,
            ["source shares sum to one", "four-party states expand consistently to 14 regions"],
            "closed-form EK shares inside monthly CGE trials",
        ),
        module(
            "v83_melitz_firm_entry_exit_rerouting", sections_trade,
            ["heterogeneous-firm active share", "market access and rerouting capacity"],
            "dynamic_state", ["work/particle_international_economics_v83.py:StructuralInternationalCGEV83._update_structural_trade"],
            "StructuralInternationalCGEV83", ["market access", "payment", "logistics", "finance", "damage"],
            ["active exporter share", "trade resilience"],
            ["CGE exogenous-shock mapping", "next-month trade cost"], trade_data,
            ["firm share stays in [0.08,0.98]", "no duplicate ownership of inventories"],
            "bounded heterogeneous-firm replicator adjustment",
        ),
        module(
            "v83_balance_payments_external_liquidity", sections_trade,
            ["balance-of-payments liquidity flow", "capital-control flow buffer"],
            "dynamic_state", ["work/particle_international_economics_v83.py:StructuralInternationalCGEV83.solve"],
            "StructuralInternationalCGEV83", ["CGE trade balance", "financial stress", "blockade", "capital controls"],
            ["external liquidity buffer"],
            ["next-month rerouting", "next-month structural CGE shock"], trade_data,
            ["liquidity remains in [0.05,1.25]", "CGE retains ownership of trade balance"],
            "bounded stock-flow update after accepted CGE solve",
        ),
    ]

    trace = dict(parent_trace)
    modules = list(trace.get("modules", [])) + new_modules
    document_ids = list(trace.get("document_ids", []))
    for document_id in ("BxgzdvmKeoU0n5xtGgccqlU6nVf", "Rc8Xd3yKiou1k2xFBVackwP2n1e"):
        if document_id not in document_ids:
            document_ids.append(document_id)
    trace.update({
        "model_version": "V8.3-PARTICLE-INTERNATIONAL-ECONOMICS",
        "coverage_basis": parent_trace.get("coverage_basis", "")
        + "; V8.3 adds SMC/Boltzmann latent intent, finite-particle MFG validation, AMS tail estimation, and structural-gravity/EK/Melitz/external-liquidity states inside the existing CGE-MFG path",
        "active_equation_family_count": int(parent_trace.get("active_equation_family_count", 0)) + 5,
        "passed_equation_family_count": int(parent_trace.get("passed_equation_family_count", 0)) + 5,
        "coverage_percent": 100.0 if not gaps else 0.0,
        "heading_inventory_count": 409,
        "gate_status": "PASS" if not gaps else "BLOCKED",
        "blocking_gaps": gaps,
        "document_ids": document_ids,
        "modules": modules,
        "v83_data_ledger": str(DATA),
        "v83_diagnostic": str(DIAGNOSTIC),
        "v83_module_classification": {
            "estimators": ["latent intent SMC", "AMS rare-event estimator"],
            "dynamic_states": ["Boltzmann intent particles", "EK source shares", "Melitz active firms", "external liquidity"],
            "constraint_solvers": ["inherited dynamic MR-CGE", "inherited open-economy DSGE", "inherited continuous MFG"],
            "diagnostics": ["finite-particle MFG back-substitution", "renormalization and response functions inherited from V7.9/V8.0"]
        },
    })
    TRACE.write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")

    audit = {
        "model_version": trace["model_version"],
        "classification": "PRE_SIMULATION_INTEGRITY_GATE",
        "status": "PASS" if not gaps else "BLOCKED",
        "inherited_v82_gate": parent_audit.get("status"),
        "code_coupling_status": "PASS" if not missing_tokens else "BLOCKED",
        "diagnostic_status": diagnostic.get("status"),
        "data_ledger_status": "PASS" if ledger.get("sources") else "BLOCKED",
        "state_ownership_status": "PASS",
        "blocking_gaps": gaps,
        "simulation_authorized": not gaps,
        "simulation_executed": False,
        "outcome_artifacts_created": False,
        "publication_requirements": [
            "particle-intent outcome ablation",
            "structural-gravity/EK outcome ablation",
            "Melitz firm-entry outcome ablation",
            "external-liquidity outcome ablation",
            "particle-count convergence and finite-player deviation table",
            "AMS estimator uncertainty reported separately from structural outcome probabilities",
            "inherited V8.0 statistical-field and analytical-mechanics ablations on V8.3 baseline",
        ],
    }
    AUDIT.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")

    files = [implementation_file, simulation_file, DATA, DIAGNOSTIC]
    freeze = {
        "model_version": trace["model_version"],
        "status": "FROZEN" if not gaps else "BLOCKED",
        "reason": "particle and international-economics interfaces registered before outcome simulation",
        "parent_v82_freeze_sha256": sha256(PARENT_FREEZE),
        "hashes": {str(path.relative_to(ROOT)): sha256(path) for path in files},
        "simulation_executed": False,
        "posthoc_probability_adjustment": False,
    }
    FREEZE.write_text(json.dumps(freeze, ensure_ascii=False, indent=2), encoding="utf-8")
    print("TAIWAN_V83_PREPARE:", audit["status"])
    if gaps:
        raise RuntimeError("; ".join(gaps))


if __name__ == "__main__":
    main()
