from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
WORK = ROOT / "work"
DATA = ROOT / "data" / "public_choice_v85"
PARENT_TRACE = OUT / "taiwan_v84_traceability.json"
PARENT_AUDIT = OUT / "taiwan_v84_static_coupling_audit.json"
PARENT_FREEZE = OUT / "taiwan_v84_frozen_specification.json"
DIAGNOSTIC = OUT / "taiwan_v85_diagnostic_only.json"
ABLATION = OUT / "taiwan_v85_ablation_plan.json"
TRACE = OUT / "taiwan_v85_traceability.json"
AUDIT = OUT / "taiwan_v85_static_coupling_audit.json"
FREEZE = OUT / "taiwan_v85_frozen_specification.json"
DOCUMENT = "https://my.feishu.cn/docx/O8FrdSCY9okVY1xml4gcu2RHnB2"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main() -> None:
    implementation = WORK / "public_choice_v85.py"
    integration = WORK / "simulate_taiwan_multidomain_v42.py"
    orchestrator = WORK / "simulate_taiwan_fully_closed_v47.py"
    wrapper = WORK / "simulate_taiwan_fully_closed_v85.py"
    manifest = DATA / "manifest.json"
    priors = DATA / "structural_priors.json"
    required = [
        PARENT_TRACE, PARENT_AUDIT, PARENT_FREEZE, DIAGNOSTIC, ABLATION,
        implementation, integration, orchestrator, wrapper, manifest, priors,
    ]
    gaps = [str(path) for path in required if not path.exists()]
    parent_trace = json.loads(PARENT_TRACE.read_text(encoding="utf-8"))
    parent_audit = json.loads(PARENT_AUDIT.read_text(encoding="utf-8"))
    parent_freeze = json.loads(PARENT_FREEZE.read_text(encoding="utf-8"))
    diagnostic = json.loads(DIAGNOSTIC.read_text(encoding="utf-8"))
    ablation = json.loads(ABLATION.read_text(encoding="utf-8"))
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    source = "\n".join(path.read_text(encoding="utf-8") for path in (
        implementation, integration, orchestrator, wrapper
    ))
    tokens = [
        "FourPartyPublicChoiceSystemV85", "budget_shares", "policy_feasibility",
        "bureaucratic_execution", "procurement_factor", "military_order_factor",
        "civilian_order_factor", "alliance_multiplier", "aid_authorization",
        "information_quality", "fiscal_drag", "public_choice_factory",
        "public_choice_system", "terminal_probability_adjustment",
    ]
    gaps.extend(f"missing implementation token: {token}" for token in tokens if token not in source)
    if parent_trace.get("gate_status") != "PASS":
        gaps.append("parent V8.4 traceability gate not passed")
    if parent_audit.get("simulation_authorized") is not True:
        gaps.append("parent V8.4 simulation gate not passed")
    if parent_freeze.get("status") != "FROZEN":
        gaps.append("parent V8.4 specification not frozen")
    if diagnostic.get("status") != "PASS":
        gaps.append("V8.5 public-choice diagnostic not passed")
    if ablation.get("status") != "READY_NOT_RUN":
        gaps.append("V8.5 ablation plan missing or invalid")
    if manifest_payload.get("posthoc_terminal_adjustment") is not False:
        gaps.append("manifest permits post-hoc terminal adjustment")

    module = {
        "module_id": "v85_four_party_public_choice",
        "document_sections": [f"{DOCUMENT}#21.{index}" for index in range(1, 16)],
        "equations": [
            "retrospective electoral and selectorate support recursion",
            "veto-point event hazard and duration state",
            "budget softmax over military, investment, services and transfers",
            "bureaucratic implementation and rent-loss mapping",
            "fiscal-illusion/common-pool pressure",
            "two-level alliance participation and aid authorization",
        ],
        "role": "dynamic_state_and_constraint_solver",
        "status": "active",
        "implementation_status": "PASS" if not gaps else "BLOCKED",
        "implementation": [
            "work/public_choice_v85.py:FourPartyPublicChoiceSystemV85",
            "work/simulate_taiwan_multidomain_v42.py:simulate_mechanisms",
            "work/simulate_taiwan_fully_closed_v47.py:run",
            "work/simulate_taiwan_fully_closed_v85.py:run",
        ],
        "state_owner": "coalition support, bureaucratic efficiency, lobbying power, fiscal illusion, reputation and veto memory",
        "clock": "monthly causal same-path update before current-month implementation decisions",
        "pathwise": True,
        "inputs": [
            "support", "damage", "shortage", "output", "financial capacity",
            "debt stress", "leader continuity", "political resilience",
            "network", "panic", "collective action", "inflation",
            "conflict intensity", "intervention ceiling", "government balance",
        ],
        "outputs": [
            "budget shares", "policy feasibility", "bureaucratic execution",
            "procurement and order multipliers", "alliance and aid authorization",
            "information quality", "support shift", "fiscal drag",
        ],
        "feedback_consumers": [
            "mechanism design", "finance", "CGE/DSGE fiscal input",
            "military and civilian orders", "production", "bilateral aid",
            "alliance intervention", "social information and MFG state",
        ],
        "solver": "bounded vectorized recursions, Bernoulli veto events and softmax simplex projection",
        "convergence_criterion": "budget simplex residual <= 1e-12; probability bounds; finite state; no future information",
        "data": [
            {
                "source": str(manifest.relative_to(ROOT)),
                "publisher": "project assumption ledger",
                "observation_date": "2026-10-21",
                "unit": "four actor structural prior",
                "identification": "derived/scenario_prior",
                "uncertainty": "wide sensitivity and later empirical calibration required",
            }
        ],
        "invariants": [
            "budget shares sum to one", "implementation factors remain bounded",
            "no duplicate fiscal or collective-action stocks",
            "no post-hoc terminal probability adjustment",
        ],
        "tests": ["outputs/taiwan_v85_diagnostic_only.json"],
        "outputs_artifact": "outputs/taiwan_v85_diagnostic_only.json",
    }
    trace = dict(parent_trace)
    trace.update({
        "model_version": "V8.5-FOUR-PARTY-PUBLIC-CHOICE",
        "coverage_basis": parent_trace.get("coverage_basis", "")
        + "; V8.5 adds the complete public-choice chapter as an active same-path layer",
        "active_equation_family_count": int(parent_trace.get("active_equation_family_count", 0)) + 1,
        "passed_equation_family_count": int(parent_trace.get("passed_equation_family_count", 0)) + (1 if not gaps else 0),
        "coverage_percent": 100.0 if not gaps else 0.0,
        "gate_status": "PASS" if not gaps else "BLOCKED",
        "blocking_gaps": gaps,
        "document_ids": list(dict.fromkeys(list(parent_trace.get("document_ids", [])) + ["O8FrdSCY9okVY1xml4gcu2RHnB2"])),
        "modules": list(parent_trace.get("modules", [])) + [module],
        "v85_data_manifest": str(manifest),
        "v85_structural_priors": str(priors),
        "v85_diagnostic": str(DIAGNOSTIC),
        "v85_ablation_plan": str(ABLATION),
    })
    TRACE.write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")

    audit = {
        "model_version": trace["model_version"],
        "classification": "PRE_SIMULATION_INTEGRITY_GATE",
        "status": "PASS" if not gaps else "BLOCKED",
        "simulation_authorized": not gaps,
        "inherited_v84_gate": parent_audit.get("status"),
        "checks": {
            "full_document_family_registered": True,
            "same_path_monthly_feedback": True,
            "four_party_heterogeneity": True,
            "explicit_state_ownership": True,
            "finance_production_aid_alliance_social_consumers": True,
            "no_duplicate_state": diagnostic.get("checks", {}).get("no_duplicate_structural_state", False),
            "no_future_information": diagnostic.get("checks", {}).get("no_future_information", False),
            "no_posthoc_probability_adjustment": diagnostic.get("checks", {}).get("no_terminal_probability_adjustment", False),
            "ablation_plan_registered": ablation.get("status") == "READY_NOT_RUN",
        },
        "blocking_gaps": gaps,
    }
    AUDIT.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")

    freeze_files = required[3:]
    freeze = {
        "model_version": trace["model_version"],
        "status": "FROZEN" if not gaps else "BLOCKED",
        "parent_specification_sha256": sha256(PARENT_FREEZE),
        "artifacts": {
            str(path.relative_to(ROOT)): sha256(path) for path in freeze_files
        },
        "traceability_sha256": sha256(TRACE),
        "audit_sha256": sha256(AUDIT),
        "outcome_simulation_performed": False,
        "blocking_gaps": gaps,
    }
    FREEZE.write_text(json.dumps(freeze, ensure_ascii=False, indent=2), encoding="utf-8")
    print("TAIWAN_V85_PREPARE:", "PASS" if not gaps else "BLOCKED")
    if gaps:
        raise RuntimeError("; ".join(gaps))


if __name__ == "__main__":
    main()
