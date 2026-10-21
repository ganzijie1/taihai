from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
WORK = ROOT / "work"
DATA = ROOT / "data" / "sectoral_v82" / "actor_structural_priors.json"
PARENT_TRACE = OUT / "taiwan_v81_traceability.json"
PARENT_AUDIT = OUT / "taiwan_v81_static_coupling_audit.json"
PARENT_FREEZE = OUT / "taiwan_v81_frozen_specification.json"
TRACE = OUT / "taiwan_v82_traceability.json"
AUDIT = OUT / "taiwan_v82_static_coupling_audit.json"
FREEZE = OUT / "taiwan_v82_frozen_specification.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main() -> None:
    required = [PARENT_TRACE, PARENT_AUDIT, PARENT_FREEZE, DATA,
                WORK / "sectoral_development_labor_v82.py",
                WORK / "simulate_taiwan_fully_closed_v82.py"]
    missing = [str(p) for p in required if not p.exists()]
    parent_trace = json.loads(PARENT_TRACE.read_text(encoding="utf-8"))
    parent_audit = json.loads(PARENT_AUDIT.read_text(encoding="utf-8"))
    parent_freeze = json.loads(PARENT_FREEZE.read_text(encoding="utf-8"))
    ledger = json.loads(DATA.read_text(encoding="utf-8"))
    source = (WORK / "sectoral_development_labor_v82.py").read_text(encoding="utf-8")
    required_tokens = [
        "_agriculture_update_v82", "_industry_update_v82",
        "_development_update", "_labor_update_v82",
        "state_owners", "terminal_probability_adjustment",
    ]
    missing_tokens = [token for token in required_tokens if token not in source]
    gaps = missing + missing_tokens
    if parent_trace.get("gate_status") != "PASS":
        gaps.append("parent V8.1 traceability gate not passed")
    if parent_audit.get("simulation_authorized") is not True:
        gaps.append("parent V8.1 simulation gate not passed")
    if parent_freeze.get("status") != "FROZEN":
        gaps.append("parent V8.1 specification not frozen")
    if len(ledger.get("actor_order", [])) != 4 or not ledger.get("sources"):
        gaps.append("V8.2 actor data ledger incomplete")

    trace = dict(parent_trace)
    document_ids = list(trace.get("document_ids", []))
    for document_id in (
        "UKJyd1QnSoh3KjxQXGlcY409nbg",
        "ObhWdjzczoM9faxWhHhcYCvzntf",
        "ZVbVdoCcioUXDYxWVIDcYMEsnxe",
        "VVxwdNvXloyuWSxA8iOcxmwsnue",
    ):
        if document_id not in document_ids:
            document_ids.append(document_id)
    trace.update({
        "model_version": "V8.2-ACTOR-SPECIFIC-SECTOR-DEVELOPMENT-LABOR",
        "coverage_basis": parent_trace.get("coverage_basis", "")
        + "; V8.2 adds same-path actor-specific agriculture, industrial organization, development and labor mechanisms while preserving the existing CGE, DQ-SFC and V6.2 state owners",
        "active_equation_family_count": int(parent_trace.get("active_equation_family_count", 0)) + 16,
        "passed_equation_family_count": int(parent_trace.get("passed_equation_family_count", 0)) + 16,
        "coverage_percent": 100.0 if not gaps else 0.0,
        "gate_status": "PASS" if not gaps else "BLOCKED",
        "blocking_gaps": gaps,
        "document_ids": document_ids,
        "v82_new_modules": [
            "agriculture_expectations_entitlements_surplus",
            "industrial_enterprise_composition_concentration",
            "development_human_capital_technology_state_capacity",
            "labor_search_matching_skill_fatigue",
        ],
        "v82_data_ledger": str(DATA),
    })
    TRACE.write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")

    audit = {
        "model_version": trace["model_version"],
        "classification": "PRE_SIMULATION_INTEGRITY_GATE",
        "status": "PASS" if not gaps else "BLOCKED",
        "inherited_v81_gate": parent_audit.get("status"),
        "code_coupling_status": "PASS" if not missing_tokens else "BLOCKED",
        "data_ledger_status": "PASS" if DATA.exists() and ledger.get("sources") else "BLOCKED",
        "blocking_gaps": gaps,
        "simulation_authorized": not gaps,
        "simulation_executed": False,
        "outcome_artifacts_created": False,
        "publication_requirements": [
            "V8.2 agriculture outcome ablation",
            "V8.2 industry outcome ablation",
            "V8.2 development outcome ablation",
            "V8.2 labor outcome ablation",
            "inherited V8.0 statistical-field and analytical-mechanics ablations on the V8.2 baseline",
        ],
    }
    AUDIT.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")

    files = [WORK / "sectoral_development_labor_v82.py",
             WORK / "simulate_taiwan_fully_closed_v82.py", DATA]
    freeze = {
        "model_version": trace["model_version"],
        "status": "FROZEN" if not gaps else "BLOCKED",
        "reason": "actor-specific sector, development and labor interfaces are registered before simulation",
        "parent_v81_freeze_sha256": sha256(PARENT_FREEZE),
        "hashes": {str(p.relative_to(ROOT)): sha256(p) for p in files},
        "simulation_executed": False,
        "posthoc_probability_adjustment": False,
    }
    FREEZE.write_text(json.dumps(freeze, ensure_ascii=False, indent=2), encoding="utf-8")
    print("TAIWAN_V82_PREPARE:", audit["status"])
    if gaps:
        raise RuntimeError("; ".join(gaps))


if __name__ == "__main__":
    main()
