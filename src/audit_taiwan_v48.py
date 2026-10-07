from __future__ import annotations

import json
from pathlib import Path

import audit_taiwan_v47 as prior


ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "work"
OUT = ROOT / "outputs"

MECHANISM_DOC = {
    "source": "https://my.feishu.cn/docx/R7qkdxtxCoIgqQxs0lgcrSofnVM",
    "publisher": "Mechanism-design extension specification",
    "observation_date": "2026-10-05",
    "unit": "dimensionless documented scenario prior",
    "identification": "scenario_prior",
}

NEW_MODULES = [
    ("robust_dynamic_mechanism", ["3", "12", "13", "14"], ["RobustDynamicMechanismDesigner", "risk_scenarios", "robust_premium"], "robust target and recursive reputation"),
    ("alliance_burden_vcg", ["5"], ["_vcg_alliance", "truthful_utility", "alliance_multiplier"], "responsibility, balanced transfers and alliance response"),
    ("aid_moral_hazard_contract", ["6"], ["contract_effort", "milestone", "aid_multiplier"], "hidden effort, audited milestone and realized aid"),
    ("procurement_delivery_mechanism", ["7"], ["procurement_factor", "delay_risk", "resilience"], "multi-attribute procurement and military production"),
    ("information_design_bce", ["8"], ["information_support_shift", "panic_reduction", "precision"], "disclosure precision, support and panic"),
    ("ceasefire_verification_contract", ["9"], ["ceasefire_entry_multiplier", "ceasefire_break_multiplier", "settlement_multiplier"], "verification, compliance and semi-Markov transitions"),
    ("multitask_governance_contract", ["10"], ["governance_recruitment_multiplier", "governance_trust_bonus", "governance_leakage_reduction"], "agent recruitment, trust and leakage"),
    ("trade_relief_index_insurance", ["11"], ["sanction_relief", "insurance_transfer", "gross_claim"], "trade relief, blockade input and zero-sum insurance"),
]

REQUIRED_EDGES = [
    ("state->mechanism", "simulate_taiwan_multidomain_v42.py", ["mechanism_system.update", "actor_support=s[\"actor_support\"]", "financial_capacity=s[\"actor_financial_capacity\"]"]),
    ("alliance mechanism->intervention", "simulate_taiwan_multidomain_v42.py", ["alliance_multiplier", "us = np.minimum", "jp = np.minimum"]),
    ("information mechanism->social state", "simulate_taiwan_multidomain_v42.py", ["information_support_shift", "panic_reduction"]),
    ("verification->regime", "simulate_taiwan_multidomain_v42.py", ["ceasefire_entry_multiplier", "ceasefire_break_multiplier", "settlement_multiplier"]),
    ("procurement->production", "simulate_taiwan_multidomain_v42.py", ["production_cn *= mechanism_factors.procurement_factor", "production_function[:, :, 0] *= mechanism_factors.procurement_factor"]),
    ("governance->principal-agent", "simulate_taiwan_multidomain_v42.py", ["governance_recruitment_multiplier", "governance_leakage_reduction", "governance_trust_bonus"]),
    ("aid contract->realized aid", "simulate_taiwan_multidomain_v42.py", ["us_aid *= mechanism_factors.aid_multiplier", "register_realized_aid"]),
    ("trade/insurance->macro-finance", "simulate_taiwan_multidomain_v42.py", ["sanction_relief", "mechanism_factors.transfer / 12.0", "mechanism_factors.insurance_transfer", "trade_volume_loss = trade_volume_loss"]),
    ("mechanism diagnostics->runtime gate", "simulate_taiwan_fully_closed_v47.py", ["mechanism.diagnostics", "max_incentive_compatibility_residual", "mechanism nonconverged"]),
]


def source(name: str) -> str:
    return (WORK / name).read_text(encoding="utf-8")


def main() -> None:
    OUT.mkdir(exist_ok=True)
    # First regenerate and validate the inherited 23-module baseline.
    prior.main()
    old_trace = json.loads((OUT / "taiwan_v47_traceability.json").read_text(encoding="utf-8"))
    old_modules = [m for m in old_trace["modules"] if m.get("status") == "active"]
    blocking = []
    mechanism_source = source("dynamic_mechanism_design_v48.py")
    new_rows = []
    for module_id, sections, symbols, outputs in NEW_MODULES:
        missing = [symbol for symbol in symbols if symbol not in mechanism_source]
        status = "PASS" if not missing else "FAIL"
        if missing:
            blocking.append({"module": module_id, "missing_symbols": missing})
        new_rows.append({
            "module_id": module_id,
            "document_sections": sections,
            "equations": ["IC", "IR", "budget/resource feasibility", "dynamic commitment"],
            "role": "constraint_solver",
            "status": "active",
            "implementation_status": status,
            "implementation": ["work/dynamic_mechanism_design_v48.py", "work/simulate_taiwan_multidomain_v42.py"],
            "symbols": symbols,
            "state_owner": f"RobustDynamicMechanismDesigner.{module_id}",
            "clock": "monthly wartime before downstream state transitions",
            "pathwise": True,
            "inputs": ["same-path public state", "private-type scenario priors", "resource ceilings"],
            "outputs": [outputs],
            "feedback_consumers": ["campaign_deds_multidomain", "finance_fiscal_money_tax_debt_assets", "dynamic_mr_cge_mrio_gravity_chips", "principal_agent_governance"],
            "solver": "closed-form allocation plus finite robust menu and recursive contract state",
            "convergence_criterion": "IC, IR, budget, resource, limited-liability, commitment and transfer residuals below 1e-9",
            "data": [MECHANISM_DOC, prior.PRIOR_DATA],
            "invariants": ["finite bounded factors", "zero-sum transfers", "truthful utility nonnegative"],
            "tests": ["work/diagnostic_taiwan_v48.py", "outputs/taiwan_v48_static_coupling_audit.json"],
            "outputs_artifact": "outputs/taiwan_v48_mechanism_simulation.json",
        })

    edge_rows = []
    for edge, file_name, tokens in REQUIRED_EDGES:
        body = source(file_name)
        missing = [token for token in tokens if token not in body]
        status = "PASS" if not missing else "FAIL"
        if missing:
            blocking.append({"edge": edge, "file": file_name, "missing_tokens": missing})
        edge_rows.append({"edge": edge, "file": file_name, "status": status})

    coverage = 100.0 * sum(
        row["implementation_status"] == "PASS" for row in old_modules + new_rows
    ) / len(old_modules + new_rows)
    if old_trace["gate_status"] != "PASS":
        blocking.append({"inherited_v47_gate": old_trace["gate_status"]})
    status = "PASS" if coverage == 100.0 and not blocking else "FAIL"
    trace = {
        "model_version": "V4.8",
        "document_ids": ["UEnMdV2AGoaAuKxAsSwcg563nSb", "R7qkdxtxCoIgqQxs0lgcrSofnVM"],
        "coverage_basis": "all 23 inherited V4.7 active equation families plus all eight executable mechanism-design families",
        "active_equation_family_count": len(old_modules) + len(new_rows),
        "passed_equation_family_count": sum(row["implementation_status"] == "PASS" for row in old_modules + new_rows),
        "coverage_percent": coverage,
        "gate_status": status,
        "modules": old_modules + new_rows,
        "blocking_gaps": blocking,
    }
    audit = {
        "model_version": "V4.8",
        "status": status,
        "master_clock": "monthly same-path update; mechanism decisions precede downstream campaign, social, financial and macro transitions",
        "required_edges": edge_rows,
        "blocking_gaps": blocking,
        "simulation_authorized": status == "PASS",
    }
    (OUT / "taiwan_v48_traceability.json").write_text(
        json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUT / "taiwan_v48_static_coupling_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"TAIWAN_V48_STATIC_AUDIT: {status}; coverage={coverage:.2f}%")
    if blocking:
        print(json.dumps(blocking, ensure_ascii=False, indent=2))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
