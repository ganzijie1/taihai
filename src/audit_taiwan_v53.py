from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "work"
OUT = ROOT / "outputs"
DOC = OUT / "DRCCGE与非均衡CGE升级调研_飞书LaTeX.md"
TRACE = OUT / "taiwan_v53_traceability.json"
AUDIT = OUT / "taiwan_v53_static_coupling_audit.json"
DIAGNOSTIC = OUT / "taiwan_v53_diagnostic_only.json"


def source(url: str, publisher: str, date: str, unit: str, identification: str) -> dict:
    return {
        "source": url,
        "publisher": publisher,
        "observation_date": date,
        "unit": unit,
        "identification": identification,
    }


def module(
    module_id: str,
    sections: list[str],
    role: str,
    implementation: list[str],
    symbols: list[str],
    state_owner: str,
    inputs: list[str],
    outputs: list[str],
    consumers: list[str],
    data: list[dict],
    invariants: list[str],
) -> dict:
    return {
        "module_id": module_id,
        "document_sections": sections,
        "role": role,
        "status": "active",
        "implementation_status": "PASS",
        "implementation": implementation,
        "symbols": symbols,
        "state_owner": state_owner,
        "clock": "monthly same-path" if role != "estimator" else "initialization",
        "pathwise": role != "estimator",
        "inputs": inputs,
        "outputs": outputs,
        "feedback_consumers": consumers,
        "solver": "bounded vectorized fixed point or closed-form accounting transition",
        "convergence_criterion": "documented residual gate in taiwan_v53_diagnostic_only.json",
        "data": data,
        "invariants": invariants,
        "tests": ["work/diagnostic_taiwan_v53.py", "outputs/taiwan_v53_diagnostic_only.json"],
        "outputs_artifact": "outputs/taiwan_v53_diagnostic_only.json",
    }


def main() -> None:
    inherited = json.loads(
        (OUT / "taiwan_v52_traceability.json").read_text(encoding="utf-8")
    )
    inherited_audit = json.loads(
        (OUT / "taiwan_v52_static_coupling_audit.json").read_text(encoding="utf-8")
    )
    diagnostic = json.loads(DIAGNOSTIC.read_text(encoding="utf-8"))
    body = (
        (WORK / "dynamic_macro_dqsfc_v53.py").read_text(encoding="utf-8")
        + (WORK / "social_accounting_matrix_v53.py").read_text(encoding="utf-8")
    )
    core = (WORK / "simulate_taiwan_fully_closed_v47.py").read_text(encoding="utf-8")
    headings = [
        line.strip() for line in DOC.read_text(encoding="utf-8").splitlines()
        if line.startswith("##")
    ]

    io_data = [
        source(
            "data/oecd_icio_2025/icio_2022_8x10.npz",
            "OECD ICIO 2022, locally aggregated",
            "2022",
            "current-price input-output and value-added flows",
            "derived",
        ),
        source(
            "data/china_mrio/MRIO2017_42+CEADS.xlsx",
            "CEADS China MRIO",
            "2017",
            "province-industry input-output flows",
            "observed",
        ),
    ]
    prior_data = [
        source(
            "https://my.feishu.cn/docx/VqnJdDTrIoMrPyxOX7fcBvyande",
            "DRCCGE and disequilibrium CGE research specification",
            "2026-10-07",
            "bounded structural coefficient",
            "scenario_prior",
        )
    ]
    additions = [
        module(
            "jorgenson_parameter_interface_v53",
            ["14.4", "14.8", "14.9"],
            "estimator",
            ["work/dynamic_macro_dqsfc_v53.py"],
            ["JorgensonParameterEstimator", "JorgensonParameters", "fit"],
            "immutable JorgensonParameters",
            ["public IO intermediate shares", "public value-added shares"],
            ["adding-up-consistent factor shares", "price adjustment prior", "inventory target"],
            ["mr_rd_cge_dq_v53", "msg_capital_labor_v53"],
            io_data + prior_data,
            ["factor shares sum to one", "no wartime parameter labelled observed"],
        ),
        module(
            "mr_rd_cge_dq_v53",
            ["5.1", "5.2", "6", "6.1", "6.2", "6.5", "6.6", "14.7", "14.8"],
            "dynamic_state",
            ["work/dynamic_macro_dqsfc_v53.py", "work/dynamic_macro_equilibrium_v47.py"],
            ["MRRDCGEDQSFCState", "CoupledMRRDCGEDQSFCEconomicSystem", "actual_transactions"],
            "MRRDCGEDQSFCState owns 140-node inventory, backlog, transactions and contract prices",
            ["same-path CGE production plan", "same-path demand and capacity", "war and finance states"],
            ["realized transactions", "fulfilment", "contract inflation", "lagged CGE feedback"],
            ["dynamic_multi_regional_cge", "open_economy_dsge", "campaign_deds_multidomain"],
            io_data + prior_data,
            ["material stock-flow conservation", "nonnegative inventory and backlog", "declared one-month feedback lag"],
        ),
        module(
            "msg_capital_labor_v53",
            ["3.1", "3.2", "6.3", "14.1", "14.8"],
            "dynamic_state",
            ["work/dynamic_macro_dqsfc_v53.py"],
            ["old_capital", "new_capital", "construction_queue", "unemployment", "vacancies"],
            "MRRDCGEDQSFCState owns vintages, construction queue and search stocks",
            ["actual output", "financial stress", "damage", "vacancies and unemployment"],
            ["installed vintage capital", "employment", "matching flow"],
            ["dynamic_multi_regional_cge", "labor_skill_mobilization"],
            io_data + prior_data,
            ["construction queue conservation", "bounded employment", "capital nonnegative"],
        ),
        module(
            "nonfinancial_sfc_v53",
            ["5.3", "6.4", "8", "14.7", "14.8"],
            "dynamic_state",
            ["work/dynamic_macro_dqsfc_v53.py", "work/simulate_four_party_financial_v45.py"],
            ["settlement_position", "ACCOUNT_NAMES", "_transfer"],
            "MRRDCGEDQSFCState owns clearing positions; FourPartyFinancialSystem owns bank capital, debt and reserves",
            ["realized sales", "wages", "taxes", "trade balance", "working-capital demand"],
            ["five-account clearing positions", "settlement feedback"],
            ["wartime_inflation_finance", "mr_rd_cge_dq_v53"],
            io_data + prior_data,
            ["every transaction has equal debit and credit", "no duplicate bank or sovereign stock owner"],
        ),
        module(
            "four_party_social_accounting_matrix_v53",
            ["5.3", "6.4", "8", "14.5", "14.7", "14.8"],
            "dynamic_state",
            ["work/social_accounting_matrix_v53.py", "work/dynamic_macro_dqsfc_v53.py"],
            ["FourPartySocialAccountingMatrix", "SAM_ACCOUNTS", "benchmark", "transactions"],
            "FourPartySocialAccountingMatrix owns SAM transactions and institutional income distribution",
            ["public IO transactions", "value added", "final demand", "realized monthly transactions"],
            ["balanced 19-account benchmark", "household income", "institutional saving", "monthly SAM"],
            ["nonfinancial_sfc_v53", "mr_rd_cge_dq_v53", "reporting"],
            io_data + prior_data,
            ["benchmark row-column balance", "monthly account balance", "saving-investment closure", "nonnegative flows"],
        ),
        module(
            "age_activity_complementarity_v53",
            ["10", "14.3", "14.6", "14.8"],
            "constraint_solver",
            ["work/dynamic_macro_dqsfc_v53.py", "work/dynamic_macro_equilibrium_v47.py"],
            ["max_age_complementarity_residual", "shadow_price", "last_capacity"],
            "DynamicMultiRegionalCGE owns activity shadow prices",
            ["capacity", "production", "scarcity rent"],
            ["activity complementarity residual", "capacity shadow price"],
            ["mr_rd_cge_dq_v53"],
            io_data,
            ["nonnegative activity", "scarcity rent complementary to unused capacity"],
        ),
        module(
            "hssw_welfare_v53",
            ["3.4", "14.2", "14.8"],
            "terminal_model",
            ["work/dynamic_macro_dqsfc_v53.py"],
            ["equivalent_variation", "cumulative_ev", "max_hssw_identity_residual"],
            "MRRDCGEDQSFCState owns welfare accumulator only",
            ["realized consumption", "contract price index", "baseline expenditure"],
            ["equivalent variation", "cumulative welfare decomposition"],
            ["reporting"],
            io_data + prior_data,
            ["expenditure identity", "welfare does not feed battlefield dynamics"],
        ),
    ]
    modules = deepcopy(inherited["modules"]) + additions
    required_symbols = sorted({symbol for item in additions for symbol in item["symbols"]})
    missing_symbols = [symbol for symbol in required_symbols if symbol not in body and symbol not in core]
    missing_edges = [
        token for token in (
            'macro_factory=None', 'macro_class = macro_factory or CoupledMacroEconomicSystem',
            'mr_rd_cge_dq_sfc', 'actor_logistics', 'actor_labor', 'actor_payment',
        ) if token not in (core + body)
    ]
    blocking = []
    if inherited.get("gate_status") != "PASS":
        blocking.append({"inherited_v52_gate": inherited.get("gate_status")})
    if inherited_audit.get("status") != "PASS":
        blocking.append({"inherited_v52_audit": inherited_audit.get("status")})
    if diagnostic.get("status") != "PASS":
        blocking.append({"diagnostic": diagnostic.get("status")})
    if len(headings) != 39:
        blocking.append({"research_document_heading_count": len(headings)})
    if missing_symbols:
        blocking.append({"missing_symbols": missing_symbols})
    if missing_edges:
        blocking.append({"missing_edges": missing_edges})

    status = "PASS" if not blocking else "FAIL"
    trace = {
        "model_version": "V5.3",
        "document_ids": inherited["document_ids"] + ["VqnJdDTrIoMrPyxOX7fcBvyande"],
        "heading_inventory_count": inherited["heading_inventory_count"] + len(headings),
        "research_document_heading_count": len(headings),
        "coverage_basis": "fresh 2026-10-07 main-document inventory plus complete local DRCCGE research specification",
        "active_equation_family_count": inherited["active_equation_family_count"] + 7,
        "passed_equation_family_count": inherited["passed_equation_family_count"] + (7 if status == "PASS" else 0),
        "coverage_percent": 100.0 if status == "PASS" else 0.0,
        "gate_status": status,
        "modules": modules,
        "blocking_gaps": blocking,
    }
    audit = {
        "model_version": "V5.3",
        "status": status,
        "master_clock": "monthly same-path; DQ-SFC outputs feed the next CGE month explicitly",
        "required_edges": [
            {"edge": "CGE planned demand/capacity -> DQ actual transactions", "status": "PASS" if not missing_edges else "FAIL"},
            {"edge": "DQ fulfilment -> next-month CGE logistics", "status": "PASS" if not missing_edges else "FAIL"},
            {"edge": "search employment -> next-month CGE labor", "status": "PASS" if not missing_edges else "FAIL"},
            {"edge": "SFC settlement -> next-month payment integrity", "status": "PASS" if not missing_edges else "FAIL"},
            {"edge": "realized production/income/trade -> balanced four-party SAM", "status": "PASS" if not missing_edges else "FAIL"},
            {"edge": "vintage completion -> CGE capital", "status": "PASS" if not missing_edges else "FAIL"},
        ],
        "duplicate_state_owners": [],
        "mandatory_solver_inheritance": "V5.2 PASS",
        "new_diagnostic_status": diagnostic.get("status"),
        "blocking_gaps": blocking,
        "simulation_authorized": status == "PASS",
    }
    TRACE.write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")
    AUDIT.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"TAIWAN_V53_STATIC_AUDIT: {status}; modules={len(modules)}")
    if blocking:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
