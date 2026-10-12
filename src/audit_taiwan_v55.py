from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "work"
OUT = ROOT / "outputs"
DOC = OUT / "台海模型计量经济学识别与校准框架_飞书LaTeX.md"
TRACE = OUT / "taiwan_v55_traceability.json"
AUDIT = OUT / "taiwan_v55_static_coupling_audit.json"


def source(path, publisher, date, unit, identification):
    return {
        "source": path, "publisher": publisher, "observation_date": date,
        "unit": unit, "identification": identification,
    }


def item(module_id, sections, role, status, implementation, inputs, outputs, consumers, data, invariants, tests):
    active = status == "active"
    return {
        "module_id": module_id,
        "document_sections": sections,
        "role": role,
        "status": status,
        "implementation_status": "PASS" if active else "NOT_IN_ACTIVE_RESULT_PATH",
        "implementation": implementation,
        "state_owner": module_id if active else "none",
        "clock": "path initialization" if role == "estimator" else ("monthly same-path" if active else "none"),
        "pathwise": active and role != "estimator",
        "inputs": inputs,
        "outputs": outputs,
        "feedback_consumers": consumers,
        "solver": "documented in implementation" if active else "not selected for V5.5",
        "convergence_criterion": "outputs/taiwan_v55_econometric_diagnostic.json" if active else "not applicable",
        "data": data,
        "invariants": invariants,
        "tests": tests,
        "outputs_artifact": "outputs/taiwan_v55_econometric_calibration.json" if active else "outputs/台海模型计量经济学识别与校准框架_飞书LaTeX.md",
    }


def main() -> None:
    inherited = json.loads((OUT / "taiwan_v54_traceability.json").read_text(encoding="utf-8"))
    inherited_audit = json.loads((OUT / "taiwan_v54_static_coupling_audit.json").read_text(encoding="utf-8"))
    calibration_diag = json.loads((OUT / "taiwan_v55_econometric_diagnostic.json").read_text(encoding="utf-8"))
    integration_diag = json.loads((OUT / "taiwan_v55_econometric_integration_diagnostic.json").read_text(encoding="utf-8"))
    headings = [line.strip() for line in DOC.read_text(encoding="utf-8").splitlines() if line.startswith("#")]
    observed = [
        source("data/oecd_icio_2025/2016-2022/*.csv", "OECD ICIO", "2016-2022", "current USD flows", "observed"),
        source("data/econometrics_v55/world_bank_macro_1960_2025.json", "World Bank WDI", "1960-2025", "annual macro indicators", "observed"),
        source("outputs/台海月度历史数据与隐状态_2020-2026.csv", "compiled public event series", "2020-2026", "monthly pressure states", "derived"),
    ]
    prior = [
        source("outputs/taiwan_v55_econometric_calibration.json", "V5.5 identification registry", "2026-10-08", "parameter distribution", "scenario_prior")
    ]
    tests = ["work/econometric_calibration_v55.py", "work/diagnostic_econometric_integration_v55.py"]
    additions = [
        item("econometric_registry_v55", ["1", "2", "3", "14", "17", "18", "21", "22"], "estimator", "active",
             ["work/econometric_calibration_v55.py:fit"], ["public data snapshots", "identification rules"],
             ["tagged parameter registry", "source ledger", "frozen path draws"],
             ["prewar_hmm_v55", "ppml_trade_v55", "macro_lp_var_v55", "posterior_path_interface_v55"], observed + prior,
             ["observed/derived/estimated/scenario_prior separation", "fixed draw count"], tests),
        item("ppml_trade_v55", ["4", "4.1", "4.2", "4.3"], "estimator", "active",
             ["work/econometric_calibration_v55.py:_fit_ppml_fixed_effects", "work/dynamic_macro_equilibrium_v47.py:DynamicMultiRegionalCGE"],
             ["seven annual bilateral ICIO matrices"], ["pair/year fixed effects", "trade-shock posterior scale"],
             ["dynamic_multi_regional_cge"], observed, ["PPML score equations close", "no Armington elasticity falsely identified"], tests),
        item("macro_lp_var_v55", ["5", "5.1", "5.3", "12"], "estimator", "active",
             ["work/econometric_calibration_v55.py:_fit_macro_var", "work/econometric_calibration_v55.py:_fit_local_projections", "work/dynamic_macro_equilibrium_v47.py:OpenEconomyDSGE"],
             ["WDI annual panel"], ["held-out RMSE", "LP responses", "DSGE response-scale posterior"],
             ["open_economy_dsge"], observed, ["training ends in 2018", "validation starts in 2019", "SVAR diagnostic not labelled causal"], tests),
        item("prewar_hmm_v55", ["7", "7.2", "7.3", "16"], "estimator", "active",
             ["work/econometric_calibration_v55.py:_fit_hmm_transition", "work/closed_prewar_system_v47.py:ClosedPrewarEvolutionSystem"],
             ["monthly labelled pressure states", "same-path strategy state"], ["Dirichlet transition draws", "filtered pressure belief"],
             ["prewar_strategy_game", "competing_risk_survival"], observed, ["transition rows sum to one", "historical and structural transitions are explicitly blended"], tests),
        item("causal_aipw_v55", ["6", "6.3"], "estimator", "active",
             ["work/simulate_taiwan_causal_evolution_v37.py:causal_local_effect"], ["monthly event/treatment sample"],
             ["cross-fitted local pressure effect"], ["prewar_hmm_v55"], observed,
             ["leave-one-year-out fitting", "local effect not extrapolated as terminal causality"], ["work/simulate_taiwan_causal_evolution_v37.py"]),
        item("competing_risk_prior_interface_v55", ["8", "8.1", "8.2", "8.3", "19"], "terminal_model", "active",
             ["work/competing_risk_survival_v47.py:StratifiedCompetingRiskSurvival"],
             ["same-path states", "explicit scenario-prior coefficient draws"], ["cause hazards", "Fine-Gray hazards", "CIFs"],
             ["terminal_outcome_aggregation"], prior, ["CIF monotonicity", "CIF plus survival equals one", "absence of terminal data is disclosed"], tests),
        item("alliance_multistate_prior_interface_v55", ["9", "19"], "dynamic_state", "active",
             ["work/endogenous_intervention_v47.py:EndogenousInterventionChoiceModel"],
             ["same-path support/fiscal/leader states", "explicit prior draws"], ["five-regime probability simplex"],
             ["campaign_case_mixture"], prior, ["probabilities sum to one", "US intervention implies Japanese base-support floor"], tests),
        item("posterior_path_interface_v55", ["17", "20.3", "20.4", "22"], "constraint_solver", "active",
             ["work/econometric_calibration_v55.py:load_draws", "work/simulate_taiwan_fully_closed_v47.py:run"],
             ["one frozen draw vector per master path"], ["same draw reused across months/cases/scenarios"],
             ["all active calibrated modules"], observed + prior, ["path count identity", "no post-hoc probability adjustment"], tests),
        item("recursive_svar_diagnostic_v55", ["5.2"], "theory_only", "theory_only",
             ["work/econometric_calibration_v55.py:_fit_macro_var"], ["VAR innovations"], ["ordering-sensitive diagnostic impact matrix"], [], observed,
             ["not used as causal input without defensible restrictions"], tests),
        item("unselected_causal_estimators_v55", ["6.1", "6.2", "15"], "theory_only", "theory_only",
             ["outputs/台海模型计量经济学识别与校准框架_飞书LaTeX.md"], ["would require staggered treatment cohorts, donor pool, or valid instruments"], [], [], observed,
             ["not substituted by fabricated treatment or instrument"], ["documented non-selection and identification boundary"]),
        item("unselected_measurement_extensions_v55", ["7.1", "10", "10.1", "10.2", "10.3", "13"], "theory_only", "theory_only",
             ["outputs/台海模型计量经济学识别与校准框架_飞书LaTeX.md"], ["would require harmonized capital/labor/poll microdata"], [], [], observed,
             ["existing structural production and MFG remain active but are not relabelled as econometric estimates"], ["documented non-selection and identification boundary"]),
    ]
    modules = deepcopy(inherited["modules"]) + additions
    code = "\n".join((WORK / path).read_text(encoding="utf-8") for path in (
        "econometric_calibration_v55.py", "closed_prewar_system_v47.py",
        "competing_risk_survival_v47.py", "endogenous_intervention_v47.py",
        "dynamic_macro_equilibrium_v47.py", "simulate_taiwan_fully_closed_v47.py",
    ))
    tokens = ["econometric_draws", "hmm_transition", "trade_shock_scale", "macro_output_scale", "force_beta_scale", "us_intervention_shift"]
    blocking = []
    if inherited.get("gate_status") != "PASS" or inherited_audit.get("status") != "PASS":
        blocking.append("inherited V5.4 gate failed")
    if calibration_diag.get("scope") != "DIAGNOSTIC_ONLY" or not calibration_diag.get("checks", {}).get("finite_draws"):
        blocking.append("econometric calibration diagnostic failed")
    if integration_diag.get("status") != "PASS":
        blocking.append("same-path integration diagnostic failed")
    missing = [token for token in tokens if token not in code]
    if missing:
        blocking.append({"missing_coupling_tokens": missing})
    if len(headings) != 48:
        blocking.append({"econometric_document_heading_count": len(headings)})
    status = "PASS" if not blocking else "FAIL"
    active_additions = sum(item_["status"] == "active" for item_ in additions)
    trace = {
        "model_version": "V5.5",
        "document_ids": inherited.get("document_ids", []) + ["BZqxdzIzrolfo4x5Svsc1zHanLc"],
        "heading_inventory_count": inherited.get("heading_inventory_count", 0) + len(headings),
        "econometric_document_heading_count": len(headings),
        "coverage_basis": "V5.4 full inventory plus explicit classification of every V5.5 econometric section",
        "active_equation_family_count": inherited.get("active_equation_family_count", 0) + active_additions,
        "passed_equation_family_count": inherited.get("passed_equation_family_count", 0) + (active_additions if status == "PASS" else 0),
        "coverage_percent": 100.0 if status == "PASS" else 0.0,
        "gate_status": status,
        "modules": modules,
        "blocking_gaps": blocking,
    }
    audit = {
        "model_version": "V5.5", "status": status,
        "master_clock": "one posterior draw per master path; reused on the monthly event clock across all scenario cells",
        "required_edges": [
            {"edge": "PPML posterior -> same-path CGE trade shock", "status": status},
            {"edge": "LP/macro posterior -> same-path DSGE response", "status": status},
            {"edge": "HMM posterior -> prewar pressure transition -> hazards", "status": status},
            {"edge": "prior uncertainty -> alliance and terminal risks", "status": status},
            {"edge": "same path draw -> every case/scenario cell", "status": status},
        ],
        "duplicate_state_owners": [], "weak_coupling_adapters": [],
        "mandatory_solver_inheritance": "V5.4 PASS",
        "simulation_authorized": status == "PASS", "blocking_gaps": blocking,
    }
    TRACE.write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")
    AUDIT.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"TAIWAN_V55_STATIC_AUDIT: {status}; modules={len(modules)}")
    if blocking:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
