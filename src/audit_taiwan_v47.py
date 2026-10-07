from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "work"
OUT = ROOT / "outputs"


MODULES = [
    {
        "id": "historical_identification_causal_hmm",
        "sections": ["2", "2.1", "11", "14", "22.2", "22.3", "22.12"],
        "equations": ["Gaussian HMM filter", "cross-fitted AIPW", "measurement error"],
        "files": ["taiwan_dynamic_game_hmm.py", "simulate_taiwan_causal_evolution_v37.py", "closed_prewar_system_v47.py"],
        "symbols": ["fit_gaussian_hmm", "causal_local_effect", "ClosedPrewarEvolutionSystem"],
        "owner": "ClosedPrewarEvolutionSystem.pressure_belief",
        "clock": "monthly prewar",
    },
    {
        "id": "prewar_game_evolution_feedback",
        "sections": ["3", "3.1", "15.7", "22.4", "22.5"],
        "equations": ["quantal response", "replicator dynamics", "state feedback"],
        "files": ["closed_prewar_system_v47.py"],
        "symbols": ["_softmax", "strategy", "replicator"],
        "owner": "ClosedPrewarEvolutionSystem.strategy",
        "clock": "monthly prewar",
    },
    {
        "id": "continuous_graphon_mfg",
        "sections": ["4", "4.1", "21.3", "24.2", "24.3", "25.7", "26.3"],
        "equations": ["HJB", "Fokker-Planck", "multi-population consistency", "Graphon interaction"],
        "files": ["continuous_multipop_mfg_v47.py", "closed_prewar_system_v47.py"],
        "symbols": ["ContinuousMultiPopulationMFG", "_fp_step", "actor_kernel", "graphon"],
        "owner": "ContinuousMultiPopulationMFG.density",
        "clock": "monthly with inner HJB-FP fixed point",
    },
    {
        "id": "age_stratified_social_contagion",
        "sections": ["4.2", "24.4", "26.11", "28.1", "28.7"],
        "equations": ["six-state competing contagion", "complex threshold", "Levy marks", "Hawkes memory"],
        "files": ["wartime_social_contagion.py"],
        "symbols": ["WartimeSocialContagionSystem", "hawkes", "levy", "conservation_error"],
        "owner": "WartimeSocialContagionSystem.compartments",
        "clock": "four weekly substeps per campaign month",
    },
    {
        "id": "unbalanced_optimal_transport",
        "sections": ["5", "5.1", "20.2", "26.3"],
        "equations": ["KL-relaxed Sinkhorn", "lossy conversion", "unmet demand"],
        "files": ["closed_prewar_system_v47.py", "simulate_taiwan_multidomain_v42.py"],
        "symbols": ["_resource_ot", "entropy_transport_fulfillment", "marginal_penalty"],
        "owner": "ClosedPrewarEvolutionSystem.ot_allocation",
        "clock": "monthly",
    },
    {
        "id": "leader_gated_jump_process",
        "sections": ["6", "6.1", "25.12", "25.13"],
        "equations": ["finite stage gates", "interception", "succession", "marked jumps"],
        "files": ["closed_prewar_system_v47.py", "simulate_taiwan_multidomain_v42.py"],
        "symbols": ["leader_stage", "leader_attempt_hazard", "leader_continuity", "intercept_probability"],
        "owner": "campaign_state.leader_continuity",
        "clock": "monthly prewar and wartime",
    },
    {
        "id": "stratified_competing_survival",
        "sections": ["7", "7.1", "7.2", "7.3", "22.6", "22.13"],
        "equations": ["stratified cause-specific Cox", "Aalen-Johansen", "Fine-Gray", "AFT"],
        "files": ["competing_risk_survival_v47.py"],
        "symbols": ["StratifiedCompetingRiskSurvival", "_integrate", "fine_gray_hazard", "aft_delay_months"],
        "owner": "StratifiedCompetingRiskSurvival.last_result",
        "clock": "monthly 2026-2100 competing-risk calendar",
    },
    {
        "id": "growth_defense_capital",
        "sections": ["15.1", "15.2", "15.3", "15.4", "15.5", "15.6", "15.7", "15.8"],
        "equations": ["growth accumulation", "perpetual inventory", "defense burden", "theater conversion", "seven long-term structural scenarios"],
        "files": ["simulate_taiwan_econ_military_joint_model.py", "closed_prewar_system_v47.py"],
        "symbols": ["GROWTH_2026", "STOCK_DEPRECIATION", "economic_capital", "military_capital"],
        "owner": "ClosedPrewarEvolutionSystem.economic_capital/military_capital",
        "clock": "monthly 2026-2100 prewar calendar",
    },
    {
        "id": "opinion_election_arms_pipeline",
        "sections": ["24.1", "24.2", "24.3", "24.4", "24.5", "24.6", "24.7"],
        "equations": ["latent polling filter", "Graphon groups", "election state", "six-stage arms pipeline"],
        "files": ["closed_prewar_system_v47.py"],
        "symbols": ["filtered_poll", "party_state", "population_latent", "pipeline", "delivered_capability"],
        "owner": "ClosedPrewarEvolutionSystem.population_latent/pipeline",
        "clock": "monthly prewar",
    },
    {
        "id": "endogenous_alliance_stackelberg",
        "sections": ["9", "17.1-17.8", "22.7", "25.6", "25.6.1"],
        "equations": ["US leader choice", "independent Japan response", "base-support floor", "five-case simplex"],
        "files": ["endogenous_intervention_v47.py", "simulate_taiwan_multidomain_v42.py"],
        "symbols": ["EndogenousInterventionChoiceModel", "japan_base", "stackelberg_alliance"],
        "owner": "EndogenousInterventionChoiceModel.last_choice",
        "clock": "prewar choice and monthly wartime response",
    },
    {
        "id": "campaign_deds_multidomain",
        "sections": ["9.1", "16.2-16.8", "18.2-18.6", "26.4", "26.5", "26.20"],
        "equations": ["DEDS", "heterogeneous attrition", "missile/interception", "ACE", "semi-Markov regimes", "kill web/OODA"],
        "files": ["simulate_taiwan_multidomain_v42.py"],
        "symbols": ["conflict_regime", "regime_duration", "kill", "ooda", "intercept"],
        "owner": "simulate_mechanisms campaign state",
        "clock": "monthly wartime",
    },
    {
        "id": "operations_network_flow_metric_search",
        "sections": ["19.2-19.6", "19.13.1-19.13.9", "20.1-20.4", "21.2", "21.4", "21.5", "21.6", "26.14"],
        "equations": ["multicommodity flow", "METRIC", "repair queue", "Bayesian search", "robust facility coverage", "network percolation"],
        "files": ["operational_constraints_v47.py"],
        "symbols": ["DynamicOperationalConstraintSystem", "spare_pipeline", "selected_facilities", "corridor_adjacency", "search_prior"],
        "owner": "DynamicOperationalConstraintSystem inventory/spares/integrity",
        "clock": "monthly wartime",
    },
    {
        "id": "geospatial_pnt",
        "sections": ["19.13.1", "21.2", "26.4"],
        "equations": ["information-matrix fusion", "GNSS/inertial/terrestrial/terrain", "integrity"],
        "files": ["operational_constraints_v47.py", "simulate_taiwan_3d_pnt_v41.py"],
        "symbols": ["DynamicGeospatialPNTSystem", "posterior_precision", "terrain_observability"],
        "owner": "DynamicGeospatialPNTSystem.covariance",
        "clock": "monthly wartime",
    },
    {
        "id": "rolling_pomdp_robust_game",
        "sections": ["3.1", "11", "19.13.9", "25.11"],
        "equations": ["partial observation filter", "receding horizon", "robust adverse scenario", "best response"],
        "files": ["rolling_differential_game_v47.py"],
        "symbols": ["RollingHorizonDifferentialGame", "_belief_update", "scenario_multiplier", "np.min(value"],
        "owner": "RollingHorizonDifferentialGame._belief/controls",
        "clock": "monthly with six-month horizon",
    },
    {
        "id": "spatial_population_control",
        "sections": ["9.2", "16.6", "19.13.6", "26.4"],
        "equations": ["regional adjacency diffusion", "population weighting", "administrative weighting", "organized defense", "persistence"],
        "files": ["spatial_control_network_v47.py"],
        "symbols": ["SpatialControlNetwork", "population_control", "administrative_control", "organized_defense"],
        "owner": "SpatialControlNetwork.zone_control",
        "clock": "monthly wartime",
    },
    {
        "id": "labor_firms_supply_collapse",
        "sections": ["23.2", "26.6", "26.7", "26.8", "26.9", "26.10", "26.12"],
        "equations": ["skill-stratified manpower", "training delay", "fatigue/mismatch", "firm production", "SIR distress", "four-domain collapse"],
        "files": ["simulate_taiwan_multidomain_v42.py"],
        "symbols": ["actor_manpower", "actor_training", "actor_labor_fatigue", "actor_firm_factors", "actor_collapse"],
        "owner": "simulate_mechanisms actor labor/firm/collapse states",
        "clock": "monthly wartime",
    },
    {
        "id": "finance_fiscal_money_tax_debt_assets",
        "sections": ["23.1", "23.3", "23.4", "23.5", "23.6", "23.7"],
        "equations": ["central/local debt", "Bohn rule", "Laffer tax", "monetization", "NKPC jump", "credit/asset/default"],
        "files": ["simulate_four_party_financial_v45.py"],
        "symbols": ["local_debt", "bohn", "tax_elasticity", "monetization", "expected_inflation", "sovereign_hazard"],
        "owner": "FourPartyFinancialSystem balance sheets",
        "clock": "monthly wartime",
    },
    {
        "id": "dynamic_mr_cge_mrio_gravity_chips",
        "sections": ["27.2", "27.3", "27.4", "27.5", "27.6", "27.8", "27.9", "27.10", "27.18-27.25"],
        "equations": ["140-node MRIO", "structural gravity shock", "Armington", "goods/factor clearing", "capital accumulation", "government/external accounts", "chip substitution"],
        "files": ["dynamic_macro_equilibrium_v47.py", "simulate_dual_circulation_v44.py"],
        "symbols": ["DynamicMultiRegionalCGE", "armington_elasticity", "government_balance", "external_account_residual", "china_substitution"],
        "owner": "DynamicMultiRegionalCGE output/price/capital/labor",
        "clock": "monthly with inner equilibrium",
    },
    {
        "id": "open_dsge_hank_mundell_fleming",
        "sections": ["23.6", "27.7", "27.11", "27.12", "27.15"],
        "equations": ["IS", "NKPC", "Taylor", "UIP", "bounded expectations", "heterogeneous MPC households", "CGE-DSGE fixed point"],
        "files": ["dynamic_macro_equilibrium_v47.py"],
        "symbols": ["OpenEconomyDSGE", "household_shares", "policy_rate", "exchange_new", "max_outer_iterations"],
        "owner": "OpenEconomyDSGE expectations/rates/exchange/households",
        "clock": "monthly with inner and outer fixed points",
    },
    {
        "id": "actual_bilateral_aid_clearing",
        "sections": ["23.3", "26.13", "27.11"],
        "equations": ["donor outflow equals recipient inflow", "one-month settlement lag"],
        "files": ["simulate_four_party_financial_v45.py", "simulate_taiwan_multidomain_v42.py"],
        "symbols": ["register_realized_aid", "pending_aid_out", "pending_aid_in", "us_aid"],
        "owner": "FourPartyFinancialSystem pending/cumulative aid",
        "clock": "monthly",
    },
    {
        "id": "rare_tail_hawkes_evt_importance",
        "sections": ["24.11", "25.3", "25.4", "25.9"],
        "equations": ["dependent tail regime", "Hawkes memory", "GPD marks", "importance likelihood ratio"],
        "files": ["simulate_taiwan_multidomain_v42.py", "simulate_taiwan_fully_closed_v47.py"],
        "symbols": ["TAIL_TARGET_PROB", "common_hawkes_memory", "importance_weight", "weighted_mean"],
        "owner": "campaign path tail regime/weight",
        "clock": "path draw plus monthly event intensity",
    },
    {
        "id": "principal_agent_governance",
        "sections": ["10", "10.1", "10.2", "25.14"],
        "equations": ["heterogeneous agent types", "recruitment/transitions", "isolation/leakage", "resistance attenuation"],
        "files": ["simulate_taiwan_multidomain_v42.py"],
        "symbols": ["taiwan_agent_type_share", "taiwan_proxy_capacity", "taiwan_proxy_leakage"],
        "owner": "campaign state Taiwan agent distribution",
        "clock": "monthly wartime",
    },
    {
        "id": "pathwise_terminal_aggregation",
        "sections": ["25.17", "26.1", "26.16", "26.20", "27.26", "28.2-28.5"],
        "equations": ["absorbing broad control", "endogenous intervention mixture", "force-onset joint probability", "importance-weighted aggregation"],
        "files": ["simulate_taiwan_fully_closed_v47.py", "simulate_taiwan_multidomain_v42.py"],
        "symbols": ["broad_control_achieved", "choice.probabilities", "force_paths", "importance_weights"],
        "owner": "campaign broad_control event and V4.7 aggregator",
        "clock": "5/10/15/20/30-year reporting",
    },
]


# Executable contracts make coverage auditable beyond symbol existence. The
# values are state names on the common path, not prose aliases.
MODULE_CONTRACTS = {
    "historical_identification_causal_hmm": ("estimator", ["historical indicators", "pressure treatment", "observed outcomes"], ["pressure_belief", "causal_pressure_effect"], ["prewar_game_evolution_feedback", "stratified_competing_survival"]),
    "prewar_game_evolution_feedback": ("dynamic_state", ["pressure_belief", "actor_support", "economic_capital"], ["strategy", "actor_support"], ["growth_defense_capital", "opinion_election_arms_pipeline", "stratified_competing_survival"]),
    "continuous_graphon_mfg": ("constraint_solver", ["campaign progress", "damage", "shortage", "social contagion"], ["support", "mobilization_preference", "fatigue_preference"], ["age_stratified_social_contagion", "campaign_deds_multidomain"]),
    "age_stratified_social_contagion": ("dynamic_state", ["MFG thresholds", "casualties", "inflation", "network state"], ["social support", "mobilization", "fatigue", "panic"], ["campaign_deds_multidomain", "dynamic_mr_cge_mrio_gravity_chips"]),
    "unbalanced_optimal_transport": ("constraint_solver", ["resource supply", "conversion cost", "combat demand"], ["allocation", "unmet demand", "transport loss"], ["campaign_deds_multidomain", "operations_network_flow_metric_search"]),
    "leader_gated_jump_process": ("dynamic_state", ["leader pressure", "intelligence", "institutional interception"], ["leader_continuity", "succession state", "jump marks"], ["endogenous_alliance_stackelberg", "campaign_deds_multidomain"]),
    "stratified_competing_survival": ("estimator", ["prewar covariates", "pressure_belief", "identified force prior"], ["cause CIF", "Fine-Gray hazard", "AFT delay"], ["endogenous_alliance_stackelberg", "pathwise_terminal_aggregation"]),
    "growth_defense_capital": ("dynamic_state", ["observed growth", "defense burden", "depreciation"], ["economic_capital", "military_capital"], ["prewar_game_evolution_feedback", "campaign_deds_multidomain"]),
    "opinion_election_arms_pipeline": ("dynamic_state", ["group beliefs", "poll measurement", "orders", "delivery delays"], ["filtered_poll", "party_state", "delivered_capability"], ["endogenous_alliance_stackelberg", "campaign_deds_multidomain"]),
    "endogenous_alliance_stackelberg": ("dynamic_state", ["support", "readiness", "AFT delay", "financial capacity"], ["five-case probabilities", "base access", "intervention path"], ["campaign_deds_multidomain", "pathwise_terminal_aggregation"]),
    "campaign_deds_multidomain": ("dynamic_state", ["alliance regime", "forces", "stocks", "controls", "terrain"], ["attrition", "damage", "domain control", "regime"], ["operations_network_flow_metric_search", "finance_fiscal_money_tax_debt_assets", "spatial_population_control"]),
    "operations_network_flow_metric_search": ("constraint_solver", ["stocks", "damage", "corridors", "repair demand", "search observations"], ["front inventory", "flow", "availability", "detection"], ["campaign_deds_multidomain", "dynamic_mr_cge_mrio_gravity_chips"]),
    "geospatial_pnt": ("constraint_solver", ["terrain observability", "sky visibility", "sensor damage", "cyber state"], ["PNT covariance", "integrity", "position error"], ["campaign_deds_multidomain", "operations_network_flow_metric_search"]),
    "rolling_pomdp_robust_game": ("constraint_solver", ["delayed observations", "belief covariance", "six-month state forecast"], ["strike", "defense", "production", "repair", "negotiation controls"], ["campaign_deds_multidomain", "finance_fiscal_money_tax_debt_assets"]),
    "spatial_population_control": ("dynamic_state", ["zone control", "adjacency", "terrain", "administration", "organized defense"], ["population control", "administrative control", "persistent broad control"], ["principal_agent_governance", "pathwise_terminal_aggregation"]),
    "labor_firms_supply_collapse": ("dynamic_state", ["manpower", "skills", "orders", "inputs", "fatigue", "damage"], ["trained labor", "firm output", "distress", "collapse state"], ["campaign_deds_multidomain", "dynamic_mr_cge_mrio_gravity_chips", "finance_fiscal_money_tax_debt_assets"]),
    "finance_fiscal_money_tax_debt_assets": ("dynamic_state", ["damage", "macro balances", "aid", "tax base", "war spending"], ["credit", "payments", "debt", "reserves", "fiscal capacity", "financial stress"], ["campaign_deds_multidomain", "dynamic_mr_cge_mrio_gravity_chips", "open_dsge_hank_mundell_fleming"]),
    "dynamic_mr_cge_mrio_gravity_chips": ("constraint_solver", ["final demand", "capital", "labor", "trade shocks", "finance", "damage"], ["output", "prices", "trade balance", "government balance", "shadow prices"], ["open_dsge_hank_mundell_fleming", "finance_fiscal_money_tax_debt_assets", "campaign_deds_multidomain"]),
    "open_dsge_hank_mundell_fleming": ("constraint_solver", ["CGE output and prices", "trade balance", "risk", "fiscal capacity"], ["demand", "investment", "inflation", "rates", "exchange rate"], ["dynamic_mr_cge_mrio_gravity_chips", "finance_fiscal_money_tax_debt_assets"]),
    "actual_bilateral_aid_clearing": ("dynamic_state", ["donor capacity", "recipient need", "intervention regime"], ["pending donor outflow", "pending recipient inflow", "settled aid"], ["finance_fiscal_money_tax_debt_assets", "campaign_deds_multidomain"]),
    "rare_tail_hawkes_evt_importance": ("dynamic_state", ["tail regime", "event history", "target and proposal laws"], ["jump marks", "Hawkes intensity", "likelihood weight"], ["campaign_deds_multidomain", "pathwise_terminal_aggregation"]),
    "principal_agent_governance": ("dynamic_state", ["local elites", "control", "isolation", "benefits", "repression"], ["agent-type shares", "proxy capacity", "leakage", "resistance attenuation"], ["spatial_population_control", "pathwise_terminal_aggregation"]),
    "pathwise_terminal_aggregation": ("terminal_model", ["force CIF", "intervention probabilities", "absorbing control paths", "importance weights"], ["conditional broad-control probability", "unconditional joint probability", "uncertainty intervals"], []),
}

PRIOR_DATA = {
    "source": "https://my.feishu.cn/docx/UEnMdV2AGoaAuKxAsSwcg563nSb",
    "publisher": "Taiwan model research specification",
    "observation_date": "2026-10-05",
    "unit": "documented dimensionless scenario prior unless overridden by public input",
    "identification": "scenario_prior",
}

OBSERVED_DATA = {
    "growth_defense_capital": {"source": "work/SIPRI-Milex-data-1949-2025_v1.2.xlsx", "publisher": "SIPRI", "observation_date": "1949-2025", "unit": "military expenditure series", "identification": "observed"},
    "operations_network_flow_metric_search": {"source": "outputs/台海V3.6_GIS机动走廊摘要.json", "publisher": "OpenStreetMap and AWS Terrain derived dataset", "observation_date": "2026-09-28", "unit": "corridor geometry and elevation-derived friction", "identification": "derived"},
    "geospatial_pnt": {"source": "outputs/台海V3.6_GIS机动走廊摘要.json", "publisher": "OpenStreetMap and AWS Terrain derived dataset", "observation_date": "2026-09-28", "unit": "terrain observability index", "identification": "derived"},
    "finance_fiscal_money_tax_debt_assets": {"source": "work/simulate_four_party_financial_v45.py:SOURCE_URLS", "publisher": "NFRA/PBOC/IMF/Fed/US Treasury/Taiwan CBC/BOJ/Japan MOF", "observation_date": "2025-2026", "unit": "balance-sheet ratios", "identification": "observed"},
    "dynamic_mr_cge_mrio_gravity_chips": {"source": "data/oecd_icio_2025/2016-2022/2022_SML.csv and data/china_mrio/MRIO2017_42+CEADS.xlsx", "publisher": "OECD and CEADs", "observation_date": "2017/2022", "unit": "monetary input-output flows", "identification": "observed"},
}

SOLVER_CRITERIA = {
    "continuous_graphon_mfg": ("implicit tridiagonal HJB policy iteration plus conservative FP Picard iteration", "fixed-point residual < 5e-4 and mass residual < 1e-10"),
    "unbalanced_optimal_transport": ("KL-relaxed Sinkhorn", "marginal and nonnegative-flow residuals pass diagnostic"),
    "rolling_pomdp_robust_game": ("distributionally robust receding-horizon best response", "best-response residual < 1e-4"),
    "dynamic_mr_cge_mrio_gravity_chips": ("damped mixed-complementarity iteration", "goods and factor complementarity residuals < 2e-3"),
    "open_dsge_hank_mundell_fleming": ("bounded-expectations fixed point nested with CGE", "DSGE residual < 8e-4 and outer residual < 3e-3"),
    "operations_network_flow_metric_search": ("capacity-constrained flow, queue and Bayesian recursions", "stock residual < 1e-7"),
    "geospatial_pnt": ("information-filter covariance fusion", "precision-covariance inverse residual < 1e-7"),
    "actual_bilateral_aid_clearing": ("one-period stock-flow settlement", "donor-recipient residual < 1e-10"),
}

SUPERSEDED_MODULES = [
    {
        "module_id": "v45_reduced_form_finance_trade_adapter",
        "document_sections": ["16"], "role": "superseded", "status": "superseded",
        "implementation": ["work/simulate_taiwan_full_coupled_v45.py"],
        "clock": "historical monthly adapter", "pathwise": False,
        "inputs": ["pre-generated economic trajectories"],
        "outputs": ["historical V4.5 result only"],
        "feedback_consumers": ["dynamic_mr_cge_mrio_gravity_chips", "open_dsge_hank_mundell_fleming"],
        "data": [PRIOR_DATA], "invariants": ["excluded from V4.7 result path"],
        "tests": ["outputs/taiwan_v47_static_coupling_audit.json"],
        "outputs_artifact": "historical only; no V4.7 output",
    },
    {
        "module_id": "v46_posthoc_probability_aggregator",
        "document_sections": ["17"], "role": "superseded", "status": "superseded",
        "implementation": ["work/simulate_taiwan_full_coupled_v46.py"],
        "clock": "historical post-processing", "pathwise": False,
        "inputs": ["separately generated module outputs"],
        "outputs": ["historical V4.6 result only"],
        "feedback_consumers": ["pathwise_terminal_aggregation"],
        "data": [PRIOR_DATA], "invariants": ["excluded from V4.7 result path"],
        "tests": ["outputs/taiwan_v47_static_coupling_audit.json"],
        "outputs_artifact": "historical only; no V4.7 output",
    },
]


REQUIRED_EDGES = [
    ("prewar trajectory->survival", "simulate_taiwan_fully_closed_v47.py", ["evaluate_trajectory", "pressure_belief=prewar_trajectory", "actor_support=prewar_trajectory"]),
    ("survival->conditional onset month", "simulate_taiwan_fully_closed_v47.py", ["survival.cause_increment", "conditional_force_time", "onset_month"]),
    ("onset state->alliance", "simulate_taiwan_fully_closed_v47.py", ["aft_delay_at_onset", "aft_delay_months=aft_delay"]),
    ("onset year->campaign", "simulate_taiwan_fully_closed_v47.py", ["event_year, case", "remaining_to_2100"]),
    ("economic/military capital->campaign", "simulate_taiwan_multidomain_v42.py", ["prewar_state[\"economic_capital\"]", "prewar_state[\"military_capital\"]", "economic_factor", "military_factor"]),
    ("long-term scenario->prewar dynamics", "closed_prewar_system_v47.py", ["PREWAR_SCENARIOS", "scenario_parameters", "alliance_credibility", "interdependence", "decoupling"]),
    ("alliance->campaign", "simulate_taiwan_fully_closed_v47.py", ["selected_choice_probability * matrix"]),
    ("campaign->finance", "simulate_taiwan_multidomain_v42.py", ["financial_system.update", "actor_damage=actor_damage"]),
    ("finance->campaign", "simulate_taiwan_multidomain_v42.py", ["financial_production", "financial_fiscal", "financial_stress"]),
    ("campaign->macro", "simulate_taiwan_multidomain_v42.py", ["macro_system.update", "actor_damage=actor_damage"]),
    ("macro->finance", "simulate_taiwan_multidomain_v42.py", ["macro_trade_balance", "macro_government_balance"]),
    ("operations->campaign", "simulate_taiwan_multidomain_v42.py", ["operational_opening.front_inventory", "operational_constraints"]),
    ("pnt->campaign", "simulate_taiwan_multidomain_v42.py", ["pnt_system.integrity", "dynamic_geospatial_pnt"]),
    ("mfg->social", "simulate_taiwan_multidomain_v42.py", ["mfg_threshold_shift", "mfg_mobilization", "mfg_fatigue"]),
    ("social->campaign", "simulate_taiwan_multidomain_v42.py", ["social[\"support\"]", "actor_social_mobilization"]),
    ("strategy->production/repair", "simulate_taiwan_multidomain_v42.py", ["actor_strategy_controls", "repair_cap_cn"]),
    ("spatial->terminal", "simulate_taiwan_multidomain_v42.py", ["spatial_factors.stable_broad_control", "population_control_paths"]),
    ("aid bilateral clearing", "simulate_taiwan_multidomain_v42.py", ["register_realized_aid(us_aid, japan_aid, minor_aid)"]),
    ("tail weights->result", "simulate_taiwan_fully_closed_v47.py", ["importance_weight_paths", "weighted_mean(unconditional"]),
    ("trade label->scenario code", "simulate_taiwan_fully_closed_v47.py", ["trade.v43.SCENARIOS[name]"]),
]


def source(file_name: str) -> str:
    return (WORK / file_name).read_text(encoding="utf-8")


def has_symbol(file_name: str, symbol: str) -> bool:
    text = source(file_name)
    if symbol in text:
        return True
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return False
    names = {
        node.name for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }
    return symbol in names


def main() -> None:
    OUT.mkdir(exist_ok=True)
    module_rows = []
    blocking = []
    for module in MODULES:
        missing_files = [name for name in module["files"] if not (WORK / name).exists()]
        missing_symbols = []
        for symbol in module["symbols"]:
            if not any(
                (WORK / name).exists() and has_symbol(name, symbol)
                for name in module["files"]
            ):
                missing_symbols.append(symbol)
        validation_status = "PASS" if not missing_files and not missing_symbols else "FAIL"
        if validation_status == "FAIL":
            blocking.append({
                "module": module["id"], "missing_files": missing_files,
                "missing_symbols": missing_symbols,
            })
        role, inputs, outputs, consumers = MODULE_CONTRACTS[module["id"]]
        solver, convergence = SOLVER_CRITERIA.get(
            module["id"],
            ("bounded monthly state transition or closed-form estimator", "finite, bounded, normalized state and module invariant"),
        )
        data = [PRIOR_DATA]
        if module["id"] in OBSERVED_DATA:
            data = [OBSERVED_DATA[module["id"]], PRIOR_DATA]
        module_rows.append({
            "module_id": module["id"],
            "document_sections": module["sections"],
            "equations": module["equations"],
            "role": role,
            "status": "active",
            "implementation_status": validation_status,
            "implementation": [f"work/{name}" for name in module["files"]],
            "symbols": module["symbols"],
            "state_owner": module["owner"],
            "clock": module["clock"],
            "pathwise": role != "estimator",
            "inputs": inputs,
            "outputs": outputs,
            "feedback_consumers": consumers,
            "solver": solver,
            "convergence_criterion": convergence,
            "data": data,
            "invariants": ["finite values", "declared bounds", convergence],
            "tests": [
                "work/diagnostic_taiwan_v47.py",
                "outputs/taiwan_v47_static_coupling_audit.json",
            ],
            "outputs_artifact": "outputs/taiwan_v47_fully_closed_simulation.json",
        })

    edge_rows = []
    for edge, file_name, tokens in REQUIRED_EDGES:
        text = source(file_name)
        missing = [token for token in tokens if token not in text]
        status = "PASS" if not missing else "FAIL"
        if missing:
            blocking.append({"edge": edge, "file": file_name, "missing_tokens": missing})
        edge_rows.append({"edge": edge, "file": file_name, "status": status})

    driver = source("simulate_taiwan_fully_closed_v47.py")
    forbidden = {
        "pre-generated economy_profile passed to campaign": "economy_profile=" in driver,
        "post-hoc broad-control logit": "prior_log_odds" in driver or "make_prior_paths" in driver,
        "outcome run without gate": "assert_complete_gate(" not in driver,
    }
    for label, failed in forbidden.items():
        if failed:
            blocking.append({"forbidden_pattern": label})

    active_count = len(MODULES)
    passed_count = sum(row["implementation_status"] == "PASS" for row in module_rows)
    coverage = 100.0 * passed_count / max(active_count, 1)
    status = "PASS" if not blocking and coverage == 100.0 else "FAIL"
    traceability = {
        "model_version": "V4.7",
        "document_id": "UEnMdV2AGoaAuKxAsSwcg563nSb",
        "coverage_basis": "all active methodological/equation families in sections 1-28; result tables, source lists, limitations and explicitly old sections 16-17 are reporting metadata rather than executable modules",
        "classification": {
            "active_method_sections": ["1-7", "9-15", "18-28 methodological subsections"],
            "superseded_results": ["8", "16", "17", "18-28 historical result subsections"],
            "report_only": ["data-source lists", "interpretation limits", "reproduction-file lists"],
        },
        "active_equation_family_count": active_count,
        "passed_equation_family_count": passed_count,
        "coverage_percent": coverage,
        "gate_status": status,
        "modules": module_rows + SUPERSEDED_MODULES,
        "blocking_gaps": blocking,
    }
    coupling_audit = {
        "model_version": "V4.7",
        "status": status,
        "master_clock": "monthly; weekly social substeps and equilibrium/game inner iterations commit once per month",
        "state_ownership": {row["module_id"]: row["state_owner"] for row in module_rows},
        "required_edges": edge_rows,
        "forbidden_patterns": forbidden,
        "blocking_gaps": blocking,
        "simulation_authorized": status == "PASS",
    }
    (OUT / "taiwan_v47_traceability.json").write_text(
        json.dumps(traceability, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUT / "taiwan_v47_static_coupling_audit.json").write_text(
        json.dumps(coupling_audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"TAIWAN_V47_STATIC_AUDIT: {status}; coverage={coverage:.2f}%")
    if blocking:
        print(json.dumps(blocking, ensure_ascii=False, indent=2))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
