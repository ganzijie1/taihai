"""Dynamic validation and same-seed grouped ablations for the V4.7 model.

This is a validation runner, not an outcome-probability estimator.  It uses a
small path count and short horizons to prove that every active equation family
has an executable state owner and that disabling each campaign group changes
the same-path trajectory.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import simulate_taiwan_causal_evolution_v37 as causal
import simulate_taiwan_full_coupled_v45 as prewar_anchor
import simulate_taiwan_multidomain_v42 as battle
from closed_prewar_system_v47 import ClosedPrewarEvolutionSystem
from competing_risk_survival_v47 import StratifiedCompetingRiskSurvival
from continuous_multipop_mfg_v47 import ContinuousMultiPopulationMFG
from dynamic_macro_equilibrium_v47 import CoupledMacroEconomicSystem
from endogenous_intervention_v47 import EndogenousInterventionChoiceModel
from operational_constraints_v47 import DynamicGeospatialPNTSystem, DynamicOperationalConstraintSystem
from rolling_differential_game_v47 import RollingHorizonDifferentialGame
from simulate_four_party_financial_v45 import FourPartyFinancialSystem
from spatial_control_network_v47 import SpatialControlNetwork
from wartime_social_contagion import WartimeSocialContagionSystem


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "taiwan_v47_dynamic_ablation_validation.json"
TRACE = ROOT / "outputs" / "taiwan_v47_traceability.json"
PATHS = 4
MONTHS = 12
ACTORS = ("china", "taiwan", "united_states", "japan")


def _flatten_numeric(value):
    if isinstance(value, dict):
        for key in sorted(value):
            yield from _flatten_numeric(value[key])
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _flatten_numeric(item)
    elif isinstance(value, (int, float, np.number)) and np.isfinite(value):
        yield float(value)


def _systems(seed, terrain, support, enabled):
    return {
        "financial_system": FourPartyFinancialSystem(PATHS, MONTHS, seed + 1, "limited", "baseline") if "finance" in enabled else None,
        "social_system": WartimeSocialContagionSystem(PATHS, seed + 2, support) if "social" in enabled else None,
        "mfg_system": ContinuousMultiPopulationMFG(PATHS, seed + 3, support) if "mfg" in enabled else None,
        "macro_system": CoupledMacroEconomicSystem(PATHS, seed + 4, "baseline") if "macro" in enabled else None,
        "operations_system": DynamicOperationalConstraintSystem(PATHS, seed + 5, terrain) if "operations" in enabled else None,
        "pnt_system": DynamicGeospatialPNTSystem(PATHS, terrain) if "pnt" in enabled else None,
        "strategy_system": RollingHorizonDifferentialGame(PATHS, seed + 6) if "strategy" in enabled else None,
        "spatial_system": SpatialControlNetwork(PATHS, seed + 7, terrain) if "spatial" in enabled else None,
    }


def _campaign(seed, terrain, sipri, prewar, enabled, flags=None):
    systems = _systems(seed, terrain, prewar["actor_support"], enabled)
    snapshots, _ = battle.simulate_mechanisms(
        np.random.default_rng(seed + 8), np.full(PATHS, 2035), "limited", "central",
        terrain, sipri, prewar_state=prewar, module_flags=flags, **systems,
    )
    diag = snapshots[MONTHS]["diagnostics"]
    vector = np.asarray(list(_flatten_numeric(diag)), dtype=float)
    return diag, vector


def main():
    trace = json.loads(TRACE.read_text(encoding="utf-8"))
    active = [m["module_id"] for m in trace["modules"] if m["status"] == "active"]
    battle.PATHS = PATHS
    battle.MONTHS = MONTHS
    battle.HORIZON_MONTHS = (MONTHS,)
    _, _, terrain, sipri = battle.load_inputs()

    observed = prewar_anchor.prewar_state(PATHS, 4701)
    local_effect = causal.causal_local_effect(4702)
    prewar_system = ClosedPrewarEvolutionSystem(
        PATHS, 4703, observed, causal_pressure_effect=local_effect["estimate_log_points"]
    )
    trajectory = prewar_system.run_trajectory(months=24)
    prewar = {
        key: trajectory[key][-1]
        for key in ("actor_support", "taiwan_readiness", "leader_continuity", "economic_capital", "military_capital")
    }
    enabled = {"finance", "social", "mfg", "macro", "operations", "pnt", "strategy", "spatial"}
    base_diag, base_vector = _campaign(4711, terrain, sipri, prewar, enabled)

    variants = {
        "without_social_mfg": (enabled - {"social", "mfg"}, {"graphon_social": False}),
        "without_macro_finance": (enabled - {"macro", "finance"}, None),
        "without_operations_pnt": (enabled - {"operations", "pnt"}, {"logistics": False, "terrain_pnt": False}),
        "without_rolling_game": (enabled - {"strategy"}, None),
        "without_spatial_governance": (enabled - {"spatial"}, {"principal_agent": False}),
        "without_alliance": (enabled, {"stackelberg_alliance": False}),
        "without_optimal_transport": (enabled, {"optimal_transport": False}),
        "without_coupled_legacy_mechanisms": (
            enabled,
            {
                "stackelberg_alliance": False,
                "optimal_transport": False,
                "graphon_social": False,
                "principal_agent": False,
            },
        ),
    }
    group_rows = []
    for name, (active_systems, flags) in variants.items():
        _, vector = _campaign(4711, terrain, sipri, prewar, active_systems, flags)
        common = min(len(base_vector), len(vector))
        delta = float(np.linalg.norm(base_vector[:common] - vector[:common]) / max(np.sqrt(common), 1.0))
        group_rows.append({"ablation": name, "rms_numeric_trajectory_delta": delta, "pass": delta > 1e-10})

    survival_model = StratifiedCompetingRiskSurvival(PATHS, 4721)
    surv = survival_model.evaluate_trajectory(
        actor_support=trajectory["actor_support"], taiwan_readiness=trajectory["taiwan_readiness"],
        leader_continuity=trajectory["leader_continuity"], pressure_belief=trajectory["pressure_belief"],
        economic_capital=trajectory["economic_capital"], interdependence=trajectory["interdependence"],
        target_force_probability=0.01861111111111111, calibration_months=24,
    )
    choice = EndogenousInterventionChoiceModel(PATHS, 4722).probabilities(
        actor_support=trajectory["actor_support"][-1], taiwan_readiness=trajectory["taiwan_readiness"][-1],
        leader_continuity=trajectory["leader_continuity"][-1], arms_support_us=trajectory["arms_support_us"][-1],
        arms_support_japan=trajectory["arms_support_japan"][-1], fiscal_capacity=np.ones((PATHS, 4)) * .65,
        financial_stress=np.ones((PATHS, 4)) * .15, aft_delay_months=np.ones(PATHS) * .8,
    )
    structural = {
        "causal_effect_nonzero": abs(local_effect["estimate_log_points"]) > 1e-12,
        "prewar_state_moves": float(np.linalg.norm(trajectory["actor_support"][-1] - trajectory["actor_support"][0])) > 1e-10,
        "survival_simplex": surv.simplex_residual < 1e-10,
        "alliance_simplex": float(np.max(np.abs(choice.probabilities.sum(axis=1) - 1.0))) < 1e-10,
        "arms_pipeline_conserves": trajectory["diagnostics"]["arms_pipeline_stock_residual"] < 1e-10,
        "mfg_converges": base_diag["continuous_mfg_fixed_point"]["nonconverged_months"] == 0,
        "macro_converges": base_diag["cge_dsge_fixed_point"]["nonconverged_months"] == 0,
        "operations_conserves": base_diag["operational_constraints"]["max_stock_accounting_residual"] < 1e-7,
        "pnt_inverts": base_diag["dynamic_geospatial_pnt"]["information_inverse_residual"] < 1e-7,
        "rolling_game_converges": base_diag["rolling_differential_game"]["nonconverged_months"] == 0,
        "spatial_bounds": base_diag["spatial_control_network"]["population_weight_residual"] < 1e-12,
        "deds_regime_simplex": abs(sum(base_diag["conflict_regime_shares"].values()) - 1.0) < 1e-10,
        "tail_importance_ess": base_diag["importance_sampling_effective_sample_size"] > 0.35 * PATHS,
    }
    evidence_map = {
        "historical_identification_causal_hmm": "causal_effect_nonzero",
        "prewar_game_evolution_feedback": "prewar_state_moves",
        "continuous_graphon_mfg": "mfg_converges",
        "age_stratified_social_contagion": "without_social_mfg",
        "unbalanced_optimal_transport": "without_optimal_transport",
        "leader_gated_jump_process": "prewar_state_moves",
        "stratified_competing_survival": "survival_simplex",
        "growth_defense_capital": "prewar_state_moves",
        "opinion_election_arms_pipeline": "arms_pipeline_conserves",
        "endogenous_alliance_stackelberg": "alliance_simplex",
        "campaign_deds_multidomain": "deds_regime_simplex",
        "operations_network_flow_metric_search": "operations_conserves",
        "geospatial_pnt": "pnt_inverts",
        "rolling_pomdp_robust_game": "rolling_game_converges",
        "spatial_population_control": "spatial_bounds",
        "labor_firms_supply_collapse": "without_macro_finance",
        "finance_fiscal_money_tax_debt_assets": "without_macro_finance",
        "dynamic_mr_cge_mrio_gravity_chips": "macro_converges",
        "open_dsge_hank_mundell_fleming": "macro_converges",
        "actual_bilateral_aid_clearing": "without_macro_finance",
        "rare_tail_hawkes_evt_importance": "tail_importance_ess",
        "principal_agent_governance": "without_spatial_governance",
        "pathwise_terminal_aggregation": "survival_simplex",
    }
    group_pass = {row["ablation"]: row["pass"] for row in group_rows}
    module_rows = []
    for module in active:
        evidence = evidence_map[module]
        passed = structural.get(evidence, group_pass.get(evidence, False))
        module_rows.append({"module_id": module, "dynamic_evidence": evidence, "pass": bool(passed)})
    errors = [row["module_id"] for row in module_rows if not row["pass"]]
    errors += [row["ablation"] for row in group_rows if not row["pass"]]
    payload = {
        "status": "PASS" if not errors else "FAIL",
        "purpose": "dynamic validation only; no outcome probabilities",
        "active_module_count": len(active),
        "passed_module_count": sum(row["pass"] for row in module_rows),
        "structural_checks": structural,
        "grouped_same_seed_ablations": group_rows,
        "module_evidence": module_rows,
        "errors": errors,
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if errors:
        raise RuntimeError("dynamic ablation validation failed: " + ", ".join(errors))
    print(f"TAIWAN_V47_DYNAMIC_ABLATION: PASS; modules={len(active)}; groups={len(group_rows)}")


if __name__ == "__main__":
    main()
