"""DIAGNOSTIC_ONLY: isolated shape, conservation, and finite-residual checks."""

from __future__ import annotations

import numpy as np

from continuous_multipop_mfg_v47 import ContinuousMultiPopulationMFG
from competing_risk_survival_v47 import StratifiedCompetingRiskSurvival
from closed_prewar_system_v47 import ClosedPrewarEvolutionSystem
from dynamic_macro_equilibrium_v47 import CoupledMacroEconomicSystem
from endogenous_intervention_v47 import EndogenousInterventionChoiceModel
from operational_constraints_v47 import DynamicGeospatialPNTSystem, DynamicOperationalConstraintSystem
from rolling_differential_game_v47 import RollingHorizonDifferentialGame
from spatial_control_network_v47 import SpatialControlNetwork
import simulate_taiwan_multidomain_v42 as battle
from simulate_four_party_financial_v45 import FourPartyFinancialSystem
from wartime_social_contagion import WartimeSocialContagionSystem
from dynamic_mechanism_design_v48 import RobustDynamicMechanismDesigner
from four_party_energy_network_v49 import FourPartyEnergyNetwork


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    paths = 2
    support = np.full((paths, 4), 0.55)
    zero = np.zeros((paths, 4))
    one = np.ones((paths, 4))
    terrain = {
        "land_share_above_500m": 0.42,
        "sky_view_proxy_p10_p50_p90": [0.45, 0.68, 0.88],
    }
    prewar = ClosedPrewarEvolutionSystem(paths, 101, {
        "actor_support": support,
        "taiwan_readiness": np.ones(paths),
    }).run(months=2)
    require(prewar.diagnostics["pressure_simplex_residual"] < 1e-12, "prewar HMM simplex")
    require(prewar.diagnostics["strategy_simplex_residual"] < 1e-12, "prewar strategy simplex")
    require(prewar.diagnostics["arms_pipeline_stock_residual"] < 1e-10, "arms pipeline stock conservation")

    mfg = ContinuousMultiPopulationMFG(paths, 1, support)
    mfg_result = mfg.update(
        progress=zero, damage=zero, shortage=zero, inflation=zero,
        financial_stress=zero, command=one, network=one,
        contagion_mobilization=zero, contagion_fatigue=zero,
    )
    require(mfg_result.support.shape == (paths, 4), "MFG support shape")
    require(mfg_result.mass_residual < 1e-9, "MFG mass conservation")
    require(np.isfinite(mfg_result.fixed_point_residual), "MFG finite residual")
    require(mfg_result.density_residual < 5e-4, "MFG density fixed point")
    require(mfg_result.bellman_residual < 2e-5, "MFG Bellman residual")
    require(mfg_result.max_policy_iterations_used <= 18, "MFG policy iteration cap")

    social = WartimeSocialContagionSystem(paths, 2, support)
    social_result = social.update(
        prior_support=support, actor_damage=zero, actor_shortage=zero,
        actor_inflation=zero, actor_backlash=zero, actor_network=one,
        actor_norm=support, actor_resilience=one, progress_signal=zero,
        command_integrity=one, conflict_intensity=np.zeros(paths),
        mfg_threshold_shift=mfg_result.threshold_shift,
        mfg_mobilization=mfg_result.mobilization_preference,
        mfg_fatigue=mfg_result.fatigue_preference,
    )
    require(social_result["conservation_error"] < 1e-9, "social conservation")
    require(social.random_batches_generated == 1, "social monthly random batching")

    ops = DynamicOperationalConstraintSystem(paths, 3, terrain)
    ops_result = ops.update(
        production=zero, consumption=zero, damage=zero,
        interdiction=zero, command=one, pnt=one, financial_capacity=one,
    )
    require(ops_result.front_inventory.shape == (paths, 4, 3), "operations inventory shape")
    require(ops_result.stock_residual < 1e-8, "operations stock conservation")
    require(ops_result.flow_residual < 1e-8, "operations flow capacity")

    pnt = DynamicGeospatialPNTSystem(paths, terrain)
    pnt_result = pnt.update(
        infrastructure=one, cyber_damage=zero, command=one, terrain_mask=0.68
    )
    require(pnt_result.integrity.shape == (paths, 4), "PNT integrity shape")
    require(pnt_result.information_residual < 1e-8, "PNT information inverse")

    spatial = SpatialControlNetwork(paths, 31, terrain)
    spatial_result = spatial.update(
        aggregate_land=np.zeros(paths), attacker_logistics=np.ones(paths),
        attacker_command=np.ones(paths), defender_command=np.ones(paths),
        defender_resistance=np.ones(paths), external_intervention=np.ones(paths),
        local_agent_capacity=np.zeros(paths), damage=np.zeros(paths),
    )
    require(np.all((spatial_result.zone_control >= 0.0) & (spatial_result.zone_control <= 1.0)), "spatial control bounds")
    require(spatial.diagnostics()["population_weight_residual"] < 1e-12, "population weights")

    finance = FourPartyFinancialSystem(paths, 2, 4, "limited", "baseline")
    firm_distress = np.zeros((paths, 4, 4))
    sector_output = np.ones((paths, 4, 4))
    finance.update(
        month=0, actor_damage=zero, actor_shortage=zero,
        actor_mobilization=zero, firm_distress=firm_distress,
        sector_output=sector_output, trade_gdp_factor=one,
        trade_industry_factor=one, trade_inflation=zero,
        trade_loss=np.zeros(paths), macro_trade_balance=zero,
        macro_government_balance=zero, us_level=np.ones(paths) * 0.5,
        japan_level=np.ones(paths) * 0.4,
    )
    finance.register_realized_aid(
        np.ones(paths) * 0.002, np.ones(paths) * 0.001, np.ones(paths) * 0.0005
    )
    require(finance.max_bilateral_aid_residual < 1e-12, "bilateral aid clearing")

    choice_model = EndogenousInterventionChoiceModel(paths, 5)
    survival_model = StratifiedCompetingRiskSurvival(paths, 4)
    survival = survival_model.evaluate(
        actor_support=support, taiwan_readiness=np.ones(paths),
        leader_continuity=one, fiscal_capacity=one,
        financial_stress=zero, arms_support_us=np.ones(paths) * 0.62,
        arms_support_japan=np.ones(paths) * 0.55,
        target_force_probability=0.12, months=60,
        pressure_belief=prewar.pressure_belief,
    )
    require(survival.simplex_residual < 1e-10, "competing-risk simplex")
    require(survival.calibration_residual < 1e-7, "Cox force-risk calibration")
    require(np.min(survival.fine_gray_hazard) >= 0.0, "Fine-Gray nonnegative hazard")
    choice = choice_model.probabilities(
        actor_support=support, taiwan_readiness=np.ones(paths),
        leader_continuity=one, arms_support_us=np.ones(paths) * 0.62,
        arms_support_japan=np.ones(paths) * 0.55,
        fiscal_capacity=one, financial_stress=zero,
        aft_delay_months=survival.aft_delay_months,
    )
    require(np.max(np.abs(choice.probabilities.sum(axis=1) - 1.0)) < 1e-12, "intervention simplex")
    require(np.min(choice.probabilities) >= 0.0, "intervention nonnegative")

    strategy = RollingHorizonDifferentialGame(paths)
    strategy_result = strategy.update(
        progress=zero, stocks=one, damage=zero, economic_capacity=one,
        support=support, command=one, intervention=one,
    )
    require(
        np.max(np.abs(strategy_result.controls.sum(axis=2) - 1.0)) < 1e-12,
        "strategy control simplex",
    )
    require(strategy_result.converged, "rolling game best-response convergence")

    macro = CoupledMacroEconomicSystem(paths, 6, "baseline")
    macro_result = macro.update(
        month=0, actor_damage=zero, actor_logistics=one,
        actor_labor=one, actor_financial_stress=zero, actor_payment=one,
        actor_fiscal=one, actor_social_action=zero, actor_panic=zero,
        actor_mobilization=zero, blockade=np.zeros(paths),
    )
    require(macro_result.gdp_factor.shape == (paths, 4), "macro factor shape")
    require(np.all(np.isfinite(macro_result.gdp_factor)), "macro finite state")
    require(np.isfinite(macro_result.outer_residual), "macro finite outer residual")
    require(macro_result.outer_iterations <= 18, "macro outer iteration cap")
    require(
        macro_result.outer_converged_path_fraction == 1.0,
        "macro pathwise convergence",
    )
    require(
        macro_result.external_account_residual < 1e-10,
        "CGE external account clearing",
    )
    require(macro_result.converged, "CGE-DSGE fixed point convergence")

    # One-month all-module path test. It is deliberately not an outcome run:
    # the horizon is one transition and no probability estimate is reported.
    battle.PATHS = paths
    battle.MONTHS = 1
    battle.HORIZON_MONTHS = (1,)
    _, _, loaded_terrain, sipri = battle.load_inputs()
    path_finance = FourPartyFinancialSystem(paths, 1, 201, "limited", "baseline")
    path_social = WartimeSocialContagionSystem(paths, 202, prewar.actor_support)
    path_mfg = ContinuousMultiPopulationMFG(paths, 203, prewar.actor_support)
    path_macro = CoupledMacroEconomicSystem(paths, 204, "baseline")
    path_ops = DynamicOperationalConstraintSystem(paths, 205, loaded_terrain)
    path_pnt = DynamicGeospatialPNTSystem(paths, loaded_terrain)
    path_strategy = RollingHorizonDifferentialGame(paths, 206)
    path_spatial = SpatialControlNetwork(paths, 207, loaded_terrain)
    path_mechanism = RobustDynamicMechanismDesigner(paths, 209)
    path_energy = FourPartyEnergyNetwork(paths, 210)
    path_prewar = {
        "actor_support": prewar.actor_support,
        "taiwan_readiness": prewar.taiwan_readiness,
        "leader_continuity": prewar.leader_continuity,
        "economic_capital": prewar.economic_capital,
        "military_capital": prewar.military_capital,
    }
    path_snapshots, _ = battle.simulate_mechanisms(
        np.random.default_rng(208), 2030, "limited", "central",
        loaded_terrain, sipri, financial_system=path_finance,
        prewar_state=path_prewar, social_system=path_social,
        macro_system=path_macro, mfg_system=path_mfg,
        operations_system=path_ops, pnt_system=path_pnt,
        strategy_system=path_strategy, spatial_system=path_spatial,
        mechanism_system=path_mechanism,
        energy_system=path_energy,
    )
    require(1 in path_snapshots, "all-module path snapshot")
    path_diag = path_snapshots[1]["diagnostics"]
    for key in (
        "continuous_mfg_fixed_point", "cge_dsge_fixed_point",
        "operational_constraints", "dynamic_geospatial_pnt",
        "rolling_differential_game", "spatial_control_network",
    ):
        require(path_diag[key] is not None, f"all-module path missing {key}")
    mechanism_diag = path_mechanism.diagnostics()
    require(mechanism_diag["nonconverged_months"] == 0, "mechanism convergence")
    require(mechanism_diag["max_incentive_compatibility_residual"] < 1e-9, "mechanism IC")
    require(mechanism_diag["max_participation_residual"] < 1e-9, "mechanism IR")
    require(mechanism_diag["max_budget_residual"] < 1e-9, "mechanism budget")
    energy_diag = path_energy.diagnostics()
    require(energy_diag["max_flow_min_cut_residual"] < 1e-9, "energy max-flow/min-cut")
    require(energy_diag["max_delivery_above_demand_residual"] < 1e-10, "energy demand feasibility")
    print("DIAGNOSTIC_ONLY_TAIWAN_V47: PASS")


if __name__ == "__main__":
    main()
