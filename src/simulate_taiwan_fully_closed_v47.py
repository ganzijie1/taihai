from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
from datetime import datetime, timezone

import numpy as np

import simulate_dual_circulation_v44 as trade
import simulate_taiwan_full_coupled_v45 as prewar_model
import simulate_taiwan_causal_evolution_v37 as causal_model
import simulate_taiwan_multidomain_v42 as battle
from continuous_multipop_mfg_v47 import ContinuousMultiPopulationMFG
from competing_risk_survival_v47 import StratifiedCompetingRiskSurvival
from closed_prewar_system_v47 import ClosedPrewarEvolutionSystem
from dynamic_macro_equilibrium_v47 import CoupledMacroEconomicSystem
from endogenous_intervention_v47 import EndogenousInterventionChoiceModel
from operational_constraints_v47 import (
    DynamicGeospatialPNTSystem,
    DynamicOperationalConstraintSystem,
)
from rolling_differential_game_v47 import RollingHorizonDifferentialGame
from spatial_control_network_v47 import SpatialControlNetwork
from simulate_four_party_financial_v45 import FourPartyFinancialSystem, PUBLIC_CALIBRATION
from wartime_social_contagion import WartimeSocialContagionSystem
from dynamic_mechanism_design_v48 import RobustDynamicMechanismDesigner
from four_party_energy_network_v49 import FourPartyEnergyNetwork
from geometric_multiscale_v50 import GeometricMultiscaleCoupler
from hybrid_filippov_impulse_v51 import HybridFilippovImpulseSystem
from topological_dynamics_v51 import TopologicalRegimeDynamics


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
TRACEABILITY = OUT / "taiwan_v47_traceability.json"
COUPLING_AUDIT = OUT / "taiwan_v47_static_coupling_audit.json"
OUT_CSV = OUT / "taiwan_v47_fully_closed_results.csv"
OUT_JSON = OUT / "taiwan_v47_fully_closed_simulation.json"
LONG_TERM_START_YEAR = 2026
LONG_TERM_END_YEAR = 2100
LONG_TERM_MONTHS = (LONG_TERM_END_YEAR - LONG_TERM_START_YEAR + 1) * 12


def _hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sum(values * weights) / np.maximum(np.sum(weights), 1e-12))


def weighted_quantile(
    values: np.ndarray, weights: np.ndarray, probabilities=(0.1, 0.5, 0.9)
) -> list[float]:
    order = np.argsort(values)
    ordered_values = values[order]
    ordered_weights = weights[order]
    cumulative = (np.cumsum(ordered_weights) - 0.5 * ordered_weights) / np.maximum(
        np.sum(ordered_weights), 1e-12
    )
    return [float(np.interp(p, cumulative, ordered_values)) for p in probabilities]


def assert_complete_gate(
    traceability: Path = TRACEABILITY, coupling_audit: Path = COUPLING_AUDIT
) -> None:
    missing = [str(p) for p in (traceability, coupling_audit) if not p.exists()]
    if missing:
        raise RuntimeError(f"complete-model gate blocked; missing audits: {missing}")
    trace = json.loads(traceability.read_text(encoding="utf-8"))
    audit = json.loads(coupling_audit.read_text(encoding="utf-8"))
    failures = []
    if trace.get("coverage_percent") != 100.0 or trace.get("gate_status") != "PASS":
        failures.append("document-to-code traceability is not 100% PASS")
    if audit.get("status") != "PASS" or audit.get("blocking_gaps"):
        failures.append("static coupling audit has blocking gaps")
    if failures:
        raise RuntimeError("complete-model gate blocked: " + "; ".join(failures))


def _initial_financial_choice_state(paths: int, continuity: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    anchors = [PUBLIC_CALIBRATION[a] for a in ("china", "taiwan", "united_states", "japan")]
    debt = np.array([a["debt_to_gdp"] + a["local_debt_to_gdp"] for a in anchors])[None, :]
    npl = np.array([a["npl"] for a in anchors])[None, :]
    reserve = np.array([a["reserve_buffer"] for a in anchors])[None, :]
    access = np.array([[0.55, 0.74, 0.98, 0.90]])
    fiscal = np.clip(
        access * (0.72 + 0.28 * continuity)
        * (1.0 - 0.10 * np.minimum(debt, 3.0))
        + 0.08 * np.minimum(reserve, 1.0),
        0.08, 1.0,
    )
    stress = np.clip(
        4.0 * npl + 0.13 * np.maximum(debt - 0.8, 0.0)
        + 0.20 * (1.0 - continuity) - 0.05 * np.minimum(reserve, 1.0),
        0.0, 1.0,
    )
    return np.repeat(fiscal, paths, axis=0) if fiscal.shape[0] == 1 else fiscal, np.repeat(stress, paths, axis=0) if stress.shape[0] == 1 else stress


def endogenous_force_onset(prewar: dict, mean_probability: float, seed: int) -> np.ndarray:
    """Same-path cause-specific force-onset probability calibrated to V3.7."""
    rng = np.random.default_rng(seed)
    support = prewar["actor_support"]
    continuity = prewar["leader_continuity"]
    readiness = prewar["taiwan_readiness"]
    mean_probability = float(np.clip(mean_probability, 1e-5, 1.0 - 1e-5))
    baseline = math.log(mean_probability / (1.0 - mean_probability))
    latent = (
        baseline + 0.72 * (support[:, 0] - np.mean(support[:, 0]))
        - 0.28 * (support[:, 1] - np.mean(support[:, 1]))
        - 0.18 * (support[:, 2] - np.mean(support[:, 2]))
        + 0.34 * (1.0 - continuity[:, 0])
        - 0.22 * np.log(np.maximum(readiness, 0.35))
        + rng.normal(0.0, 0.18, len(readiness))
    )
    raw = 1.0 / (1.0 + np.exp(-latent))
    # Preserve the identified marginal hazard while retaining path dependence.
    shift = baseline - math.log(np.mean(raw) / max(1.0 - np.mean(raw), 1e-12))
    return 1.0 / (1.0 + np.exp(-(latent + shift)))


def run(
    paths: int = 600, months: int = 360,
    scenario_names: list[str] | None = None, prewar_scenario: str = "central",
    case_names: list[str] | None = None,
    mechanism_enabled: bool = False, mechanism_flags: dict | None = None,
    energy_enabled: bool = False,
    geometry_enabled: bool = False,
    hybrid_enabled: bool = False,
    hybrid_config: dict | None = None,
    topology_enabled: bool = False,
    mfg_config: dict | None = None,
    macro_factory=None,
    traceability_path: Path = TRACEABILITY,
    coupling_audit_path: Path = COUPLING_AUDIT,
):
    assert_complete_gate(traceability_path, coupling_audit_path)
    battle.PATHS = paths
    battle.MONTHS = months
    battle.HORIZON_MONTHS = tuple(y * 12 for y in battle.HORIZON_YEARS if y * 12 <= months)
    if not battle.HORIZON_MONTHS:
        raise ValueError("campaign horizon must include at least the five-year checkpoint")
    _, _, terrain, sipri = battle.load_inputs()
    all_scenarios = list(trade.v43.SCENARIOS)
    scenarios = all_scenarios if scenario_names is None else scenario_names
    cases = list(battle.CASES) if case_names is None else case_names
    if not cases or any(case not in battle.CASES for case in cases):
        raise ValueError("case_names must be a nonempty subset of configured cases")
    observed_prewar = prewar_model.prewar_state(paths, battle.SEED + 810_001)
    causal_estimate = causal_model.causal_local_effect(battle.SEED + 811_009)
    prewar_system = ClosedPrewarEvolutionSystem(
        paths, battle.SEED + 812_003, observed_prewar,
        causal_pressure_effect=causal_estimate["estimate_log_points"],
        scenario=prewar_scenario,
    )
    prewar_trajectory = prewar_system.run_trajectory(months=LONG_TERM_MONTHS)
    _, force_row = prewar_model.force_onset_paths(paths, battle.SEED + 930_007)
    survival_model = StratifiedCompetingRiskSurvival(paths, battle.SEED + 915_001)
    survival = survival_model.evaluate_trajectory(
        actor_support=prewar_trajectory["actor_support"],
        taiwan_readiness=prewar_trajectory["taiwan_readiness"],
        leader_continuity=prewar_trajectory["leader_continuity"],
        pressure_belief=prewar_trajectory["pressure_belief"],
        economic_capital=prewar_trajectory["economic_capital"],
        interdependence=prewar_trajectory["interdependence"],
        target_force_probability=force_row["force_onset"],
        calibration_months=60,
    )
    force_paths = survival.cumulative_incidence[:, 0]
    force_increment = survival.cause_increment[:, :, 0]
    conditional_force_time = force_increment / np.maximum(force_paths[:, None], 1e-12)
    conditional_force_time /= np.maximum(
        conditional_force_time.sum(axis=1, keepdims=True), 1e-12
    )
    onset_rng = np.random.default_rng(battle.SEED + 916_003)
    onset_draw = onset_rng.random(paths)
    onset_month = np.sum(
        np.cumsum(conditional_force_time, axis=1) < onset_draw[:, None], axis=1
    )
    onset_month = np.clip(onset_month, 0, LONG_TERM_MONTHS - 1)
    path_index = np.arange(paths)
    prewar = {
        key: prewar_trajectory[key][onset_month, path_index]
        for key in (
            "actor_support", "taiwan_readiness", "arms_support_us",
            "arms_support_japan", "leader_continuity", "pressure_belief",
            "military_capital", "economic_capital",
        )
    }
    prewar["metadata"] = {
        "observed_anchor": observed_prewar["metadata"],
        "closed_prewar_diagnostics": prewar_trajectory["diagnostics"],
        "causal_local_effect": causal_estimate,
        "prewar_scenario": prewar_scenario,
        "conditional_force_onset_year_p10_p50_p90": [
            float(x) for x in np.quantile(
                LONG_TERM_START_YEAR + onset_month / 12.0, (0.1, 0.5, 0.9)
            )
        ],
    }
    event_year = LONG_TERM_START_YEAR + onset_month // 12
    remaining_to_2100 = LONG_TERM_MONTHS - onset_month
    prewar_battle = {k: v for k, v in prewar.items() if k != "metadata"}
    initial_fiscal, initial_stress = _initial_financial_choice_state(
        paths, prewar["leader_continuity"]
    )
    aft_delay = survival_model.aft_delay_at_onset(
        actor_support=prewar["actor_support"],
        taiwan_readiness=prewar["taiwan_readiness"],
        leader_continuity=prewar["leader_continuity"],
        arms_support_us=prewar["arms_support_us"],
        arms_support_japan=prewar["arms_support_japan"],
        fiscal_capacity=initial_fiscal,
        financial_stress=initial_stress,
    )
    survival.aft_delay_months = aft_delay
    choice_model = EndogenousInterventionChoiceModel(paths, battle.SEED + 920_003)
    choice = choice_model.probabilities(
        actor_support=prewar["actor_support"],
        taiwan_readiness=prewar["taiwan_readiness"],
        leader_continuity=prewar["leader_continuity"],
        arms_support_us=prewar["arms_support_us"],
        arms_support_japan=prewar["arms_support_japan"],
        fiscal_capacity=initial_fiscal,
        financial_stress=initial_stress,
        aft_delay_months=aft_delay,
    )

    conditional: dict[str, dict[str, dict[int, np.ndarray]]] = {}
    importance_weights: dict[str, np.ndarray] = {}
    rows = []
    diagnostics = {}
    common_seed = battle.SEED + LONG_TERM_END_YEAR * 101
    conditional_by_2100: dict[str, dict[str, np.ndarray]] = {}
    for case_index, case in enumerate(cases):
        conditional[case] = {}
        conditional_by_2100[case] = {}
        for name in scenarios:
            scenario_index = all_scenarios.index(name)
            innovation_seed = common_seed + scenario_index * 100_003
            finance = FourPartyFinancialSystem(
                paths, months, innovation_seed + 11_003, case, trade.v43.SCENARIOS[name]
            )
            social = WartimeSocialContagionSystem(
                paths, innovation_seed + 23_009, prewar["actor_support"]
            )
            mfg = ContinuousMultiPopulationMFG(
                paths, innovation_seed + 37_013, prewar["actor_support"],
                geometry_enabled=geometry_enabled,
                **(mfg_config or {}),
            )
            macro_class = macro_factory or CoupledMacroEconomicSystem
            macro = macro_class(
                paths, innovation_seed + 41_021, trade.v43.SCENARIOS[name]
            )
            operations = DynamicOperationalConstraintSystem(
                paths, innovation_seed + 47_027, terrain,
                geometry_enabled=geometry_enabled,
            )
            pnt = DynamicGeospatialPNTSystem(
                paths, terrain, geometry_enabled=geometry_enabled
            )
            strategy = RollingHorizonDifferentialGame(paths, innovation_seed + 53_033)
            spatial = SpatialControlNetwork(
                paths, innovation_seed + 59_041, terrain,
                geometry_enabled=geometry_enabled,
            )
            mechanism = (
                RobustDynamicMechanismDesigner(
                    paths, innovation_seed + 61_043, module_flags=mechanism_flags
                )
                if mechanism_enabled else None
            )
            energy = (
                FourPartyEnergyNetwork(
                    paths, innovation_seed + 67_049,
                    geometry_enabled=geometry_enabled,
                )
                if energy_enabled else None
            )
            geometry = GeometricMultiscaleCoupler(paths) if geometry_enabled else None
            hybrid = (
                HybridFilippovImpulseSystem(paths, **(hybrid_config or {}))
                if hybrid_enabled else None
            )
            topology = TopologicalRegimeDynamics(paths) if topology_enabled else None
            snapshots, _ = battle.simulate_mechanisms(
                np.random.default_rng(innovation_seed), event_year, case, "central", terrain, sipri,
                financial_system=finance, prewar_state=prewar_battle,
                social_system=social, macro_system=macro, mfg_system=mfg,
                operations_system=operations, pnt_system=pnt,
                strategy_system=strategy, spatial_system=spatial,
                mechanism_system=mechanism,
                energy_system=energy,
                geometry_system=geometry,
                hybrid_system=hybrid,
                topology_system=topology,
            )
            key = f"{case}|{name}"
            diagnostics[key] = {
                "finance": finance.summary(), "social": social.summary(),
                "mfg": mfg.diagnostics(), "macro": macro.diagnostics(),
                "operations": operations.diagnostics(), "pnt": pnt.diagnostics(),
                "strategy": strategy.diagnostics(),
                "spatial": spatial.diagnostics(),
                "mechanism": mechanism.diagnostics() if mechanism is not None else None,
                "energy": energy.diagnostics() if energy is not None else None,
                "geometry": geometry.diagnostics() if geometry is not None else None,
                "hybrid": hybrid.diagnostics() if hybrid is not None else None,
                "topology": topology.diagnostics() if topology is not None else None,
                "finance_accounting": {
                    "max_bilateral_aid_residual": finance.max_bilateral_aid_residual,
                },
                "campaign_horizons": {},
            }
            conditional[case][name] = {}
            for horizon in battle.HORIZON_MONTHS:
                achieved = snapshots[horizon]["broad_control_paths"].astype(float)
                weights = snapshots[horizon]["importance_weight_paths"]
                kernel_score = snapshots[horizon]["mfg_kernel_score_paths"]
                kernel_low, kernel_high = np.quantile(kernel_score, [1.0 / 3.0, 2.0 / 3.0])
                kernel_groups = (
                    kernel_score <= kernel_low,
                    (kernel_score > kernel_low) & (kernel_score <= kernel_high),
                    kernel_score > kernel_high,
                )
                kernel_rates = [
                    weighted_mean(achieved[mask], weights[mask])
                    for mask in kernel_groups
                ]
                horizon_diag = snapshots[horizon]["diagnostics"]
                diagnostics[key]["campaign_horizons"][str(horizon)] = {
                    **horizon_diag,
                    "importance_sampling_effective_sample_size": horizon_diag[
                        "importance_sampling_effective_sample_size"
                    ],
                    "absorbing_broad_control_share_unweighted": horizon_diag[
                        "absorbing_broad_control_share"
                    ],
                    "mechanical_majority_control_share_unweighted": horizon_diag[
                        "mechanical_majority_control_share"
                    ],
                }
                if name in importance_weights:
                    if not np.allclose(importance_weights[name], weights):
                        raise RuntimeError("common-random-number tail weights diverged across cases")
                else:
                    importance_weights[name] = weights.copy()
                conditional[case][name][horizon] = achieved
                rows.append({
                    "conflict_year": "pathwise_2026_2100",
                    "horizon_years": horizon // 12,
                    "case": case,
                    "trade_scenario": trade.v43.SCENARIOS[name],
                    "conditional_broad_control_rate": weighted_mean(achieved, weights),
                    "kernel_uncertainty_low_rate": kernel_rates[0],
                    "kernel_uncertainty_mid_rate": kernel_rates[1],
                    "kernel_uncertainty_high_rate": kernel_rates[2],
                    "importance_sampling_effective_sample_size": float(
                        np.sum(weights) ** 2 / np.sum(weights ** 2)
                    ),
                    "mechanical_majority_control_share": weighted_mean(
                        (
                            (snapshots[horizon]["population_control_paths"] >= 0.50)
                            & (snapshots[horizon]["administrative_control_paths"] >= 0.50)
                            & (snapshots[horizon]["organized_defense_paths"] <= 0.35)
                        ).astype(float), weights
                    ),
                    "status_quo_settlement_share": weighted_mean(
                        (snapshots[horizon]["settlement_outcome_paths"] == 1).astype(float), weights
                    ),
                    "unification_settlement_share": weighted_mean(
                        (snapshots[horizon]["settlement_outcome_paths"] == 2).astype(float), weights
                    ),
                })
            terminal = snapshots[max(battle.HORIZON_MONTHS)]
            broad_month = terminal["broad_control_month_paths"]
            conditional_by_2100[case][name] = (
                terminal["broad_control_paths"]
                & (broad_month > 0)
                & (broad_month <= remaining_to_2100)
            ).astype(float)
            if os.environ.get("TAIWAN_PROGRESS_LOG") == "1":
                safe_case = case.encode("unicode_escape").decode("ascii")
                safe_name = name.encode("unicode_escape").decode("ascii")
                print(
                    f"TAIWAN_CELL_COMPLETE case={safe_case} scenario={safe_name} "
                    f"utc={datetime.now(timezone.utc).isoformat()}",
                    flush=True,
                )

    aggregate = []
    calendar_2100 = []
    case_indices = [battle.CASES.index(case) for case in cases]
    selected_choice_probability = choice.probabilities[:, case_indices]
    selected_choice_probability /= np.maximum(
        selected_choice_probability.sum(axis=1, keepdims=True), 1e-12
    )
    for name in scenarios:
        for horizon in battle.HORIZON_MONTHS:
            matrix = np.column_stack([
                conditional[case][name][horizon] for case in cases
            ])
            weighted = np.sum(selected_choice_probability * matrix, axis=1)
            unconditional = force_paths * weighted
            tail_weights = importance_weights[name]
            aggregate.append({
                "trade_scenario": trade.v43.SCENARIOS[name],
                "horizon_years": horizon // 12,
                "conditional_mean": weighted_mean(weighted, tail_weights),
                "conditional_p10_p50_p90": weighted_quantile(weighted, tail_weights),
                "unconditional_mean": weighted_mean(unconditional, tail_weights),
                "unconditional_p10_p50_p90": weighted_quantile(unconditional, tail_weights),
            })
        by_2100_matrix = np.column_stack([
            conditional_by_2100[case][name] for case in cases
        ])
        by_2100_conditional = np.sum(
            selected_choice_probability * by_2100_matrix, axis=1
        )
        by_2100_joint = force_paths * by_2100_conditional
        weights = importance_weights[name]
        calendar_2100.append({
            "trade_scenario": trade.v43.SCENARIOS[name],
            "force_onset_cif": weighted_mean(force_paths, weights),
            "peace_arrangement_cif": weighted_mean(survival.cumulative_incidence[:, 1], weights),
            "legal_separation_cif": weighted_mean(survival.cumulative_incidence[:, 2], weights),
            "status_quo_survival": weighted_mean(survival.survival, weights),
            "conditional_broad_control_by_2100": weighted_mean(by_2100_conditional, weights),
            "joint_force_and_broad_control_by_2100": weighted_mean(by_2100_joint, weights),
            "force_without_broad_control_by_2100": weighted_mean(
                force_paths * (1.0 - by_2100_conditional), weights
            ),
            "joint_p10_p50_p90": weighted_quantile(by_2100_joint, weights),
        })
    errors = []
    if survival.simplex_residual > 1e-10:
        errors.append("long-term competing-risk simplex residual")
    if survival.calibration_residual > 1e-8:
        errors.append("2030 force-risk calibration residual")
    if np.any(np.diff(survival.cumulative_incidence_history, axis=1) < -1e-12):
        errors.append("nonmonotone long-term cumulative incidence")
    if prewar_trajectory["diagnostics"]["arms_pipeline_stock_residual"] > 1e-8:
        errors.append("prewar arms pipeline stock residual")
    hybrid_switch_totals = {
        surface: 0 for surface in ("logistics", "financial", "mobilization", "command")
    }
    hybrid_nonsaturated_cells = {surface: 0 for surface in hybrid_switch_totals}
    topology_active_cells = 0
    topology_visited_symbols: set[int] = set()
    topology_max_feedback_deviation = 0.0
    for cell, diag in diagnostics.items():
        if diag["macro"]["nonconverged_months"]:
            errors.append(
                f"{cell}: macro nonconverged months="
                f"{diag['macro']['nonconverged_months']}, outer="
                f"{diag['macro']['max_outer_residual']:.3g}, cge_goods="
                f"{diag['macro']['last_cge_goods_residual']:.3g}, cge_factor="
                f"{diag['macro']['last_cge_factor_residual']:.3g}, dsge="
                f"{diag['macro']['last_dsge_residual']:.3g}, component_counts="
                f"{diag['macro']['outer_nonconverged_months']}/"
                f"{diag['macro']['cge_nonconverged_months']}/"
                f"{diag['macro']['dsge_nonconverged_months']}, component_max="
                f"{diag['macro']['max_cge_solver_residual']:.3g}/"
                f"{diag['macro']['max_dsge_solver_residual']:.3g}"
            )
        if diag["mfg"]["nonconverged_months"]:
            errors.append(
                f"{cell}: MFG nonconverged months="
                f"{diag['mfg']['nonconverged_months']}, last="
                f"{diag['mfg']['last_fixed_point_residual']:.3g}, max="
                f"{diag['mfg']['max_fixed_point_residual']:.3g}"
            )
        if diag["mfg"].get("advanced_graphon"):
            mfg_diag = diag["mfg"]
            if mfg_diag["max_major_fixed_point_residual"] > 5.0e-4:
                errors.append(f"{cell}: major-player fixed-point residual")
            if mfg_diag["max_mean_exploitability_bound"] > 1.0e-2:
                errors.append(f"{cell}: mean exploitability bound")
            if mfg_diag["max_worst_type_exploitability_bound"] > 5.0e-2:
                errors.append(f"{cell}: worst-type exploitability bound")
            if mfg_diag["max_finite_network_field_error"] > 2.5e-1:
                errors.append(f"{cell}: finite-network field error")
            if mfg_diag["max_finite_network_payoff_deviation"] > 2.0e-1:
                errors.append(f"{cell}: finite-network payoff deviation")
            if min(mfg_diag["layer_activity"]) <= 0.0:
                errors.append(f"{cell}: inactive Graphon layer")
            if not (
                0.05 <= mfg_diag["minimum_engagement_mass"]
                <= mfg_diag["maximum_engagement_mass"] <= 2.50
            ):
                errors.append(f"{cell}: engagement mass bounds")
            fp_diag = mfg_diag["fp_solver"]
            if fp_diag["backend"] != "torch_fvm":
                errors.append(f"{cell}: production FP backend is not torch_fvm")
            if fp_diag["max_mass_residual_before_normalisation"] > 1.0e-10:
                errors.append(f"{cell}: FP pre-normalisation mass residual")
            if fp_diag["minimum_density_before_projection"] < -1.0e-10:
                errors.append(f"{cell}: FP positivity violation")
            geometry_diag = mfg_diag["information_geometry"]
            if geometry_diag["geomloss_calls"] <= 0:
                errors.append(f"{cell}: GeomLoss was not exercised")
            if geometry_diag["geomloss_failures"]:
                errors.append(f"{cell}: GeomLoss failure")
        if diag["strategy"]["nonconverged_months"]:
            errors.append(
                f"{cell}: strategy nonconverged months="
                f"{diag['strategy']['nonconverged_months']}, max="
                f"{diag['strategy']['max_nonconverged_residual']:.3g}, last="
                f"{diag['strategy']['last_best_response_residual']:.3g}"
            )
        if diag["macro"]["max_cge_external_account_residual"] > 1e-9:
            errors.append(f"{cell}: external account residual")
        dq_diag = diag["macro"].get("mr_rd_cge_dq_sfc")
        if dq_diag is not None:
            if dq_diag["max_material_residual"] > 1e-10:
                errors.append(f"{cell}: DQ material accounting residual")
            if dq_diag["max_sfc_residual"] > 1e-10:
                errors.append(f"{cell}: SFC transaction residual")
            if dq_diag["max_capital_queue_residual"] > 1e-10:
                errors.append(f"{cell}: capital construction queue residual")
            if dq_diag["max_age_complementarity_residual"] > 5e-3:
                errors.append(f"{cell}: AGE complementarity residual")
            if dq_diag["max_jorgenson_share_residual"] > 1e-12:
                errors.append(f"{cell}: Jorgenson adding-up residual")
            if dq_diag["max_hssw_identity_residual"] > 1e-10:
                errors.append(f"{cell}: HSSW expenditure identity residual")
            if dq_diag["max_inner_residual"] > 5e-4:
                errors.append(f"{cell}: DQ price-quantity inner residual")
            if dq_diag["minimum_inventory"] < -1e-12:
                errors.append(f"{cell}: negative economic inventory")
            if dq_diag["minimum_backlog"] < -1e-12:
                errors.append(f"{cell}: negative economic backlog")
        if diag["operations"]["max_stock_accounting_residual"] > 1e-7:
            errors.append(f"{cell}: stock accounting residual")
        if diag["pnt"]["information_inverse_residual"] > 1e-7:
            errors.append(f"{cell}: PNT inverse residual")
        if geometry_enabled:
            if diag["mfg"]["max_mass_residual"] > 1e-10:
                errors.append(f"{cell}: Fisher-Rao simplex mass residual")
            if diag["operations"]["max_uot_solver_residual"] > 1e-5:
                errors.append(f"{cell}: unbalanced OT solver residual")
            if diag["pnt"]["minimum_covariance_eigenvalue"] <= 0.0:
                errors.append(f"{cell}: SPD covariance lost positivity")
            if diag["spatial"]["graph_spectral_gap"] <= 0.0:
                errors.append(f"{cell}: spatial graph disconnected")
            if diag["geometry"]["max_aggregation_residual"] > 1e-12:
                errors.append(f"{cell}: multiscale aggregation residual")
        if diag["finance_accounting"]["max_bilateral_aid_residual"] > 1e-10:
            errors.append(f"{cell}: bilateral aid residual")
        if diag["mechanism"] is not None:
            mechanism_diag = diag["mechanism"]
            if mechanism_diag["nonconverged_months"]:
                errors.append(
                    f"{cell}: mechanism nonconverged months="
                    f"{mechanism_diag['nonconverged_months']}"
                )
            for key in (
                "max_incentive_compatibility_residual",
                "max_participation_residual",
                "max_budget_residual",
                "max_resource_residual",
                "max_limited_liability_residual",
                "max_dynamic_commitment_residual",
                "max_transfer_balance_residual",
            ):
                if mechanism_diag[key] > 1e-9:
                    errors.append(f"{cell}: mechanism {key}")
        if diag["energy"] is not None:
            if diag["energy"]["max_flow_min_cut_residual"] > 1e-9:
                errors.append(f"{cell}: energy max-flow/min-cut residual")
            if diag["energy"]["max_delivery_above_demand_residual"] > 1e-10:
                errors.append(f"{cell}: energy delivery feasibility residual")
            if geometry_enabled:
                if diag["energy"]["max_hodge_divergence_residual"] > 1e-10:
                    errors.append(f"{cell}: Hodge divergence residual")
                if diag["energy"]["max_dirac_interconnection_residual"] > 1e-12:
                    errors.append(f"{cell}: Dirac interconnection residual")
                if diag["energy"]["max_port_hamiltonian_residual"] > 1e-12:
                    errors.append(f"{cell}: port-Hamiltonian storage residual")
        if diag["hybrid"] is not None:
            hybrid_diag = diag["hybrid"]
            if hybrid_diag["max_reset_residual"] > 1e-12:
                errors.append(f"{cell}: impulse reset residual")
            if hybrid_diag["max_sliding_normal_residual"] > 1e-12:
                errors.append(f"{cell}: Filippov sliding normal residual")
            if hybrid_diag["max_hysteresis_violation"] > 0.0:
                errors.append(f"{cell}: Filippov hysteresis violation")
            if any(value != months for value in hybrid_diag["event_calls"].values()):
                errors.append(f"{cell}: impulse event ledger clock mismatch")
            for surface, count in hybrid_diag["switch_count"].items():
                hybrid_switch_totals[surface] += count
                if hybrid_diag["terminal_regime_share"][surface] < 0.995:
                    hybrid_nonsaturated_cells[surface] += 1
        if diag["topology"] is not None:
            topology_diag = diag["topology"]
            if topology_diag["transition_accounting_residual"] != 0:
                errors.append(f"{cell}: topological transition accounting residual")
            topology_active_cells += int(topology_diag["observed_transition_edges"] > 1)
            topology_visited_symbols.update(topology_diag["visited_symbols"])
            topology_max_feedback_deviation = max(
                topology_max_feedback_deviation,
                topology_diag["max_feedback_deviation"],
            )
            if not np.isfinite(topology_diag["topological_entropy_upper_proxy"]):
                errors.append(f"{cell}: nonfinite topological entropy")
        for horizon, horizon_diag in diag["campaign_horizons"].items():
            if horizon_diag["importance_sampling_effective_sample_size"] < 0.35 * paths:
                errors.append(f"{cell}|{horizon}: importance ESS")
    if hybrid_enabled:
        for surface, count in hybrid_switch_totals.items():
            if count == 0:
                errors.append(
                    f"global Filippov {surface} surface has no dynamic state movement"
                )
            if hybrid_nonsaturated_cells[surface] == 0:
                errors.append(f"global Filippov {surface} surface is permanently saturated")
    if topology_enabled:
        if topology_active_cells == 0:
            errors.append("global topological dynamics has no nontrivial transition graph")
        if len(topology_visited_symbols) <= 1:
            errors.append("global topological symbolic partition has no state movement")
        if topology_max_feedback_deviation <= 0.0:
            errors.append("global topological feedback is inactive")
    for case in cases:
        for name in scenarios:
            sequence = [
                weighted_mean(conditional[case][name][h], importance_weights[name])
                for h in battle.HORIZON_MONTHS
            ]
            if any(right + 1e-12 < left for left, right in zip(sequence, sequence[1:])):
                errors.append(f"{case}|{name}: nonmonotone absorbing event")
    for row in calendar_2100:
        decomposition = (
            row["peace_arrangement_cif"] + row["legal_separation_cif"]
            + row["status_quo_survival"] + row["joint_force_and_broad_control_by_2100"]
            + row["force_without_broad_control_by_2100"]
        )
        if abs(decomposition - 1.0) > 1e-9:
            errors.append(f"{row['trade_scenario']}: calendar-2100 decomposition")
    if errors:
        version = (
            "V5.2" if (mfg_config or {}).get("advanced_graphon") else (
            "V5.1" if (hybrid_enabled or topology_enabled) else ("V5.0" if geometry_enabled else
            ("V4.9" if energy_enabled else ("V4.8" if mechanism_enabled else "V4.7"))
            )
            )
        )
        failure_path = traceability_path.with_name(
            f"taiwan_{version.lower().replace('.', '')}_runtime_validation_failure.json"
        )
        failure_path.write_text(json.dumps({
            "model_version": version,
            "scope": "runtime validation failure; no substantive results published",
            "paths": paths,
            "months": months,
            "errors": errors,
            "diagnostic_cells": len(diagnostics),
            "hybrid_switch_totals": hybrid_switch_totals,
            "hybrid_nonsaturated_cells": hybrid_nonsaturated_cells,
            "topology_active_cells": topology_active_cells,
            "topology_visited_symbols": sorted(topology_visited_symbols),
            "topology_max_feedback_deviation": topology_max_feedback_deviation,
            "module_diagnostics": diagnostics,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        raise RuntimeError(f"{version} runtime validation failed: " + "; ".join(errors))
    return rows, aggregate, diagnostics, {
        "intervention": choice_model.diagnostics(),
        "survival": survival_model.diagnostics(),
        "prewar_closed_system": prewar_trajectory["diagnostics"],
        "intervention_probability_means": dict(zip(battle.CASES, np.mean(choice.probabilities, axis=0).tolist())),
        "force_onset_mean": float(np.mean(force_paths)),
        "prewar_metadata": prewar["metadata"],
        "calendar_2100": calendar_2100,
        "long_term_checkpoints": [
            {
                "year": year,
                "force": float(np.mean(survival.cumulative_incidence_history[:, min((year - LONG_TERM_START_YEAR + 1) * 12 - 1, LONG_TERM_MONTHS - 1), 0])),
                "peace": float(np.mean(survival.cumulative_incidence_history[:, min((year - LONG_TERM_START_YEAR + 1) * 12 - 1, LONG_TERM_MONTHS - 1), 1])),
                "separation": float(np.mean(survival.cumulative_incidence_history[:, min((year - LONG_TERM_START_YEAR + 1) * 12 - 1, LONG_TERM_MONTHS - 1), 2])),
                "status_quo": float(np.mean(survival.survival_history[:, min((year - LONG_TERM_START_YEAR + 1) * 12 - 1, LONG_TERM_MONTHS - 1)])),
            }
            for year in (2030, 2035, 2050, 2075, 2100)
        ],
        "runtime_validation": (
            "TAIWAN_V51_RUNTIME_VALIDATION: PASS" if (hybrid_enabled or topology_enabled) else (
            "TAIWAN_V50_RUNTIME_VALIDATION: PASS" if geometry_enabled else (
                "TAIWAN_V49_RUNTIME_VALIDATION: PASS" if energy_enabled else (
                "TAIWAN_V48_RUNTIME_VALIDATION: PASS"
                if mechanism_enabled else "TAIWAN_V47_RUNTIME_VALIDATION: PASS"
                )
            ))
        ),
    }


def write(rows, aggregate, diagnostics, verification) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    verification["result_sha256"] = _hash(OUT_CSV)
    OUT_JSON.write_text(json.dumps({
        "verification": verification,
        "aggregate": aggregate,
        "conditional": rows,
        "module_diagnostics": diagnostics,
    }, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    if os.environ.get("TAIWAN_ALLOW_COMPLETE_RUN") != "1":
        raise RuntimeError(
            "outcome simulation disabled; set TAIWAN_ALLOW_COMPLETE_RUN=1 only after the complete-model gate passes"
        )
    rows, aggregate, diagnostics, verification = run()
    write(rows, aggregate, diagnostics, verification)


if __name__ == "__main__":
    main()
