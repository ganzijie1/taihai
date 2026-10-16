"""DIAGNOSTIC_ONLY: V7.0 local gates and common-random-number ablations."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from simulate_dual_circulation_v44 import v43
from tinbergen_algebraic_v70 import (
    AsymmetricAdjustmentMacroV70,
    TinbergenAlgebraicRollingGameV70,
)


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "outputs" / "taiwan_v70_tinbergen_algebraic_diagnostic.json"


def strategy_inputs(paths: int, month: int) -> dict:
    phase = month / 11.0
    actor = np.array([[0.08, 0.18, 0.10, 0.12]])
    damage = np.repeat(np.clip(actor + 0.52 * phase, 0.0, 0.86), paths, axis=0)
    return {
        "progress": np.repeat(np.array([[0.08, -0.08, -0.05, -0.05]]), paths, axis=0),
        "stocks": np.clip(0.76 - 0.48 * phase - 0.25 * damage, 0.08, 1.0),
        "damage": damage,
        "economic_capacity": np.clip(0.88 - 0.55 * damage, 0.12, 1.0),
        "support": np.clip(0.68 - 0.32 * damage, 0.12, 0.9),
        "command": np.clip(0.90 - 0.58 * damage, 0.10, 1.0),
        "intervention": np.repeat(np.array([[1.0, 1.0, 0.82, 0.72]]), paths, axis=0),
    }


def macro_inputs(paths: int, month: int) -> dict:
    phase = month / 7.0
    damage = np.repeat(
        np.clip(np.array([[0.05, 0.13, 0.04, 0.06]]) + 0.32 * phase, 0.0, 0.65),
        paths, axis=0,
    )
    logistics = np.clip(0.96 - 0.52 * damage, 0.12, 1.0)
    return {
        "month": month,
        "actor_damage": damage,
        "actor_logistics": logistics,
        "actor_labor": np.clip(0.98 - 0.34 * damage, 0.15, 1.05),
        "actor_financial_stress": np.clip(0.06 + 0.48 * damage, 0.0, 1.0),
        "actor_payment": np.clip(0.97 - 0.30 * damage, 0.12, 1.0),
        "actor_fiscal": np.clip(0.95 - 0.24 * damage, 0.12, 1.0),
        "actor_social_action": np.clip(0.08 + 0.20 * damage, 0.0, 1.0),
        "actor_panic": np.clip(0.05 + 0.32 * damage, 0.0, 1.0),
        "actor_mobilization": np.clip(0.18 + 0.36 * damage, 0.0, 1.0),
        "blockade": np.full(paths, 0.18 + 0.32 * phase),
    }


def main() -> None:
    paths = 8
    active_game = TinbergenAlgebraicRollingGameV70(paths, 70_101)
    ablated_game = TinbergenAlgebraicRollingGameV70(
        paths, 70_101, reachability_enabled=False,
        algebraic_filter_enabled=False,
    )
    tropical_ablated_game = TinbergenAlgebraicRollingGameV70(
        paths, 70_101, tropical_enabled=False,
    )
    strategy_delta = 0.0
    tropical_delta = 0.0
    for month in range(12):
        inputs = strategy_inputs(paths, month)
        active_factor = active_game.update(**inputs)
        ablated_factor = ablated_game.update(**inputs)
        tropical_ablated_factor = tropical_ablated_game.update(**inputs)
        strategy_delta = max(
            strategy_delta,
            float(np.max(np.abs(active_factor.controls - ablated_factor.controls))),
        )
        tropical_delta = max(
            tropical_delta,
            float(np.max(np.abs(
                active_factor.controls - tropical_ablated_factor.controls
            ))),
        )

    # Registered near-boundary probe: selected control leaves the polynomial
    # safe set while the recovery allocation remains feasible.
    filter_probe = TinbergenAlgebraicRollingGameV70(paths, 88)
    stress = 0.793
    filter_probe.update(
        progress=np.full((paths, 4), -0.10),
        stocks=np.full((paths, 4), 1.0 - stress),
        damage=np.full((paths, 4), stress),
        economic_capacity=np.full((paths, 4), 1.0 - stress),
        support=np.full((paths, 4), 0.40),
        command=np.full((paths, 4), 1.0 - stress),
        intervention=np.ones((paths, 4)),
    )
    filter_diagnostic = filter_probe.diagnostics()["tinbergen_algebraic_v70"]

    central_scenario = next(iter(v43.SCENARIOS.values()))
    active_macro = AsymmetricAdjustmentMacroV70(
        paths, 70_201, central_scenario, adjustment_enabled=True
    )
    ablated_macro = AsymmetricAdjustmentMacroV70(
        paths, 70_201, central_scenario, adjustment_enabled=False
    )
    macro_delta = 0.0
    for month in range(8):
        inputs = macro_inputs(paths, month)
        active_factor = active_macro.update(**inputs)
        ablated_factor = ablated_macro.update(**inputs)
        macro_delta = max(
            macro_delta,
            float(np.max(np.abs(
                active_factor.gdp_factor - ablated_factor.gdp_factor
            ))),
        )

    strategy = active_game.diagnostics()["tinbergen_algebraic_v70"]
    full_macro_diagnostic = active_macro.diagnostics()
    macro = full_macro_diagnostic["tinbergen_asymmetric_adjustment_v70"]
    checks = {
        "full_target_tool_rank": strategy["minimum_local_response_rank"] == 4,
        "reachability_gate_active": strategy["reachability_gate_months"] == 12,
        "strategy_same_path_ablation": strategy_delta > 1.0e-8,
        "control_simplex": strategy["maximum_simplex_residual"] < 1.0e-12,
        "certifiable_barrier": strategy["maximum_barrier_violation"] < 1.0e-9,
        "polynomial_filter_active_on_registered_boundary": (
            filter_diagnostic["safety_filter_months"] == 1
            and filter_diagnostic["maximum_barrier_violation"] < 1.0e-9
        ),
        "identifiability_computed": strategy["identification_rank"] > 0,
        "multiple_equilibrium_diagnostic_active": strategy["multistable_path_months"] > 0,
        "tropical_same_path_ablation": tropical_delta > 1.0e-8,
        "tropical_min_plus_identity": (
            strategy["maximum_tropical_min_identity_residual"] < 1.0e-12
        ),
        "tropical_bottleneck_simplex": (
            abs(sum(strategy["tropical_type_shares"]) - 1.0) < 1.0e-12
        ),
        "tropical_switch_surface_observed": (
            strategy["tropical_near_switch_surface_share"] > 0.0
        ),
        "macro_adjustment_clock": macro["months"] == 8,
        "asymmetric_cost_active": macro["maximum_real_input_wedge"] > 0.0,
        "macro_same_path_ablation": macro_delta > 1.0e-10,
        "no_probability_adjustment": (
            not strategy["terminal_probability_adjustment"]
            and not macro["terminal_probability_adjustment"]
        ),
    }
    payload = {
        "classification": "DIAGNOSTIC_ONLY; no outcome probabilities",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "strategy": strategy,
        "registered_boundary_probe": filter_diagnostic,
        "macro": macro,
        "macro_convergence": {
            key: full_macro_diagnostic.get(key)
            for key in (
                "nonconverged_months", "max_outer_residual",
                "max_cge_solver_residual", "max_dsge_solver_residual",
            )
        },
        "common_random_number_ablation": {
            "maximum_control_delta": strategy_delta,
            "maximum_tropical_control_delta": tropical_delta,
            "maximum_gdp_factor_delta": macro_delta,
        },
    }
    TARGET.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"TAIWAN_V70_DIAGNOSTIC: {payload['status']}")
    if payload["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
