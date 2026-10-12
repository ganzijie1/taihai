"""DIAGNOSTIC_ONLY: V6.2 structural and optimal-transport invariants."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import simulate_taiwan_fully_closed_v60 as v60
import simulate_trade_war_v43 as trade
from continuous_multipop_mfg_v62 import ContinuousMultiPopulationMFGV62
from sectoral_economy_v61 import CoupledSectoralEconomicSystem
from sectoral_economy_v62 import CoupledSectoralEconomicSystemV62


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "outputs" / "taiwan_v62_diagnostic.json"
AUDIT = ROOT / "outputs" / "taiwan_v62_static_coupling_audit.json"


def sector_inputs(paths: int, month: int) -> dict:
    intensity = np.linspace(0.08, 0.92, paths)[:, None]
    phase = min(month / 8.0, 1.0)
    damage = np.clip(
        intensity * phase * np.array([[0.90, 1.05, 0.72, 0.82]]), 0.0, 0.75
    )
    financial = np.clip(0.04 + 0.48 * damage, 0.0, 0.75)
    panic = np.clip(0.05 + 0.42 * damage, 0.0, 0.80)
    return {
        "month": month,
        "actor_damage": damage,
        "actor_logistics": np.clip(1.0 - 0.62 * damage, 0.18, 1.0),
        "actor_labor": np.clip(1.0 - 0.35 * damage, 0.25, 1.0),
        "actor_financial_stress": financial,
        "actor_payment": np.clip(1.0 - 0.46 * financial, 0.25, 1.0),
        "actor_fiscal": np.clip(0.94 - 0.36 * financial, 0.25, 1.0),
        "actor_social_action": np.clip(0.04 + 0.32 * damage, 0.0, 0.65),
        "actor_panic": panic,
        "actor_mobilization": np.clip(0.08 + 0.52 * damage, 0.0, 0.75),
        "blockade": np.clip(np.linspace(0.0, 0.58, paths) * phase, 0.0, 0.85),
        "max_outer_iterations": 18,
        "tolerance": 3.0e-3,
    }


def run_sector(cls):
    paths = 4
    model = cls(paths, 62_001, list(trade.SCENARIOS.values())[0])
    factors = None
    for month in range(18):
        factors = model.update(**sector_inputs(paths, month))
    return model, factors


def run_mfg() -> ContinuousMultiPopulationMFGV62:
    paths = 4
    initial = np.array([
        [0.62, 0.48, 0.55, 0.52],
        [0.55, 0.44, 0.58, 0.56],
        [0.66, 0.40, 0.52, 0.49],
        [0.59, 0.46, 0.60, 0.54],
    ])
    model = ContinuousMultiPopulationMFGV62(
        paths, 62_101, initial, **v60.MFG_CONFIG
    )
    for month in range(4):
        phase = (month + 1) / 4.0
        actor_axis = np.array([[0.16, 0.35, 0.20, 0.24]])
        damage = phase * np.linspace(0.8, 1.2, paths)[:, None] * actor_axis
        model.update(
            progress=np.clip(0.15 - damage, -1.0, 1.0),
            damage=damage,
            shortage=np.clip(0.08 + 0.55 * damage, 0.0, 1.0),
            inflation=np.clip(0.04 + 0.42 * damage, 0.0, 1.0),
            financial_stress=np.clip(0.06 + 0.48 * damage, 0.0, 1.0),
            command=np.clip(0.94 - 0.35 * damage, 0.0, 1.0),
            network=np.clip(0.90 - 0.30 * damage, 0.0, 1.0),
            contagion_mobilization=np.clip(0.18 + 0.25 * damage, 0.0, 1.0),
            contagion_fatigue=np.clip(0.10 + 0.38 * damage, 0.0, 1.0),
            common_noise=np.broadcast_to(
                np.array([[0.08, -0.04, 0.05, 0.02]]) * phase,
                (paths, 4),
            ),
            jump_signal=np.clip(damage * (month == 2), 0.0, 1.0),
            max_fixed_point_iterations=48,
            tolerance=5.0e-4,
        )
    return model


def main() -> None:
    v62, final = run_sector(CoupledSectoralEconomicSystemV62)
    v61, old_final = run_sector(CoupledSectoralEconomicSystem)
    mfg = run_mfg()
    sector = v62.diagnostics()["three_sector_physical_economy"]
    structural = sector["v62_structural_transformation"]
    mfg_diag = mfg.diagnostics()
    transport = mfg_diag["partial_observation"]["v62_dynamic_transport"]
    joint = mfg_diag["v62_joint_state_action_wfr"]
    switch_difference = max(
        float(np.max(np.abs(final.gdp_factor - old_final.gdp_factor))),
        float(np.max(np.abs(final.industry_factor - old_final.industry_factor))),
        float(np.max(np.abs(final.inflation - old_final.inflation))),
    )
    checks = {
        "labor_simplex": sector["max_labor_share_residual"] < 1.0e-12,
        "employment_value_added_separated": structural[
            "employment_and_value_added_separated"
        ],
        "labor_data_are_distinct": not np.allclose(
            structural["baseline_labor_share"], structural["value_added_share"]
        ),
        "uot_exercised": structural["max_monthly_labor_share_move"] > 0.0,
        "structural_multiplier_exercised": (
            structural["minimum_structural_multiplier"] < 1.0
            and structural["maximum_structural_multiplier"] > 1.0
        ),
        "v62_changes_macro_path": switch_difference > 1.0e-7,
        "posterior_simplex": mfg_diag["partial_observation"][
            "max_posterior_simplex_residual"
        ] < 1.0e-10,
        "bridge_stochastic": transport["max_bridge_transition_residual"] < 1.0e-10,
        "dynamic_fgw_exercised": transport["dynamic_fgw_refresh_count"] > 0,
        "causal_information": not transport["future_information_used"],
        "online_dro_exercised": transport["maximum_online_radius"] > 0.0,
        "joint_action_wfr_exercised": joint["max_joint_action_reaction"] > 0.0,
        "joint_action_bound": joint["max_joint_action_mass_residual"] < 1.0e-12,
        "mfg_density_mass": mfg_diag["max_mass_residual"] < 1.0e-10,
        "network_kernel_rows": mfg_diag["max_kernel_row_residual"] < 1.0e-10,
    }
    checks = {key: bool(value) for key, value in checks.items()}
    payload = {
        "classification": "DIAGNOSTIC_ONLY; no outcome probabilities",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "v61_v62_switch_difference": switch_difference,
        "three_sector": sector,
        "mfg": mfg_diag,
    }
    TARGET.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if payload["status"] == "PASS":
        audit = json.loads(AUDIT.read_text(encoding="utf-8"))
        audit["diagnostic_status"] = "PASS"
        AUDIT.write_text(
            json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(f"TAIWAN_V62_DIAGNOSTIC: {payload['status']}")
    if payload["status"] != "PASS":
        print(json.dumps({k: v for k, v in checks.items() if not v}, indent=2))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
