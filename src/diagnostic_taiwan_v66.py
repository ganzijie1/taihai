"""DIAGNOSTIC_ONLY: V6.6 OT and symplectic invariants; no outcomes."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import simulate_taiwan_fully_closed_v60 as v60
from continuous_multipop_mfg_v66 import ContinuousMultiPopulationMFGV66
from transport_symplectic_v66 import CoupledTransportSymplecticSystemV66


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "outputs" / "taiwan_v66_ot_symplectic_diagnostic.json"


def main() -> None:
    paths = 6
    terrain = {"land_share_above_500m": 0.47}
    transport = CoupledTransportSymplecticSystemV66(paths, 66_101, terrain)
    initial = np.array([
        [0.62, 0.48, 0.55, 0.52], [0.55, 0.44, 0.58, 0.56],
        [0.66, 0.40, 0.52, 0.49], [0.59, 0.46, 0.60, 0.54],
        [0.64, 0.43, 0.57, 0.51], [0.57, 0.47, 0.53, 0.55],
    ])
    mfg = ContinuousMultiPopulationMFGV66(
        paths, 66_201, initial, **v60.MFG_CONFIG
    )
    for month in range(12):
        phase = (month + 1) / 12.0
        path_axis = np.linspace(0.78, 1.18, paths)[:, None]
        actor_axis = np.array([[0.15, 0.32, 0.18, 0.22]])
        damage = np.clip(phase * path_axis * actor_axis, 0.0, 0.72)
        intensity = np.linspace(0.10, 0.88, paths) * phase
        network = np.clip(0.94 - 0.42 * damage, 0.08, 1.0)
        command = np.clip(0.95 - 0.36 * damage, 0.08, 1.0)
        shortage = np.clip(0.06 + 0.60 * damage, 0.0, 1.0)
        transport.update(
            month=month, conflict_intensity=intensity,
            actor_damage=damage,
            actor_output=np.clip(1.0 - 0.36 * damage, 0.12, 1.0),
            financial_capacity=np.clip(0.93 - 0.48 * damage, 0.10, 1.0),
            actor_network=network, actor_command=command,
            actor_support=np.clip(initial - 0.10 * damage, 0.05, 0.95),
            actor_mobilization=np.clip(0.12 + 0.58 * damage, 0.0, 1.0),
            stock_shortage=shortage,
            geometry_logistics=np.clip(0.96 - 0.44 * damage, 0.08, 1.0),
            geometry_network=np.clip(0.95 - 0.40 * damage, 0.08, 1.0),
        )
        mfg.update(
            progress=np.clip(0.14 - damage, -1.0, 1.0), damage=damage,
            shortage=shortage,
            inflation=np.clip(0.03 + 0.35 * damage, 0.0, 1.0),
            financial_stress=np.clip(0.05 + 0.45 * damage, 0.0, 1.0),
            command=command, network=network,
            contagion_mobilization=np.clip(0.16 + 0.25 * damage, 0.0, 1.0),
            contagion_fatigue=np.clip(0.09 + 0.35 * damage, 0.0, 1.0),
            common_noise=np.broadcast_to(
                np.array([[0.06, -0.03, 0.04, 0.02]]) * phase, (paths, 4)
            ),
            jump_signal=np.clip(damage * (month in (4, 9)), 0.0, 1.0),
            max_fixed_point_iterations=48, tolerance=5.0e-4,
        )
    td = transport.diagnostics()
    md = mfg.diagnostics()
    deep = md["partial_observation"]["v66_deep_causal_transport"]
    checks = {
        "transport_clock": td["months"] == 12,
        "dynamic_flow": td["dynamic_flow_months"] > 0,
        "mass_balance": td["max_mass_balance_residual"] < 1.0e-10,
        "shared_capacity": td["max_shared_capacity_violation"] < 1.0e-12,
        "nonnegative_inventory": td["max_nonnegativity_violation"] < 1.0e-12,
        "wfr_account": td["max_wfr_mass_residual"] < 1.0e-12,
        "port_power": td["max_port_power_residual"] < 1.0e-10,
        "symplectic_defect": td["max_symplectic_defect"] < 1.0e-12,
        "so3_orthogonality": td["max_so3_orthogonality_residual"] < 1.0e-10,
        "so3_determinant": td["max_so3_determinant_residual"] < 1.0e-10,
        "jump_account": td["max_jump_account_residual"] < 1.0e-12,
        "causal_transport": not td["future_information_used"],
        "path_bridge": deep["max_path_kernel_row_residual"] < 1.0e-10,
        "temporal_marginal": deep["max_temporal_marginal_residual"] < 1.0e-10,
        "partial_mass": deep["max_partial_mass_residual"] < 1.0e-10,
        "multilayer_fgw": deep["multilayer_refresh_count"] > 0,
        "causal_ot": not deep["future_information_used"],
    }
    checks = {key: bool(value) for key, value in checks.items()}
    payload = {
        "classification": "DIAGNOSTIC_ONLY; no outcome probabilities",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "transport_symplectic": td,
        "deep_optimal_transport": deep,
    }
    TARGET.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"TAIWAN_V66_DIAGNOSTIC: {payload['status']}")
    if payload["status"] != "PASS":
        print(json.dumps({k: v for k, v in checks.items() if not v}, indent=2))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
