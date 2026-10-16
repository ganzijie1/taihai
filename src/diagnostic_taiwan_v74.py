"""DIAGNOSTIC_ONLY: V7.4 Finsler invariants; no outcome probabilities."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from continuous_multipop_mfg_v47 import ContinuousMultiPopulationMFG
from diagnostic_taiwan_v72 import inputs, state_delta
from finsler_network_v74 import FinslerOperationalNetworkV74
from continuous_graph_network_v73 import ContinuousGraphOperationalNetworkV73


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "outputs" / "taiwan_v74_finsler_diagnostic.json"


def _mfg_args(paths: int) -> dict:
    progress = np.tile(np.array([[0.18, -0.10, 0.05, 0.02]]), (paths, 1))
    return {
        "progress": progress,
        "damage": np.full((paths, 4), 0.18),
        "shortage": np.full((paths, 4), 0.22),
        "inflation": np.full((paths, 4), 0.08),
        "financial_stress": np.full((paths, 4), 0.20),
        "command": np.full((paths, 4), 0.78),
        "network": np.full((paths, 4), 0.74),
        "contagion_mobilization": np.full((paths, 4), 0.30),
        "contagion_fatigue": np.full((paths, 4), 0.24),
        "common_noise": np.zeros((paths, 4)),
        "jump_signal": np.zeros((paths, 4)),
    }


def main() -> None:
    paths, months = 8, 24
    terrain = {
        "land_share_above_500m": 0.42,
        "sky_view_proxy_p10_p50_p90": [0.31, 0.58, 0.83],
    }
    active = FinslerOperationalNetworkV74(
        paths, 74_101, terrain, geometry_enabled=True, advanced_geometry=True
    )
    ablated = ContinuousGraphOperationalNetworkV73(
        paths, 74_101, terrain, geometry_enabled=True, advanced_geometry=True
    )
    maximum_state_delta = 0.0
    for month in range(months):
        args = inputs(paths, month)
        active.update(**args)
        ablated.update(**args)
        maximum_state_delta = max(maximum_state_delta, state_delta(active, ablated))

    mfg_paths = 2
    initial_support = np.full((mfg_paths, 4), 0.56)
    baseline = ContinuousMultiPopulationMFG(
        mfg_paths, 74_201, initial_support, grid_points=15
    )
    neutral = ContinuousMultiPopulationMFG(
        mfg_paths, 74_201, initial_support, grid_points=15
    )
    shifted = ContinuousMultiPopulationMFG(
        mfg_paths, 74_201, initial_support, grid_points=15
    )
    args = _mfg_args(mfg_paths)
    baseline_factor = baseline.update(**args)
    neutral_factor = neutral.update(**args, mobility_cost=np.ones((mfg_paths, 4)))
    shifted_factor = shifted.update(
        **args,
        mobility_cost=np.tile(np.array([[1.35, 1.18, 1.28, 1.22]]), (mfg_paths, 1)),
    )
    neutral_regression = float(
        np.max(np.abs(baseline_factor.support - neutral_factor.support))
    )
    mfg_state_delta = float(
        np.max(np.abs(shifted_factor.support - neutral_factor.support))
    )

    diag = active.diagnostics()
    finsler = diag["finsler_network_v74"]
    continuous = diag["continuous_graph_network_v73"]
    inherited = diag["combinatorial_network_v71"]
    shortest_path_residual = float(np.max(np.abs(
        active.last_finsler_sink_arrival
        - np.min(active.last_finsler_path_cost, axis=-1)
    )))
    checks = {
        "same_path_ablation_changes_state": maximum_state_delta > 1.0e-7,
        "strictly_positive_cost": finsler["minimum_finsler_cost"] > 0.0,
        "randers_strong_convexity": finsler["maximum_randers_one_form_norm"] < 1.0,
        "direction_reversal_is_asymmetric": finsler["maximum_direction_asymmetry"] > 0.0,
        "zero_drift_recovers_symmetric_metric": finsler["maximum_zero_drift_error"] <= 1.0e-12,
        "positive_homogeneity": finsler["maximum_positive_homogeneity_residual"] <= 1.0e-12,
        "shortest_path_consistency": shortest_path_residual <= 1.0e-10,
        "eikonal_inequality": finsler["maximum_eikonal_residual"] <= 1.0e-10,
        "finite_volume_conservative": continuous["maximum_metric_balance_residual"] <= 1.0e-10,
        "hard_capacity_feasible": inherited["maximum_capacity_residual"] <= 1.0e-10,
        "stock_conserved": inherited["maximum_stock_residual"] <= 1.0e-9,
        "single_inventory_owner": not finsler["continuous_geometry_owns_physical_inventory"],
        "mfg_neutral_regression": neutral_regression <= 1.0e-12,
        "mfg_cost_changes_state": mfg_state_delta > 1.0e-9,
        "mfg_cost_recorded": shifted.diagnostics()["finsler_mobility_cost_active"],
        "no_probability_postprocessing": not finsler["terminal_probability_adjustment"],
    }
    payload = {
        "model_version": "V7.4-FINSLER-CLOSED-LOOP",
        "classification": "DIAGNOSTIC_ONLY; no terminal probabilities",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "active_ablation_max_state_delta": maximum_state_delta,
        "shortest_path_residual": shortest_path_residual,
        "mfg_neutral_regression": neutral_regression,
        "mfg_shifted_state_delta": mfg_state_delta,
        "diagnostics": finsler,
        "mfg_diagnostics": shifted.diagnostics(),
    }
    TARGET.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"TAIWAN_V74_DIAGNOSTIC: {payload['status']}")
    if payload["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
