"""DIAGNOSTIC_ONLY: V7.3 continuous-graph invariants; no outcomes."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from algebraic_discrete_network_v72 import AlgebraicDiscreteOperationalNetworkV72
from continuous_graph_network_v73 import ContinuousGraphOperationalNetworkV73
from diagnostic_taiwan_v72 import inputs, state_delta


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "outputs" / "taiwan_v73_continuous_graph_diagnostic.json"


def main() -> None:
    paths, months = 8, 24
    terrain = {
        "land_share_above_500m": 0.42,
        "sky_view_proxy_p10_p50_p90": [0.31, 0.58, 0.83],
    }
    active = ContinuousGraphOperationalNetworkV73(
        paths, 73_101, terrain, geometry_enabled=True, advanced_geometry=True
    )
    ablated = AlgebraicDiscreteOperationalNetworkV72(
        paths, 73_101, terrain, geometry_enabled=True, advanced_geometry=True
    )
    maximum_state_delta = 0.0
    for month in range(months):
        args = inputs(paths, month)
        active.update(**args)
        ablated.update(**args)
        maximum_state_delta = max(maximum_state_delta, state_delta(active, ablated))

    diag = active.diagnostics()
    graph = diag["continuous_graph_network_v73"]
    inherited = diag["combinatorial_network_v71"]
    algebraic = diag["algebraic_discrete_network_v72"]
    checks = {
        "same_path_ablation_changes_state": maximum_state_delta > 1.0e-6,
        "continuous_state_evolves": graph["maximum_continuous_state_delta"] > 1.0e-6,
        "monthly_clock": graph["months"] == months,
        "finite_volume_conservative": graph["maximum_metric_balance_residual"] <= 1.0e-10,
        "finite_volume_nonnegative": graph["maximum_negative_density"] <= 1.0e-12,
        "finite_volume_stable": graph["maximum_cfl"] < 1.0,
        "telegraph_bounded": graph["maximum_signal_bound_violation"] <= 1.0e-12,
        "control_bounded": graph["maximum_control_bound_violation"] <= 1.0e-12,
        "finite_graph_rollback": graph["maximum_finite_graphon_rollback_error"] <= 0.10,
        "sparse_rollback_graph": 0.0 < graph["finite_network_density"] < 0.25,
        "continuous_capacity_gate_bounded": 0.0 < graph["minimum_continuous_capacity_gate"] <= 1.0,
        "enterprise_gate_bounded": 0.0 < graph["minimum_enterprise_gate"] <= 1.0,
        "physical_inventory_has_single_owner": not graph["continuous_state_owns_physical_inventory"],
        "capacity_feasible": inherited["maximum_capacity_residual"] <= 1.0e-10,
        "stock_conserved": inherited["maximum_stock_residual"] <= 1.0e-9,
        "cascade_converged": inherited["maximum_cascade_residual"] <= 1.0e-7,
        "spectral_interlacing": algebraic["maximum_interlacing_violation"] <= 1.0e-9,
        "no_probability_postprocessing": not graph["terminal_probability_adjustment"],
    }
    payload = {
        "model_version": "V7.3-CONTINUOUS-GRAPH",
        "classification": "DIAGNOSTIC_ONLY; no terminal probabilities",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "active_ablation_max_state_delta": maximum_state_delta,
        "diagnostics": graph,
        "inherited_algebraic_diagnostics": algebraic,
        "inherited_network_diagnostics": inherited,
    }
    TARGET.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"TAIWAN_V73_DIAGNOSTIC: {payload['status']}")
    if payload["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
