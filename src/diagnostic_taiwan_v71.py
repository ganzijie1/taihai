"""DIAGNOSTIC_ONLY: combinatorial network invariants; no outcome probabilities."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

from combinatorial_network_v71 import CombinatorialOperationalNetworkV71
from operational_constraints_v47 import DynamicOperationalConstraintSystem


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "outputs" / "taiwan_v71_combinatorial_network_diagnostic.json"
LEGACY = ROOT / "taihai-publish" / "src" / "operational_constraints_v47.py"


def load_legacy_class():
    spec = importlib.util.spec_from_file_location("legacy_operational_v47", LEGACY)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.DynamicOperationalConstraintSystem


def inputs(paths: int, month: int) -> dict:
    phase = (month + 1) / 18.0
    damage = np.repeat(
        np.clip(
            np.array([[0.08, 0.16, 0.06, 0.09]]) + 0.30 * phase,
            0.0, 0.72,
        ), paths, axis=0,
    )
    interdiction = np.repeat(
        np.clip(
            np.array([[0.22, 0.36, 0.14, 0.18]]) + 0.28 * phase,
            0.0, 0.80,
        ), paths, axis=0,
    )
    return {
        "production": np.clip(0.10 - 0.035 * damage, 0.015, 0.12),
        "consumption": np.clip(0.11 + 0.025 * phase + 0.02 * damage, 0.02, 0.18),
        "damage": damage,
        "interdiction": interdiction,
        "command": np.clip(0.94 - 0.38 * damage, 0.20, 0.98),
        "pnt": np.clip(0.96 - 0.42 * damage, 0.18, 0.99),
        "financial_capacity": np.clip(0.95 - 0.28 * damage, 0.20, 0.99),
    }


def max_state_delta(left, right) -> float:
    names = (
        "inventory", "spares", "spare_pipeline", "repair_queue_state",
        "integrity", "search_prior",
    )
    return max(
        float(np.max(np.abs(getattr(left, name) - getattr(right, name))))
        for name in names
    )


def main() -> None:
    paths = 8
    terrain = {
        "land_share_above_500m": 0.42,
        "sky_view_proxy_p10_p50_p90": [0.31, 0.58, 0.83],
    }

    # Refactoring the base class into hooks must not alter any inherited model.
    legacy_class = load_legacy_class()
    refactored = DynamicOperationalConstraintSystem(
        paths, 71_001, terrain, geometry_enabled=True, advanced_geometry=True
    )
    legacy = legacy_class(
        paths, 71_001, terrain, geometry_enabled=True, advanced_geometry=True
    )
    base_regression_delta = 0.0
    for month in range(6):
        args = inputs(paths, month)
        refactored.update(**args)
        legacy.update(**args)
        base_regression_delta = max(
            base_regression_delta, max_state_delta(refactored, legacy)
        )

    active = CombinatorialOperationalNetworkV71(
        paths, 71_101, terrain, geometry_enabled=True, advanced_geometry=True
    )
    ablated = DynamicOperationalConstraintSystem(
        paths, 71_101, terrain, geometry_enabled=True, advanced_geometry=True
    )
    state_delta = 0.0
    pipeline_observed = False
    for month in range(18):
        args = inputs(paths, month)
        active.update(**args)
        ablated.update(**args)
        state_delta = max(state_delta, max_state_delta(active, ablated))
        pipeline_observed |= bool(np.any(active.shipment_pipeline > 1.0e-12))

    diag = active.diagnostics()["combinatorial_network_v71"]
    checks = {
        "legacy_zero_change_regression": base_regression_delta < 1.0e-14,
        "active_same_path_effect": state_delta > 1.0e-8,
        "time_expanded_pipeline_active": pipeline_observed,
        "monthly_clock": diag["months"] == 18,
        "shared_capacity_feasible": diag["maximum_capacity_residual"] < 1.0e-12,
        "stock_flow_accounting": diag["maximum_stock_residual"] < 1.0e-10,
        "matching_feasible": diag["maximum_matching_residual"] < 1.0e-12,
        "maxflow_equals_mincut": diag["maximum_maxflow_mincut_residual"] < 1.0e-12,
        "capacity_monotonicity": diag["maximum_monotonicity_violation"] < 1.0e-14,
        "nonnegative_pipeline": diag["maximum_pipeline_negative_mass"] < 1.0e-14,
        "cascade_fixed_point": (
            diag["maximum_cascade_residual"] < 1.0e-8
            and diag["cascade_nonconvergence_months"] == 0
        ),
        "menger_connectivity": diag["minimum_edge_connectivity"] >= 1,
        "partition_matroid_budget": diag["protection_budget_violations"] == 0,
        "hypergraph_gate_active": diag["minimum_hypergraph_gate"] < 0.999,
        "no_terminal_probability_adjustment": (
            not diag["terminal_probability_adjustment"]
        ),
    }
    payload = {
        "classification": "DIAGNOSTIC_ONLY; no outcome probabilities",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "legacy_refactor_max_state_delta": base_regression_delta,
        "active_ablation_max_state_delta": state_delta,
        "diagnostics": diag,
    }
    TARGET.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"TAIWAN_V71_DIAGNOSTIC: {payload['status']}")
    if payload["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
