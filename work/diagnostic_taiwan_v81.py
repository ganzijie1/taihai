from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from differential_manifold_mfg_v81 import DifferentialManifoldMFGV81
from differential_manifold_v81 import DifferentialManifoldConstraintV81


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
TARGET = OUT / "taiwan_v81_diagnostic_only.json"


def inputs(paths: int, stress: float, jump: float = 0.0) -> dict:
    shape = (paths, 4)
    return {
        "damage": np.full(shape, 1.15 * stress),
        "shortage": np.full(shape, 1.05 * stress),
        "inflation": np.full(shape, 0.55 * stress),
        "financial_stress": np.full(shape, 1.10 * stress),
        "command": np.full(shape, np.clip(0.92 - 0.72 * stress, 0.08, 0.95)),
        "network": np.full(shape, np.clip(0.90 - 0.58 * stress, 0.08, 0.95)),
        "contagion_mobilization": np.full(shape, 0.22 + 0.25 * stress),
        "contagion_fatigue": np.full(shape, 0.12 + 0.50 * stress),
        "jump_signal": np.full(shape, jump),
    }


def standalone_fixture() -> dict:
    support = np.array([
        [0.62, 0.48, 0.55, 0.57],
        [0.58, 0.52, 0.51, 0.54],
        [0.60, 0.50, 0.53, 0.56],
    ])
    system = DifferentialManifoldConstraintV81(3, support)
    targets = support.copy()
    for month in range(18):
        stress = 0.16 if month < 4 else (0.84 if month < 11 else 0.24)
        jump = 0.75 if month == 7 else 0.0
        system.prepare(inputs(3, stress, jump))
        targets = np.clip(targets + (0.015 if stress > 0.5 else -0.008), 0.05, 0.95)
        system.retract_support(targets)
    return system.diagnostics()


def integrated_fixture() -> tuple[dict, object]:
    support = np.array([
        [0.62, 0.48, 0.55, 0.57],
        [0.58, 0.52, 0.51, 0.54],
    ])
    system = DifferentialManifoldMFGV81(
        2,
        81_013,
        support,
        advanced_graphon=True,
        graphon_mode="multilayer",
        partial_observation_geometry=True,
        network_geometry_enabled=True,
    )
    factors = None
    for month in range(3):
        factors = system.update(
            progress=np.array([
                [0.70, -0.62, 0.58, -0.54],
                [0.64, -0.56, 0.52, -0.48],
            ]),
            common_noise=np.full((2, 4), 0.04 + 0.01 * month),
            **inputs(2, 0.32 + 0.12 * month, 0.18 if month == 1 else 0.0),
        )
    assert factors is not None
    return system.diagnostics(), factors


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    standalone = standalone_fixture()
    integrated, factors = integrated_fixture()
    manifold = integrated["v81_differential_manifold"]
    checks = {
        "standalone_simplex": standalone["maximum_probability_simplex_residual"] <= 1.0e-12,
        "standalone_spd": standalone["minimum_spd_eigenvalue"] > 0.0,
        "standalone_slow_active": standalone["slow_manifold_updates"] == 18,
        "standalone_hybrid_transition": standalone["hybrid_chart_transition_count"] > 0,
        "standalone_jump_fast_exit": standalone["maximum_jump_fast_exit"] > 0.0,
        "standalone_three_basins": len(standalone["attractor_basin_shares"]) == 3,
        "integrated_mfg_converged": factors.converged and factors.fixed_point_residual <= 5.0e-4,
        "integrated_simplex_clock": manifold["simplex_updates"] == 3,
        "integrated_feedback_clock": manifold["feedback_updates"] == 3,
        "integrated_support_feedback": manifold["maximum_support_output_shift"] > 0.0,
        "integrated_spd": manifold["minimum_spd_eigenvalue"] > 0.0,
        "bounded_support": bool(np.all((factors.support > 0.0) & (factors.support < 1.0))),
        "no_probability_patch": not manifold["terminal_probability_adjustment"],
    }
    checks = {key: bool(value) for key, value in checks.items()}
    payload = {
        "model_version": "V8.1-DIFFERENTIAL-MANIFOLD-CONSTRAINTS",
        "classification": "DIAGNOSTIC_ONLY; no outcome probabilities",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "standalone_product_manifold": standalone,
        "integrated_product_manifold": manifold,
        "outcome_simulation_executed": False,
    }
    TARGET.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"TAIWAN_V81_DIAGNOSTIC_ONLY: {payload['status']}")
    if payload["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
