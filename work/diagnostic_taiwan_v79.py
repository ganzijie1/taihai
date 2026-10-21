from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from information_criticality_mfg_v79 import InformationCriticalityMFGV79
from information_criticality_v79 import (
    ConditionalTransferEntropyV79,
    QuasicriticalCascadeV79,
)
from renormalization_scale_audit_v79 import RenormalizationScaleAuditV79


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
TARGET = OUT / "taiwan_v79_diagnostic_only.json"


def transfer_entropy_fixture() -> dict:
    estimator = ConditionalTransferEntropyV79(minimum_periods=8)
    paths = 12
    source = np.linspace(-0.9, 0.9, paths)
    lagged = np.zeros(paths)
    for month in range(12):
        signals = np.zeros((paths, 4, 4))
        current = np.tanh(source + 0.08 * month)
        signals[:, 0, 0] = current
        signals[:, 0, 1] = lagged
        signals[:, 1, :] = 0.30 * current[:, None]
        signals[:, 2, :] = -0.20 * current[:, None]
        signals[:, 3, :] = 0.10 * current[:, None]
        estimator.observe(signals)
        lagged = current.copy()
    return estimator.diagnostics()


def cascade_fixture() -> tuple[dict, object]:
    kernel = np.array([
        [0.58, 0.10, 0.16, 0.16],
        [0.12, 0.54, 0.18, 0.16],
        [0.12, 0.16, 0.58, 0.14],
        [0.14, 0.18, 0.18, 0.50],
    ])
    cascade = QuasicriticalCascadeV79(3, kernel)
    shape = (3, 4)
    result = None
    for _ in range(3):
        result = cascade.update(
            damage=np.full(shape, 0.95),
            shortage=np.full(shape, 0.90),
            inflation=np.full(shape, 0.55),
            financial_stress=np.full(shape, 0.80),
            command=np.full(shape, 0.42),
            network=np.full(shape, 0.38),
            contagion_mobilization=np.full(shape, 0.65),
            contagion_fatigue=np.full(shape, 0.70),
            jump_signal=np.full(shape, 0.85),
            service_action=np.full(shape, 0.12),
        )
    assert result is not None
    return cascade.diagnostics(), result


def scale_fixture() -> dict:
    audit = RenormalizationScaleAuditV79(np.array([0.24, 0.36, 0.25, 0.15]))
    grid = np.linspace(-1.0, 1.0, 25)
    dx = grid[1] - grid[0]
    for month in range(10):
        center = np.tanh(
            0.12 * month
            + np.arange(4)[None, :, None] * 0.04
            + np.arange(4)[None, None, :] * 0.03
        )
        density = np.exp(
            -0.5 * np.square(
                (grid[None, None, None, :] - center[..., None]) / 0.28
            )
        )
        density = np.broadcast_to(density, (3, 4, 4, len(grid))).copy()
        density /= np.sum(density, axis=-1, keepdims=True) * dx
        audit.observe(density, grid)
    return audit.diagnostics()


def integrated_fixture() -> tuple[dict, object]:
    support = np.array([
        [0.62, 0.48, 0.55, 0.57],
        [0.58, 0.52, 0.51, 0.54],
    ])
    system = InformationCriticalityMFGV79(
        2, 790_013, support,
        advanced_graphon=True,
        graphon_mode="multilayer",
        partial_observation_geometry=True,
        network_geometry_enabled=True,
    )
    shape = (2, 4)
    factors = system.update(
        progress=np.array([
            [0.70, -0.62, 0.58, -0.54],
            [0.64, -0.56, 0.52, -0.48],
        ]),
        damage=np.full(shape, 0.18),
        shortage=np.full(shape, 0.24),
        inflation=np.full(shape, 0.18),
        financial_stress=np.full(shape, 0.18),
        command=np.full(shape, 0.72),
        network=np.full(shape, 0.72),
        contagion_mobilization=np.full(shape, 0.20),
        contagion_fatigue=np.full(shape, 0.16),
        common_noise=np.full(shape, 0.05),
        jump_signal=np.full(shape, 0.08),
    )
    return system.diagnostics(), factors


def main() -> None:
    te = transfer_entropy_fixture()
    cascade, cascade_result = cascade_fixture()
    scale = scale_fixture()
    integrated, factors = integrated_fixture()
    info = integrated["partial_observation"][
        "v79_correlated_rate_distortion_information"
    ]
    flow = integrated["v79_information_flow"]
    integrated_cascade = integrated["v79_quasicritical_cascade"]
    checks = {
        "transfer_entropy_estimator_active": te["fit_updates"] > 0
        and np.isfinite(te["maximum_transfer_entropy"]),
        "transfer_entropy_lag_explicit": "next month" in te["causal_clock"],
        "cascade_triggered": cascade_result.event_count > 0,
        "cascade_bounded": all(
            np.all((value >= 0.0) & (value <= 1.0))
            for value in (
                cascade_result.social, cascade_result.financial,
                cascade_result.supply, cascade_result.command,
            )
        ),
        "cascade_mass_balance": cascade[
            "maximum_fast_mass_balance_residual"
        ] <= 1.0e-10,
        "scale_mass": scale["maximum_conditional_mass_residual"] <= 1.0e-10,
        "scale_conservation": scale[
            "maximum_scale_conservation_residual"
        ] <= 1.0e-10,
        "scale_jacobian_fitted": scale["jacobian_fit_updates"] > 0,
        "integrated_mfg_converged": factors.converged
        and factors.fixed_point_residual <= 5.0e-4,
        "correlated_information_active": info["updates"] == 1
        and info["maximum_budget_residual"] <= 1.0e-10,
        "cascade_contract_returned": factors.cascade_social_loss is not None
        and factors.cascade_recovery_need is not None,
        "same_path_feedback_active": integrated_cascade[
            "feedback_updates"
        ] == 1,
        "graphon_information_interface": flow["graphon_feedback"],
        "no_probability_patch": not info["terminal_probability_adjustment"]
        and not flow["terminal_probability_adjustment"]
        and not integrated_cascade["terminal_probability_adjustment"]
        and not scale["terminal_probability_adjustment"],
    }
    checks = {name: bool(value) for name, value in checks.items()}
    payload = {
        "model_version": "V7.9-INFORMATION-CRITICALITY-SCALE",
        "classification": "DIAGNOSTIC_ONLY; no outcome probabilities",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "transfer_entropy_fixture": te,
        "cascade_fixture": cascade,
        "scale_fixture": scale,
        "integrated_information": info,
        "integrated_information_flow": flow,
        "integrated_cascade": integrated_cascade,
        "integrated_scale": integrated["v79_renormalization_scale_audit"],
        "outcome_simulation_executed": False,
    }
    TARGET.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"TAIWAN_V79_DIAGNOSTIC_ONLY: {payload['status']}")
    if payload["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
