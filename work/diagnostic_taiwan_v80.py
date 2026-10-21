from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from analytical_mechanics_v80 import AnalyticalMechanicsV80
from statistical_field_mfg_v80 import StatisticalFieldMFGV80
from statistical_field_soc_v80 import (
    NonEquilibriumStatisticalFieldV80,
    QuasicriticalCascadeV80,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
TARGET = OUT / "taiwan_v80_diagnostic_only.json"
KERNEL = np.array([
    [0.58, 0.10, 0.16, 0.16],
    [0.12, 0.54, 0.18, 0.16],
    [0.12, 0.16, 0.58, 0.14],
    [0.14, 0.18, 0.18, 0.50],
])


def inputs(paths: int) -> dict:
    shape = (paths, 4)
    return {
        "damage": np.full(shape, 0.38),
        "shortage": np.full(shape, 0.42),
        "inflation": np.full(shape, 0.22),
        "financial_stress": np.full(shape, 0.35),
        "command": np.full(shape, 0.64),
        "network": np.full(shape, 0.61),
        "contagion_mobilization": np.full(shape, 0.24),
        "contagion_fatigue": np.full(shape, 0.28),
        "jump_signal": np.full(shape, 0.16),
    }


def cascade_fixture() -> dict:
    cascade = QuasicriticalCascadeV80(4, KERNEL)
    kw = inputs(4)
    for month in range(14):
        cascade.update(
            **kw,
            service_action=np.full((4, 4), 0.10 + 0.002 * month),
        )
    return cascade.diagnostics()


def field_fixture() -> dict:
    field = NonEquilibriumStatisticalFieldV80(
        4, KERNEL, np.array([0.24, 0.36, 0.25, 0.15]), 80_101
    )
    kw = inputs(4)
    mfg_mean = np.zeros((4, 4, 4))
    cascade_loss = np.full((4, 4, 4), 0.08)
    for month in range(14):
        local = dict(kw)
        local["jump_signal"] = np.full((4, 4), 0.08 + 0.01 * (month % 3))
        field.update(
            **local, mfg_mean=mfg_mean, cascade_loss=cascade_loss
        )
    return field.diagnostics()


def mechanics_fixture() -> dict:
    mechanics = AnalyticalMechanicsV80(4)
    kw = inputs(4)
    for _ in range(14):
        mechanics.update(**kw)
    return mechanics.diagnostics()


def integrated_fixture() -> tuple[dict, object]:
    support = np.array([
        [0.62, 0.48, 0.55, 0.57],
        [0.58, 0.52, 0.51, 0.54],
    ])
    system = StatisticalFieldMFGV80(
        2, 80_013, support,
        advanced_graphon=True,
        graphon_mode="multilayer",
        partial_observation_geometry=True,
        network_geometry_enabled=True,
    )
    kw = inputs(2)
    factors = None
    for month in range(2):
        factors = system.update(
            progress=np.array([
                [0.70, -0.62, 0.58, -0.54],
                [0.64, -0.56, 0.52, -0.48],
            ]),
            common_noise=np.full((2, 4), 0.05 + 0.01 * month),
            **kw,
        )
    assert factors is not None
    return system.diagnostics(), factors


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cascade = cascade_fixture()
    field = field_fixture()
    mechanics = mechanics_fixture()
    integrated, factors = integrated_fixture()
    integrated_field = integrated["v80_non_equilibrium_statistical_field"]
    integrated_mechanics = integrated["v80_analytical_mechanics"]
    evidence = cascade["v80_soc_evidence"]
    checks = {
        "soc_events_observed": evidence["event_sample_size"] >= 8,
        "soc_claim_bounded": evidence["claim_boundary"]
        == "quasi-critical cascade, not confirmed SOC",
        "field_dmft_fitted": field["dmft"]["fit_updates"] > 0,
        "field_memory_stable": field["dmft"][
            "maximum_memory_spectral_radius"
        ] < 1.0,
        "field_covariance_psd": field["dmft"][
            "innovation_covariance_minimum_eigenvalue"
        ] >= -1.0e-10,
        "field_cavity_resolved": field[
            "cavity_equilibrium_resolve_updates"
        ] > 0,
        "field_finite_scale_statistics": all(
            np.isfinite(value)
            for value in field["binder_cumulant_by_resolution"].values()
        ),
        "mechanics_constraint": mechanics[
            "maximum_constraint_residual"
        ] <= 1.0e-12,
        "mechanics_dirac_zero_power": mechanics[
            "maximum_dirac_power_residual"
        ] <= 1.0e-12,
        "mechanics_balance_finite": np.isfinite(
            mechanics["maximum_discrete_energy_balance_residual"]
        ),
        "integrated_mfg_converged": factors.converged
        and factors.fixed_point_residual <= 5.0e-4,
        "lagged_field_feedback_active": integrated_field[
            "maximum_lagged_mfg_input_shift"
        ] > 0.0,
        "current_outer_feedback_active": integrated_field[
            "maximum_current_outer_factor_contribution"
        ] > 0.0,
        "mechanics_same_path_active": integrated_mechanics["updates"] == 2,
        "no_probability_patch": not field["terminal_probability_adjustment"]
        and not mechanics["terminal_probability_adjustment"]
        and not evidence["terminal_probability_adjustment"],
    }
    checks = {key: bool(value) for key, value in checks.items()}
    payload = {
        "model_version": "V8.0-STATISTICAL-FIELD-ANALYTICAL-MECHANICS",
        "classification": "DIAGNOSTIC_ONLY; no outcome probabilities",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "soc_fixture": cascade,
        "statistical_field_fixture": field,
        "analytical_mechanics_fixture": mechanics,
        "integrated_statistical_field": integrated_field,
        "integrated_analytical_mechanics": integrated_mechanics,
        "outcome_simulation_executed": False,
    }
    TARGET.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"TAIWAN_V80_DIAGNOSTIC_ONLY: {payload['status']}")
    if payload["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
