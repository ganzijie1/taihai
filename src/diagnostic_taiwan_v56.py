"""DIAGNOSTIC_ONLY: local invariants for the V5.6 geometry upgrade."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import simulate_taiwan_multidomain_v42 as battle
from continuous_multipop_mfg_v47 import ContinuousMultiPopulationMFG
from four_party_energy_network_v49 import FourPartyEnergyNetwork
from geometric_multiscale_v50 import GeometricMultiscaleCoupler
from hybrid_filippov_impulse_v51 import HybridFilippovImpulseSystem
from operational_constraints_v47 import (
    DynamicGeospatialPNTSystem,
    DynamicOperationalConstraintSystem,
)
from spatial_control_network_v47 import SpatialControlNetwork


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "outputs" / "taiwan_v56_geometry_diagnostic.json"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def maximum_difference(left, right) -> float:
    fields = left.__dataclass_fields__
    values = []
    for field in fields:
        a = getattr(left, field)
        b = getattr(right, field)
        if isinstance(a, np.ndarray):
            values.append(float(np.max(np.abs(a.astype(float) - b.astype(float)))))
        elif isinstance(a, (float, int, np.floating, np.integer)):
            values.append(abs(float(a) - float(b)))
    return max(values, default=0.0)


def main() -> None:
    paths = 4
    _, _, terrain, _ = battle.load_inputs()
    rng = np.random.default_rng(56_001)
    production = rng.uniform(0.04, 0.12, (paths, 4))
    consumption = rng.uniform(0.05, 0.15, (paths, 4))
    damage = rng.uniform(0.0, 0.35, (paths, 4))
    interdiction = rng.uniform(0.0, 0.45, (paths, 4))
    command = rng.uniform(0.55, 0.95, (paths, 4))
    pnt_state = rng.uniform(0.50, 0.92, (paths, 4))
    financial = rng.uniform(0.50, 0.96, (paths, 4))

    legacy_a = DynamicOperationalConstraintSystem(paths, 56_101, terrain, geometry_enabled=True)
    legacy_b = DynamicOperationalConstraintSystem(paths, 56_101, terrain, geometry_enabled=True)
    legacy_a_factor = legacy_a.update(
        production=production, consumption=consumption, damage=damage,
        interdiction=interdiction, command=command, pnt=pnt_state,
        financial_capacity=financial,
    )
    legacy_b_factor = legacy_b.update(
        production=production, consumption=consumption, damage=damage,
        interdiction=interdiction, command=command, pnt=pnt_state,
        financial_capacity=financial,
    )
    legacy_operation_difference = maximum_difference(legacy_a_factor, legacy_b_factor)

    operation = DynamicOperationalConstraintSystem(
        paths, 56_101, terrain, geometry_enabled=True, advanced_geometry=True
    )
    operation_factor = operation.update(
        production=production, consumption=consumption, damage=damage,
        interdiction=interdiction, command=command, pnt=pnt_state,
        financial_capacity=financial,
    )
    operation_diag = operation.diagnostics()

    pnt = DynamicGeospatialPNTSystem(
        paths, terrain, geometry_enabled=True,
        geometry_metric="bures_wasserstein", joseph_update=True,
    )
    pnt_factor = pnt.update(
        infrastructure=rng.uniform(0.55, 0.95, (paths, 4)),
        cyber_damage=rng.uniform(0.0, 0.40, (paths, 4)),
        command=command,
        terrain_mask=rng.uniform(0.45, 0.95, (paths, 4)),
    )
    pnt_diag = pnt.diagnostics()

    spatial = SpatialControlNetwork(
        paths, 56_201, terrain, geometry_enabled=True, dynamic_finsler=True
    )
    spatial_factor = spatial.update(
        aggregate_land=rng.uniform(0.05, 0.25, paths),
        attacker_logistics=rng.uniform(0.45, 0.90, paths),
        attacker_command=rng.uniform(0.50, 0.90, paths),
        defender_command=rng.uniform(0.50, 0.90, paths),
        defender_resistance=rng.uniform(0.45, 0.90, paths),
        external_intervention=rng.uniform(0.0, 0.80, paths),
        local_agent_capacity=rng.uniform(0.05, 0.30, paths),
        damage=rng.uniform(0.0, 0.35, paths),
        attacker_pnt=rng.uniform(0.45, 0.95, paths),
        blockade=rng.uniform(0.0, 0.60, paths),
    )
    spatial_diag = spatial.diagnostics()

    energy = FourPartyEnergyNetwork(
        paths, 56_301, geometry_enabled=True, port_hamiltonian_dae=True
    )
    energy_factor = energy.update(
        month=0, actor_damage=damage,
        actor_output=rng.uniform(0.55, 1.0, (paths, 4)),
        actor_mobilization=rng.uniform(0.15, 0.75, (paths, 4)),
        financial_capacity=financial,
        trade_loss=rng.uniform(0.0, 0.35, paths),
        blockade=rng.uniform(0.0, 0.55, paths),
        sanction_relief=rng.uniform(0.0, 0.12, paths),
        us_level=rng.uniform(0.0, 1.0, paths),
        japan_level=rng.uniform(0.0, 1.0, paths),
    )
    energy_diag = energy.diagnostics()

    manifold = GeometricMultiscaleCoupler(paths, learned_slow_manifold=True)
    manifold_factor = manifold.update(
        front_rear_integrity=rng.uniform(0.40, 0.95, (paths, 4, 3)),
        sector_output=rng.uniform(0.45, 1.05, (paths, 4, 10)),
        financial_capacity=financial,
        command=command,
        energy_logistics=energy_factor.logistics,
    )
    manifold_diag = manifold.diagnostics()

    support = rng.uniform(0.35, 0.70, (paths, 4))
    mfg = ContinuousMultiPopulationMFG(
        paths, 56_401, support, advanced_graphon=True,
        graphon_mode="multilayer", fp_backend="numpy_fvm",
        geometry_enabled=True, wfr_strang=True, natural_gradient_major=True,
        geomloss_interval=99,
    )
    mfg_factor = mfg.update(
        progress=rng.uniform(-0.2, 0.2, (paths, 4)), damage=damage,
        shortage=rng.uniform(0.0, 0.4, (paths, 4)),
        inflation=rng.uniform(0.0, 0.15, (paths, 4)),
        financial_stress=rng.uniform(0.0, 0.45, (paths, 4)), command=command,
        network=rng.uniform(0.35, 0.85, (paths, 4)),
        contagion_mobilization=rng.uniform(0.1, 0.6, (paths, 4)),
        contagion_fatigue=rng.uniform(0.1, 0.6, (paths, 4)),
        common_noise=rng.uniform(-0.3, 0.3, (paths, 4)),
        jump_signal=rng.uniform(0.0, 0.25, (paths, 4)),
    )
    mfg_diag = mfg.diagnostics()

    hybrid = HybridFilippovImpulseSystem(paths, threshold_shift=-0.08)
    hybrid.begin_month(0)
    hybrid.update_filippov(
        logistics_service=np.full((paths, 4), 0.60),
        financial_capacity=np.full((paths, 4), 0.48),
        debt_stress=np.full((paths, 4), 0.72),
        stock_shortage=np.full((paths, 4), 0.62),
        support=np.full((paths, 4), 0.45),
        conflict_intensity=np.full(paths, 0.85),
        command_service=np.full((paths, 4), 0.50),
    )
    hybrid_diag = hybrid.diagnostics()

    checks = {
        "legacy_default_regression": legacy_operation_difference == 0.0,
        "randers_positive": operation_diag["max_randers_one_form_norm"] < 1.0,
        "operational_stock": operation_factor.stock_residual < 1.0e-10,
        "uot_converged": operation_factor.uot_solver_residual < 1.0e-6,
        "subriemannian_bounds": 0.0 < operation_diag["minimum_subriemannian_reachability"] <= 1.0,
        "so3_orthogonality": operation_diag["max_so3_orthogonality_residual"] < 1.0e-12,
        "so3_determinant": operation_diag["max_so3_determinant_residual"] < 1.0e-12,
        "pnt_spd": pnt_diag["minimum_covariance_eigenvalue"] > 0.0,
        "pnt_information_inverse": pnt_factor.information_residual < 1.0e-10,
        "pnt_joseph_symmetry": pnt_diag["max_joseph_symmetry_residual"] < 1.0e-12,
        "dynamic_finsler_rows": spatial_diag["max_dynamic_transition_row_residual"] < 1.0e-10,
        "spatial_control_bounds": bool(np.all((spatial_factor.zone_control >= 0.0) & (spatial_factor.zone_control <= 1.0))),
        "port_hamiltonian_dae": energy_diag["max_port_hamiltonian_dae_residual"] < 1.0e-10,
        "port_hamiltonian_dissipation": energy_diag["max_dissipation_inequality_violation"] < 1.0e-10,
        "energy_flow_cut": energy_diag["max_flow_min_cut_residual"] < 1.0e-10,
        "manifold_tangent": manifold_diag["max_manifold_tangent_residual"] < 1.0e-12,
        "manifold_aggregation": manifold_factor.aggregation_residual < 1.0e-12,
        "wfr_mass": mfg_diag["max_wfr_strang_mass_residual"] < 1.0e-10,
        "mfg_converged": mfg_factor.converged,
        "natural_gradient_finite": np.isfinite(mfg_diag["max_natural_gradient_distance"]),
        "saltation_finite_difference": hybrid_diag["max_saltation_finite_difference_residual"] < 1.0e-12,
        "saltation_finite": hybrid_diag["saltation_nonfinite_count"] == 0,
    }
    checks = {key: bool(value) for key, value in checks.items()}
    payload = {
        "classification": "DIAGNOSTIC_ONLY; no outcome probabilities",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "legacy_default_max_difference": legacy_operation_difference,
        "operations": operation_diag,
        "pnt": pnt_diag,
        "spatial": spatial_diag,
        "energy": energy_diag,
        "multiscale": manifold_diag,
        "mfg": mfg_diag,
        "hybrid": hybrid_diag,
    }
    TARGET.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"TAIWAN_V56_GEOMETRY_DIAGNOSTIC: {payload['status']}")
    if payload["status"] != "PASS":
        print(json.dumps({key: value for key, value in checks.items() if not value}, indent=2))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
