"""DIAGNOSTIC_ONLY: local invariants for the V6.0 Riemannian upgrade."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import simulate_taiwan_multidomain_v42 as battle
from continuous_multipop_mfg_v47 import ContinuousMultiPopulationMFG
from geometric_multiscale_v50 import GeometricMultiscaleCoupler
from operational_constraints_v47 import DynamicGeospatialPNTSystem


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "outputs" / "taiwan_v60_riemannian_diagnostic.json"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def dataclass_difference(left, right) -> float:
    differences = []
    for field in left.__dataclass_fields__:
        a = getattr(left, field)
        b = getattr(right, field)
        if isinstance(a, np.ndarray):
            differences.append(float(np.max(np.abs(a.astype(float) - b.astype(float)))))
        elif isinstance(a, (float, int, np.floating, np.integer)):
            differences.append(abs(float(a) - float(b)))
    return max(differences, default=0.0)


def main() -> None:
    paths = 4
    rng = np.random.default_rng(60_001)
    _, _, terrain, _ = battle.load_inputs()

    infrastructure = rng.uniform(0.52, 0.96, (paths, 4))
    cyber = rng.uniform(0.0, 0.42, (paths, 4))
    command = rng.uniform(0.48, 0.94, (paths, 4))
    terrain_mask = rng.uniform(0.38, 0.96, (paths, 4))

    legacy_pnt_a = DynamicGeospatialPNTSystem(
        paths, terrain, geometry_enabled=True,
        geometry_metric="bures_wasserstein", joseph_update=True,
    )
    legacy_pnt_b = DynamicGeospatialPNTSystem(
        paths, terrain, geometry_enabled=True,
        geometry_metric="bures_wasserstein", joseph_update=True,
        adaptive_bures_enabled=False, bures_barycenter_enabled=False,
    )
    legacy_pnt_difference = dataclass_difference(
        legacy_pnt_a.update(
            infrastructure=infrastructure, cyber_damage=cyber,
            command=command, terrain_mask=terrain_mask,
        ),
        legacy_pnt_b.update(
            infrastructure=infrastructure, cyber_damage=cyber,
            command=command, terrain_mask=terrain_mask,
        ),
    )

    pnt = DynamicGeospatialPNTSystem(
        paths, terrain, geometry_enabled=True,
        geometry_metric="bures_wasserstein", joseph_update=True,
        adaptive_bures_enabled=True, bures_barycenter_enabled=True,
    )
    pnt_factor = None
    for month in range(3):
        pnt_factor = pnt.update(
            infrastructure=np.clip(infrastructure - 0.03 * month, 0.25, 1.0),
            cyber_damage=np.clip(cyber + 0.04 * month, 0.0, 0.8),
            command=command, terrain_mask=terrain_mask,
        )
    pnt_diag = pnt.diagnostics()

    legacy_manifold_a = GeometricMultiscaleCoupler(paths, learned_slow_manifold=True)
    legacy_manifold_b = GeometricMultiscaleCoupler(
        paths, learned_slow_manifold=True,
        regime_riemannian_pod=False, product_manifold_metric=False,
    )
    front = rng.uniform(0.42, 0.96, (paths, 4, 3))
    sector = rng.uniform(0.44, 1.06, (paths, 4, 10))
    financial = rng.uniform(0.42, 0.98, (paths, 4))
    energy = rng.uniform(0.40, 0.98, (paths, 4))
    legacy_manifold_difference = dataclass_difference(
        legacy_manifold_a.update(
            front_rear_integrity=front, sector_output=sector,
            financial_capacity=financial, command=command,
            energy_logistics=energy,
        ),
        legacy_manifold_b.update(
            front_rear_integrity=front, sector_output=sector,
            financial_capacity=financial, command=command,
            energy_logistics=energy,
        ),
    )

    manifold = GeometricMultiscaleCoupler(
        paths, learned_slow_manifold=True,
        regime_riemannian_pod=True, product_manifold_metric=True,
    )
    for month, level in enumerate((0.88, 0.70, 0.44, 0.60, 0.76)):
        gradient = np.linspace(-0.10, 0.10, paths)[:, None]
        actor_level = np.repeat(
            np.clip(level + gradient, 0.12, 1.05), 4, axis=1
        )
        manifold.update(
            front_rear_integrity=np.repeat(actor_level[:, :, None], 3, axis=2),
            sector_output=np.repeat(actor_level[:, :, None], 10, axis=2),
            financial_capacity=actor_level,
            command=actor_level,
            energy_logistics=actor_level,
        )
    manifold_diag = manifold.diagnostics()

    support = rng.uniform(0.36, 0.68, (paths, 4))
    mfg = ContinuousMultiPopulationMFG(
        paths, 60_401, support,
        grid_points=25, numerical_backend="numpy", anderson_depth=3,
        geometry_enabled=True, advanced_graphon=True, graphon_mode="multilayer",
        fp_backend="numpy_fvm", geomloss_interval=99,
        wfr_strang=False, natural_gradient_major=True,
        partial_observation_geometry=True, full_wfr_jko=True,
        trust_region_enabled=True, network_geometry_enabled=True,
        riemannian_anderson_enabled=True,
        riemannian_anderson_regularization=1.0e-5,
        riemannian_trust_region_enabled=True,
    )
    mfg_factor = mfg.update(
        progress=rng.uniform(-0.20, 0.20, (paths, 4)),
        damage=rng.uniform(0.0, 0.35, (paths, 4)),
        shortage=rng.uniform(0.0, 0.40, (paths, 4)),
        inflation=rng.uniform(0.0, 0.18, (paths, 4)),
        financial_stress=rng.uniform(0.0, 0.42, (paths, 4)),
        command=command,
        network=rng.uniform(0.38, 0.90, (paths, 4)),
        contagion_mobilization=rng.uniform(0.10, 0.58, (paths, 4)),
        contagion_fatigue=rng.uniform(0.10, 0.55, (paths, 4)),
        common_noise=rng.uniform(-0.28, 0.28, (paths, 4)),
        jump_signal=rng.uniform(0.0, 0.25, (paths, 4)),
    )
    mfg_diag = mfg.diagnostics()
    ram = mfg_diag["riemannian_anderson"]
    major_trust = mfg_diag["major_riemannian_trust_region"]
    minor_trust = mfg_diag["minor_riemannian_trust_region"]

    checks = {
        "legacy_pnt_default_regression": legacy_pnt_difference == 0.0,
        "legacy_manifold_default_regression": legacy_manifold_difference == 0.0,
        "pnt_positive_definite": pnt_diag["minimum_covariance_eigenvalue"] > 0.0,
        "pnt_inverse_residual": pnt_factor.information_residual < 1.0e-8,
        "pnt_barycenter_converged": pnt_diag["max_barycenter_residual"] < 5.0e-7,
        "pnt_source_weights": pnt_diag["max_source_weight_residual"] < 1.0e-12,
        "pnt_adaptive_weight": (
            0.0 < pnt_diag["minimum_adaptive_weight"]
            <= pnt_diag["maximum_adaptive_weight"] <= 1.0
        ),
        "product_metric_positive": manifold_diag["minimum_product_metric_eigenvalue"] > 0.0,
        "regime_pod_exercised": manifold_diag["regime_transition_count"] > 0,
        "regime_reconstruction_finite": np.isfinite(
            manifold_diag["maximum_geodesic_reconstruction_error"]
        ),
        "mfg_converged": mfg_factor.converged,
        "mfg_mass": mfg_diag["max_mass_residual"] < 1.0e-10,
        "ram_exercised": ram is not None and ram["attempted"] > 0,
        "ram_tangent": ram["maximum_transport_tangency_residual"] < 1.0e-10,
        "ram_sphere": ram["maximum_sphere_residual"] < 1.0e-10,
        "major_trust_exercised": major_trust["calls"] > 0,
        "minor_trust_exercised": minor_trust["calls"] > 0,
        "major_trust_radius": 0.0 < major_trust["current_radius"] <= 0.14,
        "minor_trust_radius": 0.0 < minor_trust["current_radius"] <= 0.10,
    }
    checks = {key: bool(value) for key, value in checks.items()}
    payload = {
        "classification": "DIAGNOSTIC_ONLY; no outcome probabilities",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "legacy_regression": {
            "pnt_max_difference": legacy_pnt_difference,
            "manifold_max_difference": legacy_manifold_difference,
        },
        "pnt": pnt_diag,
        "multiscale": manifold_diag,
        "mfg": mfg_diag,
    }
    TARGET.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"TAIWAN_V60_RIEMANNIAN_DIAGNOSTIC: {payload['status']}")
    if payload["status"] != "PASS":
        print(json.dumps({key: value for key, value in checks.items() if not value}, indent=2))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
