"""DIAGNOSTIC_ONLY: structure-preserving geometry fixtures for V5.0.

This file must not compute campaign outcomes or publish probabilities.
"""

from __future__ import annotations

import numpy as np

from continuous_multipop_mfg_v47 import ContinuousMultiPopulationMFG
from four_party_energy_network_v49 import FourPartyEnergyNetwork
from geometric_multiscale_v50 import GeometricMultiscaleCoupler
from operational_constraints_v47 import (
    DynamicGeospatialPNTSystem,
    DynamicOperationalConstraintSystem,
)
from spatial_control_network_v47 import SpatialControlNetwork


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    paths = 6
    terrain = {
        "land_share_above_500m": 0.42,
        "sky_view_proxy_p10_p50_p90": [0.31, 0.58, 0.83],
    }
    support = np.full((paths, 4), 0.55)
    zero = np.zeros((paths, 4))
    one = np.ones((paths, 4))

    # Zero-geometry regression: explicit False must remain bitwise-equivalent
    # to the accepted V4.9 default constructor path.
    legacy_mfg = ContinuousMultiPopulationMFG(paths, 4999, support)
    explicit_mfg = ContinuousMultiPopulationMFG(
        paths, 4999, support, geometry_enabled=False
    )
    legacy_args = dict(
        progress=zero, damage=zero, shortage=zero, inflation=zero,
        financial_stress=zero, command=one * 0.8, network=one * 0.7,
        contagion_mobilization=one * 0.5, contagion_fatigue=one * 0.2,
    )
    legacy_factor = legacy_mfg.update(**legacy_args)
    explicit_factor = explicit_mfg.update(**legacy_args)
    require(np.array_equal(legacy_mfg.density, explicit_mfg.density),
            "V4.9 zero-geometry MFG regression")
    require(np.array_equal(legacy_factor.support, explicit_factor.support),
            "V4.9 zero-geometry support regression")

    mfg = ContinuousMultiPopulationMFG(
        paths, 5001, support, geometry_enabled=True
    )
    mf = mfg.update(
        progress=zero, damage=zero, shortage=zero, inflation=zero,
        financial_stress=zero, command=one * 0.8, network=one * 0.7,
        contagion_mobilization=one * 0.5, contagion_fatigue=one * 0.2,
    )
    require(mf.mass_residual < 1.0e-10, "Fisher-Rao simplex mass")
    require(np.isfinite(mf.fisher_rao_distance), "Fisher-Rao distance")

    pnt = DynamicGeospatialPNTSystem(paths, terrain, geometry_enabled=True)
    pf = pnt.update(
        infrastructure=one * 0.85, cyber_damage=one * 0.1,
        command=one * 0.8, terrain_mask=one * 0.58,
    )
    require(pnt.diagnostics()["minimum_covariance_eigenvalue"] > 0.0, "SPD positivity")
    require(np.isfinite(pf.spd_geodesic_distance), "SPD geodesic distance")

    operations = DynamicOperationalConstraintSystem(
        paths, 5002, terrain, geometry_enabled=True
    )
    of = operations.update(
        production=one * 0.06, consumption=one * 0.08,
        damage=one * 0.04, interdiction=one * 0.15,
        command=one * 0.8, pnt=pf.integrity,
        financial_capacity=one * 0.75,
    )
    require(of.stock_residual < 1.0e-8, "operational stock balance")
    require(of.uot_solver_residual < 1.0e-5, "unbalanced Sinkhorn convergence")
    require(np.all((of.finsler_reachability > 0.0) & (of.finsler_reachability <= 1.0)),
            "Finsler reachability bounds")

    energy = FourPartyEnergyNetwork(paths, 5003, geometry_enabled=True)
    ef = energy.update(
        month=0, actor_damage=one * 0.03, actor_output=one * 0.85,
        actor_mobilization=one * 0.35, financial_capacity=one * 0.75,
        trade_loss=np.full(paths, 0.15), blockade=np.full(paths, 0.20),
        sanction_relief=np.zeros(paths), us_level=np.full(paths, 0.7),
        japan_level=np.full(paths, 0.6),
    )
    ed = energy.diagnostics()
    require(ed["max_hodge_divergence_residual"] < 1.0e-10, "Hodge divergence")
    require(ed["max_dirac_interconnection_residual"] < 1.0e-12, "Dirac antisymmetry")
    require(ed["max_port_hamiltonian_residual"] < 1.0e-12, "port-Hamiltonian balance")
    require(np.all((ef.graph_resilience > 0.0) & (ef.graph_resilience <= 1.0)),
            "graph resilience bounds")

    spatial = SpatialControlNetwork(paths, 5004, terrain, geometry_enabled=True)
    sf = spatial.update(
        aggregate_land=np.full(paths, 0.15), attacker_logistics=one[:, 0] * 0.7,
        attacker_command=one[:, 0] * 0.75, defender_command=one[:, 0] * 0.72,
        defender_resistance=one[:, 0] * 0.68,
        external_intervention=one[:, 0] * 0.55,
        local_agent_capacity=one[:, 0] * 0.18, damage=one[:, 0] * 0.06,
        attacker_pnt=pf.integrity[:, 0], blockade=one[:, 0] * 0.25,
    )
    require(sf.graph_spectral_gap > 0.0, "spatial graph connectivity")
    require(np.all((sf.finsler_reachability > 0.0) & (sf.finsler_reachability <= 1.0)),
            "spatial Finsler bounds")

    multiscale = GeometricMultiscaleCoupler(paths)
    gf = multiscale.update(
        front_rear_integrity=of.echelon_integrity,
        sector_output=np.ones((paths, 4, 5)) * 0.82,
        financial_capacity=one * 0.76, command=one * 0.74,
        energy_logistics=ef.logistics,
    )
    require(gf.aggregation_residual < 1.0e-12, "multiscale reconstruction")
    require(np.all((gf.consistency > 0.0) & (gf.consistency <= 1.0)),
            "multiscale consistency bounds")
    print("DIAGNOSTIC_ONLY_TAIWAN_V50: PASS")


if __name__ == "__main__":
    main()
