from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from complete_theory_mfg_v77 import CompleteTheoryMFGV77
from nested_ces_cge_v77 import NestedCESCGEV77


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "outputs" / "taiwan_v77_diagnostic_only.json"


def main() -> None:
    paths = 4
    support = np.array([
        [0.62, 0.48, 0.55, 0.57],
        [0.58, 0.52, 0.51, 0.54],
        [0.60, 0.46, 0.56, 0.53],
        [0.56, 0.50, 0.49, 0.55],
    ])
    system = CompleteTheoryMFGV77(
        paths, 770_013, support,
        advanced_graphon=True,
        graphon_mode="multilayer",
        partial_observation_geometry=True,
        network_geometry_enabled=True,
    )
    damage = np.full((paths, 4), 0.18)
    shortage = np.full((paths, 4), 0.24)
    network = np.full((paths, 4), 0.72)
    noise = np.full((paths, 4), 0.05)
    jump = np.full((paths, 4), 0.08)
    signal = np.array([
        [0.70, -0.62, 0.58, -0.54],
        [0.64, -0.56, 0.52, -0.48],
        [0.60, -0.48, 0.45, -0.42],
        [0.55, -0.40, 0.38, -0.35],
    ])
    kernel, weights, intensity = system._effective_graphon(
        damage, shortage, network, noise, jump, signal
    )
    information = system.belief_system.update(
        progress=signal,
        damage=damage,
        shortage=shortage,
        command=network,
        network=network,
        financial_stress=damage,
        inflation=damage,
        common_noise=noise,
        jump_signal=jump,
    )
    for month in range(1, 6):
        drifting_signal = np.clip(signal + 0.015 * month * np.sign(signal), -0.9, 0.9)
        information = system.belief_system.update(
            progress=drifting_signal,
            damage=damage + 0.005 * month,
            shortage=shortage + 0.004 * month,
            command=network,
            network=network - 0.003 * month,
            financial_stress=damage + 0.004 * month,
            inflation=damage + 0.003 * month,
            common_noise=noise,
            jump_signal=jump,
        )
    mean, _ = system._moments(system.density)
    system.online_tda.update(
        month=12,
        mean=mean,
        joint_action=system.previous_joint_action,
        signed_kernel=system._last_signed_kernel,
        layer_weights=system._last_layer_weights,
        hyper_tail=system.hyper_tail,
        hyper_head=system.hyper_head,
        major_action=system.major_action,
        hyperedge_weight=system.current_hyperedge_weight,
    )
    shifted = mean.copy()
    shifted[:, 0] = np.clip(shifted[:, 0] + 0.20, -1.0, 1.0)
    system.online_tda.update(
        month=24,
        mean=shifted,
        joint_action=system.previous_joint_action,
        signed_kernel=system._last_signed_kernel,
        layer_weights=system._last_layer_weights,
        hyper_tail=system.hyper_tail,
        hyper_head=system.hyper_head,
        major_action=system.major_action,
        hyperedge_weight=system.current_hyperedge_weight,
    )
    cge = NestedCESCGEV77(paths, 770_701, "limited")
    ones = np.ones((paths, cge.n_nodes))
    zeros = np.zeros((paths, cge.n_nodes))
    cge._production_capacity(ones, ones, ones, zeros)

    mfg_diag = system.diagnostics()
    belief_diag = system.belief_system.diagnostics()[
        "v77_information_bottleneck_active_observation"
    ]
    network_diag = system.network_geometry.diagnostics()[
        "v77_complete_network_geometry"
    ]
    tda_diag = system.online_tda.diagnostics()
    oscillator_diag = system.oscillator.diagnostics()
    ces_diag = cge.extension_diagnostics()["v77_nested_ces"]
    checks = {
        "finite_graphon": bool(
            np.all(np.isfinite(kernel))
            and np.max(np.abs(np.sum(kernel, axis=3) - 1.0)) <= 1.0e-12
            and np.all(np.isfinite(intensity))
        ),
        "information_state_changes": belief_diag["maximum_bottleneck_state_change"] > 0.0,
        "information_bottleneck_converges": belief_diag["maximum_ib_fixed_point_residual"] <= 1.0e-7,
        "information_bottleneck_objective_monotone": belief_diag[
            "maximum_ib_objective_increase"
        ] <= 2.0e-10,
        "information_bottleneck_excluded_mass_bounded": belief_diag[
            "maximum_ib_excluded_prior_mass"
        ] <= 1.0e-7,
        "active_observation_kkt": belief_diag["maximum_attention_kkt_residual"] <= 1.0e-10,
        "fgw_active": network_diag["maximum_fgw_proximal_change"] > 0.0,
        "fgw_marginals": network_diag["maximum_fgw_marginal_residual"] <= 2.0e-6,
        "ollivier_active": network_diag["directed_ollivier_updates"] == 1,
        "ollivier_marginals": network_diag["maximum_ollivier_marginal_residual"] <= 2.0e-4,
        "pot_plan_oracle": network_diag["pot_validation_calls"] == 2
        and network_diag["maximum_pot_marginal_residual"] <= 1.0e-8,
        "hodge_exact": network_diag["maximum_hodge_reconstruction_residual"] <= 1.0e-10,
        "oscillator_active": oscillator_diag["updates"] == 1 and mfg_diag["v77_coupled_oscillator"]["maximum_graphon_kernel_change"] > 0.0,
        "persistent_homology": tda_diag["maximum_h0_barcode_total_lifetime"] > 0.0,
        "clique_h1": tda_diag["maximum_vietoris_rips_h1_betti"] >= 0,
        "mapper_full": tda_diag["maximum_mapper_vertices"] > 0,
        "bipersistence_rank": tda_diag["maximum_bipersistence_image_rank"] > 0,
        "zigzag_two_snapshots": tda_diag["updates"] == 2,
        "hypergraph_homology": "maximum_hypergraph_simplicial_h1_betti" in tda_diag,
        "nested_ces_active": ces_diag["updates"] == 1,
        "ces_limit": ces_diag["maximum_cobb_douglas_limit_residual"] <= 2.0e-6,
        "no_probability_patch": all(
            not block["terminal_probability_adjustment"]
            for block in (belief_diag, network_diag, oscillator_diag, ces_diag)
        ),
        "information_output_finite": bool(np.all(np.isfinite(information.posterior_mean))),
    }
    payload = {
        "model_version": "V7.7-COMPLETE-THEORY-CLOSURE",
        "classification": "DIAGNOSTIC_ONLY; no outcome probabilities",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "information": belief_diag,
        "network": network_diag,
        "tda": tda_diag,
        "oscillator": mfg_diag["v77_coupled_oscillator"],
        "nested_ces": ces_diag,
    }
    TARGET.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"TAIWAN_V77_DIAGNOSTIC_ONLY: {payload['status']}")
    if payload["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
