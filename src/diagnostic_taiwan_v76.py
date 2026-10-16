from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scf_signed_potential_mfg_v76 import (
    AdaptiveTrustRegionSCFAnderson,
    SignedPotentialHypergraphMFGV76,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
TARGET = OUT / "taiwan_v76_diagnostic_only.json"


def main() -> None:
    paths = 12
    support = np.tile(np.array([
        [0.62, 0.48, 0.55, 0.57],
        [0.58, 0.52, 0.51, 0.54],
    ]), (paths // 2, 1))
    system = SignedPotentialHypergraphMFGV76(
        paths,
        760_013,
        support,
        advanced_graphon=True,
        graphon_mode="multilayer",
        partial_observation_geometry=False,
        network_geometry_enabled=False,
    )
    centers = np.array([-0.82, 0.78, -0.68, 0.72])
    density = np.exp(
        -0.5 * ((system.x[None, None, None, :] - centers[None, :, None, None]) / 0.16) ** 2
    )
    density = np.broadcast_to(density, (paths, 4, 4, len(system.x))).copy()
    density /= np.sum(density, axis=-1, keepdims=True) * system.dx
    system.density = density
    off_diagonal = ~np.eye(4, dtype=bool)
    for layer in (0, 2):
        system.positive_relation[:, layer, off_diagonal] = 0.18
        system.negative_relation[:, layer, off_diagonal] = 0.72
    damage = np.tile(np.array([[0.18, 0.28, 0.09, 0.11], [0.22, 0.25, 0.10, 0.12]]), (paths // 2, 1))
    shortage = np.tile(np.array([[0.24, 0.31, 0.12, 0.14], [0.26, 0.29, 0.13, 0.15]]), (paths // 2, 1))
    network = np.tile(np.array([[0.72, 0.66, 0.78, 0.75], [0.69, 0.64, 0.76, 0.73]]), (paths // 2, 1))
    noise = np.tile(np.array([[0.08, 0.05, 0.03, 0.04], [0.06, 0.07, 0.02, 0.03]]), (paths // 2, 1))
    jump = np.tile(np.array([[0.05, 0.12, 0.03, 0.04], [0.07, 0.10, 0.02, 0.05]]), (paths // 2, 1))
    signal = np.tile(np.array([[0.70, -0.62, 0.58, -0.54], [0.64, -0.56, 0.52, -0.48]]), (paths // 2, 1))
    kernel, weights, intensity = system._effective_graphon(
        damage, shortage, network, noise, jump, signal
    )
    mean, _ = system._moments(system.density)
    action = np.linspace(0.05, 0.85, paths * 4 * 4 * 5).reshape(paths, 4, 4, 5)
    state, action_field, major = system._multilayer_fields(
        mean, action, kernel, weights, system.major_action, intensity
    )
    finite_state, finite_action, finite_error = system._finite_network_fields(
        mean, action, kernel, weights, intensity
    )
    system._reduced_branch_probe(signal, damage, shortage)
    system.major_action[:] = np.array([0.72, 0.48, 0.66, 0.54])[None, None, :]
    tda_factors = system.online_tda.update(
        month=3,
        mean=mean,
        joint_action=action,
        signed_kernel=system._last_signed_kernel,
        layer_weights=system._last_layer_weights,
        hyper_tail=system.hyper_tail,
        hyper_head=system.hyper_head,
        major_action=system.major_action,
    )
    system.tda_delayed_shock = tda_factors.delayed_topology_shock
    accelerator = AdaptiveTrustRegionSCFAnderson(depth=2, damping=0.52)
    mapped = 0.82 * density + 0.18 * density[..., ::-1]
    mapped /= np.sum(mapped, axis=-1, keepdims=True) * system.dx
    candidate = density
    for _ in range(4):
        candidate = accelerator.step(candidate, mapped, system.dx)
    diag = system.diagnostics()["v76_signed_potential_scf"]
    tda_diag = system.diagnostics()["v76_online_tda"]
    checks = {
        "finite_fields": bool(
            np.all(np.isfinite(state))
            and np.all(np.isfinite(action_field))
            and np.all(np.isfinite(major))
            and np.all(np.isfinite(finite_state))
            and np.all(np.isfinite(finite_action))
            and np.isfinite(finite_error)
        ),
        "finite_subsample_exercised": finite_state.shape[0] == 8,
        "signed_edges_active": bool(np.any(system.current_sign < 0.0)),
        "signed_field_active": diag["maximum_signed_field_correction"] > 0.0,
        "decomposition_exact": diag["maximum_decomposition_residual"] <= 1.0e-12,
        "pair_circulation_measured": diag["maximum_pair_nonpotential_ratio"] > 0.0,
        "hypergraph_circulation_measured": diag["maximum_hypergraph_nonpotential_ratio"] > 0.0,
        "signed_laplacian_psd": diag["minimum_signed_laplacian_eigenvalue"] >= -1.0e-10,
        "relation_memory_active": diag["relation_memory_updates"] == 1 and diag["maximum_relation_step"] > 0.0,
        "atomization_not_absorbing": diag["minimum_zero_boundary_formation_rate"] > 0.0 and not diag["atomized_boundary_absorbing"],
        "bridge_index_bounded": 0.0 <= diag["minimum_bridge_index"] <= diag["maximum_bridge_index"] <= 1.0,
        "nonautonomous_spectrum_finite": bool(np.isfinite(diag["maximum_nonautonomous_growth_rate"])),
        "scf_trust_active": diag["scf_calls"] == 4 and diag["minimum_scf_trust_ratio"] >= 1.0,
        "branch_probe_active": diag["reduced_branch_evaluations"] == 1,
        "tda_update_active": tda_diag["updates"] == 1,
        "tda_feedback_active": tda_diag["maximum_feedback_shock"] > 0.0,
        "tda_bipersistence_active": tda_diag["maximum_bipersistence_rank"] > 0.0,
        "tda_directed_flag_active": tda_diag["maximum_directed_flag_triangles"] >= 0.0,
        "tda_no_terminal_adjustment": not tda_diag["terminal_probability_adjustment"],
        "no_repelling": not diag["repelling_rule_used"],
        "no_terminal_adjustment": not diag["terminal_probability_adjustment"],
    }
    payload = {
        "model_version": "V7.6-SIGNED-POTENTIAL-SCF",
        "classification": "DIAGNOSTIC_ONLY; no outcome probabilities",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "diagnostics": diag,
        "tda_diagnostics": tda_diag,
    }
    TARGET.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"TAIWAN_V76_DIAGNOSTIC_ONLY: {payload['status']}")
    if payload["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
