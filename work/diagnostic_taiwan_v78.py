from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from mixed_potential_nonpotential_mfg_v78 import MixedPotentialNonpotentialMFGV78


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
TARGET = OUT / "taiwan_v78_diagnostic_only.json"


def main() -> None:
    subprocess.run(
        [sys.executable, str(ROOT / "work" / "diagnostic_taiwan_v77.py")],
        check=True,
    )
    inherited = json.loads(
        (OUT / "taiwan_v77_diagnostic_only.json").read_text(encoding="utf-8")
    )
    paths = 4
    support = np.array([
        [0.62, 0.48, 0.55, 0.57],
        [0.58, 0.52, 0.51, 0.54],
        [0.60, 0.46, 0.56, 0.53],
        [0.56, 0.50, 0.49, 0.55],
    ])
    system = MixedPotentialNonpotentialMFGV78(
        paths, 780_013, support,
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
    progress = np.array([
        [0.70, -0.62, 0.58, -0.54],
        [0.64, -0.56, 0.52, -0.48],
        [0.60, -0.48, 0.45, -0.42],
        [0.55, -0.40, 0.38, -0.35],
    ])
    factors = system.update(
        progress=progress,
        damage=damage,
        shortage=shortage,
        inflation=damage,
        financial_stress=damage,
        command=network,
        network=network,
        contagion_mobilization=np.full((paths, 4), 0.20),
        contagion_fatigue=np.full((paths, 4), 0.16),
        common_noise=noise,
        jump_signal=jump,
    )
    diagnostic = system.diagnostics()["v78_mixed_potential_nonpotential"]
    inherited_hypergraph = system.diagnostics()[
        "v75_directed_multilayer_hypergraph_mfg"
    ]
    inherited_signed = system.diagnostics()["v76_signed_potential_scf"]
    ablations = diagnostic["maximum_ablation_field_deviation"]
    checks = {
        "v77_inherited_gate": inherited["status"] == "PASS",
        "continuous_hjb_fp_converged": factors.converged
        and factors.fixed_point_residual <= 5.0e-4,
        "weighted_adjoint_exact": diagnostic[
            "maximum_weighted_adjoint_residual"
        ] <= 1.0e-12,
        "potential_split_exact": diagnostic[
            "maximum_potential_decomposition_residual"
        ] <= 1.0e-12,
        "mirror_prox_active": diagnostic["mirror_prox_evaluations"] > 0
        and diagnostic["maximum_mirror_prox_predictor_step"] > 0.0
        and diagnostic["maximum_mirror_prox_correction"] > 0.0,
        "potential_prox_active": diagnostic["potential_proximal_updates"] > 0,
        "nonpotential_operator_active": diagnostic[
            "maximum_nonpotential_operator_norm"
        ] > 0.0,
        "inherited_higher_order_metric_active": inherited_hypergraph[
            "max_higher_order_contribution"
        ] > 0.0,
        "inherited_signed_metric_active": inherited_signed[
            "maximum_signed_field_correction"
        ] > 0.0,
        "vi_gap_finite": np.isfinite(diagnostic["maximum_mixed_vi_gap"]),
        "finite_network_field_within_runtime_gate": (
            0.0 < factors.finite_network_field_error <= 2.5e-1
        ),
        "finite_network_payoff_within_runtime_gate": (
            0.0 < factors.finite_network_payoff_deviation <= 2.0e-1
        ),
        "four_ablation_families": set(ablations) == {
            "no_graph_mfg", "block_gmfg", "low_rank_graphon", "finite_graph"
        } and all(value > 0.0 for value in ablations.values()),
        "single_state_owner": diagnostic["state_owner"].startswith("single"),
        "no_probability_patch": not diagnostic["terminal_probability_adjustment"],
    }
    checks = {name: bool(value) for name, value in checks.items()}
    payload = {
        "model_version": "V7.8-MIXED-POTENTIAL-NONPOTENTIAL-MFG",
        "classification": "DIAGNOSTIC_ONLY; no outcome probabilities",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "mfg_factors": {
            "fixed_point_residual": factors.fixed_point_residual,
            "mean_exploitability_bound": factors.mean_exploitability_bound,
            "worst_type_exploitability_bound": factors.worst_type_exploitability_bound,
            "finite_network_field_error": factors.finite_network_field_error,
            "finite_network_payoff_deviation": (
                factors.finite_network_payoff_deviation
            ),
        },
        "mixed_solver": diagnostic,
        "inherited_hypergraph": inherited_hypergraph,
        "inherited_signed": inherited_signed,
    }
    TARGET.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"TAIWAN_V78_DIAGNOSTIC_ONLY: {payload['status']}")
    if payload["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
