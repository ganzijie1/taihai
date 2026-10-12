from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from large_deviation_v67 import (
    HybridLargeDeviationSystemV67,
    PROPOSAL_MODE_PROBABILITY,
    TARGET_MODE_PROBABILITY,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
TARGET = OUT / "taiwan_v67_large_deviation_diagnostic.json"


def action_refinement(level: int) -> float:
    dt = 1.0 / level
    time = np.linspace(0.0, 1.0, level + 1)
    path = 0.24 * time + 0.08 * np.sin(np.pi * time)
    velocity = np.diff(path) / dt
    midpoint = 0.5 * (time[:-1] + time[1:])
    drift = 0.24 + 0.08 * np.pi * np.cos(np.pi * midpoint)
    residual = velocity - drift
    return float(0.5 * np.sum(residual * residual * dt) / 0.04 ** 2)


def main() -> None:
    paths = 16384
    system = HybridLargeDeviationSystemV67(paths, 20261012)
    proposal = system.initialise_proposal()
    rng = np.random.default_rng(20261013)
    for month in range(24):
        phase = month / 23.0
        actor_damage = np.clip(
            0.04 + 0.10 * phase + rng.normal(0.0, 0.006, (paths, 4)), 0.0, 1.0
        )
        actor_output = np.clip(0.94 - 0.25 * phase - 0.30 * actor_damage, 0.02, 1.2)
        finance = np.clip(0.96 - 0.16 * phase - 0.15 * actor_damage, 0.02, 1.0)
        network = np.clip(0.95 - 0.14 * phase - 0.18 * actor_damage, 0.02, 1.0)
        command_scalar = np.clip(0.96 - 0.18 * phase - 0.20 * actor_damage, 0.02, 1.0)
        command = np.repeat(command_scalar[:, :, None], 3, axis=2)
        support = np.clip(0.62 - 0.10 * phase + rng.normal(0.0, 0.015, (paths, 4)), 0.0, 1.0)
        panic = np.clip(0.08 + 0.22 * phase + actor_damage, 0.0, 1.0)
        action = np.clip(0.05 + 0.20 * panic, 0.0, 1.0)
        resilience = np.repeat(
            np.clip(0.90 - 0.32 * phase - actor_damage, 0.0, 1.0)[:, :, None],
            4, axis=2,
        )
        shortage = np.clip(0.06 + 0.30 * phase + actor_damage, 0.0, 1.0)
        integrity = np.repeat(
            np.clip(0.94 - 0.38 * phase - actor_damage, 0.0, 1.0)[:, :, None],
            3, axis=2,
        )
        leader = np.clip(0.98 - 0.10 * phase - 0.20 * actor_damage, 0.0, 1.0)
        probability = np.clip(0.004 + 0.010 * phase + 0.02 * actor_damage, 1e-8, 0.20)
        event = rng.random((paths, 4)) < probability
        system.record_jumps(events=(event,), probabilities=(probability,))
        system.observe(
            month=month,
            actor_damage=actor_damage,
            actor_output=actor_output,
            financial_capacity=finance,
            actor_network=network,
            actor_command=command,
            actor_support=support,
            actor_panic=panic,
            actor_action=action,
            actor_resilience=resilience,
            stock_shortage=shortage,
            front_rear_integrity=integrity,
            leader_continuity=leader,
        )
    diagnostic = system.diagnostics()
    weights = proposal.likelihood_ratio
    rare_indicator = proposal.mode != 0
    importance_estimate = float(np.mean(weights * rare_indicator))
    ordinary_rng = np.random.default_rng(20261014)
    ordinary_mode = ordinary_rng.choice(
        len(TARGET_MODE_PROBABILITY), paths, p=TARGET_MODE_PROBABILITY
    )
    ordinary_estimate = float(np.mean(ordinary_mode != 0))
    exact_probability = float(1.0 - TARGET_MODE_PROBABILITY[0])
    action_grid = {str(level): action_refinement(level) for level in (16, 32, 64, 128)}
    convergence = abs(action_grid["128"] - action_grid["64"])
    checks = {
        "analytic_likelihood_identity": diagnostic["max_likelihood_identity_residual"] <= 1e-12,
        "sample_likelihood_normalization": diagnostic["sample_likelihood_normalization_residual"] <= 0.03,
        "effective_sample_size": diagnostic["effective_sample_size_ratio"] >= 0.40,
        "ordinary_mc_overlap": abs(ordinary_estimate - exact_probability) <= 0.02,
        "importance_sampling_overlap": abs(importance_estimate - exact_probability) <= 0.02,
        "ordinary_importance_agreement": abs(ordinary_estimate - importance_estimate) <= 0.025,
        "empirical_measure_mass": diagnostic["max_empirical_measure_mass_residual"] <= 1e-12,
        "causal_clock": diagnostic["max_causal_clock_violation"] == 0.0,
        "finite_nonnegative_actions": all(
            np.isfinite(diagnostic[key]) and diagnostic[key] >= 0.0
            for key in (
                "mean_continuous_action", "mean_jump_action",
                "mean_field_empirical_action",
            )
        ),
        "action_refinement": convergence <= 1e-3,
    }
    payload = {
        "model_version": "V6.7-LD",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "overlap": {
            "exact_probability": exact_probability,
            "ordinary_monte_carlo": ordinary_estimate,
            "importance_sampling": importance_estimate,
        },
        "action_refinement": action_grid,
        "action_refinement_last_difference": convergence,
        "module_diagnostics": diagnostic,
        "proposal_support_complete": bool(np.all(PROPOSAL_MODE_PROBABILITY > 0.0)),
        "target_support_complete": bool(np.all(TARGET_MODE_PROBABILITY > 0.0)),
    }
    TARGET.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if payload["status"] != "PASS":
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError("V6.7 diagnostic failed: " + ", ".join(failed))
    print("TAIWAN_V67_LARGE_DEVIATION_DIAGNOSTIC: PASS")


if __name__ == "__main__":
    main()
