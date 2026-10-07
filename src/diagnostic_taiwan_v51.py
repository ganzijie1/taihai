"""DIAGNOSTIC_ONLY: isolated Filippov and impulse-reset invariants."""

from __future__ import annotations

import numpy as np

from hybrid_filippov_impulse_v51 import HybridFilippovImpulseSystem
from topological_dynamics_v51 import TopologicalRegimeDynamics


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def fixture_state(paths: int) -> dict:
    return {
        "actor_homeland_damage": np.full((paths, 4), 0.10),
        "actor_island_chain_damage": np.full((paths, 4), 0.12),
        "actor_backlash": np.full((paths, 4), 0.08),
        "actor_panic": np.full((paths, 4), 0.15),
        "leader_continuity": np.full((paths, 4), 0.92),
        "leader_disruption_count": np.zeros((paths, 4), dtype=np.int16),
        "actor_command": np.full((paths, 4, 3), 0.85),
        "actor_res": np.full((paths, 4, 4), 0.72),
        "actor_support": np.full((paths, 4), 0.65),
        "actor_coup": np.zeros((paths, 4), dtype=bool),
        "actor_collapse": np.zeros((paths, 4, 4), dtype=bool),
        "cog_cn": np.full(paths, 0.62),
        "cog_def": np.full(paths, 0.60),
    }


def main() -> None:
    paths = 5
    system = HybridFilippovImpulseSystem(paths)
    topology = TopologicalRegimeDynamics(paths)
    one = np.ones((paths, 4))
    support = one * 0.66

    system.begin_month(0)
    factors = system.update_filippov(
        logistics_service=one * 0.70,
        financial_capacity=one * 0.82,
        debt_stress=one * 0.30,
        stock_shortage=one * 0.28,
        support=support,
        conflict_intensity=np.full(paths, 0.42),
        command_service=one * 0.76,
    )
    require(np.all((factors.sliding_weight >= 0.0) & (factors.sliding_weight <= 1.0)),
            "Filippov convex weight bounds")
    topological = topology.update(
        factors.regime, np.zeros(paths, dtype=np.int8),
        np.zeros((paths, 4, 4), dtype=bool),
    )
    require(np.all((topological.command >= 0.90) & (topological.command <= 1.02)),
            "topological command feedback bounds")

    # Force entry, retain the regime inside the hysteresis band, then allow exit.
    for month, service in ((1, 0.40), (2, 0.60), (3, 0.60), (4, 0.80)):
        system.begin_month(month)
        factors = system.update_filippov(
            logistics_service=one * service,
            financial_capacity=one * 0.82,
            debt_stress=one * 0.30,
            stock_shortage=one * 0.28,
            support=support,
            conflict_intensity=np.full(paths, 0.42),
            command_service=one * 0.76,
        )
        topological = topology.update(
            factors.regime, np.zeros(paths, dtype=np.int8),
            np.zeros((paths, 4, 4), dtype=bool),
        )
        zero = np.zeros((paths, 4), dtype=bool)
        marks = np.zeros((paths, 4))
        state = fixture_state(paths)
        system.apply_damage_reset(state, zero, marks, zero, marks)
        system.apply_leader_reset(state, zero, zero, marks)
        system.apply_coup_reset(state, zero)
        system.apply_collapse_reset(
            state, np.zeros((paths, 4, 4), dtype=bool),
            np.zeros((paths, 4, 4), dtype=bool), zero,
        )

    diag = system.diagnostics()
    require(diag["switch_count"]["logistics"] > 0, "Filippov switching movement")
    require(diag["max_hysteresis_violation"] == 0.0, "minimum dwell hysteresis")
    require(diag["max_sliding_normal_residual"] < 1e-12, "sliding normal velocity")
    require(diag["max_reset_residual"] < 1e-12, "impulse reset identity")
    topology_diag = topology.diagnostics()
    require(topology_diag["transition_accounting_residual"] == 0,
            "topological transition accounting")
    require(topology_diag["observed_transition_edges"] > 1,
            "nontrivial symbolic transition graph")
    require(len(topology_diag["morse_sets"]) >= 1, "Morse decomposition")
    require(topology_diag["max_feedback_deviation"] > 0.0,
            "topological feedback movement")

    duplicate = HybridFilippovImpulseSystem(1)
    duplicate.begin_month(0)
    state = fixture_state(1)
    zero = np.zeros((1, 4), dtype=bool)
    mark = np.zeros((1, 4))
    duplicate.apply_damage_reset(state, zero, mark, zero, mark)
    rejected = False
    try:
        duplicate.apply_damage_reset(state, zero, mark, zero, mark)
    except RuntimeError:
        rejected = True
    require(rejected, "duplicate event settlement must be rejected")
    print("DIAGNOSTIC_ONLY_TAIWAN_V51: PASS")


if __name__ == "__main__":
    main()
