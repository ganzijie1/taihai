"""Same-path Filippov switching and impulse-reset ledger for Taiwan V5.1.

The system owns four hard-regime indicators and the reset semantics of event
processes that were previously applied inline.  It does not sample event times;
the existing DEDS/Cox/Hawkes layers remain the unique event generators.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


ACTORS = 4
SURFACES = ("logistics", "financial", "mobilization", "command")


@dataclass(frozen=True)
class HybridFactors:
    logistics: np.ndarray
    financial: np.ndarray
    mobilization: np.ndarray
    command: np.ndarray
    sliding_weight: np.ndarray
    regime: np.ndarray


class HybridFilippovImpulseSystem:
    """Filippov differential inclusion with hysteresis and unique pulse resets."""

    def __init__(
        self, paths: int, enabled: bool = True,
        threshold_shift: float = 0.0, control_capacity_scale: float = 1.0,
    ) -> None:
        self.paths = paths
        self.enabled = enabled
        self.regime = np.zeros((paths, ACTORS, len(SURFACES)), dtype=bool)
        self.dwell = np.zeros_like(self.regime, dtype=np.int16)
        self.previous_stress = np.zeros_like(self.regime, dtype=float)
        self.initialized = False
        self.minimum_dwell = 2
        self.threshold_shift = float(threshold_shift)
        self.control_capacity_scale = float(control_capacity_scale)
        self.on_threshold = np.clip(
            np.array([0.34, 0.78, 0.50, 0.44]) + self.threshold_shift, 0.15, 0.95
        )[None, None, :]
        self.off_threshold = np.clip(
            np.array([0.24, 0.62, 0.36, 0.31]) + self.threshold_shift, 0.08, 0.88
        )[None, None, :]
        self.control_capacity = (
            np.array([0.10, 0.09, 0.11, 0.10]) * self.control_capacity_scale
        )[None, None, :]
        self.boundary_band = 0.025
        self.current_month = -1
        self.month_event_types: set[str] = set()
        self.event_calls = {name: 0 for name in (
            "homeland_island_damage", "leader_disruption", "coup", "collapse"
        )}
        self.event_counts = {name: 0 for name in self.event_calls}
        self.max_reset_residual = 0.0
        self.max_sliding_residual = 0.0
        self.max_hysteresis_violation = 0.0
        self.switch_count = np.zeros(len(SURFACES), dtype=np.int64)
        self.sliding_observations = np.zeros(len(SURFACES), dtype=np.int64)
        self.stress_min = np.full(len(SURFACES), np.inf)
        self.stress_max = np.full(len(SURFACES), -np.inf)
        self.stress_sum = np.zeros(len(SURFACES))
        self.stress_observations = 0
        self.max_simultaneous_event_types = 0

    def begin_month(self, month: int) -> None:
        if month <= self.current_month:
            raise RuntimeError("hybrid event clock must advance strictly")
        self.current_month = month
        self.month_event_types.clear()

    def _record(self, event_type: str, mask: np.ndarray) -> None:
        if event_type in self.month_event_types:
            raise RuntimeError(f"duplicate impulse settlement in month {self.current_month}: {event_type}")
        self.month_event_types.add(event_type)
        self.event_calls[event_type] += 1
        self.event_counts[event_type] += int(np.count_nonzero(mask))
        self.max_simultaneous_event_types = max(
            self.max_simultaneous_event_types, len(self.month_event_types)
        )

    @staticmethod
    def _reset_residual(actual: np.ndarray, expected: np.ndarray) -> float:
        return float(np.max(np.abs(actual - expected))) if actual.size else 0.0

    def update_filippov(
        self,
        logistics_service: np.ndarray,
        financial_capacity: np.ndarray,
        debt_stress: np.ndarray,
        stock_shortage: np.ndarray,
        support: np.ndarray,
        conflict_intensity: np.ndarray,
        command_service: np.ndarray,
    ) -> HybridFactors:
        """Advance four switching surfaces and return bounded flow multipliers.

        On a switching band, ``v_minus`` is the uncontrolled outward normal
        velocity and ``v_plus`` is the controlled velocity.  When they point
        toward one another, the Filippov convex weight makes the normal
        velocity zero up to floating-point error.
        """
        intensity = np.broadcast_to(conflict_intensity[:, None], (self.paths, ACTORS))
        stress = np.stack((
            1.0 - np.clip(logistics_service, 0.0, 1.0),
            np.clip(0.58 * (1.0 - financial_capacity) + 0.42 * debt_stress, 0.0, 1.5),
            np.clip(0.52 * stock_shortage + 0.30 * intensity + 0.18 * (1.0 - support), 0.0, 1.5),
            1.0 - np.clip(command_service, 0.0, 1.0),
        ), axis=2)
        self.stress_min = np.minimum(self.stress_min, np.min(stress, axis=(0, 1)))
        self.stress_max = np.maximum(self.stress_max, np.max(stress, axis=(0, 1)))
        self.stress_sum += np.sum(stress, axis=(0, 1))
        self.stress_observations += self.paths * ACTORS
        delta = np.zeros_like(stress) if not self.initialized else stress - self.previous_stress
        old = self.regime.copy()
        can_exit = self.dwell >= self.minimum_dwell
        self.regime |= stress >= self.on_threshold
        self.regime &= ~((stress <= self.off_threshold) & can_exit)
        switched = self.regime != old
        self.switch_count += np.sum(switched, axis=(0, 1))
        self.dwell = np.where(
            switched, 0,
            np.minimum(self.dwell + 1, np.iinfo(np.int16).max),
        )

        active_threshold = np.where(self.regime, self.off_threshold, self.on_threshold)
        h = stress - active_threshold
        endogenous_pressure = 0.55 * delta + 0.12 * h
        v_minus = endogenous_pressure + 0.5 * self.control_capacity
        v_plus = endogenous_pressure - 0.5 * self.control_capacity
        sliding = (
            (np.abs(h) <= self.boundary_band)
            & (v_minus > 0.0) & (v_plus < 0.0)
        )
        denominator = np.maximum(v_minus - v_plus, 1e-12)
        lam = np.where(sliding, np.clip(v_minus / denominator, 0.0, 1.0), self.regime)
        normal_velocity = (1.0 - lam) * v_minus + lam * v_plus
        if np.any(sliding):
            self.max_sliding_residual = max(
                self.max_sliding_residual,
                float(np.max(np.abs(normal_velocity[sliding]))),
            )
        self.sliding_observations += np.sum(sliding, axis=(0, 1))

        premature_exit = old & (~self.regime) & (~can_exit)
        self.max_hysteresis_violation = max(
            self.max_hysteresis_violation, float(np.any(premature_exit))
        )
        self.previous_stress = stress
        self.initialized = True
        if not self.enabled:
            lam = np.zeros_like(lam)
            self.regime[:] = False

        return HybridFactors(
            logistics=np.clip(1.0 - 0.20 * lam[:, :, 0], 0.70, 1.0),
            financial=np.clip(1.0 - 0.16 * lam[:, :, 1], 0.72, 1.0),
            mobilization=np.clip(1.0 + 0.14 * lam[:, :, 2], 1.0, 1.18),
            command=np.clip(1.0 - 0.22 * lam[:, :, 3], 0.68, 1.0),
            sliding_weight=lam.copy(),
            regime=self.regime.copy(),
        )

    def apply_damage_reset(
        self,
        state: dict,
        homeland_event: np.ndarray,
        homeland_mark: np.ndarray,
        island_event: np.ndarray,
        island_mark: np.ndarray,
    ) -> None:
        mask = homeland_event | island_event
        self._record("homeland_island_damage", mask)
        old_home = state["actor_homeland_damage"].copy()
        old_island = state["actor_island_chain_damage"].copy()
        old_backlash = state["actor_backlash"].copy()
        expected_home = np.clip(0.95 * old_home + homeland_mark, 0.0, 0.35)
        expected_island = np.clip(0.93 * old_island + island_mark, 0.0, 0.55)
        expected_backlash = np.clip(
            0.975 * old_backlash + 1.6 * homeland_mark * (1.0 - state["actor_panic"]),
            0.0, 1.0,
        )
        state["actor_homeland_damage"] = expected_home
        state["actor_island_chain_damage"] = expected_island
        state["actor_backlash"] = expected_backlash
        self.max_reset_residual = max(
            self.max_reset_residual,
            self._reset_residual(state["actor_homeland_damage"], expected_home),
            self._reset_residual(state["actor_island_chain_damage"], expected_island),
            self._reset_residual(state["actor_backlash"], expected_backlash),
        )

    def apply_leader_reset(
        self,
        state: dict,
        leader_attempt: np.ndarray,
        leader_disruption: np.ndarray,
        leader_severity: np.ndarray,
    ) -> None:
        self._record("leader_disruption", leader_attempt)
        old_continuity = state["leader_continuity"].copy()
        expected = np.where(
            leader_disruption,
            np.maximum(0.12, old_continuity * (1.0 - leader_severity)),
            old_continuity,
        )
        replacement_rate = np.array([0.045, 0.070, 0.055, 0.060])[None, :]
        expected = np.clip(
            expected + replacement_rate * (1.0 - expected) * state["actor_res"][:, :, 1],
            0.10, 1.0,
        )
        state["leader_continuity"] = expected
        state["leader_disruption_count"] += leader_disruption
        state["actor_command"] *= (0.985 + 0.015 * expected)[:, :, None]
        state["actor_panic"] = np.clip(
            state["actor_panic"] + 0.12 * leader_disruption
            + 0.035 * leader_attempt * (~leader_disruption), 0.0, 1.0,
        )
        state["actor_support"] = np.clip(
            state["actor_support"] - 0.08 * leader_disruption
            + 0.025 * leader_attempt * (~leader_disruption), 0.01, 0.99,
        )
        self.max_reset_residual = max(
            self.max_reset_residual,
            self._reset_residual(state["leader_continuity"], expected),
        )

    def apply_coup_reset(self, state: dict, new_coup: np.ndarray) -> None:
        self._record("coup", new_coup)
        state["actor_coup"] |= new_coup
        expected_res = np.clip(
            state["actor_res"]
            - new_coup[:, :, None] * np.array([0.12, 0.35, 0.18, 0.28])[None, None, :],
            0.0, 1.0,
        )
        state["actor_res"] = expected_res
        signal = new_coup.astype(float)
        state["cog_cn"] = np.clip(
            state["cog_cn"] + 0.018 * (signal[:, 1] + signal[:, 2] + signal[:, 3])
            - 0.030 * signal[:, 0], 0.04, 0.96,
        )
        state["cog_def"] = np.clip(
            state["cog_def"] + 0.018 * signal[:, 0] - 0.020 * signal[:, 1]
            - 0.012 * signal[:, 2] - 0.012 * signal[:, 3], 0.04, 0.96,
        )
        self.max_reset_residual = max(
            self.max_reset_residual,
            self._reset_residual(state["actor_res"], expected_res),
        )

    def apply_collapse_reset(
        self,
        state: dict,
        persistent_collapse: np.ndarray,
        jump_collapse: np.ndarray,
        new_coup: np.ndarray,
    ) -> None:
        event = persistent_collapse | jump_collapse
        event[:, :, 1] |= new_coup
        new_event = event & (~state["actor_collapse"])
        self._record("collapse", new_event)
        expected = state["actor_collapse"] | event
        state["actor_collapse"] = expected
        self.max_reset_residual = max(
            self.max_reset_residual,
            self._reset_residual(state["actor_collapse"].astype(float), expected.astype(float)),
        )

    def diagnostics(self) -> dict:
        total = max(self.paths * max(self.current_month + 1, 1) * ACTORS, 1)
        return {
            "enabled": self.enabled,
            "surfaces": list(SURFACES),
            "switch_count": dict(zip(SURFACES, self.switch_count.tolist())),
            "sliding_share": dict(zip(SURFACES, (self.sliding_observations / total).tolist())),
            "terminal_regime_share": dict(zip(SURFACES, np.mean(self.regime, axis=(0, 1)).tolist())),
            "stress_min": dict(zip(SURFACES, self.stress_min.tolist())),
            "stress_mean": dict(zip(
                SURFACES,
                (self.stress_sum / max(self.stress_observations, 1)).tolist(),
            )),
            "stress_max": dict(zip(SURFACES, self.stress_max.tolist())),
            "event_calls": self.event_calls.copy(),
            "event_counts": self.event_counts.copy(),
            "max_simultaneous_event_types": self.max_simultaneous_event_types,
            "max_reset_residual": self.max_reset_residual,
            "max_sliding_normal_residual": self.max_sliding_residual,
            "max_hysteresis_violation": self.max_hysteresis_violation,
            "minimum_dwell_months": self.minimum_dwell,
            "threshold_shift": self.threshold_shift,
            "control_capacity_scale": self.control_capacity_scale,
        }
