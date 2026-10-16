from __future__ import annotations

from dataclasses import dataclass

import numpy as np


CONTROLS = ("strike", "defense", "production", "repair", "negotiation")


@dataclass
class GameFactors:
    controls: np.ndarray
    value: np.ndarray
    best_response_residual: float
    belief_residual: float
    converged: bool


class RollingHorizonDifferentialGame:
    """Four-player constrained receding-horizon feedback game.

    A finite control lattice is evaluated over a six-month reduced transition
    model. The US move is updated first, followed by Japan, Taiwan, and China;
    iteration continues until no player can materially improve its allocation.
    """

    def __init__(self, paths: int, seed: int = 4701):
        self.paths = paths
        self.rng = np.random.default_rng(seed)
        raw = np.array([
            [0.34, 0.24, 0.18, 0.16, 0.08],
            [0.24, 0.34, 0.18, 0.16, 0.08],
            [0.24, 0.20, 0.30, 0.18, 0.08],
            [0.22, 0.20, 0.18, 0.32, 0.08],
            [0.18, 0.18, 0.18, 0.16, 0.30],
            [0.29, 0.21, 0.24, 0.18, 0.08],
            [0.20, 0.29, 0.22, 0.21, 0.08],
            [0.20, 0.20, 0.24, 0.18, 0.18],
        ])
        self.candidates = raw / raw.sum(axis=1, keepdims=True)
        self._scenario_multiplier = np.array([0.82, 1.00, 1.24])
        self._candidate_cost = np.sum(self.candidates ** 2, axis=1)
        self.controls = np.repeat(
            np.array([0.26, 0.25, 0.21, 0.19, 0.09])[None, None, :],
            paths * 4, axis=0,
        ).reshape(paths, 4, 5)
        self.last_residual = np.inf
        self.last_belief_residual = np.inf
        self.nonconverged_months = 0
        self.max_nonconverged_residual = 0.0
        self._belief = None
        self._belief_variance = None
        self._last_observation = None
        self.candidate_batch_calls = 0

    def _belief_update(self, observed: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """One-step bounded Gaussian filter used as the POMDP belief state."""
        process_variance = {
            "progress": 0.018, "stocks": 0.012, "damage": 0.014,
            "economic_capacity": 0.010, "support": 0.008, "command": 0.012,
        }
        observation_variance = {
            "progress": 0.055, "stocks": 0.070, "damage": 0.045,
            "economic_capacity": 0.035, "support": 0.060, "command": 0.050,
        }
        if self._belief is None:
            self._belief = {name: value.copy() for name, value in observed.items()}
            self._belief_variance = {
                name: np.full_like(value, observation_variance[name])
                for name, value in observed.items()
            }
            self._last_observation = {name: value.copy() for name, value in observed.items()}
            self.last_belief_residual = 0.0
            return {name: value.copy() for name, value in self._belief.items()}

        largest_innovation = 0.0
        for name, truth in observed.items():
            # Reports are delayed and noisy; the filter never receives the true
            # campaign state directly.
            delayed = 0.72 * truth + 0.28 * self._last_observation[name]
            measurement = delayed + self.rng.normal(
                0.0, np.sqrt(observation_variance[name]), truth.shape
            )
            prior_variance = self._belief_variance[name] + process_variance[name]
            gain = prior_variance / (prior_variance + observation_variance[name])
            innovation = measurement - self._belief[name]
            self._belief[name] = self._belief[name] + gain * innovation
            self._belief_variance[name] = np.maximum((1.0 - gain) * prior_variance, 1e-8)
            self._last_observation[name] = truth.copy()
            largest_innovation = max(largest_innovation, float(np.max(np.abs(innovation))))
        self.last_belief_residual = largest_innovation
        return {name: value.copy() for name, value in self._belief.items()}

    def _evaluate(
        self, player: int, candidate: np.ndarray, controls: np.ndarray,
        progress: np.ndarray, stocks: np.ndarray, damage: np.ndarray,
        economic_capacity: np.ndarray, support: np.ndarray,
        command: np.ndarray, intervention: np.ndarray,
    ) -> np.ndarray:
        return self._evaluate_candidates(
            player, candidate[None, :], controls, progress, stocks, damage,
            economic_capacity, support, command, intervention,
        )[:, 0]

    def _evaluate_candidates(
        self, player: int, candidates: np.ndarray, controls: np.ndarray,
        progress: np.ndarray, stocks: np.ndarray, damage: np.ndarray,
        economic_capacity: np.ndarray, support: np.ndarray,
        command: np.ndarray, intervention: np.ndarray,
    ) -> np.ndarray:
        """Evaluate the complete candidate lattice in one vectorized batch.

        Axes are ``[path, candidate, transition_scenario]``.  Candidate-independent
        beliefs and intervention means are broadcast views, so the rolling game no
        longer copies and advances the full state once per candidate.
        """
        candidates = np.asarray(candidates, dtype=float)
        candidate_count = candidates.shape[0]
        trial = np.broadcast_to(
            controls[:, None, :, :],
            (self.paths, candidate_count, controls.shape[1], controls.shape[2]),
        ).copy()
        trial[:, :, player, :] = candidates[None, :, :]
        scenario_multiplier = self._scenario_multiplier[None, None, :]
        state_shape = (self.paths, candidate_count, 3)
        own_progress = np.broadcast_to(
            progress[:, player, None, None], state_shape
        ).copy()
        own_stock = np.broadcast_to(stocks[:, player, None, None], state_shape).copy()
        own_damage = np.broadcast_to(damage[:, player, None, None], state_shape).copy()
        own_economy = np.broadcast_to(
            economic_capacity[:, player, None, None], state_shape
        ).copy()
        own_support = np.broadcast_to(support[:, player, None, None], state_shape).copy()
        value = np.zeros(state_shape)
        opponent = (
            np.sum(trial, axis=2) - trial[:, :, player, :]
        ) / max(trial.shape[2] - 1, 1)
        intervention_mean = np.mean(intervention, axis=1)[:, None, None]
        strike = trial[:, :, player, 0]
        defense = trial[:, :, player, 1]
        production = trial[:, :, player, 2]
        repair = trial[:, :, player, 3]
        negotiation = trial[:, :, player, 4]
        command_player = command[:, player, None, None]
        control_cost = np.sum(trial[:, :, player, :] ** 2, axis=2)[..., None]
        for step in range(6):
            threat = (
                opponent[:, :, 0, None] * intervention_mean
            ) * scenario_multiplier
            own_progress += (
                0.045 * strike[..., None] * command_player
            ) - 0.038 * threat * (1.0 - defense[..., None])
            own_damage = np.clip(
                own_damage + 0.030 * threat * (1.0 - defense[..., None])
                - 0.022 * repair[..., None] * own_economy,
                0.0, 1.0,
            )
            own_stock = np.clip(
                own_stock + 0.040 * production[..., None] * own_economy
                - 0.035 * strike[..., None] - 0.018 * defense[..., None],
                0.0, 1.4,
            )
            own_economy = np.clip(
                own_economy - 0.020 * strike[..., None] - 0.012 * defense[..., None]
                - 0.025 * own_damage + 0.010 * production[..., None],
                0.05, 1.2,
            )
            own_support = np.clip(
                own_support + 0.012 * own_progress - 0.020 * own_damage
                - 0.012 * (strike + defense)[..., None]
                + 0.010 * negotiation[..., None],
                0.0, 1.0,
            )
            flow = (
                0.48 * own_progress + 0.18 * own_stock + 0.14 * own_economy
                + 0.12 * own_support - 0.28 * own_damage
                + 0.08 * negotiation[..., None] * (1.0 - np.abs(own_progress))
                - 0.10 * control_cost
            )
            value += (0.96 ** step) * flow
        # Distributionally robust rolling objective: retain expected value but
        # explicitly penalize the adverse transition scenario.
        self.candidate_batch_calls += 1
        return 0.75 * np.mean(value, axis=2) + 0.25 * np.min(value, axis=2)

    def update(
        self, *, progress: np.ndarray, stocks: np.ndarray, damage: np.ndarray,
        economic_capacity: np.ndarray, support: np.ndarray,
        command: np.ndarray, intervention: np.ndarray,
        max_iterations: int = 36, tolerance: float = 1e-4,
    ) -> GameFactors:
        belief = self._belief_update({
            "progress": progress, "stocks": stocks, "damage": damage,
            "economic_capacity": economic_capacity, "support": support,
            "command": command,
        })
        controls = self.controls.copy()
        values = np.zeros((self.paths, 4))
        residual = np.inf
        # US leader signal, then Japan and Taiwan, then China best response.
        update_order = (2, 3, 1, 0)
        for _ in range(max_iterations):
            previous = controls.copy()
            for player in update_order:
                candidate_values = self._evaluate_candidates(
                    player, self.candidates, controls, belief["progress"],
                    belief["stocks"], belief["damage"],
                    belief["economic_capacity"], belief["support"],
                    belief["command"], intervention,
                )
                centered = candidate_values - np.max(candidate_values, axis=1, keepdims=True)
                response_weights = np.exp(np.clip(centered / 0.045, -35.0, 0.0))
                response_weights /= np.maximum(
                    response_weights.sum(axis=1, keepdims=True), 1e-12
                )
                selected = response_weights @ self.candidates
                controls[:, player, :] = 0.45 * controls[:, player, :] + 0.55 * selected
                controls[:, player, :] /= np.maximum(
                    controls[:, player, :].sum(axis=1, keepdims=True), 1e-12
                )
                values[:, player] = np.sum(response_weights * candidate_values, axis=1)
            residual = float(np.max(np.abs(controls - previous)))
            if residual < tolerance:
                break
        self.controls = controls
        self.last_residual = residual
        if residual >= tolerance:
            self.nonconverged_months += 1
            self.max_nonconverged_residual = max(
                self.max_nonconverged_residual, residual
            )
        return GameFactors(
            controls, values, residual, self.last_belief_residual,
            residual < tolerance,
        )

    def diagnostics(self) -> dict:
        return {
            "last_best_response_residual": self.last_residual,
            "last_belief_innovation": self.last_belief_residual,
            "maximum_belief_variance": float(max(
                np.max(value) for value in self._belief_variance.values()
            )) if self._belief_variance is not None else None,
            "simplex_residual": float(np.max(np.abs(self.controls.sum(axis=2) - 1.0))),
            "minimum_control": float(np.min(self.controls)),
            "nonconverged_months": self.nonconverged_months,
            "max_nonconverged_residual": self.max_nonconverged_residual,
            "candidate_evaluation_backend": "vectorized_candidate_batch",
            "candidate_batch_calls": self.candidate_batch_calls,
        }
