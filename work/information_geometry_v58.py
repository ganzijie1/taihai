from __future__ import annotations

from dataclasses import dataclass

import numpy as np


ACTORS = ("china", "taiwan", "united_states", "japan")
SOURCES = (
    "official", "legacy_media", "platform_media", "social_network",
    "foreign_media", "direct_experience",
)
RELIABILITY_STATES = ("reliable", "selective", "misleading")


def _normalise(values: np.ndarray, axis: int = -1) -> np.ndarray:
    values = np.maximum(values, 1.0e-300)
    return values / np.maximum(np.sum(values, axis=axis, keepdims=True), 1.0e-300)


def _logsumexp(values: np.ndarray, axis: int = -1, keepdims: bool = False):
    maximum = np.max(values, axis=axis, keepdims=True)
    result = maximum + np.log(np.maximum(
        np.sum(np.exp(values - maximum), axis=axis, keepdims=True), 1.0e-300
    ))
    return result if keepdims else np.squeeze(result, axis=axis)


@dataclass
class InformationGeometryFactors:
    truth: np.ndarray
    posterior_mean: np.ndarray
    posterior_std: np.ndarray
    self_belief: np.ndarray
    self_truth: np.ndarray
    robust_self_belief: np.ndarray
    credibility: np.ndarray
    observation_uncertainty: np.ndarray
    bridge_cost: np.ndarray
    fgw_alignment: np.ndarray
    ambiguity_radius: np.ndarray
    source_allocation: np.ndarray
    expected_information_value: np.ndarray


class PartiallyObservableInformationGeometry:
    """Pathwise belief, credibility, bridge and network-alignment system.

    Truth is supplied by the owning campaign/economic modules.  This class owns
    only actor-specific beliefs, source-reliability posteriors and the lagged
    information-action channel.  No future observations are used.
    """

    def __init__(
        self, paths: int, seed: int, initial_support: np.ndarray,
        actor_kernel: np.ndarray, age_kernel: np.ndarray,
        grid_points: int = 9,
        ambiguity_radius: float = 0.025,
        belief_filter_enabled: bool = True,
        credibility_hmm_enabled: bool = True,
        schrodinger_bridge_enabled: bool = True,
        fgw_alignment_enabled: bool = True,
        robust_information_enabled: bool = True,
        information_value_enabled: bool = True,
    ):
        if grid_points < 7 or grid_points % 2 == 0:
            raise ValueError("belief grid must be odd and at least seven")
        self.paths = int(paths)
        self.rng = np.random.default_rng(seed)
        self.grid = np.linspace(-1.0, 1.0, grid_points)
        self.ambiguity_radius_base = float(ambiguity_radius)
        self.belief_filter_enabled = bool(belief_filter_enabled)
        self.credibility_hmm_enabled = bool(credibility_hmm_enabled)
        self.schrodinger_bridge_enabled = bool(schrodinger_bridge_enabled)
        self.fgw_alignment_enabled = bool(fgw_alignment_enabled)
        self.robust_information_enabled = bool(robust_information_enabled)
        self.information_value_enabled = bool(information_value_enabled)
        self.month = 0
        self.transition = self._transition_matrix(0.16)
        self.bridge_kernel = self._transition_matrix(0.24)
        self.bridge_cost_matrix = 0.5 * np.square(
            self.grid[:, None] - self.grid[None, :]
        )

        initial_center = np.clip(
            2.0 * np.asarray(initial_support, dtype=float) - 1.0, -0.90, 0.90
        )
        target_centers = np.broadcast_to(
            initial_center[:, None, :, None],
            (paths, len(ACTORS), len(ACTORS), 1),
        )
        observer_bias = np.array([0.03, -0.01, -0.02, -0.01])[None, :, None, None]
        posterior = np.exp(-0.5 * np.square(
            (self.grid[None, None, None, :] - target_centers - observer_bias) / 0.30
        ))
        self.posterior = _normalise(posterior)

        base_reliability = np.array([0.76, 0.19, 0.05])
        self.reliability = np.broadcast_to(
            base_reliability,
            (paths, len(ACTORS), len(SOURCES), len(RELIABILITY_STATES)),
        ).copy()
        self.lagged_information_action = np.zeros((paths, len(ACTORS)))
        self.source_exposure = np.array([
            [0.88, 0.64, 0.72, 0.58, 0.24, 0.52],
            [0.70, 0.74, 0.84, 0.72, 0.46, 0.54],
            [0.58, 0.78, 0.86, 0.68, 0.58, 0.50],
            [0.72, 0.82, 0.80, 0.66, 0.50, 0.52],
        ])
        self.source_noise = np.array([0.12, 0.16, 0.20, 0.24, 0.22, 0.10])
        self.state_noise_scale = np.array([0.85, 1.35, 2.10])
        self.state_bias_scale = np.array([0.00, 0.10, 0.24])
        self.fgw_alignment, self.fgw_cost = self._fgw_alignment(
            np.asarray(actor_kernel, dtype=float), np.asarray(age_kernel, dtype=float),
            np.mean(initial_center, axis=0),
        )
        if not self.fgw_alignment_enabled:
            self.fgw_alignment = np.ones_like(self.fgw_alignment)

        self.max_posterior_simplex_residual = 0.0
        self.max_reliability_simplex_residual = 0.0
        self.max_bridge_marginal_residual = 0.0
        self.max_fgw_marginal_residual = 0.0
        self.max_ambiguity_radius = 0.0
        self.max_observation_uncertainty = 0.0
        self.max_belief_truth_gap = 0.0
        self.max_information_budget_residual = 0.0
        self.brier_sum = 0.0
        self.log_score_sum = 0.0
        self.score_count = 0

    def _transition_matrix(self, width: float) -> np.ndarray:
        kernel = np.exp(
            -0.5 * np.square(
                (self.grid[:, None] - self.grid[None, :]) / max(width, 1.0e-6)
            )
        )
        return _normalise(kernel, axis=1)

    @staticmethod
    def _sinkhorn_coupling(
        left: np.ndarray, right: np.ndarray, kernel: np.ndarray,
        iterations: int = 20,
    ) -> tuple[np.ndarray, float]:
        left = _normalise(left)
        right = _normalise(right)
        u = np.ones_like(left)
        v = np.ones_like(right)
        for _ in range(iterations):
            u = left / np.maximum(np.einsum("ij,...j->...i", kernel, v), 1.0e-15)
            v = right / np.maximum(np.einsum("ji,...j->...i", kernel, u), 1.0e-15)
        coupling = (
            u[..., :, None] * kernel
            * v[..., None, :]
        )
        residual = float(max(
            np.max(np.abs(np.sum(coupling, axis=-1) - left)),
            np.max(np.abs(np.sum(coupling, axis=-2) - right)),
        ))
        return coupling, residual

    def _fgw_alignment(
        self, actor_kernel: np.ndarray, age_kernel: np.ndarray,
        initial_center: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        ages = age_kernel.shape[0]
        weights = np.full(ages, 1.0 / ages)
        age_axis = np.linspace(-1.0, 1.0, ages)
        structures = []
        features = []
        for actor in range(len(ACTORS)):
            tilt = 0.06 * (actor - 1.5)
            structure = np.maximum(
                age_kernel * (1.0 + tilt * (age_axis[:, None] - age_axis[None, :])),
                1.0e-6,
            )
            structures.append(_normalise(structure, axis=1))
            features.append(initial_center[actor] + 0.10 * age_axis)
        structures = np.asarray(structures)
        features = np.asarray(features)
        costs = np.zeros((len(ACTORS), len(ACTORS)))
        max_residual = 0.0
        for a in range(len(ACTORS)):
            for b in range(len(ACTORS)):
                feature_cost = np.square(features[a][:, None] - features[b][None, :])
                coupling = np.outer(weights, weights)
                for _ in range(12):
                    structural_cost = np.zeros((ages, ages))
                    # Explicit four-index contraction is small and avoids a
                    # fragile tensor identity in this estimator.
                    for i in range(ages):
                        for j in range(ages):
                            structural_cost[i, j] = np.sum(
                                np.square(
                                    structures[a][i, :, None]
                                    - structures[b][j, None, :]
                                ) * coupling
                            )
                    fused = 0.45 * feature_cost + 0.55 * structural_cost
                    kernel = np.exp(-fused / 0.10)
                    coupling, residual = self._sinkhorn_coupling(
                        weights, weights, kernel, iterations=24
                    )
                    max_residual = max(max_residual, residual)
                costs[a, b] = float(np.sum(fused * coupling))
        scale = max(float(np.median(costs[costs > 0.0])), 1.0e-4)
        similarity = np.exp(-costs / scale) * np.maximum(actor_kernel, 1.0e-4)
        similarity = _normalise(similarity, axis=1)
        self.max_fgw_marginal_residual = max_residual
        return similarity, costs

    def _credibility_transition(
        self, shock: np.ndarray, network: np.ndarray,
    ) -> np.ndarray:
        base = np.array([
            [0.91, 0.075, 0.015],
            [0.16, 0.76, 0.08],
            [0.08, 0.22, 0.70],
        ])
        matrix = np.broadcast_to(
            base,
            (self.paths, len(ACTORS), len(SOURCES), 3, 3),
        ).copy()
        pressure = np.clip(
            0.10 * shock[:, :, None]
            + 0.06 * (1.0 - network[:, :, None])
            + 0.05 * np.abs(self.lagged_information_action[:, :, None]),
            0.0, 0.20,
        )
        matrix[..., 0, 0] -= pressure
        matrix[..., 0, 1] += 0.75 * pressure
        matrix[..., 0, 2] += 0.25 * pressure
        matrix[..., 1, 0] -= 0.20 * pressure
        matrix[..., 1, 1] -= 0.25 * pressure
        matrix[..., 1, 2] += 0.45 * pressure
        return _normalise(matrix, axis=-1)

    def _schrodinger_bridge(
        self, opening: np.ndarray, closing: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        coupling, residual = self._sinkhorn_coupling(
            opening, closing, self.bridge_kernel, iterations=18
        )
        cost = np.sum(coupling * self.bridge_cost_matrix, axis=(-2, -1))
        self.max_bridge_marginal_residual = max(
            self.max_bridge_marginal_residual, residual
        )
        return cost, coupling

    def update(
        self, *, progress: np.ndarray, damage: np.ndarray,
        shortage: np.ndarray, inflation: np.ndarray,
        financial_stress: np.ndarray, command: np.ndarray,
        network: np.ndarray, common_noise: np.ndarray,
        jump_signal: np.ndarray,
    ) -> InformationGeometryFactors:
        progress = np.asarray(progress, dtype=float)
        truth = np.clip(
            0.38 * progress + 0.19 * command + 0.12 * network
            - 0.13 * damage - 0.08 * shortage
            - 0.06 * financial_stress - 0.04 * np.clip(inflation, 0.0, 1.0),
            -1.0, 1.0,
        )
        if self.transition.ndim == 3:
            predicted = np.einsum("potg,pgh->poth", self.posterior, self.transition)
        else:
            predicted = np.einsum("potg,gh->poth", self.posterior, self.transition)

        if self.credibility_hmm_enabled:
            transition = self._credibility_transition(jump_signal, network)
            reliability_predicted = np.einsum(
                "posi,posij->posj", self.reliability, transition
            )
        else:
            reliability_predicted = self.reliability.copy()
        expected_state = np.sum(
            reliability_predicted * np.arange(3)[None, None, None, :], axis=-1
        )
        expected_sigma = (
            self.source_noise[None, None, :]
            * np.sum(
                reliability_predicted * self.state_noise_scale[None, None, None, :],
                axis=-1,
            )
        )
        source_direction = np.array([1.0, 0.6, 0.8, 0.4, -0.25, 0.0])
        source_bias = (
            expected_state[..., None]
            * self.state_bias_scale[-1]
            * source_direction[None, None, :, None]
            * self.lagged_information_action[:, :, None, None]
        )
        observation_noise = self.rng.normal(
            0.0, 1.0,
            (self.paths, len(ACTORS), len(SOURCES), len(ACTORS)),
        )
        observations = np.clip(
            truth[:, None, None, :]
            + source_bias
            + expected_sigma[..., None] * observation_noise,
            -1.25, 1.25,
        )

        residual = np.mean(
            np.abs(observations - truth[:, None, None, :]), axis=-1
        )
        emission_sigma = (
            self.source_noise[None, None, :, None]
            * self.state_noise_scale[None, None, None, :]
        )
        emission = np.exp(
            -0.5 * np.square(residual[..., None] / np.maximum(emission_sigma, 1e-6))
        ) / np.maximum(emission_sigma, 1e-6)
        if self.credibility_hmm_enabled:
            self.reliability = _normalise(reliability_predicted * emission)
        credibility = self.reliability[..., 0] + 0.48 * self.reliability[..., 1]

        prior_mean = np.sum(
            predicted * self.grid[None, None, None, :], axis=-1
        )
        prior_variance = np.sum(
            predicted * np.square(
                self.grid[None, None, None, :] - prior_mean[..., None]
            ), axis=-1,
        )
        information_budget = np.clip(
            0.45 + 0.35 * np.abs(self.lagged_information_action), 0.25, 0.80
        )
        source_exposure = np.asarray(self.source_exposure, dtype=float)
        if source_exposure.ndim == 2:
            source_exposure = source_exposure[None, :, :]
        if source_exposure.shape not in (
            (1, len(ACTORS), len(SOURCES)),
            (self.paths, len(ACTORS), len(SOURCES)),
        ):
            raise ValueError("source exposure must be [actor,source] or [path,actor,source]")
        source_score = (
            source_exposure
            * credibility
            * (0.15 + np.mean(prior_variance, axis=2)[:, :, None])
        )
        if self.information_value_enabled:
            allocation_weights = _normalise(source_score, axis=2)
        else:
            allocation_weights = np.full_like(source_score, 1.0 / len(SOURCES))
        source_allocation = information_budget[:, :, None] * allocation_weights
        budget_residual = float(np.max(np.abs(
            np.sum(source_allocation, axis=2) - information_budget
        )))
        self.max_information_budget_residual = max(
            self.max_information_budget_residual, budget_residual
        )
        allocation_share = source_allocation / np.maximum(
            information_budget[:, :, None], 1.0e-12
        )
        exposure = (
            source_exposure[..., None]
            * (0.65 + 0.70 * allocation_share[..., None])
        )
        weight = exposure * credibility[..., None]
        sigma = np.maximum(expected_sigma[..., None, None], 0.06)
        innovation = (
            observations[..., None]
            - self.grid[None, None, None, None, :]
        )
        log_likelihood = -0.5 * np.square(innovation / sigma)
        weighted_log_likelihood = np.sum(
            weight[..., None] * log_likelihood, axis=2
        ) / np.maximum(np.sum(weight, axis=2)[..., None], 1.0e-12)
        if self.belief_filter_enabled:
            log_posterior = (
                np.log(np.maximum(predicted, 1.0e-300)) + weighted_log_likelihood
            )
            log_posterior -= _logsumexp(log_posterior, axis=-1, keepdims=True)
            updated = np.exp(log_posterior)
        else:
            direct = np.exp(-0.5 * np.square(
                (self.grid[None, None, None, :] - truth[:, None, :, None]) / 0.12
            ))
            updated = _normalise(np.broadcast_to(
                direct, (self.paths, len(ACTORS), len(ACTORS), len(self.grid))
            ).copy())

        if self.schrodinger_bridge_enabled:
            bridge_cost, _ = self._schrodinger_bridge(self.posterior, updated)
        else:
            bridge_cost = np.zeros(updated.shape[:-1])
        self.posterior = _normalise(updated)
        posterior_mean = np.sum(
            self.posterior * self.grid[None, None, None, :], axis=-1
        )
        posterior_variance = np.sum(
            self.posterior
            * np.square(self.grid[None, None, None, :] - posterior_mean[..., None]),
            axis=-1,
        )
        posterior_std = np.sqrt(np.maximum(posterior_variance, 0.0))
        expected_information_value = (
            np.maximum(np.mean(prior_variance - posterior_variance, axis=2), 0.0)
            if self.information_value_enabled else np.zeros_like(truth)
        )

        if self.robust_information_enabled:
            ambiguity = np.clip(
                self.ambiguity_radius_base
                + 0.035 * jump_signal + 0.020 * np.abs(common_noise)
                + 0.020 * (1.0 - np.mean(credibility, axis=2)),
                0.005, 0.18,
            )
        else:
            ambiguity = np.zeros_like(truth)
        index = np.arange(len(ACTORS))
        self_belief = posterior_mean[:, index, index]
        self_truth = truth
        self_std = posterior_std[:, index, index]
        robust_self = np.clip(
            self_belief - np.sqrt(2.0 * ambiguity) * self_std,
            -1.0, 1.0,
        )
        observation_uncertainty = np.mean(posterior_std, axis=2)

        positive_probability = np.sum(
            self.posterior[..., self.grid > 0.0], axis=-1
        )
        observed_positive = (truth[:, None, :] > 0.0).astype(float)
        self.brier_sum += float(np.sum(np.square(
            positive_probability - observed_positive
        )))
        selected_probability = np.where(
            observed_positive > 0.5, positive_probability, 1.0 - positive_probability
        )
        self.log_score_sum += float(-np.sum(np.log(
            np.maximum(selected_probability, 1.0e-12)
        )))
        self.score_count += positive_probability.size

        self.max_posterior_simplex_residual = max(
            self.max_posterior_simplex_residual,
            float(np.max(np.abs(np.sum(self.posterior, axis=-1) - 1.0))),
        )
        self.max_reliability_simplex_residual = max(
            self.max_reliability_simplex_residual,
            float(np.max(np.abs(np.sum(self.reliability, axis=-1) - 1.0))),
        )
        self.max_ambiguity_radius = max(
            self.max_ambiguity_radius, float(np.max(ambiguity))
        )
        self.max_observation_uncertainty = max(
            self.max_observation_uncertainty, float(np.max(observation_uncertainty))
        )
        self.max_belief_truth_gap = max(
            self.max_belief_truth_gap,
            float(np.max(np.abs(self_belief - self_truth))),
        )
        self.month += 1
        return InformationGeometryFactors(
            truth=truth,
            posterior_mean=posterior_mean,
            posterior_std=posterior_std,
            self_belief=self_belief,
            self_truth=self_truth,
            robust_self_belief=robust_self,
            credibility=credibility,
            observation_uncertainty=observation_uncertainty,
            bridge_cost=bridge_cost,
            fgw_alignment=self.fgw_alignment.copy(),
            ambiguity_radius=ambiguity,
            source_allocation=source_allocation,
            expected_information_value=expected_information_value,
        )

    def register_information_action(self, major_action: np.ndarray) -> None:
        platform = major_action[..., 3]
        government = major_action[..., 0]
        self.lagged_information_action = np.clip(
            0.62 * platform + 0.38 * government, -1.0, 1.0
        )

    def diagnostics(self) -> dict:
        return {
            "enabled": True,
            "actors": list(ACTORS),
            "sources": list(SOURCES),
            "reliability_states": list(RELIABILITY_STATES),
            "belief_grid_points": len(self.grid),
            "max_posterior_simplex_residual": self.max_posterior_simplex_residual,
            "max_reliability_simplex_residual": self.max_reliability_simplex_residual,
            "max_bridge_marginal_residual": self.max_bridge_marginal_residual,
            "max_fgw_marginal_residual": self.max_fgw_marginal_residual,
            "max_ambiguity_radius": self.max_ambiguity_radius,
            "max_observation_uncertainty": self.max_observation_uncertainty,
            "max_belief_truth_gap": self.max_belief_truth_gap,
            "max_information_budget_residual": self.max_information_budget_residual,
            "mean_brier_score": self.brier_sum / max(self.score_count, 1),
            "mean_log_score": self.log_score_sum / max(self.score_count, 1),
            "fgw_cost": self.fgw_cost.tolist(),
            "fgw_alignment": self.fgw_alignment.tolist(),
            "observation_clock": "monthly same-path, current-state observation",
            "information_action_lag": "one explicit month",
            "schrodinger_role": "causal-filter estimator and path-friction diagnostic",
            "fgw_role": "network-kernel alignment estimator",
            "identification": "structural scenario prior pending held-out public-data calibration",
            "feature_flags": {
                "belief_filter": self.belief_filter_enabled,
                "credibility_hmm": self.credibility_hmm_enabled,
                "schrodinger_bridge": self.schrodinger_bridge_enabled,
                "fgw_alignment": self.fgw_alignment_enabled,
                "robust_information": self.robust_information_enabled,
                "information_value": self.information_value_enabled,
            },
        }
