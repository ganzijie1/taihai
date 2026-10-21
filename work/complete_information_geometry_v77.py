from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import os

import numpy as np
from numba import njit, prange
from scipy.optimize import least_squares, minimize, root
import warnings

from deep_optimal_transport_v66 import DeepCausalOptimalTransportGeometryV66
from information_geometry_v58 import ACTORS, SOURCES, _normalise


@njit(cache=True, parallel=True)
def _parallel_blahut_arimoto(prior: np.ndarray, encoder: np.ndarray,
                              observation: np.ndarray, beta: float,
                              tolerance: float, max_iterations: int,
                              support_threshold: float = 1.0e-7,
                              support_mass_limit: float = 1.0e-7):
    """Solve each local information-bottleneck problem to its own tolerance."""
    count, states = prior.shape
    latent = encoder.shape[2]
    residuals = np.zeros(count)
    raw_conditional_residuals = np.zeros(count)
    iterations = np.zeros(count, dtype=np.int64)
    distortions = np.zeros((count, states, latent))
    objective_increases = np.zeros(count)
    for item in prange(count):
        active_support = np.ones(states, dtype=np.bool_)
        order = np.argsort(prior[item])
        excluded_mass = 0.0
        for position in range(states):
            x = order[position]
            if prior[item, x] >= support_threshold:
                break
            if excluded_mass + prior[item, x] <= support_mass_limit:
                active_support[x] = False
                excluded_mass += prior[item, x]
            else:
                break
        pz = np.empty(latent)
        pyz = np.empty((latent, states))
        updated = np.empty((states, latent))
        local_distortion = np.empty((states, latent))
        local_residual = 1.0
        raw_conditional_residual = 1.0
        previous_objective = 1.0e300
        maximum_objective_increase = 0.0
        used = 0
        for iteration in range(max_iterations):
            for z in range(latent):
                mass = 0.0
                for x in range(states):
                    mass += prior[item, x] * encoder[item, x, z]
                pz[z] = max(mass, 1.0e-300)
                for y in range(states):
                    value = 0.0
                    for x in range(states):
                        value += (prior[item, x] * encoder[item, x, z]
                                  * observation[item, x, y])
                    pyz[z, y] = max(value / pz[z], 1.0e-300)
            local_residual = 0.0
            raw_conditional_residual = 0.0
            objective = 0.0
            for x in range(states):
                maximum_log_weight = -1.0e300
                for z in range(latent):
                    divergence = 0.0
                    for y in range(states):
                        likelihood = max(observation[item, x, y], 1.0e-300)
                        divergence += likelihood * (
                            np.log(likelihood) - np.log(pyz[z, y])
                        )
                    local_distortion[x, z] = divergence
                    if encoder[item, x, z] > 0.0:
                        objective += prior[item, x] * encoder[item, x, z] * (
                            np.log(encoder[item, x, z]) - np.log(pz[z])
                            + beta * divergence
                        )
                    updated[x, z] = np.log(pz[z]) - beta * divergence
                    maximum_log_weight = max(maximum_log_weight, updated[x, z])
                row_sum = 0.0
                for z in range(latent):
                    updated[x, z] = np.exp(updated[x, z] - maximum_log_weight)
                    row_sum += updated[x, z]
                row_sum = max(row_sum, 1.0e-300)
                for z in range(latent):
                    updated[x, z] /= row_sum
                    difference = abs(updated[x, z] - encoder[item, x, z])
                    raw_conditional_residual = max(
                        raw_conditional_residual, difference
                    )
                    local_residual = max(
                        local_residual, prior[item, x] * difference
                    )
                    if active_support[x]:
                        local_residual = max(local_residual, difference)
            if previous_objective < 1.0e299:
                maximum_objective_increase = max(
                    maximum_objective_increase, objective - previous_objective
                )
            previous_objective = objective
            for x in range(states):
                for z in range(latent):
                    encoder[item, x, z] = updated[x, z]
            used = iteration + 1
            if local_residual <= tolerance:
                break
        residuals[item] = local_residual
        raw_conditional_residuals[item] = raw_conditional_residual
        iterations[item] = used
        objective_increases[item] = maximum_objective_increase
        for x in range(states):
            for z in range(latent):
                distortions[item, x, z] = local_distortion[x, z]
    return (encoder, residuals, raw_conditional_residuals, iterations,
            distortions, objective_increases)


def _ib_map(encoder: np.ndarray, prior: np.ndarray,
            observation: np.ndarray, beta: float) -> tuple[np.ndarray, np.ndarray]:
    pz = np.sum(prior[:, None] * encoder, axis=0)
    pyz = np.einsum("x,xz,xy->zy", prior, encoder, observation) / np.maximum(
        pz[:, None], 1.0e-300
    )
    distortion = np.sum(
        observation[:, None, :] * (
            np.log(np.maximum(observation[:, None, :], 1.0e-300))
            - np.log(np.maximum(pyz[None, :, :], 1.0e-300))
        ),
        axis=-1,
    )
    logits = np.log(np.maximum(pz[None, :], 1.0e-300)) - beta * distortion
    logits -= np.max(logits, axis=1, keepdims=True)
    updated = np.exp(logits)
    updated /= np.maximum(np.sum(updated, axis=1, keepdims=True), 1.0e-300)
    return updated, distortion


def _ib_objective(encoder: np.ndarray, prior: np.ndarray,
                  observation: np.ndarray, beta: float) -> float:
    _, distortion = _ib_map(encoder, prior, observation, beta)
    pz = np.sum(prior[:, None] * encoder, axis=0)
    rate = np.sum(
        prior[:, None] * encoder * (
            np.log(np.maximum(encoder, 1.0e-300))
            - np.log(np.maximum(pz[None, :], 1.0e-300))
        )
    )
    loss = np.sum(prior[:, None] * encoder * distortion)
    return float(rate + beta * loss)


def _ib_logit_fixed_point(theta: np.ndarray, prior: np.ndarray,
                          observation: np.ndarray, beta: float,
                          states: int, latent: int) -> np.ndarray:
    logits = np.concatenate(
        (theta.reshape(states, latent - 1), np.zeros((states, 1))), axis=1
    )
    logits -= np.max(logits, axis=1, keepdims=True)
    encoder = np.exp(logits)
    encoder /= np.sum(encoder, axis=1, keepdims=True)
    updated, _ = _ib_map(encoder, prior, observation, beta)
    target = (
        np.log(np.maximum(updated[:, :-1], 1.0e-300))
        - np.log(np.maximum(updated[:, -1:], 1.0e-300))
    )
    return (target - theta.reshape(states, latent - 1)).ravel()


class CompleteInformationGeometryV77(DeepCausalOptimalTransportGeometryV66):
    """Causal active sensing plus a discrete information-bottleneck state.

    Source attention is solved pathwise from expected entropy reduction with an
    entropic resource cost.  Its exposure plan becomes effective one month
    later.  The posterior is then compressed by the classical Blahut--Arimoto
    information-bottleneck equations on the full nine-point belief grid.
    """

    def __init__(self, *args, bottleneck_states: int = 3,
                 bottleneck_beta: float = 4.0,
                 attention_temperature: float = 0.16, **kwargs):
        super().__init__(*args, **kwargs)
        self.bottleneck_states = max(2, min(int(bottleneck_states), len(self.grid)))
        self.bottleneck_beta = float(max(bottleneck_beta, 1.0e-3))
        self.attention_temperature = float(max(attention_temperature, 1.0e-3))
        self.base_source_exposure = np.asarray(self.source_exposure, dtype=float).copy()
        self.planned_source_exposure = np.broadcast_to(
            self.base_source_exposure, (self.paths,) + self.base_source_exposure.shape
        ).copy()
        self.maximum_attention_kkt_residual = 0.0
        self.maximum_ib_fixed_point_residual = 0.0
        self.maximum_ib_rate = 0.0
        self.maximum_ib_distortion = 0.0
        self.maximum_bottleneck_state_change = 0.0
        self.active_observation_updates = 0
        self.ib_updates = 0
        self._ib_encoder: np.ndarray | None = None
        self.maximum_ib_iterations = 0
        self.total_ib_iterations = 0
        self.maximum_ib_objective_increase = 0.0
        self.maximum_ib_raw_conditional_residual = 0.0
        self.ib_support_threshold = 1.0e-7
        self.ib_support_mass_limit = 1.0e-7
        self.maximum_ib_excluded_prior_mass = 0.0
        self.ib_continuation_fallbacks = 0
        self.ib_broyden_fallbacks = 0
        self.ib_least_squares_fallbacks = 0
        self.ib_lbfgsb_fallbacks = 0
        self.ib_final_polishing_fallbacks = 0

    @staticmethod
    def _entropy(probability: np.ndarray, axis: int = -1) -> np.ndarray:
        probability = np.maximum(probability, 1.0e-300)
        return -np.sum(probability * np.log(probability), axis=axis)

    def _active_observation_plan(self) -> np.ndarray:
        posterior_variance = np.sum(
            self.posterior * np.square(
                self.grid[None, None, None, :]
                - np.sum(self.posterior * self.grid, axis=-1)[..., None]
            ),
            axis=-1,
        )
        uncertainty = np.mean(posterior_variance, axis=2)
        reliability = self.reliability[..., 0] + 0.48 * self.reliability[..., 1]
        effective_sigma = (
            self.source_noise[None, None, :]
            * np.sum(
                self.reliability * self.state_noise_scale[None, None, None, :],
                axis=-1,
            )
        )
        information_gain = 0.5 * np.log1p(
            uncertainty[..., None] / np.maximum(effective_sigma ** 2, 1.0e-8)
        ) * reliability
        baseline = _normalise(
            np.broadcast_to(
                self.base_source_exposure,
                (self.paths,) + self.base_source_exposure.shape,
            ),
            axis=2,
        )
        logits = (
            information_gain / self.attention_temperature
            + np.log(np.maximum(baseline, 1.0e-300))
        )
        logits -= np.max(logits, axis=2, keepdims=True)
        share = _normalise(np.exp(logits), axis=2)
        # KKT stationarity for max_q <q,g>-tau KL(q||q0), sum q=1.
        gradient = information_gain - self.attention_temperature * (
            np.log(np.maximum(share, 1.0e-300))
            - np.log(np.maximum(baseline, 1.0e-300)) + 1.0
        )
        residual = gradient - np.mean(gradient, axis=2, keepdims=True)
        self.maximum_attention_kkt_residual = max(
            self.maximum_attention_kkt_residual, float(np.max(np.abs(residual)))
        )
        budget = np.clip(
            0.45 + 0.35 * np.abs(self.lagged_information_action), 0.25, 0.80
        )
        planned = budget[..., None] * share
        self.active_observation_updates += 1
        return np.clip(planned, 1.0e-6, 1.0)

    def _information_bottleneck(self) -> tuple[np.ndarray, float, float, float]:
        prior = _normalise(self.posterior)
        states = len(self.grid)
        latent = self.bottleneck_states
        centers = np.linspace(-0.85, 0.85, latent)
        canonical = np.exp(
            -np.square(self.grid[None, :] - centers[:, None]) / 0.16
        ).T
        canonical = _normalise(canonical, axis=1)
        canonical = np.broadcast_to(canonical, prior.shape + (latent,)).copy()
        if self._ib_encoder is None or self._ib_encoder.shape != prior.shape + (latent,):
            q = canonical.copy()
        else:
            q = self._ib_encoder.copy()

        credibility = np.mean(
            self.reliability[..., 0] + 0.48 * self.reliability[..., 1], axis=2
        )
        sigma = np.clip(0.32 - 0.18 * credibility, 0.08, 0.32)
        observation = np.exp(
            -0.5 * np.square(
                self.grid[None, None, None, :, None]
                - self.grid[None, None, None, None, :]
            ) / np.maximum(sigma[:, :, None, None, None] ** 2, 1.0e-8)
        )
        observation = _normalise(observation, axis=-1)
        original_shape = prior.shape
        flat_prior = np.ascontiguousarray(prior.reshape(-1, states))
        excluded_mass = np.zeros(len(flat_prior))
        for item, row in enumerate(flat_prior):
            running = 0.0
            for value in np.sort(row):
                if value >= self.ib_support_threshold:
                    break
                if running + value <= self.ib_support_mass_limit:
                    running += value
                else:
                    break
            excluded_mass[item] = running
        self.maximum_ib_excluded_prior_mass = max(
            self.maximum_ib_excluded_prior_mass, float(np.max(excluded_mass))
        )
        flat_encoder = np.ascontiguousarray(q.reshape(-1, states, latent))
        expanded_observation = np.broadcast_to(
            observation, prior.shape[:-1] + (states, states)
        )
        flat_observation = np.ascontiguousarray(
            expanded_observation.reshape(-1, states, states)
        )
        (flat_encoder, residuals, raw_conditional_residuals, iteration_counts,
         flat_distortion, objective_increases) = (
            _parallel_blahut_arimoto(
                flat_prior, flat_encoder, flat_observation,
                self.bottleneck_beta, 5.0e-8, 5000,
                self.ib_support_threshold, self.ib_support_mass_limit,
            )
        )
        failed = residuals > 1.0e-7
        if np.any(failed):
            continuation_encoder = canonical.reshape(-1, states, latent)[failed].copy()
            continuation_prior = flat_prior[failed]
            continuation_observation = flat_observation[failed]
            continuation_iterations = np.zeros(np.sum(failed), dtype=np.int64)
            continuation_increase = np.zeros(np.sum(failed))
            for continuation_beta in (8.0, 6.0, 5.0, self.bottleneck_beta):
                (continuation_encoder, continuation_residual,
                 continuation_raw_residual, used, continuation_distortion,
                 increase) = _parallel_blahut_arimoto(
                    continuation_prior, continuation_encoder,
                    continuation_observation, continuation_beta, 5.0e-8, 12000,
                    self.ib_support_threshold, self.ib_support_mass_limit,
                )
                continuation_iterations += used
                continuation_increase = np.maximum(continuation_increase, increase)
            flat_encoder[failed] = continuation_encoder
            residuals[failed] = continuation_residual
            raw_conditional_residuals[failed] = continuation_raw_residual
            iteration_counts[failed] += continuation_iterations
            flat_distortion[failed] = continuation_distortion
            objective_increases[failed] = continuation_increase
            self.ib_continuation_fallbacks += int(np.sum(failed))
        unresolved = residuals > 1.0e-7
        if np.any(unresolved):
            unresolved_indices = np.flatnonzero(unresolved)
            accelerated = flat_encoder[unresolved].copy()
            accepted = np.zeros(len(unresolved_indices), dtype=bool)
            accepted_broyden = np.zeros(len(unresolved_indices), dtype=bool)
            for local, item in enumerate(unresolved_indices):
                opening_objective = _ib_objective(
                    accelerated[local], flat_prior[item],
                    flat_observation[item], self.bottleneck_beta,
                )
                theta = (
                    np.log(np.maximum(accelerated[local, :, :-1], 1.0e-12))
                    - np.log(np.maximum(accelerated[local, :, -1:], 1.0e-12))
                ).ravel()
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    solution = root(
                        _ib_logit_fixed_point, theta,
                        args=(flat_prior[item], flat_observation[item],
                              self.bottleneck_beta, states, latent),
                        method="broyden1",
                        options={"maxiter": 6000, "fatol": 1.0e-10},
                    )
                solve_method = "broyden"
                if (not np.all(np.isfinite(solution.x))) or np.max(np.abs(
                    _ib_logit_fixed_point(
                        solution.x, flat_prior[item], flat_observation[item],
                        self.bottleneck_beta, states, latent,
                    )
                )) > 1.0e-7:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        solution = least_squares(
                            _ib_logit_fixed_point, np.clip(theta, -35.0, 35.0),
                            bounds=(-35.0, 35.0),
                            args=(flat_prior[item], flat_observation[item],
                                  self.bottleneck_beta, states, latent),
                            method="trf", max_nfev=6000,
                            ftol=1.0e-12, xtol=1.0e-12, gtol=1.0e-12,
                        )
                    solve_method = "least_squares"
                logits = np.concatenate((
                    solution.x.reshape(states, latent - 1),
                    np.zeros((states, 1)),
                ), axis=1)
                logits -= np.max(logits, axis=1, keepdims=True)
                candidate = np.exp(logits)
                candidate /= np.sum(candidate, axis=1, keepdims=True)
                closing_objective = _ib_objective(
                    candidate, flat_prior[item], flat_observation[item],
                    self.bottleneck_beta,
                )
                if (not np.all(np.isfinite(candidate))) or (
                    closing_objective > opening_objective + 2.0e-10
                ):
                    def objective_from_theta(value):
                        local_logits = np.concatenate((
                            value.reshape(states, latent - 1),
                            np.zeros((states, 1)),
                        ), axis=1)
                        local_logits -= np.max(local_logits, axis=1, keepdims=True)
                        local_encoder = np.exp(local_logits)
                        local_encoder /= np.sum(local_encoder, axis=1, keepdims=True)
                        return _ib_objective(
                            local_encoder, flat_prior[item], flat_observation[item],
                            self.bottleneck_beta,
                        )
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        solution = minimize(
                            objective_from_theta, np.clip(theta, -35.0, 35.0),
                            method="L-BFGS-B", bounds=[(-35.0, 35.0)] * theta.size,
                            options={"maxiter": 1600, "ftol": 1.0e-14,
                                     "gtol": 1.0e-10, "maxls": 60},
                        )
                    logits = np.concatenate((
                        solution.x.reshape(states, latent - 1),
                        np.zeros((states, 1)),
                    ), axis=1)
                    logits -= np.max(logits, axis=1, keepdims=True)
                    candidate = np.exp(logits)
                    candidate /= np.sum(candidate, axis=1, keepdims=True)
                    closing_objective = _ib_objective(
                        candidate, flat_prior[item], flat_observation[item],
                        self.bottleneck_beta,
                    )
                    solve_method = "lbfgsb"
                if np.all(np.isfinite(candidate)) and (
                    closing_objective <= opening_objective + 2.0e-10
                ):
                    accelerated[local] = candidate
                    accepted[local] = True
                    if solve_method == "least_squares":
                        self.ib_least_squares_fallbacks += 1
                    elif solve_method == "lbfgsb":
                        self.ib_lbfgsb_fallbacks += 1
                    else:
                        accepted_broyden[local] = True
            if np.any(accepted):
                selected = unresolved_indices[accepted]
                (refined, refined_residual, refined_raw_residual,
                 refined_iterations, refined_distortion,
                 refined_increase) = _parallel_blahut_arimoto(
                    flat_prior[selected], accelerated[accepted],
                    flat_observation[selected], self.bottleneck_beta,
                    5.0e-8, 20000, self.ib_support_threshold,
                    self.ib_support_mass_limit,
                )
                flat_encoder[selected] = refined
                residuals[selected] = refined_residual
                raw_conditional_residuals[selected] = refined_raw_residual
                iteration_counts[selected] += refined_iterations
                flat_distortion[selected] = refined_distortion
                objective_increases[selected] = np.maximum(
                    objective_increases[selected], refined_increase
                )
                self.ib_broyden_fallbacks += int(np.sum(accepted_broyden))
        # Near-degenerate support can leave accelerated solvers just above the
        # acceptance gate even though the classical BA map remains monotone.
        # Polish only those unresolved rows with the original fixed-point map;
        # this changes neither the objective nor the feasible set.
        unresolved = residuals > 1.0e-7
        if np.any(unresolved):
            selected = np.flatnonzero(unresolved)
            (polished, polished_residual, polished_raw_residual,
             polished_iterations, polished_distortion,
             polished_increase) = _parallel_blahut_arimoto(
                flat_prior[selected], flat_encoder[selected].copy(),
                flat_observation[selected], self.bottleneck_beta,
                2.5e-8, 100000, self.ib_support_threshold,
                self.ib_support_mass_limit,
            )
            flat_encoder[selected] = polished
            residuals[selected] = polished_residual
            raw_conditional_residuals[selected] = polished_raw_residual
            iteration_counts[selected] += polished_iterations
            flat_distortion[selected] = polished_distortion
            objective_increases[selected] = np.maximum(
                objective_increases[selected], polished_increase
            )
            self.ib_final_polishing_fallbacks += len(selected)

        q = flat_encoder.reshape(original_shape + (latent,))
        distortion = flat_distortion.reshape(original_shape + (latent,))
        residual = float(np.max(residuals))
        iterations = int(np.max(iteration_counts))
        objective_increase = float(np.max(objective_increases))
        if residual > 1.0e-7:
            failed_items = np.flatnonzero(residuals > 1.0e-7)
            diagnostic_path = (
                Path(__file__).resolve().parents[1] / "outputs"
                / f"taiwan_v77_ib_failure_{os.getpid()}.npz"
            )
            np.savez_compressed(
                diagnostic_path,
                prior=flat_prior[failed_items],
                observation=flat_observation[failed_items],
                encoder=flat_encoder[failed_items],
                residual=residuals[failed_items],
                iteration_count=iteration_counts[failed_items],
                beta=np.array([self.bottleneck_beta]),
            )
            raise RuntimeError(
                "information-bottleneck BA failed to converge: "
                f"residual={residual:.3e}; diagnostic={diagnostic_path.name}"
            )
        if objective_increase > 2.0e-10:
            raise RuntimeError(
                "information-bottleneck BA objective is not monotone: "
                f"increase={objective_increase:.3e}"
            )

        self._ib_encoder = q.copy()
        self.maximum_ib_iterations = max(self.maximum_ib_iterations, iterations)
        self.total_ib_iterations += float(np.mean(iteration_counts))
        self.maximum_ib_objective_increase = max(
            self.maximum_ib_objective_increase, objective_increase
        )
        self.maximum_ib_raw_conditional_residual = max(
            self.maximum_ib_raw_conditional_residual,
            float(np.max(raw_conditional_residuals)),
        )

        pz = np.sum(prior[..., :, None] * q, axis=-2)
        rate = np.sum(
            prior[..., :, None] * q * (
                np.log(np.maximum(q, 1.0e-300))
                - np.log(np.maximum(pz[..., None, :], 1.0e-300))
            ),
            axis=(-2, -1),
        )
        mean_distortion = np.sum(
            prior[..., :, None] * q * distortion, axis=(-2, -1)
        )
        # Push the latent mass back to the physical belief grid.  This is the
        # executable compressed state used by next month's Bayesian filter.
        decoder = np.exp(
            -0.5 * np.square(
                self.grid[None, :] - centers[:, None]
            ) / 0.10
        )
        decoder = _normalise(decoder, axis=1)
        compressed = np.einsum("...z,zx->...x", pz, decoder)
        compressed = _normalise(compressed)
        return compressed, residual, float(np.max(rate)), float(np.max(mean_distortion))

    def update(self, **kwargs):
        self.source_exposure = self.planned_source_exposure
        factors = super().update(**kwargs)
        opening = self.posterior.copy()
        compressed, residual, rate, distortion = self._information_bottleneck()
        self.posterior = compressed
        posterior_mean = np.sum(compressed * self.grid, axis=-1)
        posterior_variance = np.sum(
            compressed * np.square(self.grid - posterior_mean[..., None]), axis=-1
        )
        posterior_std = np.sqrt(np.maximum(posterior_variance, 0.0))
        index = np.arange(len(ACTORS))
        self_belief = posterior_mean[:, index, index]
        self_std = posterior_std[:, index, index]
        robust = np.clip(
            self_belief
            - np.sqrt(2.0 * factors.ambiguity_radius) * self_std,
            -1.0,
            1.0,
        )
        self.maximum_ib_fixed_point_residual = max(
            self.maximum_ib_fixed_point_residual, residual
        )
        self.maximum_ib_rate = max(self.maximum_ib_rate, rate)
        self.maximum_ib_distortion = max(self.maximum_ib_distortion, distortion)
        self.maximum_bottleneck_state_change = max(
            self.maximum_bottleneck_state_change,
            float(np.max(np.abs(compressed - opening))),
        )
        self.ib_updates += 1
        self.planned_source_exposure = self._active_observation_plan()
        return replace(
            factors,
            posterior_mean=posterior_mean,
            posterior_std=posterior_std,
            self_belief=self_belief,
            robust_self_belief=robust,
            observation_uncertainty=np.mean(posterior_std, axis=2),
            source_allocation=self.planned_source_exposure.copy(),
        )

    def diagnostics(self) -> dict:
        payload = super().diagnostics()
        payload["v77_information_bottleneck_active_observation"] = {
            "active_observation_updates": self.active_observation_updates,
            "information_bottleneck_updates": self.ib_updates,
            "latent_states": self.bottleneck_states,
            "beta": self.bottleneck_beta,
            "attention_temperature": self.attention_temperature,
            "maximum_attention_kkt_residual": self.maximum_attention_kkt_residual,
            "maximum_ib_fixed_point_residual": self.maximum_ib_fixed_point_residual,
            "maximum_ib_rate": self.maximum_ib_rate,
            "maximum_ib_distortion": self.maximum_ib_distortion,
            "maximum_bottleneck_state_change": self.maximum_bottleneck_state_change,
            "maximum_ib_iterations": self.maximum_ib_iterations,
            "mean_ib_iterations": self.total_ib_iterations / max(self.ib_updates, 1),
            "maximum_ib_objective_increase": self.maximum_ib_objective_increase,
            "maximum_ib_raw_conditional_residual": self.maximum_ib_raw_conditional_residual,
            "ib_support_probability_threshold": self.ib_support_threshold,
            "ib_support_excluded_mass_limit": self.ib_support_mass_limit,
            "maximum_ib_excluded_prior_mass": self.maximum_ib_excluded_prior_mass,
            "continuation_fallback_count": self.ib_continuation_fallbacks,
            "broyden_fallback_count": self.ib_broyden_fallbacks,
            "least_squares_fallback_count": self.ib_least_squares_fallbacks,
            "lbfgsb_fallback_count": self.ib_lbfgsb_fallbacks,
            "final_ba_polishing_fallback_count": self.ib_final_polishing_fallbacks,
            "solver_backend": (
                "Numba-parallel log-domain Blahut-Arimoto with deterministic "
                "continuation, SciPy nonlinear acceleration, and final monotone "
                "BA polishing for near-degenerate support"
            ),
            "causal_clock": "month-t posterior plans month-(t+1) source exposure",
            "pathwise": True,
            "terminal_probability_adjustment": False,
        }
        return payload
