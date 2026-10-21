from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from information_criticality_v79 import (
    CascadeFactorsV79,
    QuasicriticalCascadeV79,
)


DOMAINS = ("social", "financial", "supply", "command")


def _row_normalise(values: np.ndarray) -> np.ndarray:
    values = np.maximum(np.asarray(values, dtype=float), 0.0)
    return values / np.maximum(np.sum(values, axis=-1, keepdims=True), 1.0e-12)


@dataclass
class StatisticalFieldFactorsV80:
    social: np.ndarray
    financial: np.ndarray
    supply: np.ndarray
    command: np.ndarray
    recovery_need: np.ndarray
    maximum_fixed_point_residual: float


class QuasicriticalCascadeV80(QuasicriticalCascadeV79):
    """V7.9 cascade plus event-level evidence needed to test SOC claims."""

    def __init__(self, *args, event_history: int = 512, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.event_sizes: deque[float] = deque(maxlen=event_history)
        self.event_durations: deque[int] = deque(maxlen=event_history)
        self.event_branching: deque[float] = deque(maxlen=event_history)
        self.outer_steps_with_events = 0
        self.last_event_count = 0
        self.last_microsteps = 0
        self.last_branching_ratio = 0.0
        self.last_avalanche_mass = 0.0

    def update(self, **kwargs) -> CascadeFactorsV79:
        result = super().update(**kwargs)
        avalanche_mass = float(
            np.sum(result.social)
            + np.sum(result.financial)
            + np.sum(result.supply)
            + np.sum(result.command)
        )
        self.last_event_count = int(result.event_count)
        self.last_microsteps = int(result.maximum_microsteps)
        self.last_branching_ratio = float(result.branching_ratio)
        self.last_avalanche_mass = avalanche_mass
        if result.event_count > 0:
            self.outer_steps_with_events += 1
            self.event_sizes.append(max(avalanche_mass, 1.0e-12))
            self.event_durations.append(max(result.maximum_microsteps, 1))
            self.event_branching.append(max(result.branching_ratio, 0.0))
        return result

    def _tail_statistics(self) -> dict:
        sizes = np.asarray(self.event_sizes, dtype=float)
        if sizes.size < 8:
            return {
                "event_sample_size": int(sizes.size),
                "tail_sample_size": 0,
                "power_law_alpha": None,
                "power_law_vs_exponential_loglikelihood": None,
                "evidence_status": "insufficient event sample",
            }
        xmin = float(np.quantile(sizes, 0.50))
        tail = sizes[sizes >= max(xmin, 1.0e-12)]
        alpha = 1.0 + tail.size / max(
            np.sum(np.log(tail / max(xmin, 1.0e-12))), 1.0e-12
        )
        power_loglik = (
            tail.size * np.log(max(alpha - 1.0, 1.0e-12))
            - tail.size * np.log(max(xmin, 1.0e-12))
            - alpha * np.sum(np.log(tail / max(xmin, 1.0e-12)))
        )
        shifted = np.maximum(tail - xmin, 0.0)
        rate = 1.0 / max(float(np.mean(shifted)), 1.0e-12)
        exponential_loglik = tail.size * np.log(rate) - rate * np.sum(shifted)
        return {
            "event_sample_size": int(sizes.size),
            "tail_sample_size": int(tail.size),
            "tail_xmin": xmin,
            "power_law_alpha": float(alpha),
            "power_law_vs_exponential_loglikelihood": float(
                power_loglik - exponential_loglik
            ),
            "evidence_status": (
                "candidate tail comparison only; finite-size scaling remains required"
            ),
        }

    def diagnostics(self) -> dict:
        payload = super().diagnostics()
        payload["v80_soc_evidence"] = self._tail_statistics() | {
            "outer_steps_with_events": self.outer_steps_with_events,
            "event_rate_per_outer_step": self.outer_steps_with_events
            / max(self.updates, 1),
            "mean_fast_microsteps": float(np.mean(self.event_durations))
            if self.event_durations else 0.0,
            "mean_observed_branching_ratio": float(np.mean(self.event_branching))
            if self.event_branching else 0.0,
            "slow_clock": "one outer model month",
            "fast_clock": "bounded within-month relaxation microsteps",
            "overlap_rule": "outer driving is suspended during fast relaxation",
            "claim_boundary": "quasi-critical cascade, not confirmed SOC",
            "terminal_probability_adjustment": False,
        }
        return payload


class SelfConsistentDMFTV80:
    """Lagged self-consistent memory and colored-noise closure.

    The closure owns only the residual field memory. It does not own the
    underlying MFG density or any economic, financial or command stock.
    """

    def __init__(
        self, paths: int, actors: int, ages: int, seed: int,
        memory_lags: int = 4, history: int = 36, ridge: float = 2.0e-4,
    ) -> None:
        self.paths = paths
        self.actors = actors
        self.ages = ages
        self.memory_lags = memory_lags
        self.ridge = ridge
        self.history: deque[np.ndarray] = deque(maxlen=history)
        self.kernel = np.zeros((len(DOMAINS), memory_lags))
        self.kernel[:, 0] = 0.12
        self.noise_state = np.zeros((paths, actors, ages, len(DOMAINS)))
        self.innovation_covariance = np.eye(len(DOMAINS)) * 2.5e-4
        self.rng = np.random.default_rng(seed)
        self.updates = 0
        self.fit_updates = 0
        self.maximum_self_consistency_residual = 0.0
        self.maximum_kernel_spectral_radius = 0.0
        self.maximum_colored_noise = 0.0

    def memory(self) -> np.ndarray:
        result = np.zeros_like(self.noise_state)
        for lag in range(min(self.memory_lags, len(self.history))):
            result += self.history[-1 - lag] * self.kernel[:, lag]
        return result

    def colored_noise(self) -> np.ndarray:
        covariance = self.innovation_covariance + 1.0e-10 * np.eye(len(DOMAINS))
        factor = np.linalg.cholesky(covariance)
        innovation = self.rng.normal(
            size=(self.paths, self.actors, self.ages, len(DOMAINS))
        ) @ factor.T
        self.noise_state = 0.72 * self.noise_state + innovation
        self.maximum_colored_noise = max(
            self.maximum_colored_noise, float(np.max(np.abs(self.noise_state)))
        )
        return self.noise_state.copy()

    def observe(self, field: np.ndarray) -> None:
        field = np.asarray(field, dtype=float)
        if self.history:
            predicted = self.memory()
            residual = field - predicted
            flattened = residual.reshape(-1, len(DOMAINS))
            covariance = np.cov(flattened, rowvar=False, bias=True)
            covariance = np.atleast_2d(covariance)
            self.innovation_covariance = (
                0.90 * self.innovation_covariance
                + 0.10 * covariance
                + 1.0e-8 * np.eye(len(DOMAINS))
            )
            self.maximum_self_consistency_residual = max(
                self.maximum_self_consistency_residual,
                float(np.max(np.abs(residual))),
            )
        self.history.append(field.copy())
        self.updates += 1
        if len(self.history) <= self.memory_lags + 2:
            return

        sample = np.stack(tuple(self.history), axis=0)
        for domain in range(len(DOMAINS)):
            target = sample[self.memory_lags :, ..., domain].reshape(-1)
            design = np.stack([
                sample[self.memory_lags - lag - 1 : -lag - 1, ..., domain]
                .reshape(-1)
                for lag in range(self.memory_lags)
            ], axis=1)
            gram = design.T @ design + self.ridge * np.eye(self.memory_lags)
            coefficient = np.linalg.solve(gram, design.T @ target)
            radius = float(np.sum(np.abs(coefficient)))
            if radius > 0.92:
                coefficient *= 0.92 / radius
            self.kernel[domain] = coefficient
        companion = np.zeros((self.memory_lags, self.memory_lags))
        companion[0] = np.mean(self.kernel, axis=0)
        if self.memory_lags > 1:
            companion[1:, :-1] = np.eye(self.memory_lags - 1)
        radius = float(np.max(np.abs(np.linalg.eigvals(companion))))
        self.maximum_kernel_spectral_radius = max(
            self.maximum_kernel_spectral_radius, radius
        )
        self.fit_updates += 1

    def diagnostics(self) -> dict:
        return {
            "updates": self.updates,
            "fit_updates": self.fit_updates,
            "memory_lags": self.memory_lags,
            "memory_kernel": self.kernel.tolist(),
            "maximum_memory_spectral_radius": self.maximum_kernel_spectral_radius,
            "maximum_self_consistency_residual": self.maximum_self_consistency_residual,
            "maximum_colored_noise_amplitude": self.maximum_colored_noise,
            "innovation_covariance_minimum_eigenvalue": float(
                np.min(np.linalg.eigvalsh(self.innovation_covariance))
            ),
            "state_owner": "residual field memory only",
            "causal_clock": "past residual fields affect the next monthly update",
            "terminal_probability_adjustment": False,
        }


class NonEquilibriumStatisticalFieldV80:
    """Pathwise four-domain non-equilibrium residual field closure."""

    def __init__(
        self, paths: int, actor_kernel: np.ndarray, age_weights: np.ndarray,
        seed: int,
    ) -> None:
        self.paths = paths
        self.actors = actor_kernel.shape[0]
        self.ages = len(age_weights)
        self.actor_kernel = _row_normalise(actor_kernel)
        self.age_weights = np.asarray(age_weights, dtype=float)
        self.age_weights /= np.sum(self.age_weights)
        age = np.zeros((self.ages, self.ages))
        for index in range(self.ages):
            age[index, index] = 1.0
            if index:
                age[index, index - 1] = 0.45
            if index + 1 < self.ages:
                age[index, index + 1] = 0.45
        self.age_kernel = _row_normalise(age)
        self.field = np.zeros((paths, self.actors, self.ages, len(DOMAINS)))
        self.previous_field = self.field.copy()
        self.relaxation = np.array([0.44, 0.38, 0.40, 0.46])
        self.diffusion_actor = np.array([0.10, 0.08, 0.10, 0.07])
        self.diffusion_age = np.array([0.08, 0.06, 0.07, 0.05])
        self.nonreciprocal = np.array([
            [0.00, 0.10, 0.14, 0.08],
            [0.05, 0.00, 0.16, 0.04],
            [0.08, 0.14, 0.00, 0.09],
            [0.10, 0.04, 0.12, 0.00],
        ])
        self.dmft = SelfConsistentDMFTV80(
            paths, self.actors, self.ages, seed + 8_003
        )
        self.updates = 0
        self.fixed_point_iterations = 5
        self.maximum_fixed_point_residual = 0.0
        self.maximum_field = 0.0
        self.maximum_msrjd_action = 0.0
        self.maximum_response = 0.0
        self.maximum_correlation_length = 0.0
        self.minimum_recovery_rate = np.inf
        self.binder_by_resolution = {"16": 0.0, "8": 0.0, "4": 0.0}
        self.maximum_scale_spread = 0.0
        self.cavity_importance = np.zeros(self.actors)
        self.cavity_updates = 0

    def _target(
        self, *, damage: np.ndarray, shortage: np.ndarray,
        inflation: np.ndarray, financial_stress: np.ndarray,
        command: np.ndarray, network: np.ndarray,
        contagion_mobilization: np.ndarray, contagion_fatigue: np.ndarray,
        jump_signal: np.ndarray, mfg_mean: np.ndarray,
        cascade_loss: np.ndarray,
    ) -> np.ndarray:
        social = (
            0.24 * damage + 0.20 * contagion_fatigue
            + 0.15 * shortage + 0.10 * (1.0 - network)
            - 0.08 * contagion_mobilization + 0.10 * cascade_loss[..., 0]
        )
        financial = (
            0.30 * financial_stress + 0.20 * inflation + 0.14 * shortage
            + 0.08 * jump_signal + 0.10 * cascade_loss[..., 1]
        )
        supply = (
            0.34 * shortage + 0.18 * damage + 0.14 * (1.0 - network)
            + 0.08 * jump_signal + 0.10 * cascade_loss[..., 2]
        )
        command_pressure = (
            0.32 * (1.0 - command) + 0.18 * damage
            + 0.14 * (1.0 - network) + 0.08 * jump_signal
            + 0.10 * cascade_loss[..., 3]
        )
        actor_target = np.stack(
            (social, financial, supply, command_pressure), axis=-1
        )
        age_sensitivity = np.array([1.08, 1.00, 0.94, 0.88])
        target = actor_target[:, :, None, :] * age_sensitivity[None, None, :, None]
        target[..., 0] += 0.08 * np.maximum(-mfg_mean, 0.0)
        target[..., 3] += 0.04 * np.maximum(-mfg_mean, 0.0)
        return np.clip(target, 0.0, 1.5)

    def _drift(
        self, candidate: np.ndarray, target: np.ndarray,
        memory: np.ndarray, colored_noise: np.ndarray,
        jump_signal: np.ndarray,
    ) -> np.ndarray:
        actor_mix = np.einsum("ab,pbgd->pagd", self.actor_kernel, candidate)
        age_mix = np.einsum("gh,pahd->pagd", self.age_kernel, candidate)
        cross = np.einsum("de,page->pagd", self.nonreciprocal, candidate)
        saturation = 0.10 * np.power(candidate, 3)
        pulse = 0.035 * jump_signal[:, :, None, None]
        return (
            self.relaxation * (target - candidate)
            + self.diffusion_actor * (actor_mix - candidate)
            + self.diffusion_age * (age_mix - candidate)
            + cross - saturation + 0.16 * memory
            + 0.18 * colored_noise + pulse
        )

    def _field_statistics(self) -> None:
        centered = self.field - np.mean(self.field, axis=(1, 2), keepdims=True)
        distances = np.abs(
            np.arange(self.actors)[:, None] - np.arange(self.actors)[None, :]
        )
        covariance = np.einsum(
            "pagd,pbgd->pabd", centered, centered
        ) / max(self.ages, 1)
        positive = np.maximum(np.mean(covariance, axis=(0, 3)), 0.0)
        numerator = np.sum(np.square(distances) * positive)
        denominator = 2.0 * np.sum(positive)
        correlation_length = float(np.sqrt(numerator / max(denominator, 1.0e-12)))
        self.maximum_correlation_length = max(
            self.maximum_correlation_length, correlation_length
        )

        scalar = np.mean(self.field, axis=3)
        fine = scalar.reshape(self.paths, -1)
        coarse8 = scalar.reshape(self.paths, self.actors, 2, 2).mean(axis=3).reshape(
            self.paths, -1
        )
        coarse4 = scalar.mean(axis=2)
        for label, values in (("16", fine), ("8", coarse8), ("4", coarse4)):
            centered_values = values - np.mean(values, axis=1, keepdims=True)
            second = np.mean(np.square(centered_values))
            fourth = np.mean(np.power(centered_values, 4))
            binder = 1.0 - fourth / max(3.0 * second * second, 1.0e-12)
            self.binder_by_resolution[label] = float(binder)
        binder_values = np.fromiter(self.binder_by_resolution.values(), dtype=float)
        self.maximum_scale_spread = max(
            self.maximum_scale_spread, float(np.ptp(binder_values))
        )

        jacobian = -np.diag(self.relaxation) + self.nonreciprocal
        recovery_rate = -float(np.max(np.real(np.linalg.eigvals(jacobian))))
        self.minimum_recovery_rate = min(self.minimum_recovery_rate, recovery_rate)
        response = np.linalg.solve(
            np.eye(len(DOMAINS)) - 0.20 * jacobian,
            np.eye(len(DOMAINS)),
        )
        self.maximum_response = max(
            self.maximum_response, float(np.max(np.abs(response)))
        )

    def _cavity_resolve(self, target: np.ndarray) -> None:
        if self.updates % 12 != 0:
            return
        actor_target = np.mean(target, axis=(0, 2))
        coupling = 0.18 * self.actor_kernel
        base_operator = np.eye(self.actors) - coupling
        full = np.linalg.solve(base_operator, actor_target)
        for removed in range(self.actors):
            keep = np.arange(self.actors) != removed
            reduced = np.linalg.solve(
                base_operator[np.ix_(keep, keep)], actor_target[keep]
            )
            reconstructed = np.zeros_like(full)
            reconstructed[keep] = reduced
            self.cavity_importance[removed] += float(
                np.linalg.norm(full - reconstructed)
            )
        self.cavity_updates += 1

    def update(
        self, *, damage: np.ndarray, shortage: np.ndarray,
        inflation: np.ndarray, financial_stress: np.ndarray,
        command: np.ndarray, network: np.ndarray,
        contagion_mobilization: np.ndarray, contagion_fatigue: np.ndarray,
        jump_signal: np.ndarray, mfg_mean: np.ndarray,
        cascade_loss: np.ndarray,
    ) -> StatisticalFieldFactorsV80:
        target = self._target(
            damage=damage, shortage=shortage, inflation=inflation,
            financial_stress=financial_stress, command=command,
            network=network, contagion_mobilization=contagion_mobilization,
            contagion_fatigue=contagion_fatigue, jump_signal=jump_signal,
            mfg_mean=mfg_mean, cascade_loss=cascade_loss,
        )
        memory = self.dmft.memory()
        colored_noise = self.dmft.colored_noise()
        opening = self.field.copy()
        candidate = opening.copy()
        residual = 0.0
        for _ in range(self.fixed_point_iterations):
            proposal = np.clip(
                opening + 0.20 * self._drift(
                    candidate, target, memory, colored_noise, jump_signal
                ),
                0.0, 1.5,
            )
            residual = float(np.max(np.abs(proposal - candidate)))
            candidate = 0.55 * candidate + 0.45 * proposal
        self.previous_field = self.field.copy()
        self.field = candidate
        self.dmft.observe(self.field)
        self.maximum_fixed_point_residual = max(
            self.maximum_fixed_point_residual, residual
        )
        self.maximum_field = max(self.maximum_field, float(np.max(self.field)))

        realized = self.field - opening
        predicted = 0.20 * self._drift(
            self.field, target, memory, np.zeros_like(colored_noise), jump_signal
        )
        variance = np.maximum(
            np.diag(self.dmft.innovation_covariance), 1.0e-8
        )
        action = 0.5 * np.mean(np.square(realized - predicted) / variance)
        self.maximum_msrjd_action = max(self.maximum_msrjd_action, float(action))
        self.updates += 1
        self._field_statistics()
        self._cavity_resolve(target)

        actor_field = np.sum(
            self.field * self.age_weights[None, None, :, None], axis=2
        )
        recovery_need = np.clip(np.mean(actor_field, axis=2), 0.0, 1.0)
        return StatisticalFieldFactorsV80(
            social=np.clip(actor_field[..., 0], 0.0, 1.0),
            financial=np.clip(actor_field[..., 1], 0.0, 1.0),
            supply=np.clip(actor_field[..., 2], 0.0, 1.0),
            command=np.clip(actor_field[..., 3], 0.0, 1.0),
            recovery_need=recovery_need,
            maximum_fixed_point_residual=residual,
        )

    def diagnostics(self) -> dict:
        return {
            "updates": self.updates,
            "field_shape": list(self.field.shape),
            "domains": list(DOMAINS),
            "maximum_field_amplitude": self.maximum_field,
            "maximum_inner_fixed_point_residual": self.maximum_fixed_point_residual,
            "maximum_correlation_length": self.maximum_correlation_length,
            "minimum_linear_recovery_rate": 0.0
            if not np.isfinite(self.minimum_recovery_rate)
            else self.minimum_recovery_rate,
            "maximum_linear_susceptibility": self.maximum_response,
            "binder_cumulant_by_resolution": self.binder_by_resolution,
            "maximum_binder_scale_spread": self.maximum_scale_spread,
            "maximum_msrjd_action_density": self.maximum_msrjd_action,
            "cavity_equilibrium_resolve_updates": self.cavity_updates,
            "mean_cavity_equilibrium_importance": (
                self.cavity_importance / max(self.cavity_updates, 1)
            ).tolist(),
            "dmft": self.dmft.diagnostics(),
            "state_owner": (
                "residual mesoscopic field; primary social, financial, supply and command states remain external"
            ),
            "feedback_clock": (
                "current field affects outer consumers immediately and MFG inputs with one explicit monthly lag"
            ),
            "identification": (
                "reaction, diffusion, memory and noise coefficients are structural scenario priors pending historical calibration"
            ),
            "terminal_probability_adjustment": False,
        }
