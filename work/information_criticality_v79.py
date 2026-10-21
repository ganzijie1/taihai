from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from complete_information_geometry_v77 import CompleteInformationGeometryV77


def _normalise(values: np.ndarray, axis: int = -1) -> np.ndarray:
    values = np.maximum(np.asarray(values, dtype=float), 1.0e-300)
    return values / np.maximum(np.sum(values, axis=axis, keepdims=True), 1.0e-300)


class InformationTheoryV79(CompleteInformationGeometryV77):
    """Correlated-source and decision-weighted extension of V7.7 beliefs.

    Correlation changes the source exposure used by the inherited likelihood;
    asymmetric decision loss and derived channel capacity change next month's
    active-observation allocation. No terminal probability is adjusted here.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        source_count = self.base_source_exposure.shape[-1]
        index = np.arange(source_count)
        distance = np.abs(index[:, None] - index[None, :])
        correlation = 0.12 + 0.46 * np.exp(-distance / 1.35)
        np.fill_diagonal(correlation, 1.0)
        self.source_correlation = correlation
        self.false_negative_cost = np.array([1.20, 1.05, 0.88, 0.92])
        self.false_positive_cost = np.array([0.88, 1.08, 0.96, 0.94])
        self.current_network = np.ones((self.paths, 4))
        self.current_jump = np.zeros((self.paths, 4))
        self.current_common_noise = np.zeros((self.paths, 4))
        self.effective_sample_size = np.full((self.paths, 4), source_count, dtype=float)
        self.effective_capacity = np.ones((self.paths, 4))
        self.maximum_redundancy_penalty = 0.0
        self.minimum_effective_sample_size = float(source_count)
        self.minimum_effective_capacity = 1.0
        self.maximum_rate_distortion_budget_residual = 0.0
        self.maximum_asymmetric_decision_risk = 0.0
        self.rate_distortion_updates = 0

    def _source_statistics(self, allocation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        share = _normalise(allocation, axis=2)
        quadratic = np.einsum(
            "pas,st,pat->pa", share, self.source_correlation, share
        )
        effective = 1.0 / np.maximum(quadratic, 1.0e-12)
        overlap = np.einsum("st,pat->pas", self.source_correlation, share)
        uniqueness = 1.0 / np.maximum(overlap, 1.0e-8)
        uniqueness /= np.maximum(np.mean(uniqueness, axis=2, keepdims=True), 1.0e-12)
        return effective, uniqueness

    def _active_observation_plan(self) -> np.ndarray:
        base_plan = super()._active_observation_plan()
        base_budget = np.sum(base_plan, axis=2, keepdims=True)
        effective, uniqueness = self._source_statistics(base_plan)

        positive = np.sum(self.posterior[..., self.grid > 0.0], axis=-1)
        self_positive = positive[:, np.arange(4), np.arange(4)]
        false_negative = self_positive * self.false_negative_cost[None, :]
        false_positive = (1.0 - self_positive) * self.false_positive_cost[None, :]
        decision_risk = np.minimum(false_negative, false_positive)
        risk_weight = 0.72 + 0.56 * decision_risk / np.maximum(
            np.mean(decision_risk, axis=1, keepdims=True), 1.0e-8
        )

        credibility = self.reliability[..., 0] + 0.48 * self.reliability[..., 1]
        authenticity = np.mean(credibility, axis=2)
        loss = np.clip(
            0.42 * (1.0 - self.current_network) + 0.22 * self.current_jump,
            0.0, 0.80,
        )
        delay = np.clip(
            0.85 * (1.0 - self.current_network)
            + 0.30 * np.abs(self.current_common_noise),
            0.0, 1.5,
        )
        capacity = np.clip(
            self.current_network * (1.0 - loss) * authenticity
            * np.exp(-0.60 * delay),
            0.08, 1.0,
        )
        score = base_plan * uniqueness * risk_weight[..., None]
        share = _normalise(score, axis=2)
        planned = base_budget * capacity[..., None] * share

        budget_residual = np.max(np.abs(
            np.sum(planned, axis=2) - base_budget[..., 0] * capacity
        ))
        self.effective_sample_size = effective
        self.effective_capacity = capacity
        self.maximum_redundancy_penalty = max(
            self.maximum_redundancy_penalty,
            float(np.max(np.abs(1.0 - uniqueness))),
        )
        self.minimum_effective_sample_size = min(
            self.minimum_effective_sample_size, float(np.min(effective))
        )
        self.minimum_effective_capacity = min(
            self.minimum_effective_capacity, float(np.min(capacity))
        )
        self.maximum_asymmetric_decision_risk = max(
            self.maximum_asymmetric_decision_risk, float(np.max(decision_risk))
        )
        self.maximum_rate_distortion_budget_residual = max(
            self.maximum_rate_distortion_budget_residual, float(budget_residual)
        )
        self.rate_distortion_updates += 1
        return np.clip(planned, 1.0e-8, 1.0)

    def update(self, **kwargs):
        self.current_network = np.clip(np.asarray(kwargs["network"]), 0.0, 1.0)
        self.current_jump = np.clip(np.asarray(kwargs["jump_signal"]), 0.0, 1.0)
        self.current_common_noise = np.clip(
            np.asarray(kwargs["common_noise"]), -1.0, 1.0
        )
        return super().update(**kwargs)

    def diagnostics(self) -> dict:
        payload = super().diagnostics()
        payload["v79_correlated_rate_distortion_information"] = {
            "source_count": int(self.source_correlation.shape[0]),
            "source_correlation_positive_semidefinite": bool(
                np.min(np.linalg.eigvalsh(self.source_correlation)) >= -1.0e-12
            ),
            "minimum_effective_sample_size": self.minimum_effective_sample_size,
            "maximum_redundancy_penalty": self.maximum_redundancy_penalty,
            "minimum_effective_channel_capacity": self.minimum_effective_capacity,
            "maximum_asymmetric_decision_risk": self.maximum_asymmetric_decision_risk,
            "maximum_budget_residual": self.maximum_rate_distortion_budget_residual,
            "updates": self.rate_distortion_updates,
            "likelihood_coupling": (
                "correlation-adjusted source exposure enters the inherited Bayesian likelihood"
            ),
            "decision_coupling": (
                "asymmetric false-negative/false-positive loss and derived capacity allocate next-month attention"
            ),
            "identification": "source-correlation and loss matrices are scenario priors",
            "terminal_probability_adjustment": False,
        }
        return payload


class ConditionalTransferEntropyV79:
    """Lag-one discrete conditional-transfer-entropy estimator.

    The estimator consumes only present and past aggregate layer signals. Its
    current gain is applied before the latest observation is added, preserving
    an explicit one-month lag. Online estimates are structural adaptation, not
    causal estimates from historical treatment variation.
    """

    def __init__(
        self, layers: int = 4, actors: int = 4, history: int = 24,
        minimum_periods: int = 8, pseudocount: float = 0.5,
    ) -> None:
        self.layers = layers
        self.actors = actors
        self.minimum_periods = minimum_periods
        self.pseudocount = pseudocount
        self.history: deque[np.ndarray] = deque(maxlen=history)
        self.transfer_entropy = np.zeros((layers, actors, actors))
        self.gain = np.ones((layers, actors, actors))
        self.updates = 0
        self.fit_updates = 0
        self.maximum_transfer_entropy = 0.0
        self.maximum_gain_shift = 0.0
        self.minimum_sample_count = np.inf

    @staticmethod
    def _bins(values: np.ndarray) -> np.ndarray:
        return np.digitize(values, (-0.25, 0.25)).astype(np.int64)

    def _estimate_pair(
        self, next_target: np.ndarray, target: np.ndarray,
        source: np.ndarray, condition: np.ndarray,
    ) -> float:
        y1 = self._bins(next_target)
        y0 = self._bins(target)
        x0 = self._bins(source)
        z0 = self._bins(condition)
        flat = ((y1 * 3 + y0) * 3 + x0) * 3 + z0
        counts = np.bincount(flat, minlength=81).reshape(3, 3, 3, 3).astype(float)
        counts += self.pseudocount
        probability = counts / np.sum(counts)
        y0xz = np.sum(counts, axis=0, keepdims=True)
        y1y0z = np.sum(counts, axis=2, keepdims=True)
        y0z = np.sum(counts, axis=(0, 2), keepdims=True)
        ratio = (counts / y0xz) / np.maximum(y1y0z / y0z, 1.0e-300)
        return float(np.sum(probability * np.log(np.maximum(ratio, 1.0e-300))))

    def current_gain(self) -> np.ndarray:
        return self.gain.copy()

    def observe(self, signals: np.ndarray) -> None:
        signals = np.asarray(signals, dtype=float)
        if signals.ndim != 3 or signals.shape[1:] != (self.layers, self.actors):
            raise ValueError("signals must have shape (paths, layers, actors)")
        self.history.append(np.clip(signals, -1.5, 1.5).copy())
        self.updates += 1
        if len(self.history) < self.minimum_periods:
            return
        sample = np.stack(tuple(self.history), axis=0)
        condition = np.mean(sample[:-1], axis=3)
        estimate = np.zeros_like(self.transfer_entropy)
        for layer in range(self.layers):
            z = condition[:, :, layer].ravel()
            for target in range(self.actors):
                y1 = sample[1:, :, layer, target].ravel()
                y0 = sample[:-1, :, layer, target].ravel()
                for source in range(self.actors):
                    if source == target:
                        continue
                    x0 = sample[:-1, :, layer, source].ravel()
                    estimate[layer, target, source] = self._estimate_pair(
                        y1, y0, x0, z
                    )
        row_mean = np.mean(estimate, axis=2, keepdims=True)
        gain = np.exp(np.clip(1.8 * (estimate - row_mean), -0.18, 0.18))
        for layer in range(self.layers):
            np.fill_diagonal(gain[layer], 1.0)
        self.transfer_entropy = estimate
        self.gain = gain
        self.fit_updates += 1
        self.maximum_transfer_entropy = max(
            self.maximum_transfer_entropy, float(np.max(estimate))
        )
        self.maximum_gain_shift = max(
            self.maximum_gain_shift, float(np.max(np.abs(gain - 1.0)))
        )
        self.minimum_sample_count = min(
            self.minimum_sample_count, float(y1.size)
        )

    def diagnostics(self) -> dict:
        return {
            "updates": self.updates,
            "fit_updates": self.fit_updates,
            "history_months": len(self.history),
            "maximum_transfer_entropy": self.maximum_transfer_entropy,
            "maximum_directional_gain_shift": self.maximum_gain_shift,
            "minimum_fit_sample_count": (
                0.0 if not np.isfinite(self.minimum_sample_count)
                else self.minimum_sample_count
            ),
            "conditioning": "same-layer global mean plus target lag",
            "causal_clock": "previous estimate affects current kernel; current observation affects next month",
            "identification": (
                "online structural adaptation; historical causal interpretation requires event-study or local-projection validation"
            ),
            "terminal_probability_adjustment": False,
        }


@dataclass
class CascadeFactorsV79:
    social: np.ndarray
    financial: np.ndarray
    supply: np.ndarray
    command: np.ndarray
    recovery_need: np.ndarray
    event_count: int
    maximum_microsteps: int
    branching_ratio: float
    mass_balance_residual: float


class QuasicriticalCascadeV79:
    """Four-domain slow-drive/fast-relaxation cascade on a directed network."""

    DOMAINS = ("social", "financial", "supply", "command")

    def __init__(self, paths: int, actor_kernel: np.ndarray) -> None:
        self.paths = paths
        self.actors = actor_kernel.shape[0]
        actor = _normalise(np.maximum(actor_kernel, 0.0), axis=1)
        domain = np.array([
            [0.50, 0.18, 0.14, 0.18],
            [0.16, 0.50, 0.24, 0.10],
            [0.12, 0.24, 0.50, 0.14],
            [0.18, 0.12, 0.16, 0.54],
        ])
        self.transition = np.kron(actor, domain)
        self.transition = _normalise(self.transition, axis=1)
        actor_threshold = np.array([0.76, 0.72, 0.80, 0.78])[:, None]
        domain_shift = np.array([0.00, 0.04, -0.03, 0.02])[None, :]
        self.threshold = np.broadcast_to(
            actor_threshold + domain_shift,
            (paths, self.actors, len(self.DOMAINS)),
        ).copy()
        self.pressure = 0.34 * self.threshold
        self.transmission = 0.72
        self.maximum_microsteps_allowed = 16
        self.updates = 0
        self.total_events = 0
        self.maximum_microsteps_used = 0
        self.maximum_branching_ratio = 0.0
        self.maximum_mass_balance_residual = 0.0
        self.maximum_pressure = float(np.max(self.pressure))
        self.maximum_loss = 0.0

    def update(
        self, *, damage: np.ndarray, shortage: np.ndarray,
        inflation: np.ndarray, financial_stress: np.ndarray,
        command: np.ndarray, network: np.ndarray,
        contagion_mobilization: np.ndarray,
        contagion_fatigue: np.ndarray, jump_signal: np.ndarray,
        service_action: np.ndarray,
    ) -> CascadeFactorsV79:
        damage = np.clip(np.asarray(damage), 0.0, 1.5)
        shortage = np.clip(np.asarray(shortage), 0.0, 1.5)
        inflation = np.clip(np.asarray(inflation), 0.0, 1.0)
        financial_stress = np.clip(np.asarray(financial_stress), 0.0, 1.5)
        command = np.clip(np.asarray(command), 0.0, 1.0)
        network = np.clip(np.asarray(network), 0.0, 1.0)
        fatigue = np.clip(np.asarray(contagion_fatigue), 0.0, 1.0)
        mobilization = np.clip(np.asarray(contagion_mobilization), 0.0, 1.0)
        jump_signal = np.clip(np.asarray(jump_signal), 0.0, 1.0)
        service = np.clip(np.asarray(service_action), 0.0, 1.0)

        drive = np.stack((
            0.12 * damage + 0.10 * shortage + 0.08 * inflation
            + 0.08 * fatigue + 0.05 * jump_signal,
            0.11 * financial_stress + 0.08 * inflation + 0.06 * shortage
            + 0.04 * jump_signal,
            0.13 * shortage + 0.09 * damage + 0.06 * (1.0 - network)
            + 0.04 * jump_signal,
            0.12 * (1.0 - command) + 0.08 * damage
            + 0.06 * (1.0 - network) + 0.05 * jump_signal,
        ), axis=2)
        # A directed hyperedge: financial and supply pressure jointly amplify
        # social and command stress; it is not reduced to an independent event.
        joint_gate = np.sqrt(
            np.clip(self.pressure[:, :, 1] / self.threshold[:, :, 1], 0.0, 2.0)
            * np.clip(self.pressure[:, :, 2] / self.threshold[:, :, 2], 0.0, 2.0)
        )
        drive[:, :, 0] += 0.025 * joint_gate
        drive[:, :, 3] += 0.018 * joint_gate
        recovery = np.stack((
            0.045 * network + 0.035 * service,
            0.036 * network + 0.026 * service,
            0.040 * network + 0.030 * service,
            0.042 * command + 0.026 * service,
        ), axis=2)
        recovery += 0.012 * (1.0 - mobilization)[:, :, None]
        opening = self.pressure.copy()
        raw = opening + drive - recovery
        self.pressure = np.clip(raw, 0.0, 2.5)

        avalanche = np.zeros_like(self.pressure)
        prior_active = np.zeros((self.paths, self.actors, len(self.DOMAINS)), dtype=bool)
        ratios: list[float] = []
        event_count = 0
        fast_opening = self.pressure.copy()
        dissipated = np.zeros(self.paths)
        used = 0
        for microstep in range(self.maximum_microsteps_allowed):
            excess = np.maximum(self.pressure - self.threshold, 0.0)
            active = excess > 1.0e-10
            count = int(np.sum(active))
            if count == 0:
                break
            used = microstep + 1
            event_count += count
            if np.any(prior_active):
                previous_count = int(np.sum(prior_active))
                ratios.append(count / max(previous_count, 1))
            prior_active = active
            flat_excess = excess.reshape(self.paths, -1)
            transferred = self.transmission * (flat_excess @ self.transition)
            self.pressure -= excess
            self.pressure += transferred.reshape(self.pressure.shape)
            avalanche += excess
            dissipated += (1.0 - self.transmission) * np.sum(
                flat_excess, axis=1
            )

        residual = np.max(np.abs(
            np.sum(self.pressure, axis=(1, 2))
            - np.sum(fast_opening, axis=(1, 2)) + dissipated
        ))
        scaled = np.clip(avalanche / np.maximum(self.threshold, 1.0e-8), 0.0, 1.0)
        branching = max(ratios, default=0.0)
        recovery_need = np.clip(
            0.55 * scaled + 0.20 * self.pressure / self.threshold,
            0.0, 1.0,
        )
        self.updates += 1
        self.total_events += event_count
        self.maximum_microsteps_used = max(self.maximum_microsteps_used, used)
        self.maximum_branching_ratio = max(self.maximum_branching_ratio, branching)
        self.maximum_mass_balance_residual = max(
            self.maximum_mass_balance_residual, float(residual)
        )
        self.maximum_pressure = max(self.maximum_pressure, float(np.max(self.pressure)))
        self.maximum_loss = max(self.maximum_loss, float(np.max(scaled)))
        return CascadeFactorsV79(
            social=scaled[:, :, 0],
            financial=scaled[:, :, 1],
            supply=scaled[:, :, 2],
            command=scaled[:, :, 3],
            recovery_need=np.mean(recovery_need, axis=2),
            event_count=event_count,
            maximum_microsteps=used,
            branching_ratio=branching,
            mass_balance_residual=float(residual),
        )

    def diagnostics(self) -> dict:
        return {
            "domains": list(self.DOMAINS),
            "updates": self.updates,
            "total_threshold_events": self.total_events,
            "maximum_microsteps_used": self.maximum_microsteps_used,
            "maximum_branching_ratio": self.maximum_branching_ratio,
            "maximum_fast_mass_balance_residual": self.maximum_mass_balance_residual,
            "maximum_pressure": self.maximum_pressure,
            "maximum_scaled_loss": self.maximum_loss,
            "slow_clock": "monthly same-path pressure accumulation",
            "fast_clock": "within-month deterministic DEDS relaxation",
            "trigger_owner": "existing jump and Hawkes event signals",
            "terminal_probability_adjustment": False,
        }
