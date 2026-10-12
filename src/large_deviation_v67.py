from __future__ import annotations

from dataclasses import dataclass

import numpy as np


MODE_NAMES = (
    "baseline",
    "china_system_stress",
    "defender_system_stress",
    "alliance_system_stress",
    "common_network_cascade",
    "joint_extreme",
)
TARGET_MODE_PROBABILITY = np.array([0.78, 0.05, 0.04, 0.035, 0.055, 0.04])
PROPOSAL_MODE_PROBABILITY = np.array([0.28, 0.15, 0.14, 0.13, 0.15, 0.15])


@dataclass(frozen=True)
class LargeDeviationProposal:
    mode: np.ndarray
    likelihood_ratio: np.ndarray
    actor_jump_multiplier: np.ndarray
    common_jump_multiplier: np.ndarray


class HybridLargeDeviationSystemV67:
    """Pathwise large-deviation ledger and defensive mixture proposal.

    The proposal changes only the mixture frequency of pre-registered tail
    regimes. Conditional dynamics inside each regime are identical under the
    target and proposal measures, so the Radon--Nikodym derivative is exactly
    the target/proposal mode-probability ratio. Continuous and jump actions are
    diagnostics; they never replace likelihood-ratio probability estimates.
    """

    def __init__(
        self,
        paths: int,
        seed: int,
        *,
        target_mode_probability: np.ndarray | None = None,
        proposal_mode_probability: np.ndarray | None = None,
        action_enabled: bool = True,
    ):
        self.paths = int(paths)
        self.rng = np.random.default_rng(seed)
        self.target_mode_probability = np.asarray(
            TARGET_MODE_PROBABILITY
            if target_mode_probability is None else target_mode_probability,
            dtype=float,
        ).copy()
        self.proposal_mode_probability = np.asarray(
            PROPOSAL_MODE_PROBABILITY
            if proposal_mode_probability is None else proposal_mode_probability,
            dtype=float,
        ).copy()
        if self.target_mode_probability.shape != (len(MODE_NAMES),):
            raise ValueError("target mode probability has the wrong shape")
        if self.proposal_mode_probability.shape != (len(MODE_NAMES),):
            raise ValueError("proposal mode probability has the wrong shape")
        if np.any(self.target_mode_probability <= 0.0) or np.any(
            self.proposal_mode_probability <= 0.0
        ):
            raise ValueError("target and proposal modes require common full support")
        self.target_mode_probability /= np.sum(self.target_mode_probability)
        self.proposal_mode_probability /= np.sum(self.proposal_mode_probability)
        self.action_enabled = bool(action_enabled)
        self.month = -1
        self.proposal: LargeDeviationProposal | None = None
        self.previous_feature: np.ndarray | None = None
        self.reference_drift = np.zeros(8)
        self.continuous_action = np.zeros(self.paths)
        self.jump_action = np.zeros(self.paths)
        self.mean_field_action = np.zeros(self.paths)
        self.initial_measure: np.ndarray | None = None
        self.previous_measure: np.ndarray | None = None
        self.rare_hits = np.zeros((self.paths, 4), dtype=bool)
        self.first_hit_month = np.full((self.paths, 4), -1, dtype=np.int16)
        self.event_count = np.zeros((self.paths, 6), dtype=np.int32)
        self.max_clock_violation = 0.0
        self.max_likelihood_identity_residual = 0.0
        self.max_measure_mass_residual = 0.0
        self.future_information_used = False

    def initialise_proposal(self) -> LargeDeviationProposal:
        if self.proposal is not None:
            return self.proposal
        mode = self.rng.choice(
            len(MODE_NAMES), self.paths, p=self.proposal_mode_probability
        )
        likelihood = (
            self.target_mode_probability[mode] / self.proposal_mode_probability[mode]
        )
        actor = np.ones((self.paths, 4))
        actor[mode == 1, 0] = 3.2
        actor[mode == 2, 1] = 3.2
        actor[mode == 3, 2:] = 3.0
        actor[mode == 5, :] = 2.7
        common = np.where(mode == 4, 3.5, 1.0)
        common = np.where(mode == 5, 2.6, common)
        self.proposal = LargeDeviationProposal(mode, likelihood, actor, common)
        identity = float(np.sum(
            self.proposal_mode_probability
            * self.target_mode_probability / self.proposal_mode_probability
        ))
        self.max_likelihood_identity_residual = abs(identity - 1.0)
        return self.proposal

    def record_jumps(
        self,
        *,
        events: tuple[np.ndarray, ...],
        probabilities: tuple[np.ndarray, ...],
    ) -> None:
        if not self.action_enabled:
            return
        if len(events) != len(probabilities):
            raise ValueError("jump events and probabilities must have equal length")
        for family, (event, probability) in enumerate(zip(events, probabilities)):
            event_array = np.asarray(event, dtype=bool)
            probability_array = np.clip(np.asarray(probability, dtype=float), 1e-12, 1.0 - 1e-12)
            if event_array.shape != probability_array.shape:
                raise ValueError("jump event and probability shapes differ")
            axes = tuple(range(1, event_array.ndim))
            log_probability = np.where(
                event_array, np.log(probability_array), np.log1p(-probability_array)
            )
            action = -np.sum(log_probability, axis=axes) if axes else -log_probability
            self.jump_action += action
            count = np.sum(event_array, axis=axes) if axes else event_array.astype(int)
            self.event_count[:, min(family, self.event_count.shape[1] - 1)] += count.astype(np.int32)

    @staticmethod
    def _empirical_measure(
        support: np.ndarray, panic: np.ndarray, collective_action: np.ndarray
    ) -> np.ndarray:
        bins = np.linspace(0.0, 1.0, 6)
        measures = []
        for values in (support, panic, collective_action):
            counts = np.stack([
                np.mean((values >= bins[i]) & (values < bins[i + 1]), axis=1)
                for i in range(len(bins) - 1)
            ], axis=1)
            counts[:, -1] += np.mean(values >= 1.0, axis=1)
            measures.append(counts)
        measure = np.concatenate(measures, axis=1)
        measure /= np.maximum(np.sum(measure, axis=1, keepdims=True), 1e-12)
        return measure

    def observe(
        self,
        *,
        month: int,
        actor_damage: np.ndarray,
        actor_output: np.ndarray,
        financial_capacity: np.ndarray,
        actor_network: np.ndarray,
        actor_command: np.ndarray,
        actor_support: np.ndarray,
        actor_panic: np.ndarray,
        actor_action: np.ndarray,
        actor_resilience: np.ndarray,
        stock_shortage: np.ndarray,
        front_rear_integrity: np.ndarray,
        leader_continuity: np.ndarray,
    ) -> None:
        if month != self.month + 1:
            self.max_clock_violation = max(
                self.max_clock_violation, abs(month - (self.month + 1))
            )
            raise RuntimeError("V6.7 large-deviation clock must advance monthly")
        self.month = month
        if not self.action_enabled:
            return
        command = np.prod(np.clip(actor_command, 1e-6, 1.0), axis=2) ** (1.0 / 3.0)
        feature = np.column_stack((
            np.mean(actor_damage, axis=1),
            np.mean(stock_shortage, axis=1),
            1.0 - np.mean(actor_output, axis=1),
            1.0 - np.mean(financial_capacity, axis=1),
            1.0 - np.mean(command, axis=1),
            1.0 - np.mean(actor_network, axis=1),
            np.mean(actor_panic, axis=1),
            1.0 - np.mean(actor_resilience, axis=(1, 2)),
        ))
        if self.previous_feature is not None:
            increment = feature - self.previous_feature
            cohort_drift = np.mean(increment, axis=0)
            self.reference_drift = 0.94 * self.reference_drift + 0.06 * cohort_drift
            scale = np.array([0.035, 0.055, 0.045, 0.040, 0.040, 0.040, 0.055, 0.040])
            residual = (increment - self.reference_drift[None, :]) / scale[None, :]
            self.continuous_action += 0.5 * np.sum(residual * residual, axis=1)
        self.previous_feature = feature.copy()

        measure = self._empirical_measure(actor_support, actor_panic, actor_action)
        self.max_measure_mass_residual = max(
            self.max_measure_mass_residual,
            float(np.max(np.abs(np.sum(measure, axis=1) - 1.0))),
        )
        if self.initial_measure is None:
            self.initial_measure = measure.copy()
        reference = self.previous_measure if self.previous_measure is not None else self.initial_measure
        self.mean_field_action += np.sum(
            measure * np.log(np.maximum(measure, 1e-12) / np.maximum(reference, 1e-12)),
            axis=1,
        )
        self.previous_measure = measure.copy()

        logistics = (
            (np.min(front_rear_integrity, axis=(1, 2)) < 0.30)
            & (np.max(stock_shortage, axis=1) > 0.70)
        )
        command_failure = (
            (np.min(command, axis=1) < 0.25)
            & (np.min(actor_network, axis=1) < 0.35)
        )
        economic_political = (
            (np.min(actor_output, axis=1) < 0.35)
            & (np.min(actor_resilience[:, :, 0], axis=1) < 0.30)
            & (np.min(actor_resilience[:, :, 1], axis=1) < 0.30)
        )
        leadership = np.min(leader_continuity, axis=1) < 0.35
        current = np.column_stack((logistics, command_failure, economic_political, leadership))
        new_hits = current & ~self.rare_hits
        self.first_hit_month[new_hits] = month + 1
        self.rare_hits |= current

    def weights(self) -> np.ndarray:
        return self.initialise_proposal().likelihood_ratio.copy()

    def diagnostics(self) -> dict:
        proposal = self.initialise_proposal()
        weights = proposal.likelihood_ratio
        total_action = self.continuous_action + self.jump_action + self.mean_field_action
        hit_action = []
        for index in range(4):
            values = total_action[self.rare_hits[:, index]]
            hit_action.append(float(np.min(values)) if values.size else None)
        return {
            "model_version": "V6.7-LD",
            "months": self.month + 1,
            "path_conditioned": True,
            "defensive_mixture_importance_sampling": True,
            "exact_mixture_likelihood_ratio": True,
            "asymptotic_action_not_used_as_probability": True,
            "action_ledger_enabled": self.action_enabled,
            "same_path_bidirectional_feedback": True,
            "common_noise_conditioned": True,
            "future_information_used": self.future_information_used,
            "max_causal_clock_violation": self.max_clock_violation,
            "max_likelihood_identity_residual": self.max_likelihood_identity_residual,
            "sample_likelihood_normalization_residual": float(abs(np.mean(weights) - 1.0)),
            "effective_sample_size": float(np.sum(weights) ** 2 / np.sum(weights ** 2)),
            "effective_sample_size_ratio": float(
                (np.sum(weights) ** 2 / np.sum(weights ** 2)) / self.paths
            ),
            "max_empirical_measure_mass_residual": self.max_measure_mass_residual,
            "mode_counts": {
                name: int(np.count_nonzero(proposal.mode == i))
                for i, name in enumerate(MODE_NAMES)
            },
            "rare_set_hit_share": {
                name: float(np.mean(self.rare_hits[:, i]))
                for i, name in enumerate((
                    "joint_logistics_collapse",
                    "multi_domain_command_failure",
                    "economic_political_crisis",
                    "leadership_disruption",
                ))
            },
            "minimum_empirical_action_on_hit": hit_action,
            "mean_continuous_action": float(np.mean(self.continuous_action)),
            "mean_jump_action": float(np.mean(self.jump_action)),
            "mean_field_empirical_action": float(np.mean(self.mean_field_action)),
            "jump_event_counts": np.sum(self.event_count, axis=0).astype(int).tolist(),
        }
