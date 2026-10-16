from __future__ import annotations

from dataclasses import dataclass

import numpy as np


ACTORS = ("china", "taiwan", "united_states", "japan")
EVENTS = (
    "none", "peaceful_protest", "general_strike", "riot", "coup",
    "revolution", "negotiated_reform", "order_restored",
)
REGIMES = ("normal", "mobilized", "repressed", "reformed", "fractured")


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -35.0, 35.0)))


@dataclass
class CollectiveActionRegimeFactors:
    collective_action: np.ndarray
    network_action: np.ndarray
    household_pressure: np.ndarray
    common_knowledge: np.ndarray
    organization_capital: np.ndarray
    government_controls: np.ndarray
    loyalty_probability: np.ndarray
    compliance: np.ndarray
    political_fragility: np.ndarray
    labor_factor: np.ndarray
    logistics_factor: np.ndarray
    morale_factor: np.ndarray
    fiscal_drag: np.ndarray
    support_shift: np.ndarray
    event_code: np.ndarray
    new_coup: np.ndarray
    new_revolution: np.ndarray
    revolution_active: np.ndarray
    revolutionary_capacity: np.ndarray
    external_intervention_pressure: np.ndarray
    external_first_share: np.ndarray
    government_replaced: np.ndarray
    revolution_suppressed: np.ndarray
    war_continuation: np.ndarray
    negotiation_pressure: np.ndarray
    negotiated_reform: np.ndarray
    order_restored: np.ndarray


class WartimeCollectiveActionRegimeSystem:
    """Pathwise monthly collective-action and political-regime closure.

    The system owns livelihood-pressure memory, common knowledge,
    organization capital, a four-state security-loyalty HMM, government MPC
    controls, social-event competing risks, and semi-Markov regime duration.
    It consumes only current same-path campaign states.  Its output factors are
    applied immediately to politics and with an explicit one-month lag to
    labor, finance, logistics, and combat availability.
    """

    def __init__(
        self, paths: int, seed: int, initial_support: np.ndarray,
        features: dict[str, bool] | None = None,
    ):
        self.paths = paths
        self.rng = np.random.default_rng(seed)
        self.features = {
            "common_knowledge_network": True,
            "organization_capital": True,
            "government_mpc": True,
            "loyalty_hmm": True,
            "competing_risks": True,
            "revolution_episode": True,
            "external_intervention_nationalism": True,
            "closed_feedback": True,
        }
        if features:
            self.features.update(features)
        self.month = 0
        self.pressure = np.clip(0.10 + 0.12 * (1.0 - initial_support), 0.02, 0.35)
        self.common_knowledge = np.clip(
            0.08 + 0.10 * (1.0 - initial_support), 0.02, 0.30
        )
        self.organization = np.broadcast_to(
            np.array([0.10, 0.14, 0.09, 0.09]), (paths, 4)
        ).copy()
        self.controls = np.broadcast_to(
            np.array([0.20, 0.10, 0.22, 0.24, 0.08]), (paths, 4, 5)
        ).copy()
        initial_loyalty = np.array([
            [0.91, 0.070, 0.017, 0.003],
            [0.86, 0.100, 0.034, 0.006],
            [0.88, 0.090, 0.026, 0.004],
            [0.89, 0.085, 0.021, 0.004],
        ])
        self.loyalty = np.broadcast_to(initial_loyalty, (paths, 4, 4)).copy()
        self.regime = np.zeros((paths, 4), dtype=np.int8)
        self.regime_duration = np.zeros((paths, 4), dtype=np.int16)
        self.first_event = np.zeros((paths, 4), dtype=np.int8)
        self.first_event_month = np.zeros((paths, 4), dtype=np.int16)
        self.event_counts = np.zeros((4, len(EVENTS)), dtype=np.int64)
        self.coup_occurred = np.zeros((paths, 4), dtype=bool)
        self.revolution_occurred = np.zeros((paths, 4), dtype=bool)
        self.revolution_active = np.zeros((paths, 4), dtype=bool)
        self.revolution_duration = np.zeros((paths, 4), dtype=np.int16)
        self.revolutionary_capacity = np.zeros((paths, 4))
        self.external_intervention_pressure = np.zeros((paths, 4))
        self.external_first_share = np.zeros((paths, 4))
        self.revolution_resolution = np.zeros((paths, 4), dtype=np.int8)
        self.revolution_episode_count = np.zeros((paths, 4), dtype=np.int16)
        self.max_loyalty_simplex_residual = 0.0
        self.max_event_probability_sum = 0.0
        self.max_control_jump = 0.0
        self.min_feedback_factor = 1.0
        self.actor_kernel = np.array([
            [1.00, 0.18, 0.08, 0.08],
            [0.22, 1.00, 0.34, 0.27],
            [0.08, 0.26, 1.00, 0.38],
            [0.08, 0.25, 0.42, 1.00],
        ])
        # Rows are targets and columns are external actors.  The two matrices
        # distinguish aid to revolutionary challengers from aid to incumbents.
        self.revolutionary_support_kernel = np.array([
            [0.00, 0.12, 0.34, 0.24],
            [0.38, 0.00, 0.18, 0.14],
            [0.16, 0.06, 0.00, 0.08],
            [0.24, 0.08, 0.16, 0.00],
        ])
        self.incumbent_support_kernel = np.array([
            [0.00, 0.03, 0.02, 0.02],
            [0.03, 0.00, 0.38, 0.27],
            [0.01, 0.05, 0.00, 0.22],
            [0.01, 0.06, 0.42, 0.00],
        ])
        self.policy_candidates = np.array([
            [0.18, 0.08, 0.24, 0.24, 0.06],
            [0.52, 0.20, 0.18, 0.16, 0.03],
            [0.32, 0.42, 0.18, 0.14, 0.02],
            [0.18, 0.10, 0.42, 0.25, 0.05],
            [0.15, 0.05, 0.20, 0.48, 0.18],
            [0.08, 0.03, 0.14, 0.40, 0.48],
            [0.34, 0.18, 0.22, 0.30, 0.10],
            [0.24, 0.30, 0.26, 0.20, 0.05],
        ])

    def _government_mpc(
        self, pressure: np.ndarray, action_seed: np.ndarray,
        fiscal: np.ndarray, support: np.ndarray, organization: np.ndarray,
    ) -> np.ndarray:
        # Six-month receding-horizon evaluation over transparent policy
        # archetypes.  The selected first control is smoothed against the
        # previous solution to avoid artificial bang-bang switching.
        n, actors = pressure.shape
        candidates = self.policy_candidates
        k = candidates.shape[0]
        controls = np.broadcast_to(candidates, (n, actors, k, 5)).copy()
        controls[..., 0] *= np.clip(fiscal[:, :, None], 0.10, 1.0)
        controls[..., 1] *= np.clip(0.55 + 0.45 * support[:, :, None], 0.30, 1.0)
        p = np.repeat(pressure[:, :, None], k, axis=2)
        a = np.repeat(action_seed[:, :, None], k, axis=2)
        legitimacy = np.repeat(support[:, :, None], k, axis=2)
        org = np.repeat(organization[:, :, None], k, axis=2)
        cost = np.zeros((n, actors, k))
        discount = 1.0
        for _ in range(6):
            relief, concession, information, policing, repression = np.moveaxis(
                controls, -1, 0
            )
            participation_risk = 0.55 * policing + 0.85 * repression
            p = np.clip(
                0.86 * p + 0.14 * pressure[:, :, None]
                - 0.11 * relief - 0.08 * concession,
                0.0, 1.0,
            )
            org = np.clip(
                0.97 * org + 0.035 * a - 0.030 * policing - 0.045 * repression,
                0.0, 1.0,
            )
            a = _sigmoid(
                -3.25 + 2.10 * p + 1.25 * org
                + 0.85 * action_seed[:, :, None]
                - 1.20 * participation_risk - 0.45 * information
            )
            legitimacy = np.clip(
                0.985 * legitimacy + 0.020 * relief + 0.030 * concession
                - 0.035 * repression * (0.4 + a), 0.0, 1.0,
            )
            fiscal_cost = (
                0.44 * relief ** 2 + 0.16 * concession ** 2
                + 0.12 * information ** 2 + 0.20 * policing ** 2
                + 0.34 * repression ** 2
            )
            cost += discount * (
                0.42 * a + 0.22 * p + 0.18 * (1.0 - legitimacy)
                + 0.18 * fiscal_cost
            )
            discount *= 0.94
        chosen = np.argmin(cost, axis=2)
        selected = np.take_along_axis(
            controls, chosen[:, :, None, None], axis=2
        )[:, :, 0, :]
        updated = 0.62 * self.controls + 0.38 * selected
        self.max_control_jump = max(
            self.max_control_jump, float(np.max(np.abs(updated - self.controls)))
        )
        return np.clip(updated, 0.0, 0.85)

    def _update_loyalty(
        self, *, command: np.ndarray, fiscal: np.ndarray,
        progress: np.ndarray, leader_continuity: np.ndarray,
        elite_split: np.ndarray, repression: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        n = self.paths
        base = np.array([
            [0.984, 0.014, 0.0016, 0.0004],
            [0.038, 0.935, 0.0240, 0.0030],
            [0.008, 0.075, 0.8840, 0.0330],
            [0.003, 0.012, 0.0750, 0.9100],
        ])
        transition = np.broadcast_to(base, (n, 4, 4, 4)).copy()
        deterioration = np.clip(
            0.32 * (1.0 - fiscal) + 0.24 * (1.0 - command)
            + 0.18 * np.clip(-progress, 0.0, 1.0)
            + 0.16 * elite_split + 0.10 * repression,
            0.0, 1.0,
        )
        stabilization = np.clip(
            0.35 * fiscal + 0.25 * command + 0.20 * np.clip(progress, 0.0, 1.0)
            + 0.20 * leader_continuity, 0.0, 1.0,
        )
        for state in range(3):
            shift = (0.010 + 0.018 * state) * deterioration
            transition[:, :, state, state] -= shift
            transition[:, :, state, state + 1] += shift
        for state in range(1, 4):
            shift = (0.006 + 0.008 * (3 - state)) * stabilization
            transition[:, :, state, state] -= shift
            transition[:, :, state, state - 1] += shift
        transition = np.clip(transition, 1e-8, None)
        transition /= transition.sum(axis=3, keepdims=True)
        predicted = np.einsum("pai,paij->paj", self.loyalty, transition)

        observed_execution = np.clip(
            0.62 * command + 0.20 * fiscal + 0.18 * leader_continuity
            - 0.22 * elite_split, 0.0, 1.0,
        )
        emission_mean = np.array([0.91, 0.72, 0.49, 0.24])
        emission_scale = np.array([0.16, 0.18, 0.20, 0.22])
        likelihood = np.exp(
            -0.5 * (
                (observed_execution[:, :, None] - emission_mean[None, None, :])
                / emission_scale[None, None, :]
            ) ** 2
        ) + 1e-12
        posterior = predicted * likelihood
        posterior /= posterior.sum(axis=2, keepdims=True)
        residual = float(np.max(np.abs(posterior.sum(axis=2) - 1.0)))
        self.max_loyalty_simplex_residual = max(
            self.max_loyalty_simplex_residual, residual
        )
        compliance = posterior @ np.array([1.00, 0.72, 0.42, 0.12])
        return posterior, np.clip(compliance, 0.05, 1.0)

    def _draw_competing_event(
        self, *, action: np.ndarray, pressure: np.ndarray,
        organization: np.ndarray, common_knowledge: np.ndarray,
        fragility: np.ndarray, compliance: np.ndarray,
        concession: np.ndarray, repression: np.ndarray,
    ) -> np.ndarray:
        mobilized = self.regime == 1
        repressed = self.regime == 2
        fractured = self.regime == 4
        low_loyalty = 1.0 - compliance
        # Monthly cause-specific intensities.  Coefficients are transparent
        # structural priors, not observations of a future Taiwan conflict.
        hazards = np.stack((
            0.0015 + 0.028 * action + 0.012 * common_knowledge,
            0.0006 + 0.016 * action * organization + 0.006 * pressure,
            0.00015 + 0.010 * action * pressure + 0.006 * repression * action,
            0.00001 + 0.006 * fragility * low_loyalty * (0.4 + organization),
            0.000002 + 0.004 * fragility * low_loyalty * organization * common_knowledge,
            0.00008 + 0.012 * concession * (action + organization) * (mobilized | repressed),
            0.00010 + 0.020 * (1.0 - action) * (1.0 - pressure)
            * (mobilized | repressed | fractured),
        ), axis=2)
        hazards[:, :, 0] *= np.where(mobilized | repressed, 0.55, 1.0)
        hazards[:, :, 1] *= np.where(mobilized, 1.25, 1.0)
        hazards[:, :, 2] *= np.where(repressed, 1.35, 1.0)
        hazards[:, :, 3] *= np.where(fractured, 1.80, 1.0)
        hazards[:, :, 4] *= np.where(fractured, 2.20, 1.0)
        hazards[:, :, 4] *= (~self.revolution_active)
        hazards[:, :, 6] *= (~self.revolution_active)
        hazards = np.clip(hazards, 0.0, 0.15)
        total = hazards.sum(axis=2)
        p_any = 1.0 - np.exp(-total)
        probabilities = hazards / np.maximum(total[:, :, None], 1e-12)
        probabilities *= p_any[:, :, None]
        self.max_event_probability_sum = max(
            self.max_event_probability_sum, float(np.max(probabilities.sum(axis=2)))
        )
        draw = self.rng.random((self.paths, 4))
        cumulative = np.cumsum(probabilities, axis=2)
        selected = 1 + np.sum(draw[:, :, None] > cumulative, axis=2)
        selected = np.where(draw < p_any, selected, 0)
        return np.clip(selected, 0, 7).astype(np.int8)

    def _update_revolution_episode(
        self, *, new_revolution: np.ndarray, action: np.ndarray,
        organization: np.ndarray, fragility: np.ndarray,
        compliance: np.ndarray, repression: np.ndarray,
        support: np.ndarray, actor_norm: np.ndarray, pressure: np.ndarray,
        progress: np.ndarray, conflict_intensity: np.ndarray,
        war_intervention: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if not self.features["revolution_episode"]:
            self.revolution_active.fill(False)
            self.revolution_duration.fill(0)
            self.revolutionary_capacity.fill(0.0)
            self.external_intervention_pressure *= 0.90
            self.external_first_share *= 0.90
            war_continuation = _sigmoid(
                -0.10 + 0.72 * support + 0.52 * actor_norm
                + 0.45 * np.clip(progress, -1.0, 1.0)
                - 0.85 * pressure - 0.62 * action
            )
            return (
                np.zeros_like(new_revolution), np.zeros_like(new_revolution),
                war_continuation, 1.0 - war_continuation,
            )
        starting = new_revolution & (~self.revolution_active)
        self.revolution_active |= starting
        self.revolution_episode_count += starting
        self.revolution_duration = np.where(
            starting, 0,
            np.where(
                self.revolution_active,
                np.minimum(self.revolution_duration + 1, np.iinfo(np.int16).max),
                0,
            ),
        )
        self.revolutionary_capacity = np.where(
            starting,
            np.maximum(
                self.revolutionary_capacity,
                np.clip(0.18 + 0.34 * organization + 0.24 * action, 0.0, 0.75),
            ),
            self.revolutionary_capacity,
        )
        external_activity = np.clip(war_intervention, 0.0, 1.0)
        if not self.features["external_intervention_nationalism"]:
            external_activity = np.zeros_like(external_activity)
        support_revolution = np.clip(
            external_activity @ self.revolutionary_support_kernel.T,
            0.0, 1.0,
        ) * self.revolution_active * (0.25 + 0.75 * fragility)
        support_incumbent = np.clip(
            external_activity @ self.incumbent_support_kernel.T,
            0.0, 1.0,
        ) * self.revolution_active * (0.30 + 0.70 * compliance)
        external_total = np.clip(support_revolution + support_incumbent, 0.0, 1.0)
        self.external_intervention_pressure = np.clip(
            0.72 * self.external_intervention_pressure + 0.28 * external_total,
            0.0, 1.0,
        )
        external_first_target = _sigmoid(
            -1.35 + 1.85 * self.external_intervention_pressure
            + 0.85 * actor_norm + 0.55 * support
            + 0.65 * conflict_intensity[:, None] - 0.70 * pressure
        )
        self.external_first_share = np.clip(
            0.82 * self.external_first_share + 0.18 * external_first_target,
            0.0, 1.0,
        )
        if not self.features["external_intervention_nationalism"]:
            self.external_first_share.fill(0.0)
        growth = (
            0.045 * organization + 0.040 * action
            + 0.035 * (1.0 - compliance) + 0.030 * support_revolution
        )
        attrition = (
            0.045 * repression + 0.040 * compliance
            + 0.030 * support_incumbent + 0.040 * self.external_first_share
        )
        self.revolutionary_capacity = np.where(
            self.revolution_active,
            np.clip(0.94 * self.revolutionary_capacity + growth - attrition, 0.0, 1.0),
            0.90 * self.revolutionary_capacity,
        )
        eligible = self.revolution_active & (self.revolution_duration >= 1)
        replace_hazard = np.clip(
            0.0005 + 0.055 * self.revolutionary_capacity
            * (1.0 - compliance) * (1.0 - 0.55 * self.external_first_share)
            + 0.018 * support_revolution - 0.012 * support_incumbent,
            0.0, 0.20,
        ) * eligible
        suppress_hazard = np.clip(
            0.0010 + 0.045 * compliance * (0.35 + repression)
            + 0.022 * support_incumbent + 0.018 * self.external_first_share
            - 0.020 * self.revolutionary_capacity,
            0.0, 0.20,
        ) * eligible
        total = replace_hazard + suppress_hazard
        p_resolution = 1.0 - np.exp(-total)
        draw = self.rng.random((self.paths, 4))
        replace_probability = p_resolution * replace_hazard / np.maximum(total, 1e-12)
        government_replaced = eligible & (draw < replace_probability)
        revolution_suppressed = eligible & (
            (draw >= replace_probability) & (draw < p_resolution)
        )
        resolved = government_replaced | revolution_suppressed
        self.revolution_resolution = np.where(
            government_replaced, 2,
            np.where(revolution_suppressed, 1, self.revolution_resolution),
        )
        self.revolution_active &= ~resolved
        self.revolution_duration = np.where(resolved, 0, self.revolution_duration)
        self.revolutionary_capacity = np.where(
            government_replaced, 0.18 * self.revolutionary_capacity,
            np.where(revolution_suppressed, 0.08 * self.revolutionary_capacity,
                     self.revolutionary_capacity),
        )
        war_continuation = _sigmoid(
            -0.10 + 1.10 * self.external_first_share + 0.72 * support
            + 0.52 * actor_norm + 0.45 * np.clip(progress, -1.0, 1.0)
            - 0.85 * pressure - 0.62 * action
        )
        # Resolution changes institutions and leaders, but does not impose an
        # ideology-specific foreign-policy rule.  War/talk remains endogenous.
        war_continuation = np.clip(
            war_continuation + 0.06 * revolution_suppressed
            - 0.03 * government_replaced, 0.0, 1.0,
        )
        negotiation_pressure = np.clip(1.0 - war_continuation, 0.0, 1.0)
        return (
            government_replaced, revolution_suppressed,
            war_continuation, negotiation_pressure,
        )

    def update(
        self, *, month: int, support: np.ndarray, action_seed: np.ndarray,
        network_action: np.ndarray, actor_network: np.ndarray,
        actor_norm: np.ndarray, actor_panic: np.ndarray,
        shortage: np.ndarray, inflation: np.ndarray, output: np.ndarray,
        damage: np.ndarray, financial_stress: np.ndarray,
        fiscal_capacity: np.ndarray, progress: np.ndarray,
        command: np.ndarray, leader_continuity: np.ndarray,
        social_mobilization: np.ndarray, social_fatigue: np.ndarray,
        conflict_intensity: np.ndarray, war_intervention: np.ndarray,
    ) -> CollectiveActionRegimeFactors:
        self.month = month + 1
        lagged_relief = self.controls[:, :, 0]
        livelihood_gap = np.clip(
            0.34 * shortage + 0.18 * np.clip(4.0 * inflation, 0.0, 1.0)
            + 0.18 * (1.0 - output) + 0.12 * financial_stress
            + 0.12 * damage + 0.10 * social_fatigue - 0.14 * lagged_relief,
            0.0, 1.0,
        )
        target_pressure = _sigmoid(5.0 * (livelihood_gap - 0.34))
        self.pressure = np.clip(
            0.84 * self.pressure + 0.16 * target_pressure, 0.0, 1.0
        )

        verified_signal = np.clip(
            0.24 * shortage + 0.20 * damage + 0.18 * financial_stress
            + 0.18 * np.clip(-progress, 0.0, 1.0)
            + 0.20 * social_fatigue, 0.0, 1.0,
        )
        if self.features["common_knowledge_network"]:
            cross_actor = np.clip(network_action @ self.actor_kernel.T, 0.0, 1.0)
        else:
            cross_actor = np.clip(network_action, 0.0, 1.0)
        information_control = self.controls[:, :, 2]
        credibility_gap = np.clip(
            verified_signal - 0.45 * information_control, 0.0, 1.0
        )
        self.common_knowledge = np.clip(
            0.76 * self.common_knowledge
            + 0.10 * cross_actor + 0.08 * verified_signal
            + 0.08 * credibility_gap - 0.05 * information_control,
            0.0, 1.0,
        )

        if self.features["government_mpc"]:
            self.controls = self._government_mpc(
                self.pressure, action_seed, fiscal_capacity, support,
                self.organization,
            )
        relief, concession, information, policing, repression = np.moveaxis(
            self.controls, -1, 0
        )
        participation_risk = np.clip(0.52 * policing + 0.88 * repression, 0.0, 1.0)
        organization_effect = (
            self.organization
            if self.features["organization_capital"]
            else np.zeros_like(self.organization)
        )
        success_belief = np.clip(
            0.32 * self.common_knowledge + 0.28 * organization_effect
            + 0.22 * (1.0 - leader_continuity)
            + 0.18 * np.clip(-progress, 0.0, 1.0), 0.0, 1.0,
        )
        action = _sigmoid(
            -4.05 + 1.85 * self.pressure + 1.35 * self.common_knowledge
            + 1.20 * organization_effect + 0.85 * network_action
            + 0.55 * success_belief - 1.25 * participation_risk
            - 0.55 * self.external_first_share
            + 0.45 * action_seed + 0.25 * social_mobilization
        )
        if self.features["organization_capital"]:
            self.organization = np.clip(
                0.965 * self.organization + 0.030 * action
                + 0.018 * self.common_knowledge + 0.012 * actor_network
                - 0.024 * policing - 0.040 * repression
                - 0.010 * (1.0 - actor_network), 0.0, 1.0,
            )

        elite_split = np.clip(
            0.32 * (1.0 - leader_continuity) + 0.24 * financial_stress
            + 0.22 * np.clip(-progress, 0.0, 1.0)
            + 0.22 * (1.0 - support), 0.0, 1.0,
        )
        if self.features["loyalty_hmm"]:
            self.loyalty, compliance = self._update_loyalty(
                command=command, fiscal=fiscal_capacity, progress=progress,
                leader_continuity=leader_continuity, elite_split=elite_split,
                repression=repression,
            )
        else:
            compliance = self.loyalty @ np.array([1.00, 0.72, 0.42, 0.12])
        fragility = np.clip(
            0.20 * self.pressure + 0.16 * action
            + 0.13 * organization_effect + 0.13 * self.common_knowledge
            + 0.16 * (1.0 - compliance) + 0.10 * elite_split
            + 0.07 * actor_panic + 0.05 * (1.0 - actor_norm), 0.0, 1.0,
        )
        if self.features["competing_risks"]:
            event = self._draw_competing_event(
                action=action, pressure=self.pressure,
                organization=organization_effect,
                common_knowledge=self.common_knowledge, fragility=fragility,
                compliance=compliance, concession=concession,
                repression=repression,
            )
        else:
            event = np.zeros((self.paths, 4), dtype=np.int8)
        for actor in range(4):
            self.event_counts[actor] += np.bincount(
                event[:, actor], minlength=len(EVENTS)
            )
        first = (self.first_event == 0) & (event > 0)
        self.first_event[first] = event[first]
        self.first_event_month[first] = month + 1

        protest = event == 1
        strike = event == 2
        riot = event == 3
        new_coup = (event == 4) & (~self.coup_occurred)
        new_revolution = (
            (event == 5) & (~self.revolution_occurred)
            & self.features["revolution_episode"]
        )
        reform = event == 6
        restored = event == 7
        self.coup_occurred |= new_coup
        self.revolution_occurred |= new_revolution
        (
            government_replaced, revolution_suppressed,
            war_continuation, negotiation_pressure,
        ) = self._update_revolution_episode(
            new_revolution=new_revolution,
            action=action,
            organization=organization_effect,
            fragility=fragility,
            compliance=compliance,
            repression=repression,
            support=support,
            actor_norm=actor_norm,
            pressure=self.pressure,
            progress=progress,
            conflict_intensity=conflict_intensity,
            war_intervention=war_intervention,
        )
        self.regime = np.where(protest | strike, 1, self.regime)
        self.regime = np.where(riot, 2, self.regime)
        self.regime = np.where(new_coup | new_revolution | self.revolution_active, 4, self.regime)
        self.regime = np.where(revolution_suppressed, 2, self.regime)
        self.regime = np.where(government_replaced, 3, self.regime)
        self.regime = np.where(reform, 3, self.regime)
        self.regime = np.where(restored, 0, self.regime)
        changed = (
            protest | strike | riot | new_coup | new_revolution | reform | restored
            | government_replaced | revolution_suppressed
        )
        self.regime_duration = np.where(
            changed, 0,
            np.minimum(self.regime_duration + 1, np.iinfo(np.int16).max),
        )

        event_disruption = (
            0.04 * protest + 0.18 * strike + 0.14 * riot
            + 0.22 * new_coup + 0.34 * new_revolution
            + 0.18 * self.revolution_active
        )
        labor_factor = np.clip(
            1.0 - 0.24 * action - event_disruption, 0.35, 1.0
        )
        logistics_factor = np.clip(
            1.0 - 0.12 * action - 0.08 * strike - 0.12 * riot
            - 0.18 * new_coup - 0.25 * new_revolution, 0.45, 1.0,
        )
        morale_factor = np.clip(
            0.70 + 0.30 * compliance - 0.08 * riot
            - 0.16 * new_coup - 0.24 * new_revolution
            - 0.10 * self.revolution_active - 0.14 * government_replaced
            + 0.04 * restored, 0.35, 1.02,
        )
        fiscal_drag = np.clip(
            0.030 * relief + 0.010 * information + 0.018 * policing
            + 0.035 * repression + 0.018 * action + 0.020 * riot
            + 0.018 * self.external_intervention_pressure,
            0.0, 0.12,
        )
        support_shift = np.clip(
            0.014 * relief + 0.020 * concession - 0.018 * repression * (0.4 + action)
            - 0.020 * riot - 0.045 * new_coup - 0.070 * new_revolution
            + 0.040 * reform + 0.020 * restored, -0.12, 0.08,
        )
        support_shift += np.clip(
            -0.035 * government_replaced
            - 0.025 * self.revolution_active
            + 0.018 * revolution_suppressed,
            -0.08, 0.04,
        )
        support_shift = np.clip(support_shift, -0.12, 0.08)
        if not self.features["closed_feedback"]:
            labor_factor = np.ones_like(labor_factor)
            logistics_factor = np.ones_like(logistics_factor)
            morale_factor = np.ones_like(morale_factor)
            fiscal_drag = np.zeros_like(fiscal_drag)
            support_shift = np.zeros_like(support_shift)
        self.min_feedback_factor = min(
            self.min_feedback_factor, float(np.min(labor_factor)),
            float(np.min(logistics_factor)), float(np.min(morale_factor)),
        )
        self.common_knowledge = np.clip(
            self.common_knowledge + 0.025 * protest + 0.035 * strike
            + 0.050 * riot - 0.030 * restored, 0.0, 1.0,
        )
        self.organization = np.clip(
            self.organization + 0.020 * protest + 0.025 * strike
            - 0.040 * new_coup - 0.055 * new_revolution
            - 0.025 * restored, 0.0, 1.0,
        )
        network_action_out = np.clip(action @ self.actor_kernel.T, 0.0, 1.0)
        return CollectiveActionRegimeFactors(
            collective_action=action,
            network_action=network_action_out,
            household_pressure=self.pressure.copy(),
            common_knowledge=self.common_knowledge.copy(),
            organization_capital=self.organization.copy(),
            government_controls=self.controls.copy(),
            loyalty_probability=self.loyalty.copy(),
            compliance=compliance,
            political_fragility=fragility,
            labor_factor=labor_factor,
            logistics_factor=logistics_factor,
            morale_factor=morale_factor,
            fiscal_drag=fiscal_drag,
            support_shift=support_shift,
            event_code=event,
            new_coup=new_coup,
            new_revolution=new_revolution,
            revolution_active=self.revolution_active.copy(),
            revolutionary_capacity=self.revolutionary_capacity.copy(),
            external_intervention_pressure=self.external_intervention_pressure.copy(),
            external_first_share=self.external_first_share.copy(),
            government_replaced=government_replaced,
            revolution_suppressed=revolution_suppressed,
            war_continuation=war_continuation,
            negotiation_pressure=negotiation_pressure,
            negotiated_reform=reform,
            order_restored=restored,
        )

    def diagnostics(self) -> dict:
        regime_share = {
            label: float(np.mean(self.regime == i))
            for i, label in enumerate(REGIMES)
        }
        return {
            "model": "V5.7 pathwise household-network-organization-MPC-HMM-competing-risk closure",
            "features": self.features,
            "months": self.month,
            "max_loyalty_simplex_residual": self.max_loyalty_simplex_residual,
            "max_event_probability_sum": self.max_event_probability_sum,
            "max_control_jump": self.max_control_jump,
            "minimum_feedback_factor": self.min_feedback_factor,
            "event_counts": {
                actor: {
                    event: int(self.event_counts[i, j])
                    for j, event in enumerate(EVENTS)
                }
                for i, actor in enumerate(ACTORS)
            },
            "first_event_shares": {
                actor: {
                    event: float(np.mean(self.first_event[:, i] == j))
                    for j, event in enumerate(EVENTS)
                }
                for i, actor in enumerate(ACTORS)
            },
            "terminal_regime_shares": regime_share,
            "terminal_pressure_p10_p50_p90": {
                actor: [float(x) for x in np.quantile(self.pressure[:, i], (0.1, 0.5, 0.9))]
                for i, actor in enumerate(ACTORS)
            },
            "terminal_organization_p10_p50_p90": {
                actor: [float(x) for x in np.quantile(self.organization[:, i], (0.1, 0.5, 0.9))]
                for i, actor in enumerate(ACTORS)
            },
            "terminal_loyalty_state_mean": {
                actor: [float(x) for x in np.mean(self.loyalty[:, i, :], axis=0)]
                for i, actor in enumerate(ACTORS)
            },
            "revolution_episode_share": {
                actor: float(np.mean(self.revolution_episode_count[:, i] > 0))
                for i, actor in enumerate(ACTORS)
            },
            "active_revolution_share": {
                actor: float(np.mean(self.revolution_active[:, i]))
                for i, actor in enumerate(ACTORS)
            },
            "revolution_resolution_shares": {
                actor: {
                    "unresolved_or_none": float(np.mean(self.revolution_resolution[:, i] == 0)),
                    "suppressed": float(np.mean(self.revolution_resolution[:, i] == 1)),
                    "government_replaced": float(np.mean(self.revolution_resolution[:, i] == 2)),
                }
                for i, actor in enumerate(ACTORS)
            },
        }
