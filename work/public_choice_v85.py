from __future__ import annotations

from dataclasses import dataclass

import numpy as np


ACTORS = ("china", "taiwan", "united_states", "japan")
BUDGET_USES = ("military", "public_investment", "public_services", "transfers")


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -35.0, 35.0)))


def _softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - np.max(x, axis=-1, keepdims=True)
    exp = np.exp(np.clip(shifted, -35.0, 35.0))
    return exp / np.maximum(exp.sum(axis=-1, keepdims=True), 1.0e-12)


@dataclass(frozen=True)
class PublicChoiceFactors:
    budget_shares: np.ndarray
    policy_feasibility: np.ndarray
    bureaucratic_execution: np.ndarray
    procurement_factor: np.ndarray
    military_order_factor: np.ndarray
    civilian_order_factor: np.ndarray
    public_service_factor: np.ndarray
    alliance_multiplier: np.ndarray
    aid_authorization: np.ndarray
    information_quality: np.ndarray
    support_shift: np.ndarray
    fiscal_drag: np.ndarray
    rent_loss: np.ndarray
    common_pool_pressure: np.ndarray
    policy_block_event: np.ndarray


class FourPartyPublicChoiceSystemV85:
    """Monthly same-path public-choice state for the four actors.

    This layer owns political feasibility, coalition support, bureaucratic
    execution, lobbying/rent pressure, fiscal illusion and commitment
    reputation. It does not own fiscal stocks, procurement inventories,
    public opinion densities, election observations or terminal hazards.
    Outputs alter the feasible execution of those owner modules in the same
    month and are never applied as a post-hoc probability adjustment.
    """

    def __init__(
        self,
        paths: int,
        seed: int,
        initial_support: np.ndarray,
        features: dict[str, bool] | None = None,
    ) -> None:
        self.paths = int(paths)
        self.rng = np.random.default_rng(seed)
        self.features = {
            "electoral_competition": True,
            "selectorate_coalition": True,
            "bureaucratic_execution": True,
            "interest_group_rent": True,
            "fiscal_illusion_common_pool": True,
            "veto_points": True,
            "commitment_reputation": True,
            "alliance_two_level": True,
            "feedback_enabled": True,
        }
        if features:
            unknown = set(features) - set(self.features)
            if unknown:
                raise ValueError(f"unknown public-choice features: {sorted(unknown)}")
            self.features.update(features)

        support = np.asarray(initial_support, dtype=float)
        if support.shape != (self.paths, 4):
            raise ValueError("initial_support must have shape (paths, 4)")

        self.months = 0
        self.coalition_support = np.clip(0.35 + 0.55 * support, 0.08, 0.96)
        self.bureaucratic_efficiency = np.broadcast_to(
            np.array([0.78, 0.74, 0.76, 0.79]), (self.paths, 4)
        ).copy()
        self.lobby_power = np.broadcast_to(
            np.array([0.20, 0.28, 0.48, 0.38]), (self.paths, 4)
        ).copy()
        self.fiscal_illusion = np.broadcast_to(
            np.array([0.42, 0.34, 0.46, 0.39]), (self.paths, 4)
        ).copy()
        self.reputation = np.broadcast_to(
            np.array([0.70, 0.68, 0.72, 0.74]), (self.paths, 4)
        ).copy()
        self.veto_memory = np.zeros((self.paths, 4))
        self.block_duration = np.zeros((self.paths, 4), dtype=np.int8)
        self.base_budget_shares = np.array([
            [0.34, 0.25, 0.23, 0.18],
            [0.32, 0.24, 0.24, 0.20],
            [0.36, 0.23, 0.22, 0.19],
            [0.31, 0.26, 0.24, 0.19],
        ])
        self.electoral_responsiveness = np.array([0.18, 0.78, 0.74, 0.62])
        self.selectorate_responsiveness = np.array([0.82, 0.46, 0.42, 0.50])
        self.veto_friction = np.array([0.24, 0.62, 0.78, 0.66])
        self.lobby_openness = np.array([0.30, 0.48, 0.86, 0.70])
        self.review_period = np.array([60, 48, 48, 36], dtype=int)
        self.review_phase = np.array([12, 0, 18, 9], dtype=int)

        self.max_budget_simplex_residual = 0.0
        self.max_probability_bound_violation = 0.0
        self.max_feedback_shift = 0.0
        self.max_state_jump = 0.0
        self.max_fiscal_drag = 0.0
        self.min_execution = 1.0
        self.policy_block_counts = np.zeros(4, dtype=np.int64)
        self.future_information_used = False
        self.terminal_probability_adjustment = False
        self.structural_state_duplication = False

    def _cycle_proximity(self, month: int) -> np.ndarray:
        position = (month - self.review_phase) % self.review_period
        distance = np.minimum(position, self.review_period - position)
        width = np.maximum(self.review_period / 8.0, 2.0)
        proximity = np.exp(-0.5 * (distance / width) ** 2)
        return np.broadcast_to(proximity, (self.paths, 4)).copy()

    def update(
        self,
        *,
        month: int,
        actor_support: np.ndarray,
        actor_damage: np.ndarray,
        actor_shortage: np.ndarray,
        actor_output: np.ndarray,
        financial_capacity: np.ndarray,
        debt_stress: np.ndarray,
        leader_continuity: np.ndarray,
        political_resilience: np.ndarray,
        actor_network: np.ndarray,
        actor_panic: np.ndarray,
        actor_action: np.ndarray,
        actor_inflation: np.ndarray,
        conflict_intensity: np.ndarray,
        intervention_ceiling: np.ndarray,
        macro_government_balance: np.ndarray,
    ) -> PublicChoiceFactors:
        arrays = {
            "actor_support": actor_support,
            "actor_damage": actor_damage,
            "actor_shortage": actor_shortage,
            "actor_output": actor_output,
            "financial_capacity": financial_capacity,
            "debt_stress": debt_stress,
            "leader_continuity": leader_continuity,
            "political_resilience": political_resilience,
            "actor_network": actor_network,
            "actor_panic": actor_panic,
            "actor_action": actor_action,
            "actor_inflation": actor_inflation,
            "macro_government_balance": macro_government_balance,
        }
        clean: dict[str, np.ndarray] = {}
        for name, value in arrays.items():
            item = np.asarray(value, dtype=float)
            if item.shape != (self.paths, 4):
                raise ValueError(f"{name} must have shape (paths, 4)")
            if not np.all(np.isfinite(item)):
                raise ValueError(f"{name} contains non-finite values")
            clean[name] = item
        intensity = np.asarray(conflict_intensity, dtype=float)
        if intensity.shape != (self.paths,):
            raise ValueError("conflict_intensity must have shape (paths,)")
        ceilings = np.asarray(intervention_ceiling, dtype=float)
        if ceilings.shape == (2,):
            ceilings = np.broadcast_to(ceilings, (self.paths, 2)).copy()
        if ceilings.shape != (self.paths, 2):
            raise ValueError("intervention_ceiling must have shape (paths, 2) or (2,)")

        support = np.clip(clean["actor_support"], 0.01, 0.99)
        damage = np.clip(clean["actor_damage"], 0.0, 1.0)
        shortage = np.clip(clean["actor_shortage"], 0.0, 1.0)
        output = np.clip(clean["actor_output"], 0.0, 1.3)
        fiscal = np.clip(clean["financial_capacity"], 0.02, 1.0)
        debt = np.clip(clean["debt_stress"], 0.0, 1.5)
        continuity = np.clip(clean["leader_continuity"], 0.0, 1.0)
        resilience = np.clip(clean["political_resilience"], 0.0, 1.0)
        network = np.clip(clean["actor_network"], 0.0, 1.0)
        panic = np.clip(clean["actor_panic"], 0.0, 1.0)
        action = np.clip(clean["actor_action"], 0.0, 1.0)
        inflation = np.clip(clean["actor_inflation"], -0.05, 0.80)
        government_balance = np.clip(clean["macro_government_balance"], -1.0, 1.0)
        conflict = np.broadcast_to(np.clip(intensity, 0.0, 1.0)[:, None], (self.paths, 4))

        cycle = self._cycle_proximity(month)
        electoral = (
            cycle * self.electoral_responsiveness[None, :]
            if self.features["electoral_competition"] else np.zeros_like(cycle)
        )
        selectorate = (
            self.selectorate_responsiveness[None, :]
            if self.features["selectorate_coalition"] else np.zeros((1, 4))
        )
        coalition_target = _sigmoid(
            -0.15 + 1.25 * (support - 0.50) + 0.48 * (output - 0.65)
            + 0.35 * resilience + 0.24 * continuity - 0.62 * damage
            - 0.50 * shortage - 0.22 * panic
            + selectorate * (0.20 * fiscal - 0.12 * debt)
        )
        prior_coalition = self.coalition_support.copy()
        self.coalition_support = np.clip(
            0.90 * self.coalition_support + 0.10 * coalition_target, 0.02, 0.99
        )

        lobby_target = np.clip(
            self.lobby_openness[None, :]
            * (0.18 + 0.42 * conflict + 0.22 * shortage + 0.18 * debt),
            0.0, 0.95,
        )
        if self.features["interest_group_rent"]:
            self.lobby_power = np.clip(
                0.92 * self.lobby_power + 0.08 * lobby_target, 0.0, 0.95
            )
            rent_loss = np.clip(
                0.055 * self.lobby_power * (0.55 + 0.45 * (1.0 - network))
                + 0.020 * self.lobby_power * debt,
                0.0, 0.09,
            )
        else:
            rent_loss = np.zeros((self.paths, 4))

        if self.features["fiscal_illusion_common_pool"]:
            illusion_target = np.clip(
                0.25 + 0.32 * np.maximum(inflation, 0.0)
                + 0.18 * np.maximum(-government_balance, 0.0)
                + 0.18 * (1.0 - network), 0.05, 0.90,
            )
            self.fiscal_illusion = np.clip(
                0.94 * self.fiscal_illusion + 0.06 * illusion_target, 0.02, 0.95
            )
            common_pool = np.clip(
                self.fiscal_illusion * (0.42 * debt + 0.28 * conflict + 0.30 * electoral),
                0.0, 1.0,
            )
        else:
            common_pool = np.zeros((self.paths, 4))

        bureaucracy_target = np.clip(
            0.28 + 0.30 * fiscal + 0.18 * network + 0.14 * continuity
            + 0.10 * resilience - 0.24 * damage - 0.16 * shortage
            - 1.10 * rent_loss,
            0.08, 0.98,
        )
        prior_bureaucracy = self.bureaucratic_efficiency.copy()
        if self.features["bureaucratic_execution"]:
            self.bureaucratic_efficiency = np.clip(
                0.88 * self.bureaucratic_efficiency + 0.12 * bureaucracy_target,
                0.05, 0.99,
            )
        else:
            self.bureaucratic_efficiency[:] = 1.0

        acceptance_score = (
            0.10 + 0.90 * self.coalition_support + 0.35 * support
            + 0.20 * continuity - 0.45 * damage - 0.35 * shortage
            - self.veto_friction[None, :] * (0.28 * electoral + 0.22 * action)
        )
        if self.features["veto_points"]:
            feasibility = np.clip(_sigmoid(2.7 * (acceptance_score - 0.52)), 0.04, 0.99)
            block_hazard = np.clip(0.045 * (0.42 - feasibility), 0.0, 0.025)
            new_block = self.rng.random((self.paths, 4)) < block_hazard
            self.block_duration = np.where(new_block, 2, np.maximum(self.block_duration - 1, 0))
            blocked = self.block_duration > 0
            feasibility = np.where(blocked, 0.55 * feasibility, feasibility)
            self.veto_memory = np.clip(0.85 * self.veto_memory + 0.15 * blocked, 0.0, 1.0)
            self.policy_block_counts += np.sum(new_block, axis=0)
        else:
            blocked = np.zeros((self.paths, 4), dtype=bool)
            feasibility = np.ones((self.paths, 4))

        scores = np.log(np.maximum(self.base_budget_shares[None, :, :], 1.0e-9))
        scores = np.broadcast_to(scores, (self.paths, 4, 4)).copy()
        scores[:, :, 0] += 1.05 * conflict + 0.50 * damage + 0.30 * self.lobby_power
        scores[:, :, 1] += 0.62 * damage + 0.42 * shortage + 0.24 * (1.0 - network)
        scores[:, :, 2] += 0.55 * shortage + 0.42 * panic + 0.24 * (1.0 - output)
        scores[:, :, 3] += 0.45 * action + 0.32 * electoral + 0.30 * np.maximum(inflation, 0.0)
        scores[:, :, 0] += np.array([0.12, 0.08, 0.10, 0.06])[None, :]
        raw_shares = _softmax(scores)
        base = np.broadcast_to(self.base_budget_shares, raw_shares.shape)
        budget_shares = base + feasibility[:, :, None] * (raw_shares - base)
        budget_shares /= np.maximum(budget_shares.sum(axis=2, keepdims=True), 1.0e-12)

        execution = np.clip(
            (0.55 + 0.45 * self.bureaucratic_efficiency)
            * (0.72 + 0.28 * feasibility) * (1.0 - rent_loss),
            0.45, 1.05,
        )
        military_ratio = budget_shares[:, :, 0] / self.base_budget_shares[None, :, 0]
        civilian_ratio = (
            budget_shares[:, :, 1:].sum(axis=2)
            / self.base_budget_shares[None, :, 1:].sum(axis=2)
        )
        procurement = np.clip(
            0.82 + 0.18 * execution + 0.16 * (military_ratio - 1.0)
            - 0.80 * rent_loss,
            0.72, 1.16,
        )
        military_order = np.clip(0.80 + 0.20 * execution + 0.24 * (military_ratio - 1.0), 0.70, 1.22)
        civilian_order = np.clip(0.84 + 0.16 * execution + 0.20 * (civilian_ratio - 1.0), 0.72, 1.18)
        public_service = np.clip(
            execution * (
                0.45 * budget_shares[:, :, 1] / self.base_budget_shares[None, :, 1]
                + 0.55 * budget_shares[:, :, 2] / self.base_budget_shares[None, :, 2]
            ),
            0.55, 1.20,
        )

        if self.features["commitment_reputation"]:
            promised = np.clip(0.55 * feasibility + 0.45 * self.coalition_support, 0.0, 1.0)
            delivered = np.clip(execution * (1.0 - 0.5 * rent_loss), 0.0, 1.0)
            self.reputation = np.clip(
                0.94 * self.reputation + 0.06 * delivered
                - 0.08 * np.maximum(promised - delivered, 0.0),
                0.05, 0.99,
            )
        else:
            self.reputation[:] = 0.70

        information_quality = np.clip(
            (0.48 + 0.30 * network + 0.22 * self.reputation)
            * (1.0 - 0.55 * rent_loss) * (0.82 + 0.18 * feasibility),
            0.25, 1.0,
        )
        support_shift = np.clip(
            0.010 * (public_service - 1.0)
            + 0.008 * (self.reputation - 0.68)
            - 0.020 * rent_loss - 0.006 * blocked,
            -0.025, 0.018,
        )
        fiscal_drag = np.clip(
            0.55 * rent_loss + 0.020 * common_pool + 0.012 * blocked
            + 0.010 * np.maximum(0.65 - execution, 0.0),
            0.0, 0.10,
        )

        domestic_score = np.clip(
            0.34 * feasibility[:, 2:] + 0.30 * self.coalition_support[:, 2:]
            + 0.20 * support[:, 2:] + 0.16 * self.reputation[:, 2:]
            - 0.16 * damage[:, 2:] - 0.12 * debt[:, 2:],
            0.0, 1.0,
        )
        if self.features["alliance_two_level"]:
            alliance_multiplier = np.clip(
                1.0 + 0.22 * (domestic_score - 0.58) * np.clip(ceilings, 0.0, 1.0),
                0.84, 1.10,
            )
            aid_authorization = np.clip(
                0.76 + 0.42 * domestic_score + 0.10 * information_quality[:, 2:]
                - 0.15 * rent_loss[:, 2:],
                0.65, 1.22,
            )
        else:
            alliance_multiplier = np.ones((self.paths, 2))
            aid_authorization = np.ones((self.paths, 2))

        if not self.features["feedback_enabled"]:
            budget_shares = base.copy()
            feasibility = np.ones((self.paths, 4))
            execution = np.ones((self.paths, 4))
            procurement = np.ones((self.paths, 4))
            military_order = np.ones((self.paths, 4))
            civilian_order = np.ones((self.paths, 4))
            public_service = np.ones((self.paths, 4))
            alliance_multiplier = np.ones((self.paths, 2))
            aid_authorization = np.ones((self.paths, 2))
            information_quality = np.ones((self.paths, 4))
            support_shift = np.zeros((self.paths, 4))
            fiscal_drag = np.zeros((self.paths, 4))
            rent_loss = np.zeros((self.paths, 4))
            common_pool = np.zeros((self.paths, 4))
            blocked = np.zeros((self.paths, 4), dtype=bool)

        simplex = float(np.max(np.abs(budget_shares.sum(axis=2) - 1.0)))
        probability_violation = float(max(
            np.max(np.maximum(-feasibility, 0.0)),
            np.max(np.maximum(feasibility - 1.0, 0.0)),
            np.max(np.maximum(-self.coalition_support, 0.0)),
            np.max(np.maximum(self.coalition_support - 1.0, 0.0)),
        ))
        feedback_shift = float(max(
            np.max(np.abs(procurement - 1.0)),
            np.max(np.abs(alliance_multiplier - 1.0)),
            np.max(np.abs(execution - 1.0)),
            np.max(np.abs(support_shift)),
        ))
        state_jump = float(max(
            np.max(np.abs(self.coalition_support - prior_coalition)),
            np.max(np.abs(self.bureaucratic_efficiency - prior_bureaucracy)),
        ))
        self.months += 1
        self.max_budget_simplex_residual = max(self.max_budget_simplex_residual, simplex)
        self.max_probability_bound_violation = max(self.max_probability_bound_violation, probability_violation)
        self.max_feedback_shift = max(self.max_feedback_shift, feedback_shift)
        self.max_state_jump = max(self.max_state_jump, state_jump)
        self.max_fiscal_drag = max(self.max_fiscal_drag, float(np.max(fiscal_drag)))
        self.min_execution = min(self.min_execution, float(np.min(execution)))

        return PublicChoiceFactors(
            budget_shares=budget_shares,
            policy_feasibility=feasibility,
            bureaucratic_execution=execution,
            procurement_factor=procurement,
            military_order_factor=military_order,
            civilian_order_factor=civilian_order,
            public_service_factor=public_service,
            alliance_multiplier=alliance_multiplier,
            aid_authorization=aid_authorization,
            information_quality=information_quality,
            support_shift=support_shift,
            fiscal_drag=fiscal_drag,
            rent_loss=rent_loss,
            common_pool_pressure=common_pool,
            policy_block_event=blocked,
        )

    def diagnostics(self) -> dict:
        return {
            "months": self.months,
            "enabled": dict(self.features),
            "maximum_budget_simplex_residual": self.max_budget_simplex_residual,
            "maximum_probability_bound_violation": self.max_probability_bound_violation,
            "maximum_feedback_shift": self.max_feedback_shift,
            "maximum_state_jump": self.max_state_jump,
            "maximum_fiscal_drag": self.max_fiscal_drag,
            "minimum_bureaucratic_execution": self.min_execution,
            "policy_block_event_counts": self.policy_block_counts.tolist(),
            "coalition_support_mean_by_actor": np.mean(self.coalition_support, axis=0).tolist(),
            "bureaucratic_efficiency_mean_by_actor": np.mean(
                self.bureaucratic_efficiency, axis=0
            ).tolist(),
            "reputation_mean_by_actor": np.mean(self.reputation, axis=0).tolist(),
            "future_information_used": self.future_information_used,
            "terminal_probability_adjustment": self.terminal_probability_adjustment,
            "structural_state_duplication": self.structural_state_duplication,
        }
