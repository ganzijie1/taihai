from __future__ import annotations

from dataclasses import replace

import numpy as np

from rolling_differential_game_v47 import (
    GameFactors,
    RollingHorizonDifferentialGame,
)
from sectoral_economy_v62 import CoupledSectoralEconomicSystemV62


RECOVERY_CONTROL = np.array([0.08, 0.30, 0.27, 0.27, 0.08])


class TinbergenAlgebraicRollingGameV70(RollingHorizonDifferentialGame):
    """Reachability-gated rolling game with a polynomial safety filter.

    The layer changes controls before the campaign transition consumes them. It
    never edits terminal events or outcome probabilities.
    """

    def __init__(self, paths: int, seed: int = 7001, *,
                 reachability_enabled: bool = True,
                 algebraic_filter_enabled: bool = True,
                 tropical_enabled: bool = True):
        super().__init__(paths, seed)
        self.reachability_enabled = reachability_enabled
        self.algebraic_filter_enabled = algebraic_filter_enabled
        self.tropical_enabled = tropical_enabled
        self.feature_gram = np.zeros((7, 7))
        self.months = 0
        self.reachability_gate_months = 0
        self.safety_filter_months = 0
        self.uncertifiable_path_months = 0
        self.multistable_path_months = 0
        self.max_unreachable_residual = 0.0
        self.max_condition_number = 0.0
        self.minimum_rank = 5
        self.minimum_barrier = np.inf
        self.maximum_barrier_violation = 0.0
        self.maximum_simplex_residual = 0.0
        self.mean_gate_accumulator = 0.0
        self.mean_adjustment_cost_accumulator = 0.0
        self.last_identification_rank = 0
        self.last_identification_condition = np.inf
        self.last_multistable_share = 0.0
        self.last_tropical_bottleneck = np.ones((paths, 4))
        self.tropical_attribution = np.full((paths, 4, 4), 0.25)
        self.last_bottleneck_type = np.full((paths, 4), -1, dtype=np.int8)
        self.tropical_switches = 0
        self.tropical_near_surface_path_months = 0
        self.tropical_observations = 0
        self.tropical_type_counts = np.zeros(4, dtype=np.int64)
        self.max_tropical_min_residual = 0.0
        self.mean_tropical_gap_sum = 0.0
        self.mean_parallel_relief_sum = 0.0

        # Scenario priors: positive and negative reallocations need not have
        # equal organizational, training, contracting, and mothballing costs.
        self.phi_up = np.array([0.18, 0.24, 0.13, 0.15])
        self.phi_down = np.array([0.31, 0.38, 0.21, 0.23])
        self.psi = np.array([0.10, 0.13, 0.08, 0.09])

    def _evaluate_candidates(self, player, candidates, controls, *args):
        value = super()._evaluate_candidates(
            player, candidates, controls, *args
        )
        delta = candidates[None, :, :] - controls[:, player, None, :]
        positive = np.maximum(delta, 0.0)
        negative = np.maximum(-delta, 0.0)
        task_weight = np.array([1.10, 1.00, 1.25, 1.20, 0.75])[None, None, :]
        cost = (
            0.5 * self.phi_up[player] * np.sum(task_weight * positive ** 2, axis=2)
            + 0.5 * self.phi_down[player] * np.sum(task_weight * negative ** 2, axis=2)
            + self.psi[player] / 3.0 * np.sum(np.abs(delta) ** 3, axis=2)
        )
        self.mean_adjustment_cost_accumulator += float(np.mean(cost))
        return value - 0.12 * cost

    @staticmethod
    def _response_matrix(command: np.ndarray, economy: np.ndarray,
                         intervention: np.ndarray) -> np.ndarray:
        shape = command.shape + (4, 5)
        matrix = np.zeros(shape)
        threat = np.clip(np.mean(intervention, axis=1), 0.0, 1.0)[:, None]
        matrix[..., 0, 0] = 0.27 * command
        matrix[..., 0, 1] = 0.10 * threat
        matrix[..., 0, 4] = -0.06
        matrix[..., 1, 0] = -0.21
        matrix[..., 1, 1] = -0.108
        matrix[..., 1, 2] = 0.24 * economy
        matrix[..., 2, 1] = 0.18 * threat
        matrix[..., 2, 3] = 0.132 * economy
        matrix[..., 2, 4] = 0.03
        matrix[..., 3, 0] = -0.12
        matrix[..., 3, 1] = -0.072
        matrix[..., 3, 2] = 0.06 * economy
        matrix[..., 3, 3] = 0.03 * economy
        matrix[..., 3, 4] = 0.02
        return matrix

    def _identification_confidence(self, state: np.ndarray) -> float:
        flat = state.reshape(-1, 4)
        features = np.column_stack((
            np.ones(len(flat)), flat,
            flat[:, 0] * flat[:, 1], flat[:, 2] * flat[:, 3],
        ))
        self.feature_gram += features.T @ features
        eigenvalues = np.linalg.eigvalsh(self.feature_gram)
        largest = max(float(eigenvalues[-1]), 1.0e-12)
        rank = int(np.sum(eigenvalues > largest * 1.0e-9))
        positive = eigenvalues[eigenvalues > largest * 1.0e-9]
        condition = largest / max(float(positive[0]), 1.0e-12)
        self.last_identification_rank = rank
        self.last_identification_condition = condition
        rank_score = rank / len(eigenvalues)
        condition_score = 1.0 / (1.0 + condition / 1.0e8)
        return float(np.clip(0.35 + 0.65 * rank_score * condition_score, 0.35, 1.0))

    @staticmethod
    def _barrier(state: np.ndarray) -> np.ndarray:
        scale = np.array([0.82, 0.78, 0.82, 0.80])
        return 1.0 - np.sum(0.25 * (state / scale) ** 2, axis=2)

    def update(self, *, progress: np.ndarray, stocks: np.ndarray,
               damage: np.ndarray, economic_capacity: np.ndarray,
               support: np.ndarray, command: np.ndarray,
               intervention: np.ndarray, **kwargs) -> GameFactors:
        opening = self.controls.copy()
        result = super().update(
            progress=progress, stocks=stocks, damage=damage,
            economic_capacity=economic_capacity, support=support,
            command=command, intervention=intervention, **kwargs,
        )
        selected = result.controls.copy()
        shortage = np.clip(1.0 - stocks, 0.0, 1.0)
        state = np.stack((
            np.clip(damage, 0.0, 1.0), shortage,
            np.clip(1.0 - economic_capacity, 0.0, 1.0),
            np.clip(1.0 - command, 0.0, 1.0),
        ), axis=2)

        matrix = self._response_matrix(command, economic_capacity, intervention)
        singular = np.linalg.svd(matrix, compute_uv=False)
        rank = np.sum(singular > np.maximum(singular[..., :1], 1.0e-12) * 1.0e-7, axis=2)
        condition = singular[..., 0] / np.maximum(singular[..., -1], 1.0e-10)
        self.minimum_rank = min(self.minimum_rank, int(np.min(rank)))
        self.max_condition_number = max(
            self.max_condition_number, float(np.max(condition))
        )

        target = 0.05 * np.stack((
            np.clip(0.10 - progress, -0.20, 0.20),
            np.clip(0.68 - stocks, -0.25, 0.25),
            np.clip(damage - 0.24, -0.20, 0.30),
            np.clip(0.72 - economic_capacity, -0.25, 0.25),
        ), axis=2)
        normal = np.swapaxes(matrix, -1, -2) @ matrix
        normal += 2.0e-3 * np.eye(5)[None, None, :, :]
        rhs = np.einsum("...ji,...j->...i", matrix, target)
        reachable_move = np.linalg.solve(normal, rhs)
        residual = np.linalg.norm(
            np.einsum("...ij,...j->...i", matrix, reachable_move) - target,
            axis=2,
        )
        self.max_unreachable_residual = max(
            self.max_unreachable_residual, float(np.max(residual))
        )
        confidence = self._identification_confidence(state)
        rank_gate = np.clip(rank / 4.0, 0.0, 1.0)
        residual_gate = np.exp(-8.0 * residual)
        gate = np.clip(confidence * rank_gate * residual_gate, 0.20, 1.0)
        if self.reachability_enabled:
            proposal = (
                opening
                + gate[..., None] * (0.78 * (selected - opening)
                                      + 0.22 * reachable_move)
            )
            proposal = np.maximum(proposal, 1.0e-8)
            proposal /= proposal.sum(axis=2, keepdims=True)
            selected = proposal
            self.reachability_gate_months += 1
        self.mean_gate_accumulator += float(np.mean(gate))

        resources = np.stack((stocks, economic_capacity, command, support), axis=2)
        bottleneck_type = np.argmin(resources, axis=2)
        bottleneck = np.min(resources, axis=2)
        ordered = np.sort(resources, axis=2)
        tropical_gap = ordered[:, :, 1] - ordered[:, :, 0]
        near_surface = tropical_gap < 0.025
        current_attribution = np.eye(4)[bottleneck_type]
        self.tropical_attribution = np.where(
            near_surface[..., None],
            0.65 * self.tropical_attribution + 0.35 * current_attribution,
            current_attribution,
        )
        switched = (
            (self.last_bottleneck_type >= 0)
            & (bottleneck_type != self.last_bottleneck_type)
        )
        self.tropical_switches += int(np.sum(switched))
        self.tropical_near_surface_path_months += int(np.sum(near_surface))
        self.tropical_observations += bottleneck.size
        self.tropical_type_counts += np.bincount(
            bottleneck_type.ravel(), minlength=4
        )
        self.last_bottleneck_type = bottleneck_type.copy()
        self.max_tropical_min_residual = max(
            self.max_tropical_min_residual,
            float(np.max(np.abs(
                bottleneck
                - np.take_along_axis(resources, bottleneck_type[..., None], axis=2)[..., 0]
            ))),
        )
        self.mean_tropical_gap_sum += float(np.mean(tropical_gap))

        # Serial feasibility is min-plus; alternative recovery channels are
        # parallel and therefore max-plus. The latter can ease, but cannot
        # erase, the serial bottleneck.
        parallel_relief = np.max(np.stack((
            0.72 * selected[:, :, 2],
            0.78 * selected[:, :, 3],
            0.42 * selected[:, :, 4],
        ), axis=2), axis=2)
        self.mean_parallel_relief_sum += float(np.mean(parallel_relief))
        effective_bottleneck = np.clip(
            bottleneck + 0.08 * parallel_relief, 0.0, 1.0
        )
        self.last_tropical_bottleneck = effective_bottleneck.copy()
        if self.tropical_enabled:
            bottleneck_allocations = np.array([
                [0.12, 0.18, 0.38, 0.24, 0.08],
                [0.10, 0.16, 0.36, 0.24, 0.14],
                [0.10, 0.28, 0.18, 0.36, 0.08],
                [0.08, 0.24, 0.16, 0.16, 0.36],
            ])
            tropical_allocation = np.einsum(
                "...r,rc->...c", self.tropical_attribution,
                bottleneck_allocations,
            )
            tropical_intensity = np.clip(
                0.28 * (0.42 - effective_bottleneck) / 0.42,
                0.0, 0.28,
            )
            selected = (
                (1.0 - tropical_intensity[..., None]) * selected
                + tropical_intensity[..., None] * tropical_allocation
            )
            strike_cap = 0.12 + 0.34 * effective_bottleneck
            excess = np.maximum(selected[:, :, 0] - strike_cap, 0.0)
            selected[:, :, 0] -= excess
            selected[:, :, 1] += 0.38 * excess
            selected[:, :, 2] += 0.34 * excess
            selected[:, :, 3] += 0.28 * excess

        # Cusp discriminant diagnoses two locally stable branches. In that
        # region the state follows its previous branch instead of switching
        # instantaneously to a newly computed allocation.
        stress = np.mean(state, axis=2)
        cusp_a = -0.34 + 0.18 * stress
        cusp_b = 0.10 * (support - 0.5) + 0.06 * progress
        discriminant = -4.0 * cusp_a ** 3 - 27.0 * cusp_b ** 2
        multistable = discriminant > 0.0
        self.last_multistable_share = float(np.mean(multistable))
        self.multistable_path_months += int(np.sum(multistable))
        selected = np.where(
            multistable[..., None],
            0.84 * selected + 0.16 * opening,
            selected,
        )

        if self.algebraic_filter_enabled:
            opponent_strike = (
                np.sum(selected[:, :, 0], axis=1, keepdims=True)
                - selected[:, :, 0]
            ) / 3.0

            def predicted_barrier(control: np.ndarray) -> np.ndarray:
                next_state = state.copy()
                next_state[:, :, 0] = np.clip(
                    damage + 0.030 * opponent_strike * (1.0 - control[:, :, 1])
                    - 0.022 * control[:, :, 3] * economic_capacity,
                    0.0, 1.0,
                )
                next_stock = np.clip(
                    stocks + 0.040 * control[:, :, 2] * economic_capacity
                    - 0.035 * control[:, :, 0] - 0.018 * control[:, :, 1],
                    0.0, 1.4,
                )
                next_state[:, :, 1] = np.clip(1.0 - next_stock, 0.0, 1.0)
                next_state[:, :, 2] = np.clip(
                    1.0 - np.clip(
                        economic_capacity - 0.020 * control[:, :, 0]
                        - 0.012 * control[:, :, 1] - 0.025 * next_state[:, :, 0]
                        + 0.010 * control[:, :, 2], 0.05, 1.2,
                    ), 0.0, 1.0,
                )
                next_state[:, :, 3] = np.clip(
                    state[:, :, 3] + 0.018 * next_state[:, :, 0]
                    - 0.020 * control[:, :, 3], 0.0, 1.0,
                )
                return self._barrier(next_state)

            selected_barrier = predicted_barrier(selected)
            recovery = np.broadcast_to(RECOVERY_CONTROL, selected.shape).copy()
            recovery_barrier = predicted_barrier(recovery)
            unsafe = selected_barrier < 0.0
            certifiable = recovery_barrier >= 0.0
            unsafe_certifiable = unsafe & certifiable
            self.uncertifiable_path_months += int(np.sum(unsafe & ~certifiable))
            denominator = np.maximum(recovery_barrier - selected_barrier, 1.0e-12)
            selected_weight = np.clip(
                (recovery_barrier - 0.0) / denominator, 0.0, 1.0
            )
            selected = np.where(
                unsafe_certifiable[..., None],
                selected_weight[..., None] * selected
                + (1.0 - selected_weight[..., None]) * recovery,
                selected,
            )
            final_barrier = predicted_barrier(selected)
            self.minimum_barrier = min(
                self.minimum_barrier, float(np.min(final_barrier))
            )
            self.maximum_barrier_violation = max(
                self.maximum_barrier_violation,
                float(np.max(np.where(
                    certifiable, np.maximum(-final_barrier, 0.0), 0.0
                ))),
            )
            if np.any(unsafe_certifiable):
                self.safety_filter_months += 1

        selected = np.maximum(selected, 1.0e-10)
        selected /= selected.sum(axis=2, keepdims=True)
        self.controls = selected
        self.maximum_simplex_residual = max(
            self.maximum_simplex_residual,
            float(np.max(np.abs(selected.sum(axis=2) - 1.0))),
        )
        self.months += 1
        return replace(result, controls=selected)

    def diagnostics(self) -> dict:
        result = super().diagnostics()
        result["tinbergen_algebraic_v70"] = {
            "months": self.months,
            "reachability_enabled": self.reachability_enabled,
            "algebraic_filter_enabled": self.algebraic_filter_enabled,
            "tropical_enabled": self.tropical_enabled,
            "reachability_gate_months": self.reachability_gate_months,
            "safety_filter_months": self.safety_filter_months,
            "uncertifiable_path_months": self.uncertifiable_path_months,
            "minimum_local_response_rank": self.minimum_rank,
            "maximum_local_condition_number": self.max_condition_number,
            "maximum_unreachable_target_residual": self.max_unreachable_residual,
            "identification_rank": self.last_identification_rank,
            "identification_condition_number": self.last_identification_condition,
            "multistable_path_months": self.multistable_path_months,
            "last_multistable_share": self.last_multistable_share,
            "minimum_polynomial_barrier": self.minimum_barrier,
            "maximum_barrier_violation": self.maximum_barrier_violation,
            "maximum_simplex_residual": self.maximum_simplex_residual,
            "mean_reachability_gate": self.mean_gate_accumulator / max(self.months, 1),
            "mean_candidate_adjustment_cost": (
                self.mean_adjustment_cost_accumulator
                / max(self.candidate_batch_calls, 1)
            ),
            "mean_tropical_bottleneck": float(np.mean(self.last_tropical_bottleneck)),
            "tropical_bottleneck_types": [
                "stock", "economic_capacity", "command", "support"
            ],
            "tropical_type_shares": (
                self.tropical_type_counts
                / max(self.tropical_observations, 1)
            ).tolist(),
            "tropical_switches": self.tropical_switches,
            "tropical_near_switch_surface_share": (
                self.tropical_near_surface_path_months
                / max(self.tropical_observations, 1)
            ),
            "mean_tropical_first_second_gap": (
                self.mean_tropical_gap_sum / max(self.months, 1)
            ),
            "mean_max_plus_parallel_relief": (
                self.mean_parallel_relief_sum / max(self.months, 1)
            ),
            "maximum_tropical_min_identity_residual": self.max_tropical_min_residual,
            "terminal_probability_adjustment": False,
        }
        return result


class AsymmetricAdjustmentMacroV70(CoupledSectoralEconomicSystemV62):
    """Four-party asymmetric real adjustment costs inside macro clearing."""

    def __init__(self, *args, adjustment_enabled: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.adjustment_enabled = adjustment_enabled
        self.adjustment_state = np.ones((self.paths, 4))
        self.phi_expand = np.array([0.74, 0.92, 0.58, 0.66])[None, :]
        self.phi_contract = np.array([1.05, 1.18, 0.78, 0.86])[None, :]
        self.cubic_cost = np.array([0.30, 0.34, 0.22, 0.25])[None, :]
        self.expand_speed = np.array([0.085, 0.072, 0.105, 0.095])[None, :]
        self.contract_speed = np.array([0.125, 0.140, 0.115, 0.120])[None, :]
        self.months = 0
        self.max_cost = 0.0
        self.mean_cost_sum = 0.0
        self.max_input_wedge = 0.0
        self.cumulative_cost = np.zeros((self.paths, 4))

    def update(self, **kwargs):
        desired = np.clip(
            0.34 * kwargs["actor_logistics"]
            + 0.26 * kwargs["actor_labor"]
            + 0.20 * kwargs["actor_payment"]
            + 0.20 * kwargs["actor_fiscal"],
            0.05, 1.15,
        )
        gap = desired - self.adjustment_state
        speed = np.where(gap >= 0.0, self.expand_speed, self.contract_speed)
        move = speed * gap
        positive = np.maximum(move, 0.0)
        negative = np.maximum(-move, 0.0)
        cost = (
            0.5 * self.phi_expand * positive ** 2
            + 0.5 * self.phi_contract * negative ** 2
            + self.cubic_cost / 3.0 * np.abs(move) ** 3
        )
        if self.adjustment_enabled:
            wedge = np.clip(np.exp(-18.0 * cost), 0.86, 1.0)
            adjusted = dict(kwargs)
            for name in ("actor_logistics", "actor_labor", "actor_payment", "actor_fiscal"):
                adjusted[name] = np.clip(kwargs[name] * wedge, 0.02, 1.15)
            adjusted["actor_financial_stress"] = np.clip(
                kwargs["actor_financial_stress"] + 3.5 * cost, 0.0, 1.0
            )
        else:
            wedge = np.ones_like(cost)
            adjusted = kwargs
        factors = super().update(**adjusted)
        self.adjustment_state = np.clip(
            self.adjustment_state + move, 0.05, 1.15
        )
        self.cumulative_cost += cost
        self.max_cost = max(self.max_cost, float(np.max(cost)))
        self.mean_cost_sum += float(np.mean(cost))
        self.max_input_wedge = max(
            self.max_input_wedge, float(np.max(1.0 - wedge))
        )
        self.months += 1
        if self.adjustment_enabled:
            self.current_factors = replace(
                factors,
                inflation=np.clip(factors.inflation + 1.8 * cost, -0.10, 1.75),
            )
            return self.current_factors
        return factors

    def diagnostics(self) -> dict:
        result = super().diagnostics()
        result["tinbergen_asymmetric_adjustment_v70"] = {
            "months": self.months,
            "enabled": self.adjustment_enabled,
            "mean_monthly_cost": self.mean_cost_sum / max(self.months, 1),
            "maximum_monthly_cost": self.max_cost,
            "maximum_real_input_wedge": self.max_input_wedge,
            "mean_cumulative_cost_by_actor": np.mean(
                self.cumulative_cost, axis=0
            ).tolist(),
            "expansion_coefficients": self.phi_expand[0].tolist(),
            "contraction_coefficients": self.phi_contract[0].tolist(),
            "identification": "bounded scenario priors; ablation required",
            "terminal_probability_adjustment": False,
        }
        return result
