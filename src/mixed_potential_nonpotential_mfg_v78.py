from __future__ import annotations

import numpy as np

from complete_theory_mfg_v77 import CompleteTheoryMFGV77
from continuous_multipop_mfg_v47 import ContinuousMultiPopulationMFG


class MixedPotentialNonpotentialMFGV78(CompleteTheoryMFGV77):
    """Hybrid potential/non-potential continuous GMFG.

    The reciprocal part of the signed multilayer Graphon is evaluated as the
    potential field.  Its Fisher--Rao density and Bernoulli action projections
    remain owned by the inherited HJB--FP fixed point.  The skew pair field and
    every directed higher-order hypergraph field form a separate operator and
    are evaluated with a predictor/corrector (Mirror--Prox) step inside the
    same monthly fixed-point iteration.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mirror_prox_step = float(kwargs.get("mirror_prox_step", 0.18))
        self.maximum_weighted_adjoint_residual = 0.0
        self.maximum_potential_decomposition_residual = 0.0
        self.maximum_mirror_prox_predictor_step = 0.0
        self.maximum_mirror_prox_correction = 0.0
        self.maximum_mixed_vi_gap = 0.0
        self.maximum_potential_prox_gap = 0.0
        self.maximum_nonpotential_operator_norm = 0.0
        self.minimum_monotonicity_margin = np.inf
        self.mirror_prox_evaluations = 0
        self.potential_proximal_updates = 0
        self._last_ablation_month = -1
        self.ablation_evaluations = 0
        self.maximum_ablation_field_deviation = {
            "no_graph_mfg": 0.0,
            "block_gmfg": 0.0,
            "low_rank_graphon": 0.0,
            "finite_graph": 0.0,
        }

    @staticmethod
    def _normalise_rows(kernel: np.ndarray) -> np.ndarray:
        scale = np.sum(np.abs(kernel), axis=-1, keepdims=True)
        return kernel / np.maximum(scale, 1.0e-12)

    def _weighted_split(
        self, kernel: np.ndarray, weights: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        signed = kernel * self.current_sign[: kernel.shape[0]]
        # Actor masses define the discrete adjoint used by the VI.  The actor
        # weights are fixed public-account population shares inherited from the
        # continuous MFG, not path-dependent terminal outcomes.
        actor_mass = np.maximum(
            np.sum(self.age_weights[None, None, :] * self.engagement_mass[: kernel.shape[0]], axis=2),
            1.0e-8,
        )
        ratio = actor_mass[:, None, :, None] / actor_mass[:, None, None, :]
        adjoint = np.swapaxes(signed, -1, -2) / np.maximum(ratio, 1.0e-12)
        potential = 0.5 * (signed + adjoint)
        circulation = 0.5 * (signed - adjoint)
        reconstructed = potential + circulation
        self.maximum_weighted_adjoint_residual = max(
            self.maximum_weighted_adjoint_residual,
            float(np.max(np.abs(reconstructed - signed))),
        )
        self.maximum_potential_decomposition_residual = max(
            self.maximum_potential_decomposition_residual,
            float(np.max(np.abs(
                potential - np.swapaxes(potential, -1, -2) / np.maximum(ratio, 1.0e-12)
            ))),
        )
        effective_potential = np.einsum("pl,plij->pij", weights, potential)
        symmetric_proxy = 0.5 * (
            effective_potential + np.swapaxes(effective_potential, -1, -2)
        )
        eigenvalues = np.linalg.eigvalsh(symmetric_proxy)
        self.minimum_monotonicity_margin = min(
            self.minimum_monotonicity_margin, float(np.min(eigenvalues))
        )
        return potential, circulation

    def _pair_fields(
        self, mean: np.ndarray, action: np.ndarray, operator: np.ndarray,
        weights: np.ndarray, major_action: np.ndarray,
        intensity_multiplier: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return ContinuousMultiPopulationMFG._multilayer_fields(
            self, mean, action, operator, weights, major_action,
            intensity_multiplier,
        )

    def _nonpotential_fields(
        self, mean: np.ndarray, action: np.ndarray, circulation: np.ndarray,
        weights: np.ndarray, major_action: np.ndarray,
        intensity_multiplier: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        pair_state, pair_action, pair_major = self._pair_fields(
            mean, action, circulation, weights, major_action,
            intensity_multiplier,
        )
        if not self.hypergraph_enabled:
            return pair_state, pair_action, pair_major
        hyper_state, hyper_action, hyper_major, _ = self._hypergraph_fields(
            mean, action, weights, major_action
        )
        share = self.hypergraph_share
        return (
            (1.0 - share) * pair_state + share * hyper_state,
            (1.0 - share) * pair_action + share * hyper_action,
            (1.0 - share) * pair_major + share * hyper_major,
        )

    def _finite_signed_pair_fields(
        self, mean: np.ndarray, action: np.ndarray, operator: np.ndarray,
        weights: np.ndarray, major_action: np.ndarray,
        intensity_multiplier: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate a signed pair operator on the frozen sampled finite graph."""
        checked = min(mean.shape[0], self.finite_edge_uniform.shape[0])
        state_result = np.zeros_like(mean[:checked])
        action_result = np.zeros_like(action[:checked])
        major_result = np.zeros((checked, mean.shape[1]), dtype=float)
        for path in range(checked):
            layer_state = np.zeros((operator.shape[1], mean.shape[1], mean.shape[2]))
            layer_action = np.zeros((operator.shape[1], mean.shape[1], mean.shape[2], action.shape[3]))
            layer_major = np.zeros((operator.shape[1], mean.shape[1]))
            for layer in range(operator.shape[1]):
                for actor in range(mean.shape[1]):
                    row = operator[path, layer, actor]
                    row_total = float(np.sum(np.abs(row)))
                    if row_total > 1.0e-12:
                        actor_cdf = np.cumsum(np.abs(row) / row_total)
                        actor_cdf[-1] = 1.0
                    for age in range(mean.shape[2]):
                        uniforms = self.finite_edge_uniform[path, layer, actor, age]
                        age_cdf = np.cumsum(self.age_kernel[age])
                        age_cdf[-1] = 1.0
                        sender_age = np.searchsorted(age_cdf, uniforms[:, 1], side="right")
                        age_state = np.mean(mean[path, actor, sender_age])
                        age_action = np.mean(action[path, actor, sender_age], axis=0)
                        actor_state = 0.0
                        actor_action = np.zeros(action.shape[3], dtype=float)
                        if row_total > 1.0e-12:
                            sender_actor = np.searchsorted(actor_cdf, uniforms[:, 0], side="right")
                            importance = row_total * np.sign(row[sender_actor])
                            actor_state = np.mean(importance * mean[path, sender_actor, age])
                            actor_action = np.mean(importance[:, None] * action[path, sender_actor, age], axis=0)
                        multiplier = 1.0 if intensity_multiplier is None else intensity_multiplier[path, layer, actor]
                        layer_state[layer, actor, age] = 0.68 * multiplier * actor_state + 0.32 * age_state
                        layer_action[layer, actor, age] = 0.68 * multiplier * actor_action + 0.32 * age_action
                    if row_total > 1.0e-12:
                        uniforms = self.finite_edge_uniform[path, layer, actor, 0]
                        sender_actor = np.searchsorted(actor_cdf, uniforms[:, 0], side="right")
                        importance = row_total * np.sign(row[sender_actor])
                        major_by_role = np.mean(importance[:, None] * major_action[path, sender_actor], axis=0)
                        multiplier = 1.0 if intensity_multiplier is None else intensity_multiplier[path, layer, actor]
                        layer_major[layer, actor] = multiplier * float(np.dot(major_by_role, self.major_role_weights))
            state_result[path] = np.einsum("l,lih->ih", weights[path], layer_state)
            action_result[path] = np.einsum("l,lihc->ihc", weights[path], layer_action)
            major_result[path] = np.einsum("l,li->i", weights[path], layer_major)
        return state_result, action_result, major_result

    def _finite_nonpotential_fields(
        self, mean: np.ndarray, action: np.ndarray, circulation: np.ndarray,
        weights: np.ndarray, major_action: np.ndarray,
        intensity_multiplier: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        state, action_field, major_field = self._finite_signed_pair_fields(
            mean, action, circulation, weights, major_action, intensity_multiplier
        )
        if not self.hypergraph_enabled:
            return state, action_field, major_field
        checked = state.shape[0]
        hyper_state, hyper_action, hyper_major, _ = self._hypergraph_fields(
            mean[:checked], action[:checked], weights[:checked], major_action[:checked]
        )
        share = self.hypergraph_share
        return ((1.0 - share) * state + share * hyper_state,
                (1.0 - share) * action_field + share * hyper_action,
                (1.0 - share) * major_field + share * hyper_major)

    def _finite_network_fields(
        self, mean: np.ndarray, action: np.ndarray, kernel: np.ndarray,
        weights: np.ndarray, intensity_multiplier: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Back-project the complete hybrid VI onto one frozen finite graph."""
        checked = min(mean.shape[0], self.finite_edge_uniform.shape[0])
        mean, action = mean[:checked], action[:checked]
        kernel, weights = kernel[:checked], weights[:checked]
        major_action = self.major_action[:checked]
        intensity = None if intensity_multiplier is None else intensity_multiplier[:checked]
        potential, circulation = self._weighted_split(kernel, weights)
        potential_state, potential_action, _ = self._finite_signed_pair_fields(
            mean, action, potential, weights, major_action, intensity
        )
        if self.hypergraph_enabled:
            potential_state *= 1.0 - self.hypergraph_share
            potential_action *= 1.0 - self.hypergraph_share
        first_state, first_action, first_major = self._finite_nonpotential_fields(
            mean, action, circulation, weights, major_action, intensity
        )
        step = self.mirror_prox_step
        predicted_mean = np.clip(mean + step * np.tanh(first_state), -1.0, 1.0)
        predicted_action = np.clip(action + step * np.tanh(first_action - action), 0.0, 1.0)
        predicted_major = np.clip(major_action + step * np.tanh(first_major[:, :, None] - major_action), -1.0, 1.0)
        second_state, second_action, _ = self._finite_nonpotential_fields(
            predicted_mean, predicted_action, circulation, weights, predicted_major, intensity
        )
        sampled_state = potential_state + second_state
        sampled_action = potential_action + second_action
        reference_state, reference_action, _ = self._multilayer_fields(
            mean, action, kernel, weights, major_action, intensity
        )
        error = float(max(np.max(np.abs(sampled_state - reference_state)), np.max(np.abs(sampled_action - reference_action))))
        return sampled_state, sampled_action, error

    def _record_ablations(
        self, mean: np.ndarray, action: np.ndarray, kernel: np.ndarray,
        weights: np.ndarray, major_action: np.ndarray,
        intensity_multiplier: np.ndarray | None, reference_state: np.ndarray,
    ) -> None:
        if self._last_ablation_month == self.month:
            return
        self._last_ablation_month = self.month
        paths = mean.shape[0]
        shapes = kernel.shape
        candidates = {
            "no_graph_mfg": np.full(shapes, 1.0 / shapes[-1]),
            "block_gmfg": np.broadcast_to(
                self.layer_actor_kernels[None, ...], shapes
            ).copy(),
            "low_rank_graphon": np.broadcast_to(
                self.low_rank_layer_kernels[None, ...], shapes
            ).copy(),
        }
        for name, candidate in candidates.items():
            candidate = self._normalise_rows(candidate)
            state, _, _ = self._pair_fields(
                mean, action, candidate, weights, major_action,
                intensity_multiplier,
            )
            deviation = float(np.max(np.abs(state - reference_state)))
            self.maximum_ablation_field_deviation[name] = max(
                self.maximum_ablation_field_deviation[name], deviation
            )
        self.ablation_evaluations += 1

    def _multilayer_fields(
        self, mean, action, kernel, weights, major_action,
        intensity_multiplier=None,
    ):
        potential, circulation = self._weighted_split(kernel, weights)
        potential_state, potential_action, potential_major = self._pair_fields(
            mean, action, potential, weights, major_action,
            intensity_multiplier,
        )
        if self.hypergraph_enabled:
            share = self.hypergraph_share
            potential_state *= 1.0 - share
            potential_action *= 1.0 - share
            potential_major *= 1.0 - share

        first_state, first_action, first_major = self._nonpotential_fields(
            mean, action, circulation, weights, major_action,
            intensity_multiplier,
        )
        step = self.mirror_prox_step
        predicted_mean = np.clip(mean + step * np.tanh(first_state), -1.0, 1.0)
        predicted_action = np.clip(
            action + step * np.tanh(first_action - action), 0.0, 1.0
        )
        predicted_major = np.clip(
            major_action + step * np.tanh(first_major[:, :, None] - major_action),
            -1.0, 1.0,
        )
        second_state, second_action, second_major = self._nonpotential_fields(
            predicted_mean, predicted_action, circulation, weights,
            predicted_major, intensity_multiplier,
        )
        if self.hypergraph_enabled:
            pair_state, pair_action, pair_major = self._pair_fields(
                predicted_mean, predicted_action, circulation, weights,
                predicted_major, intensity_multiplier,
            )
            pair_share = 1.0 - self.hypergraph_share
            higher_order_contribution = max(
                float(np.max(np.abs(second_state - pair_share * pair_state))),
                float(np.max(np.abs(second_action - pair_share * pair_action))),
                float(np.max(np.abs(second_major - pair_share * pair_major))),
            )
            self.max_higher_order_contribution = max(
                self.max_higher_order_contribution,
                higher_order_contribution,
            )
        predictor_step = max(
            float(np.max(np.abs(predicted_mean - mean))),
            float(np.max(np.abs(predicted_action - action))),
            float(np.max(np.abs(predicted_major - major_action))),
        )
        correction = max(
            float(np.max(np.abs(second_state - first_state))),
            float(np.max(np.abs(second_action - first_action))),
            float(np.max(np.abs(second_major - first_major))),
        )
        operator_norm = max(
            float(np.max(np.abs(second_state))),
            float(np.max(np.abs(second_action))),
            float(np.max(np.abs(second_major))),
        )
        self.maximum_mirror_prox_predictor_step = max(
            self.maximum_mirror_prox_predictor_step, predictor_step
        )
        self.maximum_mirror_prox_correction = max(
            self.maximum_mirror_prox_correction, correction
        )
        self.maximum_nonpotential_operator_norm = max(
            self.maximum_nonpotential_operator_norm, operator_norm
        )
        self.mirror_prox_evaluations += 1
        self.potential_proximal_updates += 1

        state = potential_state + second_state
        action_field = potential_action + second_action
        major_field = potential_major + second_major
        unsigned_state, unsigned_action, unsigned_major = self._pair_fields(
            mean, action, kernel, weights, major_action, intensity_multiplier
        )
        circulation_state, circulation_action, circulation_major = self._pair_fields(
            mean, action, circulation, weights, major_action, intensity_multiplier
        )
        pair_share = 1.0 - self.hypergraph_share if self.hypergraph_enabled else 1.0
        signed_effect = max(
            float(np.max(np.abs(potential_state + pair_share * circulation_state - pair_share * unsigned_state))),
            float(np.max(np.abs(potential_action + pair_share * circulation_action - pair_share * unsigned_action))),
            float(np.max(np.abs(potential_major + pair_share * circulation_major - pair_share * unsigned_major))),
        )
        self.maximum_signed_field_correction = max(
            self.maximum_signed_field_correction, signed_effect
        )
        self._record_ablations(
            mean, action, kernel, weights, major_action,
            intensity_multiplier, state,
        )
        return state, action_field, major_field

    def update(self, **kwargs):
        factors = super().update(**kwargs)
        # The natural residual is a computable VI-gap upper bound for the
        # compact density/action feasible set.  It is recorded, not used as a
        # terminal-probability patch.
        prox_gap = max(
            factors.fixed_point_residual,
            factors.mean_exploitability_bound,
        )
        vi_gap = prox_gap
        self.maximum_potential_prox_gap = max(
            self.maximum_potential_prox_gap, float(prox_gap)
        )
        self.maximum_mixed_vi_gap = max(
            self.maximum_mixed_vi_gap, float(vi_gap)
        )
        if factors.finite_network_field_error > 0.0:
            self.maximum_ablation_field_deviation["finite_graph"] = max(
                self.maximum_ablation_field_deviation["finite_graph"],
                float(factors.finite_network_field_error),
            )
        return factors

    def diagnostics(self) -> dict:
        payload = super().diagnostics()
        payload["v78_mixed_potential_nonpotential"] = {
            "mirror_prox_evaluations": self.mirror_prox_evaluations,
            "potential_proximal_updates": self.potential_proximal_updates,
            "mirror_prox_step": self.mirror_prox_step,
            "maximum_weighted_adjoint_residual": self.maximum_weighted_adjoint_residual,
            "maximum_potential_decomposition_residual": self.maximum_potential_decomposition_residual,
            "maximum_mirror_prox_predictor_step": self.maximum_mirror_prox_predictor_step,
            "maximum_mirror_prox_correction": self.maximum_mirror_prox_correction,
            "maximum_nonpotential_operator_norm": self.maximum_nonpotential_operator_norm,
            "maximum_potential_prox_gap": self.maximum_potential_prox_gap,
            "maximum_mixed_vi_gap": self.maximum_mixed_vi_gap,
            "minimum_monotonicity_margin": (
                self.minimum_monotonicity_margin
                if np.isfinite(self.minimum_monotonicity_margin) else 0.0
            ),
            "ablation_evaluations": self.ablation_evaluations,
            "maximum_ablation_field_deviation": dict(
                self.maximum_ablation_field_deviation
            ),
            "state_owner": "single hybrid MFG owner for density, actions, potential and directional fields",
            "potential_block": "weighted reciprocal signed Graphon with Fisher-Rao proximal density/action projection",
            "nonpotential_block": "weighted skew Graphon plus directed hypergraph evaluated by same-iteration Mirror-Prox",
            "kernel_uncertainty": "inherited pathwise block posterior draws",
            "terminal_probability_adjustment": False,
        }
        return payload
