from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np

from continuous_multipop_mfg_v47 import ACTORS, AGES, GRAPHON_LAYERS
from hypergraph_mfg_v75 import DirectedMultilayerHypergraphMFGV75
from online_tda_v76 import OnlineSocialTDAV76
from riemannian_geometry_v60 import RegularizedRiemannianAnderson


@dataclass
class SCFTracker:
    nonpotential_ratio: float = 0.0
    calls: int = 0
    trust_contractions: int = 0
    trust_expansions: int = 0
    maximum_contraction_ratio: float = 0.0
    maximum_spectral_radius_proxy: float = 0.0
    minimum_trust_ratio: float = np.inf
    maximum_trust_ratio: float = 0.0


_SCF_TRACKER = SCFTracker()


def reset_scf_tracker() -> None:
    global _SCF_TRACKER
    _SCF_TRACKER = SCFTracker()


def set_nonpotential_ratio(value: float) -> None:
    _SCF_TRACKER.nonpotential_ratio = float(np.clip(value, 0.0, 1.0))


class AdaptiveTrustRegionSCFAnderson(RegularizedRiemannianAnderson):
    """SCF-style Anderson mixing with a path-local contraction safeguard."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("trust_ratio", 2.2)
        super().__init__(*args, **kwargs)
        self._previous_residual_norm: float | None = None

    def step(self, density: np.ndarray, mapped: np.ndarray, dx: float) -> np.ndarray:
        flat = (mapped - density).reshape(density.shape[0], -1)
        residual_norm = float(np.median(np.linalg.norm(flat, axis=1)))
        contraction = 0.0
        if self._previous_residual_norm is not None:
            contraction = residual_norm / max(self._previous_residual_norm, 1.0e-15)
            _SCF_TRACKER.maximum_contraction_ratio = max(
                _SCF_TRACKER.maximum_contraction_ratio, contraction
            )
            _SCF_TRACKER.maximum_spectral_radius_proxy = max(
                _SCF_TRACKER.maximum_spectral_radius_proxy, contraction
            )
        nonpotential_cap = 2.2 / (1.0 + 1.35 * _SCF_TRACKER.nonpotential_ratio)
        if contraction > 1.02:
            self.trust_ratio = max(1.05, 0.72 * self.trust_ratio)
            self.damping = max(0.24, 0.84 * self.damping)
            _SCF_TRACKER.trust_contractions += 1
        elif 0.0 < contraction < 0.72:
            self.trust_ratio = min(nonpotential_cap, 1.08 * self.trust_ratio)
            self.damping = min(0.62, 1.04 * self.damping)
            _SCF_TRACKER.trust_expansions += 1
        self.trust_ratio = min(self.trust_ratio, nonpotential_cap)
        _SCF_TRACKER.calls += 1
        _SCF_TRACKER.minimum_trust_ratio = min(
            _SCF_TRACKER.minimum_trust_ratio, self.trust_ratio
        )
        _SCF_TRACKER.maximum_trust_ratio = max(
            _SCF_TRACKER.maximum_trust_ratio, self.trust_ratio
        )
        self._previous_residual_norm = residual_norm
        return super().step(density, mapped, dx)


class SignedPotentialHypergraphMFGV76(DirectedMultilayerHypergraphMFGV75):
    """V7.5 GMFG with signed influence and potential/circulation diagnostics.

    Influence magnitude remains nonnegative.  A separate bounded sign field
    flips the sender target for information and identity layers (opposing
    dynamics).  It never turns a convex disagreement cost into an unbounded
    negative quadratic reward.
    """

    def __init__(self, *args, **kwargs):
        reset_scf_tracker()
        super().__init__(*args, **kwargs)
        self.current_sign = np.ones(
            (self.paths, len(GRAPHON_LAYERS), len(ACTORS), len(ACTORS))
        )
        self.positive_relation = np.full_like(self.current_sign, 0.58)
        self.negative_relation = np.full_like(self.current_sign, 0.12)
        opening_mean, _ = self._moments(self.density)
        opening_actor_mean = np.sum(
            opening_mean * self.age_weights[None, None, :], axis=2
        )
        opening_distance = np.abs(
            opening_actor_mean[:, :, None] - opening_actor_mean[:, None, :]
        )
        opening_positive = 0.15 + 0.70 * self._sigmoid(
            (0.42 - opening_distance) / 0.12
        )
        opening_negative = 0.08 + 0.70 * self._sigmoid(
            (opening_distance - 0.42) / 0.12
        )
        self.positive_relation.fill(1.0)
        self.negative_relation.fill(0.0)
        for layer in (
            GRAPHON_LAYERS.index("information"),
            GRAPHON_LAYERS.index("identity"),
        ):
            self.positive_relation[:, layer] = opening_positive
            self.negative_relation[:, layer] = opening_negative
        diagonal = np.arange(len(ACTORS))
        self.positive_relation[:, :, diagonal, diagonal] = 1.0
        self.negative_relation[:, :, diagonal, diagonal] = 0.0
        self.relation_updates = 0
        self.maximum_relation_step = 0.0
        self.minimum_positive_relation = 1.0
        self.maximum_negative_relation = 0.0
        self.minimum_zero_boundary_formation_rate = np.inf
        self.minimum_bridge_index = np.inf
        self.maximum_bridge_index = 0.0
        self._spectral_vector = np.full((self.paths, len(ACTORS)), 1.0 / np.sqrt(len(ACTORS)))
        self._spectral_log_growth = np.zeros(self.paths)
        self.maximum_nonautonomous_growth_rate = -np.inf
        self.minimum_nonautonomous_growth_rate = np.inf
        self.maximum_pair_nonpotential_ratio = 0.0
        self.maximum_hypergraph_nonpotential_ratio = 0.0
        self.maximum_decomposition_residual = 0.0
        self.maximum_signed_field_correction = 0.0
        self.maximum_structural_frustration = 0.0
        self.minimum_signed_laplacian_eigenvalue = np.inf
        self.signed_graphon_updates = 0
        self.branch_evaluations = 0
        self.maximum_reduced_branch_count = 1
        self.maximum_reduced_branch_spectral_radius = 0.0
        self.reduced_bifurcation_events = 0
        self.maximum_potential_energy_spread = 0.0
        self._last_signed_kernel = None
        self._last_layer_weights = None
        self.online_tda = OnlineSocialTDAV76(self.paths, interval=3)
        self.tda_delayed_shock = np.zeros(self.paths)

    @staticmethod
    def _sigmoid(value: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(value, -40.0, 40.0)))

    def _update_relation_memory(
        self,
        actor_mean: np.ndarray,
        damage: np.ndarray,
        shortage: np.ndarray,
        network: np.ndarray,
        jump_signal: np.ndarray,
    ) -> None:
        distance = np.abs(actor_mean[:, :, None] - actor_mean[:, None, :])
        common_loss = 0.25 * (
            damage[:, :, None] + damage[:, None, :]
            + shortage[:, :, None] + shortage[:, None, :]
        )
        network_pair = np.sqrt(np.maximum(network[:, :, None] * network[:, None, :], 0.0))
        jump_pair = 0.5 * (jump_signal[:, :, None] + jump_signal[:, None, :])
        loss_gap = 0.5 * (
            np.abs(damage[:, :, None] - damage[:, None, :])
            + np.abs(shortage[:, :, None] - shortage[:, None, :])
        )
        topology_pressure = self.tda_delayed_shock[:, None, None]
        information = GRAPHON_LAYERS.index("information")
        identity = GRAPHON_LAYERS.index("identity")
        maximum_step = 0.0
        for layer, persistence in ((information, 1.0), (identity, 0.72)):
            positive = self.positive_relation[:, layer]
            negative = self.negative_relation[:, layer]
            formation_barrier = 0.22 + 0.18 * (1.0 - positive)
            fracture_barrier = 0.38 + 0.14 * positive
            formation = (
                0.008
                + persistence * (0.070 * common_loss + 0.040 * network_pair)
                * self._sigmoid((common_loss - formation_barrier) / 0.08)
                * (1.0 - 0.55 * distance)
            ) * (1.0 - 0.25 * topology_pressure)
            fracture = (
                0.012
                + persistence * (0.075 * distance + 0.045 * (1.0 - network_pair) + 0.040 * jump_pair)
                * self._sigmoid((distance - fracture_barrier) / 0.08)
            ) + 0.045 * topology_pressure
            negative_generation = (
                0.006 + persistence * (0.060 * distance + 0.035 * loss_gap + 0.030 * jump_pair)
                + 0.035 * topology_pressure
            )
            negative_decay = 0.018 + persistence * 0.055 * network_pair * (1.0 - 0.65 * distance)
            updated_positive = np.clip(
                positive + formation * (1.0 - positive) - fracture * positive,
                0.0, 1.0,
            )
            updated_negative = np.clip(
                negative + negative_generation * (1.0 - negative) - negative_decay * negative,
                0.0, 1.0,
            )
            maximum_step = max(
                maximum_step,
                float(np.max(np.abs(updated_positive - positive))),
                float(np.max(np.abs(updated_negative - negative))),
            )
            self.positive_relation[:, layer] = updated_positive
            self.negative_relation[:, layer] = updated_negative
            self.minimum_zero_boundary_formation_rate = min(
                self.minimum_zero_boundary_formation_rate, float(np.min(formation))
            )
        diagonal = np.arange(len(ACTORS))
        self.positive_relation[:, :, diagonal, diagonal] = 1.0
        self.negative_relation[:, :, diagonal, diagonal] = 0.0
        self.maximum_relation_step = max(self.maximum_relation_step, maximum_step)
        self.minimum_positive_relation = min(
            self.minimum_positive_relation, float(np.min(self.positive_relation))
        )
        self.maximum_negative_relation = max(
            self.maximum_negative_relation, float(np.max(self.negative_relation))
        )
        mask = ~np.eye(len(ACTORS), dtype=bool)
        bridge = np.mean(self.positive_relation[:, (information, identity)][:, :, mask], axis=(1, 2))
        self.minimum_bridge_index = min(self.minimum_bridge_index, float(np.min(bridge)))
        self.maximum_bridge_index = max(self.maximum_bridge_index, float(np.max(bridge)))
        self.relation_updates += 1

    def _update_nonautonomous_spectrum(self, signed_kernel: np.ndarray, weights: np.ndarray) -> None:
        operator = np.einsum("pl,plij->pij", weights, signed_kernel)
        propagated = np.einsum("pij,pj->pi", operator, self._spectral_vector)
        norm = np.linalg.norm(propagated, axis=1)
        self._spectral_log_growth += np.log(np.maximum(norm, 1.0e-12))
        self._spectral_vector = propagated / np.maximum(norm[:, None], 1.0e-12)
        rate = self._spectral_log_growth / max(self.relation_updates, 1)
        self.maximum_nonautonomous_growth_rate = max(
            self.maximum_nonautonomous_growth_rate, float(np.max(rate))
        )
        self.minimum_nonautonomous_growth_rate = min(
            self.minimum_nonautonomous_growth_rate, float(np.min(rate))
        )

    def _signed_structure(self, kernel: np.ndarray, weights: np.ndarray) -> None:
        signed_kernel = kernel * self.current_sign
        transpose = np.swapaxes(signed_kernel, -1, -2)
        potential = 0.5 * (signed_kernel + transpose)
        circulation = 0.5 * (signed_kernel - transpose)
        reconstruction = potential + circulation - signed_kernel
        self.maximum_decomposition_residual = max(
            self.maximum_decomposition_residual,
            float(np.max(np.abs(reconstruction))),
        )
        pair_norm = np.linalg.norm(signed_kernel.reshape(self.paths, -1), axis=1)
        circulation_norm = np.linalg.norm(circulation.reshape(self.paths, -1), axis=1)
        pair_ratio = circulation_norm / np.maximum(pair_norm, 1.0e-15)
        self.maximum_pair_nonpotential_ratio = max(
            self.maximum_pair_nonpotential_ratio, float(np.max(pair_ratio))
        )

        tail_mass = np.maximum(np.sum(self.hyper_tail, axis=1, keepdims=True), 1.0e-12)
        hyper_operator = self.hyper_head.T @ (self.hyper_tail / tail_mass)
        hyper_operator /= np.maximum(
            np.sum(np.abs(hyper_operator), axis=1, keepdims=True), 1.0e-12
        )
        hyper_circulation = 0.5 * (hyper_operator - hyper_operator.T)
        hyper_ratio = np.linalg.norm(hyper_circulation) / max(
            np.linalg.norm(hyper_operator), 1.0e-15
        )
        self.maximum_hypergraph_nonpotential_ratio = max(
            self.maximum_hypergraph_nonpotential_ratio, float(hyper_ratio)
        )
        combined = (
            (1.0 - self.hypergraph_share) * pair_ratio
            + self.hypergraph_share * hyper_ratio
        )
        set_nonpotential_ratio(float(np.max(combined)))
        self._last_signed_kernel = signed_kernel
        self._last_layer_weights = weights.copy()
        self._update_nonautonomous_spectrum(signed_kernel, weights)

    def _structural_balance(self) -> None:
        checked_layers = [GRAPHON_LAYERS.index("information"), GRAPHON_LAYERS.index("identity")]
        for layer in checked_layers:
            symmetric_sign = np.sign(
                self.current_sign[:, layer]
                + np.swapaxes(self.current_sign[:, layer], -1, -2)
            )
            symmetric_sign[symmetric_sign == 0.0] = 1.0
            frustrated = np.zeros(self.paths)
            triangles = list(combinations(range(len(ACTORS)), 3))
            for i, j, k in triangles:
                product = (
                    symmetric_sign[:, i, j]
                    * symmetric_sign[:, j, k]
                    * symmetric_sign[:, k, i]
                )
                frustrated += product < 0.0
            frustrated /= max(len(triangles), 1)
            self.maximum_structural_frustration = max(
                self.maximum_structural_frustration, float(np.max(frustrated))
            )
            adjacency = self._last_signed_kernel[:, layer]
            adjacency = 0.5 * (adjacency + np.swapaxes(adjacency, -1, -2))
            degree = np.sum(np.abs(adjacency), axis=2)
            laplacian = -adjacency
            index = np.arange(len(ACTORS))
            laplacian[:, index, index] += degree
            minimum = float(np.min(np.linalg.eigvalsh(laplacian)))
            self.minimum_signed_laplacian_eigenvalue = min(
                self.minimum_signed_laplacian_eigenvalue, minimum
            )

    def _effective_graphon(self, *args, **kwargs):
        kernel, weights, intensity = super()._effective_graphon(*args, **kwargs)
        damage = np.asarray(args[0])
        shortage = np.asarray(args[1])
        network = np.asarray(args[2])
        jump_signal = np.asarray(args[4])
        mean, _ = self._moments(self.density)
        actor_mean = np.sum(mean * self.age_weights[None, None, :], axis=2)
        self._update_relation_memory(
            actor_mean, damage, shortage, network, jump_signal
        )
        denominator = self.positive_relation + self.negative_relation + 1.0e-12
        endogenous_sign = (
            self.positive_relation - self.negative_relation
        ) / denominator
        diagonal = np.arange(len(ACTORS))
        endogenous_sign[:, :, diagonal, diagonal] = 1.0
        self.current_sign = np.clip(endogenous_sign, -1.0, 1.0)
        self._signed_structure(kernel, weights)
        self._structural_balance()
        self.signed_graphon_updates += 1
        return kernel, weights, intensity

    def _signed_state_correction(
        self,
        mean: np.ndarray,
        kernel: np.ndarray,
        weights: np.ndarray,
        intensity_multiplier: np.ndarray | None,
    ) -> np.ndarray:
        paths = mean.shape[0]
        sign = self.current_sign[:paths]
        correction = np.einsum(
            "plij,pjh->plih", kernel * (sign - 1.0), mean
        )
        if intensity_multiplier is not None:
            correction *= intensity_multiplier[..., None]
        correction = 0.68 * np.einsum("pl,plih->pih", weights, correction)
        return (1.0 - self.hypergraph_share) * correction

    def _multilayer_fields(
        self, mean, action, kernel, weights, major_action,
        intensity_multiplier=None,
    ):
        state, action_field, major_field = super()._multilayer_fields(
            mean, action, kernel, weights, major_action, intensity_multiplier
        )
        correction = self._signed_state_correction(
            mean, kernel, weights, intensity_multiplier
        )
        self.maximum_signed_field_correction = max(
            self.maximum_signed_field_correction, float(np.max(np.abs(correction)))
        )
        return state + correction, action_field, major_field

    def _finite_network_fields(
        self, mean, action, kernel, weights, intensity_multiplier=None,
    ):
        state, action_field, _ = super()._finite_network_fields(
            mean, action, kernel, weights, intensity_multiplier
        )
        checked = state.shape[0]
        state = state + self._signed_state_correction(
            mean[:checked], kernel[:checked], weights[:checked],
            None if intensity_multiplier is None else intensity_multiplier[:checked],
        )
        reference_state, reference_action, _ = self._multilayer_fields(
            mean[:checked], action[:checked], kernel[:checked], weights[:checked],
            self.major_action[:checked],
            None if intensity_multiplier is None else intensity_multiplier[:checked],
        )
        error = float(max(
            np.max(np.abs(state - reference_state)),
            np.max(np.abs(action_field - reference_action)),
        ))
        return state, action_field, error

    def _reduced_branch_probe(self, progress, damage, shortage) -> None:
        if self._last_signed_kernel is None:
            return
        effective = np.einsum(
            "pl,plij->pij", self._last_layer_weights, self._last_signed_kernel
        )
        potential = 0.5 * (effective + np.swapaxes(effective, -1, -2))
        circulation = 0.5 * (effective - np.swapaxes(effective, -1, -2))
        operator = 0.62 * potential + 0.30 * circulation
        bias = np.clip(0.34 * progress - 0.24 * damage - 0.20 * shortage, -0.8, 0.8)
        solutions = []
        for start in (-0.82, 0.0, 0.82):
            value = np.full_like(bias, start)
            for _ in range(80):
                mapped = np.tanh(bias + np.einsum("pij,pj->pi", operator, value))
                updated = 0.42 * value + 0.58 * mapped
                if np.max(np.abs(updated - value)) < 1.0e-9:
                    value = updated
                    break
                value = updated
            solutions.append(value)
        stack = np.stack(solutions, axis=1)
        branch_count = np.ones(self.paths, dtype=int)
        for left, right in combinations(range(stack.shape[1]), 2):
            branch_count += (
                np.max(np.abs(stack[:, left] - stack[:, right]), axis=1) > 1.0e-3
            )
        branch_count = np.minimum(branch_count, stack.shape[1])
        spectral = np.zeros(self.paths)
        energies = []
        for solution in solutions:
            slope = 1.0 - solution ** 2
            jacobian = slope[:, :, None] * operator
            spectral = np.maximum(
                spectral,
                np.max(np.abs(np.linalg.eigvals(jacobian)), axis=1),
            )
            energy = (
                -0.5 * np.einsum("pi,pij,pj->p", solution, potential, solution)
                - np.sum(bias * solution, axis=1)
            )
            energies.append(energy)
        energy_spread = np.ptp(np.stack(energies, axis=1), axis=1)
        self.branch_evaluations += 1
        self.maximum_reduced_branch_count = max(
            self.maximum_reduced_branch_count, int(np.max(branch_count))
        )
        self.maximum_reduced_branch_spectral_radius = max(
            self.maximum_reduced_branch_spectral_radius, float(np.max(spectral))
        )
        self.reduced_bifurcation_events += int(np.count_nonzero(branch_count > 1))
        self.maximum_potential_energy_spread = max(
            self.maximum_potential_energy_spread, float(np.max(energy_spread))
        )

    def update(self, **kwargs):
        result = super().update(**kwargs)
        current_mean, _ = self._moments(self.density)
        tda = self.online_tda.update(
            month=self.month,
            mean=current_mean,
            joint_action=self.previous_joint_action,
            signed_kernel=self._last_signed_kernel,
            layer_weights=self._last_layer_weights,
            hyper_tail=self.hyper_tail,
            hyper_head=self.hyper_head,
            major_action=self.major_action,
            hyperedge_weight=self.current_hyperedge_weight,
        )
        self.tda_delayed_shock = tda.delayed_topology_shock
        if self.month % 12 == 0:
            self._reduced_branch_probe(
                np.asarray(kwargs["progress"]),
                np.asarray(kwargs["damage"]),
                np.asarray(kwargs["shortage"]),
            )
        return result

    def diagnostics(self) -> dict:
        payload = super().diagnostics()
        payload["v76_signed_potential_scf"] = {
            "signed_graphon_updates": self.signed_graphon_updates,
            "relation_memory_updates": self.relation_updates,
            "maximum_relation_step": self.maximum_relation_step,
            "minimum_positive_relation": self.minimum_positive_relation,
            "maximum_negative_relation": self.maximum_negative_relation,
            "minimum_zero_boundary_formation_rate": self.minimum_zero_boundary_formation_rate,
            "minimum_bridge_index": self.minimum_bridge_index,
            "maximum_bridge_index": self.maximum_bridge_index,
            "maximum_nonautonomous_growth_rate": self.maximum_nonautonomous_growth_rate,
            "minimum_nonautonomous_growth_rate": self.minimum_nonautonomous_growth_rate,
            "atomized_boundary_absorbing": self.minimum_zero_boundary_formation_rate <= 0.0,
            "relation_hysteresis": "distinct formation and fracture barriers with persistent positive/negative kernels",
            "opposing_layers": ["information", "identity"],
            "negative_edge_rule": "bounded_state_flipping_opposing_dynamics",
            "repelling_rule_used": False,
            "maximum_pair_nonpotential_ratio": self.maximum_pair_nonpotential_ratio,
            "maximum_hypergraph_nonpotential_ratio": self.maximum_hypergraph_nonpotential_ratio,
            "maximum_decomposition_residual": self.maximum_decomposition_residual,
            "maximum_signed_field_correction": self.maximum_signed_field_correction,
            "maximum_structural_frustration": self.maximum_structural_frustration,
            "minimum_signed_laplacian_eigenvalue": self.minimum_signed_laplacian_eigenvalue,
            "scf_calls": _SCF_TRACKER.calls,
            "scf_trust_contractions": _SCF_TRACKER.trust_contractions,
            "scf_trust_expansions": _SCF_TRACKER.trust_expansions,
            "maximum_scf_contraction_ratio": _SCF_TRACKER.maximum_contraction_ratio,
            "maximum_scf_spectral_radius_proxy": _SCF_TRACKER.maximum_spectral_radius_proxy,
            "minimum_scf_trust_ratio": (
                _SCF_TRACKER.minimum_trust_ratio
                if np.isfinite(_SCF_TRACKER.minimum_trust_ratio) else 0.0
            ),
            "maximum_scf_trust_ratio": _SCF_TRACKER.maximum_trust_ratio,
            "reduced_branch_evaluations": self.branch_evaluations,
            "maximum_reduced_branch_count": self.maximum_reduced_branch_count,
            "maximum_reduced_branch_spectral_radius": self.maximum_reduced_branch_spectral_radius,
            "reduced_bifurcation_events": self.reduced_bifurcation_events,
            "maximum_potential_energy_spread": self.maximum_potential_energy_spread,
            "branch_scope": "reduced-order diagnostic; not asserted as a second full GMFG equilibrium",
            "potential_scope": "only reciprocal component; circulation remains non-potential",
            "terminal_probability_adjustment": False,
        }
        payload["v76_online_tda"] = self.online_tda.diagnostics()
        return payload
