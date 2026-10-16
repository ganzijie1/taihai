from __future__ import annotations

import numpy as np

from continuous_multipop_mfg_v47 import (
    ACTION_CHANNELS,
    ACTORS,
    AGES,
    GRAPHON_LAYERS,
)
from continuous_multipop_mfg_v66 import ContinuousMultiPopulationMFGV66


class DirectedMultilayerHypergraphMFGV75(ContinuousMultiPopulationMFGV66):
    """Graphon GMFG augmented by directed higher-order interactions.

    Pairwise Graphon fields remain active.  A directed multilayer hypergraph
    adds group interactions among actor-age populations.  Hyperedge signals
    include within-edge dispersion and a major-player gate, so the operator is
    not equivalent to a fixed clique expansion or a post-hoc field multiplier.
    """

    def __init__(self, *args, **kwargs):
        self.hypergraph_enabled = bool(kwargs.pop("hypergraph_enabled", True))
        self.hypergraph_share = float(
            np.clip(kwargs.pop("hypergraph_share", 0.32), 0.0, 0.65)
        )
        self.hypergraph_uncertainty_scale = float(
            max(kwargs.pop("hypergraph_uncertainty_scale", 0.05), 0.0)
        )
        super().__init__(*args, **kwargs)
        self.hyper_tail, self.hyper_head, self.hyperedge_labels = (
            self._build_directed_hypergraph()
        )
        edge_count = self.hyper_tail.shape[1]
        perturbation = self.rng.normal(
            0.0,
            self.hypergraph_uncertainty_scale,
            (self.paths, len(GRAPHON_LAYERS), edge_count),
        )
        base = np.linspace(0.88, 1.12, edge_count)[None, None, :]
        self.path_hyperedge_prior = np.exp(perturbation) * base
        self.current_hyperedge_weight = self.path_hyperedge_prior.copy()
        self.max_hypergraph_incidence_residual = 0.0
        self.max_hypergraph_field = 0.0
        self.max_higher_order_contribution = 0.0
        self.max_clique_rollback_error = 0.0
        self.hypergraph_field_evaluations = 0
        self.hypergraph_weight_updates = 0

    @staticmethod
    def _node(actor: int, age: int) -> int:
        return actor * len(AGES) + age

    def _build_directed_hypergraph(
        self,
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        node_count = len(ACTORS) * len(AGES)
        tails: list[np.ndarray] = []
        heads: list[np.ndarray] = []
        labels: list[str] = []

        def add(label: str, tail_nodes: list[int], head_nodes: list[int]) -> None:
            tail = np.zeros(node_count)
            head = np.zeros(node_count)
            tail[tail_nodes] = 1.0
            head[head_nodes] = 1.0
            tails.append(tail)
            heads.append(head)
            labels.append(label)

        # Within-actor age coalitions: a joint cohort signal reaches all ages.
        for actor in range(len(ACTORS)):
            nodes = [self._node(actor, age) for age in range(len(AGES))]
            add(f"within_{ACTORS[actor]}", nodes, nodes)

        # Same-age cross-actor fields capture common information and price
        # environments that cannot be represented by one sender-receiver edge.
        for age in range(len(AGES)):
            nodes = [self._node(actor, age) for actor in range(len(ACTORS))]
            add(f"cohort_{AGES[age]}", nodes, nodes)

        # Directed triadic fields.  These are anonymous actor-level relations,
        # not operational alliances or target lists.
        actor_triads = (
            ((1, 2, 3), (0,)),
            ((0, 2, 3), (1,)),
            ((0, 1, 3), (2,)),
            ((0, 1, 2), (3,)),
        )
        for edge, (tail_actors, head_actors) in enumerate(actor_triads):
            tail_nodes = [
                self._node(actor, age)
                for actor in tail_actors for age in range(len(AGES))
            ]
            head_nodes = [
                self._node(actor, age)
                for actor in head_actors for age in range(len(AGES))
            ]
            add(f"directed_triad_{edge}", tail_nodes, head_nodes)

        # Cross-generational transmission uses multiple cohorts jointly.
        for actor in range(len(ACTORS)):
            add(
                f"intergenerational_{ACTORS[actor]}",
                [self._node(actor, 2), self._node(actor, 3)],
                [self._node(actor, 0), self._node(actor, 1)],
            )

        tail = np.stack(tails, axis=0)
        head = np.stack(heads, axis=0)
        if np.any(np.sum(tail, axis=1) <= 1.0):
            raise ValueError("every hyperedge must contain at least two tail nodes")
        if np.any(np.sum(head, axis=1) <= 0.0):
            raise ValueError("every hyperedge must have a receiving population")
        return tail, head, labels

    def _effective_graphon(self, *args, **kwargs):
        kernel, weights, intensity = super()._effective_graphon(*args, **kwargs)
        if not self.hypergraph_enabled:
            return kernel, weights, intensity
        damage, shortage, network, common_noise, jump_signal, actor_signal = args[:6]
        tail_by_actor = self.hyper_tail.reshape(
            self.hyper_tail.shape[0], len(ACTORS), len(AGES)
        ).sum(axis=2)
        tail_by_actor /= np.maximum(
            np.sum(tail_by_actor, axis=1, keepdims=True), 1.0e-12
        )
        edge_damage = np.einsum("ea,pa->pe", tail_by_actor, damage)
        edge_shortage = np.einsum("ea,pa->pe", tail_by_actor, shortage)
        edge_network = np.einsum("ea,pa->pe", tail_by_actor, network)
        edge_noise = np.einsum("ea,pa->pe", tail_by_actor, common_noise)
        edge_jump = np.einsum("ea,pa->pe", tail_by_actor, jump_signal)
        edge_signal = np.einsum("ea,pa->pe", tail_by_actor, actor_signal)
        layer_driver = np.stack((
            0.26 * edge_network + 0.20 * edge_noise + 0.16 * edge_signal,
            0.24 * edge_shortage + 0.18 * edge_damage + 0.10 * edge_jump,
            0.18 * edge_signal + 0.12 * edge_network - 0.08 * edge_noise,
            0.28 * edge_damage + 0.22 * edge_jump + 0.10 * edge_shortage,
        ), axis=1)
        dynamic = self.path_hyperedge_prior * np.exp(
            np.clip(layer_driver, -0.55, 0.75)
        )
        # Preserve the meaning of Graphon layer weights: hyperedges redistribute
        # interaction inside a layer rather than adding unbounded total mass.
        dynamic /= np.maximum(np.mean(dynamic, axis=2, keepdims=True), 1.0e-12)
        self.current_hyperedge_weight = dynamic
        self.hypergraph_weight_updates += 1
        return kernel, weights, intensity

    def _hypergraph_fields(
        self,
        mean: np.ndarray,
        action: np.ndarray,
        weights: np.ndarray,
        major_action: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        paths = mean.shape[0]
        nodes = len(ACTORS) * len(AGES)
        flat_state = mean.reshape(paths, nodes)
        flat_action = action.reshape(paths, nodes, len(ACTION_CHANNELS))
        tail_mass = np.sum(self.hyper_tail, axis=1)
        head_degree = np.einsum(
            "pe,en->pn", np.mean(self.current_hyperedge_weight, axis=1), self.hyper_head
        )
        incidence_residual = float(np.max(np.maximum(1.0e-12 - head_degree, 0.0)))
        self.max_hypergraph_incidence_residual = max(
            self.max_hypergraph_incidence_residual, incidence_residual
        )

        edge_state = np.einsum("en,pn->pe", self.hyper_tail, flat_state)
        edge_state /= tail_mass[None, :]
        edge_second = np.einsum("en,pn->pe", self.hyper_tail, flat_state ** 2)
        edge_second /= tail_mass[None, :]
        edge_dispersion = np.maximum(edge_second - edge_state ** 2, 0.0)
        edge_action = np.einsum("en,pnc->pec", self.hyper_tail, flat_action)
        edge_action /= tail_mass[None, :, None]

        actor_tail = self.hyper_tail.reshape(
            self.hyper_tail.shape[0], len(ACTORS), len(AGES)
        ).sum(axis=2)
        actor_tail /= np.maximum(np.sum(actor_tail, axis=1, keepdims=True), 1.0e-12)
        major_by_actor = np.einsum(
            "pam,m->pa", major_action, self.major_role_weights
        )
        edge_major = np.einsum("ea,pa->pe", actor_tail, major_by_actor)
        major_gate = np.clip(1.0 + 0.16 * np.tanh(edge_major), 0.78, 1.22)

        nonlinear_state = np.tanh(
            edge_state + 0.20 * edge_state * np.abs(edge_state)
            - 0.10 * edge_dispersion
        )
        nonlinear_action = np.clip(
            edge_action * (1.0 + 0.12 * edge_state[..., None])
            + 0.05 * edge_action * np.mean(edge_action, axis=-1, keepdims=True),
            0.0,
            1.5,
        )
        layer_state = np.empty((paths, len(GRAPHON_LAYERS), nodes))
        layer_action = np.empty(
            (paths, len(GRAPHON_LAYERS), nodes, len(ACTION_CHANNELS))
        )
        layer_major = np.empty((paths, len(GRAPHON_LAYERS), nodes))
        for layer in range(len(GRAPHON_LAYERS)):
            # Finite-network validation intentionally uses a path subsample.
            # Slice the pathwise hyperedge process to the caller's batch rather
            # than broadcasting the full Monte Carlo population into it.
            edge_weight = self.current_hyperedge_weight[:paths, layer] * major_gate
            denominator = np.einsum("pe,en->pn", edge_weight, self.hyper_head)
            denominator = np.maximum(denominator, 1.0e-12)
            layer_state[:, layer] = (
                np.einsum(
                    "pe,pe,en->pn", edge_weight, nonlinear_state, self.hyper_head
                ) / denominator
            )
            layer_action[:, layer] = (
                np.einsum(
                    "pe,pec,en->pnc", edge_weight, nonlinear_action, self.hyper_head
                ) / denominator[..., None]
            )
            layer_major[:, layer] = (
                np.einsum(
                    "pe,pe,en->pn", edge_weight, edge_major, self.hyper_head
                ) / denominator
            )

        state = np.einsum("pl,pln->pn", weights, layer_state).reshape(mean.shape)
        action_field = np.einsum(
            "pl,plnc->pnc", weights, layer_action
        ).reshape(action.shape)
        major_nodes = np.einsum("pl,pln->pn", weights, layer_major).reshape(
            paths, len(ACTORS), len(AGES)
        )
        major_field = np.sum(
            major_nodes * self.age_weights[None, None, :], axis=2
        )

        # Clique expansion is a diagnostic comparator only.  The executable
        # field above retains dispersion and major-gated nonlinear terms.
        clique = self.hyper_head.T @ (
            self.hyper_tail / tail_mass[:, None]
        )
        clique /= np.maximum(np.sum(clique, axis=1, keepdims=True), 1.0e-12)
        clique_field = (flat_state @ clique.T).reshape(mean.shape)
        rollback_error = np.abs(state - clique_field)
        self.max_clique_rollback_error = max(
            self.max_clique_rollback_error, float(np.max(rollback_error))
        )
        self.max_hypergraph_field = max(
            self.max_hypergraph_field,
            float(max(np.max(np.abs(state)), np.max(np.abs(action_field)))),
        )
        self.hypergraph_field_evaluations += 1
        return state, action_field, major_field, rollback_error

    def _multilayer_fields(
        self,
        mean: np.ndarray,
        action: np.ndarray,
        kernel: np.ndarray,
        weights: np.ndarray,
        major_action: np.ndarray,
        intensity_multiplier: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        pair_state, pair_action, pair_major = super()._multilayer_fields(
            mean, action, kernel, weights, major_action, intensity_multiplier
        )
        if not self.hypergraph_enabled:
            return pair_state, pair_action, pair_major
        hyper_state, hyper_action, hyper_major, _ = self._hypergraph_fields(
            mean, action, weights, major_action
        )
        share = self.hypergraph_share
        state = (1.0 - share) * pair_state + share * hyper_state
        action_field = (1.0 - share) * pair_action + share * hyper_action
        major_field = (1.0 - share) * pair_major + share * hyper_major
        contribution = max(
            np.max(np.abs(state - pair_state)),
            np.max(np.abs(action_field - pair_action)),
            np.max(np.abs(major_field - pair_major)),
        )
        self.max_higher_order_contribution = max(
            self.max_higher_order_contribution, float(contribution)
        )
        return state, action_field, major_field

    def _finite_network_fields(
        self,
        mean: np.ndarray,
        action: np.ndarray,
        kernel: np.ndarray,
        weights: np.ndarray,
        intensity_multiplier: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Back-project pairwise edges and higher-order hyperedges together."""
        # The inherited finite-network sampler calls ``self._multilayer_fields``
        # for its pairwise reference.  Disable only the hypergraph extension for
        # that call so the pairwise term is not mixed twice below.
        enabled = self.hypergraph_enabled
        self.hypergraph_enabled = False
        try:
            pair_state, pair_action, _ = super()._finite_network_fields(
                mean, action, kernel, weights, intensity_multiplier
            )
        finally:
            self.hypergraph_enabled = enabled
        checked = pair_state.shape[0]
        if not self.hypergraph_enabled:
            reference_state, reference_action, _ = super()._multilayer_fields(
                mean[:checked], action[:checked], kernel[:checked], weights[:checked],
                self.major_action[:checked],
                None if intensity_multiplier is None else intensity_multiplier[:checked],
            )
            error = float(max(
                np.max(np.abs(pair_state - reference_state)),
                np.max(np.abs(pair_action - reference_action)),
            ))
            return pair_state, pair_action, error
        hyper_state, hyper_action, _, _ = self._hypergraph_fields(
            mean[:checked], action[:checked], weights[:checked],
            self.major_action[:checked],
        )
        share = self.hypergraph_share
        sampled_state = (1.0 - share) * pair_state + share * hyper_state
        sampled_action = (1.0 - share) * pair_action + share * hyper_action
        reference_state, reference_action, _ = self._multilayer_fields(
            mean[:checked], action[:checked], kernel[:checked], weights[:checked],
            self.major_action[:checked],
            None if intensity_multiplier is None else intensity_multiplier[:checked],
        )
        error = float(max(
            np.max(np.abs(sampled_state - reference_state)),
            np.max(np.abs(sampled_action - reference_action)),
        ))
        return sampled_state, sampled_action, error

    def diagnostics(self) -> dict:
        payload = super().diagnostics()
        payload["v75_directed_multilayer_hypergraph_mfg"] = {
            "enabled": self.hypergraph_enabled,
            "layers": list(GRAPHON_LAYERS),
            "nodes": len(ACTORS) * len(AGES),
            "directed_hyperedges": int(self.hyper_tail.shape[0]),
            "hyperedge_labels": list(self.hyperedge_labels),
            "pairwise_graphon_share": 1.0 - self.hypergraph_share,
            "higher_order_share": self.hypergraph_share,
            "uncertainty_log_scale": self.hypergraph_uncertainty_scale,
            "weight_updates": self.hypergraph_weight_updates,
            "field_evaluations": self.hypergraph_field_evaluations,
            "max_incidence_residual": self.max_hypergraph_incidence_residual,
            "max_hypergraph_field": self.max_hypergraph_field,
            "max_higher_order_contribution": self.max_higher_order_contribution,
            "max_clique_rollback_error": self.max_clique_rollback_error,
            "major_minor_coupling": "major actions gate directed hyperedge fields inside each fixed-point iteration",
            "non_pairwise_mechanisms": [
                "tail-group conditional mean",
                "within-hyperedge dispersion penalty",
                "state-action cross-channel synergy",
                "major-player endogenous edge gate",
            ],
            "hard_capacity_owner": "specialized operational network solvers",
            "terminal_probability_adjustment": False,
        }
        return payload
