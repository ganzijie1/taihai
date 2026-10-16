from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from continuous_multipop_mfg_v47 import ACTION_CHANNELS, ACTORS, AGES


@dataclass
class OnlineTDAFactors:
    delayed_topology_shock: np.ndarray
    beta0_persistence_area: np.ndarray
    mapper_components: np.ndarray
    zigzag_change: np.ndarray
    relative_bridge_gain: np.ndarray


class OnlineSocialTDAV76:
    """Low-dimensional online TDA on the executable actor-age state.

    This computes H0 persistence summaries and rank-invariant slices.  It does
    not infer political class labels and does not claim unavailable microdata.
    """

    def __init__(self, paths: int, interval: int = 3):
        self.paths = int(paths)
        self.interval = max(int(interval), 1)
        self.nodes = len(ACTORS) * len(AGES)
        self.epsilon_grid = np.array([0.22, 0.34, 0.48, 0.66, 0.88])
        self.authority_grid = np.array([0.35, 0.62])
        self.delayed_shock = np.zeros(paths)
        self.previous_beta0 = np.full(paths, self.nodes, dtype=float)
        self.updates = 0
        self.hold_months = 0
        self.maximum_beta0_area = 0.0
        self.maximum_mapper_components = 0
        self.maximum_mapper_branches = 0
        self.zigzag_splits = 0
        self.zigzag_merges = 0
        self.maximum_zigzag_change = 0.0
        self.maximum_relative_bridge_gain = 0.0
        self.maximum_directed_flag_triangles = 0.0
        self.maximum_bipersistence_rank = 0.0
        self.maximum_feedback_shock = 0.0
        self.minimum_mass_residual = 0.0
        self.last_bipersistence_rank = np.zeros((paths, 2, len(self.epsilon_grid)))

    @staticmethod
    def _components(adjacency: np.ndarray, active: np.ndarray | None = None) -> np.ndarray:
        paths, nodes, _ = adjacency.shape
        reach = adjacency.copy()
        diagonal = np.arange(nodes)
        if active is None:
            active = np.ones((paths, nodes), dtype=bool)
        reach[:, diagonal, diagonal] |= active
        reach &= active[:, :, None] & active[:, None, :]
        for pivot in range(nodes):
            reach |= reach[:, :, pivot, None] & reach[:, None, pivot, :]
        first = np.argmax(reach, axis=2)
        leaders = active & (first == diagonal[None, :])
        return np.sum(leaders, axis=1).astype(float)

    def _mapper_nerve(self, opinion: np.ndarray, authority: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        opinion_centers = (-0.62, 0.0, 0.62)
        authority_centers = (0.28, 0.72)
        memberships = []
        for center_x in opinion_centers:
            for center_a in authority_centers:
                memberships.append(
                    (np.abs(opinion - center_x) <= 0.48)
                    & (np.abs(authority - center_a) <= 0.38)
                )
        membership = np.stack(memberships, axis=1)
        overlap = np.einsum("pkn,pln->pkl", membership.astype(float), membership.astype(float)) > 0.0
        active = np.any(membership, axis=2)
        components = self._components(overlap, active)
        degree = np.sum(overlap, axis=2) - active
        branches = np.sum(active & (degree == 1), axis=1)
        return components, branches.astype(float)

    def update(
        self,
        month: int,
        mean: np.ndarray,
        joint_action: np.ndarray,
        signed_kernel: np.ndarray,
        layer_weights: np.ndarray,
        hyper_tail: np.ndarray,
        hyper_head: np.ndarray,
        major_action: np.ndarray,
        hyperedge_weight: np.ndarray | None = None,
    ) -> OnlineTDAFactors:
        if month % self.interval != 0:
            self.hold_months += 1
            zeros = np.zeros(self.paths)
            return OnlineTDAFactors(self.delayed_shock.copy(), zeros, zeros, zeros, zeros)
        opinion = mean.reshape(self.paths, self.nodes)
        collective = joint_action[..., ACTION_CHANNELS.index("collective_action")].reshape(self.paths, self.nodes)
        migration = joint_action[..., ACTION_CHANNELS.index("migration")].reshape(self.paths, self.nodes)
        actor_authority = np.mean(np.abs(major_action), axis=2)
        authority = np.repeat(actor_authority, len(AGES), axis=1)
        age_coordinate = np.tile(np.linspace(0.0, 1.0, len(AGES)), len(ACTORS))[None, :]
        age_coordinate = np.broadcast_to(age_coordinate, opinion.shape)
        features = np.stack((
            opinion,
            collective,
            migration,
            authority,
            age_coordinate,
        ), axis=2)
        scale = np.array([1.0, 0.7, 0.7, 0.8, 0.6])[None, None, :]
        features = features * scale
        difference = features[:, :, None, :] - features[:, None, :, :]
        distance = np.sqrt(np.sum(difference ** 2, axis=3))

        beta0_curve = []
        component_graphs = []
        for epsilon in self.epsilon_grid:
            adjacency = distance <= epsilon
            component_graphs.append(adjacency)
            beta0_curve.append(self._components(adjacency))
        beta0_curve = np.stack(beta0_curve, axis=1)
        beta0_area = np.trapz(
            (beta0_curve - 1.0) / max(self.nodes - 1, 1),
            self.epsilon_grid,
            axis=1,
        ) / (self.epsilon_grid[-1] - self.epsilon_grid[0])

        rank_slices = []
        for threshold in self.authority_grid:
            active = authority >= threshold
            ranks = [self._components(graph, active) for graph in component_graphs]
            rank_slices.append(np.stack(ranks, axis=1))
        self.last_bipersistence_rank = np.stack(rank_slices, axis=1)

        effective_actor = np.einsum("pl,plij->pij", layer_weights, signed_kernel)
        actor_age = np.repeat(
            np.repeat(effective_actor, len(AGES), axis=1), len(AGES), axis=2
        )
        tail_mass = np.maximum(np.sum(hyper_tail, axis=1, keepdims=True), 1.0e-12)
        hyper_operator = hyper_head.T @ (hyper_tail / tail_mass)
        hyper_operator /= np.maximum(np.max(np.abs(hyper_operator)), 1.0e-12)
        directed_weight = 0.72 * actor_age + 0.28 * hyper_operator[None, :, :]
        flag_counts = []
        for threshold in (0.12, 0.24, 0.36, 0.52):
            directed = directed_weight > threshold
            np.fill_diagonal(directed[0], False) if self.paths == 1 else None
            diagonal = np.arange(self.nodes)
            directed[:, diagonal, diagonal] = False
            triangles = np.einsum(
                "pij,pjk,pki->p", directed.astype(float), directed.astype(float), directed.astype(float)
            ) / 3.0
            flag_counts.append(triangles)
        directed_flags = np.stack(flag_counts, axis=1)

        mapper_components, mapper_branches = self._mapper_nerve(opinion, authority)
        middle_beta0 = beta0_curve[:, len(self.epsilon_grid) // 2]
        zigzag_change = middle_beta0 - self.previous_beta0
        self.zigzag_splits += int(np.count_nonzero(zigzag_change > 0.0))
        self.zigzag_merges += int(np.count_nonzero(zigzag_change < 0.0))
        self.previous_beta0 = middle_beta0

        base_components = middle_beta0
        major_bridge = np.maximum(authority[:, :, None], authority[:, None, :]) >= 0.62
        bridged = component_graphs[len(self.epsilon_grid) // 2] | major_bridge
        bridged_components = self._components(bridged)
        relative_bridge_gain = np.maximum(base_components - bridged_components, 0.0)

        shock = np.clip(
            0.38 * beta0_area
            + 0.18 * np.clip((mapper_components - 1.0) / 4.0, 0.0, 1.0)
            + 0.18 * np.clip(np.abs(zigzag_change) / 5.0, 0.0, 1.0)
            + 0.14 * np.clip(np.mean(self.last_bipersistence_rank, axis=(1, 2)) / 8.0, 0.0, 1.0)
            - 0.12 * np.clip(relative_bridge_gain / 5.0, 0.0, 1.0),
            0.0, 1.0,
        )
        self.delayed_shock = shock
        self.updates += 1
        self.maximum_beta0_area = max(self.maximum_beta0_area, float(np.max(beta0_area)))
        self.maximum_mapper_components = max(self.maximum_mapper_components, int(np.max(mapper_components)))
        self.maximum_mapper_branches = max(self.maximum_mapper_branches, int(np.max(mapper_branches)))
        self.maximum_zigzag_change = max(self.maximum_zigzag_change, float(np.max(np.abs(zigzag_change))))
        self.maximum_relative_bridge_gain = max(self.maximum_relative_bridge_gain, float(np.max(relative_bridge_gain)))
        self.maximum_directed_flag_triangles = max(self.maximum_directed_flag_triangles, float(np.max(directed_flags)))
        self.maximum_bipersistence_rank = max(self.maximum_bipersistence_rank, float(np.max(self.last_bipersistence_rank)))
        self.maximum_feedback_shock = max(self.maximum_feedback_shock, float(np.max(shock)))
        return OnlineTDAFactors(shock.copy(), beta0_area, mapper_components, zigzag_change, relative_bridge_gain)

    def diagnostics(self) -> dict:
        return {
            "enabled": True,
            "evaluation_interval_months": self.interval,
            "updates": self.updates,
            "hold_months": self.hold_months,
            "nodes": self.nodes,
            "epsilon_grid": self.epsilon_grid.tolist(),
            "authority_grid": self.authority_grid.tolist(),
            "maximum_beta0_persistence_area": self.maximum_beta0_area,
            "maximum_mapper_components": self.maximum_mapper_components,
            "maximum_mapper_branches": self.maximum_mapper_branches,
            "zigzag_splits": self.zigzag_splits,
            "zigzag_merges": self.zigzag_merges,
            "maximum_zigzag_change": self.maximum_zigzag_change,
            "maximum_relative_bridge_gain": self.maximum_relative_bridge_gain,
            "maximum_directed_flag_triangles": self.maximum_directed_flag_triangles,
            "maximum_bipersistence_rank": self.maximum_bipersistence_rank,
            "maximum_feedback_shock": self.maximum_feedback_shock,
            "feedback_timing": "quarterly TDA state becomes a lagged input to subsequent monthly relation dynamics",
            "multiparameter_output": "rank-invariant slices; no fictitious unique two-parameter barcode",
            "scope": "actor-age executable state; no unobserved individual elite or hereditary-class claims",
            "terminal_probability_adjustment": False,
        }
