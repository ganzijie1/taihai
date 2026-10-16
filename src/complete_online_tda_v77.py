from __future__ import annotations

from itertools import combinations

import numpy as np

from continuous_multipop_mfg_v47 import ACTION_CHANNELS, ACTORS, AGES
from online_tda_v76 import OnlineSocialTDAV76, OnlineTDAFactors


def _gf2_rank(columns: list[int]) -> int:
    pivots: dict[int, int] = {}
    for column in columns:
        value = int(column)
        while value:
            pivot = value.bit_length() - 1
            if pivot in pivots:
                value ^= pivots[pivot]
            else:
                pivots[pivot] = value
                break
    return len(pivots)


class CompleteOnlineTDAV77(OnlineSocialTDAV76):
    """Exact finite-complex TDA on the executable actor-age state.

    The annual TDA clock is a declared multirate solver.  Between evaluations
    its latest pathwise topological state is held and continues to affect the
    monthly signed-network equations.
    """

    def __init__(self, paths: int, interval: int = 12):
        super().__init__(paths, interval=interval)
        self.previous_graph: np.ndarray | None = None
        self.maximum_h0_barcode_lifetime = 0.0
        self.maximum_h1_betti = 0
        self.maximum_directed_flag_h1 = 0
        self.maximum_hypergraph_h1 = 0
        self.maximum_mapper_vertices = 0
        self.maximum_mapper_edges = 0
        self.maximum_bipersistence_image_rank = 0
        self.maximum_zigzag_through_rank = 0
        self.maximum_zigzag_births = 0
        self.maximum_zigzag_deaths = 0
        self.maximum_simplicial_boundary_residual = 0

    @staticmethod
    def _labels(adjacency: np.ndarray, active: np.ndarray | None = None) -> np.ndarray:
        nodes = adjacency.shape[0]
        if active is None:
            active = np.ones(nodes, dtype=bool)
        labels = np.full(nodes, -1, dtype=int)
        label = 0
        for start in range(nodes):
            if not active[start] or labels[start] >= 0:
                continue
            stack = [start]
            labels[start] = label
            while stack:
                node = stack.pop()
                for neighbor in np.flatnonzero(adjacency[node] & active):
                    if labels[neighbor] < 0:
                        labels[neighbor] = label
                        stack.append(int(neighbor))
            label += 1
        return labels

    @staticmethod
    def _mst_deaths(distance: np.ndarray) -> np.ndarray:
        nodes = distance.shape[0]
        parent = list(range(nodes))

        def root(value: int) -> int:
            while parent[value] != value:
                parent[value] = parent[parent[value]]
                value = parent[value]
            return value

        edges = sorted(
            (float(distance[i, j]), i, j)
            for i in range(nodes) for j in range(i + 1, nodes)
        )
        deaths = []
        for weight, left, right in edges:
            a, b = root(left), root(right)
            if a != b:
                parent[b] = a
                deaths.append(weight)
                if len(deaths) == nodes - 1:
                    break
        return np.asarray(deaths)

    @staticmethod
    def _clique_betti_one(adjacency: np.ndarray) -> tuple[int, int]:
        nodes = adjacency.shape[0]
        labels = CompleteOnlineTDAV77._labels(adjacency)
        beta0 = len(set(labels.tolist()))
        edges = [(i, j) for i in range(nodes) for j in range(i + 1, nodes) if adjacency[i, j]]
        edge_index = {edge: index for index, edge in enumerate(edges)}
        columns: list[int] = []
        for i, j, k in combinations(range(nodes), 3):
            if adjacency[i, j] and adjacency[i, k] and adjacency[j, k]:
                column = 0
                for edge in ((i, j), (i, k), (j, k)):
                    column ^= 1 << edge_index[edge]
                columns.append(column)
        rank_boundary_two = _gf2_rank(columns)
        beta1 = len(edges) - nodes + beta0 - rank_boundary_two
        return beta0, max(int(beta1), 0)

    def _exact_mapper(self, opinion: np.ndarray, authority: np.ndarray,
                      metric_graph: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        components = np.zeros(self.paths)
        branches = np.zeros(self.paths)
        vertices_count = np.zeros(self.paths)
        covers = [
            (cx, ca) for cx in (-0.62, 0.0, 0.62) for ca in (0.28, 0.72)
        ]
        for path in range(self.paths):
            mapper_vertices: list[set[int]] = []
            for center_x, center_a in covers:
                active = (
                    (np.abs(opinion[path] - center_x) <= 0.48)
                    & (np.abs(authority[path] - center_a) <= 0.38)
                )
                labels = self._labels(metric_graph[path], active)
                for label in sorted(set(labels[labels >= 0].tolist())):
                    mapper_vertices.append(set(np.flatnonzero(labels == label).tolist()))
            count = len(mapper_vertices)
            nerve = np.zeros((count, count), dtype=bool)
            for left in range(count):
                nerve[left, left] = True
                for right in range(left + 1, count):
                    if mapper_vertices[left] & mapper_vertices[right]:
                        nerve[left, right] = nerve[right, left] = True
            labels = self._labels(nerve) if count else np.empty(0, dtype=int)
            components[path] = len(set(labels.tolist())) if count else 0
            degree = np.sum(nerve, axis=1) - 1 if count else np.empty(0)
            branches[path] = np.count_nonzero(degree == 1)
            vertices_count[path] = count
            self.maximum_mapper_edges = max(
                self.maximum_mapper_edges, int((np.count_nonzero(nerve) - count) // 2)
            )
        return components, branches, vertices_count

    def _bipersistence(self, graphs: list[np.ndarray], authority: np.ndarray) -> np.ndarray:
        ranks = np.zeros((self.paths, len(self.authority_grid), len(graphs)))
        max_image = 0
        for path in range(self.paths):
            labels_grid: list[list[np.ndarray]] = []
            active_grid: list[np.ndarray] = []
            for quantile in self.authority_grid:
                threshold = float(np.quantile(authority[path], quantile))
                active = authority[path] >= threshold
                active_grid.append(active)
                row = []
                for graph in graphs:
                    labels = self._labels(graph[path], active)
                    row.append(labels)
                labels_grid.append(row)
            for a, row in enumerate(labels_grid):
                for e, labels in enumerate(row):
                    ranks[path, a, e] = len(set(labels[labels >= 0].tolist()))
            # Parameter order: authority threshold decreases and epsilon grows.
            for source_a in range(len(self.authority_grid) - 1, -1, -1):
                for target_a in range(source_a, -1, -1):
                    source_active = active_grid[source_a]
                    for source_e in range(len(graphs)):
                        source_labels = labels_grid[source_a][source_e]
                        source_nodes = source_active & (source_labels >= 0)
                        for target_e in range(source_e, len(graphs)):
                            target_labels = labels_grid[target_a][target_e]
                            image = len(set(target_labels[source_nodes].tolist()) - {-1})
                            max_image = max(max_image, image)
        self.maximum_bipersistence_image_rank = max(
            self.maximum_bipersistence_image_rank, max_image
        )
        return ranks

    def _zigzag(self, current: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        through = np.zeros(self.paths)
        births = np.zeros(self.paths)
        deaths = np.zeros(self.paths)
        if self.previous_graph is None:
            self.previous_graph = current.copy()
            return through, births, deaths
        for path in range(self.paths):
            old_labels = self._labels(self.previous_graph[path])
            new_labels = self._labels(current[path])
            union_labels = self._labels(self.previous_graph[path] | current[path])
            old_count = len(set(old_labels.tolist()))
            new_count = len(set(new_labels.tolist()))
            union_count = len(set(union_labels.tolist()))
            through[path] = union_count
            births[path] = max(new_count - union_count, 0)
            deaths[path] = max(old_count - union_count, 0)
        self.previous_graph = current.copy()
        return through, births, deaths

    def _directed_flag_h1(self, directed_actor: np.ndarray) -> np.ndarray:
        result = np.zeros(self.paths)
        for path in range(self.paths):
            edges = [(i, j) for i in range(4) for j in range(4) if i != j and directed_actor[path, i, j]]
            edge_index = {edge: index for index, edge in enumerate(edges)}
            boundary_one_columns = [(1 << i) ^ (1 << j) for i, j in edges]
            rank_b1 = _gf2_rank(boundary_one_columns)
            triangles: list[int] = []
            for i, j, k in combinations(range(4), 3):
                for order in ((i, j, k), (i, k, j), (j, i, k), (j, k, i), (k, i, j), (k, j, i)):
                    a, b, c = order
                    if (a, b) in edge_index and (b, c) in edge_index and (a, c) in edge_index:
                        triangles.append(
                            (1 << edge_index[(a, b)])
                            ^ (1 << edge_index[(b, c)])
                            ^ (1 << edge_index[(a, c)])
                        )
            rank_b2 = _gf2_rank(triangles)
            result[path] = max(len(edges) - rank_b1 - rank_b2, 0)
        return result

    def _hypergraph_homology(self, hyper_tail: np.ndarray, hyper_head: np.ndarray,
                             edge_weight: np.ndarray) -> np.ndarray:
        result = np.zeros(self.paths)
        node_sets = [
            np.flatnonzero((hyper_tail[edge] + hyper_head[edge]) > 0.0).tolist()
            for edge in range(hyper_tail.shape[0])
        ]
        for path in range(self.paths):
            threshold = float(np.median(edge_weight[path]))
            adjacency = np.eye(self.nodes, dtype=bool)
            active_sets = []
            for edge, nodes in enumerate(node_sets):
                if edge_weight[path, edge] >= threshold:
                    active_sets.append(nodes)
                    for i, j in combinations(nodes, 2):
                        adjacency[i, j] = adjacency[j, i] = True
            edges = [(i, j) for i in range(self.nodes) for j in range(i + 1, self.nodes) if adjacency[i, j]]
            edge_index = {edge: index for index, edge in enumerate(edges)}
            triangles = set()
            for nodes in active_sets:
                for triangle in combinations(sorted(nodes), 3):
                    triangles.add(triangle)
            columns = []
            for i, j, k in triangles:
                columns.append(
                    (1 << edge_index[(i, j)])
                    ^ (1 << edge_index[(i, k)])
                    ^ (1 << edge_index[(j, k)])
                )
            beta0 = len(set(self._labels(adjacency).tolist()))
            result[path] = max(
                len(edges) - self.nodes + beta0 - _gf2_rank(columns), 0
            )
        return result

    def update(self, month: int, mean: np.ndarray, joint_action: np.ndarray,
               signed_kernel: np.ndarray, layer_weights: np.ndarray,
               hyper_tail: np.ndarray, hyper_head: np.ndarray,
               major_action: np.ndarray, hyperedge_weight: np.ndarray | None = None):
        if month % self.interval != 0:
            self.hold_months += 1
            zeros = np.zeros(self.paths)
            return OnlineTDAFactors(self.delayed_shock.copy(), zeros, zeros, zeros, zeros)
        opinion = mean.reshape(self.paths, self.nodes)
        collective = joint_action[..., ACTION_CHANNELS.index("collective_action")].reshape(self.paths, self.nodes)
        migration = joint_action[..., ACTION_CHANNELS.index("migration")].reshape(self.paths, self.nodes)
        actor_authority = np.mean(np.abs(major_action), axis=2)
        authority = np.repeat(actor_authority, len(AGES), axis=1)
        age_coordinate = np.broadcast_to(
            np.tile(np.linspace(0.0, 1.0, len(AGES)), len(ACTORS))[None, :], opinion.shape
        )
        features = np.stack((opinion, collective, migration, authority, age_coordinate), axis=2)
        features *= np.array([1.0, 0.7, 0.7, 0.8, 0.6])[None, None, :]
        distance = np.linalg.norm(features[:, :, None, :] - features[:, None, :, :], axis=3)
        graphs = [distance <= epsilon for epsilon in self.epsilon_grid]
        beta0_curve = np.stack([self._components(graph) for graph in graphs], axis=1)
        beta0_area = np.trapz(
            (beta0_curve - 1.0) / max(self.nodes - 1, 1), self.epsilon_grid, axis=1
        ) / (self.epsilon_grid[-1] - self.epsilon_grid[0])
        barcode_lifetime = np.zeros(self.paths)
        h1_curve = np.zeros((self.paths, len(graphs)))
        for path in range(self.paths):
            deaths = self._mst_deaths(distance[path])
            barcode_lifetime[path] = float(np.sum(deaths))
            for index, graph in enumerate(graphs):
                _, h1_curve[path, index] = self._clique_betti_one(graph[path])
        self.last_bipersistence_rank = self._bipersistence(graphs, authority)
        mapper_components, mapper_branches, mapper_vertices = self._exact_mapper(
            opinion, authority, graphs[1]
        )
        through, births, deaths = self._zigzag(graphs[len(graphs) // 2])
        zigzag_change = births - deaths

        effective_actor = np.einsum("pl,plij->pij", layer_weights, signed_kernel)
        directed_actor = effective_actor > np.quantile(
            effective_actor, 0.58, axis=(1, 2), keepdims=True
        )
        diagonal = np.arange(len(ACTORS))
        directed_actor[:, diagonal, diagonal] = False
        directed_h1 = self._directed_flag_h1(directed_actor)
        if hyperedge_weight is None:
            edge_weight = np.ones((self.paths, hyper_tail.shape[0]))
        else:
            edge_weight = np.mean(np.asarray(hyperedge_weight), axis=1)
        hypergraph_h1 = self._hypergraph_homology(hyper_tail, hyper_head, edge_weight)

        base_components = beta0_curve[:, len(graphs) // 2]
        major_bridge = np.maximum(authority[:, :, None], authority[:, None, :]) >= 0.62
        bridged_components = self._components(graphs[len(graphs) // 2] | major_bridge)
        relative_bridge_gain = np.maximum(base_components - bridged_components, 0.0)
        shock = np.clip(
            0.27 * beta0_area
            + 0.13 * np.clip(np.max(h1_curve, axis=1) / 8.0, 0.0, 1.0)
            + 0.10 * np.clip(directed_h1 / 5.0, 0.0, 1.0)
            + 0.10 * np.clip(hypergraph_h1 / 12.0, 0.0, 1.0)
            + 0.12 * np.clip((mapper_components - 1.0) / 4.0, 0.0, 1.0)
            + 0.12 * np.clip((births + deaths) / 5.0, 0.0, 1.0)
            + 0.10 * np.clip(np.mean(self.last_bipersistence_rank, axis=(1, 2)) / 8.0, 0.0, 1.0)
            - 0.06 * np.clip(relative_bridge_gain / 5.0, 0.0, 1.0), 0.0, 1.0
        )
        self.delayed_shock = shock
        self.updates += 1
        self.zigzag_splits += int(np.sum(births))
        self.zigzag_merges += int(np.sum(deaths))
        self.maximum_beta0_area = max(self.maximum_beta0_area, float(np.max(beta0_area)))
        self.maximum_beta0_persistence_area = self.maximum_beta0_area
        self.maximum_h0_barcode_lifetime = max(self.maximum_h0_barcode_lifetime, float(np.max(barcode_lifetime)))
        self.maximum_h1_betti = max(self.maximum_h1_betti, int(np.max(h1_curve)))
        self.maximum_directed_flag_h1 = max(self.maximum_directed_flag_h1, int(np.max(directed_h1)))
        self.maximum_hypergraph_h1 = max(self.maximum_hypergraph_h1, int(np.max(hypergraph_h1)))
        self.maximum_mapper_components = max(self.maximum_mapper_components, int(np.max(mapper_components)))
        self.maximum_mapper_branches = max(self.maximum_mapper_branches, int(np.max(mapper_branches)))
        self.maximum_mapper_vertices = max(self.maximum_mapper_vertices, int(np.max(mapper_vertices)))
        self.maximum_zigzag_through_rank = max(self.maximum_zigzag_through_rank, int(np.max(through)))
        self.maximum_zigzag_births = max(self.maximum_zigzag_births, int(np.max(births)))
        self.maximum_zigzag_deaths = max(self.maximum_zigzag_deaths, int(np.max(deaths)))
        self.maximum_zigzag_change = max(self.maximum_zigzag_change, float(np.max(np.abs(zigzag_change))))
        self.maximum_relative_bridge_gain = max(self.maximum_relative_bridge_gain, float(np.max(relative_bridge_gain)))
        self.maximum_bipersistence_rank = max(self.maximum_bipersistence_rank, float(np.max(self.last_bipersistence_rank)))
        self.maximum_feedback_shock = max(self.maximum_feedback_shock, float(np.max(shock)))
        return OnlineTDAFactors(shock.copy(), beta0_area, mapper_components, zigzag_change, relative_bridge_gain)

    def diagnostics(self) -> dict:
        payload = super().diagnostics()
        payload.update({
            "feedback_timing": "annual exact TDA state is held and enters subsequent monthly relation dynamics",
            "multiparameter_output": "exact finite-grid H0 image-rank invariant; no fictitious unique two-parameter barcode",
            "solver_scope": "exact finite complexes on 16 actor-age nodes; annual multirate clock",
            "maximum_h0_barcode_total_lifetime": self.maximum_h0_barcode_lifetime,
            "maximum_vietoris_rips_h1_betti": self.maximum_h1_betti,
            "maximum_directed_flag_h1_betti": self.maximum_directed_flag_h1,
            "maximum_hypergraph_simplicial_h1_betti": self.maximum_hypergraph_h1,
            "maximum_mapper_vertices": self.maximum_mapper_vertices,
            "maximum_mapper_edges": self.maximum_mapper_edges,
            "maximum_bipersistence_image_rank": self.maximum_bipersistence_image_rank,
            "maximum_zigzag_through_rank": self.maximum_zigzag_through_rank,
            "maximum_zigzag_births": self.maximum_zigzag_births,
            "maximum_zigzag_deaths": self.maximum_zigzag_deaths,
            "persistent_homology": "Kruskal H0 barcode plus exact GF(2) clique H1 Betti filtration",
            "mapper": "overlapping pullback cover with within-cover connected clustering and nerve",
            "bipersistence": "authority-distance bifiltration with exact H0 image-rank invariant",
            "zigzag": "exact consecutive H0 union-zigzag ranks, births and deaths",
            "hypergraph_homology": "dynamic hyperedge simplicial closure with exact GF(2) H1",
        })
        return payload
