from __future__ import annotations

from itertools import combinations, permutations

import numpy as np

from combinatorial_network_v71 import CombinatorialOperationalNetworkV71


# Anonymous depth--transit--front graph.  Nodes deliberately do not identify
# real facilities; public aggregate data only support this resolution.
EDGE_TAIL = np.array([0, 0, 1, 1, 2, 2, 3, 4, 5, 3, 4, 2], dtype=int)
EDGE_HEAD = np.array([1, 2, 3, 4, 3, 4, 5, 5, 6, 6, 6, 5], dtype=int)
EDGE_DELAY = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2], dtype=int)
EDGE_LAYER = np.array([0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 3, 2], dtype=int)
EDGE_SIGN = np.array([1, 1, 1, -1, -1, 1, 1, 1, 1, 1, 1, -1], dtype=float)
NODE_COUNT = 7
SOURCE = 0
SINK = 6


def _simple_paths() -> tuple[tuple[int, ...], ...]:
    outgoing = [[] for _ in range(NODE_COUNT)]
    for edge, (tail, head) in enumerate(zip(EDGE_TAIL, EDGE_HEAD)):
        outgoing[tail].append((head, edge))
    paths: list[tuple[int, ...]] = []

    def visit(node: int, used_nodes: set[int], edges: list[int]) -> None:
        if node == SINK:
            paths.append(tuple(edges))
            return
        for head, edge in outgoing[node]:
            if head not in used_nodes:
                visit(head, used_nodes | {head}, edges + [edge])

    visit(SOURCE, {SOURCE}, [])
    return tuple(sorted(paths, key=lambda p: (int(np.sum(EDGE_DELAY[list(p)])), len(p), p)))


PATHS = _simple_paths()
PATH_INCIDENCE = np.zeros((len(PATHS), len(EDGE_TAIL)), dtype=float)
for path_index, path in enumerate(PATHS):
    PATH_INCIDENCE[path_index, list(path)] = 1.0
PATH_DELAYS = np.array([int(np.sum(EDGE_DELAY[list(path)])) for path in PATHS])


def _cut_masks() -> tuple[np.ndarray, ...]:
    cuts = []
    internal = tuple(range(1, SINK))
    for code in range(1 << len(internal)):
        side = np.zeros(NODE_COUNT, dtype=bool)
        side[SOURCE] = True
        for offset, node in enumerate(internal):
            side[node] = bool((code >> offset) & 1)
        cuts.append(side[EDGE_TAIL] & ~side[EDGE_HEAD])
    return tuple(cuts)


CUT_MASKS = _cut_masks()


def _feasible_protection_sets() -> tuple[tuple[int, ...], ...]:
    conflict = {(0, 1), (9, 10)}
    feasible = [tuple()]
    for size in range(1, 4):
        for chosen in combinations(range(len(EDGE_TAIL)), size):
            if any(a in chosen and b in chosen for a, b in conflict):
                continue
            # Partition-matroid rule: at most two actions in any layer.
            if any(np.sum(EDGE_LAYER[list(chosen)] == layer) > 2 for layer in range(4)):
                continue
            feasible.append(chosen)
    return tuple(feasible)


PROTECTION_SETS = _feasible_protection_sets()


def _edge_disjoint_path_sets() -> tuple[tuple[int, ...], ...]:
    feasible: list[tuple[int, ...]] = []
    for size in range(1, min(4, len(PATHS)) + 1):
        for chosen in combinations(range(len(PATHS)), size):
            use = np.sum(PATH_INCIDENCE[list(chosen)], axis=0)
            if np.max(use) <= 1.0:
                feasible.append(chosen)
    return tuple(feasible)


DISJOINT_PATH_SETS = _edge_disjoint_path_sets()

EDGE_DEPENDENCY = np.zeros((len(EDGE_TAIL), len(EDGE_TAIL)))
for left in range(len(EDGE_TAIL)):
    left_nodes = {int(EDGE_TAIL[left]), int(EDGE_HEAD[left])}
    for right in range(len(EDGE_TAIL)):
        if left != right and left_nodes & {int(EDGE_TAIL[right]), int(EDGE_HEAD[right])}:
            EDGE_DEPENDENCY[left, right] = 1.0
EDGE_DEPENDENCY /= np.maximum(np.sum(EDGE_DEPENDENCY, axis=1, keepdims=True), 1.0)


def _exact_treewidth() -> int:
    adjacency = [set() for _ in range(NODE_COUNT)]
    for tail, head in zip(EDGE_TAIL, EDGE_HEAD):
        adjacency[tail].add(head)
        adjacency[head].add(tail)
    best = NODE_COUNT - 1
    for order in permutations(range(NODE_COUNT)):
        graph = [set(neighbors) for neighbors in adjacency]
        width = 0
        for node in order:
            neighbors = list(graph[node])
            width = max(width, len(neighbors))
            if width >= best:
                break
            for left, right in combinations(neighbors, 2):
                graph[left].add(right)
                graph[right].add(left)
            for neighbor in neighbors:
                graph[neighbor].discard(node)
            graph[node].clear()
        best = min(best, width)
    return best


TREEWIDTH = _exact_treewidth()


class AlgebraicDiscreteOperationalNetworkV72(CombinatorialOperationalNetworkV71):
    """V7.2 anonymous multilayer graph with algebraic/discrete feedback.

    All graph statistics are evaluated from each Monte Carlo path's current
    capacities before they affect protection, shipment, production and repair.
    They are diagnostics and state-transition inputs, never terminal-probability
    multipliers.
    """

    def __init__(self, paths: int, seed: int, terrain: dict, **kwargs) -> None:
        super().__init__(paths, seed, terrain, **kwargs)
        actor_scale = np.array([1.00, 0.72, 1.18, 0.92])[None, :, None]
        edge_scale = np.array(
            [0.66, 0.62, 0.54, 0.42, 0.46, 0.58, 0.50, 0.54, 0.64, 0.30, 0.32, 0.34]
        )[None, None, :]
        self.edge_base_capacity = np.repeat(actor_scale * edge_scale, paths, axis=0)
        self.edge_integrity = np.clip(
            self.rng.normal(0.94, 0.018, (paths, 4, len(EDGE_TAIL))), 0.84, 0.985
        )
        self.shipment_pipeline = np.zeros((paths, 4, 3, int(np.max(PATH_DELAYS))))
        self.last_route_flow = np.zeros((paths, 4, len(PATHS), 3))
        self.last_route_capacity = np.ones((paths, 4, len(PATHS)))
        self.last_edge_utilisation = np.zeros((paths, 4, len(EDGE_TAIL)))
        self.last_protection = np.zeros((paths, 4, len(EDGE_TAIL)), dtype=bool)
        self.last_effective_resistance = np.zeros((paths, 4, len(EDGE_TAIL)))
        self.last_edge_leverage = np.zeros((paths, 4, len(EDGE_TAIL)))
        self.last_kirchhoff = np.zeros((paths, 4))
        self.last_tree_logdet = np.zeros((paths, 4))
        self.last_directed_tree_logdet = np.zeros((paths, 4))
        self.last_nonbacktracking_radius = np.zeros((paths, 4))
        self.last_signed_frustration = np.zeros((paths, 4))
        self.last_signed_cohesion = np.ones((paths, 4))
        self.last_spectral_gap = np.zeros((paths, 4))
        self.last_coarsening_error = np.zeros((paths, 4))
        self.last_temporal_arrival = np.zeros((paths, 4))
        self.last_temporal_connectivity = np.zeros((paths, 4), dtype=int)
        self.last_graph_gate = np.ones((paths, 4))
        self.last_graph_metrics: dict[str, np.ndarray] = {}
        self.max_interlacing_violation_v72 = 0.0
        self.max_effective_resistance_symmetry_v72 = 0.0
        self.max_protection_conflict_v72 = 0
        self.max_protection_partition_v72 = 0
        self.minimum_temporal_connectivity_v72 = len(PATHS)
        self.maximum_kirchhoff_v72 = 0.0
        self.minimum_directed_tree_logdet_v72 = np.inf
        self.months_v72 = 0

    @staticmethod
    def _undirected_laplacian(edge_capacity: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        batch_shape = edge_capacity.shape[:-1]
        adjacency = np.zeros(batch_shape + (NODE_COUNT, NODE_COUNT))
        half = 0.5 * edge_capacity
        for edge, (tail, head) in enumerate(zip(EDGE_TAIL, EDGE_HEAD)):
            adjacency[..., tail, head] += half[..., edge]
            adjacency[..., head, tail] += half[..., edge]
        degree = np.sum(adjacency, axis=-1)
        laplacian = -adjacency
        diagonal = np.arange(NODE_COUNT)
        laplacian[..., diagonal, diagonal] = degree
        return laplacian, adjacency

    def _algebraic_metrics(self, edge_capacity: np.ndarray) -> dict[str, np.ndarray]:
        laplacian, adjacency = self._undirected_laplacian(edge_capacity)
        eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
        inverse = np.zeros_like(eigenvalues)
        np.divide(1.0, eigenvalues, out=inverse, where=eigenvalues > 1.0e-10)
        pseudoinverse = np.einsum(
            "...ik,...k,...jk->...ij", eigenvectors, inverse, eigenvectors
        )
        resistance = np.empty(edge_capacity.shape)
        for edge, (tail, head) in enumerate(zip(EDGE_TAIL, EDGE_HEAD)):
            resistance[..., edge] = (
                pseudoinverse[..., tail, tail]
                + pseudoinverse[..., head, head]
                - 2.0 * pseudoinverse[..., tail, head]
            )
        leverage = np.clip(0.5 * edge_capacity * resistance, 0.0, 1.0)
        kirchhoff = NODE_COUNT * np.sum(inverse, axis=-1)
        spectral_gap = eigenvalues[..., 1]

        sign, tree_logdet = np.linalg.slogdet(laplacian[..., 1:, 1:])
        tree_logdet = np.where(sign > 0, tree_logdet, -60.0)
        directed = np.zeros_like(laplacian)
        for edge, (tail, head) in enumerate(zip(EDGE_TAIL, EDGE_HEAD)):
            directed[..., tail, tail] += edge_capacity[..., edge]
            directed[..., tail, head] -= edge_capacity[..., edge]
        sign_d, directed_logdet = np.linalg.slogdet(directed[..., :-1, :-1])
        directed_logdet = np.where(sign_d > 0, directed_logdet, -60.0)

        signed_adjacency = np.zeros_like(adjacency)
        for edge, (tail, head) in enumerate(zip(EDGE_TAIL, EDGE_HEAD)):
            weight = 0.5 * EDGE_SIGN[edge] * edge_capacity[..., edge]
            signed_adjacency[..., tail, head] += weight
            signed_adjacency[..., head, tail] += weight
        signed_degree = np.sum(np.abs(signed_adjacency), axis=-1)
        signed_laplacian = -signed_adjacency
        diagonal = np.arange(NODE_COUNT)
        signed_laplacian[..., diagonal, diagonal] = signed_degree
        signed_eigen = np.linalg.eigvalsh(signed_laplacian)
        signed_frustration = np.maximum(signed_eigen[..., 0], 0.0)
        signed_cohesion = np.exp(
            -signed_frustration / np.maximum(np.mean(signed_degree, axis=-1), 1.0e-10)
        )

        membership = np.zeros((NODE_COUNT, 3))
        membership[[0, 1, 2], 0] = 1.0 / np.sqrt(3.0)
        membership[[3, 4, 5], 1] = 1.0 / np.sqrt(3.0)
        membership[6, 2] = 1.0
        coarse = np.einsum("in,...ij,jm->...nm", membership, laplacian, membership)
        coarse_eigen = np.linalg.eigvalsh(coarse)
        denom = np.sum(np.abs(eigenvalues[..., :3]), axis=-1) + 1.0e-10
        coarsening_error = np.sum(
            np.abs(eigenvalues[..., :3] - coarse_eigen), axis=-1
        ) / denom
        upper = eigenvalues[..., -3:]
        violation = np.maximum(eigenvalues[..., :3] - coarse_eigen, 0.0)
        violation = np.maximum(violation, np.maximum(coarse_eigen - upper, 0.0))
        self.max_interlacing_violation_v72 = max(
            self.max_interlacing_violation_v72, float(np.max(violation))
        )

        radius = self._nonbacktracking_radius(edge_capacity)
        return {
            "resistance": resistance,
            "leverage": leverage,
            "kirchhoff": kirchhoff,
            "spectral_gap": spectral_gap,
            "tree_logdet": tree_logdet,
            "directed_tree_logdet": directed_logdet,
            "nonbacktracking_radius": radius,
            "signed_frustration": signed_frustration,
            "signed_cohesion": signed_cohesion,
            "coarsening_error": coarsening_error,
        }

    @staticmethod
    def _nonbacktracking_radius(edge_capacity: np.ndarray) -> np.ndarray:
        tails = np.concatenate([EDGE_TAIL, EDGE_HEAD])
        heads = np.concatenate([EDGE_HEAD, EDGE_TAIL])
        weights = np.sqrt(np.concatenate([edge_capacity, edge_capacity], axis=-1))
        vector = np.ones(edge_capacity.shape[:-1] + (len(tails),))
        vector /= np.linalg.norm(vector, axis=-1, keepdims=True)
        radius = np.zeros(edge_capacity.shape[:-1])
        for _ in range(18):
            updated = np.zeros_like(vector)
            for incoming in range(len(tails)):
                mask = (tails == heads[incoming]) & (heads != tails[incoming])
                updated[..., mask] += vector[..., incoming, None] * weights[..., mask]
            norm = np.linalg.norm(updated, axis=-1)
            vector = updated / np.maximum(norm[..., None], 1.0e-12)
            radius = norm
        return radius

    def _select_protection(
        self, opening_integrity: np.ndarray, leverage: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if leverage is None:
            leverage = np.zeros_like(opening_integrity)
        risk = (
            0.34 * self.last_edge_utilisation
            + 0.24 * self.current_damage[..., None]
            + 0.18 * self.current_interdiction[..., None]
            + 0.14 * (1.0 - opening_integrity)
            + 0.10 * leverage
        )
        budget = np.clip(
            1 + (self.current_financial > 0.66).astype(int)
            + (self.current_command > 0.80).astype(int), 1, 3
        )
        scores = []
        for chosen in PROTECTION_SETS:
            if not chosen:
                scores.append(np.zeros((self.paths, 4)))
                continue
            chosen_list = list(chosen)
            covered_paths = np.sum(np.any(PATH_INCIDENCE[:, chosen_list] > 0, axis=1))
            score = np.sum(risk[..., chosen_list], axis=-1) + 0.025 * covered_paths
            score = np.where(len(chosen) <= budget, score, -np.inf)
            scores.append(score)
        best = np.argmax(np.stack(scores, axis=-1), axis=-1)
        protected = np.zeros_like(opening_integrity, dtype=bool)
        for index, chosen in enumerate(PROTECTION_SETS):
            if not chosen:
                continue
            mask = best == index
            protected[..., list(chosen)] |= mask[..., None]
        selected = np.sum(protected, axis=-1)
        self.protection_budget_violations += int(np.sum(selected > budget))
        for layer in range(4):
            count = np.sum(protected[..., EDGE_LAYER == layer], axis=-1)
            self.max_protection_partition_v72 = max(
                self.max_protection_partition_v72, int(np.max(np.maximum(count - 2, 0)))
            )
        for left, right in ((0, 1), (9, 10)):
            self.max_protection_conflict_v72 += int(
                np.sum(protected[..., left] & protected[..., right])
            )
        return protected, budget

    def _cascade_capacities(
        self, corridor: np.ndarray, supply: np.ndarray, need: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        opening = self.edge_integrity.copy()
        network_scale = np.clip(np.mean(corridor, axis=2), 0.02, 1.25)
        initial_capacity = self.edge_base_capacity * network_scale[..., None] * opening
        initial_metrics = self._algebraic_metrics(initial_capacity)
        protected, _ = self._select_protection(opening, initial_metrics["leverage"])
        contagion = np.clip(
            initial_metrics["nonbacktracking_radius"]
            / np.maximum(np.mean(initial_metrics["nonbacktracking_radius"]), 1.0e-10),
            0.70, 1.35,
        )
        cohesion = initial_metrics["signed_cohesion"]
        shock = (
            0.070 * self.current_damage[..., None]
            + 0.052 * self.current_interdiction[..., None]
            + 0.024 * (1.0 - self.current_pnt[..., None])
        ) * (1.0 - 0.48 * protected) * contagion[..., None]
        repair = (
            0.030 * self.current_command[..., None]
            * self.current_financial[..., None]
            * cohesion[..., None]
            * (1.0 - opening)
            * (1.0 + 0.30 * protected)
        )
        integrity = opening.copy()
        residual = np.inf
        edge_demand = np.einsum(
            "...pc,pe->...e", self.last_route_flow, PATH_INCIDENCE
        )
        opening_capacity = self.edge_base_capacity * network_scale[..., None] * opening
        opening_utilisation = edge_demand / np.maximum(opening_capacity, 1.0e-12)
        overload = 0.050 * np.maximum(opening_utilisation - 0.82, 0.0)
        base_target = opening + repair - shock - overload
        for _ in range(160):
            neighbor_failure = np.einsum(
                "...e,fe->...f", 1.0 - integrity, EDGE_DEPENDENCY
            )
            target = np.clip(base_target - 0.035 * neighbor_failure, 0.04, 0.995)
            updated = 0.70 * integrity + 0.30 * target
            residual = float(np.max(np.abs(updated - integrity)))
            integrity = updated
            if residual < 1.0e-8:
                break
        if residual >= 1.0e-8:
            self.cascade_nonconvergence_months += 1
        self.max_cascade_residual_v71 = max(self.max_cascade_residual_v71, residual)
        self.edge_integrity = integrity
        self.last_protection = protected
        edge_capacity = self.edge_base_capacity * network_scale[..., None] * integrity
        metrics = self._algebraic_metrics(edge_capacity)
        self.last_graph_metrics = metrics
        self.last_effective_resistance = metrics["resistance"]
        self.last_edge_leverage = metrics["leverage"]
        self.last_kirchhoff = metrics["kirchhoff"]
        self.last_tree_logdet = metrics["tree_logdet"]
        self.last_directed_tree_logdet = metrics["directed_tree_logdet"]
        self.last_nonbacktracking_radius = metrics["nonbacktracking_radius"]
        self.last_signed_frustration = metrics["signed_frustration"]
        self.last_signed_cohesion = metrics["signed_cohesion"]
        self.last_spectral_gap = metrics["spectral_gap"]
        self.last_coarsening_error = metrics["coarsening_error"]
        self.maximum_kirchhoff_v72 = max(
            self.maximum_kirchhoff_v72, float(np.max(metrics["kirchhoff"]))
        )
        self.minimum_directed_tree_logdet_v72 = min(
            self.minimum_directed_tree_logdet_v72,
            float(np.min(metrics["directed_tree_logdet"])),
        )
        path_capacity = np.min(
            edge_capacity[..., None, :] + (1.0 - PATH_INCIDENCE)[None, None, :, :] * 1.0e6,
            axis=-1,
        )
        return edge_capacity, path_capacity

    @staticmethod
    def _minimum_cut(edge_capacity: np.ndarray) -> np.ndarray:
        candidates = [np.sum(edge_capacity[..., mask], axis=-1) for mask in CUT_MASKS]
        return np.min(np.stack(candidates, axis=-1), axis=-1)

    def _temporal_metrics(self, edge_capacity: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        penalty = (self.edge_integrity < 0.58).astype(int) + (
            self.edge_integrity < 0.30
        ).astype(int)
        delays = EDGE_DELAY[None, None, :] + penalty
        arrival = np.full((self.paths, 4, NODE_COUNT), 10_000, dtype=int)
        arrival[..., SOURCE] = 0
        for node in range(NODE_COUNT):
            for edge in np.flatnonzero(EDGE_TAIL == node):
                head = EDGE_HEAD[edge]
                candidate = arrival[..., node] + delays[..., edge]
                candidate = np.where(edge_capacity[..., edge] > 1.0e-6, candidate, 10_000)
                arrival[..., head] = np.minimum(arrival[..., head], candidate)
        deadline = 6
        feasible_path = np.ones((self.paths, 4, len(PATHS)), dtype=bool)
        for path_index, path in enumerate(PATHS):
            path_list = list(path)
            feasible_path[..., path_index] = (
                np.sum(delays[..., path_list], axis=-1) <= deadline
            ) & np.all(edge_capacity[..., path_list] > 1.0e-6, axis=-1)
        connectivity = np.zeros((self.paths, 4), dtype=int)
        for chosen in DISJOINT_PATH_SETS:
            feasible = np.all(feasible_path[..., list(chosen)], axis=-1)
            connectivity = np.maximum(connectivity, feasible.astype(int) * len(chosen))
        return arrival[..., SINK].astype(float), connectivity

    def _allocate_multicommodity_flows(
        self, demand: np.ndarray, corridor: np.ndarray,
        front_priority: np.ndarray,
    ) -> np.ndarray:
        supply = self.inventory[:, :, 1, :] + self.inventory[:, :, 2, :]
        need = np.maximum(demand + 0.55 - self.inventory[:, :, 0, :], 0.0)
        if self.geometry_enabled:
            need *= front_priority[..., None]
        edge_capacity, path_capacity = self._cascade_capacities(corridor, supply, need)
        residual_edge = edge_capacity.copy()
        remaining_supply = np.maximum(supply, 0.0).copy()
        remaining_need = np.maximum(need, 0.0).copy()
        plan = np.zeros((self.paths, 4, len(PATHS), 3))
        for commodity in (0, 1, 2):
            for path_index, path in enumerate(PATHS):
                path_list = list(path)
                bottleneck = np.min(residual_edge[..., path_list], axis=-1)
                amount = np.minimum(
                    np.minimum(remaining_supply[..., commodity], remaining_need[..., commodity]),
                    bottleneck,
                )
                plan[..., path_index, commodity] = amount
                remaining_supply[..., commodity] -= amount
                remaining_need[..., commodity] -= amount
                residual_edge[..., path_list] -= amount[..., None]

        total_by_commodity = np.sum(plan, axis=2)
        from_transit = np.minimum(self.inventory[:, :, 1, :], total_by_commodity)
        from_depth = total_by_commodity - from_transit
        self.inventory[:, :, 1, :] -= from_transit
        self.inventory[:, :, 2, :] -= from_depth
        for path_index, delay in enumerate(PATH_DELAYS):
            self.shipment_pipeline[..., delay - 1] += plan[..., path_index, :]

        edge_load = np.einsum("...pc,pe->...e", plan, PATH_INCIDENCE)
        self.last_edge_utilisation = edge_load / np.maximum(edge_capacity, 1.0e-12)
        self.last_route_flow = plan
        self.last_route_capacity = path_capacity
        scheduled = np.sum(total_by_commodity, axis=-1)
        requested = np.sum(need, axis=-1)
        arrival, temporal_connectivity = self._temporal_metrics(edge_capacity)
        self.last_temporal_arrival = arrival
        self.last_temporal_connectivity = temporal_connectivity
        self.minimum_temporal_connectivity_v72 = min(
            self.minimum_temporal_connectivity_v72, int(np.min(temporal_connectivity))
        )
        temporal_factor = np.clip(temporal_connectivity / 2.0, 0.35, 1.0)
        self.last_delivery_ratio = np.clip(
            scheduled / np.maximum(requested, 1.0e-12) * temporal_factor, 0.0, 1.0
        )
        weights = np.array([1.18, 1.08, 0.92])[None, None, :]
        unmet = np.maximum(need - total_by_commodity, 0.0)
        self.last_shadow_scarcity = np.clip(
            np.sum(weights * unmet, axis=-1)
            / np.maximum(np.sum(weights * need, axis=-1), 1.0e-12), 0.0, 1.0
        )

        min_cut = self._minimum_cut(edge_capacity)
        self.last_min_cut = min_cut
        perturbed_cut = self._minimum_cut(edge_capacity + 1.0e-6)
        self.max_monotonicity_violation_v71 = max(
            self.max_monotonicity_violation_v71,
            float(np.max(np.maximum(min_cut - perturbed_cut, 0.0))),
        )
        self.max_flow_cut_residual_v71 = max(
            self.max_flow_cut_residual_v71,
            float(np.max(np.maximum(np.sum(plan, axis=(2, 3)) - min_cut, 0.0))),
        )
        active = edge_capacity > 1.0e-6
        unit_capacity = active.astype(float)
        connectivity = self._minimum_cut(unit_capacity).astype(int)
        self.last_edge_connectivity = connectivity
        self.last_node_connectivity = np.minimum(connectivity, 2)
        self.minimum_connectivity_v71 = min(
            self.minimum_connectivity_v71, int(np.min(connectivity))
        )
        capacity_residual = np.maximum(edge_load - edge_capacity, 0.0)
        self.max_capacity_residual_v71 = max(
            self.max_capacity_residual_v71, float(np.max(capacity_residual))
        )
        self.predicted_consumed = np.minimum(self.inventory[:, :, 0, :], demand)
        return total_by_commodity

    def update(self, **kwargs):
        # Previous-month graph state is part of the current production and
        # command transition.  The first month starts neutrally at one.
        if self.months_v72 == 0:
            graph_gate = np.ones((self.paths, 4))
        else:
            tree_scale = 1.0 / (1.0 + np.exp(-(self.last_tree_logdet + 10.0) / 4.0))
            temporal_scale = np.clip(self.last_temporal_connectivity / 2.0, 0.0, 1.0)
            graph_gate = np.clip(
                0.74 + 0.10 * self.last_signed_cohesion
                + 0.08 * temporal_scale + 0.08 * tree_scale,
                0.70, 1.0,
            )
        self.last_graph_gate = graph_gate
        args = dict(kwargs)
        args["production"] = np.asarray(args["production"]) * graph_gate
        args["command"] = np.asarray(args["command"]) * (
            0.92 + 0.08 * self.last_signed_cohesion
        )
        result = super().update(**args)
        self.months_v72 += 1
        return result

    def diagnostics(self) -> dict:
        base = super().diagnostics()
        base["combinatorial_network_v71"]["exact_series_parallel_maxflow_mincut"] = False
        base["combinatorial_network_v71"]["v72_general_graph_replacement"] = True
        base["algebraic_discrete_network_v72"] = {
            "months": self.months_v72,
            "nodes": NODE_COUNT,
            "directed_edges": len(EDGE_TAIL),
            "simple_paths": len(PATHS),
            "exact_treewidth": TREEWIDTH,
            "solver_branch": "exact-enumeration" if TREEWIDTH <= 4 else "decomposition",
            "effective_resistance_active": True,
            "matrix_tree_active": True,
            "directed_arborescence_active": True,
            "nonbacktracking_active": True,
            "signed_balance_active": True,
            "spectral_coarsening_active": True,
            "temporal_paths_active": True,
            "survivable_design_active": True,
            "matroid_conflict_schedule_active": True,
            "mean_kirchhoff_index": float(np.mean(self.last_kirchhoff)),
            "maximum_kirchhoff_index": self.maximum_kirchhoff_v72,
            "mean_tree_logdet": float(np.mean(self.last_tree_logdet)),
            "minimum_directed_tree_logdet": self.minimum_directed_tree_logdet_v72,
            "mean_nonbacktracking_radius": float(np.mean(self.last_nonbacktracking_radius)),
            "mean_signed_frustration": float(np.mean(self.last_signed_frustration)),
            "mean_signed_cohesion": float(np.mean(self.last_signed_cohesion)),
            "mean_spectral_gap": float(np.mean(self.last_spectral_gap)),
            "mean_coarsening_error": float(np.mean(self.last_coarsening_error)),
            "maximum_interlacing_violation": self.max_interlacing_violation_v72,
            "minimum_temporal_connectivity": self.minimum_temporal_connectivity_v72,
            "mean_temporal_arrival": float(np.mean(self.last_temporal_arrival)),
            "maximum_protection_partition_violation": self.max_protection_partition_v72,
            "protection_conflict_violations": self.max_protection_conflict_v72,
            "mean_graph_gate": float(np.mean(self.last_graph_gate)),
            "terminal_probability_adjustment": False,
        }
        return base
