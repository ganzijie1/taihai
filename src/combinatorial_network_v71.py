from __future__ import annotations

import numpy as np

from operational_constraints_v47 import DynamicOperationalConstraintSystem


ROUTE_EDGES = np.array([[0, 1], [2, 3], [4, 5]], dtype=int)
ROUTE_DELAYS = np.array([1, 2, 3], dtype=int)
COMMODITY_PRIORITY = (0, 1, 2)


class CombinatorialOperationalNetworkV71(DynamicOperationalConstraintSystem):
    """Time-expanded multicommodity, survivability and repair network.

    The aggregate graph has a super-source, three internally disjoint route
    channels and a front sink.  Every route contains two serial edges.  This
    makes max-flow/min-cut and Menger connectivity exact while retaining a
    vectorised solver suitable for every Monte Carlo path and monthly clock.
    The graph is deliberately anonymous and aggregate.
    """

    def __init__(self, paths: int, seed: int, terrain: dict, **kwargs) -> None:
        super().__init__(paths, seed, terrain, **kwargs)
        actor_scale = np.array([1.00, 0.72, 1.18, 0.92])[None, :, None]
        edge_scale = np.array([0.82, 0.74, 0.68, 0.66, 0.52, 0.58])[None, None, :]
        self.edge_base_capacity = np.repeat(
            actor_scale * edge_scale, paths, axis=0
        )
        self.edge_integrity = np.clip(
            self.rng.normal(0.94, 0.018, (paths, 4, 6)), 0.84, 0.985
        )
        self.shipment_pipeline = np.zeros((paths, 4, 3, 3))
        self.last_route_flow = np.zeros((paths, 4, 3, 3))
        self.last_route_capacity = np.ones((paths, 4, 3))
        self.last_edge_utilisation = np.zeros((paths, 4, 6))
        self.last_delivery_ratio = np.ones((paths, 4))
        self.last_shadow_scarcity = np.zeros((paths, 4))
        self.last_min_cut = np.ones((paths, 4))
        self.last_edge_connectivity = np.full((paths, 4), 3, dtype=int)
        self.last_node_connectivity = np.full((paths, 4), 3, dtype=int)
        self.last_protection = np.zeros((paths, 4, 6), dtype=bool)
        self.last_matching_flow = np.zeros((paths, 4, 3, 3))
        self.current_damage = np.zeros((paths, 4))
        self.current_interdiction = np.zeros((paths, 4))
        self.current_command = np.ones((paths, 4))
        self.current_pnt = np.ones((paths, 4))
        self.current_financial = np.ones((paths, 4))
        self.current_consumption = np.zeros((paths, 4))
        self.predicted_consumed = np.zeros((paths, 4, 3))
        self.max_capacity_residual_v71 = 0.0
        self.max_stock_residual_v71 = 0.0
        self.max_matching_residual_v71 = 0.0
        self.max_flow_cut_residual_v71 = 0.0
        self.max_cascade_residual_v71 = 0.0
        self.max_monotonicity_violation_v71 = 0.0
        self.max_pipeline_negative_v71 = 0.0
        self.minimum_connectivity_v71 = 3
        self.cascade_nonconvergence_months = 0
        self.protection_budget_violations = 0
        self.hypergraph_gate_minimum = 1.0
        self.months_v71 = 0

    def _advance_pipeline(self) -> None:
        arrivals = self.shipment_pipeline[..., 0].copy()
        self.inventory[:, :, 0, :] += arrivals
        self.shipment_pipeline[..., :-1] = self.shipment_pipeline[..., 1:]
        self.shipment_pipeline[..., -1] = 0.0

    @staticmethod
    def _route_capacity(edge_capacity: np.ndarray) -> np.ndarray:
        return np.minimum(
            edge_capacity[..., ROUTE_EDGES[:, 0]],
            edge_capacity[..., ROUTE_EDGES[:, 1]],
        )

    @staticmethod
    def _plan_routes(
        supply: np.ndarray, need: np.ndarray, route_capacity: np.ndarray,
    ) -> np.ndarray:
        """Priority-optimal flow for equal edge use on parallel serial routes."""
        remaining_supply = np.maximum(supply, 0.0).copy()
        remaining_need = np.maximum(need, 0.0).copy()
        remaining_capacity = np.maximum(route_capacity, 0.0).copy()
        allocation = np.zeros(route_capacity.shape + (3,))
        for route in range(3):
            for commodity in COMMODITY_PRIORITY:
                amount = np.minimum(
                    np.minimum(
                        remaining_supply[..., commodity],
                        remaining_need[..., commodity],
                    ),
                    remaining_capacity[..., route],
                )
                allocation[..., route, commodity] = amount
                remaining_supply[..., commodity] -= amount
                remaining_need[..., commodity] -= amount
                remaining_capacity[..., route] -= amount
        return allocation

    def _select_protection(
        self, opening_integrity: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        risk = (
            0.52 * self.last_edge_utilisation
            + 0.30 * self.current_damage[..., None]
            + 0.18 * self.current_interdiction[..., None]
            + 0.24 * (1.0 - opening_integrity)
        )
        route_risk = risk.reshape(self.paths, 4, 3, 2)
        best_local_edge = np.argmax(route_risk, axis=-1)
        best_route_score = np.max(route_risk, axis=-1)
        budget = (
            1
            + (self.current_financial > 0.66).astype(int)
            + (self.current_command > 0.80).astype(int)
        )
        budget = np.clip(budget, 1, 3)
        rank = np.argsort(np.argsort(-best_route_score, axis=-1), axis=-1)
        chosen_route = rank < budget[..., None]
        protected = np.zeros_like(opening_integrity, dtype=bool)
        for route in range(3):
            edge = 2 * route + best_local_edge[..., route]
            mask = chosen_route[..., route]
            path_index, actor_index = np.nonzero(mask)
            protected[
                path_index, actor_index, edge[path_index, actor_index]
            ] = True
        selected = np.sum(protected, axis=-1)
        self.protection_budget_violations += int(np.sum(selected > budget))
        return protected, budget

    def _cascade_capacities(
        self, corridor: np.ndarray, supply: np.ndarray, need: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        opening = self.edge_integrity.copy()
        protected, _ = self._select_protection(opening)
        network_scale = np.clip(np.mean(corridor, axis=2), 0.02, 1.25)
        shock = (
            0.075 * self.current_damage[..., None]
            + 0.055 * self.current_interdiction[..., None]
            + 0.025 * (1.0 - self.current_pnt[..., None])
        ) * (1.0 - 0.48 * protected)
        repair = (
            0.030 * self.current_command[..., None]
            * self.current_financial[..., None]
            * (1.0 - opening)
            * (1.0 + 0.30 * protected)
        )
        integrity = opening.copy()
        residual = np.inf
        for _ in range(32):
            edge_capacity = (
                self.edge_base_capacity * network_scale[..., None] * integrity
            )
            route_capacity = self._route_capacity(edge_capacity)
            plan = self._plan_routes(supply, need, route_capacity)
            route_load = np.sum(plan, axis=-1)
            utilisation = route_load / np.maximum(route_capacity, 1.0e-12)
            edge_utilisation = np.repeat(utilisation, 2, axis=-1)
            overload = 0.055 * np.maximum(edge_utilisation - 0.82, 0.0)
            target = np.clip(opening + repair - shock - overload, 0.04, 0.995)
            updated = 0.55 * integrity + 0.45 * target
            residual = float(np.max(np.abs(updated - integrity)))
            integrity = updated
            if residual < 1.0e-8:
                break
        if residual >= 1.0e-8:
            self.cascade_nonconvergence_months += 1
        self.max_cascade_residual_v71 = max(
            self.max_cascade_residual_v71, residual
        )
        self.edge_integrity = integrity
        self.last_protection = protected
        edge_capacity = (
            self.edge_base_capacity * network_scale[..., None] * integrity
        )
        return edge_capacity, self._route_capacity(edge_capacity)

    def _allocate_multicommodity_flows(
        self, demand: np.ndarray, corridor: np.ndarray,
        front_priority: np.ndarray,
    ) -> np.ndarray:
        supply = self.inventory[:, :, 1, :] + self.inventory[:, :, 2, :]
        need = np.maximum(
            demand + 0.55 - self.inventory[:, :, 0, :], 0.0
        )
        if self.geometry_enabled:
            need *= front_priority[..., None]
        edge_capacity, route_capacity = self._cascade_capacities(
            corridor, supply, need
        )
        plan = self._plan_routes(supply, need, route_capacity)
        total_by_commodity = np.sum(plan, axis=2)
        from_transit = np.minimum(
            self.inventory[:, :, 1, :], total_by_commodity
        )
        from_depth = total_by_commodity - from_transit
        self.inventory[:, :, 1, :] -= from_transit
        self.inventory[:, :, 2, :] -= from_depth
        for route, delay in enumerate(ROUTE_DELAYS):
            self.shipment_pipeline[..., delay - 1] += plan[..., route, :]

        route_load = np.sum(plan, axis=-1)
        utilisation = route_load / np.maximum(route_capacity, 1.0e-12)
        self.last_edge_utilisation = np.repeat(utilisation, 2, axis=-1)
        self.last_route_flow = plan
        self.last_route_capacity = route_capacity
        scheduled = np.sum(total_by_commodity, axis=-1)
        requested = np.sum(need, axis=-1)
        self.last_delivery_ratio = np.clip(
            scheduled / np.maximum(requested, 1.0e-12), 0.0, 1.0
        )
        weights = np.array([1.18, 1.08, 0.92])[None, None, :]
        unmet = np.maximum(need - total_by_commodity, 0.0)
        self.last_shadow_scarcity = np.clip(
            np.sum(weights * unmet, axis=-1)
            / np.maximum(np.sum(weights * need, axis=-1), 1.0e-12),
            0.0, 1.0,
        )

        min_cut = np.sum(route_capacity, axis=-1)
        cut_candidates = []
        for mask_code in range(8):
            selected = np.array([
                ROUTE_EDGES[route, (mask_code >> route) & 1]
                for route in range(3)
            ])
            cut_candidates.append(np.sum(edge_capacity[..., selected], axis=-1))
        enumerated_min_cut = np.min(np.stack(cut_candidates, axis=-1), axis=-1)
        self.max_flow_cut_residual_v71 = max(
            self.max_flow_cut_residual_v71,
            float(np.max(np.abs(min_cut - enumerated_min_cut))),
        )
        active_routes = route_capacity > 1.0e-6
        connectivity = np.sum(active_routes, axis=-1)
        self.last_edge_connectivity = connectivity
        self.last_node_connectivity = connectivity.copy()
        self.minimum_connectivity_v71 = min(
            self.minimum_connectivity_v71, int(np.min(connectivity))
        )
        self.last_min_cut = min_cut
        perturbed_cut = np.sum(self._route_capacity(edge_capacity + 1.0e-6), axis=-1)
        self.max_monotonicity_violation_v71 = max(
            self.max_monotonicity_violation_v71,
            float(np.max(np.maximum(min_cut - perturbed_cut, 0.0))),
        )
        capacity_residual = np.maximum(route_load - route_capacity, 0.0)
        self.max_capacity_residual_v71 = max(
            self.max_capacity_residual_v71, float(np.max(capacity_residual))
        )
        self.predicted_consumed = np.minimum(
            self.inventory[:, :, 0, :], demand
        )
        return total_by_commodity

    def _allocate_repairs(
        self, failures: np.ndarray, command: np.ndarray,
        financial: np.ndarray,
    ) -> np.ndarray:
        self.repair_queue_state += failures
        local_spares = np.mean(self.spares[:, :, 0, :], axis=2)
        total_capacity = np.clip(
            0.14 * local_spares * command * financial
            * np.mean(self.facility_coverage)
            * (0.70 + 0.30 * self.last_delivery_ratio),
            0.0, 0.18,
        )
        resource = total_capacity[..., None] * np.array([0.42, 0.34, 0.24])
        task = self.repair_queue_state.copy()
        utility = np.array([
            [1.00, 0.62, 0.35],
            [0.72, 0.94, 0.58],
            [0.42, 0.68, 0.90],
        ])
        edges = sorted(
            ((utility[i, j], i, j) for i in range(3) for j in range(3)),
            reverse=True,
        )
        matching = np.zeros((self.paths, 4, 3, 3))
        for _, crew, echelon in edges:
            amount = np.minimum(resource[..., crew], task[..., echelon])
            matching[..., crew, echelon] = amount
            resource[..., crew] -= amount
            task[..., echelon] -= amount
        repaired = np.sum(matching, axis=2)
        self.repair_queue_state -= repaired
        self.last_matching_flow = matching
        matching_residual = max(
            float(np.max(np.maximum(-resource, 0.0))),
            float(np.max(np.maximum(-task, 0.0))),
        )
        self.max_matching_residual_v71 = max(
            self.max_matching_residual_v71, matching_residual
        )
        return repaired

    def update(
        self, *, production: np.ndarray, consumption: np.ndarray,
        damage: np.ndarray, interdiction: np.ndarray, command: np.ndarray,
        pnt: np.ndarray, financial_capacity: np.ndarray,
    ):
        opening_total = np.sum(self.inventory, axis=(2, 3)) + np.sum(
            self.shipment_pipeline, axis=(2, 3)
        )
        self._advance_pipeline()
        self.current_damage = np.clip(damage, 0.0, 1.0)
        self.current_interdiction = np.clip(interdiction, 0.0, 1.0)
        self.current_command = np.clip(command, 0.0, 1.0)
        self.current_pnt = np.clip(pnt, 0.0, 1.0)
        self.current_financial = np.clip(financial_capacity, 0.05, 1.0)
        self.current_consumption = np.clip(consumption, 0.0, 0.20)

        # Hyperedge complementarity: production requires command, PNT,
        # finance, network integrity and recent delivery simultaneously.
        complement = np.minimum.reduce([
            self.current_command,
            self.current_pnt,
            self.current_financial,
            np.mean(self.edge_integrity, axis=-1),
            np.clip(0.30 + 0.70 * self.last_delivery_ratio, 0.0, 1.0),
        ])
        hypergraph_gate = np.clip(0.58 + 0.42 * complement, 0.58, 1.0)
        self.hypergraph_gate_minimum = min(
            self.hypergraph_gate_minimum, float(np.min(hypergraph_gate))
        )
        effective_production = np.asarray(production) * hypergraph_gate
        result = super().update(
            production=effective_production,
            consumption=consumption,
            damage=damage,
            interdiction=interdiction,
            command=command,
            pnt=pnt,
            financial_capacity=financial_capacity,
        )

        closing_total = np.sum(self.inventory, axis=(2, 3)) + np.sum(
            self.shipment_pipeline, axis=(2, 3)
        )
        produced_total = np.clip(effective_production, 0.0, 0.18)
        consumed_total = np.sum(self.predicted_consumed, axis=2)
        expected = (
            opening_total + produced_total - consumed_total
            - self.last_inventory_spill
        )
        residual = float(np.max(np.abs(closing_total - expected)))
        self.max_stock_residual_v71 = max(
            self.max_stock_residual_v71, residual
        )
        self.max_stock_residual = self.max_stock_residual_v71
        self.max_pipeline_negative_v71 = max(
            self.max_pipeline_negative_v71,
            float(np.max(np.maximum(-self.shipment_pipeline, 0.0))),
        )
        self.months_v71 += 1
        return self.current()

    def diagnostics(self) -> dict:
        base = super().diagnostics()
        base["combinatorial_network_v71"] = {
            "months": self.months_v71,
            "time_expanded_pipeline": True,
            "multicommodity_shared_capacity": True,
            "exact_series_parallel_maxflow_mincut": True,
            "menger_edge_and_node_connectivity": True,
            "fractional_b_matching_repair": True,
            "partition_matroid_protection": True,
            "hypergraph_complementarity_gate": True,
            "cascade_fixed_point": True,
            "maximum_capacity_residual": self.max_capacity_residual_v71,
            "maximum_stock_residual": self.max_stock_residual_v71,
            "maximum_matching_residual": self.max_matching_residual_v71,
            "maximum_maxflow_mincut_residual": self.max_flow_cut_residual_v71,
            "maximum_cascade_residual": self.max_cascade_residual_v71,
            "maximum_monotonicity_violation": self.max_monotonicity_violation_v71,
            "maximum_pipeline_negative_mass": self.max_pipeline_negative_v71,
            "minimum_edge_connectivity": self.minimum_connectivity_v71,
            "minimum_hypergraph_gate": self.hypergraph_gate_minimum,
            "mean_delivery_ratio": float(np.mean(self.last_delivery_ratio)),
            "mean_shadow_scarcity": float(np.mean(self.last_shadow_scarcity)),
            "minimum_cut_capacity": float(np.min(self.last_min_cut)),
            "cascade_nonconvergence_months": self.cascade_nonconvergence_months,
            "protection_budget_violations": self.protection_budget_violations,
            "terminal_probability_adjustment": False,
        }
        return base
