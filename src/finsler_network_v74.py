from __future__ import annotations

import numpy as np

from algebraic_discrete_network_v72 import (
    CUT_MASKS,
    DISJOINT_PATH_SETS,
    EDGE_DELAY,
    EDGE_HEAD,
    EDGE_TAIL,
    NODE_COUNT,
    PATH_INCIDENCE,
    PATHS,
    SINK,
    SOURCE,
)
from continuous_graph_network_v73 import ContinuousGraphOperationalNetworkV73


MAX_PIPELINE_DELAY = 12
EDGE_LENGTH = np.array(
    [1.00, 1.08, 1.04, 1.18, 1.12, 1.02, 1.06, 1.10, 1.00, 1.28, 1.30, 1.24]
)
EDGE_SLOPE = np.array(
    [0.04, 0.10, 0.08, 0.17, 0.14, 0.06, 0.09, 0.12, 0.04, 0.20, 0.22, 0.18]
)
EDGE_BACKGROUND_BIAS = np.array(
    [0.05, -0.04, 0.07, -0.09, 0.08, -0.03, 0.06, -0.05, 0.03, 0.10, -0.08, 0.07]
)


class FinslerOperationalNetworkV74(ContinuousGraphOperationalNetworkV73):
    """Pathwise Randers travel cost coupled to hard flow and the social MFG.

    The graph is deliberately anonymous and aggregate.  The Finsler layer owns
    travel time, generalized cost and reliability only.  Physical inventories,
    edge capacities and queues remain owned by the inherited discrete solvers.
    """

    def __init__(self, paths: int, seed: int, terrain: dict, **kwargs) -> None:
        super().__init__(paths, seed, terrain, **kwargs)
        edges = len(EDGE_TAIL)
        self.shipment_pipeline = np.zeros((paths, 4, 3, MAX_PIPELINE_DELAY))
        self.metric_uncertainty = self.rng.lognormal(0.0, 0.045, (paths, 4, edges))
        self.background_bias_draw = np.clip(
            EDGE_BACKGROUND_BIAS[None, None, :]
            + self.rng.normal(0.0, 0.012, (paths, 4, edges)),
            -0.18,
            0.18,
        )
        self.last_finsler_edge_cost = np.broadcast_to(
            EDGE_LENGTH[None, None, :], (paths, 4, edges)
        ).copy()
        self.last_reverse_edge_cost = self.last_finsler_edge_cost.copy()
        self.last_robust_edge_cost = self.last_finsler_edge_cost.copy()
        self.last_finsler_path_cost = np.ones((paths, 4, len(PATHS)))
        self.last_finsler_path_reliability = np.ones_like(self.last_finsler_path_cost)
        self.last_finsler_node_arrival = np.zeros((paths, 4, NODE_COUNT))
        self.last_finsler_sink_arrival = np.ones((paths, 4))
        self.last_mfg_mobility_cost = np.ones((paths, 4))
        self.minimum_finsler_cost = np.inf
        self.maximum_randers_norm = 0.0
        self.maximum_direction_asymmetry = 0.0
        self.maximum_zero_drift_error = 0.0
        self.maximum_homogeneity_residual = 0.0
        self.maximum_eikonal_residual = 0.0
        self.maximum_finsler_capacity_residual = 0.0
        self.minimum_path_reliability = 1.0
        self.months_v74 = 0

    @staticmethod
    def _randers_pair(
        symmetric_cost: np.ndarray, one_form_ratio: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        ratio = np.clip(one_form_ratio, -0.95, 0.95)
        return symmetric_cost * (1.0 + ratio), symmetric_cost * (1.0 - ratio)

    def _build_finsler_metric(
        self, edge_capacity: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        signal = np.mean(self.edge_signal, axis=-1)
        control = np.mean(self.edge_control, axis=-1)
        seasonal = np.sin(2.0 * np.pi * self.months_v74 / 12.0)
        disruption = (
            0.40 * self.current_damage[..., None]
            + 0.34 * self.current_interdiction[..., None]
            + 0.20 * (1.0 - self.current_command[..., None])
            + 0.18 * (1.0 - self.current_pnt[..., None])
        )
        terrain_metric = 1.0 + 0.70 * EDGE_SLOPE[None, None, :] ** 2
        continuum = np.sqrt(np.maximum(self.last_continuous_delay_factor, 1.0))
        symmetric = (
            EDGE_LENGTH[None, None, :]
            * np.sqrt(terrain_metric)
            * (1.0 + disruption)
            * continuum
            * self.metric_uncertainty
        )
        one_form_ratio = self.background_bias_draw * (0.78 + 0.22 * seasonal)
        one_form_ratio *= 0.78 + 0.22 * self.current_pnt[..., None]
        forward, reverse = self._randers_pair(symmetric, one_form_ratio)
        reliability = np.clip(
            self.edge_integrity
            * self.last_continuous_capacity_gate
            * (0.58 + 0.22 * signal + 0.20 * control),
            0.02,
            1.0,
        )
        forward = np.maximum(
            forward * (1.0 + 0.30 * (1.0 - reliability)), 1.0e-8
        )
        reverse = np.maximum(
            reverse * (1.0 + 0.30 * (1.0 - reliability)), 1.0e-8
        )
        robust = forward * (1.0 + 0.10 * np.abs(self.metric_uncertainty - 1.0))
        forward = np.where(edge_capacity > 1.0e-8, forward, np.inf)
        reverse = np.where(edge_capacity > 1.0e-8, reverse, np.inf)
        robust = np.where(edge_capacity > 1.0e-8, robust, np.inf)

        finite = np.isfinite(forward)
        if np.any(finite):
            self.minimum_finsler_cost = min(
                self.minimum_finsler_cost, float(np.min(forward[finite]))
            )
            self.maximum_direction_asymmetry = max(
                self.maximum_direction_asymmetry,
                float(np.max(np.abs(forward[finite] - reverse[finite]))),
            )
        self.maximum_randers_norm = max(
            self.maximum_randers_norm, float(np.max(np.abs(one_form_ratio)))
        )
        zero_forward, _ = self._randers_pair(symmetric, np.zeros_like(one_form_ratio))
        self.maximum_zero_drift_error = max(
            self.maximum_zero_drift_error,
            float(np.max(np.abs(zero_forward - symmetric))),
        )
        alpha = 1.7
        scaled, _ = self._randers_pair(alpha * symmetric, one_form_ratio)
        base_forward, _ = self._randers_pair(symmetric, one_form_ratio)
        self.maximum_homogeneity_residual = max(
            self.maximum_homogeneity_residual,
            float(np.max(np.abs(scaled - alpha * base_forward))),
        )
        self.last_finsler_edge_cost = forward
        self.last_reverse_edge_cost = reverse
        self.last_robust_edge_cost = robust
        return forward, robust, reliability

    @staticmethod
    def _shortest_arrival(
        edge_cost: np.ndarray,
    ) -> np.ndarray:
        arrival = np.full(edge_cost.shape[:-1] + (NODE_COUNT,), np.inf)
        arrival[..., SOURCE] = 0.0
        for node in range(NODE_COUNT):
            for edge in np.flatnonzero(EDGE_TAIL == node):
                head = EDGE_HEAD[edge]
                arrival[..., head] = np.minimum(
                    arrival[..., head], arrival[..., node] + edge_cost[..., edge]
                )
        return arrival

    def _path_geometry(
        self, edge_cost: np.ndarray, edge_reliability: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        path_cost = np.empty(edge_cost.shape[:-1] + (len(PATHS),))
        path_reliability = np.ones(path_cost.shape)
        for path_index, path in enumerate(PATHS):
            path_list = list(path)
            path_cost[..., path_index] = np.sum(
                edge_cost[..., path_list], axis=-1
            )
            path_reliability[..., path_index] = np.prod(
                edge_reliability[..., path_list], axis=-1
            ) ** (1.0 / len(path_list))
        arrival = self._shortest_arrival(edge_cost)
        edge_residual = np.zeros(edge_cost.shape)
        for edge, (tail, head) in enumerate(zip(EDGE_TAIL, EDGE_HEAD)):
            edge_residual[..., edge] = np.maximum(
                arrival[..., head] - arrival[..., tail] - edge_cost[..., edge], 0.0
            )
        finite = np.isfinite(edge_residual)
        if np.any(finite):
            self.maximum_eikonal_residual = max(
                self.maximum_eikonal_residual,
                float(np.max(edge_residual[finite])),
            )
        self.last_finsler_path_cost = path_cost
        self.last_finsler_path_reliability = path_reliability
        self.last_finsler_node_arrival = arrival
        self.last_finsler_sink_arrival = arrival[..., SINK]
        self.minimum_path_reliability = min(
            self.minimum_path_reliability, float(np.min(path_reliability))
        )
        return path_cost, path_reliability, arrival

    def _allocate_multicommodity_flows(
        self, demand: np.ndarray, corridor: np.ndarray,
        front_priority: np.ndarray,
    ) -> np.ndarray:
        supply = self.inventory[:, :, 1, :] + self.inventory[:, :, 2, :]
        need = np.maximum(demand + 0.55 - self.inventory[:, :, 0, :], 0.0)
        if self.geometry_enabled:
            need *= front_priority[..., None]
        edge_capacity, path_capacity = self._cascade_capacities(corridor, supply, need)
        edge_cost, _, edge_reliability = self._build_finsler_metric(edge_capacity)
        path_cost, path_reliability, arrival = self._path_geometry(
            edge_cost, edge_reliability
        )

        residual_edge = edge_capacity.copy()
        remaining_supply = np.maximum(supply, 0.0).copy()
        remaining_need = np.maximum(need, 0.0).copy()
        plan = np.zeros((self.paths, 4, len(PATHS), 3))
        order = np.argsort(path_cost, axis=-1)
        for commodity in (0, 1, 2):
            for rank in range(len(PATHS)):
                selected = order[..., rank]
                for path_index, path in enumerate(PATHS):
                    mask = selected == path_index
                    path_list = list(path)
                    bottleneck = np.min(residual_edge[..., path_list], axis=-1)
                    amount = np.where(
                        mask,
                        np.minimum(
                            np.minimum(
                                remaining_supply[..., commodity],
                                remaining_need[..., commodity],
                            ),
                            bottleneck,
                        ),
                        0.0,
                    )
                    plan[..., path_index, commodity] += amount
                    remaining_supply[..., commodity] -= amount
                    remaining_need[..., commodity] -= amount
                    residual_edge[..., path_list] -= amount[..., None]

        total_by_commodity = np.sum(plan, axis=2)
        from_transit = np.minimum(self.inventory[:, :, 1, :], total_by_commodity)
        from_depth = total_by_commodity - from_transit
        self.inventory[:, :, 1, :] -= from_transit
        self.inventory[:, :, 2, :] -= from_depth
        safe_path_cost = np.where(
            np.isfinite(path_cost), path_cost, float(MAX_PIPELINE_DELAY)
        )
        delay = np.ceil(np.clip(safe_path_cost, 1.0, MAX_PIPELINE_DELAY)).astype(int)
        for path_index in range(len(PATHS)):
            for month_delay in range(1, MAX_PIPELINE_DELAY + 1):
                mask = delay[..., path_index] == month_delay
                self.shipment_pipeline[..., month_delay - 1] += (
                    plan[..., path_index, :] * mask[..., None]
                )

        edge_load = np.einsum("...pc,pe->...e", plan, PATH_INCIDENCE)
        self.last_edge_utilisation = edge_load / np.maximum(edge_capacity, 1.0e-12)
        self.last_route_flow = plan
        self.last_route_capacity = path_capacity
        scheduled = np.sum(total_by_commodity, axis=-1)
        requested = np.sum(need, axis=-1)
        plan_mass = np.sum(plan, axis=(2, 3))
        weighted_reliability = np.sum(
            np.sum(plan, axis=-1) * path_reliability, axis=-1
        ) / np.maximum(plan_mass, 1.0e-12)
        weighted_reliability = np.where(plan_mass > 0.0, weighted_reliability, 1.0)
        temporal_factor = np.clip(7.0 / np.maximum(arrival[..., SINK], 1.0), 0.30, 1.0)
        self.last_delivery_ratio = np.clip(
            scheduled / np.maximum(requested, 1.0e-12)
            * weighted_reliability * temporal_factor,
            0.0,
            1.0,
        )
        weights = np.array([1.18, 1.08, 0.92])[None, None, :]
        unmet = np.maximum(need - total_by_commodity, 0.0)
        self.last_shadow_scarcity = np.clip(
            np.sum(weights * unmet, axis=-1)
            / np.maximum(np.sum(weights * need, axis=-1), 1.0e-12),
            0.0,
            1.0,
        )

        feasible_path = np.isfinite(path_cost) & (path_cost <= 8.0)
        connectivity = np.zeros((self.paths, 4), dtype=int)
        for chosen in DISJOINT_PATH_SETS:
            feasible = np.all(feasible_path[..., list(chosen)], axis=-1)
            connectivity = np.maximum(connectivity, feasible.astype(int) * len(chosen))
        self.last_temporal_arrival = arrival[..., SINK]
        self.last_temporal_connectivity = connectivity
        self.minimum_temporal_connectivity_v72 = min(
            self.minimum_temporal_connectivity_v72, int(np.min(connectivity))
        )
        self.last_mfg_mobility_cost = np.clip(
            0.62
            + 0.17 * arrival[..., SINK] / 3.0
            + 0.30 * (1.0 - weighted_reliability),
            0.60,
            2.50,
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
        connectivity_capacity = self._minimum_cut(active.astype(float)).astype(int)
        self.last_edge_connectivity = connectivity_capacity
        self.last_node_connectivity = np.minimum(connectivity_capacity, 2)
        self.minimum_connectivity_v71 = min(
            self.minimum_connectivity_v71, int(np.min(connectivity_capacity))
        )
        capacity_residual = np.maximum(edge_load - edge_capacity, 0.0)
        residual = float(np.max(capacity_residual))
        self.max_capacity_residual_v71 = max(self.max_capacity_residual_v71, residual)
        self.maximum_finsler_capacity_residual = max(
            self.maximum_finsler_capacity_residual, residual
        )
        self.predicted_consumed = np.minimum(self.inventory[:, :, 0, :], demand)
        return total_by_commodity

    def update(self, **kwargs):
        result = super().update(**kwargs)
        self.months_v74 += 1
        return result

    def diagnostics(self) -> dict:
        base = super().diagnostics()
        finite_arrival = self.last_finsler_sink_arrival[
            np.isfinite(self.last_finsler_sink_arrival)
        ]
        base["finsler_network_v74"] = {
            "months": self.months_v74,
            "randers_metric_active": True,
            "directed_shortest_arrival_active": True,
            "cost_ordered_hard_flow_active": True,
            "pathwise_transport_delay_active": True,
            "mfg_action_cost_export_active": True,
            "stochastic_metric_draws_active": True,
            "robust_cost_interval_active": True,
            "maximum_randers_one_form_norm": self.maximum_randers_norm,
            "minimum_finsler_cost": float(self.minimum_finsler_cost),
            "maximum_direction_asymmetry": self.maximum_direction_asymmetry,
            "maximum_zero_drift_error": self.maximum_zero_drift_error,
            "maximum_positive_homogeneity_residual": self.maximum_homogeneity_residual,
            "maximum_eikonal_residual": self.maximum_eikonal_residual,
            "maximum_capacity_residual": self.maximum_finsler_capacity_residual,
            "minimum_path_reliability": self.minimum_path_reliability,
            "mean_sink_arrival": (
                float(np.mean(finite_arrival)) if finite_arrival.size else float("inf")
            ),
            "mean_mfg_mobility_cost": float(np.mean(self.last_mfg_mobility_cost)),
            "minimum_mfg_mobility_cost": float(np.min(self.last_mfg_mobility_cost)),
            "maximum_mfg_mobility_cost": float(np.max(self.last_mfg_mobility_cost)),
            "physical_inventory_owner": "inherited hard-capacity inventory solver",
            "continuous_geometry_owns_physical_inventory": False,
            "terminal_probability_adjustment": False,
        }
        return base
