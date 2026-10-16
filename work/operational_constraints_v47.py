from __future__ import annotations

from dataclasses import dataclass

import numpy as np


ACTORS = ("china", "taiwan", "united_states", "japan")
ECHELONS = ("front", "transit", "depth")
COMMODITIES = ("munitions", "fuel", "critical_spares")


@dataclass
class OperationalFactors:
    front_inventory: np.ndarray
    echelon_integrity: np.ndarray
    repair_availability: np.ndarray
    repair_queue: np.ndarray
    search_effectiveness: np.ndarray
    flow_residual: float
    stock_residual: float
    unbalanced_transport_cost: np.ndarray
    finsler_reachability: np.ndarray
    uot_solver_residual: float


class DynamicOperationalConstraintSystem:
    """GIS-capacity, multicommodity flow, METRIC spares, and search solver."""

    def __init__(
        self, paths: int, seed: int, terrain: dict,
        geometry_enabled: bool = False, advanced_geometry: bool = False,
    ):
        self.paths = paths
        self.geometry_enabled = bool(geometry_enabled)
        self.advanced_geometry = bool(advanced_geometry)
        self.rng = np.random.default_rng(seed)
        mountain = float(terrain["land_share_above_500m"])
        sky = float(terrain["sky_view_proxy_p10_p50_p90"][1])
        actor_scale = np.array([1.00, 0.72, 1.18, 0.92])[None, :, None]
        echelon_scale = np.array([0.66, 0.82, 1.00])[None, None, :]
        terrain_penalty = np.array([0.42, 0.68, 0.18, 0.34])[None, :, None] * mountain
        self.corridor_capacity = np.clip(
            actor_scale * echelon_scale * (1.0 - terrain_penalty), 0.12, 1.25
        )
        self.corridor_capacity = np.repeat(self.corridor_capacity, paths, axis=0)
        base_stock = np.array([0.90, 0.94, 0.88])[None, None, None, :]
        echelon_stock = np.array([0.50, 0.76, 1.05])[None, None, :, None]
        self.inventory = np.clip(
            base_stock * echelon_stock
            * self.rng.lognormal(0.0, 0.045, (paths, 4, 3, 3)),
            0.02, 1.40,
        )
        self.spares = np.clip(
            self.rng.normal(0.86, 0.05, (paths, 4, 3, 3)), 0.20, 1.20
        )
        self.spare_pipeline = np.zeros((paths, 4, 3, 3, 4))
        self.repair_queue_state = np.zeros((paths, 4, 3))
        self.integrity = np.clip(
            self.rng.normal(0.92, 0.025, (paths, 4, 3)), 0.70, 0.99
        )
        self.search_prior = np.clip(
            self.rng.normal(0.45 + 0.12 * sky, 0.04, (paths, 4, 3)), 0.08, 0.90
        )
        # Robust maximal coverage: select three of six anonymous support sites
        # against normal, corridor-damaged, and depth-damaged scenarios.
        site_to_echelon = np.array([
            [0.92, 0.44, 0.18], [0.78, 0.66, 0.22], [0.55, 0.88, 0.42],
            [0.28, 0.74, 0.82], [0.20, 0.48, 0.94], [0.46, 0.62, 0.68],
        ])
        scenario_survival = np.array([
            [1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
            [0.62, 0.70, 0.82, 0.90, 0.94, 0.84],
            [0.92, 0.88, 0.76, 0.64, 0.58, 0.72],
        ])
        selected: list[int] = []
        for _ in range(3):
            best_site, best_score = -1, -np.inf
            for site in range(6):
                if site in selected:
                    continue
                trial = selected + [site]
                coverage = np.max(
                    scenario_survival[:, trial, None] * site_to_echelon[None, trial, :],
                    axis=1,
                )
                score = float(np.min(np.mean(coverage, axis=1)))
                if score > best_score:
                    best_site, best_score = site, score
            selected.append(best_site)
        self.selected_facilities = tuple(selected)
        robust_coverage = np.max(
            scenario_survival[:, selected, None] * site_to_echelon[None, selected, :],
            axis=1,
        )
        self.facility_coverage = np.min(robust_coverage, axis=0)
        self.corridor_adjacency = np.array([
            [0.0, 0.72, 0.10], [0.58, 0.0, 0.66], [0.08, 0.62, 0.0]
        ])
        self.last_flow_residual = 0.0
        self.max_stock_residual = 0.0
        self.last_transport_cost = np.zeros((paths, 4))
        self.last_finsler_reachability = np.ones((paths, 4))
        self.max_uot_solver_residual = 0.0
        self.max_uot_iterations_used = 0
        self.month = 0
        self.max_randers_norm = 0.0
        self.minimum_subriemannian_reachability = 1.0
        self.max_rotation_orthogonality_residual = 0.0
        self.max_rotation_determinant_residual = 0.0
        self.last_inventory_spill = np.zeros((paths, 4))

    @staticmethod
    def _rodrigues_z(angle: np.ndarray) -> np.ndarray:
        """Batched SO(3) exponential for aggregate heading changes."""
        cosine = np.cos(angle)
        sine = np.sin(angle)
        rotation = np.zeros(angle.shape + (3, 3))
        rotation[..., 0, 0] = cosine
        rotation[..., 0, 1] = -sine
        rotation[..., 1, 0] = sine
        rotation[..., 1, 1] = cosine
        rotation[..., 2, 2] = 1.0
        return rotation

    def _advanced_finsler_cost(
        self, interdiction: np.ndarray, command: np.ndarray,
        pnt: np.ndarray, damage: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Time-varying Randers cost and sub-Riemannian turn reachability.

        The three nodes are front, transit and depth aggregates of the GIS
        corridor graph. Hard capacity remains owned by the network-flow step.
        """
        base_length = np.array([
            [0.0, 1.15, 2.95],
            [0.92, 0.0, 1.18],
            [2.48, 0.88, 0.0],
        ])[None, None, :, :]
        slope_metric = np.array([
            [1.0, 1.18, 1.38], [1.10, 1.0, 1.22], [1.30, 1.12, 1.0]
        ])[None, None, :, :]
        threat = (
            0.78 * interdiction + 0.42 * damage + 0.34 * (1.0 - command)
            + 0.30 * (1.0 - pnt)
        )[:, :, None, None]
        weather_cycle = 0.08 * (1.0 + np.sin(2.0 * np.pi * self.month / 12.0))
        direction = np.array([
            [0.0, 0.10, 0.22], [-0.08, 0.0, 0.09], [-0.18, -0.07, 0.0]
        ])[None, None, :, :]
        randers_norm = np.abs(direction) / np.maximum(slope_metric, 1.0e-12)
        self.max_randers_norm = max(self.max_randers_norm, float(np.max(randers_norm)))
        symmetric = base_length * np.sqrt(slope_metric) * (
            1.0 + threat + weather_cycle
        )
        cost = np.maximum(symmetric + base_length * direction, 0.0)
        diagonal = np.arange(3)
        cost[..., diagonal, diagonal] = 0.0

        turn_demand = np.array([0.46, 0.72, 0.38])[None, None, :]
        angular_authority = np.clip(
            0.30 + 0.38 * command + 0.22 * pnt - 0.16 * damage,
            0.08, 0.96,
        )
        sub_reach = np.exp(
            -turn_demand * np.maximum(1.0 / angular_authority[:, :, None] - 1.0, 0.0)
        )
        self.minimum_subriemannian_reachability = min(
            self.minimum_subriemannian_reachability, float(np.min(sub_reach))
        )
        angle = 0.18 * angular_authority * (1.0 - 0.45 * interdiction)
        rotation = self._rodrigues_z(angle)
        identity = np.eye(3)[None, None, :, :]
        self.max_rotation_orthogonality_residual = max(
            self.max_rotation_orthogonality_residual,
            float(np.max(np.abs(np.swapaxes(rotation, -1, -2) @ rotation - identity))),
        )
        self.max_rotation_determinant_residual = max(
            self.max_rotation_determinant_residual,
            float(np.max(np.abs(np.linalg.det(rotation) - 1.0))),
        )
        return cost, sub_reach

    def current(self) -> OperationalFactors:
        front = np.clip(self.inventory[:, :, 0, :], 0.01, 1.35)
        availability = np.clip(1.0 - self.repair_queue_state / (1.0 + self.repair_queue_state), 0.0, 1.0)
        search = np.clip(np.mean(self.search_prior, axis=2), 0.02, 0.98)
        return OperationalFactors(
            front_inventory=front,
            echelon_integrity=self.integrity.copy(),
            repair_availability=availability,
            repair_queue=self.repair_queue_state.copy(),
            search_effectiveness=search,
            flow_residual=self.last_flow_residual,
            stock_residual=self.max_stock_residual,
            unbalanced_transport_cost=self.last_transport_cost.copy(),
            finsler_reachability=self.last_finsler_reachability.copy(),
            uot_solver_residual=self.max_uot_solver_residual,
        )

    @staticmethod
    def _entropy_allocate(supply: np.ndarray, demand: np.ndarray, cost: np.ndarray) -> np.ndarray:
        kernel = np.exp(-cost[None, None, :, :] / 0.20)
        supply = np.maximum(supply, 1e-10)
        demand = np.maximum(demand, 1e-10)
        mass = np.minimum(np.sum(supply, axis=2), np.sum(demand, axis=2))
        a = supply / np.maximum(np.sum(supply, axis=2, keepdims=True), 1e-12)
        b = demand / np.maximum(np.sum(demand, axis=2, keepdims=True), 1e-12)
        u = np.ones_like(a)
        v = np.ones_like(b)
        for _ in range(12):
            u = a / np.maximum(np.sum(kernel * v[:, :, None, :], axis=3), 1e-12)
            v = b / np.maximum(np.sum(kernel * u[:, :, :, None], axis=2), 1e-12)
        return kernel * u[:, :, :, None] * v[:, :, None, :] * mass[:, :, None, None]

    @staticmethod
    def _unbalanced_transport(
        supply: np.ndarray, demand: np.ndarray, cost: np.ndarray,
        entropy: float = 0.24, marginal_penalty: float = 1.10,
    ) -> tuple[np.ndarray, float, int]:
        """KL-relaxed Sinkhorn transport on the directed echelon graph."""
        kernel = np.exp(-cost / entropy)
        a = np.maximum(supply, 1.0e-10)
        b = np.maximum(demand, 1.0e-10)
        tau = marginal_penalty / (marginal_penalty + entropy)
        u = np.ones_like(a)
        v = np.ones_like(b)
        residual = np.inf
        iterations = 0
        for iteration in range(120):
            old_u = u
            old_v = v
            u = (a / np.maximum(np.einsum("paij,paj->pai", kernel, v), 1.0e-12)) ** tau
            v = (b / np.maximum(np.einsum("paij,pai->paj", kernel, u), 1.0e-12)) ** tau
            residual = float(max(np.max(np.abs(u - old_u)), np.max(np.abs(v - old_v))))
            iterations = iteration + 1
            if residual < 1.0e-7:
                break
        plan = kernel * u[..., :, None] * v[..., None, :]
        return plan, residual, iterations

    def _allocate_multicommodity_flows(
        self, demand: np.ndarray, corridor: np.ndarray,
        front_priority: np.ndarray,
    ) -> np.ndarray:
        """Move the three commodities through the two shared-capacity echelons.

        Kept as a separate hook so later model versions can replace the
        aggregate allocator without changing inventory ownership or the master
        monthly clock.  The base implementation is exactly the pre-V7.1 rule.
        """
        shipped_total = np.zeros((self.paths, 4, 3))
        for source_echelon, target_echelon in ((2, 1), (1, 0)):
            available = self.inventory[:, :, source_echelon, :]
            target_gap = np.maximum(
                demand + 0.55 - self.inventory[:, :, target_echelon, :], 0.0
            )
            if self.geometry_enabled and target_echelon == 0:
                target_gap *= front_priority[:, :, None]
            desired = np.minimum(available, target_gap)
            scale = np.minimum(
                1.0,
                corridor[:, :, target_echelon]
                / np.maximum(np.sum(desired, axis=2), 1e-12),
            )
            flow = desired * scale[:, :, None]
            self.inventory[:, :, source_echelon, :] -= flow
            self.inventory[:, :, target_echelon, :] += flow
            shipped_total += flow
        return shipped_total

    def _allocate_repairs(
        self, failures: np.ndarray, command: np.ndarray,
        financial: np.ndarray,
    ) -> np.ndarray:
        """Allocate divisible repair capacity across echelon backorders."""
        self.repair_queue_state += failures
        local_spares = np.mean(self.spares[:, :, 0, :], axis=2)
        service_capacity = np.clip(
            0.11 * local_spares * command * financial
            * np.mean(self.facility_coverage), 0.0, 0.16
        )
        repaired = np.minimum(
            self.repair_queue_state, service_capacity[:, :, None]
        )
        self.repair_queue_state -= repaired
        return repaired

    def update(
        self, *, production: np.ndarray, consumption: np.ndarray,
        damage: np.ndarray, interdiction: np.ndarray, command: np.ndarray,
        pnt: np.ndarray, financial_capacity: np.ndarray,
    ) -> OperationalFactors:
        opening = np.sum(self.inventory, axis=(2, 3))
        production = np.clip(production, 0.0, 0.18)
        consumption = np.clip(consumption, 0.0, 0.20)
        damage = np.clip(damage, 0.0, 1.0)
        interdiction = np.clip(interdiction, 0.0, 1.0)
        financial = np.clip(financial_capacity, 0.05, 1.0)

        produced = production[:, :, None] * np.array([0.42, 0.36, 0.22])[None, None, :]
        self.inventory[:, :, 2, :] += produced
        demand = consumption[:, :, None] * np.array([0.46, 0.38, 0.16])[None, None, :]
        corridor = np.clip(
            self.corridor_capacity * self.integrity * command[:, :, None]
            * financial[:, :, None] * (1.0 - 0.70 * interdiction[:, :, None]),
            0.0, 1.25,
        )
        failed_neighbor_share = (
            (self.integrity < 0.28).astype(float) @ self.corridor_adjacency.T
            / np.maximum(self.corridor_adjacency.sum(axis=1)[None, None, :], 1e-12)
        )
        corridor *= np.clip(1.0 - 0.42 * failed_neighbor_share, 0.10, 1.0)

        front_priority = np.ones((self.paths, 4))
        if self.geometry_enabled:
            # The directed cost is Finsler-like: depth-to-front movement is
            # more expensive than adjacent movement and reverse evacuation.
            if self.advanced_geometry:
                finsler_cost, sub_reach = self._advanced_finsler_cost(
                    interdiction, command, pnt, damage
                )
            else:
                base_cost = np.array([
                    [0.0, 1.30, 3.20],
                    [1.00, 0.0, 1.25],
                    [2.80, 1.00, 0.0],
                ])[None, None, :, :]
                dynamic_penalty = (
                    1.0 + 0.72 * interdiction + 0.36 * (1.0 - command)
                    + 0.28 * (1.0 - pnt)
                )[:, :, None, None]
                finsler_cost = base_cost * dynamic_penalty
                sub_reach = np.ones((self.paths, 4, 3))
            supply_density = np.sum(self.inventory, axis=3)
            total_mass = np.sum(supply_density, axis=2, keepdims=True)
            target_weights = np.array([0.52, 0.30, 0.18])[None, None, :]
            target_density = np.maximum(total_mass * target_weights, 1.0e-10)
            plan, uot_residual, uot_iterations = self._unbalanced_transport(
                supply_density, target_density, finsler_cost
            )
            plan_mass = np.maximum(np.sum(plan, axis=(2, 3)), 1.0e-12)
            transport_cost = np.sum(plan * finsler_cost, axis=(2, 3)) / plan_mass
            reachability = np.clip(
                np.exp(-0.42 * transport_cost) * np.mean(sub_reach, axis=2),
                0.12 if self.advanced_geometry else 0.18, 1.0,
            )
            front_share = np.sum(plan[..., 0], axis=2) / plan_mass
            front_priority = np.clip(0.72 + 0.86 * front_share, 0.70, 1.18)
            corridor *= (0.72 + 0.28 * reachability[:, :, None])
            self.last_transport_cost = transport_cost
            self.last_finsler_reachability = reachability
            self.max_uot_solver_residual = max(
                self.max_uot_solver_residual, uot_residual
            )
            self.max_uot_iterations_used = max(
                self.max_uot_iterations_used, uot_iterations
            )

        # The base hook reproduces the original two-echelon allocator.  V7.1
        # replaces it with a time-expanded, capacity-feasible network solver.
        shipped_total = self._allocate_multicommodity_flows(
            demand, corridor, front_priority
        )

        consumed = np.minimum(self.inventory[:, :, 0, :], demand)
        self.inventory[:, :, 0, :] -= consumed
        shortage = np.maximum(demand - consumed, 0.0)

        # Three-echelon METRIC approximation: failures create backorders;
        # service requires local line-replaceable units, with delayed upstream
        # replenishment tracked in an explicit four-month pipeline.
        failures = damage[:, :, None] * np.array([0.34, 0.30, 0.36])[None, None, :]
        repaired = self._allocate_repairs(failures, command, financial)
        spare_use = repaired[:, :, :, None] * np.array([0.42, 0.34, 0.24])[None, None, None, :]
        self.spares = np.maximum(self.spares - spare_use, 0.0)
        arrivals = self.spare_pipeline[..., 0].copy()
        self.spare_pipeline[..., :-1] = self.spare_pipeline[..., 1:]
        self.spare_pipeline[..., -1] = 0.0
        self.spares += arrivals
        reorder = np.maximum(0.78 - self.spares[:, :, 0, :], 0.0) * financial[:, :, None]
        self.spare_pipeline[:, :, 0, :, -1] += 0.55 * reorder
        self.spare_pipeline[:, :, 1, :, -2] += 0.30 * reorder
        self.spare_pipeline[:, :, 2, :, -1] += 0.15 * reorder

        damage_by_echelon = damage[:, :, None] * np.array([0.62, 0.26, 0.12])[None, None, :]
        self.integrity = np.clip(
            self.integrity + 0.10 * repaired - 0.18 * damage_by_echelon
            - 0.10 * shortage.mean(axis=2)[:, :, None],
            0.02, 0.995,
        )
        unclipped_inventory = self.inventory.copy()
        self.last_inventory_spill = np.sum(
            np.maximum(unclipped_inventory - 1.50, 0.0), axis=(2, 3)
        )
        self.inventory = np.clip(unclipped_inventory, 0.0, 1.50)
        self.spares = np.clip(self.spares, 0.0, 1.30)

        # Bayesian search effort allocation over air, surface, and subsurface.
        target_value = np.array([0.38, 0.34, 0.28])[None, None, :]
        information = self.search_prior * target_value * pnt[:, :, None]
        effort = information / np.maximum(np.sum(information, axis=2, keepdims=True), 1e-12)
        detection = 1.0 - np.exp(
            -2.2 * effort * pnt[:, :, None] * command[:, :, None]
        )
        self.search_prior = np.clip(
            0.72 * self.search_prior + 0.28 * detection, 0.01, 0.995
        )

        closing = np.sum(self.inventory, axis=(2, 3))
        expected = opening + np.sum(produced, axis=2) - np.sum(consumed, axis=2)
        stock_residual = float(np.max(np.abs(closing - expected)))
        flow_residual = float(np.max(np.maximum(
            np.sum(shipped_total, axis=2) - np.sum(self.corridor_capacity, axis=2), 0.0
        )))
        self.max_stock_residual = max(self.max_stock_residual, stock_residual)
        self.last_flow_residual = flow_residual
        self.month += 1
        return self.current()

    def diagnostics(self) -> dict:
        return {
            "last_flow_capacity_residual": self.last_flow_residual,
            "max_stock_accounting_residual": self.max_stock_residual,
            "minimum_inventory": float(np.min(self.inventory)),
            "minimum_spares": float(np.min(self.spares)),
            "selected_facilities": list(self.selected_facilities),
            "robust_minimum_coverage": float(np.min(self.facility_coverage)),
            "geometry_enabled": self.geometry_enabled,
            "advanced_geometry": self.advanced_geometry,
            "max_uot_solver_residual": self.max_uot_solver_residual,
            "max_uot_iterations_used": self.max_uot_iterations_used,
            "mean_unbalanced_transport_cost": float(np.mean(self.last_transport_cost)),
            "minimum_finsler_reachability": float(np.min(self.last_finsler_reachability)),
            "max_randers_one_form_norm": self.max_randers_norm,
            "minimum_subriemannian_reachability": self.minimum_subriemannian_reachability,
            "max_so3_orthogonality_residual": self.max_rotation_orthogonality_residual,
            "max_so3_determinant_residual": self.max_rotation_determinant_residual,
        }


@dataclass
class PNTFactors:
    integrity: np.ndarray
    horizontal_error: np.ndarray
    vertical_error: np.ndarray
    information_residual: float
    spd_geodesic_distance: float


class DynamicGeospatialPNTSystem:
    """Information-matrix fusion of GNSS, inertial, terrestrial, and terrain PNT."""

    def __init__(
        self, paths: int, terrain: dict, geometry_enabled: bool = False,
        geometry_metric: str = "affine_invariant", joseph_update: bool = False,
        adaptive_bures_enabled: bool = False,
        bures_barycenter_enabled: bool = False,
    ):
        self.paths = paths
        self.geometry_enabled = bool(geometry_enabled)
        if geometry_metric not in {"affine_invariant", "bures_wasserstein"}:
            raise ValueError("unsupported SPD geometry metric")
        self.geometry_metric = geometry_metric
        self.joseph_update = bool(joseph_update)
        self.adaptive_bures_enabled = bool(adaptive_bures_enabled)
        self.bures_barycenter_enabled = bool(bures_barycenter_enabled)
        self.sky = float(terrain["sky_view_proxy_p10_p50_p90"][1])
        self.covariance = np.repeat(
            np.diag([1.0, 1.0, 1.8])[None, None, :, :], paths * 4, axis=0
        ).reshape(paths, 4, 3, 3)
        self.integrity = np.full((paths, 4), 0.82)
        self.last_information_residual = 0.0
        self.max_spd_geodesic_distance = 0.0
        self.minimum_covariance_eigenvalue = 1.0
        self.max_bures_wasserstein_distance = 0.0
        self.max_joseph_symmetry_residual = 0.0
        self.max_barycenter_residual = 0.0
        self.max_barycenter_iterations = 0
        self.max_source_weight_residual = 0.0
        self.minimum_source_weight = 1.0
        self.minimum_adaptive_weight = 1.0
        self.maximum_adaptive_weight = 0.0
        self.maximum_source_dispersion = 0.0

    @staticmethod
    def _matrix_power_spd(matrix: np.ndarray, power: float) -> np.ndarray:
        eigenvalue, eigenvector = np.linalg.eigh(matrix)
        eigenvalue = np.maximum(eigenvalue, 1.0e-12) ** power
        return np.einsum(
            "...ik,...k,...jk->...ij", eigenvector, eigenvalue, eigenvector,
            optimize=True,
        )

    @classmethod
    def _affine_invariant_geodesic(
        cls, opening: np.ndarray, target: np.ndarray, weight: float,
    ) -> tuple[np.ndarray, float]:
        root = cls._matrix_power_spd(opening, 0.5)
        inverse_root = cls._matrix_power_spd(opening, -0.5)
        relative = inverse_root @ target @ inverse_root
        relative_power = cls._matrix_power_spd(relative, weight)
        mixed = root @ relative_power @ root
        eigenvalue = np.maximum(np.linalg.eigvalsh(relative), 1.0e-12)
        distance = float(np.max(np.sqrt(np.sum(np.log(eigenvalue) ** 2, axis=-1))))
        return 0.5 * (mixed + np.swapaxes(mixed, -1, -2)), distance

    @classmethod
    def _bures_wasserstein_geodesic(
        cls, opening: np.ndarray, target: np.ndarray, weight: float,
    ) -> tuple[np.ndarray, float]:
        root = cls._matrix_power_spd(opening, 0.5)
        inverse_root = cls._matrix_power_spd(opening, -0.5)
        middle = root @ target @ root
        middle_root = cls._matrix_power_spd(middle, 0.5)
        transport = inverse_root @ middle_root @ inverse_root
        identity = np.eye(opening.shape[-1])
        interpolation = (1.0 - weight) * identity + weight * transport
        mixed = interpolation @ opening @ interpolation
        trace_term = np.trace(opening, axis1=-2, axis2=-1) + np.trace(
            target, axis1=-2, axis2=-1
        ) - 2.0 * np.trace(middle_root, axis1=-2, axis2=-1)
        distance = float(np.max(np.sqrt(np.maximum(trace_term, 0.0))))
        return 0.5 * (mixed + np.swapaxes(mixed, -1, -2)), distance

    @classmethod
    def _bures_distance(cls, opening: np.ndarray, target: np.ndarray) -> np.ndarray:
        root = cls._matrix_power_spd(opening, 0.5)
        middle_root = cls._matrix_power_spd(root @ target @ root, 0.5)
        trace_term = (
            np.trace(opening, axis1=-2, axis2=-1)
            + np.trace(target, axis1=-2, axis2=-1)
            - 2.0 * np.trace(middle_root, axis1=-2, axis2=-1)
        )
        return np.sqrt(np.maximum(trace_term, 0.0))

    @classmethod
    def _bures_barycenter(
        cls, covariance: np.ndarray, weights: np.ndarray,
        maximum_iterations: int = 12, tolerance: float = 1.0e-9,
    ) -> tuple[np.ndarray, float, int]:
        """Fixed point Bures barycenter for [..., source, dimension, dimension]."""
        weights = weights / np.maximum(np.sum(weights, axis=-1, keepdims=True), 1.0e-15)
        estimate = np.sum(weights[..., None, None] * covariance, axis=-3)
        residual = np.inf
        used = 0
        for iteration in range(maximum_iterations):
            root = cls._matrix_power_spd(estimate, 0.5)
            inverse_root = cls._matrix_power_spd(estimate, -0.5)
            middle = root[..., None, :, :] @ covariance @ root[..., None, :, :]
            mean_root = np.sum(
                weights[..., None, None] * cls._matrix_power_spd(middle, 0.5),
                axis=-3,
            )
            updated = inverse_root @ mean_root @ mean_root @ inverse_root
            updated = 0.5 * (updated + np.swapaxes(updated, -1, -2))
            residual = float(np.max(cls._bures_distance(estimate, updated)))
            estimate = updated
            used = iteration + 1
            if residual <= tolerance:
                break
        return estimate, residual, used

    def update(
        self, *, infrastructure: np.ndarray, cyber_damage: np.ndarray,
        command: np.ndarray, terrain_mask: np.ndarray | float,
    ) -> PNTFactors:
        infra = np.clip(infrastructure, 0.02, 1.0)
        cyber = np.clip(cyber_damage, 0.0, 1.0)
        mask = np.clip(np.asarray(terrain_mask), 0.10, 1.0)
        if mask.ndim == 0:
            mask = np.full((self.paths, 4), float(mask))
        gnss = np.clip(self.sky * infra * (1.0 - 0.78 * cyber), 0.01, 1.0)
        inertial = np.clip(0.38 + 0.42 * command, 0.05, 0.90)
        terrestrial = np.clip(0.18 + 0.65 * infra * mask, 0.02, 0.92)
        terrain = np.clip(0.16 + 0.55 * mask * command, 0.02, 0.88)
        precision = np.zeros_like(self.covariance)
        precision[..., 0, 0] = 0.70 * gnss + 0.42 * terrestrial + 0.24 * inertial
        precision[..., 1, 1] = 0.70 * gnss + 0.42 * terrestrial + 0.24 * inertial
        precision[..., 2, 2] = 0.42 * gnss + 0.34 * terrain + 0.30 * inertial
        precision[..., 0, 1] = precision[..., 1, 0] = 0.04 * terrestrial
        measurement_covariance = np.linalg.inv(
            precision + 1.0e-12 * np.eye(3)[None, None, :, :]
        )
        source_dispersion = np.zeros((self.paths, 4))
        if self.bures_barycenter_enabled:
            source_precision = np.zeros((self.paths, 4, 4, 3, 3))
            source_precision[..., 0, 0, 0] = 0.70 * gnss
            source_precision[..., 0, 1, 1] = 0.70 * gnss
            source_precision[..., 0, 2, 2] = 0.42 * gnss
            source_precision[..., 1, 0, 0] = 0.24 * inertial
            source_precision[..., 1, 1, 1] = 0.24 * inertial
            source_precision[..., 1, 2, 2] = 0.30 * inertial
            source_precision[..., 2, 0, 0] = 0.42 * terrestrial
            source_precision[..., 2, 1, 1] = 0.42 * terrestrial
            source_precision[..., 2, 2, 2] = 0.08 * terrestrial
            source_precision[..., 2, 0, 1] = 0.04 * terrestrial
            source_precision[..., 2, 1, 0] = 0.04 * terrestrial
            source_precision[..., 3, 0, 0] = 0.10 * terrain
            source_precision[..., 3, 1, 1] = 0.10 * terrain
            source_precision[..., 3, 2, 2] = 0.34 * terrain
            source_precision += 0.015 * np.eye(3)[None, None, None, :, :]
            source_covariance = np.linalg.inv(source_precision)
            source_strength = np.stack((gnss, inertial, terrestrial, terrain), axis=-1)
            source_weights = source_strength / np.maximum(
                np.sum(source_strength, axis=-1, keepdims=True), 1.0e-15
            )
            barycenter, barycenter_residual, barycenter_iterations = self._bures_barycenter(
                source_covariance, source_weights
            )
            effective_count = np.sum(source_strength, axis=-1) / np.maximum(
                np.max(source_strength, axis=-1), 1.0e-12
            )
            measurement_covariance = barycenter / np.maximum(
                effective_count[..., None, None], 1.0
            )
            precision = np.linalg.inv(measurement_covariance)
            distances = np.stack([
                self._bures_distance(barycenter, source_covariance[..., source, :, :])
                for source in range(4)
            ], axis=-1)
            source_dispersion = np.sum(source_weights * distances, axis=-1)
            self.max_barycenter_residual = max(
                self.max_barycenter_residual, barycenter_residual
            )
            self.max_barycenter_iterations = max(
                self.max_barycenter_iterations, barycenter_iterations
            )
            self.max_source_weight_residual = max(
                self.max_source_weight_residual,
                float(np.max(np.abs(np.sum(source_weights, axis=-1) - 1.0))),
            )
            self.minimum_source_weight = min(
                self.minimum_source_weight, float(np.min(source_weights))
            )
            self.maximum_source_dispersion = max(
                self.maximum_source_dispersion, float(np.max(source_dispersion))
            )
        prior_precision = np.linalg.inv(self.covariance)
        posterior_precision = prior_precision + precision
        if self.joseph_update:
            gain = self.covariance @ np.linalg.inv(
                self.covariance + measurement_covariance
            )
            identity = np.eye(3)[None, None, :, :]
            left = identity - gain
            posterior = (
                left @ self.covariance @ np.swapaxes(left, -1, -2)
                + gain @ measurement_covariance @ np.swapaxes(gain, -1, -2)
            )
            self.max_joseph_symmetry_residual = max(
                self.max_joseph_symmetry_residual,
                float(np.max(np.abs(posterior - np.swapaxes(posterior, -1, -2)))),
            )
        else:
            posterior = np.linalg.inv(posterior_precision)
        process_noise = np.zeros_like(posterior)
        process_noise[..., 0, 0] = 0.025 + 0.12 * cyber
        process_noise[..., 1, 1] = 0.025 + 0.12 * cyber
        process_noise[..., 2, 2] = 0.040 + 0.15 * cyber
        target_covariance = posterior + process_noise
        if self.geometry_enabled:
            if self.geometry_metric == "bures_wasserstein":
                weight = 0.88
                if self.adaptive_bures_enabled:
                    innovation_distance = self._bures_distance(
                        self.covariance, target_covariance
                    )
                    weight_field = np.clip(
                        0.94 / (
                            1.0 + 0.34 * source_dispersion
                            + 0.48 * innovation_distance + 0.55 * cyber
                        ),
                        0.32, 0.94,
                    )
                    # A scalar per batch is supported by the matrix geodesic.
                    root = self._matrix_power_spd(self.covariance, 0.5)
                    inverse_root = self._matrix_power_spd(self.covariance, -0.5)
                    middle_root = self._matrix_power_spd(
                        root @ target_covariance @ root, 0.5
                    )
                    transport = inverse_root @ middle_root @ inverse_root
                    identity = np.eye(3)[None, None, :, :]
                    interpolation = identity + weight_field[..., None, None] * (
                        transport - identity
                    )
                    self.covariance = interpolation @ self.covariance @ interpolation
                    self.covariance = 0.5 * (
                        self.covariance + np.swapaxes(self.covariance, -1, -2)
                    )
                    distance = float(np.max(innovation_distance))
                    self.minimum_adaptive_weight = min(
                        self.minimum_adaptive_weight, float(np.min(weight_field))
                    )
                    self.maximum_adaptive_weight = max(
                        self.maximum_adaptive_weight, float(np.max(weight_field))
                    )
                else:
                    self.covariance, distance = self._bures_wasserstein_geodesic(
                        self.covariance, target_covariance, weight
                    )
                self.max_bures_wasserstein_distance = max(
                    self.max_bures_wasserstein_distance, distance
                )
            else:
                self.covariance, distance = self._affine_invariant_geodesic(
                    self.covariance, target_covariance, 0.88
                )
            self.max_spd_geodesic_distance = max(
                self.max_spd_geodesic_distance, distance
            )
        else:
            self.covariance = target_covariance
        horizontal = np.sqrt(np.maximum(
            self.covariance[..., 0, 0] + self.covariance[..., 1, 1], 1e-12
        ))
        vertical = np.sqrt(np.maximum(self.covariance[..., 2, 2], 1e-12))
        self.integrity = np.clip(
            np.exp(-0.34 * horizontal - 0.22 * vertical)
            * (0.55 + 0.45 * np.maximum.reduce([gnss, inertial, terrestrial, terrain])),
            0.03, 0.995,
        )
        identity = np.eye(3)[None, None, :, :]
        residual = float(np.max(np.abs(
            posterior_precision @ posterior - identity
        )))
        self.last_information_residual = residual
        self.minimum_covariance_eigenvalue = min(
            self.minimum_covariance_eigenvalue,
            float(np.min(np.linalg.eigvalsh(self.covariance))),
        )
        return PNTFactors(
            self.integrity.copy(), horizontal, vertical, residual,
            self.max_spd_geodesic_distance,
        )

    def diagnostics(self) -> dict:
        return {
            "information_inverse_residual": self.last_information_residual,
            "minimum_integrity": float(np.min(self.integrity)),
            "maximum_covariance": float(np.max(self.covariance)),
            "geometry_enabled": self.geometry_enabled,
            "geometry_metric": self.geometry_metric,
            "joseph_update": self.joseph_update,
            "max_spd_geodesic_distance": self.max_spd_geodesic_distance,
            "minimum_covariance_eigenvalue": self.minimum_covariance_eigenvalue,
            "max_bures_wasserstein_distance": self.max_bures_wasserstein_distance,
            "max_joseph_symmetry_residual": self.max_joseph_symmetry_residual,
            "adaptive_bures_enabled": self.adaptive_bures_enabled,
            "bures_barycenter_enabled": self.bures_barycenter_enabled,
            "max_barycenter_residual": self.max_barycenter_residual,
            "max_barycenter_iterations": self.max_barycenter_iterations,
            "max_source_weight_residual": self.max_source_weight_residual,
            "minimum_source_weight": self.minimum_source_weight,
            "minimum_adaptive_weight": self.minimum_adaptive_weight,
            "maximum_adaptive_weight": self.maximum_adaptive_weight,
            "maximum_source_dispersion": self.maximum_source_dispersion,
        }
