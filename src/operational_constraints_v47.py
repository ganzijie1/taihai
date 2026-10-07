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
        geometry_enabled: bool = False,
    ):
        self.paths = paths
        self.geometry_enabled = bool(geometry_enabled)
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
            supply_density = np.sum(self.inventory, axis=3)
            total_mass = np.sum(supply_density, axis=2, keepdims=True)
            target_weights = np.array([0.52, 0.30, 0.18])[None, None, :]
            target_density = np.maximum(total_mass * target_weights, 1.0e-10)
            plan, uot_residual, uot_iterations = self._unbalanced_transport(
                supply_density, target_density, finsler_cost
            )
            plan_mass = np.maximum(np.sum(plan, axis=(2, 3)), 1.0e-12)
            transport_cost = np.sum(plan * finsler_cost, axis=(2, 3)) / plan_mass
            reachability = np.clip(np.exp(-0.42 * transport_cost), 0.18, 1.0)
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

        # Depth-to-transit and transit-to-front are shared-capacity
        # multicommodity flows with explicit capacity rationing.
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

        consumed = np.minimum(self.inventory[:, :, 0, :], demand)
        self.inventory[:, :, 0, :] -= consumed
        shortage = np.maximum(demand - consumed, 0.0)

        # Three-echelon METRIC approximation: failures create backorders;
        # service requires local line-replaceable units, with delayed upstream
        # replenishment tracked in an explicit four-month pipeline.
        failures = damage[:, :, None] * np.array([0.34, 0.30, 0.36])[None, None, :]
        self.repair_queue_state += failures
        local_spares = np.mean(self.spares[:, :, 0, :], axis=2)
        service_capacity = np.clip(
            0.11 * local_spares * command * financial
            * np.mean(self.facility_coverage), 0.0, 0.16
        )
        repaired = np.minimum(self.repair_queue_state, service_capacity[:, :, None])
        self.repair_queue_state -= repaired
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
        self.inventory = np.clip(self.inventory, 0.0, 1.50)
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
            "max_uot_solver_residual": self.max_uot_solver_residual,
            "max_uot_iterations_used": self.max_uot_iterations_used,
            "mean_unbalanced_transport_cost": float(np.mean(self.last_transport_cost)),
            "minimum_finsler_reachability": float(np.min(self.last_finsler_reachability)),
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

    def __init__(self, paths: int, terrain: dict, geometry_enabled: bool = False):
        self.paths = paths
        self.geometry_enabled = bool(geometry_enabled)
        self.sky = float(terrain["sky_view_proxy_p10_p50_p90"][1])
        self.covariance = np.repeat(
            np.diag([1.0, 1.0, 1.8])[None, None, :, :], paths * 4, axis=0
        ).reshape(paths, 4, 3, 3)
        self.integrity = np.full((paths, 4), 0.82)
        self.last_information_residual = 0.0
        self.max_spd_geodesic_distance = 0.0
        self.minimum_covariance_eigenvalue = 1.0

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
        prior_precision = np.linalg.inv(self.covariance)
        posterior_precision = prior_precision + precision
        posterior = np.linalg.inv(posterior_precision)
        process_noise = np.zeros_like(posterior)
        process_noise[..., 0, 0] = 0.025 + 0.12 * cyber
        process_noise[..., 1, 1] = 0.025 + 0.12 * cyber
        process_noise[..., 2, 2] = 0.040 + 0.15 * cyber
        target_covariance = posterior + process_noise
        if self.geometry_enabled:
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
            "max_spd_geodesic_distance": self.max_spd_geodesic_distance,
            "minimum_covariance_eigenvalue": self.minimum_covariance_eigenvalue,
        }
