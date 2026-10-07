from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SpatialControlFactors:
    zone_control: np.ndarray
    population_control: np.ndarray
    administrative_control: np.ndarray
    organized_defense: np.ndarray
    majority_control: np.ndarray
    stable_broad_control: np.ndarray
    transition_residual: float
    finsler_reachability: np.ndarray
    graph_spectral_gap: float


class SpatialControlNetwork:
    """Anonymous population- and administration-weighted regional control graph."""

    def __init__(
        self, paths: int, seed: int, terrain: dict,
        geometry_enabled: bool = False,
    ):
        self.paths = paths
        self.geometry_enabled = bool(geometry_enabled)
        self.rng = np.random.default_rng(seed)
        self.zones = 12
        self.population_weights = np.array(
            [0.145, 0.130, 0.112, 0.104, 0.096, 0.088, 0.078, 0.070, 0.060, 0.050, 0.039, 0.028]
        )
        self.population_weights /= self.population_weights.sum()
        self.administrative_weights = np.array(
            [0.135, 0.125, 0.115, 0.105, 0.095, 0.085, 0.080, 0.072, 0.062, 0.052, 0.042, 0.032]
        )
        self.administrative_weights /= self.administrative_weights.sum()
        mountain_share = float(terrain.get("land_share_above_500m", 0.42))
        self.terrain_friction = np.clip(
            np.array([0.18, 0.22, 0.28, 0.32, 0.38, 0.44, 0.50, 0.56, 0.63, 0.70, 0.78, 0.86])
            * (0.72 + 0.65 * mountain_share), 0.08, 0.95
        )
        adjacency = np.zeros((self.zones, self.zones))
        for zone in range(self.zones):
            if zone > 0:
                adjacency[zone, zone - 1] = 1.0
            if zone + 1 < self.zones:
                adjacency[zone, zone + 1] = 1.0
            if zone + 3 < self.zones:
                adjacency[zone, zone + 3] = 0.45
                adjacency[zone + 3, zone] = 0.45
        self.adjacency = adjacency / np.maximum(adjacency.sum(axis=1, keepdims=True), 1.0)
        zone_gradient = self.terrain_friction[:, None] - self.terrain_friction[None, :]
        directed_cost = np.where(
            adjacency > 0.0,
            (1.0 + 0.85 * np.maximum(zone_gradient, 0.0))
            / np.maximum(adjacency, 1.0e-6),
            1.0e6,
        )
        finsler_kernel = np.where(adjacency > 0.0, np.exp(-0.72 * directed_cost), 0.0)
        self.finsler_transition = finsler_kernel / np.maximum(
            finsler_kernel.sum(axis=1, keepdims=True), 1.0e-12
        )
        symmetric = 0.5 * (adjacency + adjacency.T)
        laplacian = np.diag(symmetric.sum(axis=1)) - symmetric
        spectrum = np.sort(np.linalg.eigvalsh(laplacian))
        self.graph_spectral_gap = float(spectrum[1]) if len(spectrum) > 1 else 0.0
        self.zone_control = np.clip(
            self.rng.beta(1.2, 46.0, (paths, self.zones)), 0.0, 0.10
        )
        self.organized_defense = np.ones((paths, self.zones))
        self.control_duration = np.zeros(paths, dtype=np.int16)
        self.last_residual = 0.0
        self.last_finsler_reachability = np.ones(paths)

    def update(
        self, *, aggregate_land: np.ndarray, attacker_logistics: np.ndarray,
        attacker_command: np.ndarray, defender_command: np.ndarray,
        defender_resistance: np.ndarray, external_intervention: np.ndarray,
        local_agent_capacity: np.ndarray, damage: np.ndarray,
        attacker_pnt: np.ndarray | None = None,
        blockade: np.ndarray | None = None,
    ) -> SpatialControlFactors:
        previous = self.zone_control.copy()
        neighbor_control = previous @ self.adjacency.T
        if self.geometry_enabled:
            pnt = np.ones(self.paths) if attacker_pnt is None else np.clip(attacker_pnt, 0.05, 1.0)
            denied = np.zeros(self.paths) if blockade is None else np.clip(blockade, 0.0, 1.0)
            finsler_neighbor = previous @ self.finsler_transition.T
            reachability = np.clip(
                (0.42 + 0.58 * attacker_logistics) * (0.55 + 0.45 * pnt)
                * (1.0 - 0.48 * denied), 0.08, 1.0
            )
            neighbor_control = (
                0.52 * neighbor_control
                + 0.48 * finsler_neighbor * reachability[:, None]
            )
            self.last_finsler_reachability = reachability
        beachhead_pressure = np.clip(
            0.40 * aggregate_land + 0.24 * attacker_logistics
            + 0.18 * attacker_command + 0.12 * local_agent_capacity
            - 0.20 * external_intervention,
            -0.40, 1.20,
        )
        expansion = (
            0.052 * beachhead_pressure[:, None]
            * (0.25 + 0.75 * neighbor_control)
            * (1.0 - self.terrain_friction[None, :])
            * (1.0 - previous)
        )
        rollback = (
            0.034 * (0.45 * defender_command + 0.40 * defender_resistance
                     + 0.15 * external_intervention)[:, None]
            * previous * (0.55 + 0.45 * self.terrain_friction[None, :])
        )
        disruption = 0.020 * damage[:, None] * (1.0 - previous)
        self.zone_control = np.clip(previous + expansion + disruption - rollback, 0.0, 1.0)
        organization_capacity = np.clip(
            0.52 + 0.24 * defender_command
            + 0.18 * defender_resistance + 0.14 * external_intervention,
            0.20, 1.10,
        )
        defense_target = np.clip(
            (1.0 - self.zone_control) * organization_capacity[:, None]
            * (1.0 - 0.30 * damage[:, None]),
            0.0, 1.0,
        )
        self.organized_defense = np.clip(
            0.90 * self.organized_defense + 0.10 * defense_target, 0.0, 1.0
        )
        population_control = self.zone_control @ self.population_weights
        administrative_control = self.zone_control @ self.administrative_weights
        organized_defense = self.organized_defense @ self.population_weights
        majority = (
            (population_control >= 0.50)
            & (administrative_control >= 0.50)
            & (organized_defense <= 0.35)
        )
        self.control_duration = np.where(majority, self.control_duration + 1, 0)
        stable = self.control_duration >= 6
        self.last_residual = float(np.max(np.abs(self.zone_control - previous)))
        return SpatialControlFactors(
            zone_control=self.zone_control.copy(),
            population_control=population_control,
            administrative_control=administrative_control,
            organized_defense=organized_defense,
            majority_control=majority,
            stable_broad_control=stable,
            transition_residual=self.last_residual,
            finsler_reachability=self.last_finsler_reachability.copy(),
            graph_spectral_gap=self.graph_spectral_gap,
        )

    def diagnostics(self) -> dict:
        return {
            "minimum_zone_control": float(np.min(self.zone_control)),
            "maximum_zone_control": float(np.max(self.zone_control)),
            "last_transition_residual": self.last_residual,
            "population_weight_residual": abs(float(self.population_weights.sum()) - 1.0),
            "administrative_weight_residual": abs(float(self.administrative_weights.sum()) - 1.0),
            "geometry_enabled": self.geometry_enabled,
            "minimum_finsler_reachability": float(np.min(self.last_finsler_reachability)),
            "graph_spectral_gap": self.graph_spectral_gap,
        }
