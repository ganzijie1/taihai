from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ManifoldStepV81:
    retraction_weight: np.ndarray
    iteration_budget: int
    chart_state: np.ndarray
    normal_residual: np.ndarray
    minimum_covariance_eigenvalue: float


class DifferentialManifoldConstraintV81:
    """Pathwise product-manifold constraint and diagnostic layer.

    The class owns only the probability-simplex support state, a slow residual
    chart and an SPD innovation covariance. Physical inventories, campaign
    states, terminal hazards and discrete collapse events remain owned by their
    existing solvers.
    """

    ATTRACTORS = np.array([
        [0.12, 0.12, 0.10, 0.10],
        [0.46, 0.50, 0.42, 0.42],
        [0.82, 0.84, 0.78, 0.76],
    ])

    def __init__(
        self,
        paths: int,
        initial_support: np.ndarray,
        *,
        simplex_enabled: bool = True,
        spd_enabled: bool = True,
        slow_manifold_enabled: bool = True,
        hybrid_chart_enabled: bool = True,
        basin_diagnostics_enabled: bool = True,
    ) -> None:
        support = np.clip(np.asarray(initial_support, dtype=float), 1.0e-8, 1.0 - 1.0e-8)
        if support.shape != (paths, 4):
            raise ValueError("initial_support must have shape (paths, 4)")
        self.paths = paths
        self.simplex_enabled = bool(simplex_enabled)
        self.spd_enabled = bool(spd_enabled)
        self.slow_manifold_enabled = bool(slow_manifold_enabled)
        self.hybrid_chart_enabled = bool(hybrid_chart_enabled)
        self.basin_diagnostics_enabled = bool(basin_diagnostics_enabled)
        self.support_simplex = np.stack((1.0 - support, support), axis=-1)
        self.slow_state = np.zeros((paths, 4, 4), dtype=float)
        self.slow_initialized = False
        self.covariance = np.broadcast_to(np.eye(4) * 0.08, (paths, 4, 4)).copy()
        self.chart_state = np.zeros((paths, 4), dtype=np.int8)
        self.pending_weight = np.ones((paths, 4), dtype=float)
        self.pending_normal_residual = np.zeros((paths, 4), dtype=float)
        self.pending_jump = np.zeros((paths, 4), dtype=float)
        self.updates = 0
        self.simplex_updates = 0
        self.spd_updates = 0
        self.slow_updates = 0
        self.chart_transitions = np.zeros((3, 3), dtype=np.int64)
        self.basin_counts = np.zeros(3, dtype=np.int64)
        self.maximum_simplex_residual = 0.0
        self.maximum_retraction_shift = 0.0
        self.maximum_normal_residual = 0.0
        self.minimum_spd_eigenvalue = np.inf
        self.maximum_spd_condition = 1.0
        self.minimum_separatrix_margin = np.inf
        self.maximum_jump_fast_exit = 0.0
        self.maximum_chart_weight_difference = 0.0

    @staticmethod
    def _features(kwargs: dict) -> np.ndarray:
        return np.stack((
            np.clip(kwargs["damage"] / 1.5, 0.0, 1.0),
            np.clip(kwargs["shortage"] / 1.5, 0.0, 1.0),
            np.clip(kwargs["financial_stress"] / 1.5, 0.0, 1.0),
            np.clip(1.0 - kwargs["command"], 0.0, 1.0),
        ), axis=-1)

    @staticmethod
    def _project_spd(matrix: np.ndarray, floor: float = 1.0e-6) -> tuple[np.ndarray, np.ndarray]:
        symmetric = 0.5 * (matrix + np.swapaxes(matrix, -1, -2))
        eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
        clipped = np.maximum(eigenvalues, floor)
        projected = np.einsum("pij,pj,pkj->pik", eigenvectors, clipped, eigenvectors)
        projected = 0.5 * (projected + np.swapaxes(projected, -1, -2))
        return projected, clipped

    def _update_chart(self, stress: np.ndarray) -> None:
        previous = self.chart_state.copy()
        if self.hybrid_chart_enabled:
            low_to_mid = (previous == 0) & (stress >= 0.38)
            mid_to_low = (previous == 1) & (stress <= 0.30)
            mid_to_high = (previous == 1) & (stress >= 0.68)
            high_to_mid = (previous == 2) & (stress <= 0.56)
            self.chart_state[low_to_mid] = 1
            self.chart_state[mid_to_low] = 0
            self.chart_state[mid_to_high] = 2
            self.chart_state[high_to_mid] = 1
        for left in range(3):
            for right in range(3):
                self.chart_transitions[left, right] += int(np.sum(
                    (previous == left) & (self.chart_state == right)
                ))

    def _basin_diagnostics(self, feature: np.ndarray) -> None:
        if not self.basin_diagnostics_enabled:
            return
        aggregate = np.mean(feature, axis=1)
        inverse = np.linalg.inv(self.covariance)
        difference = aggregate[:, None, :] - self.ATTRACTORS[None, :, :]
        distance = np.einsum("pki,pij,pkj->pk", difference, inverse, difference)
        order = np.sort(distance, axis=1)
        basin = np.argmin(distance, axis=1)
        self.basin_counts += np.bincount(basin, minlength=3)
        self.minimum_separatrix_margin = min(
            self.minimum_separatrix_margin,
            float(np.min(order[:, 1] - order[:, 0])),
        )

    def prepare(self, kwargs: dict) -> ManifoldStepV81:
        feature = self._features(kwargs)
        jump = np.clip(
            np.asarray(kwargs.get("jump_signal", np.zeros((self.paths, 4)))),
            0.0,
            1.0,
        )
        if not self.slow_initialized:
            self.slow_state = feature.copy()
            self.slow_initialized = True
        innovation = feature - self.slow_state
        if self.slow_manifold_enabled:
            rate = np.clip(0.08 + 0.42 * jump, 0.08, 0.50)[..., None]
            self.slow_state += rate * innovation
            self.slow_updates += 1
        else:
            self.slow_state = feature.copy()
        residual = feature - self.slow_state
        path_residual = np.mean(residual, axis=1)
        target_covariance = np.einsum("pi,pj->pij", path_residual, path_residual)
        target_covariance += np.eye(4)[None, ...] * 1.0e-4
        if self.spd_enabled:
            candidate = 0.94 * self.covariance + 0.06 * target_covariance
            self.covariance, eigenvalues = self._project_spd(candidate)
            self.spd_updates += 1
        else:
            eigenvalues = np.linalg.eigvalsh(self.covariance)
        inverse = np.linalg.inv(self.covariance)
        normal_squared = np.einsum("pai,pij,paj->pa", residual, inverse, residual)
        normal_residual = np.sqrt(np.maximum(normal_squared, 0.0))
        self.maximum_normal_residual = max(
            self.maximum_normal_residual, float(np.max(normal_residual))
        )
        self.minimum_spd_eigenvalue = min(
            self.minimum_spd_eigenvalue, float(np.min(eigenvalues))
        )
        condition = np.max(eigenvalues, axis=1) / np.maximum(
            np.min(eigenvalues, axis=1), 1.0e-12
        )
        self.maximum_spd_condition = max(
            self.maximum_spd_condition, float(np.max(condition))
        )
        stress = (
            0.29 * feature[..., 0]
            + 0.29 * feature[..., 1]
            + 0.22 * feature[..., 2]
            + 0.20 * feature[..., 3]
        )
        self._update_chart(stress)
        chart_weight = np.choose(self.chart_state, (0.46, 0.36, 0.28))
        if not self.hybrid_chart_enabled:
            chart_weight.fill(0.40)
        residual_penalty = 1.0 / (1.0 + 0.18 * normal_residual)
        fast_exit = np.clip(0.38 * jump, 0.0, 0.38)
        weight = np.clip(chart_weight * residual_penalty + fast_exit, 0.18, 0.82)
        self.maximum_jump_fast_exit = max(
            self.maximum_jump_fast_exit, float(np.max(fast_exit))
        )
        self.maximum_chart_weight_difference = max(
            self.maximum_chart_weight_difference,
            float(np.max(weight) - np.min(weight)),
        )
        self.pending_weight = weight
        self.pending_normal_residual = normal_residual
        self.pending_jump = jump
        self._basin_diagnostics(feature)
        self.updates += 1
        iteration_budget = int(48 + 8 * np.max(self.chart_state) + 8 * (
            np.max(normal_residual) > 1.0
        ))
        return ManifoldStepV81(
            retraction_weight=weight.copy(),
            iteration_budget=iteration_budget,
            chart_state=self.chart_state.copy(),
            normal_residual=normal_residual.copy(),
            minimum_covariance_eigenvalue=float(np.min(eigenvalues)),
        )

    def retract_support(self, target_support: np.ndarray) -> np.ndarray:
        target = np.clip(np.asarray(target_support), 1.0e-8, 1.0 - 1.0e-8)
        target_simplex = np.stack((1.0 - target, target), axis=-1)
        if self.simplex_enabled:
            alpha = self.pending_weight[..., None]
            root = (
                (1.0 - alpha) * np.sqrt(self.support_simplex)
                + alpha * np.sqrt(target_simplex)
            )
            updated = np.square(root)
            updated /= np.maximum(np.sum(updated, axis=-1, keepdims=True), 1.0e-15)
            self.simplex_updates += 1
        else:
            updated = target_simplex
        shift = np.max(np.abs(updated[..., 1] - target))
        residual = np.max(np.abs(np.sum(updated, axis=-1) - 1.0))
        self.maximum_retraction_shift = max(self.maximum_retraction_shift, float(shift))
        self.maximum_simplex_residual = max(self.maximum_simplex_residual, float(residual))
        self.support_simplex = updated
        return updated[..., 1].copy()

    def diagnostics(self) -> dict:
        total_basins = max(int(np.sum(self.basin_counts)), 1)
        transition_count = int(np.sum(self.chart_transitions) - np.trace(self.chart_transitions))
        return {
            "updates": self.updates,
            "simplex_updates": self.simplex_updates,
            "spd_updates": self.spd_updates,
            "slow_manifold_updates": self.slow_updates,
            "maximum_probability_simplex_residual": self.maximum_simplex_residual,
            "maximum_fisher_rao_retraction_shift": self.maximum_retraction_shift,
            "minimum_spd_eigenvalue": self.minimum_spd_eigenvalue,
            "maximum_spd_condition_number": self.maximum_spd_condition,
            "maximum_slow_normal_residual": self.maximum_normal_residual,
            "maximum_jump_fast_exit": self.maximum_jump_fast_exit,
            "maximum_chart_weight_difference": self.maximum_chart_weight_difference,
            "hybrid_chart_transition_count": transition_count,
            "hybrid_chart_transition_matrix": self.chart_transitions.tolist(),
            "attractor_basin_shares": (
                self.basin_counts.astype(float) / total_basins
            ).tolist(),
            "minimum_attractor_separatrix_margin": self.minimum_separatrix_margin,
            "simplex_enabled": self.simplex_enabled,
            "spd_enabled": self.spd_enabled,
            "slow_manifold_enabled": self.slow_manifold_enabled,
            "hybrid_chart_enabled": self.hybrid_chart_enabled,
            "basin_diagnostics_enabled": self.basin_diagnostics_enabled,
            "state_owner": {
                "support_probability": "v81_product_manifold_constraint",
                "physical_stocks": "inherited dedicated solvers",
                "terminal_hazards": "inherited competing-risk solver",
            },
            "terminal_probability_adjustment": False,
        }
