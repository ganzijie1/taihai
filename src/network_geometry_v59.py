from __future__ import annotations

from dataclasses import dataclass

import numpy as np


LAYERS = ("information", "economic", "identity", "casualty")


def _row_normalise(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    intensity = np.sum(values, axis=-1)
    transition = values / np.maximum(intensity[..., None], 1.0e-12)
    return transition, intensity


@dataclass
class NetworkGeometryFactors:
    transition: np.ndarray
    intensity: np.ndarray
    intensity_multiplier: np.ndarray
    layer_weights: np.ndarray
    bottleneck_risk: np.ndarray
    sheaf_consistency: np.ndarray
    actor_spectral_gap: np.ndarray
    supra_spectral_gap: np.ndarray
    conductance: np.ndarray
    forman_curvature: np.ndarray


class NetworkGeometrySystem:
    """Pathwise dynamic geometry for the four-layer directed Graphon.

    The system keeps network strength separate from the row-stochastic
    transition kernel. Expensive supra-Laplacian eigensystems are refreshed on
    an explicit interval and held between refreshes; all other states update on
    every monthly master-clock step.
    """

    def __init__(
        self,
        initial_kernel: np.ndarray,
        *,
        spectral_interval: int = 12,
        intensity_transition_enabled: bool = True,
        supra_laplacian_enabled: bool = True,
        spectral_geometry_enabled: bool = True,
        forman_curvature_enabled: bool = True,
        sheaf_laplacian_enabled: bool = True,
    ) -> None:
        kernel = np.asarray(initial_kernel, dtype=float)
        if kernel.ndim != 4 or kernel.shape[1:] != (4, 4, 4):
            raise ValueError("network geometry requires [path,layer,actor,actor] kernels")
        self.paths = kernel.shape[0]
        self.initial_kernel = np.clip(kernel.copy(), 1.0e-6, 3.0)
        self.raw_kernel = self.initial_kernel.copy()
        _, self.baseline_intensity = _row_normalise(self.initial_kernel)
        self.spectral_interval = max(int(spectral_interval), 1)
        self.intensity_transition_enabled = bool(intensity_transition_enabled)
        self.supra_laplacian_enabled = bool(supra_laplacian_enabled)
        self.spectral_geometry_enabled = bool(spectral_geometry_enabled)
        self.forman_curvature_enabled = bool(forman_curvature_enabled)
        self.sheaf_laplacian_enabled = bool(sheaf_laplacian_enabled)
        self.month = 0
        self.cross_layer_base = np.array([
            [0.00, 0.11, 0.24, 0.16],
            [0.09, 0.00, 0.12, 0.20],
            [0.22, 0.08, 0.00, 0.18],
            [0.20, 0.18, 0.16, 0.00],
        ])
        self.last_supra_gap = np.zeros(self.paths)
        self.last_supra_minimum = np.zeros(self.paths)
        self.max_transition_row_residual = 0.0
        self.minimum_intensity = np.inf
        self.maximum_intensity = 0.0
        self.minimum_actor_spectral_gap = np.inf
        self.minimum_supra_spectral_gap = np.inf
        self.minimum_conductance = np.inf
        self.minimum_forman_curvature = np.inf
        self.minimum_sheaf_eigenvalue = np.inf
        self.maximum_sheaf_energy = 0.0
        self.maximum_temporal_log_step = 0.0
        self.supra_refreshes = 0
        self.maximum_bottleneck_risk = 0.0
        self.last_factors: NetworkGeometryFactors | None = None

    @staticmethod
    def _conductance(adjacency: np.ndarray) -> np.ndarray:
        degree = np.sum(adjacency, axis=-1)
        total_volume = np.sum(degree, axis=-1)
        best = np.full(adjacency.shape[0], np.inf)
        # Complements are equivalent, so seven nontrivial cuts suffice for four nodes.
        for mask_code in range(1, 8):
            mask = np.array([(mask_code >> bit) & 1 for bit in range(4)], dtype=bool)
            complement = ~mask
            cut = np.sum(adjacency[:, mask][:, :, complement], axis=(1, 2))
            volume = np.sum(degree[:, mask], axis=1)
            denominator = np.minimum(volume, total_volume - volume)
            ratio = cut / np.maximum(denominator, 1.0e-12)
            best = np.minimum(best, ratio)
        return np.clip(best, 0.0, 1.0)

    @staticmethod
    def _forman_node_curvature(adjacency: np.ndarray) -> np.ndarray:
        paths, actors, _ = adjacency.shape
        curvature_sum = np.zeros((paths, actors))
        curvature_weight = np.zeros((paths, actors))
        degree = np.maximum(np.sum(adjacency, axis=-1), 1.0e-12)
        for i in range(actors):
            for j in range(i + 1, actors):
                edge = np.maximum(adjacency[:, i, j], 1.0e-12)
                adjacent_i = np.sum(
                    adjacency[:, i, :] / np.sqrt(
                        edge[:, None] * np.maximum(adjacency[:, i, :], 1.0e-12)
                    ),
                    axis=1,
                ) - 1.0
                adjacent_j = np.sum(
                    adjacency[:, j, :] / np.sqrt(
                        edge[:, None] * np.maximum(adjacency[:, j, :], 1.0e-12)
                    ),
                    axis=1,
                ) - 1.0
                edge_curvature = 2.0 - adjacent_i - adjacent_j
                curvature_sum[:, i] += edge * edge_curvature
                curvature_sum[:, j] += edge * edge_curvature
                curvature_weight[:, i] += edge
                curvature_weight[:, j] += edge
        return curvature_sum / np.maximum(curvature_weight, 1.0e-12)

    def _sheaf_geometry(
        self, transition: np.ndarray, actor_signal: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.sheaf_laplacian_enabled:
            return (
                np.ones((self.paths, 4)),
                np.zeros(self.paths),
                np.zeros(self.paths),
            )
        information = transition[:, 0]
        identity = transition[:, 2]
        mismatch = information - identity
        node_energy = np.sqrt(np.mean(np.square(mismatch), axis=-1))
        consistency = np.exp(-2.5 * node_energy)
        # The scalar-stalk sheaf Laplacian is B^T B. Restrictions are the
        # geometric means of information and identity edge weights.
        minimum_eigenvalue = np.zeros(self.paths)
        sheaf_energy = np.zeros(self.paths)
        for path in range(self.paths):
            incidence_rows = []
            for i in range(4):
                for j in range(i + 1, 4):
                    left = np.sqrt(max(information[path, i, j], 1.0e-12))
                    right = np.sqrt(max(identity[path, j, i], 1.0e-12))
                    row = np.zeros(4)
                    row[i] = left
                    row[j] = -right
                    incidence_rows.append(row)
            incidence = np.asarray(incidence_rows)
            laplacian = incidence.T @ incidence
            eigenvalues = np.linalg.eigvalsh(laplacian)
            minimum_eigenvalue[path] = max(float(eigenvalues[0]), 0.0)
            signal = actor_signal[path]
            sheaf_energy[path] = float(signal @ laplacian @ signal)
        return consistency, minimum_eigenvalue, sheaf_energy

    def _supra_gap(
        self, symmetric_layers: np.ndarray, cross_layer: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if not self.supra_laplacian_enabled:
            return np.zeros(self.paths), np.zeros(self.paths)
        gaps = np.zeros(self.paths)
        minima = np.zeros(self.paths)
        for path in range(self.paths):
            adjacency = np.zeros((16, 16))
            for layer in range(4):
                block = symmetric_layers[path, layer]
                start = 4 * layer
                adjacency[start:start + 4, start:start + 4] = block
            for left in range(4):
                for right in range(left + 1, 4):
                    strength = 0.5 * (
                        cross_layer[path, left, right]
                        + cross_layer[path, right, left]
                    )
                    for actor in range(4):
                        i = 4 * left + actor
                        j = 4 * right + actor
                        adjacency[i, j] = strength
                        adjacency[j, i] = strength
            laplacian = np.diag(np.sum(adjacency, axis=1)) - adjacency
            eigenvalues = np.linalg.eigvalsh(laplacian)
            minima[path] = float(eigenvalues[0])
            gaps[path] = max(float(eigenvalues[1]), 0.0)
        return gaps, minima

    def update(
        self,
        *,
        target_kernel: np.ndarray,
        base_layer_weights: np.ndarray,
        damage: np.ndarray,
        shortage: np.ndarray,
        network: np.ndarray,
        common_noise: np.ndarray,
        jump_signal: np.ndarray,
        actor_signal: np.ndarray,
    ) -> NetworkGeometryFactors:
        target = np.clip(np.asarray(target_kernel, dtype=float), 1.0e-6, 3.0)
        previous = self.raw_kernel.copy()
        if self.intensity_transition_enabled:
            # Bounded proximal-style temporal update. Damage and jumps can
            # contract capacity, while intact networks permit gradual repair.
            stress = np.clip(
                0.32 * damage + 0.24 * shortage + 0.22 * jump_signal
                + 0.12 * np.maximum(common_noise, 0.0),
                0.0, 0.85,
            )
            repair = np.clip(0.025 * network * (1.0 - stress), 0.0, 0.025)
            target *= (1.0 - 0.18 * stress)[:, None, :, None]
            target *= (1.0 + repair)[:, None, :, None]
            proposed = 0.82 * previous + 0.18 * target
            log_step = np.clip(
                np.log(np.maximum(proposed, 1.0e-12) / np.maximum(previous, 1.0e-12)),
                -0.18,
                0.18,
            )
            self.raw_kernel = np.clip(previous * np.exp(log_step), 1.0e-6, 3.0)
            self.maximum_temporal_log_step = max(
                self.maximum_temporal_log_step, float(np.max(np.abs(log_step)))
            )
        else:
            self.raw_kernel = target

        transition, intensity = _row_normalise(self.raw_kernel)
        row_residual = float(np.max(np.abs(np.sum(transition, axis=-1) - 1.0)))
        self.max_transition_row_residual = max(
            self.max_transition_row_residual, row_residual
        )
        self.minimum_intensity = min(self.minimum_intensity, float(np.min(intensity)))
        self.maximum_intensity = max(self.maximum_intensity, float(np.max(intensity)))
        intensity_multiplier = np.clip(
            intensity / np.maximum(self.baseline_intensity, 1.0e-12), 0.35, 1.80
        )

        layer_mean_intensity = np.mean(intensity_multiplier, axis=-1)
        layer_weights = np.asarray(base_layer_weights, dtype=float) * np.sqrt(
            np.maximum(layer_mean_intensity, 1.0e-12)
        )
        layer_weights /= np.maximum(np.sum(layer_weights, axis=1, keepdims=True), 1.0e-12)

        symmetric_layers = 0.5 * (
            transition + np.swapaxes(transition, -1, -2)
        )
        diagonal = np.arange(4)
        symmetric_layers[:, :, diagonal, diagonal] = 0.0
        actor_adjacency = np.einsum("pl,plij->pij", layer_weights, symmetric_layers)
        actor_laplacian = (
            np.eye(4)[None, :, :] * np.sum(actor_adjacency, axis=-1)[:, :, None]
            - actor_adjacency
        )
        if self.spectral_geometry_enabled:
            actor_eigenvalues = np.linalg.eigvalsh(actor_laplacian)
            actor_gap = np.maximum(actor_eigenvalues[:, 1], 0.0)
            conductance = self._conductance(actor_adjacency)
        else:
            actor_gap = np.ones(self.paths)
            conductance = np.ones(self.paths)

        cross_pressure = np.clip(
            1.0 + 0.22 * np.mean(jump_signal, axis=1)
            - 0.18 * np.mean(damage + shortage, axis=1),
            0.45, 1.45,
        )
        cross_layer = self.cross_layer_base[None, :, :] * cross_pressure[:, None, None]
        if self.month % self.spectral_interval == 0:
            self.last_supra_gap, self.last_supra_minimum = self._supra_gap(
                symmetric_layers, cross_layer
            )
            self.supra_refreshes += 1

        if self.forman_curvature_enabled:
            forman = self._forman_node_curvature(actor_adjacency)
        else:
            forman = np.zeros((self.paths, 4))
        sheaf_consistency, sheaf_minimum, sheaf_energy = self._sheaf_geometry(
            transition, np.asarray(actor_signal, dtype=float)
        )

        degree = np.sum(actor_adjacency, axis=-1)
        relative_degree = degree / np.maximum(np.mean(degree, axis=1, keepdims=True), 1.0e-12)
        curvature_risk = 1.0 / (1.0 + np.exp(np.clip(forman, -20.0, 20.0)))
        global_bottleneck = (
            0.42 * (1.0 - conductance)
            + 0.30 * np.exp(-self.last_supra_gap)
        )[:, None]
        bottleneck_risk = np.clip(
            global_bottleneck
            + 0.16 * np.clip(1.0 - relative_degree, 0.0, 1.0)
            + 0.12 * curvature_risk
            + 0.18 * (1.0 - sheaf_consistency),
            0.0, 1.0,
        )
        # Geometry affects current propagation capacity and therefore the MFG
        # field, while the raw kernel remains the state carried to next month.
        intensity_multiplier *= np.clip(
            1.0 - 0.24 * bottleneck_risk[:, None, :]
            + 0.08 * sheaf_consistency[:, None, :],
            0.55, 1.10,
        )

        self.minimum_actor_spectral_gap = min(
            self.minimum_actor_spectral_gap, float(np.min(actor_gap))
        )
        self.minimum_supra_spectral_gap = min(
            self.minimum_supra_spectral_gap, float(np.min(self.last_supra_gap))
        )
        self.minimum_conductance = min(
            self.minimum_conductance, float(np.min(conductance))
        )
        self.minimum_forman_curvature = min(
            self.minimum_forman_curvature, float(np.min(forman))
        )
        self.minimum_sheaf_eigenvalue = min(
            self.minimum_sheaf_eigenvalue, float(np.min(sheaf_minimum))
        )
        self.maximum_sheaf_energy = max(
            self.maximum_sheaf_energy, float(np.max(sheaf_energy))
        )
        self.maximum_bottleneck_risk = max(
            self.maximum_bottleneck_risk, float(np.max(bottleneck_risk))
        )
        factors = NetworkGeometryFactors(
            transition=transition,
            intensity=intensity,
            intensity_multiplier=intensity_multiplier,
            layer_weights=layer_weights,
            bottleneck_risk=bottleneck_risk,
            sheaf_consistency=sheaf_consistency,
            actor_spectral_gap=actor_gap,
            supra_spectral_gap=self.last_supra_gap.copy(),
            conductance=conductance,
            forman_curvature=forman,
        )
        self.last_factors = factors
        self.month += 1
        return factors

    def kernel_draw_score(self) -> np.ndarray:
        return np.mean(
            np.log(
                np.maximum(self.raw_kernel, 1.0e-12)
                / np.maximum(self.initial_kernel, 1.0e-12)
            ),
            axis=(1, 2, 3),
        )

    def diagnostics(self) -> dict:
        return {
            "enabled": True,
            "layers": list(LAYERS),
            "clock": "monthly same-path master clock",
            "spectral_refresh_interval_months": self.spectral_interval,
            "supra_refreshes": self.supra_refreshes,
            "max_transition_row_residual": self.max_transition_row_residual,
            "minimum_intensity": self.minimum_intensity,
            "maximum_intensity": self.maximum_intensity,
            "minimum_actor_spectral_gap": self.minimum_actor_spectral_gap,
            "minimum_supra_spectral_gap": self.minimum_supra_spectral_gap,
            "minimum_conductance": self.minimum_conductance,
            "minimum_forman_curvature": self.minimum_forman_curvature,
            "minimum_sheaf_eigenvalue": self.minimum_sheaf_eigenvalue,
            "maximum_sheaf_energy": self.maximum_sheaf_energy,
            "maximum_temporal_log_step": self.maximum_temporal_log_step,
            "maximum_bottleneck_risk": self.maximum_bottleneck_risk,
            "feature_flags": {
                "intensity_transition": self.intensity_transition_enabled,
                "supra_laplacian": self.supra_laplacian_enabled,
                "spectral_geometry": self.spectral_geometry_enabled,
                "forman_curvature": self.forman_curvature_enabled,
                "sheaf_laplacian": self.sheaf_laplacian_enabled,
            },
        }
