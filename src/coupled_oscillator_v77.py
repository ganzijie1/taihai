from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class OscillatorFactors:
    phase_compatibility: np.ndarray
    global_order: np.ndarray
    layer_order: np.ndarray
    desynchronization: np.ndarray


class CoupledOscillatorV77:
    """Inertial signed Sakaguchi--Kuramoto system on actor-layer clocks.

    Phases describe observable adjustment cycles rather than political
    preferences: information attention, production/logistics, identity
    mobilization and casualty response.  The signed directed Graphon supplies
    coupling, while physical/economic stress changes natural frequencies.
    """

    def __init__(self, paths: int, seed: int, layers: int = 4, actors: int = 4):
        self.paths = int(paths)
        self.layers = int(layers)
        self.actors = int(actors)
        rng = np.random.default_rng(seed)
        self.phase = rng.uniform(-np.pi, np.pi, (paths, layers, actors))
        periods = np.array([5.5, 11.0, 8.0, 6.5])
        self.base_frequency = (2.0 * np.pi / periods)[None, :, None]
        self.frequency = np.broadcast_to(
            self.base_frequency, self.phase.shape
        ).copy()
        self.inertia = np.array([0.45, 0.75, 0.60, 0.50])[None, :, None]
        self.damping = np.array([0.90, 0.72, 0.78, 0.84])[None, :, None]
        self.phase_lag = np.array([0.18, 0.10, 0.26, 0.22])[None, :, None, None]
        self.last_desynchronization = np.zeros(paths)
        self.updates = 0
        self.maximum_frequency = 0.0
        self.maximum_phase_velocity = 0.0
        self.minimum_global_order = 1.0
        self.maximum_global_order = 0.0
        self.maximum_cluster_order_gap = 0.0
        self.maximum_transverse_exponent = -np.inf
        self.minimum_transverse_exponent = np.inf
        self.maximum_phase_step = 0.0

    @staticmethod
    def _order(phase: np.ndarray, axis: int) -> np.ndarray:
        return np.abs(np.mean(np.exp(1j * phase), axis=axis))

    def update(self, magnitude: np.ndarray, sign: np.ndarray,
               layer_weights: np.ndarray, damage: np.ndarray,
               shortage: np.ndarray, network: np.ndarray,
               jump_signal: np.ndarray, common_noise: np.ndarray) -> OscillatorFactors:
        stress = np.clip(
            0.34 * damage + 0.26 * shortage + 0.22 * jump_signal
            + 0.10 * np.abs(common_noise) + 0.08 * (1.0 - network),
            0.0, 1.5,
        )
        layer_stress = np.stack((
            0.55 * stress + 0.45 * np.abs(common_noise),
            0.50 * stress + 0.50 * shortage,
            0.62 * stress + 0.38 * (1.0 - network),
            0.60 * stress + 0.40 * damage,
        ), axis=1)
        natural = self.base_frequency * np.clip(1.0 + 0.30 * layer_stress, 0.55, 1.65)
        signed = magnitude * sign
        row_scale = np.maximum(np.sum(np.abs(signed), axis=3, keepdims=True), 1.0e-12)
        coupling = signed / row_scale
        substeps = 4
        dt = 1.0 / substeps
        for _ in range(substeps):
            difference = (
                self.phase[:, :, None, :]
                - self.phase[:, :, :, None]
                - self.phase_lag
            )
            interaction = np.sum(coupling * np.sin(difference), axis=3)
            acceleration = (
                natural - self.damping * self.frequency
                + 0.82 * interaction
            ) / self.inertia
            self.frequency += dt * acceleration
            step = dt * self.frequency
            self.phase = (self.phase + step + np.pi) % (2.0 * np.pi) - np.pi
            self.maximum_phase_step = max(
                self.maximum_phase_step, float(np.max(np.abs(step)))
            )
        layer_order = self._order(self.phase, axis=2)
        weighted_complex = np.sum(
            layer_weights[:, :, None] * np.exp(1j * self.phase), axis=(1, 2)
        ) / self.actors
        global_order = np.abs(weighted_complex)
        desync = np.clip(1.0 - global_order, 0.0, 1.0)
        phase_difference = self.phase[:, :, None, :] - self.phase[:, :, :, None]
        compatibility = np.clip(
            0.62 + 0.38 * np.cos(phase_difference - self.phase_lag), 0.20, 1.0
        )

        jacobian_weight = magnitude * np.cos(phase_difference - self.phase_lag)
        jacobian_weight *= sign
        transverse = np.zeros((self.paths, self.layers))
        for path in range(self.paths):
            for layer in range(self.layers):
                weight = jacobian_weight[path, layer]
                laplacian = np.diag(np.sum(weight, axis=1)) - weight
                eigenvalues = np.linalg.eigvals(laplacian)
                ordered = np.sort(np.real(eigenvalues))
                transverse[path, layer] = -0.82 * ordered[1] / float(self.inertia[0, layer, 0])
        self.last_desynchronization = desync
        self.updates += 1
        self.maximum_frequency = max(self.maximum_frequency, float(np.max(np.abs(natural))))
        self.maximum_phase_velocity = max(self.maximum_phase_velocity, float(np.max(np.abs(self.frequency))))
        self.minimum_global_order = min(self.minimum_global_order, float(np.min(global_order)))
        self.maximum_global_order = max(self.maximum_global_order, float(np.max(global_order)))
        self.maximum_cluster_order_gap = max(
            self.maximum_cluster_order_gap,
            float(np.max(np.ptp(layer_order, axis=1))),
        )
        self.maximum_transverse_exponent = max(
            self.maximum_transverse_exponent, float(np.max(transverse))
        )
        self.minimum_transverse_exponent = min(
            self.minimum_transverse_exponent, float(np.min(transverse))
        )
        return OscillatorFactors(
            phase_compatibility=compatibility,
            global_order=global_order,
            layer_order=layer_order,
            desynchronization=desync,
        )

    def diagnostics(self) -> dict:
        return {
            "enabled": True,
            "model": "inertial signed directed Sakaguchi-Kuramoto",
            "updates": self.updates,
            "state_shape": [self.paths, self.layers, self.actors],
            "minimum_global_order": self.minimum_global_order,
            "maximum_global_order": self.maximum_global_order,
            "maximum_cluster_order_gap": self.maximum_cluster_order_gap,
            "maximum_natural_frequency": self.maximum_frequency,
            "maximum_phase_velocity": self.maximum_phase_velocity,
            "maximum_phase_step": self.maximum_phase_step,
            "minimum_transverse_lyapunov_proxy": self.minimum_transverse_exponent,
            "maximum_transverse_lyapunov_proxy": self.maximum_transverse_exponent,
            "diagnostics": [
                "global and layer order parameters",
                "cluster-order gap",
                "signed weighted master-stability transverse exponent",
            ],
            "interpretation_boundary": "latent adjustment-cycle phase; not asserted as a literal political oscillator",
            "terminal_probability_adjustment": False,
        }
