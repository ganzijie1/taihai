from __future__ import annotations

from collections import deque

import numpy as np


class RenormalizationScaleAuditV79:
    """Read-only coarse-graining and empirical scale-flow diagnostic.

    The audit never changes a path state or terminal hazard. It compares the
    actor-age field at 16, 8 and 4 nodes, preserves actor mass with frozen age
    weights, and estimates a low-dimensional scale-flow Jacobian only after a
    sufficient time history exists.
    """

    def __init__(
        self, age_weights: np.ndarray, history: int = 36,
        minimum_periods: int = 8, ridge: float = 1.0e-5,
    ) -> None:
        self.age_weights = np.asarray(age_weights, dtype=float)
        self.age_weights /= np.sum(self.age_weights)
        self.minimum_periods = minimum_periods
        self.ridge = ridge
        self.fine_history: deque[np.ndarray] = deque(maxlen=history)
        self.coarse_history: deque[np.ndarray] = deque(maxlen=history)
        self.updates = 0
        self.fit_updates = 0
        self.maximum_mass_residual = 0.0
        self.maximum_scale_residual = 0.0
        self.maximum_reconstruction_error = 0.0
        self.maximum_spectral_radius = 0.0
        self.relevant_directions = 0
        self.marginal_directions = 0
        self.irrelevant_directions = 0

    @staticmethod
    def _summary(values: np.ndarray) -> np.ndarray:
        flat = values.reshape(values.shape[0], -1)
        mean = np.mean(flat, axis=1)
        spread = np.std(flat, axis=1)
        energy = np.mean(np.square(np.diff(flat, axis=1)), axis=1)
        tail = np.mean(np.abs(flat) >= 0.55, axis=1)
        return np.stack((mean, spread, energy, tail), axis=1)

    def observe(self, density: np.ndarray, grid: np.ndarray) -> None:
        density = np.asarray(density, dtype=float)
        grid = np.asarray(grid, dtype=float)
        dx = float(grid[1] - grid[0])
        conditional_mass = np.sum(density, axis=-1) * dx
        mean = np.sum(density * grid[None, None, None, :], axis=-1) * dx

        pair_weights = np.array([
            np.sum(self.age_weights[:2]), np.sum(self.age_weights[2:])
        ])
        coarse8 = np.stack((
            np.sum(mean[:, :, :2] * self.age_weights[None, None, :2], axis=2)
            / max(pair_weights[0], 1.0e-12),
            np.sum(mean[:, :, 2:] * self.age_weights[None, None, 2:], axis=2)
            / max(pair_weights[1], 1.0e-12),
        ), axis=2)
        coarse4 = np.sum(
            mean * self.age_weights[None, None, :], axis=2
        )
        reconstructed = np.repeat(coarse8, 2, axis=2)

        fine_total = np.sum(mean * self.age_weights[None, None, :], axis=(1, 2))
        coarse_total = np.sum(coarse4, axis=1)
        mass_residual = float(np.max(np.abs(conditional_mass - 1.0)))
        scale_residual = float(np.max(np.abs(fine_total - coarse_total)))
        reconstruction = float(np.max(np.abs(mean - reconstructed)))
        self.maximum_mass_residual = max(self.maximum_mass_residual, mass_residual)
        self.maximum_scale_residual = max(self.maximum_scale_residual, scale_residual)
        self.maximum_reconstruction_error = max(
            self.maximum_reconstruction_error, reconstruction
        )

        fine_summary = np.mean(self._summary(mean), axis=0)
        coarse_summary = np.mean(self._summary(coarse8), axis=0)
        self.fine_history.append(fine_summary)
        self.coarse_history.append(coarse_summary)
        self.updates += 1
        if len(self.fine_history) < self.minimum_periods:
            return

        x = np.stack(tuple(self.fine_history), axis=0)
        y = np.stack(tuple(self.coarse_history), axis=0)
        x = x - np.mean(x, axis=0, keepdims=True)
        y = y - np.mean(y, axis=0, keepdims=True)
        gram = x.T @ x + self.ridge * np.eye(x.shape[1])
        jacobian = np.linalg.solve(gram, x.T @ y).T
        eigenvalues = np.linalg.eigvals(jacobian)
        modulus = np.abs(eigenvalues)
        self.maximum_spectral_radius = max(
            self.maximum_spectral_radius, float(np.max(modulus))
        )
        self.relevant_directions = int(np.sum(modulus > 1.05))
        self.marginal_directions = int(np.sum(
            (modulus >= 0.95) & (modulus <= 1.05)
        ))
        self.irrelevant_directions = int(np.sum(modulus < 0.95))
        self.fit_updates += 1

    def diagnostics(self) -> dict:
        return {
            "updates": self.updates,
            "jacobian_fit_updates": self.fit_updates,
            "resolutions": [16, 8, 4],
            "maximum_conditional_mass_residual": self.maximum_mass_residual,
            "maximum_scale_conservation_residual": self.maximum_scale_residual,
            "maximum_coarse_reconstruction_error": self.maximum_reconstruction_error,
            "maximum_scale_flow_spectral_radius": self.maximum_spectral_radius,
            "relevant_directions": self.relevant_directions,
            "marginal_directions": self.marginal_directions,
            "irrelevant_directions": self.irrelevant_directions,
            "role": "diagnostic and model-reduction evidence only",
            "terminal_probability_adjustment": False,
        }
