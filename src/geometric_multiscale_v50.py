from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class GeometricMultiscaleFactors:
    consistency: np.ndarray
    combat_factor: np.ndarray
    macro_logistics_factor: np.ndarray
    fast_component: np.ndarray
    slow_component: np.ndarray
    aggregation_residual: float


class GeometricMultiscaleCoupler:
    """Same-path fast/slow geometry linking tactical and macro states.

    The coupler does not own physical stocks. It compares five independently
    owned scales, evolves an explicit slow manifold, and returns bounded
    consistency factors consumed by both campaign tempo and macro logistics.
    """

    def __init__(self, paths: int):
        self.paths = paths
        self.slow_state = np.ones((paths, 4))
        self.last_factors = GeometricMultiscaleFactors(
            consistency=np.ones((paths, 4)),
            combat_factor=np.ones((paths, 4)),
            macro_logistics_factor=np.ones((paths, 4)),
            fast_component=np.zeros((paths, 4)),
            slow_component=self.slow_state.copy(),
            aggregation_residual=0.0,
        )
        self.max_aggregation_residual = 0.0
        self.months = 0

    def current(self) -> GeometricMultiscaleFactors:
        return self.last_factors

    def update(
        self, *, front_rear_integrity: np.ndarray, sector_output: np.ndarray,
        financial_capacity: np.ndarray, command: np.ndarray,
        energy_logistics: np.ndarray,
    ) -> GeometricMultiscaleFactors:
        logistics = np.clip(np.min(front_rear_integrity, axis=2), 0.02, 1.20)
        output = np.clip(np.mean(sector_output, axis=2), 0.02, 1.30)
        finance = np.clip(financial_capacity, 0.02, 1.10)
        command_level = np.clip(command, 0.02, 1.0)
        energy = np.clip(energy_logistics, 0.02, 1.0)
        scales = np.stack((logistics, output, finance, command_level, energy), axis=2)
        log_scales = np.log(np.maximum(scales, 1.0e-8))
        geometric_mean = np.exp(np.mean(log_scales, axis=2))
        dispersion = np.std(log_scales, axis=2)
        consistency = np.exp(-dispersion)

        opening_slow = self.slow_state.copy()
        self.slow_state = np.clip(
            0.94 * self.slow_state + 0.06 * geometric_mean, 0.04, 1.15
        )
        fast = geometric_mean - self.slow_state
        reconstructed = self.slow_state + fast
        aggregation_residual = float(np.max(np.abs(reconstructed - geometric_mean)))
        self.max_aggregation_residual = max(
            self.max_aggregation_residual, aggregation_residual,
            float(np.max(np.abs(
                self.slow_state - (0.94 * opening_slow + 0.06 * geometric_mean)
            ))),
        )
        combat_factor = np.clip(
            0.88 + 0.08 * consistency + 0.04 * self.slow_state, 0.82, 1.04
        )
        macro_factor = np.clip(
            0.90 + 0.07 * consistency + 0.03 * np.clip(geometric_mean, 0.0, 1.2),
            0.84, 1.04,
        )
        self.months += 1
        self.last_factors = GeometricMultiscaleFactors(
            consistency=consistency,
            combat_factor=combat_factor,
            macro_logistics_factor=macro_factor,
            fast_component=fast,
            slow_component=self.slow_state.copy(),
            aggregation_residual=aggregation_residual,
        )
        return self.last_factors

    def diagnostics(self) -> dict:
        return {
            "months": self.months,
            "max_aggregation_residual": self.max_aggregation_residual,
            "minimum_consistency": float(np.min(self.last_factors.consistency)),
            "minimum_combat_factor": float(np.min(self.last_factors.combat_factor)),
            "minimum_macro_logistics_factor": float(
                np.min(self.last_factors.macro_logistics_factor)
            ),
            "clock": "monthly_same_path",
        }
