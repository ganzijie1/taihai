from __future__ import annotations

import numpy as np

from dynamic_macro_equilibrium_v47 import DynamicMultiRegionalCGE


class NestedCESCGEV77(DynamicMultiRegionalCGE):
    """Two-level CES production with actor-sector wartime substitution."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        factor_sigma_sector = np.array([
            0.42, 0.58, 0.48, 0.62, 0.92, 0.78, 0.55, 0.70, 0.82, 0.50,
        ])
        logistics_sigma_sector = np.array([
            0.28, 0.34, 0.30, 0.38, 0.66, 0.58, 0.36, 0.46, 0.54, 0.32,
        ])
        self.factor_sigma = factor_sigma_sector[self.sector]
        self.logistics_sigma = logistics_sigma_sector[self.sector]
        value_share = np.clip(
            self.labor_share + self.capital_share, 0.05, 0.95
        )
        self.capital_weight = np.clip(
            self.capital_share / value_share, 0.05, 0.95
        )
        self.value_added_weight = value_share
        self.maximum_ces_limit_residual = 0.0
        self.maximum_ces_substitution_effect = 0.0
        self.minimum_nested_capacity = np.inf
        self.nested_ces_updates = 0

    @staticmethod
    def _ces(left: np.ndarray, right: np.ndarray, weight: np.ndarray,
             sigma: np.ndarray) -> np.ndarray:
        left = np.maximum(left, 1.0e-12)
        right = np.maximum(right, 1.0e-12)
        rho = (sigma - 1.0) / np.maximum(sigma, 1.0e-12)
        near_cobb = np.abs(rho) < 1.0e-6
        power = (
            weight[None, :] * left ** rho[None, :]
            + (1.0 - weight[None, :]) * right ** rho[None, :]
        )
        result = power ** (1.0 / rho[None, :])
        if np.any(near_cobb):
            result[:, near_cobb] = (
                left[:, near_cobb] ** weight[None, near_cobb]
                * right[:, near_cobb] ** (1.0 - weight[None, near_cobb])
            )
        return result

    def _production_capacity(self, productivity, labor_node,
                             logistics_node, exogenous_shock):
        effective_labor = np.minimum(self.labor, labor_node)
        value_added = self._ces(
            self.capital, effective_labor, self.capital_weight, self.factor_sigma
        )
        nested = self._ces(
            value_added, logistics_node, self.value_added_weight,
            self.logistics_sigma,
        )
        capacity = np.clip(
            productivity * nested * (1.0 - exogenous_shock), 0.01, 1.25
        )
        cobb_limit = (
            self.capital ** self.capital_weight[None, :]
            * effective_labor ** (1.0 - self.capital_weight[None, :])
        )
        self.maximum_ces_limit_residual = max(
            self.maximum_ces_limit_residual,
            float(np.max(np.abs(
                self._ces(
                    self.capital, effective_labor, self.capital_weight,
                    np.full_like(self.factor_sigma, 1.0 + 1.0e-7),
                ) - cobb_limit
            ))),
        )
        baseline = super()._production_capacity(
            productivity, labor_node, logistics_node, exogenous_shock
        )
        self.maximum_ces_substitution_effect = max(
            self.maximum_ces_substitution_effect,
            float(np.max(np.abs(capacity - baseline))),
        )
        self.minimum_nested_capacity = min(
            self.minimum_nested_capacity, float(np.min(capacity))
        )
        self.nested_ces_updates += 1
        return capacity

    def extension_diagnostics(self) -> dict:
        return {"v77_nested_ces": {
            "updates": self.nested_ces_updates,
            "factor_nest": "capital-labor CES",
            "upper_nest": "value-added-logistics CES",
            "minimum_factor_sigma": float(np.min(self.factor_sigma)),
            "maximum_factor_sigma": float(np.max(self.factor_sigma)),
            "minimum_logistics_sigma": float(np.min(self.logistics_sigma)),
            "maximum_logistics_sigma": float(np.max(self.logistics_sigma)),
            "maximum_cobb_douglas_limit_residual": self.maximum_ces_limit_residual,
            "maximum_capacity_change_from_baseline": self.maximum_ces_substitution_effect,
            "minimum_nested_capacity": self.minimum_nested_capacity,
            "identification": "sector-specific structural prior; retained in sensitivity ledger",
            "terminal_probability_adjustment": False,
        }}
