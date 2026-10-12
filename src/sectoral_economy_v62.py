from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from sectoral_economy_v61 import (
    CoupledSectoralEconomicSystem,
    SectoralFactors,
    ThreeSectorProductionState,
)


ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "data" / "sectoral_v62" / "employment_share_ledger.json"


def _official_labor_shares() -> np.ndarray:
    payload = json.loads(LEDGER.read_text(encoding="utf-8"))
    rows = []
    for actor in payload["actor_order"]:
        values = np.asarray(payload["observations"][actor]["values"], dtype=float)
        rows.append(values / values.sum())
    return np.asarray(rows)


class ThreeSectorProductionStateV62(ThreeSectorProductionState):
    """V6.2 structural transformation with unbalanced labor transport.

    IO/SAM value-added shares remain the accounting weights. Official
    employment shares initialize labor. Slow peacetime transformation and
    fast wartime reallocation are separate flows and are reconciled before
    the within-month physical clearing.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.baseline_labor_share = _official_labor_shares()
        self.labor_share = np.broadcast_to(
            self.baseline_labor_share, (self.paths, 4, 3)
        ).copy()
        self.slow_labor_flow = np.zeros((self.paths, 4, 3, 3))
        self.fast_labor_flow = np.zeros_like(self.slow_labor_flow)
        self.cumulative_industrial_log_output = np.zeros((self.paths, 4))
        self.kaldor_productivity = np.ones((self.paths, 4))
        self.max_uot_marginal_residual = 0.0
        self.max_uot_mass_change = 0.0
        self.max_monthly_labor_move = 0.0
        self.shift_share_within = np.zeros((self.paths, 4))
        self.shift_share_between = np.zeros((self.paths, 4))
        self.minimum_structural_multiplier = 1.0
        self.maximum_structural_multiplier = 1.0

    @staticmethod
    def _unbalanced_sinkhorn(source: np.ndarray, target: np.ndarray,
                             cost: np.ndarray, epsilon: float,
                             mass_penalty: float,
                             iterations: int = 18) -> np.ndarray:
        kernel = np.exp(-cost / max(epsilon, 1.0e-6))
        u = np.ones_like(source)
        v = np.ones_like(target)
        exponent = mass_penalty / (mass_penalty + epsilon)
        for _ in range(iterations):
            kv = np.einsum("...ij,...j->...i", kernel, v)
            u = (source / np.maximum(kv, 1.0e-14)) ** exponent
            ktu = np.einsum("...ji,...j->...i", kernel, u)
            v = (target / np.maximum(ktu, 1.0e-14)) ** exponent
        return u[..., :, None] * kernel * v[..., None, :]

    def _labor_update(self, group_demand: np.ndarray, prior_output: np.ndarray,
                      actor_logistics: np.ndarray,
                      actor_panic: np.ndarray) -> np.ndarray:
        if self.labor_reallocation_enabled:
            opening = self.labor_share.copy()
            scarcity = np.clip(group_demand - prior_output, -0.8, 1.5)
            productivity = prior_output / np.maximum(opening, 0.025)
            attractiveness = (
                0.46 * scarcity
                + 0.28 * np.log(np.maximum(productivity, 1.0e-5))
                - 0.24 * actor_panic[:, :, None]
            )
            logits = np.log(np.maximum(
                self.baseline_labor_share[None, :, :], 1.0e-10
            )) + attractiveness
            logits -= np.max(logits, axis=2, keepdims=True)
            target = np.exp(logits)
            target /= np.maximum(target.sum(axis=2, keepdims=True), 1.0e-14)

            training = np.array([
                [0.00, 0.42, 0.38],
                [0.48, 0.00, 0.30],
                [0.36, 0.34, 0.00],
            ])[None, None, :, :]
            logistics_cost = (
                0.32 * (1.0 - actor_logistics) + 0.22 * actor_panic
            )[:, :, None, None]
            cost = training + logistics_cost
            slow_plan = self._unbalanced_sinkhorn(
                opening, target, cost, epsilon=0.10, mass_penalty=0.85
            )
            fast_target = np.maximum(target * (1.0 + 0.65 * np.maximum(scarcity, 0.0)), 1e-8)
            fast_target /= fast_target.sum(axis=2, keepdims=True)
            fast_plan = self._unbalanced_sinkhorn(
                opening, fast_target, 0.72 * cost,
                epsilon=0.14, mass_penalty=0.55,
            )
            slow_destination = slow_plan.sum(axis=2)
            fast_destination = fast_plan.sum(axis=2)
            slow_destination /= np.maximum(
                slow_destination.sum(axis=2, keepdims=True), 1.0e-14
            )
            fast_destination /= np.maximum(
                fast_destination.sum(axis=2, keepdims=True), 1.0e-14
            )
            slow_speed = 0.008
            fast_speed = np.clip(
                0.015 + 0.070 * actor_logistics * (1.0 - 0.70 * actor_panic),
                0.006, 0.085,
            )
            proposed = (
                opening
                + slow_speed * (slow_destination - opening)
                + fast_speed[:, :, None] * (fast_destination - opening)
            )
            proposed = np.maximum(proposed, 1.0e-7)
            proposed /= proposed.sum(axis=2, keepdims=True)
            self.slow_labor_flow = slow_plan
            self.fast_labor_flow = fast_plan
            raw_mass = fast_plan.sum(axis=(2, 3))
            self.max_uot_mass_change = max(
                self.max_uot_mass_change,
                float(np.max(np.abs(raw_mass - 1.0))),
            )
            row_residual = np.max(np.abs(
                slow_plan.sum(axis=3) - opening
            ))
            self.max_uot_marginal_residual = max(
                self.max_uot_marginal_residual, float(row_residual)
            )
            self.max_monthly_labor_move = max(
                self.max_monthly_labor_move,
                float(np.max(np.abs(proposed - opening))),
            )
            self.labor_share = proposed
        factor = self.labor_share / np.maximum(
            self.baseline_labor_share[None, :, :], 1.0e-8
        )
        residual = float(np.max(np.abs(self.labor_share.sum(axis=2) - 1.0)))
        self.max_labor_share_residual = max(self.max_labor_share_residual, residual)
        return np.clip(factor, 0.35, 1.80)

    def update(self, **kwargs) -> SectoralFactors:
        opening_labor = self.labor_share.copy()
        factors = super().update(**kwargs)
        output = np.maximum(factors.actor_sector_output, 1.0e-6)
        income = np.einsum("pag,ag->pa", output, self.actor_group_output_share)
        subsistence = np.clip(0.16 / np.maximum(income, 0.18), 0.0, 0.35)
        engel = np.stack((
            1.0 + 0.32 * subsistence,
            1.0 + 0.05 * (1.0 - income),
            1.0 - 0.18 * subsistence,
        ), axis=2)
        industrial_growth = np.log(np.maximum(output[:, :, 1], 1.0e-6))
        self.cumulative_industrial_log_output = (
            0.985 * self.cumulative_industrial_log_output + industrial_growth
        )
        self.kaldor_productivity = np.clip(
            np.exp(0.018 * self.cumulative_industrial_log_output), 0.82, 1.18
        )
        baumol = np.clip(1.0 - 0.10 * self.baumol_cost_pressure, 0.86, 1.0)
        structural = engel.copy()
        structural[:, :, 1] *= self.kaldor_productivity
        structural[:, :, 2] *= baumol
        structural = np.clip(structural, 0.78, 1.22)
        node_multiplier = self._expand_actor_group(structural)
        self.node_realization = np.clip(
            self.node_realization * node_multiplier, 0.02, 1.25
        )
        self.sector_output = np.clip(self.sector_output * structural, 0.01, 1.35)
        self.sector_capacity = np.clip(self.sector_capacity * structural, 0.01, 1.35)
        productivity = output / np.maximum(opening_labor, 1.0e-6)
        labor_change = self.labor_share - opening_labor
        self.shift_share_between = np.sum(labor_change * productivity, axis=2)
        self.shift_share_within = np.sum(
            opening_labor * (output / np.maximum(factors.actor_sector_output, 1e-6) - 1.0),
            axis=2,
        )
        self.minimum_structural_multiplier = min(
            self.minimum_structural_multiplier, float(np.min(structural))
        )
        self.maximum_structural_multiplier = max(
            self.maximum_structural_multiplier, float(np.max(structural))
        )
        self.last_factors = replace(
            factors,
            actor_sector_output=self.sector_output.copy(),
            actor_sector_capacity=self.sector_capacity.copy(),
            node_realization=self.node_realization.copy(),
        )
        return self.last_factors

    def diagnostics(self) -> dict:
        result = super().diagnostics()
        result["v62_structural_transformation"] = {
            "employment_share_source": str(LEDGER),
            "employment_and_value_added_separated": True,
            "baseline_labor_share": self.baseline_labor_share.tolist(),
            "value_added_share": self.actor_group_output_share.tolist(),
            "nonhomothetic_demand": "Stone-Geary/Engel bounded same-budget reweighting",
            "labor_transport": "two-speed entropic unbalanced OT",
            "kaldor_verdoorn": "cumulative bounded industrial productivity",
            "baumol": "service relative-cost capacity pressure",
            "shift_share_role": "diagnostic only",
            "max_uot_marginal_residual": self.max_uot_marginal_residual,
            "max_uot_mass_change_before_renormalization": self.max_uot_mass_change,
            "max_monthly_labor_share_move": self.max_monthly_labor_move,
            "minimum_structural_multiplier": self.minimum_structural_multiplier,
            "maximum_structural_multiplier": self.maximum_structural_multiplier,
        }
        result["state_owners"].update({
            "official_sectoral_employment_shares": "ThreeSectorProductionStateV62",
            "unbalanced_labor_transport": "ThreeSectorProductionStateV62",
            "io_sam_value_added_shares": "DynamicMultiRegionalCGE",
        })
        return result


class CoupledSectoralEconomicSystemV62(CoupledSectoralEconomicSystem):
    def __init__(self, paths: int, seed: int, scenario: str,
                 dqsfc_enabled: bool = True,
                 econometric_draws: dict[str, np.ndarray] | None = None, **kwargs):
        super().__init__(
            paths, seed, scenario, dqsfc_enabled, econometric_draws, **kwargs
        )
        self.sectoral = ThreeSectorProductionStateV62(
            self.cge, paths, seed + 620_062,
            agriculture_enabled=kwargs.get("agriculture_enabled", True),
            industry_enabled=kwargs.get("industry_enabled", True),
            services_enabled=kwargs.get("services_enabled", True),
            labor_reallocation_enabled=kwargs.get("labor_reallocation_enabled", True),
            structural_transformation_enabled=kwargs.get(
                "structural_transformation_enabled", True
            ),
        )
        if self.sectoral_layer_enabled and self.sectoral.services_enabled:
            self.dqsfc.inventory[:, self.sectoral.service_node_mask] = 0.0
            self.dqsfc.inventory_target[self.sectoral.service_node_mask] = 0.0
            self.dqsfc.spoilage[self.sectoral.service_node_mask] = 1.0
