from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

import simulate_trade_war_v43 as trade
from dynamic_macro_dqsfc_v53 import CoupledMRRDCGEDQSFCEconomicSystem
from dynamic_macro_equilibrium_v47 import CoupledMacroEconomicSystem, MacroFactors


SECTOR_GROUPS = ("agriculture", "industry", "services")
AGRICULTURE_SECTORS = ("primary", "food_light")
INDUSTRY_SECTORS = (
    "energy_utilities", "materials", "electronics",
    "machinery_transport", "construction",
)
SERVICE_SECTORS = ("trade_transport", "business_services", "public_services")


@dataclass(frozen=True)
class SectoralFactors:
    actor_sector_output: np.ndarray
    actor_sector_capacity: np.ndarray
    actor_sector_labor: np.ndarray
    food_security: np.ndarray
    industrial_operability: np.ndarray
    service_utilization: np.ndarray
    service_wait: np.ndarray
    node_realization: np.ndarray
    agriculture_land_share: np.ndarray
    industrial_firm_mass: np.ndarray
    industrial_quality: np.ndarray
    digital_adoption: np.ndarray
    public_service_capacity: np.ndarray


class ThreeSectorProductionState:
    """Sector-specific physical state layered on the existing IO-SAM core.

    The CGE remains the owner of installed capital and equilibrium plans. The
    DQ-SFC layer remains the owner of commodity inventories, order backlogs,
    transaction prices and settlement. This state owns only biological yield,
    water and soil condition, industrial equipment operability, non-storable
    service queues, and sectoral labor shares.
    """

    def __init__(
        self,
        cge,
        paths: int,
        seed: int,
        *,
        agriculture_enabled: bool = True,
        industry_enabled: bool = True,
        services_enabled: bool = True,
        labor_reallocation_enabled: bool = True,
        structural_transformation_enabled: bool = True,
    ):
        self.paths = paths
        self.rng = np.random.default_rng(seed)
        self.agriculture_enabled = agriculture_enabled
        self.industry_enabled = industry_enabled
        self.services_enabled = services_enabled
        self.labor_reallocation_enabled = labor_reallocation_enabled
        self.structural_transformation_enabled = structural_transformation_enabled
        self.actor_of_node = cge.actor_of_node.copy()
        self.sector_of_node = cge.sector.copy()
        self.n_nodes = cge.n_nodes
        self.n_actors = 4

        agriculture = np.isin(
            self.sector_of_node,
            [trade.SECTORS.index(name) for name in AGRICULTURE_SECTORS],
        )
        industry = np.isin(
            self.sector_of_node,
            [trade.SECTORS.index(name) for name in INDUSTRY_SECTORS],
        )
        services = np.isin(
            self.sector_of_node,
            [trade.SECTORS.index(name) for name in SERVICE_SECTORS],
        )
        self.group_of_node = np.select(
            (agriculture, industry, services), (0, 1, 2), default=2
        ).astype(int)
        self.service_node_mask = services

        self.group_weights = np.zeros((4, 3, self.n_nodes))
        self.actor_node_weights = np.zeros((4, self.n_nodes))
        observed_group_share = np.zeros((4, 3))
        for actor in range(4):
            actor_mask = self.actor_of_node == actor
            actor_va = np.maximum(cge.va[actor_mask], 0.0)
            self.actor_node_weights[actor, actor_mask] = (
                actor_va / max(actor_va.sum(), 1.0e-12)
            )
            for group in range(3):
                mask = actor_mask & (self.group_of_node == group)
                weights = np.maximum(cge.va[mask], 0.0)
                self.group_weights[actor, group, mask] = (
                    weights / max(weights.sum(), 1.0e-12)
                )
                observed_group_share[actor, group] = weights.sum()
            observed_group_share[actor] /= max(
                observed_group_share[actor].sum(), 1.0e-12
            )
        # Preserve observed value-added shares for exact sector-to-actor
        # accounting. A separately regularized copy is used only by the labor
        # reallocation law so very small sectors remain numerically reachable.
        self.actor_group_output_share = observed_group_share.copy()
        self.baseline_labor_share = np.clip(observed_group_share, 0.03, 0.94)
        self.baseline_labor_share /= self.baseline_labor_share.sum(axis=1, keepdims=True)
        self.labor_share = np.broadcast_to(
            self.baseline_labor_share, (paths, 4, 3)
        ).copy()

        self.weather = np.zeros((paths, 4))
        self.water = np.ones((paths, 4))
        self.soil = np.ones((paths, 4))
        self.biological_yield = np.ones((paths, 4))
        self.agriculture_land_share = np.broadcast_to(
            np.array([0.55, 0.25, 0.20]), (paths, 4, 3)
        ).copy()
        self.industrial_operability = np.ones((paths, 4))
        self.industrial_repair_queue = np.zeros((paths, 4))
        self.industrial_firm_mass = np.ones((paths, 4))
        self.industrial_quality = np.ones((paths, 4))
        self.industrial_learning = np.ones((paths, 4))
        self.service_capacity = np.ones((paths, 4))
        self.service_queue = np.zeros((paths, 4))
        self.service_utilization = np.zeros((paths, 4))
        self.service_wait = np.zeros((paths, 4))
        self.digital_adoption = np.full((paths, 4), 0.46)
        self.public_service_capacity = np.ones((paths, 4))
        self.baumol_cost_pressure = np.zeros((paths, 4))
        self.sector_output = np.ones((paths, 4, 3))
        self.sector_capacity = np.ones((paths, 4, 3))
        self.node_realization = np.ones((paths, self.n_nodes))
        self.expired_service = np.zeros((paths, 4))

        self.months = 0
        self.max_labor_share_residual = 0.0
        self.max_service_stock_residual = 0.0
        self.max_node_bound_violation = 0.0
        self.max_sector_accounting_residual = 0.0
        self.minimum_water = 1.0
        self.minimum_soil = 1.0
        self.minimum_industrial_operability = 1.0
        self.maximum_service_utilization = 0.0
        self.maximum_service_wait = 0.0
        self.maximum_expired_service = 0.0
        self.max_land_share_residual = 0.0
        self.minimum_firm_mass = 1.0
        self.minimum_industrial_quality = 1.0
        self.maximum_baumol_cost_pressure = 0.0
        self.last_factors = SectoralFactors(
            actor_sector_output=self.sector_output.copy(),
            actor_sector_capacity=self.sector_capacity.copy(),
            actor_sector_labor=np.ones((paths, 4, 3)),
            food_security=np.ones((paths, 4)),
            industrial_operability=self.industrial_operability.copy(),
            service_utilization=self.service_utilization.copy(),
            service_wait=self.service_wait.copy(),
            node_realization=self.node_realization.copy(),
            agriculture_land_share=self.agriculture_land_share.copy(),
            industrial_firm_mass=self.industrial_firm_mass.copy(),
            industrial_quality=self.industrial_quality.copy(),
            digital_adoption=self.digital_adoption.copy(),
            public_service_capacity=self.public_service_capacity.copy(),
        )

    def _aggregate_group(self, node_values: np.ndarray) -> np.ndarray:
        return np.einsum("pn,agn->pag", node_values, self.group_weights)

    def _expand_actor_group(self, values: np.ndarray) -> np.ndarray:
        expanded = np.ones((self.paths, self.n_nodes))
        directly_modelled = self.actor_of_node < 4
        expanded[:, directly_modelled] = values[
            :, self.actor_of_node[directly_modelled], self.group_of_node[directly_modelled]
        ]
        if np.any(~directly_modelled):
            spill = 1.0 + 0.18 * np.mean(values - 1.0, axis=1)
            expanded[:, ~directly_modelled] = spill[:, self.group_of_node[~directly_modelled]]
        return expanded

    def _labor_update(
        self,
        group_demand: np.ndarray,
        prior_output: np.ndarray,
        actor_logistics: np.ndarray,
        actor_panic: np.ndarray,
    ) -> np.ndarray:
        if self.labor_reallocation_enabled:
            scarcity = np.clip(group_demand - prior_output, -0.8, 1.5)
            logits = (
                np.log(np.maximum(self.baseline_labor_share[None, :, :], 1.0e-9))
                + 0.65 * scarcity
            )
            logits -= np.max(logits, axis=2, keepdims=True)
            target = np.exp(logits)
            target /= np.maximum(target.sum(axis=2, keepdims=True), 1.0e-12)
            mobility = np.clip(
                0.025 + 0.075 * actor_logistics * (1.0 - 0.65 * actor_panic),
                0.01,
                0.10,
            )
            self.labor_share += mobility[:, :, None] * (target - self.labor_share)
            self.labor_share = np.maximum(self.labor_share, 1.0e-6)
            self.labor_share /= self.labor_share.sum(axis=2, keepdims=True)
        labor_factor = self.labor_share / np.maximum(
            self.baseline_labor_share[None, :, :], 1.0e-8
        )
        residual = float(np.max(np.abs(self.labor_share.sum(axis=2) - 1.0)))
        self.max_labor_share_residual = max(self.max_labor_share_residual, residual)
        return np.clip(labor_factor, 0.35, 1.80)

    def update(
        self,
        *,
        month: int,
        cge,
        inventory: np.ndarray,
        inventory_target: np.ndarray,
        actor_damage: np.ndarray,
        actor_logistics: np.ndarray,
        actor_labor: np.ndarray,
        actor_financial_stress: np.ndarray,
        actor_payment: np.ndarray,
        actor_fiscal: np.ndarray,
        actor_social_action: np.ndarray,
        actor_panic: np.ndarray,
        actor_mobilization: np.ndarray,
        blockade: np.ndarray,
    ) -> SectoralFactors:
        group_demand = np.clip(self._aggregate_group(cge.last_demand), 0.01, 2.50)
        prior_output = np.clip(self._aggregate_group(cge.last_output), 0.01, 1.50)
        if self.structural_transformation_enabled:
            # Engel demand migration is relative, not a new final-demand account.
            income_proxy = np.clip(
                np.einsum(
                    "pag,ag->pa", prior_output, self.actor_group_output_share
                ),
                0.08,
                1.35,
            )
            group_demand[:, :, 0] *= np.clip(income_proxy ** -0.16, 0.82, 1.30)
            group_demand[:, :, 2] *= np.clip(income_proxy ** 0.11, 0.78, 1.16)
        labor_factor = self._labor_update(
            group_demand, prior_output, actor_logistics, actor_panic
        )
        effective_labor = np.clip(
            actor_labor[:, :, None] * labor_factor, 0.05, 1.35
        )

        if self.agriculture_enabled:
            common_weather = self.rng.normal(0.0, 0.055, (self.paths, 1))
            local_weather = self.rng.normal(0.0, 0.035, (self.paths, 4))
            self.weather = 0.72 * self.weather + common_weather + local_weather
            water_target = np.clip(
                0.42 + 0.28 * actor_logistics + 0.18 * actor_payment
                + 0.12 * actor_fiscal - 0.35 * actor_damage
                - 0.12 * blockade[:, None],
                0.18,
                1.10,
            )
            self.water = np.clip(
                self.water + 0.09 * (water_target - self.water)
                - 0.012 * np.maximum(self.weather, 0.0),
                0.12,
                1.12,
            )
            self.soil = np.clip(
                self.soil + 0.0025 * (1.0 - self.soil)
                - 0.0035 * actor_damage - 0.0015 * actor_mobilization,
                0.35,
                1.05,
            )
            land_returns = np.stack((
                0.55 + 0.35 * actor_panic,
                0.58 + 0.18 * actor_payment - 0.28 * blockade[:, None],
                0.52 + 0.24 * actor_logistics - 0.16 * actor_financial_stress,
            ), axis=2)
            average_return = np.sum(
                self.agriculture_land_share * land_returns, axis=2, keepdims=True
            )
            self.agriculture_land_share *= np.exp(
                0.018 * (land_returns - average_return)
            )
            self.agriculture_land_share /= np.maximum(
                self.agriculture_land_share.sum(axis=2, keepdims=True), 1.0e-12
            )
            land_productivity = np.sum(
                self.agriculture_land_share
                * np.array([1.00, 1.06, 0.94])[None, None, :], axis=2
            )
            season = 1.0 + 0.035 * np.sin(2.0 * np.pi * (month % 12) / 12.0)
            weather_factor = np.exp(
                -0.10 * np.abs(self.weather) - 0.08 * self.weather ** 2
            )
            self.biological_yield = np.clip(
                season * land_productivity * weather_factor
                * self.water ** 0.24 * self.soil ** 0.14
                * effective_labor[:, :, 0] ** 0.24
                * actor_logistics ** 0.18
                * (1.0 - 0.42 * actor_damage),
                0.10,
                1.25,
            )
        else:
            self.biological_yield.fill(1.0)

        industrial_nodes = self.group_of_node == 1
        inventory_ratio_node = np.clip(
            inventory / np.maximum(inventory_target[None, :], 1.0e-6), 0.0, 2.0
        )
        industrial_input = np.clip(
            self._aggregate_group(
                np.where(industrial_nodes[None, :], inventory_ratio_node, 0.0)
            )[:, :, 1],
            0.10,
            1.25,
        )
        if self.industry_enabled:
            new_damage = np.clip(
                0.080 * actor_damage + 0.025 * actor_financial_stress
                + 0.018 * blockade[:, None],
                0.0,
                0.18,
            )
            self.industrial_repair_queue = np.clip(
                self.industrial_repair_queue + new_damage
                - 0.035 * actor_logistics * actor_payment * actor_fiscal,
                0.0,
                1.50,
            )
            repair = np.minimum(
                self.industrial_repair_queue,
                0.025 * actor_logistics * actor_payment
                * (0.45 + 0.55 * actor_fiscal),
            )
            self.industrial_repair_queue -= repair
            self.industrial_operability = np.clip(
                self.industrial_operability + repair
                + 0.012 * (1.0 - self.industrial_operability)
                - new_damage,
                0.08,
                1.10,
            )
            entry_hazard = np.clip(
                0.012 * actor_payment * actor_fiscal
                * np.maximum(group_demand[:, :, 1] - prior_output[:, :, 1], 0.0),
                0.0,
                0.025,
            )
            exit_hazard = np.clip(
                0.004 + 0.035 * actor_financial_stress + 0.045 * actor_damage
                + 0.018 * blockade[:, None],
                0.0,
                0.10,
            )
            self.industrial_firm_mass = np.clip(
                self.industrial_firm_mass * (1.0 + entry_hazard - exit_hazard)
                + 0.006 * (1.0 - self.industrial_firm_mass),
                0.15,
                1.18,
            )
            output_growth = prior_output[:, :, 1] / np.maximum(
                self.industrial_learning, 1.0e-8
            ) - 1.0
            self.industrial_learning = np.clip(
                0.96 * self.industrial_learning + 0.04 * prior_output[:, :, 1],
                0.10,
                1.30,
            )
            self.industrial_quality = np.clip(
                self.industrial_quality
                + 0.006 * np.maximum(output_growth, -0.25)
                + 0.003 * actor_fiscal * actor_payment
                - 0.012 * actor_damage,
                0.55,
                1.20,
            )
            industrial_capacity = np.minimum.reduce((
                self.industrial_operability,
                np.clip(industrial_input, 0.08, 1.10),
                np.clip(effective_labor[:, :, 1], 0.08, 1.15),
                np.clip(
                    actor_logistics * actor_payment
                    * (1.0 - 0.42 * actor_financial_stress),
                    0.08,
                    1.10,
                ),
            ))
            industrial_capacity = np.clip(
                industrial_capacity
                * self.industrial_firm_mass ** 0.12
                * self.industrial_quality ** 0.14,
                0.06,
                1.18,
            )
        else:
            self.industrial_operability.fill(1.0)
            industrial_capacity = np.ones((self.paths, 4))

        if self.services_enabled:
            demand_pressure = np.clip(
                group_demand[:, :, 2]
                * (1.0 + 0.18 * actor_panic + 0.12 * actor_social_action),
                0.02,
                2.50,
            )
            capacity_target = np.clip(
                effective_labor[:, :, 2] * actor_logistics
                * (1.0 - 0.48 * actor_damage)
                * (1.0 - 0.28 * actor_financial_stress)
                * (0.55 + 0.45 * actor_payment),
                0.08,
                1.20,
            )
            adoption_flow = (
                0.055 * self.digital_adoption * (1.0 - self.digital_adoption)
                * actor_logistics * actor_payment
                - 0.025 * actor_damage * self.digital_adoption
            )
            self.digital_adoption = np.clip(
                self.digital_adoption + adoption_flow, 0.05, 0.96
            )
            self.public_service_capacity = np.clip(
                self.public_service_capacity
                + 0.08 * (
                    actor_fiscal * actor_payment * (1.0 - 0.35 * actor_damage)
                    - self.public_service_capacity
                ),
                0.10,
                1.12,
            )
            service_productivity = np.clip(
                0.62 + 0.26 * self.digital_adoption
                + 0.12 * self.public_service_capacity,
                0.30,
                1.10,
            )
            industrial_productivity = np.clip(
                self.industrial_quality * self.industrial_operability, 0.10, 1.30
            )
            if self.structural_transformation_enabled:
                self.baumol_cost_pressure = np.clip(
                    0.92 * self.baumol_cost_pressure
                    + 0.08 * np.maximum(
                        industrial_productivity
                        / np.maximum(service_productivity, 1.0e-8) - 1.0,
                        0.0,
                    ),
                    0.0,
                    1.50,
                )
            else:
                self.baumol_cost_pressure.fill(0.0)
            capacity_target *= np.clip(
                service_productivity / (1.0 + 0.18 * self.baumol_cost_pressure),
                0.45,
                1.12,
            )
            self.service_capacity = np.clip(
                self.service_capacity + 0.14 * (capacity_target - self.service_capacity),
                0.08,
                1.20,
            )
            available_service = np.maximum(
                self.service_queue * 0.82 + demand_pressure, 0.0
            )
            served = np.minimum(available_service, self.service_capacity)
            self.service_queue = np.maximum(available_service - served, 0.0)
            self.service_utilization = np.clip(
                served / np.maximum(self.service_capacity, 1.0e-8), 0.0, 1.0
            )
            self.service_wait = np.clip(
                self.service_queue / np.maximum(self.service_capacity, 1.0e-8),
                0.0,
                8.0,
            )
            service_factor = np.clip(
                served / np.maximum(demand_pressure, 1.0e-8), 0.08, 1.10
            )
        else:
            self.service_capacity.fill(1.0)
            self.service_queue.fill(0.0)
            self.service_utilization.fill(0.0)
            self.service_wait.fill(0.0)
            service_factor = np.ones((self.paths, 4))

        self.sector_capacity = np.stack(
            (self.biological_yield, industrial_capacity, self.service_capacity), axis=2
        )
        planned_group = np.maximum(group_demand, 1.0e-8)
        raw_output = np.stack(
            (
                np.minimum(prior_output[:, :, 0], self.biological_yield),
                np.minimum(prior_output[:, :, 1], industrial_capacity),
                np.minimum(prior_output[:, :, 2], service_factor),
            ),
            axis=2,
        )
        self.sector_output = np.clip(raw_output, 0.02, 1.30)
        sector_realization = np.clip(
            self.sector_output / planned_group, 0.05, 1.18
        )
        self.node_realization = self._expand_actor_group(sector_realization)

        agriculture_inventory = self._aggregate_group(inventory)[:, :, 0]
        food_security = np.clip(
            0.62 * self.biological_yield
            + 0.28 * np.minimum(agriculture_inventory, 1.20)
            + 0.10 * actor_logistics,
            0.02,
            1.20,
        )
        self.months += 1
        self.minimum_water = min(self.minimum_water, float(np.min(self.water)))
        self.minimum_soil = min(self.minimum_soil, float(np.min(self.soil)))
        self.minimum_industrial_operability = min(
            self.minimum_industrial_operability,
            float(np.min(self.industrial_operability)),
        )
        self.maximum_service_utilization = max(
            self.maximum_service_utilization,
            float(np.max(self.service_utilization)),
        )
        self.maximum_service_wait = max(
            self.maximum_service_wait, float(np.max(self.service_wait))
        )
        lower_violation = max(0.0, -float(np.min(self.node_realization)))
        upper_violation = max(0.0, float(np.max(self.node_realization)) - 1.18)
        self.max_node_bound_violation = max(
            self.max_node_bound_violation, lower_violation, upper_violation
        )
        self.max_land_share_residual = max(
            self.max_land_share_residual,
            float(np.max(np.abs(self.agriculture_land_share.sum(axis=2) - 1.0))),
        )
        self.minimum_firm_mass = min(
            self.minimum_firm_mass, float(np.min(self.industrial_firm_mass))
        )
        self.minimum_industrial_quality = min(
            self.minimum_industrial_quality, float(np.min(self.industrial_quality))
        )
        self.maximum_baumol_cost_pressure = max(
            self.maximum_baumol_cost_pressure,
            float(np.max(self.baumol_cost_pressure)),
        )
        self.last_factors = SectoralFactors(
            actor_sector_output=self.sector_output.copy(),
            actor_sector_capacity=self.sector_capacity.copy(),
            actor_sector_labor=effective_labor.copy(),
            food_security=food_security,
            industrial_operability=self.industrial_operability.copy(),
            service_utilization=self.service_utilization.copy(),
            service_wait=self.service_wait.copy(),
            node_realization=self.node_realization.copy(),
            agriculture_land_share=self.agriculture_land_share.copy(),
            industrial_firm_mass=self.industrial_firm_mass.copy(),
            industrial_quality=self.industrial_quality.copy(),
            digital_adoption=self.digital_adoption.copy(),
            public_service_capacity=self.public_service_capacity.copy(),
        )
        return self.last_factors

    def expire_service_inventory(self, dqsfc) -> None:
        expired_nodes = np.where(
            self.service_node_mask[None, :], dqsfc.inventory, 0.0
        )
        expired_actor = expired_nodes @ self.actor_node_weights.T
        self.expired_service += expired_actor
        dqsfc.inventory[:, self.service_node_mask] = 0.0
        residual = float(np.max(np.abs(dqsfc.inventory[:, self.service_node_mask])))
        self.max_service_stock_residual = max(
            self.max_service_stock_residual, residual
        )
        self.maximum_expired_service = max(
            self.maximum_expired_service, float(np.max(expired_actor))
        )

    def realized_sector_output(self, actual_nodes: np.ndarray) -> np.ndarray:
        actual = np.clip(self._aggregate_group(actual_nodes), 0.0, 1.50)
        accounting = np.einsum(
            "pag,ag->pa", actual, self.actor_group_output_share
        )
        direct = actual_nodes @ self.actor_node_weights.T
        self.max_sector_accounting_residual = max(
            self.max_sector_accounting_residual,
            float(np.max(np.abs(accounting - direct))),
        )
        self.sector_output = actual
        return actual

    def diagnostics(self) -> dict:
        return {
            "clock": "monthly_same_path_after_equilibrium_before_realized_transactions",
            "months": self.months,
            "sector_groups": list(SECTOR_GROUPS),
            "sector_mapping": {
                "agriculture": list(AGRICULTURE_SECTORS),
                "industry": list(INDUSTRY_SECTORS),
                "services": list(SERVICE_SECTORS),
            },
            "enabled": {
                "agriculture": self.agriculture_enabled,
                "industry": self.industry_enabled,
                "services": self.services_enabled,
                "labor_reallocation": self.labor_reallocation_enabled,
                "structural_transformation": self.structural_transformation_enabled,
            },
            "max_labor_share_residual": self.max_labor_share_residual,
            "max_service_stock_residual": self.max_service_stock_residual,
            "max_node_bound_violation": self.max_node_bound_violation,
            "max_sector_accounting_residual": self.max_sector_accounting_residual,
            "minimum_water": self.minimum_water,
            "minimum_soil": self.minimum_soil,
            "minimum_industrial_operability": self.minimum_industrial_operability,
            "maximum_service_utilization": self.maximum_service_utilization,
            "maximum_service_wait": self.maximum_service_wait,
            "maximum_expired_service": self.maximum_expired_service,
            "max_land_share_residual": self.max_land_share_residual,
            "minimum_firm_mass": self.minimum_firm_mass,
            "minimum_industrial_quality": self.minimum_industrial_quality,
            "minimum_digital_adoption": float(np.min(self.digital_adoption)),
            "maximum_digital_adoption": float(np.max(self.digital_adoption)),
            "minimum_public_service_capacity": float(
                np.min(self.public_service_capacity)
            ),
            "maximum_baumol_cost_pressure": self.maximum_baumol_cost_pressure,
            "state_owners": {
                "weather_water_soil_biological_yield": "ThreeSectorProductionState",
                "industrial_equipment_operability": "ThreeSectorProductionState",
                "nonstorable_service_queue": "ThreeSectorProductionState",
                "sectoral_labor_shares": "ThreeSectorProductionState",
                "agricultural_land_allocation": "ThreeSectorProductionState",
                "industrial_firm_mass_quality_and_learning": "ThreeSectorProductionState",
                "digital_adoption_and_public_service_capacity": "ThreeSectorProductionState",
                "engel_baumol_kaldor_structural_dynamics": "ThreeSectorProductionState",
                "commodity_inventory_and_order_backlog": "MRRDCGEDQSFCState",
                "installed_capital_and_equilibrium_plan": "DynamicMultiRegionalCGE",
                "financial_assets_and_liabilities": "FourPartyFinancialSystem",
            },
            "identification": {
                "sector_mapping_and_weights": "OECD_ICIO_2022_observed_value_added",
                "inventory_and_transactions": "existing_IO_SAM_DQ_SFC",
                "agriculture_industry_service_adjustment_speeds": "bounded_scenario_priors",
                "claim": "conditional_mechanism_simulation_not_point_forecast",
            },
        }


class CoupledSectoralEconomicSystem(CoupledMRRDCGEDQSFCEconomicSystem):
    """V6.1 macro core with lagged sector feedback and current physical clearing."""

    def __init__(
        self,
        paths: int,
        seed: int,
        scenario: str,
        dqsfc_enabled: bool = True,
        econometric_draws: dict[str, np.ndarray] | None = None,
        *,
        agriculture_enabled: bool = True,
        industry_enabled: bool = True,
        services_enabled: bool = True,
        labor_reallocation_enabled: bool = True,
        structural_transformation_enabled: bool = True,
        sectoral_layer_enabled: bool = True,
    ):
        super().__init__(paths, seed, scenario, dqsfc_enabled, econometric_draws)
        self.sectoral_layer_enabled = sectoral_layer_enabled
        self.sectoral = ThreeSectorProductionState(
            self.cge,
            paths,
            seed + 610_061,
            agriculture_enabled=agriculture_enabled,
            industry_enabled=industry_enabled,
            services_enabled=services_enabled,
            labor_reallocation_enabled=labor_reallocation_enabled,
            structural_transformation_enabled=structural_transformation_enabled,
        )
        # Services are flows, not storable commodities. Any within-period
        # excess is recorded as expired service after DQ-SFC clearing.
        if self.sectoral_layer_enabled and self.sectoral.services_enabled:
            self.dqsfc.inventory[:, self.sectoral.service_node_mask] = 0.0
            self.dqsfc.inventory_target[self.sectoral.service_node_mask] = 0.0
            self.dqsfc.spoilage[self.sectoral.service_node_mask] = 1.0

    def update(self, **kwargs) -> MacroFactors:
        if not self.sectoral_layer_enabled:
            return CoupledMRRDCGEDQSFCEconomicSystem.update(self, **kwargs)
        prior_dq = self.dqsfc.last_factors
        prior_sector = self.sectoral.last_factors
        composite_supply = np.clip(
            0.24 * prior_sector.food_security
            + 0.46 * prior_sector.industrial_operability
            + 0.30 * (1.0 / (1.0 + prior_sector.service_wait)),
            0.12,
            1.10,
        )
        adjusted = dict(kwargs)
        adjusted["actor_logistics"] = np.clip(
            kwargs["actor_logistics"]
            * (0.72 + 0.28 * prior_dq.fulfilment)
            * (0.78 + 0.22 * composite_supply),
            0.02,
            1.0,
        )
        adjusted["actor_labor"] = np.clip(
            kwargs["actor_labor"] * prior_dq.employment
            * (0.88 + 0.12 / (1.0 + prior_sector.service_wait)),
            0.05,
            1.15,
        )
        adjusted["actor_payment"] = np.clip(
            kwargs["actor_payment"] * prior_dq.settlement
            * (0.90 + 0.10 / (1.0 + prior_sector.service_wait)),
            0.08,
            1.0,
        )
        adjusted["actor_fiscal"] = np.clip(
            kwargs["actor_fiscal"]
            * (0.92 + 0.08 * prior_sector.food_security),
            0.08,
            1.0,
        )

        equilibrium = CoupledMacroEconomicSystem.update(self, **adjusted)
        sector = self.sectoral.update(
            month=kwargs["month"],
            cge=self.cge,
            inventory=self.dqsfc.inventory,
            inventory_target=self.dqsfc.inventory_target,
            actor_damage=kwargs["actor_damage"],
            actor_logistics=adjusted["actor_logistics"],
            actor_labor=adjusted["actor_labor"],
            actor_financial_stress=kwargs["actor_financial_stress"],
            actor_payment=adjusted["actor_payment"],
            actor_fiscal=adjusted["actor_fiscal"],
            actor_social_action=kwargs["actor_social_action"],
            actor_panic=kwargs["actor_panic"],
            actor_mobilization=kwargs["actor_mobilization"],
            blockade=kwargs["blockade"],
        )
        self.cge.last_output = np.clip(
            self.cge.last_output * sector.node_realization, 0.0, 1.35
        )
        self.cge.last_capacity = np.minimum(
            self.cge.last_capacity,
            np.clip(self.cge.last_capacity * sector.node_realization, 0.0, 1.35),
        )
        dq = self.dqsfc.update(
            cge=self.cge,
            macro=equilibrium,
            actor_damage=kwargs["actor_damage"],
            actor_financial_stress=kwargs["actor_financial_stress"],
            actor_payment=adjusted["actor_payment"],
            actor_fiscal=adjusted["actor_fiscal"],
            actor_panic=kwargs["actor_panic"],
        )
        if self.sectoral.services_enabled:
            self.sectoral.expire_service_inventory(self.dqsfc)
        realized_sector = self.sectoral.realized_sector_output(
            self.dqsfc.actual_transactions
        )
        realized_gdp = np.clip(
            self.dqsfc.actual_transactions @ self.cge.actor_gdp_weights[:4].T,
            0.005,
            1.35,
        )
        realized_industry = np.clip(realized_sector[:, :, 1], 0.005, 1.35)
        service_pressure = np.clip(
            self.sectoral.service_wait / (1.0 + self.sectoral.service_wait),
            0.0,
            1.0,
        )
        food_pressure = np.clip(1.0 - sector.food_security, 0.0, 1.0)
        self.current_factors = replace(
            equilibrium,
            gdp_factor=realized_gdp,
            industry_factor=realized_industry,
            inflation=np.clip(
                0.54 * equilibrium.inflation
                + 0.30 * dq.contract_inflation
                + 0.10 * food_pressure
                + 0.04 * service_pressure
                + 0.02 * self.sectoral.baumol_cost_pressure,
                -0.10,
                1.75,
            ),
        )
        return self.current_factors

    def diagnostics(self) -> dict:
        result = CoupledMRRDCGEDQSFCEconomicSystem.diagnostics(self)
        result["three_sector_physical_economy"] = self.sectoral.diagnostics()
        result["three_sector_physical_economy"]["layer_enabled"] = (
            self.sectoral_layer_enabled
        )
        return result
