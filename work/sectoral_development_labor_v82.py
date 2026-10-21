from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from sectoral_economy_v62 import ThreeSectorProductionStateV62
from tinbergen_algebraic_v70 import AsymmetricAdjustmentMacroV70


ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "data" / "sectoral_v82" / "actor_structural_priors.json"


class ThreeSectorDevelopmentLaborStateV82(ThreeSectorProductionStateV62):
    """Actor-specific agriculture, industry, development and labor deepening.

    Existing owners are preserved: DQ-SFC owns commodity stocks and contracts,
    CGE owns installed capital and market clearing, and V6.2 owns sector labor
    shares. This layer owns expectations, enterprise composition, vacancies,
    matching frictions, skills, fatigue, entitlements and policy credibility.
    """

    def __init__(self, *args, agriculture_deepening_enabled: bool = True,
                 industry_deepening_enabled: bool = True,
                 development_deepening_enabled: bool = True,
                 labor_deepening_enabled: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        cfg = json.loads(LEDGER.read_text(encoding="utf-8"))
        observed = cfg["observed"]
        prior = cfg["bounded_calibration_priors"]
        self.v82_config = cfg
        self.agriculture_deepening_enabled = agriculture_deepening_enabled
        self.industry_deepening_enabled = industry_deepening_enabled
        self.development_deepening_enabled = development_deepening_enabled
        self.labor_deepening_enabled = labor_deepening_enabled

        def tiled(values):
            return np.broadcast_to(np.asarray(values, dtype=float),
                                   (self.paths, 4)).copy()

        self.food_self_sufficiency = tiled(observed["food_self_sufficiency_calorie"])
        self.aging_share = tiled(observed["population_65_plus_share"])
        self.unemployment_anchor = tiled(observed["baseline_unemployment"])
        self.sme_anchor = tiled(observed["sme_employment_share"])
        self.concentration = tiled(prior["industrial_concentration"])
        self.state_enterprise_weight = tiled(prior["state_enterprise_weight"])
        self.labor_mobility = tiled(prior["labor_mobility"])
        self.finance_access = tiled(prior["market_finance_access"])
        self.ag_tfp_monthly = tiled(prior["agricultural_tfp_trend_annual"]) / 12.0

        self.expected_farm_return = np.ones((self.paths, 4))
        self.farm_return_variance = np.zeros((self.paths, 4))
        self.farm_input_intensity = np.ones((self.paths, 4))
        self.agricultural_technology = np.ones((self.paths, 4))
        self.entitlement_access = np.ones((self.paths, 4))
        self.policy_credibility = np.full((self.paths, 4), 0.82)
        self.report_bias = np.zeros((self.paths, 4))
        self.sustainable_surplus = np.zeros((self.paths, 4))
        self.johnston_mellor_contribution = np.full((self.paths, 4), 0.75)
        # Enterprise types: state/large incumbent, private specialist, SME.
        public = self.state_enterprise_weight
        sme = np.clip(self.sme_anchor, 0.22, 0.78)
        specialist = np.maximum(1.0 - public - sme, 0.05)
        enterprise = np.stack((public, specialist, sme), axis=2)
        self.enterprise_share = enterprise / enterprise.sum(axis=2, keepdims=True)
        self.enterprise_productivity = np.broadcast_to(
            np.array([0.94, 1.10, 0.88]), (self.paths, 4, 3)
        ).copy()
        self.vacancy_rate = np.full((self.paths, 4), 0.045)
        self.matching_wedge = np.ones((self.paths, 4))
        self.skill_stock = np.broadcast_to(
            np.array([0.96, 0.94, 1.04, 1.00]), (self.paths, 4)
        ).copy()
        self.fatigue = np.zeros((self.paths, 4))
        self.human_capital = np.ones((self.paths, 4))
        self.technology_capability = np.broadcast_to(
            np.array([0.88, 1.08, 1.12, 1.02]), (self.paths, 4)
        ).copy()
        self.state_capacity = np.broadcast_to(
            np.array([0.84, 0.82, 0.86, 0.88]), (self.paths, 4)
        ).copy()
        self.months_v82 = 0
        self.max_enterprise_share_residual = 0.0
        self.max_probability_violation = 0.0
        self.min_entitlement = 1.0
        self.max_fatigue = 0.0
        self.min_matching_wedge = 1.0
        self.min_sustainable_surplus = 0.0

    def _development_update(self, fiscal, payment, damage, panic, logistics):
        if not self.development_deepening_enabled:
            return np.ones((self.paths, 4))
        credibility_target = np.clip(
            0.30 + 0.30 * fiscal + 0.24 * payment + 0.16 * logistics
            - 0.18 * panic - 0.12 * damage, 0.12, 1.0,
        )
        self.policy_credibility += 0.018 * (credibility_target - self.policy_credibility)
        self.report_bias = np.clip(
            0.94 * self.report_bias + 0.035 * panic + 0.025 * damage
            - 0.030 * self.policy_credibility, 0.0, 0.45,
        )
        surplus_gain = np.clip(self.sustainable_surplus, -0.50, 0.50)
        development_contribution = np.clip(
            self.johnston_mellor_contribution - 0.75, -0.45, 0.35
        )
        self.state_capacity = np.clip(
            self.state_capacity + 0.006 * (fiscal * payment - self.state_capacity)
            - 0.003 * damage - 0.002 * self.report_bias, 0.18, 1.08,
        )
        self.state_capacity = np.clip(
            self.state_capacity
            + 0.0018 * surplus_gain
            + 0.0012 * development_contribution,
            0.18, 1.08,
        )
        self.human_capital = np.clip(
            self.human_capital + 0.0015 * self.state_capacity
            - 0.0012 * self.fatigue - 0.0010 * damage
            - 0.0008 * self.aging_share
            + 0.0015 * np.minimum(surplus_gain, 0.0), 0.60, 1.18,
        )
        frontier = np.array([1.10, 1.13, 1.16, 1.12])[None, :]
        catchup = np.maximum(frontier - self.technology_capability, 0.0)
        self.technology_capability = np.clip(
            self.technology_capability
            + 0.0025 * self.state_capacity * self.human_capital * catchup
            - 0.0018 * damage, 0.55, 1.18,
        )
        return np.clip(
            self.human_capital ** 0.30 * self.technology_capability ** 0.28
            * self.state_capacity ** 0.12, 0.72, 1.12,
        )

    def _labor_update_v82(self, factors, damage, logistics, financial, payment,
                          panic, mobilization):
        if not self.labor_deepening_enabled:
            return np.ones((self.paths, 4))
        demand = np.clip(np.mean(factors.actor_sector_output, axis=2), 0.05, 1.35)
        vacancy_target = np.clip(
            0.018 + 0.070 * np.maximum(demand - 0.72, 0.0)
            + 0.018 * payment - 0.035 * financial, 0.004, 0.16,
        )
        self.vacancy_rate += 0.16 * (vacancy_target - self.vacancy_rate)
        unemployment = np.clip(
            self.unemployment_anchor + 0.14 * financial + 0.11 * damage
            + 0.055 * panic - 0.035 * logistics, 0.012, 0.42,
        )
        tightness = self.vacancy_rate / np.maximum(unemployment, 1.0e-4)
        matching_efficiency = np.clip(
            (0.34 + 0.38 * self.labor_mobility) * logistics
            * (1.0 - 0.45 * panic), 0.06, 0.82,
        )
        job_finding = np.clip(matching_efficiency * tightness ** 0.46, 0.0, 0.82)
        separation = np.clip(
            0.012 + 0.050 * damage + 0.040 * financial
            + 0.022 * panic + 0.018 * mobilization, 0.006, 0.24,
        )
        self.matching_wedge = np.clip(
            1.0 + 0.10 * job_finding - 0.32 * separation, 0.78, 1.08,
        )
        overtime = np.clip(mobilization + np.maximum(demand - 0.92, 0.0), 0.0, 1.5)
        self.fatigue = np.clip(
            0.90 * self.fatigue + 0.055 * overtime
            - 0.040 * logistics * (1.0 - panic), 0.0, 0.70,
        )
        training = np.clip(0.45 * payment + 0.35 * logistics - 0.25 * damage, 0.0, 1.0)
        self.skill_stock = np.clip(
            self.skill_stock + 0.004 * training
            - 0.003 * self.fatigue - 0.0015 * self.aging_share, 0.62, 1.16,
        )
        return np.clip(
            self.matching_wedge * self.skill_stock * (1.0 - 0.28 * self.fatigue),
            0.58, 1.12,
        )

    def _agriculture_update_v82(self, factors, logistics, payment, fiscal,
                                financial, panic, blockade):
        if not self.agriculture_deepening_enabled:
            return np.ones((self.paths, 4)), factors.food_security
        price_signal = np.clip(
            1.0 + 0.22 * panic + 0.18 * blockade[:, None]
            + 0.12 * financial - 0.10 * logistics, 0.65, 1.55,
        )
        realized_return = np.clip(
            price_signal * payment * self.policy_credibility, 0.10, 1.45,
        )
        return_innovation = realized_return - self.expected_farm_return
        self.farm_return_variance = np.clip(
            0.90 * self.farm_return_variance + 0.10 * return_innovation ** 2,
            0.0, 0.40,
        )
        self.expected_farm_return = (
            0.88 * self.expected_farm_return + 0.12 * realized_return
        )
        risk_penalty = (
            0.08 * financial + 0.10 * (1.0 - self.policy_credibility)
            + 0.12 * np.sqrt(self.farm_return_variance)
        )
        input_target = np.clip(
            self.expected_farm_return - risk_penalty, 0.35, 1.22,
        )
        self.farm_input_intensity += 0.07 * (input_target - self.farm_input_intensity)
        technology_frontier = np.array([1.20, 1.17, 1.24, 1.18])[None, :]
        self.agricultural_technology = np.clip(
            self.agricultural_technology
            + self.ag_tfp_monthly * self.policy_credibility
            * (technology_frontier - self.agricultural_technology)
            - 0.0012 * np.maximum(0.45 - logistics, 0.0),
            0.72, technology_frontier,
        )
        import_access = np.clip(
            logistics * (1.0 - 0.72 * blockade[:, None]), 0.02, 1.0,
        )
        domestic = self.food_self_sufficiency * factors.food_security
        imported = (1.0 - self.food_self_sufficiency) * import_access
        entitlement_target = np.clip(
            0.38 * payment + 0.24 * fiscal + 0.22 * logistics
            + 0.16 * (domestic + imported) - 0.18 * panic, 0.05, 1.08,
        )
        self.entitlement_access += 0.10 * (entitlement_target - self.entitlement_access)
        subsistence = np.clip(0.52 + 0.13 * self.aging_share, 0.52, 0.62)
        seed_feed_loss = 0.18 + 0.05 * (1.0 - logistics)
        self.sustainable_surplus = (
            factors.actor_sector_output[:, :, 0] - subsistence - seed_feed_loss
        )
        food_security = np.clip(
            (domestic + imported) * self.entitlement_access, 0.02, 1.20,
        )
        labor_release = np.clip(
            1.0 - self.labor_share[:, :, 0]
            / np.maximum(self.baseline_labor_share[None, :, 0], 1.0e-6),
            -0.60, 0.60,
        )
        self.johnston_mellor_contribution = np.clip(
            0.30 * food_security
            + 0.14 * (0.70 + 0.30 * labor_release)
            + 0.18 * fiscal * np.clip(1.0 + self.sustainable_surplus, 0.0, 1.4)
            + 0.20 * payment * self.entitlement_access
            + 0.18 * logistics * (1.0 - blockade[:, None]),
            0.05, 1.20,
        )
        production_factor = np.clip(
            self.farm_input_intensity * self.agricultural_technology
            * (0.93 + 0.07 * self.policy_credibility), 0.72, 1.14,
        )
        return production_factor, food_security

    def _industry_update_v82(self, factors, logistics, payment, fiscal,
                             financial, damage, blockade):
        if not self.industry_deepening_enabled:
            return np.ones((self.paths, 4))
        resilience = np.stack((
            0.72 + 0.22 * fiscal - 0.10 * financial,
            0.68 + 0.25 * self.finance_access * payment - 0.18 * damage,
            0.62 + 0.22 * logistics - 0.25 * financial - 0.12 * damage,
        ), axis=2)
        payoff = resilience * self.enterprise_productivity
        average = np.sum(self.enterprise_share * payoff, axis=2, keepdims=True)
        self.enterprise_share *= np.exp(0.018 * (payoff - average))
        self.enterprise_share = np.maximum(self.enterprise_share, 1.0e-8)
        self.enterprise_share /= self.enterprise_share.sum(axis=2, keepdims=True)
        learning = np.clip(
            0.002 * factors.actor_sector_output[:, :, 1, None]
            * np.array([0.7, 1.2, 0.9])[None, None, :]
            - 0.0015 * damage[:, :, None], -0.004, 0.004,
        )
        self.enterprise_productivity = np.clip(
            self.enterprise_productivity + learning, 0.62, 1.25,
        )
        portfolio_productivity = np.sum(
            self.enterprise_share * self.enterprise_productivity, axis=2
        )
        concentration_loss = self.concentration * np.clip(
            0.55 * damage + 0.35 * blockade[:, None]
            + 0.25 * (1.0 - logistics), 0.0, 1.0,
        )
        return np.clip(
            portfolio_productivity * (1.0 - 0.32 * concentration_loss),
            0.56, 1.12,
        )

    def update(self, **kwargs):
        factors = super().update(**kwargs)
        development = self._development_update(
            kwargs["actor_fiscal"], kwargs["actor_payment"],
            kwargs["actor_damage"], kwargs["actor_panic"],
            kwargs["actor_logistics"],
        )
        labor = self._labor_update_v82(
            factors, kwargs["actor_damage"], kwargs["actor_logistics"],
            kwargs["actor_financial_stress"], kwargs["actor_payment"],
            kwargs["actor_panic"], kwargs["actor_mobilization"],
        )
        agriculture, food_security = self._agriculture_update_v82(
            factors, kwargs["actor_logistics"], kwargs["actor_payment"],
            kwargs["actor_fiscal"], kwargs["actor_financial_stress"],
            kwargs["actor_panic"], kwargs["blockade"],
        )
        industry = self._industry_update_v82(
            factors, kwargs["actor_logistics"], kwargs["actor_payment"],
            kwargs["actor_fiscal"], kwargs["actor_financial_stress"],
            kwargs["actor_damage"], kwargs["blockade"],
        )
        multiplier = np.stack((
            agriculture * labor * development,
            industry * labor * development,
            labor * development,
        ), axis=2)
        multiplier = np.clip(multiplier, 0.48, 1.16)
        self.sector_output = np.clip(factors.actor_sector_output * multiplier, 0.01, 1.35)
        self.sector_capacity = np.clip(factors.actor_sector_capacity * multiplier, 0.01, 1.35)
        node_multiplier = self._expand_actor_group(multiplier)
        self.node_realization = np.clip(factors.node_realization * node_multiplier, 0.01, 1.18)
        self.months_v82 += 1
        self.max_enterprise_share_residual = max(
            self.max_enterprise_share_residual,
            float(np.max(np.abs(self.enterprise_share.sum(axis=2) - 1.0))),
        )
        self.max_probability_violation = max(
            self.max_probability_violation,
            max(0.0, -float(np.min(self.enterprise_share)),
                float(np.max(self.enterprise_share)) - 1.0),
        )
        self.min_entitlement = min(self.min_entitlement, float(np.min(self.entitlement_access)))
        self.max_fatigue = max(self.max_fatigue, float(np.max(self.fatigue)))
        self.min_matching_wedge = min(self.min_matching_wedge, float(np.min(self.matching_wedge)))
        self.min_sustainable_surplus = min(
            self.min_sustainable_surplus, float(np.min(self.sustainable_surplus))
        )
        self.last_factors = replace(
            factors,
            actor_sector_output=self.sector_output.copy(),
            actor_sector_capacity=self.sector_capacity.copy(),
            actor_sector_labor=np.clip(
                factors.actor_sector_labor * labor[:, :, None], 0.03, 1.45
            ),
            food_security=food_security.copy(),
            node_realization=self.node_realization.copy(),
            industrial_quality=np.clip(
                factors.industrial_quality * industry, 0.35, 1.25
            ),
        )
        return self.last_factors

    def diagnostics(self) -> dict:
        result = super().diagnostics()
        result["v82_actor_specific_economic_deepening"] = {
            "months": self.months_v82,
            "agriculture_enabled": self.agriculture_deepening_enabled,
            "industry_enabled": self.industry_deepening_enabled,
            "development_enabled": self.development_deepening_enabled,
            "labor_enabled": self.labor_deepening_enabled,
            "data_ledger": str(LEDGER),
            "actor_order": self.v82_config["actor_order"],
            "maximum_enterprise_share_residual": self.max_enterprise_share_residual,
            "maximum_probability_bound_violation": self.max_probability_violation,
            "minimum_entitlement_access": self.min_entitlement,
            "maximum_fatigue": self.max_fatigue,
            "minimum_matching_wedge": self.min_matching_wedge,
            "minimum_sustainable_surplus": self.min_sustainable_surplus,
            "mean_farm_return_variance_by_actor": np.mean(
                self.farm_return_variance, axis=0
            ).tolist(),
            "mean_johnston_mellor_contribution_by_actor": np.mean(
                self.johnston_mellor_contribution, axis=0
            ).tolist(),
            "mean_policy_credibility_by_actor": np.mean(
                self.policy_credibility, axis=0
            ).tolist(),
            "mean_enterprise_type_share_by_actor": np.mean(
                self.enterprise_share, axis=0
            ).tolist(),
            "terminal_probability_adjustment": False,
        }
        result["state_owners"].update({
            "farm_expectations_and_input_intensity": "ThreeSectorDevelopmentLaborStateV82",
            "food_entitlement_access": "ThreeSectorDevelopmentLaborStateV82",
            "enterprise_type_composition": "ThreeSectorDevelopmentLaborStateV82",
            "vacancy_matching_skill_and_fatigue": "ThreeSectorDevelopmentLaborStateV82",
            "human_capital_technology_and_policy_credibility": "ThreeSectorDevelopmentLaborStateV82",
            "aggregate_employment": "MRRDCGEDQSFCState",
            "commodity_inventory": "MRRDCGEDQSFCState",
            "installed_capital": "DynamicMultiRegionalCGE",
        })
        return result


class ActorSpecificEconomicSystemV82(AsymmetricAdjustmentMacroV70):
    def __init__(self, *args, agriculture_deepening_enabled: bool = True,
                 industry_deepening_enabled: bool = True,
                 development_deepening_enabled: bool = True,
                 labor_deepening_enabled: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.sectoral = ThreeSectorDevelopmentLaborStateV82(
            self.cge, self.paths, kwargs.get("seed", args[1] if len(args) > 1 else 820_082) + 820_082,
            agriculture_enabled=kwargs.get("agriculture_enabled", True),
            industry_enabled=kwargs.get("industry_enabled", True),
            services_enabled=kwargs.get("services_enabled", True),
            labor_reallocation_enabled=kwargs.get("labor_reallocation_enabled", True),
            structural_transformation_enabled=kwargs.get("structural_transformation_enabled", True),
            agriculture_deepening_enabled=agriculture_deepening_enabled,
            industry_deepening_enabled=industry_deepening_enabled,
            development_deepening_enabled=development_deepening_enabled,
            labor_deepening_enabled=labor_deepening_enabled,
        )
        if self.sectoral_layer_enabled and self.sectoral.services_enabled:
            self.dqsfc.inventory[:, self.sectoral.service_node_mask] = 0.0
            self.dqsfc.inventory_target[self.sectoral.service_node_mask] = 0.0
            self.dqsfc.spoilage[self.sectoral.service_node_mask] = 1.0
