from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from dynamic_macro_equilibrium_v47 import CoupledMacroEconomicSystem, MacroFactors
from social_accounting_matrix_v53 import FourPartySocialAccountingMatrix


ACCOUNT_NAMES = ("households", "firms", "government", "banks", "external")


@dataclass
class DQSFCFactors:
    fulfilment: np.ndarray
    employment: np.ndarray
    settlement: np.ndarray
    transaction_factor: np.ndarray
    contract_inflation: np.ndarray
    equivalent_variation: np.ndarray


@dataclass(frozen=True)
class JorgensonParameters:
    factor_cost_shares: np.ndarray
    price_adjustment: np.ndarray
    inventory_target: np.ndarray


class JorgensonParameterEstimator:
    """Derive adding-up-consistent benchmark shares from the public IO base.

    This is deliberately not labelled an estimate of wartime elasticities.
    The IO shares are derived observations; adjustment speeds remain bounded
    scenario priors until an external time-series identification pass exists.
    """

    @staticmethod
    def fit(cge) -> JorgensonParameters:
        raw = np.stack(
            (cge.labor_share, cge.capital_share, cge.intermediate_input_share),
            axis=1,
        )
        shares = raw / np.maximum(raw.sum(axis=1, keepdims=True), 1e-12)
        intermediate = shares[:, 2]
        price_adjustment = np.clip(0.24 + 0.18 * (1.0 - intermediate), 0.24, 0.42)
        inventory_target = np.clip(0.16 + 0.34 * intermediate, 0.16, 0.48)
        return JorgensonParameters(shares, price_adjustment, inventory_target)


class MRRDCGEDQSFCState:
    """Pathwise monthly disequilibrium and non-financial SFC state.

    The CGE owns equilibrium production plans and factor prices.  This class
    uniquely owns realised transactions, inventories, backlogs, contract
    prices, search unemployment/vacancies, construction queues and the
    short-term settlement ledger.  Bank capital and sovereign debt remain
    owned by FourPartyFinancialSystem.
    """

    def __init__(self, cge, paths: int):
        self.paths = paths
        self.n_nodes = cge.n_nodes
        self.actor_of_node = cge.actor_of_node.copy()
        self.n_actors = cge.n_actors
        self.sector = cge.sector.copy()
        self.out = np.maximum(cge.out, 1e-9)
        self.actor_weights = cge.actor_gdp_weights[:4].copy()
        self.industry_weights = cge.actor_industry_weights[:4].copy()
        self.jorgenson = JorgensonParameterEstimator.fit(cge)
        sector_inventory_prior = np.array(
            [0.32, 0.28, 0.42, 0.36, 0.24, 0.30, 0.20, 0.18, 0.16, 0.26]
        )[self.sector]
        self.inventory_target = (
            0.55 * sector_inventory_prior + 0.45 * self.jorgenson.inventory_target
        )
        self.spoilage = np.array(
            [0.030, 0.022, 0.010, 0.008, 0.006, 0.006, 0.004, 0.012, 0.006, 0.015]
        )[self.sector]
        self.inventory = np.broadcast_to(
            self.inventory_target, (paths, self.n_nodes)
        ).copy()
        self.backlog = np.zeros((paths, self.n_nodes))
        self.actual_transactions = np.ones((paths, self.n_nodes))
        self.contract_price = np.ones((paths, self.n_nodes))
        self.unemployment = np.broadcast_to(
            np.array([0.052, 0.038, 0.042, 0.026]), (paths, 4)
        ).copy()
        self.vacancies = np.broadcast_to(
            np.array([0.030, 0.028, 0.036, 0.031]), (paths, 4)
        ).copy()
        self.old_capital = 0.76 * cge.capital.copy()
        self.new_capital = 0.24 * cge.capital.copy()
        self.construction_queue = np.zeros((paths, self.n_nodes, 3))
        self.settlement_position = np.zeros((paths, 4, len(ACCOUNT_NAMES)))
        self.sam = FourPartySocialAccountingMatrix(cge, paths)
        self.cumulative_ev = np.zeros((paths, 4))
        self.last_factors = DQSFCFactors(
            fulfilment=np.ones((paths, 4)),
            employment=1.0 - self.unemployment,
            settlement=np.ones((paths, 4)),
            transaction_factor=np.ones((paths, 4)),
            contract_inflation=np.zeros((paths, 4)),
            equivalent_variation=np.zeros((paths, 4)),
        )
        self.months = 0
        self.max_material_residual = 0.0
        self.max_sfc_residual = 0.0
        self.max_age_complementarity_residual = 0.0
        self.minimum_inventory = float(np.min(self.inventory))
        self.minimum_backlog = 0.0
        self.max_inner_residual = 0.0
        self.max_inner_iterations = 0
        self.max_capital_queue_residual = 0.0
        self.max_jorgenson_share_residual = 0.0
        self.max_hssw_identity_residual = 0.0

    def _aggregate(self, values: np.ndarray) -> np.ndarray:
        return values @ self.actor_weights.T

    def _expand_actor(self, values: np.ndarray, spill_weight: float = 0.20) -> np.ndarray:
        """Map four directly modelled actors to all CGE regions explicitly."""
        values = np.asarray(values, dtype=float)
        expanded = np.zeros((self.paths, self.n_actors))
        expanded[:, :4] = values
        expanded[:, 4:] = (
            spill_weight * np.mean(values, axis=1, keepdims=True)
        )
        return expanded

    def _transfer(
        self, ledger: np.ndarray, payer: int, recipient: int, amount: np.ndarray
    ) -> None:
        amount = np.maximum(amount, 0.0)
        ledger[:, :, payer] -= amount
        ledger[:, :, recipient] += amount

    def update(
        self,
        *,
        cge,
        macro: MacroFactors,
        actor_damage: np.ndarray,
        actor_financial_stress: np.ndarray,
        actor_payment: np.ndarray,
        actor_fiscal: np.ndarray,
        actor_panic: np.ndarray,
    ) -> DQSFCFactors:
        production = np.maximum(cge.last_output, 0.0)
        planned = np.maximum(cge.last_demand, 0.0)
        capacity = np.maximum(cge.last_capacity, 0.0)
        opening_inventory = self.inventory.copy()
        opening_backlog = self.backlog.copy()
        available = production + opening_inventory * (1.0 - self.spoilage[None, :])
        panic_all = self._expand_actor(actor_panic, 0.12)
        stress_all = self._expand_actor(actor_financial_stress, 0.18)
        payment_all = np.clip(self._expand_actor(actor_payment, 0.20), 0.10, 1.0)
        fiscal_all = np.clip(self._expand_actor(actor_fiscal, 0.25), 0.08, 1.0)
        cancellation = np.clip(
            0.015
            + 0.11 * panic_all[:, self.actor_of_node]
            + 0.08 * stress_all[:, self.actor_of_node],
            0.0,
            0.35,
        )
        effective_demand = planned + 0.12 * opening_backlog

        price = self.contract_price.copy()
        order_demand = effective_demand.copy()
        actual = np.minimum(order_demand, available)
        inner_residual = np.inf
        for iteration in range(48):
            relative_contract_price = np.maximum(
                price / np.maximum(cge.price, 0.25), 0.25
            )
            order_demand = np.clip(
                effective_demand
                * relative_contract_price ** (-cge.price_elasticity[None, :]),
                0.0,
                3.0,
            )
            new_actual = np.minimum(order_demand, available)
            unmet = np.maximum(order_demand - new_actual, 0.0)
            unsold = np.maximum(available - new_actual, 0.0)
            scale = np.maximum(order_demand + available, 1e-8)
            target = np.clip(
                0.72 * cge.price
                + 0.28 * price
                * np.exp(
                    self.jorgenson.price_adjustment[None, :] * unmet / scale
                    - 0.10 * unsold / scale
                ),
                0.25,
                8.0,
            )
            new_price = 0.58 * price + 0.42 * target
            inner_residual = float(max(
                np.max(np.abs(new_price - price)),
                np.max(np.abs(new_actual - actual)),
            ))
            price = new_price
            actual = new_actual
            if inner_residual < 1.0e-6:
                break
        self.max_inner_residual = max(self.max_inner_residual, inner_residual)
        self.max_inner_iterations = max(self.max_inner_iterations, iteration + 1)

        new_inventory = np.maximum(available - actual, 0.0)
        new_backlog = np.maximum(
            opening_backlog * (1.0 - cancellation) + order_demand - actual, 0.0
        )
        material_residual = np.max(np.abs(
            opening_inventory * (1.0 - self.spoilage[None, :])
            + production - actual - new_inventory
        ))
        self.max_material_residual = max(
            self.max_material_residual, float(material_residual)
        )
        self.inventory = new_inventory
        self.backlog = new_backlog
        self.actual_transactions = actual
        self.contract_price = price
        self.minimum_inventory = min(self.minimum_inventory, float(np.min(new_inventory)))
        self.minimum_backlog = min(self.minimum_backlog, float(np.min(new_backlog)))

        planned_actor = np.maximum(self._aggregate(order_demand), 1e-8)
        actual_actor = self._aggregate(actual)
        fulfilment = np.clip(actual_actor / planned_actor, 0.02, 1.05)
        transaction_factor = np.clip(actual_actor, 0.01, 1.35)

        # Search matching with actor-specific stocks.  The matching flow is
        # bounded by both unemployment and vacancies, so neither stock can go
        # negative under large shocks.
        output_gap = np.clip(1.0 - transaction_factor, -0.20, 0.95)
        vacancy_target = np.clip(
            0.018 + 0.10 * self._aggregate(new_backlog)
            - 0.055 * output_gap - 0.040 * actor_financial_stress,
            0.002,
            0.18,
        )
        self.vacancies += 0.18 * (vacancy_target - self.vacancies)
        matching = 0.46 * np.sqrt(
            np.maximum(self.unemployment, 0.0) * np.maximum(self.vacancies, 0.0)
        )
        matching = np.minimum(matching, np.minimum(self.unemployment, self.vacancies))
        separation = np.clip(
            0.004 + 0.028 * actor_damage + 0.020 * actor_financial_stress
            + 0.012 * actor_panic + 0.010 * output_gap,
            0.001,
            0.12,
        ) * (1.0 - self.unemployment)
        self.unemployment = np.clip(
            self.unemployment + separation - matching, 0.005, 0.65
        )
        self.vacancies = np.clip(self.vacancies - matching + 0.35 * separation, 0.001, 0.22)
        employment = 1.0 - self.unemployment

        # Putty/semi-putty capital: investment first enters a three-stage
        # construction queue.  Existing capital cannot instantly change type.
        finance_node = (
            (1.0 - stress_all) * payment_all * fiscal_all
        )[:, self.actor_of_node]
        fulfilment_all = np.clip(self._expand_actor(fulfilment, 0.30), 0.05, 1.0)
        equipment = fulfilment_all[:, self.actor_of_node]
        planned_investment = np.maximum(
            0.010 * planned * finance_node * equipment, 0.0
        )
        queue_open = self.construction_queue.copy()
        completion = queue_open[:, :, 2] / 6.0
        self.construction_queue[:, :, 2] += queue_open[:, :, 1] / 6.0 - completion
        self.construction_queue[:, :, 1] += queue_open[:, :, 0] / 6.0 - queue_open[:, :, 1] / 6.0
        self.construction_queue[:, :, 0] += planned_investment - queue_open[:, :, 0] / 6.0
        self.construction_queue = np.maximum(self.construction_queue, 0.0)
        queue_residual = np.max(np.abs(
            self.construction_queue.sum(axis=2)
            - (queue_open.sum(axis=2) + planned_investment - completion)
        ))
        self.max_capital_queue_residual = max(
            self.max_capital_queue_residual, float(queue_residual)
        )
        self.old_capital *= 1.0 - cge.depreciation[None, :]
        self.new_capital = (
            self.new_capital * (1.0 - 0.65 * cge.depreciation[None, :]) + completion
        )
        vintage_capital = np.clip(self.old_capital + self.new_capital, 0.04, 1.65)
        cge.capital = np.clip(0.70 * cge.capital + 0.30 * vintage_capital, 0.04, 1.55)

        # Five-account SFC settlement ledger.  Each transaction is posted once
        # as an equal debit and credit.  These are clearing positions; bank
        # capital, public debt and foreign reserves remain in the financial
        # module and are not duplicated here.
        value_sales = actual_actor * np.clip(self._aggregate(price), 0.25, 8.0)
        wage_bill = np.maximum(0.50 * value_sales * employment, 0.0)
        consumption = np.maximum(0.56 * value_sales * (1.0 - 0.35 * actor_panic), 0.0)
        taxes = np.maximum(0.15 * value_sales * actor_payment, 0.0)
        procurement = np.maximum(0.12 * value_sales * actor_fiscal, 0.0)
        transfers = np.maximum(0.035 * value_sales * actor_fiscal, 0.0)
        working_capital = np.maximum(
            0.10 * value_sales * actor_financial_stress
            + 0.06 * self._aggregate(new_backlog), 0.0
        )
        debt_service = np.minimum(working_capital, 0.04 * value_sales * actor_payment)
        exports = np.maximum(macro.trade_balance, 0.0)
        imports = np.maximum(-macro.trade_balance, 0.0)
        ledger = np.zeros((self.paths, 4, len(ACCOUNT_NAMES)))
        self._transfer(ledger, 1, 0, wage_bill)
        self._transfer(ledger, 0, 1, consumption)
        self._transfer(ledger, 1, 2, taxes)
        self._transfer(ledger, 2, 1, procurement)
        self._transfer(ledger, 2, 0, transfers)
        self._transfer(ledger, 3, 1, working_capital)
        self._transfer(ledger, 1, 3, debt_service)
        self._transfer(ledger, 4, 1, exports)
        self._transfer(ledger, 1, 4, imports)
        sfc_residual = float(np.max(np.abs(np.sum(ledger, axis=2))))
        self.max_sfc_residual = max(self.max_sfc_residual, sfc_residual)
        self.settlement_position += ledger
        self.sam.update(
            actual_nodes=actual,
            node_prices=price,
            employment=employment,
            consumption=consumption,
            taxes=taxes,
            procurement=procurement,
            transfers=transfers,
            investment=self._aggregate(planned_investment),
            trade_balance=macro.trade_balance,
        )
        settlement = np.clip(
            1.0 + 0.04 * np.tanh(self.settlement_position[:, :, 1])
            - 0.10 * actor_financial_stress,
            0.55,
            1.04,
        )

        # HSSW welfare decomposition uses realised consumption and the
        # contract-price index.  It is diagnostic and does not own behaviour.
        price_index = np.clip(self._aggregate(price), 0.25, 8.0)
        real_consumption = consumption / price_index
        baseline_consumption = 0.56 * np.maximum(macro.gdp_factor, 1e-8)
        ev = real_consumption - baseline_consumption
        expenditure_identity = price_index * real_consumption - consumption
        self.max_hssw_identity_residual = max(
            self.max_hssw_identity_residual,
            float(np.max(np.abs(expenditure_identity))),
        )
        self.cumulative_ev += ev

        # AGE activity complementarity inherited from the capacity-constrained
        # CGE: positive scarcity rent is complementary to unused capacity.
        slack = np.maximum(capacity - production, 0.0)
        age_residual = float(np.max(np.abs(cge.shadow_price * slack)))
        self.max_age_complementarity_residual = max(
            self.max_age_complementarity_residual, age_residual
        )

        # Jorgenson share restrictions.  Public IO value-added shares supply
        # the benchmark; no wartime observation is fabricated.
        normalized = self.jorgenson.factor_cost_shares
        self.max_jorgenson_share_residual = max(
            self.max_jorgenson_share_residual,
            float(np.max(np.abs(normalized.sum(axis=1) - 1.0))),
        )

        self.months += 1
        self.last_factors = DQSFCFactors(
            fulfilment=fulfilment,
            employment=employment,
            settlement=settlement,
            transaction_factor=transaction_factor,
            contract_inflation=np.clip(price_index - 1.0, -0.10, 2.0),
            equivalent_variation=ev,
        )
        return self.last_factors

    def diagnostics(self) -> dict:
        return {
            "clock": "monthly_same_path",
            "feedback_lag": "one explicit month into CGE-DSGE and finance",
            "months": self.months,
            "max_material_residual": self.max_material_residual,
            "max_sfc_residual": self.max_sfc_residual,
            "minimum_inventory": self.minimum_inventory,
            "minimum_backlog": self.minimum_backlog,
            "max_inner_residual": self.max_inner_residual,
            "max_inner_iterations": self.max_inner_iterations,
            "max_capital_queue_residual": self.max_capital_queue_residual,
            "max_age_complementarity_residual": self.max_age_complementarity_residual,
            "max_jorgenson_share_residual": self.max_jorgenson_share_residual,
            "max_hssw_identity_residual": self.max_hssw_identity_residual,
            "minimum_employment": float(np.min(1.0 - self.unemployment)),
            "maximum_backlog": float(np.max(self.backlog)),
            "minimum_settlement_position": float(np.min(self.settlement_position)),
            "maximum_settlement_position": float(np.max(self.settlement_position)),
            "account_names": list(ACCOUNT_NAMES),
            "social_accounting_matrix": self.sam.diagnostics(),
            "state_owners": {
                "inventory_backlog_contract_prices": "MRRDCGEDQSFCState",
                "unemployment_vacancies": "MRRDCGEDQSFCState",
                "capital_vintages_construction_queue": "MRRDCGEDQSFCState",
                "clearing_positions": "MRRDCGEDQSFCState",
                "sam_transactions_and_institutional_distribution": "FourPartySocialAccountingMatrix",
                "bank_capital_public_debt_reserves": "FourPartyFinancialSystem",
            },
            "parameter_identification": {
                "input_output_and_value_added": "observed_or_derived_public_IO",
                "price_adjustment_search_and_cancellation": "scenario_prior",
                "wartime_elasticities": "scenario_prior_with_sensitivity_required",
            },
        }


class CoupledMRRDCGEDQSFCEconomicSystem(CoupledMacroEconomicSystem):
    """V5.3 economic core with explicit lagged DQ-SFC feedback."""

    def __init__(self, paths: int, seed: int, scenario: str, dqsfc_enabled: bool = True):
        super().__init__(paths, seed, scenario)
        self.dqsfc_enabled = dqsfc_enabled
        self.dqsfc = MRRDCGEDQSFCState(self.cge, paths)

    def update(self, **kwargs) -> MacroFactors:
        if not self.dqsfc_enabled:
            return super().update(**kwargs)
        prior = self.dqsfc.last_factors
        adjusted = dict(kwargs)
        adjusted["actor_logistics"] = np.clip(
            kwargs["actor_logistics"] * (0.72 + 0.28 * prior.fulfilment), 0.02, 1.0
        )
        adjusted["actor_labor"] = np.clip(
            kwargs["actor_labor"] * prior.employment, 0.05, 1.15
        )
        adjusted["actor_payment"] = np.clip(
            kwargs["actor_payment"] * prior.settlement, 0.08, 1.0
        )
        equilibrium = super().update(**adjusted)
        dq = self.dqsfc.update(
            cge=self.cge,
            macro=equilibrium,
            actor_damage=kwargs["actor_damage"],
            actor_financial_stress=kwargs["actor_financial_stress"],
            actor_payment=kwargs["actor_payment"],
            actor_fiscal=kwargs["actor_fiscal"],
            actor_panic=kwargs["actor_panic"],
        )
        realised_gdp = np.clip(
            equilibrium.gdp_factor * dq.fulfilment, 0.005, 1.35
        )
        realised_industry = np.clip(
            equilibrium.industry_factor * dq.fulfilment, 0.005, 1.35
        )
        self.current_factors = replace(
            equilibrium,
            gdp_factor=realised_gdp,
            industry_factor=realised_industry,
            inflation=np.clip(
                0.62 * equilibrium.inflation + 0.38 * dq.contract_inflation,
                -0.10,
                1.75,
            ),
        )
        return self.current_factors

    def diagnostics(self) -> dict:
        result = super().diagnostics()
        result["mr_rd_cge_dq_sfc"] = {
            **self.dqsfc.diagnostics(), "enabled": self.dqsfc_enabled
        }
        return result
