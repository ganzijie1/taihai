from __future__ import annotations

from dataclasses import dataclass
import os
from time import perf_counter

import numpy as np

import simulate_dual_circulation_v44 as v44
import simulate_trade_war_v43 as v43
from accelerated_solver_infrastructure import JAX_AVAILABLE, JaxResidentRuntime
from jax_equilibrium_solvers import solve_cge_newton_krylov, solve_dsge_jvp


PRIMARY_ACTORS = ("CHN", "TWN", "USA", "JPN")


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -35.0, 35.0)))


@dataclass
class CGEResult:
    gdp_factor: np.ndarray
    industry_factor: np.ndarray
    chip_gap: np.ndarray
    price_inflation: np.ndarray
    trade_loss: np.ndarray
    trade_balance: np.ndarray
    government_balance: np.ndarray
    capacity_shadow_price: np.ndarray
    goods_market_residual: float
    factor_market_residual: float
    external_account_residual: float
    solver_iterations: int
    converged_path_fraction: float
    converged: bool


@dataclass
class DSGEResult:
    final_demand_factor: np.ndarray
    investment_factor: np.ndarray
    inflation: np.ndarray
    expected_inflation: np.ndarray
    policy_rate: np.ndarray
    exchange_rate: np.ndarray
    risk_premium: np.ndarray
    output_gap: np.ndarray
    equilibrium_residual: float
    solver_iterations: int
    converged_path_fraction: float
    converged: bool


@dataclass
class MacroFactors:
    gdp_factor: np.ndarray
    industry_factor: np.ndarray
    chip_gap: np.ndarray
    inflation: np.ndarray
    trade_loss: np.ndarray
    exchange_rate: np.ndarray
    policy_rate: np.ndarray
    capacity_shadow_price: np.ndarray
    trade_balance: np.ndarray
    government_balance: np.ndarray
    external_account_residual: float
    cge_goods_residual: float
    cge_factor_residual: float
    dsge_residual: float
    outer_residual: float
    outer_iterations: int
    outer_converged_path_fraction: float
    converged: bool


class DynamicMultiRegionalCGE:
    """Dynamic 14-region, 10-sector (140-node) capacity-constrained CGE.

    Quantities, delivered prices, labor, capital, intermediate inputs, Armington
    substitution, final demand, government demand, investment, and external
    accounts are solved jointly by damped complementarity iterations.
    """

    def __init__(self, paths: int, seed: int, scenario: str):
        self.paths = paths
        self.rng = np.random.default_rng(seed)
        self.data = v44.build_nested_accounts()
        self.network = v44.build_network(self.data)
        self.z = np.maximum(self.data["Z"], 0.0)
        self.fd = np.maximum(self.data["FD"], 0.0)
        self.out = np.maximum(self.data["OUT"], 1e-9)
        self.va = np.maximum(self.data["VA"], 0.0)
        self.region = self.data["region_of_node"].astype(int)
        self.sector = self.data["sector_of_node"].astype(int)
        self.n_nodes = len(self.out)
        self.n_regions = len(v44.REGIONS)
        self.n_actors = len(v44.ACTORS)
        self.actor_of_region = np.array(
            [0] * len(v44.CHINA_ZONES) + list(range(1, len(v44.ACTORS))), dtype=int
        )
        self.actor_of_node = self.actor_of_region[self.region]
        self.scenario = scenario
        self.scenario_parameters = v43.scenario_parameters(
            scenario, self.rng, paths
        )

        sales = np.maximum(self.z.sum(axis=1) + self.fd.sum(axis=1), 1e-9)
        self.intermediate_sales_weights = self.z / sales[:, None]
        self.final_sales_weights = self.fd / sales[:, None]
        self.intermediate_input_share = np.clip(self.z.sum(axis=0) / self.out, 0.0, 0.92)
        va_share = np.clip(self.va / self.out, 0.05, 0.95)
        self.labor_share = 0.58 * va_share
        self.capital_share = 0.42 * va_share
        self.price_elasticity = np.array(
            [0.70, 0.85, 0.62, 0.78, 1.15, 1.05, 0.72, 0.82, 0.90, 0.68]
        )[self.sector]
        self.armington_elasticity = np.array(
            [1.30, 1.45, 0.85, 1.15, 2.10, 1.85, 0.95, 1.35, 1.55, 0.90]
        )[self.sector]
        self.depreciation = np.array(
            [0.045, 0.055, 0.060, 0.065, 0.105, 0.085, 0.055, 0.060, 0.070, 0.045]
        )[self.sector] / 12.0
        self.military_mask = np.isin(
            self.sector,
            [v44.SECTORS.index("electronics"), v44.SECTORS.index("materials"), v44.SECTORS.index("machinery_transport")],
        )
        self.electronics_mask = self.sector == v44.SECTORS.index("electronics")

        self.output_ratio = np.ones((paths, self.n_nodes))
        self.price = np.ones((paths, self.n_nodes))
        self.capital = np.ones((paths, self.n_nodes))
        self.labor = np.ones((paths, self.n_nodes))
        self.productivity = np.ones((paths, self.n_nodes))
        self.inventory_months = self.rng.uniform(2.0, 7.0, paths)
        self.wage = np.ones((paths, self.n_actors))
        self.rental = np.ones((paths, self.n_actors))
        self.shadow_price = np.zeros((paths, self.n_nodes))
        # Expose the accepted node-level equilibrium plan to later model
        # layers.  These are plans, not realised transactions; V5.3's
        # disequilibrium layer owns inventories, backlogs and settlement.
        self.last_demand = np.ones((paths, self.n_nodes))
        self.last_capacity = np.ones((paths, self.n_nodes))
        self.last_output = np.ones((paths, self.n_nodes))
        self.last_goods_residual = np.inf
        self.last_factor_residual = np.inf
        self.max_accounting_residual = 0.0
        self.max_external_account_residual = 0.0

        self.actor_gdp_weights = np.zeros((self.n_actors, self.n_nodes))
        self.actor_industry_weights = np.zeros_like(self.actor_gdp_weights)
        self.actor_mean_weights = np.zeros_like(self.actor_gdp_weights)
        self.actor_electronics_weights = np.zeros_like(self.actor_gdp_weights)
        self.actor_output_weights = np.zeros_like(self.actor_gdp_weights)
        for actor in range(self.n_actors):
            mask = self.actor_of_node == actor
            weights = self.va[mask]
            self.actor_gdp_weights[actor, mask] = weights / max(weights.sum(), 1e-12)
            self.actor_mean_weights[actor, mask] = 1.0 / max(np.count_nonzero(mask), 1)
            self.actor_output_weights[actor, mask] = self.out[mask]
            imask = mask & self.military_mask
            iweights = self.va[imask]
            self.actor_industry_weights[actor, imask] = iweights / max(iweights.sum(), 1e-12)
            emask = mask & self.electronics_mask
            self.actor_electronics_weights[actor, emask] = 1.0 / max(
                np.count_nonzero(emask), 1
            )
        self.actor_scale = np.maximum(self.actor_output_weights.sum(axis=1), 1e-9)
        self.last_solver_iterations = 0
        self.max_solver_iterations = 0
        requested_solver = os.environ.get("TAIWAN_EQUILIBRIUM_SOLVER", "auto").lower()
        self.jax_runtime = JaxResidentRuntime() if JAX_AVAILABLE else None
        gpu_jax = self.jax_runtime is not None and self.jax_runtime.platform == "gpu"
        self.solver_backend = (
            "newton_krylov" if requested_solver == "newton_krylov"
            or (requested_solver == "auto" and gpu_jax) else "picard"
        )
        self.last_newton_diagnostics = None

    def snapshot_state(self) -> dict[str, np.ndarray | float]:
        """Capture the month-opening state for non-committing equilibrium trials."""
        return {
            "output_ratio": self.output_ratio.copy(),
            "price": self.price.copy(),
            "capital": self.capital.copy(),
            "labor": self.labor.copy(),
            "productivity": self.productivity.copy(),
            "inventory_months": self.inventory_months.copy(),
            "wage": self.wage.copy(),
            "rental": self.rental.copy(),
            "shadow_price": self.shadow_price.copy(),
            "last_demand": self.last_demand.copy(),
            "last_capacity": self.last_capacity.copy(),
            "last_output": self.last_output.copy(),
            "last_goods_residual": self.last_goods_residual,
            "last_factor_residual": self.last_factor_residual,
            "max_accounting_residual": self.max_accounting_residual,
            "max_external_account_residual": self.max_external_account_residual,
        }

    def restore_state(self, state: dict[str, np.ndarray | float]) -> None:
        for name in (
            "output_ratio", "price", "capital", "labor", "productivity",
            "inventory_months", "wage", "rental", "shadow_price",
            "last_demand", "last_capacity", "last_output",
        ):
            setattr(self, name, state[name].copy())
        self.last_goods_residual = float(state["last_goods_residual"])
        self.last_factor_residual = float(state["last_factor_residual"])
        self.max_accounting_residual = float(state["max_accounting_residual"])
        self.max_external_account_residual = float(state["max_external_account_residual"])

    def _scenario_shock(self, month: int, blockade_feedback: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        p = self.scenario_parameters
        elapsed = month + 1
        recovery = 1.0 - np.exp(-elapsed / p["recovery"])
        block = p["block_lr"] + (p["block0"] - p["block_lr"]) * (1.0 - recovery)
        block = np.clip(block + 0.30 * blockade_feedback, 0.0, 0.95)
        fab_available = p["fab0"] + (p["fab_lr"] - p["fab0"]) * recovery
        certification = 1.0 - np.exp(-elapsed / 30.0)
        china_substitution = p["china_sub"] * certification
        global_substitution = p["global_sub"] * (1.0 - np.exp(-elapsed / 54.0))
        rerouting = np.clip(global_substitution + 0.28 * certification, 0.0, 0.72)
        gravity_gap = block * (1.0 - rerouting)
        chip_raw_gap = np.clip(
            0.81 * (1.0 - fab_available) * (1.0 - global_substitution), 0.0, 0.98
        )
        self.inventory_months = np.maximum(self.inventory_months - chip_raw_gap, 0.0)
        chip_coeff = chip_raw_gap * (1.0 - np.minimum(self.inventory_months / 3.0, 1.0))
        sanction = p["sanction"] * (1.0 - 0.35 * certification)
        shipping = p["shipping"] * (1.0 - 0.25 * certification)
        bases = self.network["bases"]
        shock = (
            gravity_gap[:, None] * bases["twn_trade"][None, :]
            + chip_coeff[:, None] * bases["chip"][None, :]
            + sanction[:, None] * bases["chn_sanction"][None, :]
            + shipping[:, None] * bases["shipping"][None, :]
        )
        china_critical = (self.actor_of_node == 0) & self.military_mask
        shock[:, china_critical] *= 1.0 - 0.72 * china_substitution[:, None]
        trade_loss = np.clip(
            self.network["twn_trade_share"] * gravity_gap
            + self.network["chn_trade_share"] * sanction + 0.35 * shipping,
            0.0, 0.90,
        )
        return np.clip(shock, 0.0, 0.96), trade_loss

    def solve(
        self, *, month: int, final_demand_factor: np.ndarray,
        investment_factor: np.ndarray, actor_damage: np.ndarray,
        actor_logistics: np.ndarray, actor_labor: np.ndarray,
        actor_financial_stress: np.ndarray, actor_payment: np.ndarray,
        actor_fiscal: np.ndarray, actor_social_action: np.ndarray,
        actor_panic: np.ndarray, blockade_feedback: np.ndarray,
        max_iterations: int = 480, tolerance: float = 2.0e-3,
    ) -> CGEResult:
        # Expand four directly modeled actors; other regions receive global spillovers.
        def expand(values, external_weight=0.22):
            full = np.zeros((self.paths, self.n_actors))
            full[:, :4] = values
            spill = external_weight * np.mean(values, axis=1)
            full[:, 4:] = spill[:, None]
            return full

        def expand_level(values, external_weight=0.22):
            full = np.ones((self.paths, self.n_actors))
            full[:, :4] = values
            deviation = external_weight * np.mean(values - 1.0, axis=1)
            full[:, 4:] = 1.0 + deviation[:, None]
            return full

        damage = expand(actor_damage, 0.18)
        logistics = np.clip(expand_level(actor_logistics, 0.30), 0.05, 1.0)
        labor_supply = np.clip(expand_level(actor_labor, 0.25), 0.10, 1.15)
        finance = expand(actor_financial_stress, 0.18)
        payment = np.clip(expand_level(actor_payment, 0.20), 0.10, 1.0)
        fiscal = np.clip(expand_level(actor_fiscal, 0.25), 0.08, 1.0)
        social = expand(actor_social_action, 0.12)
        panic = expand(actor_panic, 0.12)
        fd_factor = np.clip(final_demand_factor, 0.20, 1.25)
        inv_factor = np.clip(investment_factor, 0.10, 1.40)
        exogenous_shock, trade_loss = self._scenario_shock(month, blockade_feedback)

        actor_damage_node = damage[:, self.actor_of_node]
        logistics_node = logistics[:, self.actor_of_node]
        labor_node = labor_supply[:, self.actor_of_node]
        finance_node = finance[:, self.actor_of_node]
        payment_node = payment[:, self.actor_of_node]
        social_node = social[:, self.actor_of_node]
        panic_node = panic[:, self.actor_of_node]
        productivity = np.clip(
            self.productivity * (1.0 - 0.30 * actor_damage_node)
            * (1.0 - 0.18 * finance_node) * payment_node
            * (1.0 - 0.12 * social_node - 0.08 * panic_node),
            0.08, 1.15,
        )
        capacity = np.clip(
            productivity
            * self.capital ** self.capital_share[None, :]
            * np.minimum(self.labor, labor_node) ** self.labor_share[None, :]
            * logistics_node * (1.0 - exogenous_shock),
            0.01, 1.25,
        )

        output = np.minimum(self.output_ratio, capacity)
        price = self.price.copy()
        wage = self.wage.copy()
        rental = self.rental.copy()
        goods_residual = np.inf
        factor_residual = np.inf
        government_multiplier = np.clip(
            1.0 + 0.55 * (1.0 - fiscal) + 0.35 * damage,
            0.45, 1.85,
        )
        region_fd = np.clip(
            0.82 * fd_factor[:, self.actor_of_region]
            + 0.18 * government_multiplier[:, self.actor_of_region], 0.20, 1.55
        )
        active = np.ones(self.paths, dtype=bool)
        path_iterations = np.zeros(self.paths, dtype=np.int32)
        path_relaxation = np.ones(self.paths, dtype=float)
        previous_path_residual = np.full(self.paths, np.inf)
        iterations_used = 0
        if self.solver_backend == "newton_krylov":
            solution = solve_cge_newton_krylov(
                self.jax_runtime,
                initial={"output": output, "price": price, "wage": wage, "rental": rental},
                coefficients={
                    "intermediate_weights": self.intermediate_sales_weights,
                    "final_weights": self.final_sales_weights,
                    "actor_gdp_weights": self.actor_gdp_weights,
                    "actor_of_node": self.actor_of_node,
                    "cost_wedge": 1.0 + 0.34 * exogenous_shock + 0.18 * (1.0 - payment_node),
                    "region_fd": region_fd,
                    "armington_elasticity": self.armington_elasticity[None, :],
                    "capacity": capacity,
                    "labor_endowment": labor_supply,
                    "capital_endowment": self.capital @ self.actor_mean_weights.T,
                },
                max_newton_iterations=10,
                max_krylov_iterations=36,
                tolerance=tolerance,
            )
            output, price = solution.output, solution.price
            wage, rental = solution.wage, solution.rental
            self.last_newton_diagnostics = solution.diagnostics
            delivered_price = price * (
                1.0 + 0.34 * exogenous_shock + 0.18 * (1.0 - payment_node)
            )
            intermediate_demand = output @ self.intermediate_sales_weights.T
            final_demand = region_fd @ self.final_sales_weights.T
            armington_response = np.clip(
                delivered_price ** (-self.armington_elasticity[None, :]), 0.25, 3.0
            )
            demand = np.clip(
                (intermediate_demand + final_demand) * armington_response, 0.001, 2.5
            )
            new_output = np.minimum(capacity, demand)
            labor_demand_node = price * new_output / np.maximum(
                wage[:, self.actor_of_node], 0.15
            )
            capital_demand_node = price * new_output / np.maximum(
                rental[:, self.actor_of_node], 0.15
            )
            labor_gap = labor_demand_node @ self.actor_gdp_weights.T - labor_supply
            capital_gap = (
                capital_demand_node @ self.actor_gdp_weights.T
                - self.capital @ self.actor_mean_weights.T
            )
            labor_comp = np.where(
                wage <= 0.250001, np.maximum(labor_gap, 0.0),
                np.where(wage >= 3.999999, np.maximum(-labor_gap, 0.0), np.abs(labor_gap)),
            )
            capital_comp = np.where(
                rental <= 0.200001, np.maximum(capital_gap, 0.0),
                np.where(rental >= 4.499999, np.maximum(-capital_gap, 0.0), np.abs(capital_gap)),
            )
            goods_path_residual = np.max(np.abs(new_output - output), axis=1)
            factor_path_residual = np.maximum(
                np.max(labor_comp, axis=1), np.max(capital_comp, axis=1)
            )
            current_residual = np.maximum(goods_path_residual, factor_path_residual)
            path_iterations[:] = solution.diagnostics.iterations
            iterations_used = solution.diagnostics.iterations
            max_iterations = 0
        for iteration in range(max_iterations):
            delivered_price = price * (
                1.0 + 0.34 * exogenous_shock + 0.18 * (1.0 - payment_node)
            )
            intermediate_demand = output @ self.intermediate_sales_weights.T
            final_demand = region_fd @ self.final_sales_weights.T
            armington_response = np.clip(
                delivered_price ** (-self.armington_elasticity[None, :]), 0.25, 3.0
            )
            demand = np.clip(
                (intermediate_demand + final_demand) * armington_response,
                0.001, 2.5,
            )
            scarcity = np.maximum(demand - capacity, 0.0)
            slack = np.maximum(capacity - demand, 0.0)
            new_output = np.minimum(capacity, demand)
            # The transport/payment wedge is a monthly level shock. Using
            # delivered_price here would compound the same wedge at every
            # equilibrium iteration and destroy the within-month fixed point.
            cost_wedge = 1.0 + 0.34 * exogenous_shock + 0.18 * (1.0 - payment_node)
            price_target = np.clip(
                cost_wedge * np.exp(0.20 * scarcity - 0.06 * slack), 0.35, 6.0
            )

            labor_demand_node = price * new_output / np.maximum(
                wage[:, self.actor_of_node], 0.15
            )
            capital_demand_node = price * new_output / np.maximum(
                rental[:, self.actor_of_node], 0.15
            )
            labor_demand = labor_demand_node @ self.actor_gdp_weights.T
            capital_demand = capital_demand_node @ self.actor_gdp_weights.T
            labor_endowment = labor_supply
            capital_endowment = self.capital @ self.actor_mean_weights.T
            wage_target = np.clip(
                wage * labor_demand / np.maximum(labor_endowment, 1e-8), 0.25, 4.0
            )
            rental_target = np.clip(
                rental * capital_demand / np.maximum(capital_endowment, 1e-8), 0.20, 4.5
            )
            goods_path_residual = np.max(np.abs(new_output - output), axis=1)
            labor_gap = labor_demand - labor_endowment
            capital_gap = capital_demand - capital_endowment
            # Mixed-complementarity residual: excess supply is admissible at
            # the price floor and excess demand at the ceiling. Interior
            # prices still require market clearing.
            labor_comp = np.where(
                wage <= 0.250001, np.maximum(labor_gap, 0.0),
                np.where(wage >= 3.999999, np.maximum(-labor_gap, 0.0), np.abs(labor_gap)),
            )
            capital_comp = np.where(
                rental <= 0.200001, np.maximum(capital_gap, 0.0),
                np.where(rental >= 4.499999, np.maximum(-capital_gap, 0.0), np.abs(capital_gap)),
            )
            factor_path_residual = np.maximum(
                np.max(labor_comp, axis=1), np.max(capital_comp, axis=1)
            )
            current_residual = np.maximum(goods_path_residual, factor_path_residual)
            path_iterations[active] = iteration + 1
            continuing = active & (current_residual >= tolerance)

            # Near complementarity boundaries the wage/rental fixed point can
            # enter a tiny two-cycle.  More iterations alone do not remove it.
            # Reduce all coupled relaxation steps pathwise when the residual is
            # already small but no longer contracts materially; recover the
            # nominal step gradually once contraction resumes.
            finite_previous = np.isfinite(previous_path_residual)
            near_solution = current_residual < 5.0e-2
            stalled = (
                finite_previous & near_solution
                & (current_residual > 0.985 * previous_path_residual)
            )
            path_relaxation = np.where(
                continuing & stalled,
                np.maximum(0.10, 0.65 * path_relaxation),
                np.where(
                    continuing,
                    np.minimum(1.0, 1.03 * path_relaxation),
                    path_relaxation,
                ),
            )
            output_step = (0.40 * path_relaxation)[:, None]
            price_step = (0.28 * path_relaxation)[:, None]
            factor_step = (0.48 * path_relaxation)[:, None]
            output = np.where(
                continuing[:, None],
                (1.0 - output_step) * output + output_step * new_output,
                output,
            )
            price = np.where(
                continuing[:, None],
                (1.0 - price_step) * price + price_step * price_target,
                price,
            )
            wage = np.where(
                continuing[:, None],
                (1.0 - factor_step) * wage + factor_step * wage_target,
                wage,
            )
            rental = np.where(
                continuing[:, None],
                (1.0 - factor_step) * rental + factor_step * rental_target,
                rental,
            )
            previous_path_residual = np.where(
                continuing, current_residual, previous_path_residual
            )
            active = continuing
            iterations_used = iteration + 1
            if not np.any(active):
                break
        goods_residual = float(np.max(goods_path_residual))
        factor_residual = float(np.max(factor_path_residual))
        if (
            os.environ.get("TAIWAN_CGE_DEBUG") == "1"
            and max(goods_residual, factor_residual) >= tolerance
        ):
            worst = int(np.argmax(current_residual))
            print(
                "CGE_NONCONVERGENCE "
                f"month={month} path={worst} iterations={int(path_iterations[worst])} "
                f"goods={goods_path_residual[worst]:.12g} "
                f"factor={factor_path_residual[worst]:.12g} "
                f"relax={path_relaxation[worst]:.12g} "
                f"labor={np.max(labor_comp[worst]):.12g} "
                f"capital={np.max(capital_comp[worst]):.12g}",
                flush=True,
            )
        self.last_solver_iterations = iterations_used
        self.max_solver_iterations = max(self.max_solver_iterations, iterations_used)

        self.output_ratio = np.clip(output, 0.001, 1.25)
        self.price = np.clip(price, 0.25, 6.0)
        self.wage = wage
        self.rental = rental
        self.shadow_price = np.maximum(demand - capacity, 0.0) * self.price
        self.last_demand = demand.copy()
        self.last_capacity = capacity.copy()
        self.last_output = self.output_ratio.copy()
        node_investment = inv_factor[:, self.actor_of_node]
        self.capital = np.clip(
            (1.0 - self.depreciation[None, :]) * self.capital
            + self.depreciation[None, :] * node_investment * fiscal[:, self.actor_of_node],
            0.05, 1.45,
        )
        self.labor = np.clip(
            0.88 * self.labor + 0.12 * labor_node * (1.0 - 0.20 * social_node), 0.08, 1.20
        )
        self.productivity = np.clip(
            self.productivity + 0.003 * (1.0 - self.productivity)
            - 0.010 * actor_damage_node - 0.006 * finance_node,
            0.45, 1.12,
        )

        gdp = self.output_ratio @ self.actor_gdp_weights.T
        industry = self.output_ratio @ self.actor_industry_weights.T
        chip_gap = 1.0 - self.output_ratio @ self.actor_electronics_weights.T
        price_index = self.price @ self.actor_gdp_weights.T
        exports = self.output_ratio @ self.actor_output_weights.T
        imports = demand @ self.actor_output_weights.T
        trade_balance = (exports - imports) / self.actor_scale[None, :]
        global_external = np.sum(
            trade_balance * self.actor_scale[None, :], axis=1
        )
        trade_balance -= global_external[:, None] / np.sum(self.actor_scale)
        external_residual = float(np.max(np.abs(
            np.sum(trade_balance * self.actor_scale[None, :], axis=1)
        )) / max(np.sum(self.actor_scale), 1e-9))
        tax_revenue = np.clip(
            (0.16 + 0.06 * fiscal) * gdp * payment, 0.0, 0.40
        )
        government_spending = 0.18 * government_multiplier * gdp
        government_balance = tax_revenue - government_spending
        inflation = np.clip(price_index - 1.0, -0.10, 1.50)
        shadow_actor = self.shadow_price @ self.actor_gdp_weights.T
        accounting_residual = float(np.max(np.abs(
            np.sum(self.output_ratio * self.out[None, :], axis=1)
            - np.sum(demand * self.out[None, :], axis=1)
        ) / max(np.sum(self.out), 1e-9)))
        self.max_accounting_residual = max(self.max_accounting_residual, accounting_residual)
        self.max_external_account_residual = max(
            self.max_external_account_residual, external_residual
        )
        self.last_goods_residual = goods_residual
        self.last_factor_residual = factor_residual
        return CGEResult(
            gdp_factor=np.clip(gdp, 0.01, 1.30),
            industry_factor=np.clip(industry, 0.01, 1.30),
            chip_gap=np.clip(chip_gap, 0.0, 0.99),
            price_inflation=inflation,
            trade_loss=trade_loss,
            trade_balance=np.clip(trade_balance, -1.0, 1.0),
            government_balance=np.clip(government_balance, -1.0, 1.0),
            capacity_shadow_price=np.clip(shadow_actor, 0.0, 5.0),
            goods_market_residual=goods_residual,
            factor_market_residual=factor_residual,
            external_account_residual=external_residual,
            solver_iterations=iterations_used,
            converged_path_fraction=float(np.mean(current_residual < tolerance)),
            converged=max(goods_residual, factor_residual) < tolerance,
        )


class OpenEconomyDSGE:
    """Eight-economy bounded-expectations New-Keynesian open DSGE solver."""

    def __init__(self, paths: int):
        self.paths = paths
        self.n_actors = len(v44.ACTORS)
        self.output_gap = np.zeros((paths, self.n_actors))
        self.inflation = np.repeat(
            np.array([[0.020, 0.020, 0.025, 0.020, 0.022, 0.021, 0.025, 0.024]]), paths, axis=0
        )
        self.expected_inflation = self.inflation.copy()
        self.expected_output_gap = self.output_gap.copy()
        self.policy_rate = np.repeat(
            np.array([[0.015, 0.020, 0.036, 0.005, 0.028, 0.025, 0.035, 0.030]]), paths, axis=0
        )
        self.exchange_rate = np.ones((paths, self.n_actors))
        self.risk_premium = np.zeros((paths, self.n_actors))
        self.investment = np.ones((paths, self.n_actors))
        self.household_shares = np.repeat(
            np.array([[
                [0.46, 0.34, 0.20], [0.38, 0.37, 0.25],
                [0.36, 0.39, 0.25], [0.34, 0.40, 0.26],
                [0.48, 0.32, 0.20], [0.42, 0.35, 0.23],
                [0.44, 0.34, 0.22], [0.46, 0.33, 0.21],
            ]]), paths, axis=0,
        )
        self.household_consumption = np.ones((paths, self.n_actors, 3))
        self.last_residual = np.inf
        self.last_solver_iterations = 0
        self.max_solver_iterations = 0
        requested_solver = os.environ.get("TAIWAN_EQUILIBRIUM_SOLVER", "auto").lower()
        self.jax_runtime = JaxResidentRuntime() if JAX_AVAILABLE else None
        gpu_jax = self.jax_runtime is not None and self.jax_runtime.platform == "gpu"
        self.solver_backend = (
            "jvp_newton_krylov" if requested_solver == "newton_krylov"
            or (requested_solver == "auto" and gpu_jax) else "picard"
        )
        self.last_newton_diagnostics = None
        self.total_solve_seconds = 0.0
        self.household_mapping_seconds = 0.0
        self.solve_calls = 0

    def snapshot_state(self) -> dict[str, np.ndarray | float]:
        """Capture expectations and policy states before fixed-point trials."""
        return {
            "output_gap": self.output_gap.copy(),
            "inflation": self.inflation.copy(),
            "expected_inflation": self.expected_inflation.copy(),
            "expected_output_gap": self.expected_output_gap.copy(),
            "policy_rate": self.policy_rate.copy(),
            "exchange_rate": self.exchange_rate.copy(),
            "risk_premium": self.risk_premium.copy(),
            "investment": self.investment.copy(),
            "household_consumption": self.household_consumption.copy(),
            "last_residual": self.last_residual,
        }

    def restore_state(self, state: dict[str, np.ndarray | float]) -> None:
        for name in (
            "output_gap", "inflation", "expected_inflation", "expected_output_gap",
            "policy_rate", "exchange_rate", "risk_premium", "investment",
            "household_consumption",
        ):
            setattr(self, name, state[name].copy())
        self.last_residual = float(state["last_residual"])

    def solve(
        self, *, cge_gdp: np.ndarray, cge_inflation: np.ndarray,
        trade_balance: np.ndarray, finance_stress: np.ndarray,
        fiscal_capacity: np.ndarray, war_risk: np.ndarray,
        social_action: np.ndarray, max_iterations: int = 240,
        tolerance: float = 8e-4,
    ) -> DSGEResult:
        solve_started = perf_counter()
        self.solve_calls += 1
        def expand4(values, external_weight=0.15, level=False):
            full = np.ones((self.paths, self.n_actors)) if level else np.zeros((self.paths, self.n_actors))
            full[:, :4] = values
            if level:
                spill = external_weight * np.mean(values - 1.0, axis=1)
                full[:, 4:] = 1.0 + spill[:, None]
            else:
                full[:, 4:] = np.mean(values, axis=1)[:, None] * external_weight
            return full

        stress = expand4(finance_stress)
        fiscal = np.clip(expand4(fiscal_capacity, level=True), 0.05, 1.0)
        risk = expand4(war_risk)
        action = expand4(social_action)
        potential_gap = np.clip(cge_gdp - 1.0, -0.95, 0.30)
        imported_inflation = np.clip(cge_inflation, -0.10, 1.50)
        y = self.output_gap.copy()
        pi = self.inflation.copy()
        rate = self.policy_rate.copy()
        exchange = self.exchange_rate.copy()
        exchange_anchor = self.exchange_rate.copy()
        investment = self.investment.copy()
        e_pi = self.expected_inflation.copy()
        e_y = self.expected_output_gap.copy()
        residual = np.inf
        natural_rate = np.array([[0.012, 0.014, 0.020, 0.004, 0.015, 0.012, 0.018, 0.016]])
        target_pi = np.array([[0.020, 0.020, 0.020, 0.020, 0.020, 0.020, 0.025, 0.025]])
        path_iterations = np.zeros(self.paths, dtype=np.int32)
        iterations_used = 0
        if self.solver_backend == "jvp_newton_krylov":
            solution = solve_dsge_jvp(
                self.jax_runtime,
                initial={
                    "output_gap": y, "inflation": pi, "policy_rate": rate,
                    "exchange_rate": exchange, "investment": investment,
                    "expected_inflation": e_pi, "expected_output_gap": e_y,
                },
                inputs={
                    "stress": stress, "risk": risk, "trade_balance": trade_balance,
                    "action": action, "fiscal": fiscal,
                    "potential_gap": potential_gap,
                    "imported_inflation": imported_inflation,
                    "natural_rate": natural_rate, "target_pi": target_pi,
                    "exchange_anchor": exchange_anchor,
                },
                max_newton_iterations=10,
                max_krylov_iterations=30,
                tolerance=tolerance,
            )
            y, pi, rate = solution.output_gap, solution.inflation, solution.policy_rate
            exchange, investment = solution.exchange_rate, solution.investment
            e_pi, e_y = solution.expected_inflation, solution.expected_output_gap
            self.last_newton_diagnostics = solution.diagnostics
            premium = np.clip(
                0.32 * stress + 0.28 * risk + 0.18 * np.maximum(-trade_balance, 0.0)
                + 0.12 * action + 0.10 * np.maximum(exchange - 1.0, 0.0), 0.0, 0.80,
            )
            real_rate_gap = rate - e_pi - natural_rate + premium
            path_residual = np.full(self.paths, solution.diagnostics.residual)
            path_iterations[:] = solution.diagnostics.iterations
            iterations_used = solution.diagnostics.iterations
            max_iterations = 0
        for iteration in range(max_iterations):
            premium = np.clip(
                0.32 * stress + 0.28 * risk + 0.18 * np.maximum(-trade_balance, 0.0)
                + 0.12 * action + 0.10 * np.maximum(exchange - 1.0, 0.0),
                0.0, 0.80,
            )
            real_rate_gap = rate - e_pi - natural_rate + premium
            y_new = np.clip(
                0.58 * e_y - 0.42 * real_rate_gap + 0.34 * potential_gap
                + 0.12 * (fiscal - 0.5) + 0.10 * trade_balance - 0.12 * action,
                -1.2, 0.45,
            )
            pi_new = np.clip(
                0.76 * e_pi + 0.14 * y_new + 0.32 * imported_inflation
                + 0.12 * premium, -0.10, 1.50,
            )
            rate_new = np.clip(
                0.82 * rate + 0.18 * (
                    natural_rate + target_pi + 1.45 * (pi_new - target_pi) + 0.35 * y_new
                ), 0.0, 0.45,
            )
            world_rate = np.mean(rate_new[:, 4:], axis=1, keepdims=True)
            exchange_new = np.clip(
                exchange_anchor * np.exp(0.10 * (rate_new - world_rate + premium - trade_balance)),
                0.35, 4.0,
            )
            q_tobin = np.clip(
                1.0 + 0.45 * y_new - 0.35 * real_rate_gap - 0.28 * stress,
                0.15, 1.60,
            )
            investment_new = np.clip(
                0.72 * investment + 0.28 * q_tobin * fiscal, 0.08, 1.45
            )
            e_pi_new = np.clip(0.70 * e_pi + 0.30 * pi_new, -0.10, 1.20)
            e_y_new = np.clip(0.68 * e_y + 0.32 * y_new, -1.0, 0.40)
            path_residual = np.max(np.stack((
                np.max(np.abs(y_new - y), axis=1),
                np.max(np.abs(pi_new - pi), axis=1),
                np.max(np.abs(rate_new - rate), axis=1),
                np.max(np.abs(exchange_new - exchange), axis=1),
                np.max(np.abs(e_pi_new - e_pi), axis=1),
                np.max(np.abs(e_y_new - e_y), axis=1),
            ), axis=1), axis=1)
            path_iterations[:] = iteration + 1
            y = 0.30 * y + 0.70 * y_new
            pi = 0.30 * pi + 0.70 * pi_new
            rate = 0.35 * rate + 0.65 * rate_new
            exchange = 0.40 * exchange + 0.60 * exchange_new
            investment = 0.38 * investment + 0.62 * investment_new
            e_pi = e_pi_new
            e_y = e_y_new
            iterations_used = iteration + 1
            if float(np.max(path_residual)) < tolerance:
                break
        residual = float(np.max(path_residual))
        self.last_solver_iterations = iterations_used
        self.max_solver_iterations = max(self.max_solver_iterations, iterations_used)
        self.output_gap = y
        self.inflation = pi
        self.policy_rate = rate
        self.exchange_rate = exchange
        self.investment = investment
        self.expected_inflation = e_pi
        self.expected_output_gap = e_y
        self.risk_premium = premium
        self.last_residual = residual
        household_started = perf_counter()
        mpc = np.array([0.92, 0.58, 0.34])[None, None, :]
        rate_exposure = np.array([0.10, 0.24, 0.42])[None, None, :]
        stress_exposure = np.array([0.48, 0.30, 0.16])[None, None, :]
        disposable_income = np.clip(
            1.0 + 0.72 * y - 0.20 * action - 0.12 * risk, 0.05, 1.30
        )
        household_target = np.clip(
            1.0 + mpc * (disposable_income[:, :, None] - 1.0)
            - rate_exposure * real_rate_gap[:, :, None]
            - stress_exposure * stress[:, :, None],
            0.05, 1.35,
        )
        self.household_consumption = np.clip(
            0.68 * self.household_consumption + 0.32 * household_target,
            0.05, 1.35,
        )
        consumption = np.sum(
            self.household_shares * self.household_consumption, axis=2
        )
        self.household_mapping_seconds += perf_counter() - household_started
        government = np.clip(0.55 + 0.45 * fiscal + 0.20 * risk, 0.20, 1.25)
        final_demand = np.clip(0.68 * consumption + 0.20 * investment + 0.12 * government, 0.12, 1.30)
        self.total_solve_seconds += perf_counter() - solve_started
        return DSGEResult(
            final_demand_factor=final_demand,
            investment_factor=investment,
            inflation=pi,
            expected_inflation=e_pi,
            policy_rate=rate,
            exchange_rate=exchange,
            risk_premium=premium,
            output_gap=y,
            equilibrium_residual=residual,
            solver_iterations=iterations_used,
            converged_path_fraction=float(np.mean(path_residual < tolerance)),
            converged=residual < tolerance,
        )


class CoupledMacroEconomicSystem:
    """Monthly CGE-DSGE fixed point with lagged finance and current war states."""

    def __init__(self, paths: int, seed: int, scenario: str):
        self.cge = DynamicMultiRegionalCGE(paths, seed, scenario)
        self.dsge = OpenEconomyDSGE(paths)
        self.paths = paths
        self.current_factors = MacroFactors(
            gdp_factor=np.ones((paths, 4)), industry_factor=np.ones((paths, 4)),
            chip_gap=np.zeros((paths, 4)), inflation=np.zeros((paths, 4)),
            trade_loss=np.zeros(paths), exchange_rate=np.ones((paths, 4)),
            policy_rate=np.zeros((paths, 4)), capacity_shadow_price=np.zeros((paths, 4)),
            trade_balance=np.zeros((paths, 4)), government_balance=np.zeros((paths, 4)),
            external_account_residual=0.0,
            cge_goods_residual=0.0, cge_factor_residual=0.0, dsge_residual=0.0,
            outer_residual=0.0, outer_iterations=0,
            outer_converged_path_fraction=1.0, converged=True,
        )
        self.max_outer_residual = 0.0
        self.nonconverged_months = 0
        self.outer_nonconverged_months = 0
        self.cge_nonconverged_months = 0
        self.dsge_nonconverged_months = 0
        self.max_cge_solver_residual = 0.0
        self.max_dsge_solver_residual = 0.0
        self.last_outer_iterations = 0
        self.max_outer_iterations_used = 0

    def current(self) -> MacroFactors:
        return self.current_factors

    def update(
        self, *, month: int, actor_damage: np.ndarray, actor_logistics: np.ndarray,
        actor_labor: np.ndarray, actor_financial_stress: np.ndarray,
        actor_payment: np.ndarray, actor_fiscal: np.ndarray,
        actor_social_action: np.ndarray, actor_panic: np.ndarray,
        actor_mobilization: np.ndarray, blockade: np.ndarray,
        max_outer_iterations: int = 18, tolerance: float = 3.0e-3,
    ) -> MacroFactors:
        cge_opening = self.cge.snapshot_state()
        dsge_opening = self.dsge.snapshot_state()
        fd = np.clip(self.dsge.investment * 0.20 + 0.80, 0.20, 1.30)
        inv = self.dsge.investment.copy()
        previous = self.current_factors.gdp_factor.copy()
        outer_residual = np.inf
        cge_result = None
        dsge_result = None
        active = np.ones(self.paths, dtype=bool)
        path_outer_residual = np.full(self.paths, np.inf)
        outer_iterations = 0
        for iteration in range(max_outer_iterations):
            # Equilibrium trials must not advance stocks or expectations. Each
            # trial starts from the same month-opening state; only the accepted
            # solution below is committed once.
            self.cge.restore_state(cge_opening)
            self.dsge.restore_state(dsge_opening)
            cge_result = self.cge.solve(
                month=month, final_demand_factor=fd, investment_factor=inv,
                actor_damage=actor_damage, actor_logistics=actor_logistics,
                actor_labor=actor_labor, actor_financial_stress=actor_financial_stress,
                actor_payment=actor_payment, actor_fiscal=actor_fiscal,
                actor_social_action=actor_social_action, actor_panic=actor_panic,
                blockade_feedback=blockade,
            )
            war_risk = np.clip(
                0.34 * actor_damage + 0.24 * (1.0 - actor_logistics)
                + 0.22 * actor_mobilization + 0.20 * blockade[:, None], 0.0, 1.0
            )
            dsge_result = self.dsge.solve(
                cge_gdp=cge_result.gdp_factor,
                cge_inflation=cge_result.price_inflation,
                trade_balance=cge_result.trade_balance,
                finance_stress=actor_financial_stress,
                fiscal_capacity=actor_fiscal,
                war_risk=war_risk,
                social_action=actor_social_action,
            )
            new_fd = dsge_result.final_demand_factor
            new_inv = dsge_result.investment_factor
            current_path_residual = np.maximum.reduce((
                np.max(np.abs(cge_result.gdp_factor[:, :4] - previous), axis=1),
                np.max(np.abs(new_fd - fd), axis=1),
                np.max(np.abs(new_inv - inv), axis=1),
            ))
            path_outer_residual[active] = current_path_residual[active]
            outer_residual = float(np.max(path_outer_residual))
            outer_iterations = iteration + 1
            converged_now = current_path_residual < tolerance
            if np.all(converged_now | ~active):
                fd = np.where(active[:, None], new_fd, fd)
                inv = np.where(active[:, None], new_inv, inv)
                break
            previous = np.where(
                active[:, None], cge_result.gdp_factor[:, :4], previous
            )
            fd = np.where(active[:, None], 0.35 * fd + 0.65 * new_fd, fd)
            inv = np.where(active[:, None], 0.35 * inv + 0.65 * new_inv, inv)
            active &= ~converged_now
        self.last_outer_iterations = outer_iterations
        self.max_outer_iterations_used = max(
            self.max_outer_iterations_used, outer_iterations
        )
        accepted_fd = fd.copy()
        accepted_inv = inv.copy()
        self.cge.restore_state(cge_opening)
        self.dsge.restore_state(dsge_opening)
        cge_result = self.cge.solve(
            month=month, final_demand_factor=accepted_fd,
            investment_factor=accepted_inv, actor_damage=actor_damage,
            actor_logistics=actor_logistics, actor_labor=actor_labor,
            actor_financial_stress=actor_financial_stress,
            actor_payment=actor_payment, actor_fiscal=actor_fiscal,
            actor_social_action=actor_social_action, actor_panic=actor_panic,
            blockade_feedback=blockade,
        )
        war_risk = np.clip(
            0.34 * actor_damage + 0.24 * (1.0 - actor_logistics)
            + 0.22 * actor_mobilization + 0.20 * blockade[:, None], 0.0, 1.0
        )
        dsge_result = self.dsge.solve(
            cge_gdp=cge_result.gdp_factor,
            cge_inflation=cge_result.price_inflation,
            trade_balance=cge_result.trade_balance,
            finance_stress=actor_financial_stress,
            fiscal_capacity=actor_fiscal,
            war_risk=war_risk,
            social_action=actor_social_action,
        )
        converged = bool(
            outer_residual < tolerance and cge_result.converged and dsge_result.converged
        )
        cge_solver_residual = max(
            cge_result.goods_market_residual, cge_result.factor_market_residual
        )
        self.max_cge_solver_residual = max(
            self.max_cge_solver_residual, cge_solver_residual
        )
        self.max_dsge_solver_residual = max(
            self.max_dsge_solver_residual, dsge_result.equilibrium_residual
        )
        if outer_residual >= tolerance:
            self.outer_nonconverged_months += 1
        if not cge_result.converged:
            self.cge_nonconverged_months += 1
        if not dsge_result.converged:
            self.dsge_nonconverged_months += 1
        if not converged:
            self.nonconverged_months += 1
        self.max_outer_residual = max(self.max_outer_residual, outer_residual)
        self.current_factors = MacroFactors(
            gdp_factor=cge_result.gdp_factor[:, :4],
            industry_factor=cge_result.industry_factor[:, :4],
            chip_gap=cge_result.chip_gap[:, :4],
            inflation=np.clip(
                0.55 * cge_result.price_inflation[:, :4] + 0.45 * dsge_result.inflation[:, :4],
                -0.10, 1.50,
            ),
            trade_loss=cge_result.trade_loss,
            exchange_rate=dsge_result.exchange_rate[:, :4],
            policy_rate=dsge_result.policy_rate[:, :4],
            capacity_shadow_price=cge_result.capacity_shadow_price[:, :4],
            trade_balance=cge_result.trade_balance[:, :4],
            government_balance=cge_result.government_balance[:, :4],
            external_account_residual=cge_result.external_account_residual,
            cge_goods_residual=cge_result.goods_market_residual,
            cge_factor_residual=cge_result.factor_market_residual,
            dsge_residual=dsge_result.equilibrium_residual,
            outer_residual=outer_residual,
            outer_iterations=outer_iterations,
            outer_converged_path_fraction=float(
                np.mean(path_outer_residual < tolerance)
            ),
            converged=converged,
        )
        return self.current_factors

    def diagnostics(self) -> dict:
        return {
            "max_outer_residual": self.max_outer_residual,
            "nonconverged_months": self.nonconverged_months,
            "outer_nonconverged_months": self.outer_nonconverged_months,
            "cge_nonconverged_months": self.cge_nonconverged_months,
            "dsge_nonconverged_months": self.dsge_nonconverged_months,
            "max_cge_solver_residual": self.max_cge_solver_residual,
            "max_dsge_solver_residual": self.max_dsge_solver_residual,
            "max_cge_accounting_residual": self.cge.max_accounting_residual,
            "max_cge_external_account_residual": self.cge.max_external_account_residual,
            "last_cge_goods_residual": self.cge.last_goods_residual,
            "last_cge_factor_residual": self.cge.last_factor_residual,
            "last_dsge_residual": self.dsge.last_residual,
            "last_outer_iterations": self.last_outer_iterations,
            "max_outer_iterations_used": self.max_outer_iterations_used,
            "last_cge_solver_iterations": self.cge.last_solver_iterations,
            "max_cge_solver_iterations": self.cge.max_solver_iterations,
            "last_dsge_solver_iterations": self.dsge.last_solver_iterations,
            "max_dsge_solver_iterations": self.dsge.max_solver_iterations,
            "cge_solver_backend": self.cge.solver_backend,
            "dsge_solver_backend": self.dsge.solver_backend,
            "cge_newton_diagnostics": (
                self.cge.last_newton_diagnostics.__dict__
                if self.cge.last_newton_diagnostics is not None else None
            ),
            "dsge_newton_diagnostics": (
                self.dsge.last_newton_diagnostics.__dict__
                if self.dsge.last_newton_diagnostics is not None else None
            ),
            "cge_jax_runtime": (
                self.cge.jax_runtime.diagnostics() if self.cge.jax_runtime else None
            ),
            "dsge_jax_runtime": (
                self.dsge.jax_runtime.diagnostics() if self.dsge.jax_runtime else None
            ),
            "dsge_timing": {
                "solve_calls": self.dsge.solve_calls,
                "total_solve_seconds": self.dsge.total_solve_seconds,
                "household_mapping_seconds": self.dsge.household_mapping_seconds,
                "household_share_of_dsge": (
                    self.dsge.household_mapping_seconds
                    / max(self.dsge.total_solve_seconds, 1e-12)
                ),
            },
        }
