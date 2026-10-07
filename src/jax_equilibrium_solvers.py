"""JAX residual formulations for the active CGE and DSGE solvers.

The routines are matrix-free: Jacobians are never assembled.  Exact JVPs feed
the pathwise Newton--Krylov implementation in accelerated_solver_infrastructure.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from accelerated_solver_infrastructure import (
    JaxResidentRuntime,
    NewtonDiagnostics,
    jax,
    jnp,
    newton_krylov,
)


@dataclass(frozen=True)
class CGENewtonSolution:
    output: np.ndarray
    price: np.ndarray
    wage: np.ndarray
    rental: np.ndarray
    diagnostics: NewtonDiagnostics


@dataclass(frozen=True)
class DSGEJVPSolution:
    output_gap: np.ndarray
    inflation: np.ndarray
    policy_rate: np.ndarray
    exchange_rate: np.ndarray
    investment: np.ndarray
    expected_inflation: np.ndarray
    expected_output_gap: np.ndarray
    diagnostics: NewtonDiagnostics


def _put_mapping(runtime: JaxResidentRuntime, prefix: str,
                 mapping: Mapping[str, np.ndarray], *,
                 static_names: frozenset[str] = frozenset()) -> dict[str, object]:
    return {
        name: (
            runtime.put_static(f"{prefix}.{name}", value)
            if name in static_names else runtime.put(f"{prefix}.{name}", value)
        )
        for name, value in mapping.items()
    }


def solve_cge_newton_krylov(
    runtime: JaxResidentRuntime, *, initial: Mapping[str, np.ndarray],
    coefficients: Mapping[str, np.ndarray], max_newton_iterations: int = 10,
    max_krylov_iterations: int = 36, tolerance: float = 2.0e-3,
) -> CGENewtonSolution:
    """Solve CGE goods, price, and bounded factor markets by exact JVP."""
    arrays = _put_mapping(
        runtime, "cge", coefficients,
        static_names=frozenset({
            "intermediate_weights", "final_weights", "actor_gdp_weights",
            "actor_of_node", "armington_elasticity",
        }),
    )
    paths, nodes = initial["output"].shape
    actors = initial["wage"].shape[1]
    packed = np.concatenate((
        initial["output"], initial["price"], initial["wage"], initial["rental"]
    ), axis=1)
    state0 = runtime.put("cge.state", packed)

    intermediate_weights = arrays["intermediate_weights"]
    final_weights = arrays["final_weights"]
    actor_gdp_weights = arrays["actor_gdp_weights"]
    actor_of_node = jnp.asarray(coefficients["actor_of_node"], dtype=jnp.int32)

    def residual(state):
        output = state[:, :nodes]
        price = state[:, nodes : 2 * nodes]
        wage = state[:, 2 * nodes : 2 * nodes + actors]
        rental = state[:, 2 * nodes + actors :]
        delivered = price * arrays["cost_wedge"]
        intermediate = output @ intermediate_weights.T
        final = arrays["region_fd"] @ final_weights.T
        armington = jnp.clip(
            delivered ** (-arrays["armington_elasticity"]), 0.25, 3.0
        )
        demand = jnp.clip((intermediate + final) * armington, 0.001, 2.5)
        scarcity = jnp.maximum(demand - arrays["capacity"], 0.0)
        slack = jnp.maximum(arrays["capacity"] - demand, 0.0)
        output_target = jnp.minimum(arrays["capacity"], demand)
        price_target = jnp.clip(
            arrays["cost_wedge"] * jnp.exp(0.20 * scarcity - 0.06 * slack),
            0.35, 6.0,
        )
        labor_node = price * output_target / jnp.maximum(
            wage[:, actor_of_node], 0.15
        )
        capital_node = price * output_target / jnp.maximum(
            rental[:, actor_of_node], 0.15
        )
        labor_demand = labor_node @ actor_gdp_weights.T
        capital_demand = capital_node @ actor_gdp_weights.T
        labor_gap = labor_demand - arrays["labor_endowment"]
        capital_gap = capital_demand - arrays["capital_endowment"]
        wage_projection = jnp.clip(wage + 0.35 * labor_gap, 0.25, 4.0)
        rental_projection = jnp.clip(rental + 0.35 * capital_gap, 0.20, 4.5)
        return jnp.concatenate((
            output - output_target,
            price - price_target,
            wage - wage_projection,
            rental - rental_projection,
        ), axis=1)

    residual_jit = runtime.compile("cge.residual", residual)
    solved, diagnostics = newton_krylov(
        residual_jit, state0, max_newton_iterations=max_newton_iterations,
        max_krylov_iterations=max_krylov_iterations, tolerance=tolerance,
        krylov_tolerance=max(tolerance * 0.12, 1.0e-7),
    )
    runtime.buffers["cge.state"] = solved
    host = runtime.to_host(solved)
    return CGENewtonSolution(
        output=np.clip(host[:, :nodes], 0.001, 1.25),
        price=np.clip(host[:, nodes : 2 * nodes], 0.25, 6.0),
        wage=np.clip(host[:, 2 * nodes : 2 * nodes + actors], 0.25, 4.0),
        rental=np.clip(host[:, 2 * nodes + actors :], 0.20, 4.5),
        diagnostics=diagnostics,
    )


def solve_dsge_jvp(
    runtime: JaxResidentRuntime, *, initial: Mapping[str, np.ndarray],
    inputs: Mapping[str, np.ndarray], max_newton_iterations: int = 10,
    max_krylov_iterations: int = 30, tolerance: float = 8.0e-4,
) -> DSGEJVPSolution:
    """Solve the bounded-expectations DSGE equilibrium with exact JAX JVPs."""
    arrays = _put_mapping(runtime, "dsge", inputs)
    names = (
        "output_gap", "inflation", "policy_rate", "exchange_rate",
        "investment", "expected_inflation", "expected_output_gap",
    )
    paths, actors = initial["output_gap"].shape
    state0 = runtime.put(
        "dsge.state", np.concatenate([initial[name] for name in names], axis=1)
    )

    def residual(state):
        y, pi, rate, exchange, investment, e_pi, e_y = jnp.split(state, 7, axis=1)
        premium = jnp.clip(
            0.32 * arrays["stress"] + 0.28 * arrays["risk"]
            + 0.18 * jnp.maximum(-arrays["trade_balance"], 0.0)
            + 0.12 * arrays["action"]
            + 0.10 * jnp.maximum(exchange - 1.0, 0.0),
            0.0, 0.80,
        )
        real_rate_gap = rate - e_pi - arrays["natural_rate"] + premium
        y_target = jnp.clip(
            0.58 * e_y - 0.42 * real_rate_gap
            + 0.34 * arrays["potential_gap"]
            + 0.12 * (arrays["fiscal"] - 0.5)
            + 0.10 * arrays["trade_balance"] - 0.12 * arrays["action"],
            -1.2, 0.45,
        )
        pi_target = jnp.clip(
            0.76 * e_pi + 0.14 * y_target
            + 0.32 * arrays["imported_inflation"] + 0.12 * premium,
            -0.10, 1.50,
        )
        rate_target = jnp.clip(
            0.82 * rate + 0.18 * (
                arrays["natural_rate"] + arrays["target_pi"]
                + 1.45 * (pi_target - arrays["target_pi"])
                + 0.35 * y_target
            ), 0.0, 0.45,
        )
        world_rate = jnp.mean(rate_target[:, 4:], axis=1, keepdims=True)
        exchange_target = jnp.clip(
            arrays["exchange_anchor"] * jnp.exp(
                0.10 * (rate_target - world_rate + premium - arrays["trade_balance"])
            ), 0.35, 4.0,
        )
        q_tobin = jnp.clip(
            1.0 + 0.45 * y_target - 0.35 * real_rate_gap
            - 0.28 * arrays["stress"], 0.15, 1.60,
        )
        investment_target = jnp.clip(
            0.72 * investment + 0.28 * q_tobin * arrays["fiscal"],
            0.08, 1.45,
        )
        e_pi_target = jnp.clip(0.70 * e_pi + 0.30 * pi_target, -0.10, 1.20)
        e_y_target = jnp.clip(0.68 * e_y + 0.32 * y_target, -1.0, 0.40)
        return jnp.concatenate((
            y - y_target, pi - pi_target, rate - rate_target,
            exchange - exchange_target, investment - investment_target,
            e_pi - e_pi_target, e_y - e_y_target,
        ), axis=1)

    residual_jit = runtime.compile("dsge.residual", residual)
    solved, diagnostics = newton_krylov(
        residual_jit, state0, max_newton_iterations=max_newton_iterations,
        max_krylov_iterations=max_krylov_iterations, tolerance=tolerance,
        krylov_tolerance=max(tolerance * 0.10, 1.0e-8),
    )
    runtime.buffers["dsge.state"] = solved
    host = runtime.to_host(solved)
    blocks = np.split(host, 7, axis=1)
    return DSGEJVPSolution(*blocks, diagnostics=diagnostics)
