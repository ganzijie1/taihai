"""DIAGNOSTIC_ONLY: numerical-kernel equivalence and conservation checks.

This file does not execute an outcome simulation or aggregate scenario
probabilities.  It only validates isolated numerical kernels and one-step
solver fixtures.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from continuous_multipop_mfg_v47 import ContinuousMultiPopulationMFG
from dynamic_macro_equilibrium_v47 import CoupledMacroEconomicSystem
from numerical_acceleration import (
    NUMBA_AVAILABLE,
    backend_diagnostics,
    batched_thomas,
    euler_maruyama_step,
    rk4_step,
)
from wartime_social_contagion import WartimeSocialContagionSystem


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "outputs" / "numerical_acceleration_diagnostic.json"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def tridiagonal_fixture() -> dict:
    rng = np.random.default_rng(1901)
    shape = (7, 4, 17)
    lower = rng.uniform(-0.18, -0.02, shape[:-1] + (shape[-1] - 1,))
    upper = rng.uniform(-0.18, -0.02, shape[:-1] + (shape[-1] - 1,))
    diagonal = rng.uniform(1.4, 2.2, shape)
    rhs = rng.normal(size=shape)
    reference = np.empty_like(rhs)
    for index in np.ndindex(shape[:-1]):
        matrix = np.diag(diagonal[index])
        matrix += np.diag(lower[index], -1)
        matrix += np.diag(upper[index], 1)
        reference[index] = np.linalg.solve(matrix, rhs[index])
    numpy_solution = batched_thomas(lower, diagonal, upper, rhs, backend="numpy")
    numpy_error = float(np.max(np.abs(numpy_solution - reference)))
    require(numpy_error < 2.0e-12, "NumPy Thomas solver equivalence")
    result = {"numpy_max_error": numpy_error}
    if NUMBA_AVAILABLE:
        numba_solution = batched_thomas(
            lower, diagonal, upper, rhs, backend="numba"
        )
        numba_error = float(np.max(np.abs(numba_solution - reference)))
        backend_gap = float(np.max(np.abs(numba_solution - numpy_solution)))
        require(numba_error < 2.0e-12, "Numba Thomas solver equivalence")
        require(backend_gap < 2.0e-13, "NumPy/Numba Thomas parity")
        result.update({
            "numba_max_error": numba_error,
            "numpy_numba_gap": backend_gap,
        })
    return result


def differential_equation_fixtures() -> dict:
    initial = np.array([[1.0], [2.0], [3.0]])

    def rhs(_time, state, rate):
        return rate * state

    step = rk4_step(rhs, 0.0, initial, 0.05, -0.7)
    exact = initial * np.exp(-0.7 * 0.05)
    rk4_error = float(np.max(np.abs(step - exact)))
    require(rk4_error < 2.0e-8, "RK4 exponential fixture")

    state = np.array([[0.2, -0.1], [0.4, 0.3]])
    drift = np.full_like(state, 0.06)
    diffusion = np.full_like(state, 0.15)
    normal = np.array([[0.5, -0.2], [1.1, 0.0]])
    dt = 0.04
    step = euler_maruyama_step(state, drift, diffusion, normal, dt)
    expected = state + drift * dt + diffusion * np.sqrt(dt) * normal
    em_error = float(np.max(np.abs(step - expected)))
    require(em_error < 1.0e-15, "Euler-Maruyama deterministic fixture")
    return {"rk4_max_error": rk4_error, "euler_maruyama_max_error": em_error}


def mfg_fixture() -> dict:
    paths = 3
    support = np.full((paths, 4), 0.55)
    zero = np.zeros((paths, 4))
    one = np.ones((paths, 4))
    update_args = dict(
        progress=zero, damage=zero, shortage=zero, inflation=zero,
        financial_stress=zero, command=one, network=one,
        contagion_mobilization=zero, contagion_fatigue=zero,
    )
    system = ContinuousMultiPopulationMFG(
        paths, 1902, support, numerical_backend="numpy", anderson_depth=3
    )
    factors = system.update(**update_args)
    require(factors.mass_residual < 1.0e-10, "MFG mass conservation")
    require(np.isfinite(factors.bellman_residual), "MFG Bellman residual")
    require(np.isfinite(factors.density_residual), "MFG density residual")
    require(factors.fixed_point_iterations <= 48, "MFG fixed-point cap")
    require(factors.max_policy_iterations_used <= 18, "MFG policy cap")
    result = {
        "backend": factors.numerical_backend,
        "mass_residual": factors.mass_residual,
        "mean_residual": factors.mean_residual,
        "density_residual": factors.density_residual,
        "bellman_residual": factors.bellman_residual,
        "fixed_point_iterations": factors.fixed_point_iterations,
        "max_policy_iterations_used": factors.max_policy_iterations_used,
        "converged": factors.converged,
    }
    if NUMBA_AVAILABLE:
        accelerated = ContinuousMultiPopulationMFG(
            paths, 1902, support, numerical_backend="numba", anderson_depth=3
        )
        accelerated_factors = accelerated.update(**update_args)
        density_gap = float(np.max(np.abs(system.density - accelerated.density)))
        support_gap = float(np.max(np.abs(
            factors.support - accelerated_factors.support
        )))
        require(density_gap < 2.0e-13, "MFG NumPy/Numba density parity")
        require(support_gap < 2.0e-13, "MFG NumPy/Numba output parity")
        result.update({
            "accelerated_backend": accelerated_factors.numerical_backend,
            "numpy_numba_density_gap": density_gap,
            "numpy_numba_support_gap": support_gap,
        })
    return result


def social_fixture() -> dict:
    paths = 5
    support = np.full((paths, 4), 0.56)
    zero = np.zeros((paths, 4))
    one = np.ones((paths, 4))
    kwargs = dict(
        prior_support=support, actor_damage=zero, actor_shortage=zero,
        actor_inflation=zero, actor_backlash=zero, actor_network=one,
        actor_norm=support, actor_resilience=one, progress_signal=zero,
        command_integrity=one, conflict_intensity=np.zeros(paths),
    )
    first = WartimeSocialContagionSystem(paths, 1903, support)
    second = WartimeSocialContagionSystem(paths, 1903, support)
    first_result = first.update(**kwargs)
    second_result = second.update(**kwargs)
    replay_error = float(np.max(np.abs(first.state - second.state)))
    require(replay_error == 0.0, "social fixed-seed replay")
    require(first_result["conservation_error"] < 1.0e-12, "social conservation")
    require(
        np.array_equal(first_result["support"], second_result["support"]),
        "social output replay",
    )
    return {
        "fixed_seed_replay_error": replay_error,
        "conservation_error": first_result["conservation_error"],
        "monthly_random_batches": first.random_batches_generated,
    }


def macro_fixture() -> dict:
    paths = 2
    zero = np.zeros((paths, 4))
    one = np.ones((paths, 4))
    system = CoupledMacroEconomicSystem(paths, 1904, "baseline")
    factors = system.update(
        month=0, actor_damage=zero, actor_logistics=one,
        actor_labor=one, actor_financial_stress=zero,
        actor_payment=one, actor_fiscal=one,
        actor_social_action=zero, actor_panic=zero,
        actor_mobilization=zero, blockade=np.zeros(paths),
    )
    require(np.all(np.isfinite(factors.gdp_factor)), "macro finite state")
    require(
        factors.external_account_residual < 1.0e-10,
        "CGE external account clearing",
    )
    require(factors.outer_iterations <= 18, "macro outer iteration cap")
    diagnostics = system.diagnostics()
    return {
        "converged": factors.converged,
        "outer_residual": factors.outer_residual,
        "outer_iterations": factors.outer_iterations,
        "outer_converged_path_fraction": factors.outer_converged_path_fraction,
        "cge_solver_iterations": diagnostics["last_cge_solver_iterations"],
        "dsge_solver_iterations": diagnostics["last_dsge_solver_iterations"],
        "external_account_residual": factors.external_account_residual,
    }


def main() -> None:
    report = {
        "scope": "DIAGNOSTIC_ONLY",
        "backend": backend_diagnostics(),
        "tridiagonal": tridiagonal_fixture(),
        "differential_equations": differential_equation_fixtures(),
        "mfg": mfg_fixture(),
        "social_sde_jump": social_fixture(),
        "macro": macro_fixture(),
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("DIAGNOSTIC_ONLY_NUMERICAL_ACCELERATION: PASS")


if __name__ == "__main__":
    main()
