"""DIAGNOSTIC_ONLY: reproducible microbenchmarks for numerical kernels.

The benchmark uses synthetic linear systems and a neutral one-month MFG
fixture.  It does not run campaign outcomes or estimate probabilities.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import platform
import statistics
import time

import numpy as np

from continuous_multipop_mfg_v47 import ContinuousMultiPopulationMFG
from dynamic_macro_equilibrium_v47 import CoupledMacroEconomicSystem
from fp_solvers_v52 import FokkerPlanckSolver, TORCH_AVAILABLE
from numerical_acceleration import (
    NUMBA_AVAILABLE, batched_thomas, euler_maruyama_step, rk4_step,
)
from rolling_differential_game_v47 import RollingHorizonDifferentialGame

if TORCH_AVAILABLE:
    import torch


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "numerical_acceleration_benchmark.json"


def timed(callable_, repeats: int) -> list[float]:
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        callable_()
        samples.append(time.perf_counter() - start)
    return samples


def summary(samples: list[float]) -> dict:
    return {
        "median_seconds": statistics.median(samples),
        "minimum_seconds": min(samples),
        "maximum_seconds": max(samples),
        "repeats": len(samples),
    }


def benchmark_tridiagonal() -> dict:
    rng = np.random.default_rng(2601)
    shape = (128, 4, 4, 25)
    lower = rng.uniform(-0.18, -0.02, shape[:-1] + (24,))
    upper = rng.uniform(-0.18, -0.02, shape[:-1] + (24,))
    diagonal = rng.uniform(1.4, 2.2, shape)
    rhs = rng.normal(size=shape)
    numpy_call = lambda: batched_thomas(
        lower, diagonal, upper, rhs, backend="numpy"
    )
    numpy_solution = numpy_call()
    result = {
        "systems": int(np.prod(shape[:-1])),
        "grid_points": shape[-1],
        "numpy": summary(timed(numpy_call, 40)),
    }
    if NUMBA_AVAILABLE:
        compile_start = time.perf_counter()
        numba_solution = batched_thomas(
            lower, diagonal, upper, rhs, backend="numba"
        )
        compile_and_first = time.perf_counter() - compile_start
        numba_call = lambda: batched_thomas(
            lower, diagonal, upper, rhs, backend="numba"
        )
        numba_times = timed(numba_call, 40)
        result["numba"] = summary(numba_times)
        result["numba_compile_and_first_seconds"] = compile_and_first
        result["hot_speedup_median"] = (
            result["numpy"]["median_seconds"]
            / result["numba"]["median_seconds"]
        )
        result["backend_max_absolute_gap"] = float(
            np.max(np.abs(numpy_solution - numba_solution))
        )
    return result


def benchmark_mfg() -> dict:
    paths = 128
    support = np.full((paths, 4), 0.55)
    zero = np.zeros((paths, 4))
    one = np.ones((paths, 4))
    update_args = dict(
        progress=zero, damage=zero, shortage=zero, inflation=zero,
        financial_stress=zero, command=one, network=one,
        contagion_mobilization=zero, contagion_fatigue=zero,
    )

    def run(backend: str):
        system = ContinuousMultiPopulationMFG(
            paths, 2602, support, numerical_backend=backend, anderson_depth=3
        )
        factors = system.update(**update_args)
        return system, factors

    numpy_samples = []
    numpy_last = None
    for _ in range(5):
        start = time.perf_counter()
        numpy_last = run("numpy")
        numpy_samples.append(time.perf_counter() - start)
    result = {"paths": paths, "numpy": summary(numpy_samples)}
    if NUMBA_AVAILABLE:
        run("numba")
        numba_samples = []
        numba_last = None
        for _ in range(5):
            start = time.perf_counter()
            numba_last = run("numba")
            numba_samples.append(time.perf_counter() - start)
        result["numba"] = summary(numba_samples)
        result["hot_speedup_median"] = (
            result["numpy"]["median_seconds"]
            / result["numba"]["median_seconds"]
        )
        result["density_max_absolute_gap"] = float(np.max(np.abs(
            numpy_last[0].density - numba_last[0].density
        )))
        result["support_max_absolute_gap"] = float(np.max(np.abs(
            numpy_last[1].support - numba_last[1].support
        )))
        result["fixed_point_iterations"] = {
            "numpy": numpy_last[1].fixed_point_iterations,
            "numba": numba_last[1].fixed_point_iterations,
        }
    return result


def benchmark_fp_transfer_fusion() -> dict:
    if not TORCH_AVAILABLE or not torch.cuda.is_available():
        return {"available": False}
    rng = np.random.default_rng(2603)
    shape = (128, 4, 4, 25)
    dx = 2.0 / 24.0
    density = np.maximum(rng.normal(0.5, 0.1, shape), 0.0)
    density /= density.sum(axis=-1, keepdims=True) * dx
    drift = rng.uniform(-0.5, 0.5, shape)
    sigma = np.full(shape, 0.12)

    def separate():
        solver = FokkerPlanckSolver(dx, backend="torch_fvm")
        evolved = density
        for _ in range(3):
            evolved, _ = solver.step(evolved, drift, sigma, 0.025)
        return evolved, solver

    def fused():
        solver = FokkerPlanckSolver(dx, backend="torch_fvm")
        evolved, _ = solver.step_repeated(density, drift, sigma, 0.025, 3)
        return evolved, solver

    for _ in range(3):
        separate()
        fused()
    separate_times = timed(lambda: separate(), 20)
    fused_times = timed(lambda: fused(), 20)
    reference, reference_solver = separate()
    candidate, candidate_solver = fused()
    return {
        "available": True,
        "backend": "torch_fvm_cuda",
        "separate": summary(separate_times),
        "fused": summary(fused_times),
        "hot_speedup_median": (
            statistics.median(separate_times) / statistics.median(fused_times)
        ),
        "max_absolute_gap": float(np.max(np.abs(reference - candidate))),
        "separate_transfers": {
            "host_to_device": reference_solver.host_to_device_transfers,
            "device_to_host": reference_solver.device_to_host_transfers,
        },
        "fused_transfers": {
            "host_to_device": candidate_solver.host_to_device_transfers,
            "device_to_host": candidate_solver.device_to_host_transfers,
        },
    }


def benchmark_rolling_game() -> dict:
    paths = 128
    game = RollingHorizonDifferentialGame(paths, 2604)
    rng = np.random.default_rng(2605)
    controls = game.controls.copy()
    arguments = (
        rng.uniform(-0.3, 0.3, (paths, 4)),
        rng.uniform(0.2, 1.2, (paths, 4)),
        rng.uniform(0.0, 0.6, (paths, 4)),
        rng.uniform(0.3, 1.1, (paths, 4)),
        rng.uniform(0.1, 0.9, (paths, 4)),
        rng.uniform(0.3, 1.0, (paths, 4)),
        rng.uniform(0.0, 1.0, (paths, 4)),
    )

    def scalar_candidates():
        return np.stack([
            game._evaluate(0, candidate, controls, *arguments)
            for candidate in game.candidates
        ], axis=1)

    def vectorized_candidates():
        return game._evaluate_candidates(
            0, game.candidates, controls, *arguments
        )

    reference = scalar_candidates()
    candidate = vectorized_candidates()
    scalar_times = timed(scalar_candidates, 80)
    vector_times = timed(vectorized_candidates, 80)
    return {
        "paths": paths,
        "candidates": len(game.candidates),
        "scalar_candidates": summary(scalar_times),
        "vectorized_candidate_axis": summary(vector_times),
        "hot_speedup_median": (
            statistics.median(scalar_times) / statistics.median(vector_times)
        ),
        "max_absolute_gap": float(np.max(np.abs(reference - candidate))),
    }


def _macro_case(paths: int, *, stressed: bool) -> tuple[str, dict]:
    rng = np.random.default_rng(2606)
    zero = np.zeros((paths, 4))
    one = np.ones((paths, 4))
    if not stressed:
        return "baseline", dict(
            month=0, actor_damage=zero, actor_logistics=one,
            actor_labor=one, actor_financial_stress=zero,
            actor_payment=one, actor_fiscal=one,
            actor_social_action=zero, actor_panic=zero,
            actor_mobilization=zero, blockade=np.zeros(paths),
        )
    return "combined", dict(
        month=24, actor_damage=rng.uniform(0.15, 0.65, (paths, 4)),
        actor_logistics=rng.uniform(0.20, 0.80, (paths, 4)),
        actor_labor=rng.uniform(0.45, 0.95, (paths, 4)),
        actor_financial_stress=rng.uniform(0.10, 0.70, (paths, 4)),
        actor_payment=rng.uniform(0.35, 0.95, (paths, 4)),
        actor_fiscal=rng.uniform(0.30, 0.90, (paths, 4)),
        actor_social_action=rng.uniform(0.05, 0.55, (paths, 4)),
        actor_panic=rng.uniform(0.03, 0.45, (paths, 4)),
        actor_mobilization=rng.uniform(0.10, 0.80, (paths, 4)),
        blockade=rng.uniform(0.30, 0.90, paths),
    )


def benchmark_macro_native_windows() -> dict:
    paths = 64
    previous = os.environ.get("TAIWAN_EQUILIBRIUM_SOLVER")

    def run(backend: str, stressed: bool):
        os.environ["TAIWAN_EQUILIBRIUM_SOLVER"] = backend
        scenario, inputs = _macro_case(paths, stressed=stressed)
        system = CoupledMacroEconomicSystem(paths, 2607, scenario)
        started = time.perf_counter()
        factors = system.update(**inputs)
        return factors, system.diagnostics(), time.perf_counter() - started

    try:
        # Compile the Numba kernel outside timed samples.
        warm_scenario, warm_inputs = _macro_case(2, stressed=False)
        os.environ["TAIWAN_EQUILIBRIUM_SOLVER"] = "auto"
        CoupledMacroEconomicSystem(2, 2607, warm_scenario).update(**warm_inputs)
        cases = {}
        for label, stressed in (("neutral", False), ("stressed", True)):
            reference, ref_diag, ref_time = run("picard", stressed)
            accelerated, acc_diag, acc_time = run("auto", stressed)
            fields = (
                "gdp_factor", "industry_factor", "chip_gap", "inflation",
                "trade_loss", "exchange_rate", "policy_rate",
                "capacity_shadow_price", "trade_balance", "government_balance",
            )
            gap = max(float(np.max(np.abs(
                getattr(reference, name) - getattr(accelerated, name)
            ))) for name in fields)
            cases[label] = {
                "picard_seconds": ref_time,
                "native_windows_seconds": acc_time,
                "speedup": ref_time / max(acc_time, 1.0e-12),
                "max_state_gap": gap,
                "picard_converged": reference.converged,
                "native_windows_converged": accelerated.converged,
                "picard_residuals": {
                    "outer": reference.outer_residual,
                    "cge_goods": reference.cge_goods_residual,
                    "cge_factor": reference.cge_factor_residual,
                    "dsge": reference.dsge_residual,
                },
                "native_windows_residuals": {
                    "outer": accelerated.outer_residual,
                    "cge_goods": accelerated.cge_goods_residual,
                    "cge_factor": accelerated.cge_factor_residual,
                    "dsge": accelerated.dsge_residual,
                },
                "backends": {
                    "cge": acc_diag["cge_solver_backend"],
                    "dsge": acc_diag["dsge_solver_backend"],
                },
            }
        return {"paths": paths, "cases": cases}
    finally:
        if previous is None:
            os.environ.pop("TAIWAN_EQUILIBRIUM_SOLVER", None)
        else:
            os.environ["TAIWAN_EQUILIBRIUM_SOLVER"] = previous


def benchmark_ode_sde_batches() -> dict:
    rng = np.random.default_rng(2608)
    paths = 4096
    initial = rng.normal(size=(paths, 4))
    normals = rng.standard_normal((paths, 4))

    def rhs(_time, state, rate):
        return rate * state

    vector_rk4 = lambda: rk4_step(rhs, 0.0, initial, 0.025, -0.7)
    scalar_rk4 = lambda: np.stack([
        rk4_step(rhs, 0.0, initial[index:index + 1], 0.025, -0.7)[0]
        for index in range(paths)
    ])
    vector_em = lambda: euler_maruyama_step(
        initial, -0.4 * initial, np.full_like(initial, 0.12), normals, 0.025
    )
    scalar_em = lambda: np.stack([
        euler_maruyama_step(
            initial[index:index + 1], -0.4 * initial[index:index + 1],
            np.full((1, 4), 0.12), normals[index:index + 1], 0.025,
        )[0] for index in range(paths)
    ])
    rk_reference = scalar_rk4()
    rk_vector = vector_rk4()
    em_reference = scalar_em()
    em_vector = vector_em()
    rk_scalar_times = timed(scalar_rk4, 3)
    rk_vector_times = timed(vector_rk4, 30)
    em_scalar_times = timed(scalar_em, 3)
    em_vector_times = timed(vector_em, 30)
    return {
        "paths": paths,
        "rk4": {
            "scalar": summary(rk_scalar_times),
            "vectorized": summary(rk_vector_times),
            "speedup": statistics.median(rk_scalar_times) / statistics.median(rk_vector_times),
            "max_absolute_gap": float(np.max(np.abs(rk_reference - rk_vector))),
        },
        "euler_maruyama": {
            "scalar": summary(em_scalar_times),
            "vectorized": summary(em_vector_times),
            "speedup": statistics.median(em_scalar_times) / statistics.median(em_vector_times),
            "max_absolute_gap": float(np.max(np.abs(em_reference - em_vector))),
        },
    }


def main() -> None:
    report = {
        "scope": "DIAGNOSTIC_ONLY",
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "numba_available": NUMBA_AVAILABLE,
        "timing_note": "wall clock; compile time separated; no outcome simulation",
        "tridiagonal": benchmark_tridiagonal(),
        "mfg_one_month_neutral_fixture": benchmark_mfg(),
        "fp_cuda_transfer_fusion": benchmark_fp_transfer_fusion(),
        "rolling_differential_game": benchmark_rolling_game(),
        "macro_native_windows": benchmark_macro_native_windows(),
        "ode_sde_path_batching": benchmark_ode_sde_batches(),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("DIAGNOSTIC_ONLY_NUMERICAL_BENCHMARK: PASS")


if __name__ == "__main__":
    main()
