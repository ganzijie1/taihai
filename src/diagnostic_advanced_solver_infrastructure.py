"""DIAGNOSTIC_ONLY checks for GPU residency, JVP/NK, and multigrid.

This script is prohibited from producing scenario probabilities or substantive
Taiwan-model outputs.  It uses analytic systems and a zero-shock macro fixture.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from accelerated_solver_infrastructure import (
    GeometricMultigrid1D,
    JaxResidentRuntime,
    TorchResidentRuntime,
    infrastructure_diagnostics,
    jnp,
    newton_krylov,
)
from dynamic_macro_equilibrium_v47 import CoupledMacroEconomicSystem


REPORT = Path(__file__).resolve().parents[1] / "outputs" / "advanced_solver_infrastructure_diagnostic.json"


def require(condition: bool, label: str) -> None:
    if not condition:
        raise AssertionError(label)


def torch_residency_fixture() -> dict:
    runtime = TorchResidentRuntime("cuda:0")
    state = runtime.put("state", np.linspace(0.0, 1.0, 4096))
    pointer = int(state.data_ptr())
    for _ in range(8):
        state.mul_(0.999).add_(0.001)
    runtime.buffers["state"] = state
    require(int(runtime.get("state").data_ptr()) == pointer, "stable CUDA allocation")
    require(runtime.host_transfer_count == 0, "no implicit host transfers")
    diagnostics = runtime.diagnostics()
    require(diagnostics["memory_allocated_bytes"] > 0, "CUDA memory allocation")
    runtime.to_host(state[:1])
    diagnostics["pointer_stable"] = True
    diagnostics["explicit_host_transfers_after_readback"] = runtime.host_transfer_count
    return diagnostics


def newton_jvp_fixture() -> dict:
    runtime = JaxResidentRuntime()
    target = runtime.put("analytic.target", np.array([[1.0, 4.0], [9.0, 16.0]]))

    def residual(value):
        return value * value - target

    initial = runtime.put("analytic.state", np.full((2, 2), 2.5))
    solution, diagnostics = newton_krylov(
        runtime.compile("analytic.residual", residual), initial,
        max_newton_iterations=12, max_krylov_iterations=8,
        tolerance=1.0e-10, krylov_tolerance=1.0e-11,
    )
    host = runtime.to_host(solution)
    error = float(np.max(np.abs(host - np.sqrt(np.array([[1.0, 4.0], [9.0, 16.0]])))))
    require(error < 1.0e-8, "Newton-JVP analytic accuracy")
    return {
        "runtime": runtime.diagnostics(),
        "iterations": diagnostics.iterations,
        "krylov_iterations": diagnostics.krylov_iterations,
        "residual": diagnostics.residual,
        "max_error": error,
        "converged": diagnostics.converged,
    }


def multigrid_fixture() -> dict:
    runtime = JaxResidentRuntime()
    paths, points = 3, 25
    diagonal = runtime.put("mg.diagonal", np.full((paths, points), 2.2))
    off = runtime.put("mg.off", np.full((paths, points - 1), -1.0))
    rhs = runtime.put("mg.rhs", np.vstack([
        np.sin(np.linspace(0.0, np.pi, points)),
        np.ones(points),
        np.linspace(-1.0, 1.0, points),
    ]))
    multigrid = GeometricMultigrid1D(diagonal, off)
    solution = jnp.zeros_like(rhs)
    initial_residual = float(jnp.max(jnp.linalg.norm(rhs, axis=1)))
    residual_history = [initial_residual]
    for _ in range(8):
        defect = rhs - multigrid._apply(diagonal, off, solution)
        solution = solution + multigrid.apply(defect)
        residual_history.append(multigrid.residual(solution, rhs))
    require(np.isfinite(residual_history[-1]), "multigrid finite residual")
    require(residual_history[-1] < 0.12 * residual_history[0], "multigrid contraction")
    return {
        "points": points,
        "paths": paths,
        "cycles": 8,
        "initial_residual": residual_history[0],
        "final_residual": residual_history[-1],
        "contraction": residual_history[-1] / residual_history[0],
    }


def zero_shock_macro_fixture() -> dict:
    os.environ["TAIWAN_EQUILIBRIUM_SOLVER"] = "newton_krylov"
    paths = 1
    zero = np.zeros((paths, 4))
    one = np.ones((paths, 4))
    system = CoupledMacroEconomicSystem(paths, 7721, "baseline")
    factors = system.update(
        month=0, actor_damage=zero, actor_logistics=one,
        actor_labor=one, actor_financial_stress=zero,
        actor_payment=one, actor_fiscal=one,
        actor_social_action=zero, actor_panic=zero,
        actor_mobilization=zero, blockade=np.zeros(paths),
        max_outer_iterations=4,
    )
    diagnostics = system.diagnostics()
    require(np.all(np.isfinite(factors.gdp_factor)), "finite zero-shock macro state")
    require(factors.external_account_residual < 1.0e-10, "external account identity")
    require(diagnostics["cge_newton_diagnostics"] is not None, "CGE NK executed")
    require(diagnostics["dsge_newton_diagnostics"] is not None, "DSGE JVP executed")
    return {
        "scope": "zero-shock numerical fixture; economic levels intentionally omitted",
        "cge_backend": diagnostics["cge_solver_backend"],
        "dsge_backend": diagnostics["dsge_solver_backend"],
        "cge_newton": diagnostics["cge_newton_diagnostics"],
        "dsge_newton": diagnostics["dsge_newton_diagnostics"],
        "external_account_residual": factors.external_account_residual,
        "outer_residual": factors.outer_residual,
        "outer_iterations": factors.outer_iterations,
    }


def main() -> None:
    report = {
        "scope": "DIAGNOSTIC_ONLY; no scenarios, probabilities, casualties, or rankings",
        "infrastructure": infrastructure_diagnostics(),
        "torch_cuda_residency": torch_residency_fixture(),
        "jax_newton_jvp": newton_jvp_fixture(),
        "geometric_multigrid": multigrid_fixture(),
        "zero_shock_cge_dsge": zero_shock_macro_fixture(),
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("DIAGNOSTIC_ONLY_ADVANCED_SOLVERS: PASS")


if __name__ == "__main__":
    main()
