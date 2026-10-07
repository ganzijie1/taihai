from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "work"))

from cognitive_geometry_v52 import CognitiveInformationGeometry
from continuous_multipop_mfg_v47 import ContinuousMultiPopulationMFG
from fp_solvers_v52 import FokkerPlanckSolver


OUT = ROOT / "outputs" / "taiwan_v52_diagnostic_only.json"


def fixture(paths: int = 4) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    rng = np.random.default_rng(520_031)
    initial = np.clip(0.52 + 0.08 * rng.normal(size=(paths, 4)), 0.22, 0.82)
    values = {
        "progress": 0.12 * rng.normal(size=(paths, 4)),
        "damage": 0.18 * rng.random((paths, 4)),
        "shortage": 0.16 * rng.random((paths, 4)),
        "inflation": 0.10 * rng.random((paths, 4)),
        "financial_stress": 0.12 * rng.random((paths, 4)),
        "command": 0.58 + 0.32 * rng.random((paths, 4)),
        "network": 0.50 + 0.38 * rng.random((paths, 4)),
        "contagion_mobilization": 0.10 * rng.random((paths, 4)),
        "contagion_fatigue": 0.10 * rng.random((paths, 4)),
        "common_noise": 0.12 * rng.normal(size=(paths, 4)),
        "jump_signal": 0.24 * (rng.random((paths, 4)) < 0.18),
    }
    return initial, values


def legacy_reference(density, drift, sigma, dx, dt):
    interface = 0.5 * (drift[..., 1:] + drift[..., :-1])
    flux = np.where(
        interface >= 0.0, interface * density[..., :-1],
        interface * density[..., 1:]
    ) - 0.25 * (sigma[..., 1:] ** 2 + sigma[..., :-1] ** 2) * (
        density[..., 1:] - density[..., :-1]
    ) / dx
    evolved = density.copy()
    evolved[..., 0] -= dt * flux[..., 0] / dx
    evolved[..., 1:-1] -= dt * (flux[..., 1:] - flux[..., :-1]) / dx
    evolved[..., -1] += dt * flux[..., -1] / dx
    evolved = np.maximum(evolved, 0.0)
    return evolved / (np.sum(evolved, axis=-1, keepdims=True) * dx)


def run() -> dict:
    x = np.linspace(-1.0, 1.0, 25)
    dx = float(x[1] - x[0])
    density = np.exp(-0.5 * ((x[None, :] - 0.18) / 0.24) ** 2)
    density /= np.sum(density, axis=-1, keepdims=True) * dx
    drift = np.broadcast_to(-0.22 * x[None, :], density.shape).copy()
    sigma = np.full(density.shape, 0.09)
    solutions = {}
    solver_diagnostics = {}
    for backend in ("legacy_explicit", "numpy_fvm", "torch_fvm", "torchdiffeq"):
        solver = FokkerPlanckSolver(dx, backend=backend)
        result, _ = solver.step(density, drift, sigma, 0.075)
        solutions[backend] = np.asarray(result)
        solver_diagnostics[backend] = solver.diagnostics()

    reference = legacy_reference(density, drift, sigma, dx, 0.075)
    legacy_error = float(np.max(np.abs(reference - solutions["legacy_explicit"])))
    cpu_gpu_error = float(np.max(np.abs(
        solutions["numpy_fvm"] - solutions["torch_fvm"]
    )))
    mass_residual = float(max(
        np.max(np.abs(np.sum(value, axis=-1) * dx - 1.0))
        for value in solutions.values()
    ))
    minimum_density = float(min(np.min(value) for value in solutions.values()))

    differentiable_solver = FokkerPlanckSolver(dx, backend="torchdiffeq")
    density_t = torch.tensor(density, dtype=torch.float64, device="cuda")
    drift_t = torch.tensor(
        drift, dtype=torch.float64, device="cuda", requires_grad=True
    )
    sigma_t = torch.tensor(sigma, dtype=torch.float64, device="cuda")
    differentiable, _ = differentiable_solver.step(
        density_t, drift_t, sigma_t, 0.025
    )
    objective = torch.sum(differentiable * torch.tensor(
        x, dtype=torch.float64, device="cuda"
    )) * dx
    objective.backward()
    gradient_finite = bool(torch.all(torch.isfinite(drift_t.grad)).item())
    gradient_norm = float(torch.linalg.vector_norm(drift_t.grad).detach().cpu())

    geometry = CognitiveInformationGeometry(x)
    reversed_density = np.flip(density, axis=-1).copy()
    mixed, fisher_distance = geometry.fisher_rao_mix(
        density, reversed_density, 0.5
    )
    reaction, engagement, wfr_distance = geometry.wfr_reaction_step(
        density, np.ones((1,)), 0.20 * x[None, :], 0.2
    )
    sinkhorn_distance = geometry.sinkhorn_distance(density, reversed_density)
    geometry_mass_residual = float(max(
        np.max(np.abs(np.sum(mixed, axis=-1) * dx - 1.0)),
        np.max(np.abs(np.sum(reaction, axis=-1) * dx - 1.0)),
    ))

    initial, inputs = fixture()
    trajectories = {}
    mfg_diagnostics = {}
    for mode in ("none", "block", "low_rank", "multilayer", "finite"):
        model = ContinuousMultiPopulationMFG(
            len(initial), 520_101, initial, geometry_enabled=True,
            advanced_graphon=True, graphon_mode=mode, fp_backend="torch_fvm",
            geomloss_interval=2,
        )
        snapshots = []
        for _ in range(3):
            result = model.update(**inputs)
            snapshots.append(result.support.copy())
        trajectories[mode] = np.stack(snapshots)
        mfg_diagnostics[mode] = model.diagnostics()

    main = trajectories["multilayer"]
    ablation_rmse = {
        mode: float(np.sqrt(np.mean((trajectory - main) ** 2)))
        for mode, trajectory in trajectories.items() if mode != "multilayer"
    }
    main_diag = mfg_diagnostics["multilayer"]
    assertions = {
        "legacy_zero_change": legacy_error < 1.0e-14,
        "cpu_gpu_equivalence": cpu_gpu_error < 1.0e-11,
        "fp_mass_conservation": mass_residual < 1.0e-12,
        "fp_nonnegative": minimum_density >= -1.0e-12,
        "torchdiffeq_gradient": gradient_finite and gradient_norm > 0.0,
        "fisher_wfr_mass": geometry_mass_residual < 1.0e-12,
        "geomloss_active": np.isfinite(sinkhorn_distance)
        and geometry.diagnostics()["geomloss_failures"] == 0,
        "mfg_converged": all(
            item["nonconverged_months"] == 0 for item in mfg_diagnostics.values()
        ),
        "mfg_mass_conservation": max(
            item["max_mass_residual"] for item in mfg_diagnostics.values()
        ) < 1.0e-10,
        "kernel_rows_normalised": main_diag["max_kernel_row_residual"] < 1.0e-12,
        "mean_exploitability_reported": np.isfinite(
            main_diag["max_mean_exploitability_bound"]
        ),
        "worst_exploitability_reported": np.isfinite(
            main_diag["max_worst_type_exploitability_bound"]
        ),
        "finite_network_backtest": 0.0 < main_diag["max_finite_network_field_error"] < 0.25,
        "all_graph_ablations_move_state": all(value > 1.0e-8 for value in ablation_rmse.values()),
        "kernel_uncertainty_non_degenerate": (
            main_diag["kernel_uncertainty"]["p95"]
            > main_diag["kernel_uncertainty"]["p05"]
        ),
        "hard_capacity_flow_not_replaced": "geomloss" not in (
            ROOT / "work" / "operational_constraints_v47.py"
        ).read_text(encoding="utf-8").lower(),
    }
    assertions = {key: bool(value) for key, value in assertions.items()}
    result = {
        "label": "DIAGNOSTIC_ONLY",
        "substantive_outcomes_generated": False,
        "status": "PASS" if all(assertions.values()) else "FAIL",
        "assertions": assertions,
        "fp": {
            "legacy_error": legacy_error,
            "cpu_gpu_error": cpu_gpu_error,
            "mass_residual": mass_residual,
            "minimum_density": minimum_density,
            "torchdiffeq_gradient_norm": gradient_norm,
            "backend_diagnostics": solver_diagnostics,
        },
        "geometry": {
            "fisher_rao_distance": fisher_distance,
            "wfr_distance": wfr_distance,
            "engagement_mass": engagement.tolist(),
            "sinkhorn_distance": sinkhorn_distance,
            "mass_residual": geometry_mass_residual,
            "diagnostics": geometry.diagnostics(),
        },
        "mfg": {
            "graph_ablation_support_rmse": ablation_rmse,
            "diagnostics": mfg_diagnostics,
        },
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    report = run()
    print(f"TAIWAN_V52_DIAGNOSTIC_ONLY: {report['status']}")
    if report["status"] != "PASS":
        raise SystemExit(1)
