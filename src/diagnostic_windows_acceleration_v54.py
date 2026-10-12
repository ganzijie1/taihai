"""DIAGNOSTIC_ONLY acceptance gate for the native-Windows V5.4 solvers."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "work"
OUT = ROOT / "outputs"
sys.path.insert(0, str(WORK))

from dynamic_macro_equilibrium_v47 import CoupledMacroEconomicSystem
from continuous_multipop_mfg_v47 import ContinuousMultiPopulationMFG


BENCHMARK = OUT / "numerical_acceleration_benchmark.json"
V53_DIAGNOSTIC = OUT / "taiwan_v53_diagnostic_only.json"
REPORT = OUT / "taiwan_v54_windows_acceleration_diagnostic.json"


def require(condition: bool, label: str) -> None:
    if not condition:
        raise AssertionError(label)


def main() -> None:
    benchmark = json.loads(BENCHMARK.read_text(encoding="utf-8"))
    inherited = json.loads(V53_DIAGNOSTIC.read_text(encoding="utf-8"))
    checks = {
        "inherited_v53_diagnostic_pass": inherited.get("status") == "PASS",
        "tridiagonal_equivalent": (
            benchmark["tridiagonal"].get("backend_max_absolute_gap", 1.0) < 1.0e-12
        ),
        "mfg_hjb_equivalent": (
            benchmark["mfg_one_month_neutral_fixture"].get(
                "density_max_absolute_gap", 1.0
            ) < 2.0e-12
        ),
        "fp_fused_equivalent": (
            benchmark["fp_cuda_transfer_fusion"].get("max_absolute_gap", 1.0)
            < 1.0e-12
        ),
        "fp_transfer_count_reduced": (
            benchmark["fp_cuda_transfer_fusion"]["fused_transfers"]["host_to_device"]
            < benchmark["fp_cuda_transfer_fusion"]["separate_transfers"]["host_to_device"]
            and benchmark["fp_cuda_transfer_fusion"]["fused_transfers"]["device_to_host"]
            < benchmark["fp_cuda_transfer_fusion"]["separate_transfers"]["device_to_host"]
        ),
        "differential_game_equivalent": (
            benchmark["rolling_differential_game"]["max_absolute_gap"] < 1.0e-12
        ),
        "ode_batch_equivalent": (
            benchmark["ode_sde_path_batching"]["rk4"]["max_absolute_gap"] < 1.0e-12
        ),
        "sde_batch_equivalent": (
            benchmark["ode_sde_path_batching"]["euler_maruyama"]["max_absolute_gap"]
            < 1.0e-12
        ),
    }
    for label, case in benchmark["macro_native_windows"]["cases"].items():
        checks[f"macro_{label}_state_equivalent"] = case["max_state_gap"] < 2.0e-12
        checks[f"macro_{label}_reference_converged"] = case["picard_converged"]
        checks[f"macro_{label}_accelerated_converged"] = case[
            "native_windows_converged"
        ]
        checks[f"macro_{label}_measured_speedup"] = case["speedup"] > 1.0

    previous = os.environ.pop("TAIWAN_EQUILIBRIUM_SOLVER", None)
    try:
        macro = CoupledMacroEconomicSystem(2, 5401, "baseline")
        support = np.full((2, 4), 0.55)
        mfg = ContinuousMultiPopulationMFG(
            2, 5402, support, fp_backend="torch_fvm"
        )
        selected = {
            "cge": macro.cge.solver_backend,
            "dsge": macro.dsge.solver_backend,
            "mfg_hjb": mfg.numerical_backend,
            "mfg_fp": mfg.fp_solver.backend,
            "mfg_fp_device": mfg.fp_solver.device,
        }
    finally:
        if previous is not None:
            os.environ["TAIWAN_EQUILIBRIUM_SOLVER"] = previous

    checks.update({
        "formal_cge_equivalence_preserving": selected["cge"] == "picard",
        "formal_dsge_compiled": selected["dsge"] == "numba_picard",
        "formal_mfg_hjb_compiled": selected["mfg_hjb"] == "numba",
        "formal_mfg_fp_cuda": (
            selected["mfg_fp"] == "torch_fvm"
            and selected["mfg_fp_device"].startswith("cuda")
        ),
    })
    for label, value in checks.items():
        require(bool(value), label)

    report = {
        "scope": "DIAGNOSTIC_ONLY; no probabilities or scenario rankings",
        "status": "PASS",
        "platform_policy": {
            "operating_system": "native Windows",
            "wsl_required": False,
            "jax_cuda_required": False,
            "multi_gpu_required": False,
            "formal_backends": selected,
        },
        "checks": checks,
        "benchmark": benchmark,
    }
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("TAIWAN_V54_WINDOWS_ACCELERATION_DIAGNOSTIC: PASS")


if __name__ == "__main__":
    main()
