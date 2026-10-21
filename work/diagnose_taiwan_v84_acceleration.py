from __future__ import annotations

import concurrent.futures
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import triton

from continuous_multipop_mfg_v47 import ContinuousMultiPopulationMFG
from fp_solvers_v52 import _torch_pcr, _torch_thomas, _torch_thomas_compiled
from statistical_contact_v84 import StatisticalContactSystemV84
from variant_batching import VariantBatchLayout, shared_variant_random_context


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "taiwan_v84_acceleration_diagnostic.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _pid(_: int) -> int:
    return os.getpid()


def _benchmark(function, arguments, repetitions: int = 30) -> float:
    for _ in range(3):
        function(*arguments)
    torch.cuda.synchronize()
    opening = time.perf_counter()
    for _ in range(repetitions):
        function(*arguments)
    torch.cuda.synchronize()
    return 1000.0 * (time.perf_counter() - opening) / repetitions


def tridiagonal_diagnostic() -> dict:
    records = []
    for systems in (2_048, 28_672):
        generator = torch.Generator(device="cuda").manual_seed(84_052 + systems)
        lower = -0.04 * torch.rand((systems, 24), generator=generator, device="cuda", dtype=torch.float64)
        upper = -0.04 * torch.rand((systems, 24), generator=generator, device="cuda", dtype=torch.float64)
        diagonal = 1.2 + torch.rand((systems, 25), generator=generator, device="cuda", dtype=torch.float64)
        rhs = torch.rand((systems, 25), generator=generator, device="cuda", dtype=torch.float64)
        arguments = (lower, diagonal, upper, rhs)
        reference = _torch_thomas(*arguments)
        compiled = _torch_thomas_compiled(*arguments)
        pcr = _torch_pcr(*arguments)
        records.append({
            "systems": systems,
            "compiled_max_abs_error": float(torch.max(torch.abs(compiled - reference))),
            "pcr_max_abs_error": float(torch.max(torch.abs(pcr - reference))),
            "eager_thomas_ms": _benchmark(_torch_thomas, arguments),
            "compiled_thomas_ms": _benchmark(_torch_thomas_compiled, arguments),
            "pcr_ms": _benchmark(_torch_pcr, arguments),
        })
    return {"grid_points": 25, "records": records}


def variant_diagnostic() -> dict:
    names = tuple(f"variant_{index:02d}" for index in range(14))
    layout = VariantBatchLayout(names, 8)
    with shared_variant_random_context(layout):
        generator = np.random.default_rng(84_061)
        draw = generator.normal(np.arange(layout.total_paths), 0.7)
    grouped_noise = layout.unflatten(draw - np.arange(layout.total_paths))
    crn_error = float(np.max(np.abs(grouped_noise - grouped_noise[0])))

    masks = {
        "wavelet_enabled": np.repeat([True, False, True], 4),
        "contact_enabled": np.repeat([True, True, False], 4),
    }
    batched = StatisticalContactSystemV84(12, np.eye(4), **masks)
    separate = [
        StatisticalContactSystemV84(4, np.eye(4), wavelet_enabled=w, contact_enabled=c)
        for w, c in ((True, True), (False, True), (True, False))
    ]
    generator = np.random.default_rng(84_067)
    equivalence_error = 0.0
    for _month in range(12):
        base = {
            key: generator.random((4, 4))
            for key in (
                "shortage", "inflation", "financial_stress", "contagion_fatigue",
                "damage", "network", "command", "jump_signal",
            )
        }
        base["jump_signal"] *= 0.1
        packed = {key: np.tile(value, (3, 1)) for key, value in base.items()}
        batched_factors = batched.update(**packed)
        for index, system in enumerate(separate):
            factors = system.update(**base)
            section = slice(index * 4, (index + 1) * 4)
            for field in batched_factors.__dataclass_fields__:
                equivalence_error = max(
                    equivalence_error,
                    float(np.max(np.abs(
                        getattr(batched_factors, field)[section] - getattr(factors, field)
                    ))),
                )
    return {
        "variant_count": 14, "paths_per_variant": 8,
        "common_random_number_error": crn_error,
        "statistical_layer_batch_equivalence_error": equivalence_error,
    }


def mfg_residency_diagnostic() -> dict:
    paths = 2
    initial = np.full((paths, 4), 0.60)
    model = ContinuousMultiPopulationMFG(
        paths, 84_071, initial, geometry_enabled=True, advanced_graphon=True,
        fp_backend="torch_fvm", full_wfr_jko=True,
        riemannian_anderson_enabled=True, riemannian_trust_region_enabled=True,
        riemannian_fixed_point_iterations=48,
    )
    zero, one = np.zeros((paths, 4)), np.ones((paths, 4))
    model.update(
        progress=zero, damage=zero, shortage=zero, inflation=zero,
        financial_stress=zero, command=one, network=one,
        contagion_mobilization=zero, contagion_fatigue=zero,
        max_fixed_point_iterations=1, tolerance=1.0e9,
        common_noise=zero, jump_signal=zero,
    )
    diagnostics = model.fp_solver.diagnostics()
    return {
        "density_shape": list(model.density.shape),
        "engagement_mass_shape": list(model.engagement_mass.shape),
        "maximum_mass_residual": float(np.max(np.abs(
            model.density.sum(axis=-1) * model.dx - 1.0
        ))),
        "finite": bool(
            np.all(np.isfinite(model.density))
            and np.all(np.isfinite(model.engagement_mass))
        ),
        "host_to_device_transfers_inside_fp": diagnostics["host_to_device_transfers"],
        "device_to_host_transfers_inside_fp": diagnostics["device_to_host_transfers"],
        "monthly_deferred_device_reductions": diagnostics["deferred_device_reductions"],
        "tridiagonal_solver": diagnostics["tridiagonal_solver"],
    }


def mfg_cpu_gpu_regression() -> dict:
    paths = 2
    initial = np.full((paths, 4), 0.60)
    options = dict(
        geometry_enabled=True, advanced_graphon=True, full_wfr_jko=True,
        riemannian_anderson_enabled=True, riemannian_trust_region_enabled=True,
        riemannian_fixed_point_iterations=48,
    )
    cpu = ContinuousMultiPopulationMFG(paths, 84_079, initial, fp_backend="numpy_fvm", **options)
    gpu = ContinuousMultiPopulationMFG(paths, 84_079, initial, fp_backend="torch_fvm", **options)
    zero, one = np.zeros((paths, 4)), np.ones((paths, 4))
    inputs = dict(
        progress=zero, damage=zero, shortage=zero, inflation=zero,
        financial_stress=zero, command=one, network=one,
        contagion_mobilization=zero, contagion_fatigue=zero,
        max_fixed_point_iterations=1, tolerance=1.0e9,
        common_noise=zero, jump_signal=zero,
    )
    cpu_factors, gpu_factors = cpu.update(**inputs), gpu.update(**inputs)
    fields = (
        "support", "mobilization_preference", "fatigue_preference",
        "threshold_shift", "control_effort", "engagement_mass",
    )
    field_errors = {
        field: float(np.max(np.abs(
            getattr(cpu_factors, field) - getattr(gpu_factors, field)
        ))) for field in fields
    }
    return {
        "density_max_abs_error": float(np.max(np.abs(cpu.density - gpu.density))),
        "engagement_mass_max_abs_error": float(np.max(np.abs(
            cpu.engagement_mass - gpu.engagement_mass
        ))),
        "factor_max_abs_errors": field_errors,
        "fixed_point_residual_abs_error": abs(
            cpu_factors.fixed_point_residual - gpu_factors.fixed_point_residual
        ),
        "bellman_residual_abs_error": abs(
            cpu_factors.bellman_residual - gpu_factors.bellman_residual
        ),
    }


def persistent_worker_diagnostic() -> dict:
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as pool:
        first = list(pool.map(_pid, range(8)))
        second = list(pool.map(_pid, range(8, 16)))
    unique = sorted(set(first + second))
    return {
        "submitted_tasks": 16, "configured_workers": 2,
        "unique_worker_pids": unique,
        "workers_reused_across_batches": bool(set(first) & set(second)),
    }


def main() -> None:
    if not torch.cuda.is_available() or _torch_thomas_compiled is None:
        raise RuntimeError("CUDA and Triton-compiled Thomas are required")
    payload = {
        "scope": "DIAGNOSTIC_ONLY; no outcome probabilities or substantive results",
        "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
        "triton_version": triton.__version__,
        "tridiagonal": tridiagonal_diagnostic(),
        "variant_batch": variant_diagnostic(),
        "mfg_device_residency": mfg_residency_diagnostic(),
        "mfg_cpu_gpu_zero_shock_regression": mfg_cpu_gpu_regression(),
        "persistent_workers": persistent_worker_diagnostic(),
        "source_hashes": {
            relative: _sha256(ROOT / relative)
            for relative in (
                "work/fp_solvers_v52.py", "work/torch_mfg_acceleration.py",
                "work/continuous_multipop_mfg_v47.py", "work/variant_batching.py",
                "work/simulate_taiwan_fully_closed_v47.py",
                "work/simulate_taiwan_fully_closed_v84.py",
            )
        },
    }
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("TAIWAN_V84_ACCELERATION_DIAGNOSTIC: PASS")


if __name__ == "__main__":
    main()
