"""Device-resident nonlinear and PDE solver infrastructure.

This module contains no Taiwan-scenario logic.  It provides deterministic,
diagnostic-safe numerical kernels shared by CGE, DSGE, and MFG components.
JAX is the primary differentiable backend; PyTorch is retained as an
independent CUDA residency and numerical cross-check backend.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Callable

import numpy as np

try:
    os.environ.setdefault("JAX_ENABLE_X64", "true")
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    JAX_AVAILABLE = True
except ImportError:  # pragma: no cover
    jax = None
    jnp = None
    JAX_AVAILABLE = False

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    torch = None
    TORCH_AVAILABLE = False


Array = object


@dataclass(frozen=True)
class KrylovDiagnostics:
    iterations: int
    residual: float
    converged: bool


@dataclass(frozen=True)
class NewtonDiagnostics:
    iterations: int
    residual: float
    converged: bool
    krylov_iterations: int
    line_search_reductions: int


class JaxResidentRuntime:
    """Own persistent JAX buffers and compiled kernels on one device.

    Arrays registered here remain device arrays across monthly calls.  The
    caller explicitly requests host copies only for reporting or checkpointing.
    """

    def __init__(self, platform: str | None = None):
        if not JAX_AVAILABLE:
            raise RuntimeError("JAX is not installed")
        candidates = jax.devices(platform) if platform else jax.devices()
        if not candidates:
            raise RuntimeError(f"no JAX device for platform={platform!r}")
        self.device = candidates[0]
        self.buffers: dict[str, Array] = {}
        self.compiled: dict[str, Callable] = {}
        self.host_transfer_count = 0

    @property
    def platform(self) -> str:
        return self.device.platform

    def put(self, name: str, value, dtype=jnp.float64):
        array = jax.device_put(jnp.asarray(value, dtype=dtype), self.device)
        self.buffers[name] = array
        return array

    def put_static(self, name: str, value, dtype=jnp.float64):
        """Upload an immutable coefficient once and reuse its device buffer."""
        if name in self.buffers:
            return self.buffers[name]
        return self.put(name, value, dtype=dtype)

    def get(self, name: str):
        return self.buffers[name]

    def compile(self, name: str, function: Callable, *, static_argnames=()):
        compiled = jax.jit(function, static_argnames=static_argnames, device=self.device)
        self.compiled[name] = compiled
        return compiled

    def to_host(self, value) -> np.ndarray:
        self.host_transfer_count += 1
        return np.asarray(jax.device_get(value))

    def diagnostics(self) -> dict:
        return {
            "backend": "jax",
            "platform": self.device.platform,
            "device": str(self.device),
            "resident_buffers": len(self.buffers),
            "compiled_kernels": len(self.compiled),
            "host_transfer_count": self.host_transfer_count,
            "x64_enabled": bool(jax.config.x64_enabled),
        }


class TorchResidentRuntime:
    """Persistent PyTorch CUDA buffers used for independent GPU checks."""

    def __init__(self, device: str = "cuda:0"):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is not installed")
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA PyTorch runtime is unavailable")
        self.device = torch.device(device)
        self.buffers: dict[str, torch.Tensor] = {}
        self.host_transfer_count = 0

    def put(self, name: str, value, dtype=torch.float64):
        tensor = torch.as_tensor(value, dtype=dtype, device=self.device).contiguous()
        self.buffers[name] = tensor
        return tensor

    def get(self, name: str):
        return self.buffers[name]

    def to_host(self, value) -> np.ndarray:
        self.host_transfer_count += 1
        return value.detach().cpu().numpy()

    def diagnostics(self) -> dict:
        allocated = 0
        reserved = 0
        if self.device.type == "cuda":
            allocated = int(torch.cuda.memory_allocated(self.device))
            reserved = int(torch.cuda.memory_reserved(self.device))
        return {
            "backend": "torch",
            "platform": self.device.type,
            "device": str(self.device),
            "device_name": (
                torch.cuda.get_device_name(self.device)
                if self.device.type == "cuda" else "CPU"
            ),
            "resident_buffers": len(self.buffers),
            "host_transfer_count": self.host_transfer_count,
            "memory_allocated_bytes": allocated,
            "memory_reserved_bytes": reserved,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
        }


def _path_dot(left, right):
    return jnp.sum(left * right, axis=-1, keepdims=True)


def batched_gmres(
    matvec: Callable[[Array], Array], rhs: Array, *, x0: Array | None = None,
    preconditioner: Callable[[Array], Array] | None = None,
    max_iterations: int = 40, tolerance: float = 1.0e-8,
) -> tuple[Array, KrylovDiagnostics]:
    """Independent-path GMRES with a shared iteration cap.

    ``rhs`` has shape ``[path, state]``.  Every scalar product is pathwise, so
    Monte Carlo paths never enter one another's Krylov subspace.
    """
    if not JAX_AVAILABLE:
        raise RuntimeError("batched_gmres requires JAX")
    rhs = jnp.asarray(rhs, dtype=jnp.float64)
    x = jnp.zeros_like(rhs) if x0 is None else jnp.asarray(x0, dtype=rhs.dtype)
    apply_m = (lambda value: value) if preconditioner is None else preconditioner
    residual0 = apply_m(rhs - matvec(x))
    beta = jnp.sqrt(jnp.maximum(_path_dot(residual0, residual0), 1.0e-30))
    basis = [residual0 / beta]
    hessenberg = jnp.zeros((rhs.shape[0], max_iterations + 1, max_iterations), dtype=rhs.dtype)
    iterations = 0
    solution = x
    final_norm = jnp.max(beta)
    for column in range(max_iterations):
        vector = apply_m(matvec(basis[column]))
        for row in range(column + 1):
            coefficient = _path_dot(basis[row], vector)
            hessenberg = hessenberg.at[:, row, column].set(coefficient[:, 0])
            vector = vector - coefficient * basis[row]
        next_norm = jnp.sqrt(jnp.maximum(_path_dot(vector, vector), 1.0e-30))
        hessenberg = hessenberg.at[:, column + 1, column].set(next_norm[:, 0])
        basis.append(vector / next_norm)

        h = hessenberg[:, : column + 2, : column + 1]
        target = jnp.zeros((rhs.shape[0], column + 2), dtype=rhs.dtype)
        target = target.at[:, 0].set(beta[:, 0])
        coefficients = jax.vmap(
            lambda matrix, vector: jnp.linalg.lstsq(
                matrix, vector, rcond=None
            )[0]
        )(h, target)
        stacked = jnp.stack(basis[: column + 1], axis=1)
        candidate = x + jnp.einsum("pkn,pk->pn", stacked, coefficients)
        candidate_residual = rhs - matvec(candidate)
        final_norm = jnp.max(jnp.linalg.norm(candidate_residual, axis=1))
        solution = candidate
        iterations = column + 1
        if float(final_norm) < tolerance:
            break
    residual_value = float(final_norm)
    return solution, KrylovDiagnostics(
        iterations=iterations,
        residual=residual_value,
        converged=residual_value < tolerance,
    )


def newton_krylov(
    residual_function: Callable[[Array], Array], initial: Array, *,
    preconditioner: Callable[[Array], Array] | None = None,
    max_newton_iterations: int = 12, max_krylov_iterations: int = 40,
    tolerance: float = 1.0e-8, krylov_tolerance: float = 1.0e-7,
) -> tuple[Array, NewtonDiagnostics]:
    """Matrix-free, pathwise Newton--Krylov using exact JAX JVPs."""
    if not JAX_AVAILABLE:
        raise RuntimeError("newton_krylov requires JAX")
    state = jnp.asarray(initial, dtype=jnp.float64)
    total_krylov = 0
    reductions = 0
    residual_norm = np.inf
    for iteration in range(max_newton_iterations):
        residual = residual_function(state)
        path_norm = jnp.linalg.norm(residual, axis=1)
        residual_norm = float(jnp.max(path_norm))
        if residual_norm < tolerance:
            return state, NewtonDiagnostics(
                iteration, residual_norm, True, total_krylov, reductions
            )

        def matvec(direction):
            return jax.jvp(residual_function, (state,), (direction,))[1]

        direction, krylov = batched_gmres(
            matvec, -residual, preconditioner=preconditioner,
            max_iterations=max_krylov_iterations, tolerance=krylov_tolerance,
        )
        total_krylov += krylov.iterations
        step = jnp.ones((state.shape[0], 1), dtype=state.dtype)
        accepted = jnp.zeros((state.shape[0], 1), dtype=bool)
        candidate = state
        base_norm = path_norm[:, None]
        for _ in range(10):
            trial = state + step * direction
            trial_norm = jnp.linalg.norm(residual_function(trial), axis=1, keepdims=True)
            sufficient = trial_norm <= (1.0 - 1.0e-4 * step) * base_norm
            newly = sufficient & ~accepted
            candidate = jnp.where(newly, trial, candidate)
            accepted = accepted | sufficient
            step = jnp.where(accepted, step, 0.5 * step)
            reductions += int(np.count_nonzero(np.asarray(~accepted)))
            if bool(jnp.all(accepted)):
                break
        fallback = state + step * direction
        state = jnp.where(accepted, candidate, fallback)
    residual_norm = float(jnp.max(jnp.linalg.norm(residual_function(state), axis=1)))
    return state, NewtonDiagnostics(
        max_newton_iterations, residual_norm, residual_norm < tolerance,
        total_krylov, reductions,
    )


class GeometricMultigrid1D:
    """Batched geometric V-cycle for positive 1-D diffusion operators."""

    def __init__(self, diagonal, off_diagonal, *, min_coarse: int = 3,
                 pre_smooth: int = 2, post_smooth: int = 2):
        if not JAX_AVAILABLE:
            raise RuntimeError("GeometricMultigrid1D requires JAX")
        diagonal = jnp.asarray(diagonal, dtype=jnp.float64)
        off_diagonal = jnp.asarray(off_diagonal, dtype=jnp.float64)
        if diagonal.ndim != 2 or off_diagonal.shape != (diagonal.shape[0], diagonal.shape[1] - 1):
            raise ValueError("expected diagonal [path,n] and off_diagonal [path,n-1]")
        self.diagonal = diagonal
        self.off_diagonal = off_diagonal
        self.min_coarse = max(int(min_coarse), 2)
        self.pre_smooth = int(pre_smooth)
        self.post_smooth = int(post_smooth)

    @staticmethod
    def _apply(diagonal, off, value):
        result = diagonal * value
        result = result.at[:, 1:].add(off * value[:, :-1])
        result = result.at[:, :-1].add(off * value[:, 1:])
        return result

    @staticmethod
    def _jacobi(diagonal, off, rhs, value, iterations):
        for _ in range(iterations):
            residual = rhs - GeometricMultigrid1D._apply(diagonal, off, value)
            value = value + 0.72 * residual / diagonal
        return value

    @staticmethod
    def _restrict(value):
        n = value.shape[1]
        coarse_n = (n + 1) // 2
        padded = jnp.pad(value, ((0, 0), (1, 1)), mode="edge")
        indices = 2 * jnp.arange(coarse_n) + 1
        return (
            0.25 * padded[:, indices - 1]
            + 0.50 * padded[:, indices]
            + 0.25 * padded[:, indices + 1]
        )

    @staticmethod
    def _prolong(value, target_n):
        coarse_n = value.shape[1]
        fine = jnp.zeros((value.shape[0], target_n), dtype=value.dtype)
        even_count = min(coarse_n, (target_n + 1) // 2)
        fine = fine.at[:, 0 : 2 * even_count : 2].set(value[:, :even_count])
        odd_count = target_n // 2
        if odd_count:
            right = value[:, 1 : odd_count + 1]
            left = value[:, :odd_count]
            fine = fine.at[:, 1 : 2 * odd_count : 2].set(0.5 * (left + right))
        return fine

    def _v_cycle(self, diagonal, off, rhs, value):
        n = diagonal.shape[1]
        if n <= self.min_coarse:
            matrix = jnp.zeros((diagonal.shape[0], n, n), dtype=diagonal.dtype)
            indices = jnp.arange(n)
            matrix = matrix.at[:, indices, indices].set(diagonal)
            if n > 1:
                edges = jnp.arange(n - 1)
                matrix = matrix.at[:, edges + 1, edges].set(off)
                matrix = matrix.at[:, edges, edges + 1].set(off)
            return jnp.linalg.solve(matrix, rhs[..., None])[..., 0]
        value = self._jacobi(diagonal, off, rhs, value, self.pre_smooth)
        residual = rhs - self._apply(diagonal, off, value)
        coarse_rhs = self._restrict(residual)
        coarse_diag = self._restrict(diagonal)
        coarse_off = -0.25 * jnp.sqrt(
            jnp.maximum(coarse_diag[:, :-1] * coarse_diag[:, 1:], 1.0e-16)
        )
        correction = self._v_cycle(
            coarse_diag, coarse_off, coarse_rhs, jnp.zeros_like(coarse_rhs)
        )
        value = value + self._prolong(correction, n)
        return self._jacobi(diagonal, off, rhs, value, self.post_smooth)

    def apply(self, rhs):
        rhs = jnp.asarray(rhs, dtype=jnp.float64)
        return self._v_cycle(
            self.diagonal, self.off_diagonal, rhs, jnp.zeros_like(rhs)
        )

    def residual(self, solution, rhs) -> float:
        residual = rhs - self._apply(self.diagonal, self.off_diagonal, solution)
        return float(jnp.max(jnp.linalg.norm(residual, axis=1)))


def infrastructure_diagnostics() -> dict:
    diagnostics = {
        "jax_available": JAX_AVAILABLE,
        "torch_available": TORCH_AVAILABLE,
    }
    if JAX_AVAILABLE:
        diagnostics["jax_devices"] = [str(device) for device in jax.devices()]
        diagnostics["jax_platforms"] = [device.platform for device in jax.devices()]
    if TORCH_AVAILABLE:
        diagnostics.update({
            "torch_version": torch.__version__,
            "torch_cuda_build": torch.version.cuda,
            "torch_cuda_available": bool(torch.cuda.is_available()),
            "torch_cuda_devices": (
                [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
                if torch.cuda.is_available() else []
            ),
        })
    return diagnostics
