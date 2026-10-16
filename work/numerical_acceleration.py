"""Shared numerical kernels for pathwise accelerated solvers.

The module has no mandatory dependency beyond NumPy.  Numba is used when it
is installed and selected, while the NumPy implementation remains the
reference path for deterministic equivalence checks.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Callable

import numpy as np


try:
    from numba import njit, prange

    NUMBA_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised on minimal installations
    NUMBA_AVAILABLE = False
    njit = None
    prange = range


VALID_BACKENDS = ("auto", "numpy", "numba")


def resolve_backend(requested: str | None = None) -> str:
    name = (requested or os.environ.get("TAIWAN_NUMERICAL_BACKEND", "auto")).lower()
    if name not in VALID_BACKENDS:
        raise ValueError(f"unknown numerical backend {name!r}; expected {VALID_BACKENDS}")
    if name == "auto":
        return "numba" if NUMBA_AVAILABLE else "numpy"
    if name == "numba" and not NUMBA_AVAILABLE:
        return "numpy"
    return name


def _validate_tridiagonal(
    lower: np.ndarray, diagonal: np.ndarray, upper: np.ndarray, rhs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[int, ...]]:
    diagonal = np.ascontiguousarray(diagonal, dtype=np.float64)
    rhs = np.ascontiguousarray(rhs, dtype=np.float64)
    lower = np.ascontiguousarray(lower, dtype=np.float64)
    upper = np.ascontiguousarray(upper, dtype=np.float64)
    if diagonal.shape != rhs.shape:
        raise ValueError("diagonal and rhs must have the same shape")
    if diagonal.ndim < 2:
        raise ValueError("batched systems require at least one batch axis")
    expected_band = diagonal.shape[:-1] + (diagonal.shape[-1] - 1,)
    if lower.shape != expected_band or upper.shape != expected_band:
        raise ValueError("lower and upper diagonals have incompatible shapes")
    batch_shape = diagonal.shape[:-1]
    return lower, diagonal, upper, rhs, batch_shape


def batched_thomas_numpy(
    lower: np.ndarray, diagonal: np.ndarray, upper: np.ndarray, rhs: np.ndarray,
    pivot_floor: float = 1.0e-12,
) -> np.ndarray:
    """Reference Thomas solver, vectorized over every leading batch axis."""
    lower, diagonal, upper, rhs, _ = _validate_tridiagonal(
        lower, diagonal, upper, rhs
    )
    grid = diagonal.shape[-1]
    c_prime = np.empty_like(upper)
    d_prime = np.empty_like(rhs)
    denominator = np.maximum(diagonal[..., 0], pivot_floor)
    c_prime[..., 0] = upper[..., 0] / denominator
    d_prime[..., 0] = rhs[..., 0] / denominator
    for index in range(1, grid):
        denominator = np.maximum(
            diagonal[..., index]
            - lower[..., index - 1] * c_prime[..., index - 1],
            pivot_floor,
        )
        if index < grid - 1:
            c_prime[..., index] = upper[..., index] / denominator
        d_prime[..., index] = (
            rhs[..., index] - lower[..., index - 1] * d_prime[..., index - 1]
        ) / denominator
    solved = np.empty_like(rhs)
    solved[..., -1] = d_prime[..., -1]
    for index in range(grid - 2, -1, -1):
        solved[..., index] = (
            d_prime[..., index] - c_prime[..., index] * solved[..., index + 1]
        )
    return solved


if NUMBA_AVAILABLE:

    @njit(cache=True, parallel=True)
    def _batched_thomas_numba_flat(lower, diagonal, upper, rhs, pivot_floor):
        systems, grid = diagonal.shape
        solved = np.empty_like(rhs)
        for system in prange(systems):
            c_prime = np.empty(grid - 1, dtype=np.float64)
            d_prime = np.empty(grid, dtype=np.float64)
            denominator = max(diagonal[system, 0], pivot_floor)
            c_prime[0] = upper[system, 0] / denominator
            d_prime[0] = rhs[system, 0] / denominator
            for index in range(1, grid):
                denominator = max(
                    diagonal[system, index]
                    - lower[system, index - 1] * c_prime[index - 1],
                    pivot_floor,
                )
                if index < grid - 1:
                    c_prime[index] = upper[system, index] / denominator
                d_prime[index] = (
                    rhs[system, index]
                    - lower[system, index - 1] * d_prime[index - 1]
                ) / denominator
            solved[system, grid - 1] = d_prime[grid - 1]
            for index in range(grid - 2, -1, -1):
                solved[system, index] = (
                    d_prime[index]
                    - c_prime[index] * solved[system, index + 1]
                )
        return solved


    @njit(cache=True, parallel=True)
    def _batched_hjb_policy_numba_flat(
        reward, base_drift, initial_value, control_cost, sigma2,
        discount, dx, max_iterations, tolerance, pivot_floor,
    ):
        systems, grid = initial_value.shape
        values = initial_value.copy()
        controls = np.zeros_like(values)
        iterations = np.zeros(systems, dtype=np.int32)
        for system in prange(systems):
            value_x = np.empty(grid, dtype=np.float64)
            left = np.empty(grid, dtype=np.float64)
            right = np.empty(grid, dtype=np.float64)
            diagonal = np.empty(grid, dtype=np.float64)
            lower = np.empty(grid - 1, dtype=np.float64)
            upper = np.empty(grid - 1, dtype=np.float64)
            rhs = np.empty(grid, dtype=np.float64)
            c_prime = np.empty(grid - 1, dtype=np.float64)
            d_prime = np.empty(grid, dtype=np.float64)
            solved = np.empty(grid, dtype=np.float64)
            for iteration in range(max_iterations):
                value_x[0] = (values[system, 1] - values[system, 0]) / dx
                for index in range(1, grid - 1):
                    value_x[index] = (
                        values[system, index + 1] - values[system, index - 1]
                    ) / (2.0 * dx)
                value_x[grid - 1] = (
                    values[system, grid - 1] - values[system, grid - 2]
                ) / dx
                for index in range(grid):
                    proposed = value_x[index] / control_cost[system, index]
                    proposed = min(max(proposed, -0.55), 0.55)
                    controls[system, index] = (
                        0.35 * controls[system, index] + 0.65 * proposed
                    )
                    drift = base_drift[system, index] + controls[system, index]
                    drift = min(max(drift, -0.75), 0.75)
                    left[index] = max(-drift, 0.0) / dx + (
                        0.5 * sigma2[system, index] / (dx * dx)
                    )
                    right[index] = max(drift, 0.0) / dx + (
                        0.5 * sigma2[system, index] / (dx * dx)
                    )
                left[0] = 0.0
                right[grid - 1] = 0.0
                for index in range(grid):
                    diagonal[index] = discount + left[index] + right[index]
                    rhs[index] = reward[system, index] - 0.5 * (
                        control_cost[system, index]
                        * controls[system, index] * controls[system, index]
                    )
                    if index < grid - 1:
                        upper[index] = -right[index]
                        lower[index] = -left[index + 1]
                denominator = max(diagonal[0], pivot_floor)
                c_prime[0] = upper[0] / denominator
                d_prime[0] = rhs[0] / denominator
                for index in range(1, grid):
                    denominator = max(
                        diagonal[index] - lower[index - 1] * c_prime[index - 1],
                        pivot_floor,
                    )
                    if index < grid - 1:
                        c_prime[index] = upper[index] / denominator
                    d_prime[index] = (
                        rhs[index] - lower[index - 1] * d_prime[index - 1]
                    ) / denominator
                solved[grid - 1] = d_prime[grid - 1]
                for index in range(grid - 2, -1, -1):
                    solved[index] = d_prime[index] - c_prime[index] * solved[index + 1]
                residual = 0.0
                for index in range(grid):
                    residual = max(residual, abs(solved[index] - values[system, index]))
                    values[system, index] = (
                        0.20 * values[system, index] + 0.80 * solved[index]
                    )
                iterations[system] = iteration + 1
                if residual < tolerance:
                    break
            value_x[0] = (values[system, 1] - values[system, 0]) / dx
            for index in range(1, grid - 1):
                value_x[index] = (
                    values[system, index + 1] - values[system, index - 1]
                ) / (2.0 * dx)
            value_x[grid - 1] = (
                values[system, grid - 1] - values[system, grid - 2]
            ) / dx
            for index in range(grid):
                controls[system, index] = min(max(
                    value_x[index] / control_cost[system, index], -0.55
                ), 0.55)
        return values, controls, iterations


def batched_thomas(
    lower: np.ndarray, diagonal: np.ndarray, upper: np.ndarray, rhs: np.ndarray,
    *, backend: str | None = None, pivot_floor: float = 1.0e-12,
) -> np.ndarray:
    """Solve independent tridiagonal systems with the selected backend."""
    lower, diagonal, upper, rhs, batch_shape = _validate_tridiagonal(
        lower, diagonal, upper, rhs
    )
    selected = resolve_backend(backend)
    if selected == "numba":
        grid = diagonal.shape[-1]
        solved = _batched_thomas_numba_flat(
            lower.reshape(-1, grid - 1),
            diagonal.reshape(-1, grid),
            upper.reshape(-1, grid - 1),
            rhs.reshape(-1, grid),
            pivot_floor,
        )
        return solved.reshape(batch_shape + (grid,))
    return batched_thomas_numpy(lower, diagonal, upper, rhs, pivot_floor)


def batched_hjb_policy(
    reward: np.ndarray, base_drift: np.ndarray, initial_value: np.ndarray,
    control_cost: np.ndarray, sigma2: np.ndarray, *, discount: float,
    dx: float, max_iterations: int, tolerance: float,
    backend: str | None = None, pivot_floor: float = 1.0e-10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Compile the complete independent-system HJB policy loop when possible.

    Returning ``None`` asks the caller to use its NumPy reference path.
    """
    selected = resolve_backend(backend)
    if selected != "numba":
        return None
    arrays = [
        np.ascontiguousarray(item, dtype=np.float64)
        for item in (reward, base_drift, initial_value, control_cost, sigma2)
    ]
    if any(item.shape != arrays[0].shape for item in arrays[1:]):
        raise ValueError("HJB policy arrays must have identical shapes")
    shape = arrays[0].shape
    grid = shape[-1]
    flattened = [item.reshape(-1, grid) for item in arrays]
    value, control, iterations = _batched_hjb_policy_numba_flat(
        *flattened, float(discount), float(dx), int(max_iterations),
        float(tolerance), float(pivot_floor),
    )
    return value.reshape(shape), control.reshape(shape), iterations.reshape(shape[:-1])


@dataclass
class AndersonDiagnostics:
    attempted: int = 0
    accepted_paths: int = 0
    rejected_paths: int = 0
    resets: int = 0


class PathwiseAnderson:
    """Small-memory type-II Anderson acceleration, independent by path.

    Every path obtains its own least-squares coefficients.  This is essential:
    fitting one coefficient vector across Monte Carlo paths would create an
    artificial cross-path feedback that is not part of the model.
    """

    def __init__(self, depth: int = 3, damping: float = 0.72,
                 regularization: float = 1.0e-8, trust_ratio: float = 3.0):
        self.depth = max(int(depth), 0)
        self.damping = float(damping)
        self.regularization = float(regularization)
        self.trust_ratio = float(trust_ratio)
        self._x: list[np.ndarray] = []
        self._f: list[np.ndarray] = []
        self.diagnostics = AndersonDiagnostics()

    def reset(self) -> None:
        self._x.clear()
        self._f.clear()
        self.diagnostics.resets += 1

    def step(self, x: np.ndarray, mapped: np.ndarray) -> np.ndarray:
        if x.shape != mapped.shape or x.ndim < 2:
            raise ValueError("Anderson inputs must share [path, ...] shape")
        residual = mapped - x
        picard = x + self.damping * residual
        if self.depth == 0:
            return picard
        self._x.append(x.copy())
        self._f.append(residual.copy())
        keep = self.depth + 1
        if len(self._x) > keep:
            self._x.pop(0)
            self._f.pop(0)
        if len(self._x) < 2:
            return picard

        self.diagnostics.attempted += 1
        flat_x = [item.reshape(item.shape[0], -1) for item in self._x]
        flat_f = [item.reshape(item.shape[0], -1) for item in self._f]
        delta_x = np.stack(
            [flat_x[i + 1] - flat_x[i] for i in range(len(flat_x) - 1)], axis=2
        )
        delta_f = np.stack(
            [flat_f[i + 1] - flat_f[i] for i in range(len(flat_f) - 1)], axis=2
        )
        current_f = flat_f[-1]
        gram = np.einsum("pnk,pnl->pkl", delta_f, delta_f)
        rhs = np.einsum("pnk,pn->pk", delta_f, current_f)
        eye = np.eye(gram.shape[-1])[None, :, :]
        gram = gram + self.regularization * eye
        try:
            coefficients = np.linalg.solve(gram, rhs[..., None])[..., 0]
        except np.linalg.LinAlgError:
            self.diagnostics.rejected_paths += x.shape[0]
            return picard
        correction = np.einsum(
            "pnk,pk->pn", delta_x + self.damping * delta_f, coefficients
        )
        accelerated = picard.reshape(x.shape[0], -1) - correction
        reference_step = np.linalg.norm(
            picard.reshape(x.shape[0], -1) - flat_x[-1], axis=1
        )
        accelerated_step = np.linalg.norm(accelerated - flat_x[-1], axis=1)
        accept = (
            np.all(np.isfinite(accelerated), axis=1)
            & (accelerated_step <= self.trust_ratio * np.maximum(reference_step, 1.0e-12))
        )
        self.diagnostics.accepted_paths += int(np.count_nonzero(accept))
        self.diagnostics.rejected_paths += int(np.count_nonzero(~accept))
        result = picard.reshape(x.shape[0], -1)
        result[accept] = accelerated[accept]
        return result.reshape(x.shape)


def rk4_step(
    rhs: Callable[..., np.ndarray], t: float, state: np.ndarray, dt: float,
    *args,
) -> np.ndarray:
    """Vectorized fixed-step RK4 for arrays whose leading axis is path."""
    k1 = rhs(t, state, *args)
    k2 = rhs(t + 0.5 * dt, state + 0.5 * dt * k1, *args)
    k3 = rhs(t + 0.5 * dt, state + 0.5 * dt * k2, *args)
    k4 = rhs(t + dt, state + dt * k3, *args)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def euler_maruyama_step(
    state: np.ndarray, drift: np.ndarray, diffusion: np.ndarray,
    standard_normal: np.ndarray, dt: float,
) -> np.ndarray:
    """One path-batched Euler--Maruyama step with caller-owned randomness."""
    if not (state.shape == drift.shape == diffusion.shape == standard_normal.shape):
        raise ValueError("Euler-Maruyama arrays must have identical shapes")
    return state + drift * dt + diffusion * np.sqrt(dt) * standard_normal


def backend_diagnostics(requested: str | None = None) -> dict:
    return {
        "requested": requested or os.environ.get("TAIWAN_NUMERICAL_BACKEND", "auto"),
        "selected": resolve_backend(requested),
        "numba_available": NUMBA_AVAILABLE,
    }
