from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import math

import numpy as np


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
TORCHDIFFEQ_AVAILABLE = importlib.util.find_spec("torchdiffeq") is not None

if TORCH_AVAILABLE:
    import torch
else:  # pragma: no cover - minimal environments use the NumPy reference path
    torch = None


@dataclass
class FPStepDiagnostics:
    backend: str
    mass_residual: float
    minimum_density: float
    cfl_number: float
    substeps: int
    differentiable: bool


def _normalise_numpy(density: np.ndarray, dx: float) -> tuple[np.ndarray, float]:
    mass = np.sum(density, axis=-1, keepdims=True) * dx
    residual = float(np.max(np.abs(mass - 1.0)))
    return density / np.maximum(mass, 1.0e-15), residual


def _normalise_torch(density: "torch.Tensor", dx: float) -> tuple["torch.Tensor", float]:
    mass = density.sum(dim=-1, keepdim=True) * dx
    residual = float(torch.max(torch.abs(mass - 1.0)).detach().cpu())
    return density / torch.clamp(mass, min=1.0e-15), residual


def _torch_thomas(
    lower: "torch.Tensor", diagonal: "torch.Tensor", upper: "torch.Tensor",
    rhs: "torch.Tensor",
) -> "torch.Tensor":
    """Differentiable batched tridiagonal solve on CPU or CUDA."""
    grid = diagonal.shape[-1]
    c_values = []
    d_values = []
    denominator = torch.clamp(diagonal[..., 0], min=1.0e-12)
    c_values.append(upper[..., 0] / denominator)
    d_values.append(rhs[..., 0] / denominator)
    for index in range(1, grid):
        denominator = torch.clamp(
            diagonal[..., index] - lower[..., index - 1] * c_values[index - 1],
            min=1.0e-12,
        )
        if index < grid - 1:
            c_values.append(upper[..., index] / denominator)
        d_values.append(
            (rhs[..., index] - lower[..., index - 1] * d_values[index - 1])
            / denominator
        )
    solved = [None] * grid
    solved[-1] = d_values[-1]
    for index in range(grid - 2, -1, -1):
        solved[index] = d_values[index] - c_values[index] * solved[index + 1]
    return torch.stack(solved, dim=-1)


class FokkerPlanckSolver:
    """Method-of-lines FP solver with reflecting, zero-flux boundaries.

    The production backend uses conservative upwind finite volumes for
    advection and backward Euler for diffusion. ``torchdiffeq`` is deliberately
    optional and intended for differentiable calibration on non-stiff cases.
    """

    VALID_BACKENDS = {
        "auto", "legacy_explicit", "torch_fvm", "numpy_fvm", "torchdiffeq"
    }

    def __init__(
        self, dx: float, backend: str = "auto", device: str = "auto",
        dtype: str = "float64",
    ):
        if backend not in self.VALID_BACKENDS:
            raise ValueError(f"unsupported FP backend {backend!r}")
        self.dx = float(dx)
        if device == "auto":
            device = "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu"
        self.device = device
        self.dtype_name = dtype
        self.dtype = getattr(torch, dtype) if TORCH_AVAILABLE else None
        if backend == "auto":
            backend = "torch_fvm" if TORCH_AVAILABLE and torch.cuda.is_available() else "numpy_fvm"
        if backend.startswith("torch") and not TORCH_AVAILABLE:
            backend = "numpy_fvm"
        if backend == "torchdiffeq" and not TORCHDIFFEQ_AVAILABLE:
            raise RuntimeError("torchdiffeq backend requested but package is unavailable")
        self.backend = backend
        self.max_mass_residual = 0.0
        self.minimum_density = math.inf
        self.max_cfl_number = 0.0
        self.total_substeps = 0
        self.calls = 0

    def _substeps(self, drift, dt: float) -> tuple[int, float]:
        if TORCH_AVAILABLE and torch.is_tensor(drift):
            max_speed = float(torch.max(torch.abs(drift)).detach().cpu())
        else:
            max_speed = float(np.max(np.abs(drift)))
        cfl = max_speed * dt / self.dx
        return max(1, int(math.ceil(cfl / 0.80))), cfl

    def _numpy_step(self, density, drift, sigma, dt):
        substeps, cfl = self._substeps(drift, dt)
        h = dt / substeps
        state = np.asarray(density, dtype=float).copy()
        drift = np.broadcast_to(np.asarray(drift, dtype=float), state.shape)
        sigma = np.broadcast_to(np.asarray(sigma, dtype=float), state.shape)
        for _ in range(substeps):
            interface = 0.5 * (drift[..., 1:] + drift[..., :-1])
            flux = np.where(
                interface >= 0.0,
                interface * state[..., :-1],
                interface * state[..., 1:],
            )
            advected = state.copy()
            advected[..., 0] -= h * flux[..., 0] / self.dx
            advected[..., 1:-1] -= h * (flux[..., 1:] - flux[..., :-1]) / self.dx
            advected[..., -1] += h * flux[..., -1] / self.dx
            diffusivity = 0.5 * sigma * sigma
            interface_d = 0.5 * (diffusivity[..., 1:] + diffusivity[..., :-1])
            lower = -h * interface_d / self.dx**2
            upper = lower.copy()
            diagonal = np.ones_like(state)
            diagonal[..., 0] -= upper[..., 0]
            diagonal[..., -1] -= lower[..., -1]
            diagonal[..., 1:-1] -= lower[..., :-1] + upper[..., 1:]
            from numerical_acceleration import batched_thomas_numpy

            state = batched_thomas_numpy(lower, diagonal, upper, advected)
        minimum = float(np.min(state))
        state = np.maximum(state, 0.0)
        state, residual = _normalise_numpy(state, self.dx)
        return state, FPStepDiagnostics(
            "numpy_fvm", residual, minimum, cfl, substeps, False
        )

    def _legacy_step(self, density, drift, sigma, dt):
        state = np.asarray(density, dtype=float)
        drift = np.broadcast_to(np.asarray(drift, dtype=float), state.shape)
        sigma = np.broadcast_to(np.asarray(sigma, dtype=float), state.shape)
        interface = 0.5 * (drift[..., 1:] + drift[..., :-1])
        flux = np.where(
            interface >= 0.0, interface * state[..., :-1],
            interface * state[..., 1:]
        ) - 0.25 * (sigma[..., 1:] ** 2 + sigma[..., :-1] ** 2) * (
            state[..., 1:] - state[..., :-1]
        ) / self.dx
        evolved = state.copy()
        evolved[..., 0] -= dt * flux[..., 0] / self.dx
        evolved[..., 1:-1] -= dt * (flux[..., 1:] - flux[..., :-1]) / self.dx
        evolved[..., -1] += dt * flux[..., -1] / self.dx
        minimum = float(np.min(evolved))
        evolved = np.maximum(evolved, 0.0)
        evolved, residual = _normalise_numpy(evolved, self.dx)
        _, cfl = self._substeps(drift, dt)
        return evolved, FPStepDiagnostics(
            "legacy_explicit", residual, minimum, cfl, 1, False
        )

    def _torch_fvm_step(self, density, drift, sigma, dt):
        input_is_tensor = torch.is_tensor(density)
        state = density if input_is_tensor else torch.as_tensor(
            density, dtype=self.dtype, device=self.device
        )
        drift_t = drift if torch.is_tensor(drift) else torch.tensor(
            drift, dtype=state.dtype, device=state.device
        )
        sigma_t = sigma if torch.is_tensor(sigma) else torch.tensor(
            sigma, dtype=state.dtype, device=state.device
        )
        drift_t = torch.broadcast_to(drift_t, state.shape)
        sigma_t = torch.broadcast_to(sigma_t, state.shape)
        substeps, cfl = self._substeps(drift_t, dt)
        h = dt / substeps
        for _ in range(substeps):
            interface = 0.5 * (drift_t[..., 1:] + drift_t[..., :-1])
            flux = torch.where(
                interface >= 0.0,
                interface * state[..., :-1],
                interface * state[..., 1:],
            )
            advected = state.clone()
            advected[..., 0] = advected[..., 0] - h * flux[..., 0] / self.dx
            advected[..., 1:-1] = (
                advected[..., 1:-1]
                - h * (flux[..., 1:] - flux[..., :-1]) / self.dx
            )
            advected[..., -1] = advected[..., -1] + h * flux[..., -1] / self.dx
            diffusivity = 0.5 * sigma_t * sigma_t
            interface_d = 0.5 * (diffusivity[..., 1:] + diffusivity[..., :-1])
            lower = -h * interface_d / self.dx**2
            upper = lower.clone()
            diagonal = torch.ones_like(state)
            diagonal[..., 0] = diagonal[..., 0] - upper[..., 0]
            diagonal[..., -1] = diagonal[..., -1] - lower[..., -1]
            diagonal[..., 1:-1] = (
                diagonal[..., 1:-1] - lower[..., :-1] - upper[..., 1:]
            )
            state = _torch_thomas(lower, diagonal, upper, advected)
        minimum = float(torch.min(state).detach().cpu())
        state = torch.clamp(state, min=0.0)
        state, residual = _normalise_torch(state, self.dx)
        result = state if input_is_tensor else state.detach().cpu().numpy()
        return result, FPStepDiagnostics(
            "torch_fvm", residual, minimum, cfl, substeps, bool(state.requires_grad)
        )

    def _torchdiffeq_step(self, density, drift, sigma, dt):
        from torchdiffeq import odeint

        input_is_tensor = torch.is_tensor(density)
        state = density if input_is_tensor else torch.as_tensor(
            density, dtype=self.dtype, device=self.device
        )
        drift_t = drift if torch.is_tensor(drift) else torch.tensor(
            drift, dtype=state.dtype, device=state.device
        )
        sigma_t = sigma if torch.is_tensor(sigma) else torch.tensor(
            sigma, dtype=state.dtype, device=state.device
        )
        drift_t = torch.broadcast_to(drift_t, state.shape)
        sigma_t = torch.broadcast_to(sigma_t, state.shape)
        substeps, cfl = self._substeps(drift_t, dt)

        def rhs(_time, current):
            interface = 0.5 * (drift_t[..., 1:] + drift_t[..., :-1])
            adv_flux = torch.where(
                interface >= 0.0,
                interface * current[..., :-1],
                interface * current[..., 1:],
            )
            interface_d = 0.25 * (
                sigma_t[..., 1:] ** 2 + sigma_t[..., :-1] ** 2
            )
            diff_flux = -interface_d * (
                current[..., 1:] - current[..., :-1]
            ) / self.dx
            flux = adv_flux + diff_flux
            derivative = torch.zeros_like(current)
            derivative[..., 0] = -flux[..., 0] / self.dx
            derivative[..., 1:-1] = -(flux[..., 1:] - flux[..., :-1]) / self.dx
            derivative[..., -1] = flux[..., -1] / self.dx
            return derivative

        times = torch.tensor([0.0, dt], dtype=state.dtype, device=state.device)
        step_size = dt / max(substeps, 4)
        evolved = odeint(
            rhs, state, times, method="rk4", options={"step_size": step_size}
        )[-1]
        minimum = float(torch.min(evolved).detach().cpu())
        evolved = torch.clamp(evolved, min=0.0)
        evolved, residual = _normalise_torch(evolved, self.dx)
        result = evolved if input_is_tensor else evolved.detach().cpu().numpy()
        return result, FPStepDiagnostics(
            "torchdiffeq", residual, minimum, cfl, max(substeps, 4),
            bool(evolved.requires_grad),
        )

    def step(self, density, drift, sigma, dt: float):
        if self.backend == "legacy_explicit":
            result, diagnostic = self._legacy_step(density, drift, sigma, dt)
        elif self.backend == "numpy_fvm":
            result, diagnostic = self._numpy_step(density, drift, sigma, dt)
        elif self.backend == "torch_fvm":
            result, diagnostic = self._torch_fvm_step(density, drift, sigma, dt)
        else:
            result, diagnostic = self._torchdiffeq_step(density, drift, sigma, dt)
        self.calls += 1
        self.total_substeps += diagnostic.substeps
        self.max_mass_residual = max(self.max_mass_residual, diagnostic.mass_residual)
        self.minimum_density = min(self.minimum_density, diagnostic.minimum_density)
        self.max_cfl_number = max(self.max_cfl_number, diagnostic.cfl_number)
        return result, diagnostic

    def diagnostics(self) -> dict:
        return {
            "backend": self.backend,
            "device": self.device,
            "torch_available": TORCH_AVAILABLE,
            "torchdiffeq_available": TORCHDIFFEQ_AVAILABLE,
            "max_mass_residual_before_normalisation": self.max_mass_residual,
            "minimum_density_before_projection": self.minimum_density,
            "max_advective_cfl": self.max_cfl_number,
            "calls": self.calls,
            "total_substeps": self.total_substeps,
        }
