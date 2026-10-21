from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import math

import numpy as np


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
TORCHDIFFEQ_AVAILABLE = importlib.util.find_spec("torchdiffeq") is not None
TRITON_AVAILABLE = importlib.util.find_spec("triton") is not None

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


@dataclass
class WFRStepDiagnostics:
    backend: str
    conditional_mass_residual: float
    stock_accounting_residual: float
    minimum_measure: float
    cfl_number: float
    substeps: int
    action: float


@dataclass
class WFRDeviceStepDiagnostics:
    reductions: "torch.Tensor"
    cfl_number: float
    substeps: int


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
    solved = [d_values[-1]]
    for index in range(grid - 2, -1, -1):
        solved.append(d_values[index] - c_values[index] * solved[-1])
    return torch.stack(solved[::-1], dim=-1)


def _signed_pivot_torch(value: "torch.Tensor", floor: float) -> "torch.Tensor":
    sign = torch.where(value < 0.0, -torch.ones_like(value), torch.ones_like(value))
    return sign * torch.clamp(torch.abs(value), min=floor)


def _torch_pcr(
    lower: "torch.Tensor", diagonal: "torch.Tensor", upper: "torch.Tensor",
    rhs: "torch.Tensor", pivot_floor: float = 1.0e-12,
) -> "torch.Tensor":
    """Batched parallel cyclic reduction for arbitrary grid lengths."""
    grid = diagonal.shape[-1]
    a = torch.zeros_like(diagonal)
    c = torch.zeros_like(diagonal)
    a[..., 1:] = lower
    c[..., :-1] = upper
    b = diagonal.clone()
    d = rhs.clone()
    indices = torch.arange(grid, device=diagonal.device)
    stride = 1
    while stride < grid:
        left_valid = (indices >= stride).view(*([1] * (b.ndim - 1)), grid)
        right_valid = (indices + stride < grid).view(*([1] * (b.ndim - 1)), grid)
        left_a = torch.roll(a, stride, dims=-1)
        left_b = torch.roll(b, stride, dims=-1)
        left_c = torch.roll(c, stride, dims=-1)
        left_d = torch.roll(d, stride, dims=-1)
        right_a = torch.roll(a, -stride, dims=-1)
        right_b = torch.roll(b, -stride, dims=-1)
        right_c = torch.roll(c, -stride, dims=-1)
        right_d = torch.roll(d, -stride, dims=-1)
        alpha = torch.where(
            left_valid, -a / _signed_pivot_torch(left_b, pivot_floor),
            torch.zeros_like(a),
        )
        beta = torch.where(
            right_valid, -c / _signed_pivot_torch(right_b, pivot_floor),
            torch.zeros_like(c),
        )
        a, b, c, d = (
            torch.where(left_valid, alpha * left_a, torch.zeros_like(a)),
            b + alpha * left_c + beta * right_a,
            torch.where(right_valid, beta * right_c, torch.zeros_like(c)),
            d + alpha * left_d + beta * right_d,
        )
        stride *= 2
    return d / _signed_pivot_torch(b, pivot_floor)


if TORCH_AVAILABLE and TRITON_AVAILABLE and hasattr(torch, "compile"):
    _torch_thomas_compiled = torch.compile(
        _torch_thomas, fullgraph=True, dynamic=True
    )
else:  # pragma: no cover - exercised only on CPU/minimal installations
    _torch_thomas_compiled = None


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
        dtype: str = "float64", tridiagonal_solver: str = "auto",
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
        if tridiagonal_solver not in {"auto", "compiled_thomas", "thomas", "pcr"}:
            raise ValueError(
                "tridiagonal_solver must be auto, compiled_thomas, thomas or pcr"
            )
        if tridiagonal_solver == "auto":
            tridiagonal_solver = (
                "compiled_thomas"
                if str(device).startswith("cuda") and _torch_thomas_compiled is not None
                else "thomas"
            )
        if tridiagonal_solver == "compiled_thomas" and _torch_thomas_compiled is None:
            tridiagonal_solver = "thomas"
        self.tridiagonal_solver = tridiagonal_solver
        self.max_mass_residual = 0.0
        self.minimum_density = math.inf
        self.max_cfl_number = 0.0
        self.total_substeps = 0
        self.calls = 0
        self.physical_steps = 0
        self.host_to_device_transfers = 0
        self.device_to_host_transfers = 0
        self.max_wfr_conditional_mass_residual = 0.0
        self.max_wfr_stock_accounting_residual = 0.0
        self.minimum_wfr_measure = math.inf
        self.max_wfr_action = 0.0
        self.wfr_calls = 0
        self.deferred_device_reductions = 0
        self._device_diagnostic_window: list[torch.Tensor] = []
        self._device_cfl_window: list[float] = []
        self._device_substeps_window: list[int] = []

    def _solve_tridiagonal(self, lower, diagonal, upper, rhs):
        if self.tridiagonal_solver == "compiled_thomas":
            return _torch_thomas_compiled(lower, diagonal, upper, rhs)
        if self.tridiagonal_solver == "pcr":
            return _torch_pcr(lower, diagonal, upper, rhs)
        return _torch_thomas(lower, diagonal, upper, rhs)

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
            state = self._solve_tridiagonal(lower, diagonal, upper, advected)
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
        input_is_tensor = TORCH_AVAILABLE and torch.is_tensor(density)
        if self.backend == "legacy_explicit":
            result, diagnostic = self._legacy_step(density, drift, sigma, dt)
        elif self.backend == "numpy_fvm":
            result, diagnostic = self._numpy_step(density, drift, sigma, dt)
        elif self.backend == "torch_fvm":
            result, diagnostic = self._torch_fvm_step(density, drift, sigma, dt)
        else:
            result, diagnostic = self._torchdiffeq_step(density, drift, sigma, dt)
        self.calls += 1
        self.physical_steps += 1
        if self.backend.startswith("torch") and not input_is_tensor:
            self.host_to_device_transfers += 1
            self.device_to_host_transfers += 1
        self.total_substeps += diagnostic.substeps
        self.max_mass_residual = max(self.max_mass_residual, diagnostic.mass_residual)
        self.minimum_density = min(self.minimum_density, diagnostic.minimum_density)
        self.max_cfl_number = max(self.max_cfl_number, diagnostic.cfl_number)
        return result, diagnostic

    def step_repeated(self, density, drift, sigma, dt: float, repeats: int):
        """Advance repeated physical steps with a single full-array GPU transfer."""
        repeats = int(repeats)
        if repeats < 1:
            raise ValueError("repeats must be at least one")
        if repeats == 1:
            return self.step(density, drift, sigma, dt)

        input_is_tensor = TORCH_AVAILABLE and torch.is_tensor(density)
        if self.backend.startswith("torch"):
            state = density if input_is_tensor else torch.as_tensor(
                density, dtype=self.dtype, device=self.device
            )
            drift_state = drift if torch.is_tensor(drift) else torch.as_tensor(
                drift, dtype=state.dtype, device=state.device
            )
            sigma_state = sigma if torch.is_tensor(sigma) else torch.as_tensor(
                sigma, dtype=state.dtype, device=state.device
            )
            if not input_is_tensor:
                self.host_to_device_transfers += 1
            diagnostics = []
            for _ in range(repeats):
                if self.backend == "torch_fvm":
                    state, diagnostic = self._torch_fvm_step(
                        state, drift_state, sigma_state, dt
                    )
                else:
                    state, diagnostic = self._torchdiffeq_step(
                        state, drift_state, sigma_state, dt
                    )
                diagnostics.append(diagnostic)
            if input_is_tensor:
                result = state
            else:
                result = state.detach().cpu().numpy()
                self.device_to_host_transfers += 1
        else:
            result = density
            diagnostics = []
            for _ in range(repeats):
                if self.backend == "legacy_explicit":
                    result, diagnostic = self._legacy_step(result, drift, sigma, dt)
                else:
                    result, diagnostic = self._numpy_step(result, drift, sigma, dt)
                diagnostics.append(diagnostic)

        combined = FPStepDiagnostics(
            backend=diagnostics[-1].backend,
            mass_residual=max(item.mass_residual for item in diagnostics),
            minimum_density=min(item.minimum_density for item in diagnostics),
            cfl_number=max(item.cfl_number for item in diagnostics),
            substeps=sum(item.substeps for item in diagnostics),
            differentiable=all(item.differentiable for item in diagnostics),
        )
        self.calls += 1
        self.physical_steps += repeats
        self.total_substeps += combined.substeps
        self.max_mass_residual = max(
            self.max_mass_residual, combined.mass_residual
        )
        self.minimum_density = min(
            self.minimum_density, combined.minimum_density
        )
        self.max_cfl_number = max(
            self.max_cfl_number, combined.cfl_number
        )
        return result, combined

    def _numpy_unbalanced_step(
        self, measure, drift, sigma, reaction, dt: float,
    ):
        substeps, cfl = self._substeps(drift, dt)
        h = dt / substeps
        state = np.asarray(measure, dtype=float).copy()
        drift = np.broadcast_to(np.asarray(drift, dtype=float), state.shape)
        sigma = np.broadcast_to(np.asarray(sigma, dtype=float), state.shape)
        reaction = np.broadcast_to(
            np.clip(np.asarray(reaction, dtype=float), -0.80, 0.55), state.shape
        )
        action = 0.0
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
            diagonal = np.ones_like(state) - h * reaction
            diagonal[..., 0] -= upper[..., 0]
            diagonal[..., -1] -= lower[..., -1]
            diagonal[..., 1:-1] -= lower[..., :-1] + upper[..., 1:]
            from numerical_acceleration import batched_thomas_numpy

            state = batched_thomas_numpy(lower, diagonal, upper, advected)
            state = np.maximum(state, 0.0)
            action = max(action, float(np.max(
                np.sum(
                    (np.square(drift) + np.square(reaction)) * state,
                    axis=-1,
                ) * self.dx * h
            )))
        return state, substeps, cfl, action

    def _torch_unbalanced_step(
        self, measure, drift, sigma, reaction, dt: float,
    ):
        state = measure
        drift = torch.broadcast_to(drift, state.shape)
        sigma = torch.broadcast_to(sigma, state.shape)
        reaction = torch.broadcast_to(
            torch.clamp(reaction, min=-0.80, max=0.55), state.shape
        )
        substeps, cfl = self._substeps(drift, dt)
        h = dt / substeps
        action = torch.zeros((), dtype=state.dtype, device=state.device)
        for _ in range(substeps):
            interface = 0.5 * (drift[..., 1:] + drift[..., :-1])
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
            diffusivity = 0.5 * sigma * sigma
            interface_d = 0.5 * (diffusivity[..., 1:] + diffusivity[..., :-1])
            lower = -h * interface_d / self.dx**2
            upper = lower.clone()
            diagonal = torch.ones_like(state) - h * reaction
            diagonal[..., 0] = diagonal[..., 0] - upper[..., 0]
            diagonal[..., -1] = diagonal[..., -1] - lower[..., -1]
            diagonal[..., 1:-1] = (
                diagonal[..., 1:-1] - lower[..., :-1] - upper[..., 1:]
            )
            state = torch.clamp(
                self._solve_tridiagonal(lower, diagonal, upper, advected), min=0.0
            )
            action = torch.maximum(action, torch.max(
                torch.sum(
                    (torch.square(drift) + torch.square(reaction)) * state,
                    dim=-1,
                ) * self.dx * h
            ))
        return state, substeps, cfl, action

    def step_repeated_unbalanced(
        self, density, mass, drift, sigma, reaction,
        dt: float, repeats: int, defer_diagnostics: bool = False,
    ):
        """Joint conservative transport, diffusion and Fisher--Rao reaction.

        The method advances the unnormalised measure with reaction on the same
        implicit tridiagonal operator as diffusion.  It therefore avoids the
        transport/reaction operator split used by earlier model versions.
        """
        repeats = int(repeats)
        if repeats < 1:
            raise ValueError("repeats must be at least one")
        input_is_tensor = TORCH_AVAILABLE and torch.is_tensor(density)
        use_torch = self.backend.startswith("torch") and TORCH_AVAILABLE
        if use_torch:
            state = density if input_is_tensor else torch.as_tensor(
                density, dtype=self.dtype, device=self.device
            )
            mass_state = mass if torch.is_tensor(mass) else torch.as_tensor(
                mass, dtype=state.dtype, device=state.device
            )
            drift_state = drift if torch.is_tensor(drift) else torch.as_tensor(
                drift, dtype=state.dtype, device=state.device
            )
            sigma_state = sigma if torch.is_tensor(sigma) else torch.as_tensor(
                sigma, dtype=state.dtype, device=state.device
            )
            reaction_state = reaction if torch.is_tensor(reaction) else torch.as_tensor(
                reaction, dtype=state.dtype, device=state.device
            )
            measure = state * mass_state[..., None]
            diagnostics = []
            if not input_is_tensor:
                self.host_to_device_transfers += 1
            for _ in range(repeats):
                measure, substeps, cfl, action = self._torch_unbalanced_step(
                    measure, drift_state, sigma_state, reaction_state, dt
                )
                diagnostics.append((substeps, cfl, action))
            closing_mass = torch.sum(measure, dim=-1) * self.dx
            closing_density = measure / torch.clamp(
                closing_mass[..., None], min=1.0e-15
            )
            reductions = torch.stack((
                torch.max(torch.abs(
                    torch.sum(closing_density, dim=-1) * self.dx - 1.0
                )),
                torch.max(torch.abs(
                    torch.sum(measure, dim=-1) * self.dx - closing_mass
                )),
                torch.min(measure),
                torch.stack([item[2] for item in diagnostics]).max(),
            ))
            if defer_diagnostics and input_is_tensor:
                self._device_diagnostic_window.append(reductions.detach())
                self._device_cfl_window.append(float(cfl))
                self._device_substeps_window.append(
                    sum(item[0] for item in diagnostics)
                )
                combined = WFRDeviceStepDiagnostics(
                    reductions=reductions, cfl_number=cfl,
                    substeps=sum(item[0] for item in diagnostics),
                )
                self.wfr_calls += 1
                self.calls += 1
                self.physical_steps += repeats
                self.total_substeps += combined.substeps
                return closing_density, closing_mass, combined
            reductions = reductions.detach().cpu().numpy()
            self.deferred_device_reductions += 1
            conditional_residual = float(reductions[0])
            stock_residual = float(reductions[1])
            minimum = float(reductions[2])
            maximum_action = float(reductions[3])
            if not input_is_tensor:
                closing_density = closing_density.detach().cpu().numpy()
                closing_mass = closing_mass.detach().cpu().numpy()
                self.device_to_host_transfers += 1
        else:
            measure = np.asarray(density, dtype=float) * np.asarray(mass)[..., None]
            diagnostics = []
            for _ in range(repeats):
                measure, substeps, cfl, action = self._numpy_unbalanced_step(
                    measure, drift, sigma, reaction, dt
                )
                diagnostics.append((substeps, cfl, action))
            closing_mass = np.sum(measure, axis=-1) * self.dx
            closing_density = measure / np.maximum(
                closing_mass[..., None], 1.0e-15
            )
            conditional_residual = float(np.max(np.abs(
                np.sum(closing_density, axis=-1) * self.dx - 1.0
            )))
            stock_residual = float(np.max(np.abs(
                np.sum(measure, axis=-1) * self.dx - closing_mass
            )))
            minimum = float(np.min(measure))
            maximum_action = max(item[2] for item in diagnostics)
        combined = WFRStepDiagnostics(
            backend="torch_fvm" if use_torch else "numpy_fvm",
            conditional_mass_residual=conditional_residual,
            stock_accounting_residual=stock_residual,
            minimum_measure=minimum,
            cfl_number=max(item[1] for item in diagnostics),
            substeps=sum(item[0] for item in diagnostics),
            action=maximum_action,
        )
        self.wfr_calls += 1
        self.calls += 1
        self.physical_steps += repeats
        self.total_substeps += combined.substeps
        self.max_wfr_conditional_mass_residual = max(
            self.max_wfr_conditional_mass_residual,
            combined.conditional_mass_residual,
        )
        self.max_wfr_stock_accounting_residual = max(
            self.max_wfr_stock_accounting_residual,
            combined.stock_accounting_residual,
        )
        self.minimum_wfr_measure = min(
            self.minimum_wfr_measure, combined.minimum_measure
        )
        self.max_wfr_action = max(self.max_wfr_action, combined.action)
        return closing_density, closing_mass, combined

    def flush_device_diagnostics(self) -> WFRStepDiagnostics | None:
        """Reduce a monthly CUDA diagnostic window with one host transfer."""
        if not self._device_diagnostic_window:
            return None
        values = torch.stack(self._device_diagnostic_window)
        packed = torch.stack((
            values[:, 0].max(), values[:, 1].max(), values[:, 2].min(),
            values[:, 3].max(),
        )).detach().cpu().numpy()
        self._device_diagnostic_window.clear()
        maximum_cfl = max(self._device_cfl_window)
        total_substeps = sum(self._device_substeps_window)
        self._device_cfl_window.clear()
        self._device_substeps_window.clear()
        self.deferred_device_reductions += 1
        combined = WFRStepDiagnostics(
            backend="torch_fvm", conditional_mass_residual=float(packed[0]),
            stock_accounting_residual=float(packed[1]), minimum_measure=float(packed[2]),
            cfl_number=maximum_cfl, substeps=total_substeps, action=float(packed[3]),
        )
        self.max_wfr_conditional_mass_residual = max(
            self.max_wfr_conditional_mass_residual, combined.conditional_mass_residual
        )
        self.max_wfr_stock_accounting_residual = max(
            self.max_wfr_stock_accounting_residual, combined.stock_accounting_residual
        )
        self.minimum_wfr_measure = min(self.minimum_wfr_measure, combined.minimum_measure)
        self.max_wfr_action = max(self.max_wfr_action, combined.action)
        self.max_cfl_number = max(self.max_cfl_number, combined.cfl_number)
        return combined

    def diagnostics(self) -> dict:
        return {
            "backend": self.backend,
            "tridiagonal_solver": self.tridiagonal_solver,
            "device": self.device,
            "torch_available": TORCH_AVAILABLE,
            "torchdiffeq_available": TORCHDIFFEQ_AVAILABLE,
            "triton_available": TRITON_AVAILABLE,
            "max_mass_residual_before_normalisation": self.max_mass_residual,
            "minimum_density_before_projection": self.minimum_density,
            "max_advective_cfl": self.max_cfl_number,
            "calls": self.calls,
            "physical_steps": self.physical_steps,
            "total_substeps": self.total_substeps,
            "host_to_device_transfers": self.host_to_device_transfers,
            "device_to_host_transfers": self.device_to_host_transfers,
            "wfr_calls": self.wfr_calls,
            "deferred_device_reductions": self.deferred_device_reductions,
            "max_wfr_conditional_mass_residual": self.max_wfr_conditional_mass_residual,
            "max_wfr_stock_accounting_residual": self.max_wfr_stock_accounting_residual,
            "minimum_wfr_measure": self.minimum_wfr_measure,
            "max_wfr_action": self.max_wfr_action,
        }
