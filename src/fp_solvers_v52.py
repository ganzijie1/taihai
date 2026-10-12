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


@dataclass
class WFRStepDiagnostics:
    backend: str
    conditional_mass_residual: float
    stock_accounting_residual: float
    minimum_measure: float
    cfl_number: float
    substeps: int
    action: float


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
        self.physical_steps = 0
        self.host_to_device_transfers = 0
        self.device_to_host_transfers = 0
        self.max_wfr_conditional_mass_residual = 0.0
        self.max_wfr_stock_accounting_residual = 0.0
        self.minimum_wfr_measure = math.inf
        self.max_wfr_action = 0.0
        self.wfr_calls = 0

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
        action = 0.0
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
                _torch_thomas(lower, diagonal, upper, advected), min=0.0
            )
            action = max(action, float(torch.max(
                torch.sum(
                    (torch.square(drift) + torch.square(reaction)) * state,
                    dim=-1,
                ) * self.dx * h
            ).detach().cpu()))
        return state, substeps, cfl, action

    def step_repeated_unbalanced(
        self, density, mass, drift, sigma, reaction,
        dt: float, repeats: int,
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
            conditional_residual = float(torch.max(torch.abs(
                torch.sum(closing_density, dim=-1) * self.dx - 1.0
            )).detach().cpu())
            stock_residual = float(torch.max(torch.abs(
                torch.sum(measure, dim=-1) * self.dx - closing_mass
            )).detach().cpu())
            minimum = float(torch.min(measure).detach().cpu())
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
        combined = WFRStepDiagnostics(
            backend="torch_fvm" if use_torch else "numpy_fvm",
            conditional_mass_residual=conditional_residual,
            stock_accounting_residual=stock_residual,
            minimum_measure=minimum,
            cfl_number=max(item[1] for item in diagnostics),
            substeps=sum(item[0] for item in diagnostics),
            action=max(item[2] for item in diagnostics),
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
            "physical_steps": self.physical_steps,
            "total_substeps": self.total_substeps,
            "host_to_device_transfers": self.host_to_device_transfers,
            "device_to_host_transfers": self.device_to_host_transfers,
            "wfr_calls": self.wfr_calls,
            "max_wfr_conditional_mass_residual": self.max_wfr_conditional_mass_residual,
            "max_wfr_stock_accounting_residual": self.max_wfr_stock_accounting_residual,
            "minimum_wfr_measure": self.minimum_wfr_measure,
            "max_wfr_action": self.max_wfr_action,
        }
