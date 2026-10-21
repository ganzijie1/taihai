from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


def _normalise_sphere(value: torch.Tensor) -> torch.Tensor:
    return value / torch.linalg.vector_norm(value, dim=-1, keepdim=True).clamp_min(1.0e-15)


def _sphere_log(opening: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    opening = _normalise_sphere(opening)
    target = _normalise_sphere(target)
    affinity = torch.sum(opening * target, dim=-1, keepdim=True).clamp(-1.0, 1.0)
    angle = torch.acos(affinity)
    tangent = target - affinity * opening
    sine = torch.sin(angle)
    scale = torch.where(sine > 1.0e-12, angle / sine.clamp_min(1.0e-15), torch.ones_like(angle))
    return tangent * scale, angle[..., 0]


def _sphere_exp(opening: torch.Tensor, tangent: torch.Tensor) -> torch.Tensor:
    opening = _normalise_sphere(opening)
    norm = torch.linalg.vector_norm(tangent, dim=-1, keepdim=True)
    ratio = torch.where(
        norm > 1.0e-12, torch.sin(norm) / norm.clamp_min(1.0e-15), torch.ones_like(norm)
    )
    return _normalise_sphere(torch.cos(norm) * opening + ratio * tangent)


def _parallel_transport(
    opening: torch.Tensor, target: torch.Tensor, tangent: torch.Tensor,
) -> torch.Tensor:
    opening = _normalise_sphere(opening)
    target = _normalise_sphere(target)
    denominator = (1.0 + torch.sum(opening * target, dim=-1, keepdim=True)).clamp_min(1.0e-12)
    coefficient = torch.sum(tangent * target, dim=-1, keepdim=True) / denominator
    transported = tangent - coefficient * (opening + target)
    return transported - torch.sum(transported * target, dim=-1, keepdim=True) * target


@dataclass
class DeviceAndersonDiagnostics:
    attempted: int = 0
    accepted_paths: int = 0
    rejected_paths: int = 0
    resets: int = 0
    maximum_tangent_norm: float = 0.0
    maximum_transport_tangency_residual: float = 0.0
    maximum_sphere_residual: float = 0.0
    maximum_regularized_condition: float = 0.0


class TorchRegularizedRiemannianAnderson:
    """Pathwise Fisher-sphere Anderson accelerator with CUDA-resident history."""

    def __init__(self, depth: int = 3, damping: float = 0.52,
                 regularization: float = 1.0e-5, trust_ratio: float = 2.5):
        self.depth = max(int(depth), 0)
        self.damping = float(damping)
        self.regularization = float(regularization)
        self.trust_ratio = float(trust_ratio)
        self._points: list[torch.Tensor] = []
        self._residuals: list[torch.Tensor] = []
        self._diagnostic_values: list[torch.Tensor] = []
        self._attempted = 0
        self._resets = 0
        self._accepted = torch.zeros((), dtype=torch.int64, device="cuda")
        self._rejected = torch.zeros((), dtype=torch.int64, device="cuda")

    def reset(self) -> None:
        self._points.clear()
        self._residuals.clear()
        self._resets += 1

    def step(self, density: torch.Tensor, mapped: torch.Tensor, dx: float) -> torch.Tensor:
        if density.shape != mapped.shape or density.ndim != 4:
            raise ValueError("device Riemannian Anderson expects [path, actor, age, grid]")
        opening = _normalise_sphere(torch.sqrt((density * dx).clamp_min(1.0e-30)))
        target = _normalise_sphere(torch.sqrt((mapped * dx).clamp_min(1.0e-30)))
        residual, distance = _sphere_log(opening, target)
        tangent_residual = torch.zeros((), dtype=density.dtype, device=density.device)
        condition = torch.zeros((), dtype=density.dtype, device=density.device)
        self._points.append(opening.clone())
        self._residuals.append(residual.clone())
        keep = self.depth + 1
        if len(self._points) > keep:
            self._points.pop(0)
            self._residuals.pop(0)

        direction = residual
        if self.depth > 0 and len(self._points) >= 2:
            self._attempted += 1
            transported = []
            for point, old_residual in zip(self._points, self._residuals):
                moved = _parallel_transport(point, opening, old_residual)
                tangent_residual = torch.maximum(
                    tangent_residual, torch.abs(torch.sum(moved * opening, dim=-1)).max()
                )
                transported.append(moved.reshape(density.shape[0], -1))
            stacked = torch.stack(transported, dim=2)
            gram = torch.einsum("pnk,pnl->pkl", stacked, stacked)
            size = gram.shape[-1]
            gram = gram + self.regularization * torch.eye(
                size, dtype=gram.dtype, device=gram.device
            )[None]
            ones = torch.ones((density.shape[0], size, 1), dtype=gram.dtype, device=gram.device)
            inverse_ones = torch.linalg.solve(gram, ones)[..., 0]
            coefficient_sum = inverse_ones.sum(dim=1, keepdim=True)
            signed_floor = torch.where(
                coefficient_sum >= 0.0,
                torch.full_like(coefficient_sum, 1.0e-15),
                torch.full_like(coefficient_sum, -1.0e-15),
            )
            coefficient_sum = torch.where(
                torch.abs(coefficient_sum) >= 1.0e-15, coefficient_sum, signed_floor
            )
            coefficients = inverse_ones / coefficient_sum
            combined = torch.einsum("pnk,pk->pn", stacked, coefficients)
            current = residual.reshape(density.shape[0], -1)
            reference_norm = torch.linalg.vector_norm(current, dim=1)
            combined_norm = torch.linalg.vector_norm(combined, dim=1)
            accept = torch.isfinite(combined).all(dim=1) & (
                combined_norm <= self.trust_ratio * reference_norm.clamp_min(1.0e-12)
            )
            direction = torch.where(accept[:, None], combined, current).reshape_as(residual)
            self._accepted += accept.sum()
            self._rejected += (~accept).sum()
            condition = torch.nan_to_num(torch.linalg.cond(gram), nan=0.0, posinf=0.0).max()

        updated = _normalise_sphere(_sphere_exp(opening, self.damping * direction).clamp_min(1.0e-15))
        sphere_residual = torch.abs(torch.sum(updated * updated, dim=-1) - 1.0).max()
        self._diagnostic_values.append(torch.stack((
            distance.max(), tangent_residual, sphere_residual, condition,
        )))
        result = updated.square() / dx
        return result / (result.sum(dim=-1, keepdim=True) * dx).clamp_min(1.0e-15)

    def diagnostics(self) -> DeviceAndersonDiagnostics:
        if self._diagnostic_values:
            maxima = torch.stack(self._diagnostic_values).amax(dim=0)
        else:
            maxima = torch.zeros(4, dtype=torch.float64, device=self._accepted.device)
        packed = torch.cat((
            maxima.to(torch.float64),
            self._accepted[None].to(torch.float64), self._rejected[None].to(torch.float64),
        )).detach().cpu().numpy()
        return DeviceAndersonDiagnostics(
            attempted=self._attempted,
            accepted_paths=int(packed[4]), rejected_paths=int(packed[5]),
            resets=self._resets, maximum_tangent_norm=float(packed[0]),
            maximum_transport_tangency_residual=float(packed[1]),
            maximum_sphere_residual=float(packed[2]),
            maximum_regularized_condition=float(packed[3]),
        )


def device_moments(density: torch.Tensor, x: torch.Tensor, dx: float) -> tuple[torch.Tensor, torch.Tensor]:
    mean = torch.sum(density * x, dim=-1) * dx
    positive = torch.sum(density * torch.clamp(x, 0.0, 1.0), dim=-1) * dx
    return mean, positive


def bulk_device_to_numpy(*values: torch.Tensor) -> tuple[np.ndarray, ...]:
    """Copy heterogeneous monthly states with one device synchronization."""
    shapes = [tuple(value.shape) for value in values]
    sizes = [value.numel() for value in values]
    packed = torch.cat([value.reshape(-1).to(torch.float64) for value in values])
    host = packed.detach().cpu().numpy()
    output = []
    offset = 0
    for shape, size in zip(shapes, sizes):
        output.append(host[offset:offset + size].reshape(shape))
        offset += size
    return tuple(output)
