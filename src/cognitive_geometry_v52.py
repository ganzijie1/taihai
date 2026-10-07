from __future__ import annotations

from dataclasses import dataclass
import importlib.util

import numpy as np


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
GEOMLOSS_AVAILABLE = importlib.util.find_spec("geomloss") is not None

if TORCH_AVAILABLE:
    import torch
else:  # pragma: no cover
    torch = None


@dataclass
class CognitiveGeometryDiagnostics:
    fisher_rao_distance: float
    sinkhorn_distance: float
    wfr_surrogate_distance: float
    opening_mass: float
    closing_mass: float
    mass_residual: float


class CognitiveInformationGeometry:
    """Geometry for belief distributions and engagement mass.

    Conditional beliefs live on a probability simplex. Fisher--Rao geodesics
    update their shape, while a reaction split updates engagement mass. The
    latter is the Fisher--Rao component of an unbalanced/WFR transport step.
    GeomLoss supplies differentiable Sinkhorn diagnostics when installed.
    """

    def __init__(
        self, support: np.ndarray, device: str = "auto", blur: float = 0.08,
        reach: float = 0.45,
    ):
        self.support = np.asarray(support, dtype=float)
        self.dx = float(self.support[1] - self.support[0])
        if device == "auto":
            device = "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu"
        self.device = device
        self.blur = float(blur)
        self.reach = float(reach)
        self.max_fisher_rao_distance = 0.0
        self.max_sinkhorn_distance = 0.0
        self.max_wfr_surrogate_distance = 0.0
        self.max_mass_residual = 0.0
        self.geomloss_calls = 0
        self.geomloss_failures = 0

    def fisher_rao_mix(
        self, opening: np.ndarray, target: np.ndarray, weight: float,
    ) -> tuple[np.ndarray, float]:
        p = np.maximum(opening * self.dx, 1.0e-15)
        q = np.maximum(target * self.dx, 1.0e-15)
        p /= np.maximum(np.sum(p, axis=-1, keepdims=True), 1.0e-15)
        q /= np.maximum(np.sum(q, axis=-1, keepdims=True), 1.0e-15)
        root_p = np.sqrt(p)
        root_q = np.sqrt(q)
        affinity = np.clip(np.sum(root_p * root_q, axis=-1, keepdims=True), 0.0, 1.0)
        theta = np.arccos(affinity)
        denominator = np.sin(theta)
        regular = denominator > 1.0e-8
        left = np.where(
            regular,
            np.sin((1.0 - weight) * theta) / np.maximum(denominator, 1.0e-15),
            1.0 - weight,
        )
        right = np.where(
            regular,
            np.sin(weight * theta) / np.maximum(denominator, 1.0e-15),
            weight,
        )
        root = left * root_p + right * root_q
        mixed = np.maximum(root * root / self.dx, 0.0)
        mixed /= np.maximum(np.sum(mixed, axis=-1, keepdims=True) * self.dx, 1.0e-15)
        distance = float(np.max(2.0 * theta))
        self.max_fisher_rao_distance = max(self.max_fisher_rao_distance, distance)
        return mixed, distance

    def wfr_reaction_step(
        self, density: np.ndarray, mass: np.ndarray, fitness: np.ndarray,
        dt: float,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Exact positive Fisher--Rao reaction split for unbalanced transport."""
        fitness = np.broadcast_to(fitness, density.shape)
        weighted = density * np.exp(np.clip(dt * fitness, -12.0, 12.0))
        normaliser = np.sum(weighted, axis=-1, keepdims=True) * self.dx
        closing_density = weighted / np.maximum(normaliser, 1.0e-15)
        opening_mass = np.asarray(mass, dtype=float)
        closing_mass = np.clip(
            opening_mass * normaliser[..., 0], 0.05, 2.50
        )
        reaction_distance = float(np.max(
            2.0 * np.abs(np.sqrt(closing_mass) - np.sqrt(opening_mass))
        ))
        self.max_wfr_surrogate_distance = max(
            self.max_wfr_surrogate_distance, reaction_distance
        )
        mass_residual = float(np.max(np.abs(
            np.sum(closing_density, axis=-1) * self.dx - 1.0
        )))
        self.max_mass_residual = max(self.max_mass_residual, mass_residual)
        return closing_density, closing_mass, reaction_distance

    def sinkhorn_distance(
        self, opening: np.ndarray, target: np.ndarray, *, unbalanced: bool = False,
        max_batches: int | None = None,
    ) -> float:
        profile = self.sinkhorn_profile(
            opening, target, unbalanced=unbalanced,
            chunk_size=256 if max_batches is None else max_batches,
        )
        return float(np.nanmean(profile)) if np.any(np.isfinite(profile)) else float("nan")

    def sinkhorn_profile(
        self, opening: np.ndarray, target: np.ndarray, *, unbalanced: bool = False,
        chunk_size: int = 256,
    ) -> np.ndarray:
        if not (TORCH_AVAILABLE and GEOMLOSS_AVAILABLE):
            return np.full(opening.shape[:-1], np.nan)
        from geomloss import SamplesLoss

        shape = opening.shape
        p = np.maximum(opening.reshape(-1, shape[-1]) * self.dx, 1.0e-12)
        q = np.maximum(target.reshape(-1, shape[-1]) * self.dx, 1.0e-12)
        if not unbalanced:
            p /= np.sum(p, axis=1, keepdims=True)
            q /= np.sum(q, axis=1, keepdims=True)
        loss = SamplesLoss(
            loss="sinkhorn", p=2, blur=self.blur,
            reach=self.reach if unbalanced else None,
            debias=True, backend="tensorized",
        )
        output = np.empty(p.shape[0], dtype=float)
        try:
            for start in range(0, p.shape[0], chunk_size):
                stop = min(start + chunk_size, p.shape[0])
                x = torch.as_tensor(
                    self.support[None, :, None], dtype=torch.float64,
                    device=self.device,
                ).repeat(stop - start, 1, 1).contiguous()
                p_t = torch.as_tensor(
                    p[start:stop], dtype=torch.float64, device=self.device
                )
                q_t = torch.as_tensor(
                    q[start:stop], dtype=torch.float64, device=self.device
                )
                values = loss(p_t, x, q_t, x)
                output[start:stop] = values.detach().cpu().numpy()
                self.geomloss_calls += 1
            distance = float(np.mean(output))
            self.max_sinkhorn_distance = max(self.max_sinkhorn_distance, distance)
            return output.reshape(shape[:-1])
        except (RuntimeError, ValueError):
            self.geomloss_failures += 1
            return np.full(shape[:-1], np.nan)

    def diagnostics(self) -> dict:
        return {
            "device": self.device,
            "geomloss_available": GEOMLOSS_AVAILABLE,
            "max_fisher_rao_distance": self.max_fisher_rao_distance,
            "max_sinkhorn_distance": self.max_sinkhorn_distance,
            "max_wfr_surrogate_distance": self.max_wfr_surrogate_distance,
            "max_conditional_mass_residual": self.max_mass_residual,
            "geomloss_calls": self.geomloss_calls,
            "geomloss_failures": self.geomloss_failures,
        }
