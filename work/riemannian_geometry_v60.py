from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _normalise_sphere(value: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(value, axis=-1, keepdims=True)
    return value / np.maximum(norm, 1.0e-15)


def sphere_log(opening: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    opening = _normalise_sphere(opening)
    target = _normalise_sphere(target)
    affinity = np.clip(np.sum(opening * target, axis=-1, keepdims=True), -1.0, 1.0)
    angle = np.arccos(affinity)
    tangent = target - affinity * opening
    sine = np.sin(angle)
    scale = np.where(sine > 1.0e-12, angle / np.maximum(sine, 1.0e-15), 1.0)
    return tangent * scale, angle[..., 0]


def sphere_exp(opening: np.ndarray, tangent: np.ndarray) -> np.ndarray:
    opening = _normalise_sphere(opening)
    norm = np.linalg.norm(tangent, axis=-1, keepdims=True)
    ratio = np.where(norm > 1.0e-12, np.sin(norm) / np.maximum(norm, 1.0e-15), 1.0)
    updated = np.cos(norm) * opening + ratio * tangent
    return _normalise_sphere(updated)


def sphere_parallel_transport(
    opening: np.ndarray, target: np.ndarray, tangent: np.ndarray,
) -> np.ndarray:
    opening = _normalise_sphere(opening)
    target = _normalise_sphere(target)
    denominator = 1.0 + np.sum(opening * target, axis=-1, keepdims=True)
    coefficient = np.sum(tangent * target, axis=-1, keepdims=True) / np.maximum(
        denominator, 1.0e-12
    )
    transported = tangent - coefficient * (opening + target)
    transported -= np.sum(transported * target, axis=-1, keepdims=True) * target
    return transported


@dataclass
class RiemannianAndersonDiagnostics:
    attempted: int = 0
    accepted_paths: int = 0
    rejected_paths: int = 0
    resets: int = 0
    maximum_tangent_norm: float = 0.0
    maximum_transport_tangency_residual: float = 0.0
    maximum_sphere_residual: float = 0.0
    maximum_regularized_condition: float = 0.0


class RegularizedRiemannianAnderson:
    """Pathwise regularized Anderson mixing on Fisher square-root spheres."""

    def __init__(
        self, depth: int = 3, damping: float = 0.52,
        regularization: float = 1.0e-5, trust_ratio: float = 2.5,
    ):
        self.depth = max(int(depth), 0)
        self.damping = float(damping)
        self.regularization = float(regularization)
        self.trust_ratio = float(trust_ratio)
        self._points: list[np.ndarray] = []
        self._residuals: list[np.ndarray] = []
        self.diagnostics = RiemannianAndersonDiagnostics()

    def reset(self) -> None:
        self._points.clear()
        self._residuals.clear()
        self.diagnostics.resets += 1

    def step(self, density: np.ndarray, mapped: np.ndarray, dx: float) -> np.ndarray:
        if density.shape != mapped.shape or density.ndim != 4:
            raise ValueError("Riemannian Anderson expects [path, actor, age, grid]")
        opening = _normalise_sphere(np.sqrt(np.maximum(density * dx, 1.0e-30)))
        target = _normalise_sphere(np.sqrt(np.maximum(mapped * dx, 1.0e-30)))
        residual, distance = sphere_log(opening, target)
        self.diagnostics.maximum_tangent_norm = max(
            self.diagnostics.maximum_tangent_norm, float(np.max(distance))
        )
        self._points.append(opening.copy())
        self._residuals.append(residual.copy())
        keep = self.depth + 1
        if len(self._points) > keep:
            self._points.pop(0)
            self._residuals.pop(0)

        direction = residual
        accept = np.ones(density.shape[0], dtype=bool)
        if self.depth > 0 and len(self._points) >= 2:
            self.diagnostics.attempted += 1
            transported = []
            for point, old_residual in zip(self._points, self._residuals):
                moved = sphere_parallel_transport(point, opening, old_residual)
                tangency = np.abs(np.sum(moved * opening, axis=-1))
                self.diagnostics.maximum_transport_tangency_residual = max(
                    self.diagnostics.maximum_transport_tangency_residual,
                    float(np.max(tangency)),
                )
                transported.append(moved.reshape(density.shape[0], -1))
            stacked = np.stack(transported, axis=2)
            gram = np.einsum("pnk,pnl->pkl", stacked, stacked)
            size = gram.shape[-1]
            gram += self.regularization * np.eye(size)[None, :, :]
            ones = np.ones((density.shape[0], size, 1))
            try:
                inverse_ones = np.linalg.solve(gram, ones)[..., 0]
                coefficient_sum = np.sum(inverse_ones, axis=1, keepdims=True)
                signed_floor = np.where(
                    coefficient_sum >= 0.0, 1.0e-15, -1.0e-15
                )
                coefficient_sum = np.where(
                    np.abs(coefficient_sum) >= 1.0e-15,
                    coefficient_sum,
                    signed_floor,
                )
                coefficients = inverse_ones / coefficient_sum
                combined = np.einsum("pnk,pk->pn", stacked, coefficients)
                current = residual.reshape(density.shape[0], -1)
                reference_norm = np.linalg.norm(current, axis=1)
                combined_norm = np.linalg.norm(combined, axis=1)
                accept = (
                    np.all(np.isfinite(combined), axis=1)
                    & (combined_norm <= self.trust_ratio * np.maximum(reference_norm, 1.0e-12))
                )
                flat = current.copy()
                flat[accept] = combined[accept]
                direction = flat.reshape(residual.shape)
                condition = np.linalg.cond(gram)
                self.diagnostics.maximum_regularized_condition = max(
                    self.diagnostics.maximum_regularized_condition,
                    float(np.max(np.where(np.isfinite(condition), condition, 0.0))),
                )
            except np.linalg.LinAlgError:
                accept[:] = False
                direction = residual
            self.diagnostics.accepted_paths += int(np.count_nonzero(accept))
            self.diagnostics.rejected_paths += int(np.count_nonzero(~accept))

        updated = sphere_exp(opening, self.damping * direction)
        # The Fisher sphere is the positive orthant. Safeguarding here is a
        # retraction-domain guard, followed by exact density normalisation.
        updated = _normalise_sphere(np.maximum(updated, 1.0e-15))
        self.diagnostics.maximum_sphere_residual = max(
            self.diagnostics.maximum_sphere_residual,
            float(np.max(np.abs(np.sum(updated * updated, axis=-1) - 1.0))),
        )
        result = updated * updated / dx
        result /= np.maximum(np.sum(result, axis=-1, keepdims=True) * dx, 1.0e-15)
        return result


@dataclass
class TrustRegionDiagnostics:
    calls: int = 0
    accepted: int = 0
    rejected: int = 0
    boundary_hits: int = 0
    minimum_radius: float = np.inf
    maximum_radius: float = 0.0
    maximum_gradient_norm: float = 0.0
    minimum_reduction_ratio: float = np.inf
    maximum_kl: float = 0.0


class FisherRaoTrustRegion:
    """Gauss--Newton trust region on a product of Bernoulli manifolds."""

    def __init__(
        self, radius: float, minimum_radius: float = 0.006,
        maximum_radius: float = 0.12,
    ):
        self.radius = float(radius)
        self.minimum_radius = float(minimum_radius)
        self.maximum_radius = float(maximum_radius)
        self.diagnostics = TrustRegionDiagnostics()

    @staticmethod
    def _bernoulli_embedding(probability: np.ndarray) -> np.ndarray:
        probability = np.clip(probability, 1.0e-10, 1.0 - 1.0e-10)
        return np.stack((np.sqrt(probability), np.sqrt(1.0 - probability)), axis=-1)

    @staticmethod
    def _kl(opening: np.ndarray, closing: np.ndarray) -> np.ndarray:
        p = np.clip(opening, 1.0e-10, 1.0 - 1.0e-10)
        q = np.clip(closing, 1.0e-10, 1.0 - 1.0e-10)
        return p * np.log(p / q) + (1.0 - p) * np.log((1.0 - p) / (1.0 - q))

    def step(
        self, opening: np.ndarray, target: np.ndarray, base_fraction: float,
    ) -> tuple[np.ndarray, float]:
        opening = np.clip(opening, 1.0e-10, 1.0 - 1.0e-10)
        target = np.clip(target, 1.0e-10, 1.0 - 1.0e-10)
        point = self._bernoulli_embedding(opening)
        target_point = self._bernoulli_embedding(target)
        logarithm, half_distance = sphere_log(point, target_point)
        distance = 2.0 * half_distance
        maximum_distance = float(np.max(distance))
        self.diagnostics.maximum_gradient_norm = max(
            self.diagnostics.maximum_gradient_norm, maximum_distance
        )
        fraction = np.minimum(
            float(base_fraction), self.radius / np.maximum(distance, 1.0e-12)
        )
        if np.any(fraction < float(base_fraction) - 1.0e-12):
            self.diagnostics.boundary_hits += 1
        trial_point = sphere_exp(point, fraction[..., None] * logarithm)
        trial = np.clip(trial_point[..., 0] ** 2, 1.0e-10, 1.0 - 1.0e-10)

        old_objective = 0.5 * distance * distance
        _, new_half_distance = sphere_log(
            self._bernoulli_embedding(trial), target_point
        )
        new_objective = 0.5 * (2.0 * new_half_distance) ** 2
        actual = old_objective - new_objective
        predicted = old_objective * np.maximum(2.0 * fraction - fraction * fraction, 1.0e-12)
        ratio = actual / np.maximum(predicted, 1.0e-15)
        minimum_ratio = float(np.min(ratio))
        accepted = np.isfinite(ratio) & (ratio >= 0.10)
        result = np.where(accepted, trial, opening)
        self.diagnostics.calls += 1
        self.diagnostics.accepted += int(np.count_nonzero(accepted))
        self.diagnostics.rejected += int(np.count_nonzero(~accepted))
        self.diagnostics.minimum_reduction_ratio = min(
            self.diagnostics.minimum_reduction_ratio, minimum_ratio
        )
        if minimum_ratio < 0.25:
            self.radius = max(self.minimum_radius, 0.5 * self.radius)
        elif minimum_ratio > 0.75 and np.any(fraction < float(base_fraction)):
            self.radius = min(self.maximum_radius, 1.35 * self.radius)
        self.diagnostics.minimum_radius = min(
            self.diagnostics.minimum_radius, self.radius
        )
        self.diagnostics.maximum_radius = max(
            self.diagnostics.maximum_radius, self.radius
        )
        maximum_kl = float(np.max(self._kl(result, opening)))
        self.diagnostics.maximum_kl = max(self.diagnostics.maximum_kl, maximum_kl)
        return result, maximum_distance

    def summary(self) -> dict:
        diagnostic = self.diagnostics
        return {
            "calls": diagnostic.calls,
            "accepted": diagnostic.accepted,
            "rejected": diagnostic.rejected,
            "boundary_hits": diagnostic.boundary_hits,
            "current_radius": self.radius,
            "minimum_radius": (
                diagnostic.minimum_radius if np.isfinite(diagnostic.minimum_radius)
                else self.radius
            ),
            "maximum_radius": diagnostic.maximum_radius,
            "maximum_gradient_norm": diagnostic.maximum_gradient_norm,
            "minimum_reduction_ratio": (
                diagnostic.minimum_reduction_ratio
                if np.isfinite(diagnostic.minimum_reduction_ratio) else 1.0
            ),
            "maximum_kl": diagnostic.maximum_kl,
        }
