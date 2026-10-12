from __future__ import annotations

import numpy as np


class OnlineSystemStabilityV63:
    """Pathwise reduced tangent, Lyapunov, cavity, DMFT and RMT diagnostics."""

    def __init__(self, paths: int, actors: int = 4, rank: int = 4):
        self.paths = paths
        self.actors = actors
        self.features = 9
        self.dimension = actors * self.features
        self.rank = min(rank, self.dimension)
        self.previous = None
        self.jacobian = np.broadcast_to(
            0.92 * np.eye(self.dimension), (paths, self.dimension, self.dimension)
        ).copy()
        q = np.eye(self.dimension, self.rank)
        self.q = np.broadcast_to(q, (paths,) + q.shape).copy()
        self.log_growth = np.zeros((paths, self.rank))
        self.months = 0
        self.max_tangent_response = 0.0
        self.max_real_spectral_edge = -np.inf
        self.max_numerical_abscissa = -np.inf
        self.max_mp_outlier_ratio = 0.0
        self.max_saltation_norm = 0.0
        self.cavity_sum = np.zeros((paths, actors))
        self.dmft_mean = np.zeros((paths, actors))
        self.dmft_variance = np.zeros((paths, actors))
        self.dmft_autocovariance = np.zeros((paths, actors))
        self.dmft_backprojection_error = 0.0

    def update(self, state: np.ndarray, jump: np.ndarray) -> None:
        flat = np.asarray(state, dtype=float).reshape(self.paths, self.dimension)
        if self.previous is None:
            self.previous = flat.copy()
            self.months += 1
            return
        dx = self.previous - np.mean(self.previous, axis=0, keepdims=True)
        dy = flat - np.mean(flat, axis=0, keepdims=True)
        scale = np.sum(dx * dx, axis=1) + 1.0e-8
        secant = np.einsum("pi,pj->pij", dy, dx) / scale[:, None, None]
        self.jacobian = 0.94 * self.jacobian + 0.06 * secant
        jump_flat = np.repeat(np.asarray(jump)[:, :, None], self.features, axis=2)
        jump_flat = jump_flat.reshape(self.paths, self.dimension)
        jump_norm = np.linalg.norm(jump_flat, axis=1)
        active = jump_norm > 1.0e-8
        if np.any(active):
            direction = jump_flat[active] / jump_norm[active, None]
            saltation = np.eye(self.dimension)[None, :, :] + 0.08 * np.einsum(
                "pi,pj->pij", direction, direction
            )
            self.jacobian[active] = np.einsum(
                "pij,pjk->pik", saltation, self.jacobian[active]
            )
            self.max_saltation_norm = max(
                self.max_saltation_norm,
                float(np.max(np.linalg.norm(saltation, axis=(1, 2)))),
            )
        propagated = np.einsum("pij,pjk->pik", self.jacobian, self.q)
        for p in range(self.paths):
            q, r = np.linalg.qr(propagated[p], mode="reduced")
            self.q[p] = q
            self.log_growth[p] += np.log(np.maximum(np.abs(np.diag(r)), 1.0e-12))
        self.max_tangent_response = max(
            self.max_tangent_response,
            float(np.max(np.linalg.norm(propagated, axis=(1, 2)))),
        )
        if self.months % 12 == 0:
            mean_j = np.mean(self.jacobian, axis=0)
            eig = np.linalg.eigvals(mean_j)
            self.max_real_spectral_edge = max(
                self.max_real_spectral_edge, float(np.max(np.real(eig)))
            )
            symmetric = 0.5 * (mean_j + mean_j.T)
            self.max_numerical_abscissa = max(
                self.max_numerical_abscissa,
                float(np.max(np.linalg.eigvalsh(symmetric))),
            )
            singular = np.linalg.svd(mean_j, compute_uv=False)
            noise = np.median(singular) + 1.0e-10
            self.max_mp_outlier_ratio = max(
                self.max_mp_outlier_ratio, float(singular[0] / noise)
            )
        for actor in range(self.actors):
            sl = slice(actor * self.features, (actor + 1) * self.features)
            cavity = self.jacobian.copy()
            cavity[:, :, sl] = 0.0
            impact = np.linalg.norm(self.jacobian - cavity, axis=(1, 2))
            self.cavity_sum[:, actor] += impact
        population = state[:, :, 6:9]
        current_mean = np.mean(population, axis=2)
        current_var = np.var(population, axis=2)
        previous_population = self.previous.reshape(
            self.paths, self.actors, self.features
        )[:, :, 6:9]
        previous_centered = previous_population - np.mean(previous_population, axis=2, keepdims=True)
        current_centered = population - np.mean(population, axis=2, keepdims=True)
        autocov = np.mean(previous_centered * current_centered, axis=2)
        self.dmft_mean = 0.90 * self.dmft_mean + 0.10 * current_mean
        self.dmft_variance = 0.90 * self.dmft_variance + 0.10 * current_var
        self.dmft_autocovariance = 0.90 * self.dmft_autocovariance + 0.10 * autocov
        reconstructed = np.repeat(self.dmft_mean[:, :, None], 3, axis=2)
        error = float(np.sqrt(np.mean((population - reconstructed) ** 2)))
        self.dmft_backprojection_error = max(self.dmft_backprojection_error, error)
        self.previous = flat.copy()
        self.months += 1

    def diagnostics(self) -> dict:
        lyapunov = self.log_growth / max(self.months - 1, 1)
        return {
            "role": "online_read_only_stability_diagnostic",
            "months": self.months,
            "state_dimension": self.dimension,
            "lyapunov_rank": self.rank,
            "lyapunov_mean": np.mean(lyapunov, axis=0).tolist(),
            "lyapunov_p10_p50_p90": np.quantile(
                lyapunov, (0.1, 0.5, 0.9), axis=0
            ).tolist(),
            "max_tangent_response": self.max_tangent_response,
            "max_real_spectral_edge": (
                0.0 if not np.isfinite(self.max_real_spectral_edge)
                else self.max_real_spectral_edge
            ),
            "max_numerical_abscissa": (
                0.0 if not np.isfinite(self.max_numerical_abscissa)
                else self.max_numerical_abscissa
            ),
            "max_random_matrix_outlier_ratio": self.max_mp_outlier_ratio,
            "mean_cavity_importance_by_actor": np.mean(
                self.cavity_sum / max(self.months - 1, 1), axis=0
            ).tolist(),
            "dmft_mean": np.mean(self.dmft_mean, axis=0).tolist(),
            "dmft_variance": np.mean(self.dmft_variance, axis=0).tolist(),
            "dmft_autocovariance": np.mean(
                self.dmft_autocovariance, axis=0
            ).tolist(),
            "dmft_finite_graph_backprojection_error": self.dmft_backprojection_error,
            "max_saltation_norm": self.max_saltation_norm,
            "feeds_terminal_probability": False,
            "future_information_used": False,
        }
