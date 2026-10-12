from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class GeometricMultiscaleFactors:
    consistency: np.ndarray
    combat_factor: np.ndarray
    macro_logistics_factor: np.ndarray
    fast_component: np.ndarray
    slow_component: np.ndarray
    aggregation_residual: float


class GeometricMultiscaleCoupler:
    """Same-path fast/slow geometry linking tactical and macro states.

    The coupler does not own physical stocks. It compares five independently
    owned scales, evolves an explicit slow manifold, and returns bounded
    consistency factors consumed by both campaign tempo and macro logistics.
    """

    def __init__(
        self, paths: int, learned_slow_manifold: bool = False,
        regime_riemannian_pod: bool = False,
        product_manifold_metric: bool = False,
    ):
        self.paths = paths
        self.learned_slow_manifold = bool(learned_slow_manifold)
        self.regime_riemannian_pod = bool(regime_riemannian_pod)
        self.product_manifold_metric = bool(product_manifold_metric)
        self.slow_state = np.ones((paths, 4))
        self.latent_state = np.zeros((paths, 4, 2))
        self.manifold_center, self.encoder, self.decoder, self.validation_rmse = (
            self._calibrate_manifold()
        )
        (
            self.regime_centers, self.regime_encoders,
            self.regime_decoders, self.regime_validation_rmse,
        ) = self._calibrate_regime_manifolds()
        self.regime_state = np.zeros((paths, 4), dtype=np.int8)
        self.previous_stress = np.ones((paths, 4))
        self.regime_transition_count = 0
        self.minimum_product_metric_eigenvalue = 1.0
        self.maximum_cross_metric_norm = 0.0
        self.maximum_geodesic_reconstruction_error = 0.0
        self.last_factors = GeometricMultiscaleFactors(
            consistency=np.ones((paths, 4)),
            combat_factor=np.ones((paths, 4)),
            macro_logistics_factor=np.ones((paths, 4)),
            fast_component=np.zeros((paths, 4)),
            slow_component=self.slow_state.copy(),
            aggregation_residual=0.0,
        )
        self.max_aggregation_residual = 0.0
        self.max_reconstruction_residual = 0.0
        self.max_tangent_residual = 0.0
        self.max_constraint_residual = 0.0
        self.months = 0

    @staticmethod
    def _calibrate_manifold() -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """Fit a two-dimensional log-state manifold on a frozen structural prior.

        The design is an explicit scenario prior, not a historical observation.
        A deterministic holdout is retained to prevent an untested projection.
        """
        index = np.arange(1536, dtype=float)
        phase = 2.0 * np.pi * index / len(index)
        common = 0.52 + 0.31 * np.sin(phase) + 0.09 * np.sin(5.0 * phase)
        operational = 0.56 + 0.27 * np.cos(1.7 * phase) + 0.07 * np.sin(7.0 * phase)
        sample = np.column_stack((
            0.18 + 0.82 * np.clip(0.68 * operational + 0.32 * common, 0.0, 1.0),
            0.20 + 0.80 * np.clip(0.62 * common + 0.38 * operational, 0.0, 1.0),
            0.12 + 0.88 * np.clip(0.78 * common + 0.22 * np.cos(3.0 * phase), 0.0, 1.0),
            0.16 + 0.84 * np.clip(0.74 * operational + 0.26 * np.sin(2.0 * phase), 0.0, 1.0),
            0.14 + 0.86 * np.clip(0.58 * common + 0.42 * operational, 0.0, 1.0),
        ))
        log_sample = np.log(np.maximum(sample, 1.0e-8))
        train = log_sample[:1152]
        validation = log_sample[1152:]
        center = np.mean(train, axis=0)
        _, _, vt = np.linalg.svd(train - center, full_matrices=False)
        encoder = vt[:2].T
        decoder = encoder.T
        reconstructed = (validation - center) @ encoder @ decoder + center
        rmse = float(np.sqrt(np.mean((validation - reconstructed) ** 2)))
        return center, encoder, decoder, rmse

    @staticmethod
    def _calibrate_regime_manifolds() -> tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray
    ]:
        """Fit four local log-Euclidean tangent POD charts on frozen priors."""
        centers = []
        encoders = []
        decoders = []
        errors = []
        base_levels = np.array([0.90, 0.72, 0.46, 0.66])
        feature_load = np.array([0.08, 0.05, 0.11, 0.09, 0.07])
        for regime, base in enumerate(base_levels):
            index = np.arange(768, dtype=float)
            phase = 2.0 * np.pi * index / len(index)
            common = np.sin((1.0 + 0.15 * regime) * phase)
            secondary = np.cos((2.2 + 0.20 * regime) * phase)
            sample = np.column_stack([
                np.clip(base + feature_load[j] * common
                        + (0.035 + 0.006 * j) * secondary
                        + 0.015 * np.sin((j + 3) * phase), 0.05, 1.20)
                for j in range(5)
            ])
            log_sample = np.log(sample)
            train = log_sample[:576]
            validation = log_sample[576:]
            center = np.mean(train, axis=0)
            _, _, vt = np.linalg.svd(train - center, full_matrices=False)
            encoder = vt[:2].T
            decoder = encoder.T
            reconstructed = (validation - center) @ encoder @ decoder + center
            centers.append(center)
            encoders.append(encoder)
            decoders.append(decoder)
            errors.append(float(np.sqrt(np.mean((validation - reconstructed) ** 2))))
        return (
            np.asarray(centers), np.asarray(encoders),
            np.asarray(decoders), np.asarray(errors),
        )

    def _metric_summary(
        self, log_scales: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if not self.product_manifold_metric:
            return np.exp(np.mean(log_scales, axis=2)), np.std(log_scales, axis=2)
        diagonal = np.array([1.12, 0.96, 1.08, 1.16, 0.92])
        cross = np.array([0.10, -0.06, 0.09, 0.11, 0.08])
        metric = np.diag(diagonal) + np.outer(cross, cross)
        eigenvalue = np.linalg.eigvalsh(metric)
        self.minimum_product_metric_eigenvalue = min(
            self.minimum_product_metric_eigenvalue, float(np.min(eigenvalue))
        )
        self.maximum_cross_metric_norm = max(
            self.maximum_cross_metric_norm,
            float(np.linalg.norm(np.outer(cross, cross), ord=2)),
        )
        weights = diagonal / np.sum(diagonal)
        center = np.sum(log_scales * weights[None, None, :], axis=2)
        deviation = log_scales - center[..., None]
        dispersion = np.sqrt(np.maximum(
            np.einsum("paf,fg,pag->pa", deviation, metric, deviation)
            / np.sum(diagonal), 0.0
        ))
        return np.exp(center), dispersion

    def _select_regime(self, stress: np.ndarray) -> np.ndarray:
        regime = np.where(stress > 0.78, 0, np.where(stress > 0.58, 1, 2)).astype(np.int8)
        recovering = (
            ((self.regime_state == 2) | (self.regime_state == 3))
            & (stress - self.previous_stress > 0.012)
            & (stress < 0.84)
        )
        regime[recovering] = 3
        self.regime_transition_count += int(np.count_nonzero(regime != self.regime_state))
        self.regime_state = regime
        self.previous_stress = stress.copy()
        return regime

    def current(self) -> GeometricMultiscaleFactors:
        return self.last_factors

    def update(
        self, *, front_rear_integrity: np.ndarray, sector_output: np.ndarray,
        financial_capacity: np.ndarray, command: np.ndarray,
        energy_logistics: np.ndarray,
    ) -> GeometricMultiscaleFactors:
        logistics = np.clip(np.min(front_rear_integrity, axis=2), 0.02, 1.20)
        output = np.clip(np.mean(sector_output, axis=2), 0.02, 1.30)
        finance = np.clip(financial_capacity, 0.02, 1.10)
        command_level = np.clip(command, 0.02, 1.0)
        energy = np.clip(energy_logistics, 0.02, 1.0)
        scales = np.stack((logistics, output, finance, command_level, energy), axis=2)
        log_scales = np.log(np.maximum(scales, 1.0e-8))
        geometric_mean, dispersion = self._metric_summary(log_scales)
        consistency = np.exp(-dispersion)

        opening_slow = self.slow_state.copy()
        if self.learned_slow_manifold:
            if self.regime_riemannian_pod:
                regime = self._select_regime(np.min(scales, axis=2))
                selected_center = self.regime_centers[regime]
                selected_encoder = self.regime_encoders[regime]
                selected_decoder = self.regime_decoders[regime]
                centered = log_scales - selected_center
                target_latent = np.einsum(
                    "paf,pafd->pad", centered, selected_encoder
                )
            else:
                selected_center = self.manifold_center[None, None, :]
                selected_decoder = np.broadcast_to(
                    self.decoder, (self.paths, 4) + self.decoder.shape
                )
                centered = log_scales - selected_center
                target_latent = np.einsum("paf,fd->pad", centered, self.encoder)
            opening_latent = self.latent_state.copy()
            epsilon = np.clip(0.045 + 0.035 * dispersion, 0.035, 0.095)
            self.latent_state += epsilon[..., None] * (
                target_latent - self.latent_state
            )
            reconstructed_log = np.einsum(
                "pad,padf->paf", self.latent_state, selected_decoder
            ) + selected_center
            reconstructed_scales = np.exp(reconstructed_log)
            manifold_mean = np.exp(np.mean(reconstructed_log, axis=2))
            self.slow_state = np.clip(manifold_mean, 0.04, 1.15)
            reconstruction = float(np.max(np.abs(
                reconstructed_log - log_scales
            )))
            expected_latent = opening_latent + epsilon[..., None] * (
                target_latent - opening_latent
            )
            tangent = float(np.max(np.abs(self.latent_state - expected_latent)))
            constraint = float(np.max(np.maximum(
                np.min(reconstructed_scales, axis=2) - np.max(scales, axis=2), 0.0
            )))
            self.max_reconstruction_residual = max(
                self.max_reconstruction_residual, reconstruction
            )
            self.max_tangent_residual = max(self.max_tangent_residual, tangent)
            self.max_constraint_residual = max(self.max_constraint_residual, constraint)
            if self.regime_riemannian_pod:
                geodesic_error = np.sqrt(np.mean(
                    (reconstructed_log - log_scales) ** 2, axis=2
                ))
                self.maximum_geodesic_reconstruction_error = max(
                    self.maximum_geodesic_reconstruction_error,
                    float(np.max(geodesic_error)),
                )
        else:
            self.slow_state = np.clip(
                0.94 * self.slow_state + 0.06 * geometric_mean, 0.04, 1.15
            )
        fast = geometric_mean - self.slow_state
        reconstructed = self.slow_state + fast
        aggregation_residual = float(np.max(np.abs(reconstructed - geometric_mean)))
        legacy_update_residual = (
            0.0 if self.learned_slow_manifold else float(np.max(np.abs(
                self.slow_state - (0.94 * opening_slow + 0.06 * geometric_mean)
            )))
        )
        self.max_aggregation_residual = max(
            self.max_aggregation_residual, aggregation_residual,
            legacy_update_residual,
        )
        combat_factor = np.clip(
            0.88 + 0.08 * consistency + 0.04 * self.slow_state, 0.82, 1.04
        )
        macro_factor = np.clip(
            0.90 + 0.07 * consistency + 0.03 * np.clip(geometric_mean, 0.0, 1.2),
            0.84, 1.04,
        )
        self.months += 1
        self.last_factors = GeometricMultiscaleFactors(
            consistency=consistency,
            combat_factor=combat_factor,
            macro_logistics_factor=macro_factor,
            fast_component=fast,
            slow_component=self.slow_state.copy(),
            aggregation_residual=aggregation_residual,
        )
        return self.last_factors

    def diagnostics(self) -> dict:
        return {
            "months": self.months,
            "learned_slow_manifold": self.learned_slow_manifold,
            "regime_riemannian_pod": self.regime_riemannian_pod,
            "product_manifold_metric": self.product_manifold_metric,
            "max_aggregation_residual": self.max_aggregation_residual,
            "minimum_consistency": float(np.min(self.last_factors.consistency)),
            "minimum_combat_factor": float(np.min(self.last_factors.combat_factor)),
            "minimum_macro_logistics_factor": float(
                np.min(self.last_factors.macro_logistics_factor)
            ),
            "clock": "monthly_same_path",
            "manifold_calibration_identification": "scenario_prior_with_deterministic_holdout",
            "manifold_validation_rmse": self.validation_rmse,
            "max_manifold_reconstruction_residual": self.max_reconstruction_residual,
            "max_manifold_tangent_residual": self.max_tangent_residual,
            "max_manifold_constraint_residual": self.max_constraint_residual,
            "regime_validation_rmse": self.regime_validation_rmse.tolist(),
            "regime_transition_count": self.regime_transition_count,
            "regime_state_counts": np.bincount(
                self.regime_state.reshape(-1), minlength=4
            ).tolist(),
            "minimum_product_metric_eigenvalue": self.minimum_product_metric_eigenvalue,
            "maximum_cross_metric_norm": self.maximum_cross_metric_norm,
            "maximum_geodesic_reconstruction_error": self.maximum_geodesic_reconstruction_error,
        }
