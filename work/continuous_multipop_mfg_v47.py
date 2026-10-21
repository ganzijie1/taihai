from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cognitive_geometry_v52 import CognitiveInformationGeometry
from fp_solvers_v52 import FokkerPlanckSolver
from information_geometry_v58 import PartiallyObservableInformationGeometry
from network_geometry_v59 import NetworkGeometrySystem
from riemannian_geometry_v60 import (
    FisherRaoTrustRegion,
    RegularizedRiemannianAnderson,
)
from numerical_acceleration import (
    PathwiseAnderson,
    batched_hjb_policy,
    backend_diagnostics,
    batched_thomas,
    resolve_backend,
)


ACTORS = ("china", "taiwan", "united_states", "japan")
AGES = ("18_29", "30_49", "50_64", "65_plus")
GRAPHON_LAYERS = ("information", "economic", "identity", "casualty")
MAJOR_TYPES = ("government", "military", "major_party", "platform_media")
ACTION_CHANNELS = (
    "collective_action", "service", "migration", "hoarding", "information_forward"
)


@dataclass
class MFGFactors:
    support: np.ndarray
    mobilization_preference: np.ndarray
    fatigue_preference: np.ndarray
    threshold_shift: np.ndarray
    control_effort: np.ndarray
    fixed_point_residual: float
    mean_residual: float
    density_residual: float
    bellman_residual: float
    mass_residual: float
    fisher_rao_distance: float
    fixed_point_iterations: int
    max_policy_iterations_used: int
    numerical_backend: str
    converged: bool
    network_action_field: np.ndarray | None = None
    major_action: np.ndarray | None = None
    mean_exploitability_bound: float = 0.0
    worst_type_exploitability_bound: float = 0.0
    finite_network_field_error: float = 0.0
    finite_network_payoff_deviation: float = 0.0
    joint_action_field: np.ndarray | None = None
    engagement_mass: np.ndarray | None = None
    sinkhorn_distance: float = 0.0
    wfr_distance: float = 0.0
    fp_backend: str = ""
    belief_truth_gap: float = 0.0
    observation_uncertainty: float = 0.0
    bridge_cost: float = 0.0
    network_bottleneck_risk: float = 0.0
    network_sheaf_consistency: float = 1.0
    cascade_social_loss: np.ndarray | None = None
    cascade_financial_loss: np.ndarray | None = None
    cascade_supply_loss: np.ndarray | None = None
    cascade_command_loss: np.ndarray | None = None
    cascade_recovery_need: np.ndarray | None = None


class ContinuousMultiPopulationMFG:
    """Continuous-state four-party, four-age HJB-Fokker-Planck MFG.

    The continuous state x in [-1, 1] is net mobilization commitment. The
    implementation solves a bounded-control stationary HJB against a monthly
    Fokker-Planck transition and iterates the mean-field consistency condition.
    """

    def __init__(
        self, paths: int, seed: int, initial_support: np.ndarray,
        grid_points: int = 25, numerical_backend: str | None = None,
        anderson_depth: int = 3, geometry_enabled: bool = False,
        advanced_graphon: bool = False, graphon_uncertainty_scale: float = 0.06,
        graphon_mode: str = "multilayer", fp_backend: str = "legacy_explicit",
        geomloss_interval: int = 12, finite_nodes_per_type: int = 3,
        wfr_strang: bool = False, natural_gradient_major: bool = False,
        partial_observation_geometry: bool = False,
        full_wfr_jko: bool = False,
        fisher_trust_radius: float = 0.035,
        minor_policy_trust_radius: float = 0.025,
        information_ambiguity_radius: float = 0.025,
        trust_region_enabled: bool = True,
        belief_filter_enabled: bool = True,
        credibility_hmm_enabled: bool = True,
        schrodinger_bridge_enabled: bool = True,
        fgw_alignment_enabled: bool = True,
        robust_information_enabled: bool = True,
        information_value_enabled: bool = True,
        network_geometry_enabled: bool = False,
        network_geometry_spectral_interval: int = 12,
        intensity_transition_enabled: bool = True,
        supra_laplacian_enabled: bool = True,
        spectral_geometry_enabled: bool = True,
        forman_curvature_enabled: bool = True,
        sheaf_laplacian_enabled: bool = True,
        riemannian_anderson_enabled: bool = False,
        riemannian_anderson_regularization: float = 1.0e-5,
        riemannian_trust_region_enabled: bool = False,
        riemannian_fixed_point_iterations: int = 96,
        riemannian_minor_relaxation: float = 0.48,
    ):
        if grid_points < 15 or grid_points % 2 == 0:
            raise ValueError("grid_points must be odd and at least 15")
        self.paths = paths
        self.numerical_backend = resolve_backend(numerical_backend)
        self.anderson_depth = max(int(anderson_depth), 0)
        self.geometry_enabled = bool(geometry_enabled)
        self.advanced_graphon = bool(advanced_graphon)
        self.graphon_mode = graphon_mode
        self.wfr_strang = bool(wfr_strang)
        self.natural_gradient_major = bool(natural_gradient_major)
        self.partial_observation_geometry = bool(partial_observation_geometry)
        self.full_wfr_jko = bool(full_wfr_jko)
        self.network_geometry_enabled = bool(network_geometry_enabled)
        self.riemannian_anderson_enabled = bool(riemannian_anderson_enabled)
        self.riemannian_anderson_regularization = float(
            riemannian_anderson_regularization
        )
        self.riemannian_trust_region_enabled = bool(
            riemannian_trust_region_enabled
        )
        self.riemannian_fixed_point_iterations = max(
            int(riemannian_fixed_point_iterations), 48
        )
        self.riemannian_minor_relaxation = float(
            np.clip(riemannian_minor_relaxation, 0.08, 0.48)
        )
        self.minimum_minor_relaxation_used = self.riemannian_minor_relaxation
        self.fisher_trust_radius = float(fisher_trust_radius)
        self.minor_policy_trust_radius = float(minor_policy_trust_radius)
        self.trust_region_enabled = bool(trust_region_enabled)
        if graphon_mode not in {"none", "block", "low_rank", "multilayer", "finite"}:
            raise ValueError("unsupported graphon_mode")
        self.rng = np.random.default_rng(seed)
        self.x = np.linspace(-1.0, 1.0, grid_points)
        self.dx = float(self.x[1] - self.x[0])
        self.age_weights = np.array([0.18, 0.36, 0.27, 0.19])
        self.age_mobility = np.array([1.16, 1.04, 0.88, 0.70])[None, None, :, None]
        self.control_cost = np.array([0.72, 0.82, 0.94, 1.04])[None, None, :, None]
        self.last_mobility_cost = np.ones((paths, len(ACTORS)))
        self.finsler_mobility_cost_active = False
        self.minimum_mobility_cost = 1.0
        self.maximum_mobility_cost = 1.0
        self.maximum_control_cost_shift = 0.0
        self.diffusion = np.array([0.105, 0.090, 0.078, 0.068])[None, None, :, None]
        self.discount = 0.18
        self.actor_kernel = np.array([
            [1.00, 0.18, 0.10, 0.10],
            [0.22, 1.00, 0.32, 0.26],
            [0.08, 0.26, 1.00, 0.38],
            [0.08, 0.24, 0.40, 1.00],
        ])
        self.age_kernel = np.array([
            [0.52, 0.31, 0.12, 0.05],
            [0.18, 0.48, 0.25, 0.09],
            [0.08, 0.28, 0.46, 0.18],
            [0.04, 0.14, 0.29, 0.53],
        ])
        self.layer_actor_kernels = np.array([
            [[1.00, 0.20, 0.16, 0.18], [0.14, 1.00, 0.24, 0.22],
             [0.10, 0.28, 1.00, 0.40], [0.10, 0.26, 0.42, 1.00]],
            [[1.00, 0.12, 0.24, 0.22], [0.18, 1.00, 0.32, 0.30],
             [0.26, 0.34, 1.00, 0.46], [0.24, 0.32, 0.44, 1.00]],
            [[1.00, 0.08, 0.06, 0.07], [0.10, 1.00, 0.16, 0.14],
             [0.06, 0.14, 1.00, 0.22], [0.06, 0.13, 0.24, 1.00]],
            [[1.00, 0.16, 0.12, 0.14], [0.20, 1.00, 0.24, 0.22],
             [0.12, 0.22, 1.00, 0.30], [0.13, 0.21, 0.32, 1.00]],
        ])
        self.graphon_uncertainty_scale = float(graphon_uncertainty_scale)
        perturbation = self.rng.normal(
            0.0, graphon_uncertainty_scale,
            (paths, len(GRAPHON_LAYERS), len(ACTORS), len(ACTORS)),
        )
        self.path_layer_kernels = np.clip(
            self.layer_actor_kernels[None, ...] * np.exp(perturbation), 0.01, 1.50
        )
        self.low_rank_layer_kernels = np.empty_like(self.layer_actor_kernels)
        for layer, kernel in enumerate(self.layer_actor_kernels):
            u, singular, vt = np.linalg.svd(kernel, full_matrices=False)
            self.low_rank_layer_kernels[layer] = (
                u[:, :2] * singular[:2]
            ) @ vt[:2]
        self.low_rank_layer_kernels = np.clip(
            self.low_rank_layer_kernels, 0.01, 1.50
        )
        self.network_geometry = (
            NetworkGeometrySystem(
                self.path_layer_kernels,
                spectral_interval=network_geometry_spectral_interval,
                intensity_transition_enabled=intensity_transition_enabled,
                supra_laplacian_enabled=supra_laplacian_enabled,
                spectral_geometry_enabled=spectral_geometry_enabled,
                forman_curvature_enabled=forman_curvature_enabled,
                sheaf_laplacian_enabled=sheaf_laplacian_enabled,
            )
            if self.network_geometry_enabled else None
        )
        major_bias = np.array([0.06, 0.10, -0.02, -0.05])[None, None, :]
        self.major_action = np.clip(
            (2.0 * initial_support - 1.0)[:, :, None] + major_bias, -0.90, 0.90
        )
        self.major_role_weights = np.array([0.34, 0.28, 0.20, 0.18])
        self.previous_minor_action = np.zeros((paths, len(ACTORS), len(AGES)))
        self.previous_joint_action = np.zeros(
            (paths, len(ACTORS), len(AGES), len(ACTION_CHANNELS))
        )
        if self.partial_observation_geometry:
            self.previous_joint_action.fill(0.05)
        self.engagement_mass = np.ones((paths, len(ACTORS), len(AGES)))
        self.cognitive_transport_cost = np.zeros((paths, len(ACTORS), len(AGES)))
        self.fp_solver = FokkerPlanckSolver(self.dx, backend=fp_backend)
        self.information_geometry = CognitiveInformationGeometry(
            self.x, device=self.fp_solver.device
        )
        self.belief_system = (
            PartiallyObservableInformationGeometry(
                paths, seed + 71_003, initial_support,
                self.actor_kernel, self.age_kernel,
                ambiguity_radius=information_ambiguity_radius,
                belief_filter_enabled=belief_filter_enabled,
                credibility_hmm_enabled=credibility_hmm_enabled,
                schrodinger_bridge_enabled=schrodinger_bridge_enabled,
                fgw_alignment_enabled=fgw_alignment_enabled,
                robust_information_enabled=robust_information_enabled,
                information_value_enabled=information_value_enabled,
            )
            if self.partial_observation_geometry else None
        )
        self.geomloss_interval = max(int(geomloss_interval), 1)
        self.finite_nodes_per_type = max(int(finite_nodes_per_type), 2)
        finite_degree = len(ACTORS) * len(AGES) * self.finite_nodes_per_type
        validation_paths = min(paths, 8)
        self.finite_edge_uniform = self.rng.random(
            (
                validation_paths, len(GRAPHON_LAYERS),
                len(ACTORS), len(AGES), finite_degree, 2,
            )
        )
        if self.graphon_mode == "finite" and paths > validation_paths:
            raise ValueError(
                "finite graph ablation is diagnostic-only and supports at most 8 paths"
            )
        self.month = 0
        centers = np.clip(2.0 * initial_support - 1.0, -0.88, 0.88)[:, :, None, None]
        age_shift = np.array([0.04, 0.02, -0.01, -0.04])[None, None, :, None]
        width = 0.25
        density = np.exp(-0.5 * ((self.x[None, None, None, :] - centers - age_shift) / width) ** 2)
        density /= np.maximum(np.sum(density, axis=-1, keepdims=True) * self.dx, 1e-12)
        self.density = density
        self.value = np.zeros_like(density)
        self.last_residual = np.inf
        self.max_fixed_point_residual = 0.0
        self.max_mass_residual = 0.0
        self.nonconverged_months = 0
        self.nonconverged_events: list[dict[str, object]] = []
        self.max_bellman_residual = 0.0
        self.max_density_residual = 0.0
        self.max_policy_iterations_used = 0
        self.max_fixed_point_iterations_used = 0
        self.anderson_accepted_paths = 0
        self.anderson_rejected_paths = 0
        self.max_fisher_rao_distance = 0.0
        self.max_exploitability_bound = 0.0
        self.sum_exploitability_bound = 0.0
        self.exploitability_observations = 0
        self.max_finite_network_field_error = 0.0
        self.max_finite_network_payoff_deviation = 0.0
        self.max_kernel_row_residual = 0.0
        self.max_major_fixed_point_residual = 0.0
        self.layer_activity = np.zeros(len(GRAPHON_LAYERS))
        self.max_mean_exploitability_bound = 0.0
        self.max_worst_type_exploitability_bound = 0.0
        self.max_wfr_distance = 0.0
        self.max_sinkhorn_distance = 0.0
        self.min_engagement_mass = 1.0
        self.max_engagement_mass = 1.0
        self.max_natural_gradient_distance = 0.0
        self.max_wfr_strang_mass_residual = 0.0
        self.max_full_wfr_conditional_residual = 0.0
        self.max_full_wfr_stock_residual = 0.0
        self.max_major_trust_region_kl = 0.0
        self.max_minor_trust_region_kl = 0.0
        self.major_trust_region_backtracks = 0
        self.minor_trust_region_backtracks = 0
        self.major_riemannian_trust = FisherRaoTrustRegion(
            self.fisher_trust_radius, maximum_radius=0.14
        )
        self.minor_riemannian_trust = FisherRaoTrustRegion(
            self.minor_policy_trust_radius, maximum_radius=0.10
        )
        self.riemannian_anderson_diagnostics = None

    def _fisher_rao_mix(
        self, opening: np.ndarray, target: np.ndarray, weight: float = 0.78
    ) -> tuple[np.ndarray, float]:
        if self.partial_observation_geometry and self.trust_region_enabled:
            p = np.maximum(opening * self.dx, 1.0e-15)
            q = np.maximum(target * self.dx, 1.0e-15)
            affinity = np.clip(
                np.sum(np.sqrt(p * q), axis=-1), 0.0, 1.0
            )
            full_distance = float(np.max(2.0 * np.arccos(affinity)))
            if full_distance * weight > self.fisher_trust_radius:
                weight = max(
                    0.02, self.fisher_trust_radius / max(full_distance, 1.0e-12)
                )
        return self.information_geometry.fisher_rao_mix(opening, target, weight)

    def _moments(self, density: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mean = np.sum(density * self.x[None, None, None, :], axis=-1) * self.dx
        positive = np.sum(
            density * np.clip(self.x[None, None, None, :], 0.0, 1.0), axis=-1
        ) * self.dx
        return mean, positive

    def _apply_jump_generator(
        self, density: np.ndarray, common_noise: np.ndarray,
        jump_signal: np.ndarray, network: np.ndarray, dt: float = 1.0,
    ) -> tuple[np.ndarray, float]:
        """Apply the common-noise jump generator on the master event clock."""
        x = self.x[None, None, None, :]
        directional = 0.22 * common_noise[:, :, None, None] * x
        polarisation = 0.30 * jump_signal[:, :, None, None] * (np.abs(x) - 0.44)
        recovery = 0.035 * network[:, :, None, None] * (0.25 - np.abs(x))
        fitness = directional + polarisation + recovery
        evolved, mass, distance = self.information_geometry.wfr_reaction_step(
            density, self.engagement_mass, fitness, dt=dt
        )
        attrition = np.exp(-0.10 * jump_signal[:, :, None] * dt)
        recovery_mass = np.exp(np.log1p(0.025 * network[:, :, None]) * dt)
        self.engagement_mass = np.clip(mass * attrition * recovery_mass, 0.05, 2.50)
        self.min_engagement_mass = min(
            self.min_engagement_mass, float(np.min(self.engagement_mass))
        )
        self.max_engagement_mass = max(
            self.max_engagement_mass, float(np.max(self.engagement_mass))
        )
        self.max_wfr_distance = max(self.max_wfr_distance, distance)
        return evolved, distance

    def _reaction_fitness(
        self, common_noise: np.ndarray, jump_signal: np.ndarray,
        network: np.ndarray,
    ) -> np.ndarray:
        x = self.x[None, None, None, :]
        directional = 0.22 * common_noise[:, :, None, None] * x
        polarisation = 0.30 * jump_signal[:, :, None, None] * (np.abs(x) - 0.44)
        recovery = 0.035 * network[:, :, None, None] * (0.25 - np.abs(x))
        attrition = -0.10 * jump_signal[:, :, None, None]
        reentry = np.log1p(0.025 * network[:, :, None, None])
        return np.clip(
            directional + polarisation + recovery + attrition + reentry,
            -0.80, 0.55,
        )

    @staticmethod
    def _bernoulli_kl(opening: np.ndarray, closing: np.ndarray) -> np.ndarray:
        p = np.clip(opening, 1.0e-8, 1.0 - 1.0e-8)
        q = np.clip(closing, 1.0e-8, 1.0 - 1.0e-8)
        return p * np.log(p / q) + (1.0 - p) * np.log((1.0 - p) / (1.0 - q))

    def _minor_trust_step(
        self, opening: np.ndarray, target: np.ndarray, weight: float,
    ) -> np.ndarray:
        opening = np.clip(opening, 1.0e-8, 1.0 - 1.0e-8)
        target = np.clip(target, 1.0e-8, 1.0 - 1.0e-8)
        trial_weight = float(weight)
        while True:
            trial = np.clip(
                (1.0 - trial_weight) * opening + trial_weight * target,
                1.0e-8, 1.0 - 1.0e-8,
            )
            maximum = float(np.max(self._bernoulli_kl(trial, opening)))
            if maximum <= self.minor_policy_trust_radius or trial_weight <= 0.015:
                break
            trial_weight *= 0.5
            self.minor_trust_region_backtracks += 1
        self.max_minor_trust_region_kl = max(
            self.max_minor_trust_region_kl, maximum
        )
        return trial

    def _natural_major_step(
        self, opening: np.ndarray, target: np.ndarray, weight: float,
    ) -> tuple[np.ndarray, float]:
        """Fisher--Rao update on positive/negative Bernoulli action shares."""
        opening_probability = np.clip(0.5 * (opening + 1.0), 1.0e-8, 1.0 - 1.0e-8)
        target_probability = np.clip(0.5 * (target + 1.0), 1.0e-8, 1.0 - 1.0e-8)
        p = np.stack((opening_probability, 1.0 - opening_probability), axis=-1)
        q = np.stack((target_probability, 1.0 - target_probability), axis=-1)
        root_p = np.sqrt(p)
        root_q = np.sqrt(q)
        affinity = np.clip(np.sum(root_p * root_q, axis=-1, keepdims=True), 0.0, 1.0)
        theta = np.arccos(affinity)
        sine = np.sin(theta)
        left = np.where(
            sine > 1.0e-10,
            np.sin((1.0 - weight) * theta) / np.maximum(sine, 1.0e-15),
            1.0 - weight,
        )
        right = np.where(
            sine > 1.0e-10,
            np.sin(weight * theta) / np.maximum(sine, 1.0e-15),
            weight,
        )
        mixed = np.square(left * root_p + right * root_q)
        mixed /= np.maximum(np.sum(mixed, axis=-1, keepdims=True), 1.0e-15)
        if self.partial_observation_geometry and self.trust_region_enabled:
            trial_weight = 1.0
            while True:
                trial = np.maximum(
                    (1.0 - trial_weight) * p + trial_weight * mixed, 1.0e-15
                )
                trial /= np.sum(trial, axis=-1, keepdims=True)
                kl = np.sum(
                    trial * np.log(trial / np.maximum(p, 1.0e-15)), axis=-1
                )
                maximum_kl = float(np.max(kl))
                if maximum_kl <= self.fisher_trust_radius or trial_weight <= 0.015:
                    mixed = trial
                    break
                trial_weight *= 0.5
                self.major_trust_region_backtracks += 1
            self.max_major_trust_region_kl = max(
                self.max_major_trust_region_kl, maximum_kl
            )
        distance = float(np.max(2.0 * theta))
        self.max_natural_gradient_distance = max(
            self.max_natural_gradient_distance, distance
        )
        return np.clip(2.0 * mixed[..., 0] - 1.0, -1.0, 1.0), distance

    def _riemannian_minor_step(
        self, opening: np.ndarray, target: np.ndarray, weight: float,
    ) -> np.ndarray:
        updated, _ = self.minor_riemannian_trust.step(opening, target, weight)
        self.max_minor_trust_region_kl = max(
            self.max_minor_trust_region_kl,
            self.minor_riemannian_trust.diagnostics.maximum_kl,
        )
        return updated

    def _riemannian_major_step(
        self, opening: np.ndarray, target: np.ndarray, weight: float,
    ) -> tuple[np.ndarray, float]:
        opening_probability = np.clip(0.5 * (opening + 1.0), 1.0e-10, 1.0 - 1.0e-10)
        target_probability = np.clip(0.5 * (target + 1.0), 1.0e-10, 1.0 - 1.0e-10)
        updated, distance = self.major_riemannian_trust.step(
            opening_probability, target_probability, weight
        )
        self.max_major_trust_region_kl = max(
            self.max_major_trust_region_kl,
            self.major_riemannian_trust.diagnostics.maximum_kl,
        )
        self.max_natural_gradient_distance = max(
            self.max_natural_gradient_distance, distance
        )
        return np.clip(2.0 * updated - 1.0, -1.0, 1.0), distance

    def _joint_action_response(
        self, density: np.ndarray, control: np.ndarray, damage: np.ndarray,
        shortage: np.ndarray, inflation: np.ndarray, network: np.ndarray,
    ) -> np.ndarray:
        mean, positive = self._moments(density)
        positive_control = np.sum(np.maximum(control, 0.0) * density, axis=-1) * self.dx
        negative_control = np.sum(np.maximum(-control, 0.0) * density, axis=-1) * self.dx
        collective = np.clip(positive, 0.0, 1.0)
        service = np.clip(0.48 * positive_control + 0.34 * collective, 0.0, 1.0)
        migration = np.clip(
            0.34 * negative_control + 0.30 * damage[:, :, None]
            + 0.18 * (1.0 - mean), 0.0, 1.0
        )
        hoarding = np.clip(
            0.52 * shortage[:, :, None] + 0.28 * inflation[:, :, None]
            + 0.20 * negative_control, 0.0, 1.0
        )
        information_forward = np.clip(
            network[:, :, None] * (0.25 + 0.55 * collective)
            + 0.20 * positive_control, 0.0, 1.0
        )
        return np.stack(
            [collective, service, migration, hoarding, information_forward], axis=-1
        )

    def _effective_graphon(
        self, damage: np.ndarray, shortage: np.ndarray, network: np.ndarray,
        common_noise: np.ndarray, jump_signal: np.ndarray,
        actor_signal: np.ndarray,
        fgw_alignment: np.ndarray | None = None,
        ambiguity_radius: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.graphon_mode == "none":
            kernel = np.ones(
                (self.paths, len(GRAPHON_LAYERS), len(ACTORS), len(ACTORS))
            )
        elif self.graphon_mode == "block":
            kernel = np.broadcast_to(
                self.layer_actor_kernels[None, ...], self.path_layer_kernels.shape
            ).copy()
        elif self.graphon_mode == "low_rank":
            kernel = np.broadcast_to(
                self.low_rank_layer_kernels[None, ...], self.path_layer_kernels.shape
            ).copy()
        else:
            kernel = self.path_layer_kernels.copy()

        if fgw_alignment is not None:
            kernel *= np.asarray(fgw_alignment)[None, None, :, :]
        if ambiguity_radius is not None:
            radius = np.asarray(ambiguity_radius)[:, None, :, None]
            off_diagonal = 1.0 - np.eye(len(ACTORS))[None, None, :, :]
            kernel *= np.exp(-radius * off_diagonal)

        receiver_network = (0.72 + 0.28 * network)[:, None, :, None]
        sender_damage = (1.0 + 0.34 * damage)[:, None, None, :]
        sender_shortage = (1.0 + 0.24 * shortage)[:, None, None, :]
        kernel[:, 0] *= receiver_network[:, 0]
        kernel[:, 1] *= sender_shortage[:, 0]
        kernel[:, 2] *= (0.86 + 0.14 * network)[:, :, None]
        kernel[:, 3] *= sender_damage[:, 0]
        jump_scale = np.clip(
            1.0 + 0.30 * common_noise + 0.42 * jump_signal, 0.55, 1.75
        )
        kernel *= jump_scale[:, None, :, None]
        weights = np.broadcast_to(
            np.array([0.34, 0.24, 0.24, 0.18])[None, :],
            (self.paths, len(GRAPHON_LAYERS)),
        ).copy()
        weights[:, 0] += 0.08 * np.mean(common_noise, axis=1)
        weights[:, 3] += 0.08 * np.mean(jump_signal, axis=1)
        weights /= np.sum(weights, axis=1, keepdims=True)
        if self.network_geometry is not None:
            geometry = self.network_geometry.update(
                target_kernel=kernel,
                base_layer_weights=weights,
                damage=damage,
                shortage=shortage,
                network=network,
                common_noise=common_noise,
                jump_signal=jump_signal,
                actor_signal=actor_signal,
            )
            kernel = geometry.transition
            weights = geometry.layer_weights
            intensity_multiplier = geometry.intensity_multiplier
        else:
            row_sum = np.sum(kernel, axis=3, keepdims=True)
            kernel /= np.maximum(row_sum, 1e-12)
            intensity_multiplier = np.ones(kernel.shape[:-1])
        self.max_kernel_row_residual = max(
            self.max_kernel_row_residual,
            float(np.max(np.abs(np.sum(kernel, axis=3) - 1.0))),
        )
        self.layer_activity += np.sum(weights, axis=0)
        return kernel, weights, intensity_multiplier

    def _multilayer_fields(
        self, mean: np.ndarray, action: np.ndarray, kernel: np.ndarray,
        weights: np.ndarray, major_action: np.ndarray,
        intensity_multiplier: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        actor_state = np.einsum("plij,pjh->plih", kernel, mean)
        actor_action = np.einsum("plij,pjhc->plihc", kernel, action)
        if intensity_multiplier is not None:
            actor_state *= intensity_multiplier[..., None]
            actor_action *= intensity_multiplier[..., None, None]
        age_state = np.einsum("pih,gh->pig", mean, self.age_kernel)
        age_action = np.einsum("pihc,gh->pigc", action, self.age_kernel)
        layer_state = 0.68 * actor_state + 0.32 * age_state[:, None, :, :]
        layer_action = 0.68 * actor_action + 0.32 * age_action[:, None, :, :, :]
        layer_major_by_role = np.einsum("plij,pjm->plim", kernel, major_action)
        if intensity_multiplier is not None:
            layer_major_by_role *= intensity_multiplier[..., None]
        layer_major = np.einsum(
            "plim,m->pli", layer_major_by_role, self.major_role_weights
        )
        state_field = np.einsum("pl,plih->pih", weights, layer_state)
        action_field = np.einsum("pl,plihc->pihc", weights, layer_action)
        major_field = np.einsum("pl,pli->pi", weights, layer_major)
        return state_field, action_field, major_field

    def _finite_network_fields(
        self, mean: np.ndarray, action: np.ndarray, kernel: np.ndarray,
        weights: np.ndarray, intensity_multiplier: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Back-project the Graphon equilibrium to a sampled finite network."""
        checked = min(mean.shape[0], self.finite_edge_uniform.shape[0])
        state_result = np.zeros_like(mean[:checked])
        action_result = np.zeros_like(action[:checked])
        for path in range(checked):
            layer_state = np.zeros(
                (len(GRAPHON_LAYERS), len(ACTORS), len(AGES))
            )
            layer_action = np.zeros(
                (
                    len(GRAPHON_LAYERS), len(ACTORS), len(AGES),
                    len(ACTION_CHANNELS),
                )
            )
            for layer in range(len(GRAPHON_LAYERS)):
                for actor in range(len(ACTORS)):
                    actor_cdf = np.cumsum(kernel[path, layer, actor])
                    actor_cdf[-1] = 1.0
                    for age in range(len(AGES)):
                        age_cdf = np.cumsum(self.age_kernel[age])
                        age_cdf[-1] = 1.0
                        uniforms = self.finite_edge_uniform[path, layer, actor, age]
                        actor_branch = uniforms[:, 0] < 0.68
                        sender_actor = np.searchsorted(
                            actor_cdf, uniforms[:, 1], side="right"
                        )
                        sender_age = np.searchsorted(
                            age_cdf, uniforms[:, 1], side="right"
                        )
                        sampled_state = np.where(
                            actor_branch,
                            mean[path, sender_actor, age],
                            mean[path, actor, sender_age],
                        )
                        sampled_action = np.where(
                            actor_branch[:, None],
                            action[path, sender_actor, age],
                            action[path, actor, sender_age],
                        )
                        layer_state[layer, actor, age] = np.mean(sampled_state)
                        layer_action[layer, actor, age] = np.mean(
                            sampled_action, axis=0
                        )
            if intensity_multiplier is not None:
                layer_state *= intensity_multiplier[path, :, :, None]
                layer_action *= intensity_multiplier[path, :, :, None, None]
            state_result[path] = np.einsum("l,lih->ih", weights[path], layer_state)
            action_result[path] = np.einsum(
                "l,lihc->ihc", weights[path], layer_action
            )
        graphon_state, graphon_action, _ = self._multilayer_fields(
            mean[:checked], action[:checked], kernel[:checked], weights[:checked],
            self.major_action[:checked],
            None if intensity_multiplier is None else intensity_multiplier[:checked],
        )
        error = float(max(
            np.max(np.abs(state_result - graphon_state)),
            np.max(np.abs(action_result - graphon_action)),
        ))
        return state_result, action_result, error

    def _derivatives(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        first = np.empty_like(values)
        second = np.empty_like(values)
        first[..., 1:-1] = (values[..., 2:] - values[..., :-2]) / (2.0 * self.dx)
        first[..., 0] = (values[..., 1] - values[..., 0]) / self.dx
        first[..., -1] = (values[..., -1] - values[..., -2]) / self.dx
        second[..., 1:-1] = (values[..., 2:] - 2.0 * values[..., 1:-1] + values[..., :-2]) / (self.dx ** 2)
        second[..., 0] = second[..., 1]
        second[..., -1] = second[..., -2]
        return first, second

    def _fp_step(
        self, density: np.ndarray, drift: np.ndarray, sigma: np.ndarray,
        dt: float, repeats: int = 1,
    ) -> np.ndarray:
        evolved, _ = self.fp_solver.step_repeated(
            density, drift, sigma, dt, repeats
        )
        return evolved

    def _solve_hjb(
        self, reward: np.ndarray, base_drift: np.ndarray, initial_value: np.ndarray,
        control_cost: np.ndarray | None = None,
        max_policy_iterations: int = 18, policy_tolerance: float = 2.0e-5,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """Solve the discounted HJB with vectorized tridiagonal policy iteration."""
        value = initial_value.copy()
        control = np.zeros_like(value)
        effective_control_cost = (
            self.control_cost if control_cost is None else np.asarray(control_cost)
        )
        sigma2 = np.broadcast_to(self.diffusion ** 2, value.shape)
        compiled = batched_hjb_policy(
            reward, base_drift, value,
            np.broadcast_to(effective_control_cost, value.shape), sigma2,
            discount=self.discount, dx=self.dx,
            max_iterations=max_policy_iterations, tolerance=policy_tolerance,
            backend=self.numerical_backend,
        )
        if compiled is not None:
            value, control, system_iterations = compiled
            iterations_used = int(np.max(system_iterations))
        else:
            active = np.ones(value.shape[:-1], dtype=bool)
            iterations_used = 0
            for iteration in range(max_policy_iterations):
                value_x, _ = self._derivatives(value)
                proposed = np.clip(value_x / effective_control_cost, -0.55, 0.55)
                control = 0.35 * control + 0.65 * proposed
                drift = np.clip(base_drift + control, -0.75, 0.75)
                left = np.maximum(-drift, 0.0) / self.dx + 0.5 * sigma2 / self.dx ** 2
                right = np.maximum(drift, 0.0) / self.dx + 0.5 * sigma2 / self.dx ** 2
                left[..., 0] = 0.0
                right[..., -1] = 0.0
                diagonal = self.discount + left + right
                lower = -left[..., 1:]
                upper = -right[..., :-1]
                rhs = reward - 0.5 * effective_control_cost * control ** 2
                solved = batched_thomas(
                    lower, diagonal, upper, rhs,
                    backend=self.numerical_backend, pivot_floor=1.0e-10,
                )
                system_residual = np.max(np.abs(solved - value), axis=-1)
                mixed = 0.20 * value + 0.80 * solved
                value = np.where(active[..., None], mixed, value)
                active &= system_residual >= policy_tolerance
                iterations_used = iteration + 1
                if not np.any(active):
                    break
            value_x, _ = self._derivatives(value)
            control = np.clip(value_x / effective_control_cost, -0.55, 0.55)
        drift = np.clip(base_drift + control, -0.75, 0.75)
        left = np.maximum(-drift, 0.0) / self.dx + 0.5 * sigma2 / self.dx ** 2
        right = np.maximum(drift, 0.0) / self.dx + 0.5 * sigma2 / self.dx ** 2
        left[..., 0] = 0.0
        right[..., -1] = 0.0
        diagonal = self.discount + left + right
        lower = -left[..., 1:]
        upper = -right[..., :-1]
        rhs = reward - 0.5 * effective_control_cost * control ** 2
        lhs = diagonal * value
        lhs[..., 1:] += lower * value[..., :-1]
        lhs[..., :-1] += upper * value[..., 1:]
        bellman_residual_grid = np.abs(lhs - rhs)
        return value, control, bellman_residual_grid, iterations_used

    def update(
        self, *, progress: np.ndarray, damage: np.ndarray, shortage: np.ndarray,
        inflation: np.ndarray, financial_stress: np.ndarray, command: np.ndarray,
        network: np.ndarray, contagion_mobilization: np.ndarray,
        contagion_fatigue: np.ndarray, max_fixed_point_iterations: int = 48,
        tolerance: float = 5.0e-4, common_noise: np.ndarray | None = None,
        jump_signal: np.ndarray | None = None,
        mobility_cost: np.ndarray | None = None,
    ) -> MFGFactors:
        common_noise = (
            np.zeros_like(damage) if common_noise is None
            else np.clip(common_noise, -1.0, 1.0)
        )
        jump_signal = (
            np.zeros_like(damage) if jump_signal is None
            else np.clip(jump_signal, 0.0, 1.0)
        )
        if mobility_cost is None:
            mobility = np.ones_like(damage)
        else:
            mobility = np.clip(np.asarray(mobility_cost), 0.50, 3.00)
            if mobility.shape != damage.shape:
                raise ValueError("mobility_cost must have shape (paths, actors)")
            self.finsler_mobility_cost_active = True
        self.last_mobility_cost = mobility.copy()
        self.minimum_mobility_cost = min(
            self.minimum_mobility_cost, float(np.min(mobility))
        )
        self.maximum_mobility_cost = max(
            self.maximum_mobility_cost, float(np.max(mobility))
        )
        effective_control_cost = self.control_cost * mobility[:, :, None, None]
        self.maximum_control_cost_shift = max(
            self.maximum_control_cost_shift,
            float(np.max(np.abs(effective_control_cost - self.control_cost))),
        )
        information_factors = None
        effective_progress = progress
        information_uncertainty = np.zeros_like(progress)
        if self.belief_system is not None:
            information_factors = self.belief_system.update(
                progress=progress, damage=damage, shortage=shortage,
                inflation=inflation, financial_stress=financial_stress,
                command=command, network=network, common_noise=common_noise,
                jump_signal=jump_signal,
            )
            belief_gap = (
                information_factors.robust_self_belief
                - information_factors.self_truth
            )
            effective_progress = np.clip(
                progress + 0.34 * belief_gap, -1.0, 1.0
            )
            information_uncertainty = information_factors.observation_uncertainty
            self_bridge = information_factors.bridge_cost[
                :, np.arange(len(ACTORS)), np.arange(len(ACTORS))
            ]
            self.cognitive_transport_cost = (
                0.72 * self.cognitive_transport_cost
                + 0.28 * self_bridge[:, :, None]
            )
        age_sensitivity = np.array([1.12, 1.04, 0.94, 0.84])[None, None, :]
        benefit = (
            0.70 * effective_progress - 0.55 * damage - 0.48 * shortage
            - 0.34 * inflation - 0.30 * financial_stress
            + 0.28 * command + 0.18 * network
            + 0.34 * contagion_mobilization - 0.42 * contagion_fatigue
            - 0.10 * information_uncertainty
        )[:, :, None] * age_sensitivity - 0.06 * self.cognitive_transport_cost
        fatigue_cost = (
            0.44 * damage + 0.38 * shortage + 0.30 * inflation
            + 0.28 * financial_stress + 0.30 * contagion_fatigue
        )[:, :, None]

        calendar_opening_density = self.density.copy()
        if (
            self.advanced_graphon and self.geometry_enabled
            and not self.full_wfr_jko
        ):
            opening_density, wfr_distance = self._apply_jump_generator(
                calendar_opening_density, common_noise, jump_signal, network,
                dt=0.5 if self.wfr_strang else 1.0,
            )
        else:
            opening_density = calendar_opening_density
            wfr_distance = 0.0
        opening_engagement_mass = self.engagement_mass.copy()
        candidate_engagement_mass = opening_engagement_mass.copy()
        reaction_fitness = self._reaction_fitness(
            common_noise, jump_signal, network
        )
        candidate = opening_density.copy()
        value = self.value.copy()
        residual = np.inf
        mean_residual = np.inf
        density_residual = np.inf
        bellman_residual_grid = np.full_like(candidate, np.inf)
        control = np.zeros_like(candidate)
        minor_action = self.previous_joint_action.copy()
        major_action = self.major_action.copy()
        major_target = major_action.copy()
        network_action_field = np.zeros_like(minor_action)
        major_residual = 0.0
        kernel, layer_weights, intensity_multiplier = self._effective_graphon(
            damage, shortage, network, common_noise, jump_signal,
            effective_progress,
            fgw_alignment=(
                information_factors.fgw_alignment
                if information_factors is not None else None
            ),
            ambiguity_radius=(
                information_factors.ambiguity_radius
                if information_factors is not None else None
            ),
        )
        accelerator = (
            RegularizedRiemannianAnderson(
                depth=self.anderson_depth,
                damping=0.52,
                regularization=self.riemannian_anderson_regularization,
            )
            if self.riemannian_anderson_enabled
            else PathwiseAnderson(depth=self.anderson_depth, damping=0.38)
        )
        previous_residual = np.inf
        previous_action_residual = np.inf
        minor_relaxation = self.riemannian_minor_relaxation
        minimum_minor_relaxation = minor_relaxation
        action_residual_trace: list[float] = []
        post_action_residual = np.inf
        max_policy_used = 0
        fixed_point_iterations = 0
        fisher_rao_distance = 0.0
        iteration_budget = (
            max(max_fixed_point_iterations, self.riemannian_fixed_point_iterations)
            if self.riemannian_anderson_enabled
            else max_fixed_point_iterations
        )
        for iteration in range(iteration_budget):
            mean, _ = self._moments(candidate)
            actor_mean = np.sum(mean * self.age_weights[None, None, :], axis=2)
            if self.advanced_graphon:
                state_field, network_action_field, major_field = self._multilayer_fields(
                    mean, minor_action, kernel, layer_weights, major_action,
                    intensity_multiplier,
                )
                if self.graphon_mode == "finite":
                    state_field, network_action_field, finite_error = self._finite_network_fields(
                        mean, minor_action, kernel, layer_weights,
                        intensity_multiplier,
                    )
                    self.max_finite_network_field_error = max(
                        self.max_finite_network_field_error, finite_error
                    )
                major_base = (
                    0.62 * np.sum(benefit * self.age_weights[None, None, :], axis=2)
                    + 0.34 * actor_mean + 0.24 * major_field
                    - 0.22 * damage - 0.16 * shortage
                )
                major_target = np.tanh(np.stack([
                    major_base + 0.10 * command,
                    major_base + 0.16 * command - 0.08 * shortage,
                    major_base + 0.08 * actor_mean - 0.05 * damage,
                    major_base + 0.13 * network - 0.10 * jump_signal,
                ], axis=-1))
                major_residual = float(np.max(np.abs(major_target - major_action)))
                if self.riemannian_trust_region_enabled:
                    major_action, _ = self._riemannian_major_step(
                        major_action, major_target, 0.45
                    )
                elif self.natural_gradient_major:
                    major_action, _ = self._natural_major_step(
                        major_action, major_target, 0.45
                    )
                else:
                    major_action = 0.55 * major_action + 0.45 * major_target
                action_externality = np.einsum(
                    "pihc,c->pih", network_action_field,
                    np.array([0.34, 0.20, -0.18, -0.16, 0.24]),
                )
                mean_field = (
                    0.48 * state_field + 0.30 * action_externality
                    + 0.22 * major_field[:, :, None]
                )
            else:
                cross_actor = actor_mean @ self.actor_kernel.T
                cross_age = np.einsum("pag,hg->pah", mean, self.age_kernel)
                mean_field = 0.58 * cross_age + 0.42 * cross_actor[:, :, None]
            utility = (
                benefit[..., None] * self.x[None, None, None, :]
                + 0.34 * mean_field[..., None] * self.x[None, None, None, :]
                - 0.24 * self.x[None, None, None, :] ** 2
                - fatigue_cost[..., None] * np.clip(self.x[None, None, None, :], 0.0, 1.0)
            )
            base_drift = (
                0.16 * (mean_field[..., None] - self.x[None, None, None, :])
                + 0.10 * benefit[..., None]
            ) * self.age_mobility
            value, control, bellman_residual_grid, policy_used = self._solve_hjb(
                utility, base_drift, value, effective_control_cost
            )
            action_new = self._joint_action_response(
                candidate, control, damage, shortage, inflation, network
            )
            action_residual = float(np.max(np.abs(action_new - minor_action)))
            if self.riemannian_trust_region_enabled:
                if action_residual > 1.05 * previous_action_residual:
                    minor_relaxation = max(0.08, 0.65 * minor_relaxation)
                elif action_residual < 0.75 * previous_action_residual:
                    minor_relaxation = min(
                        self.riemannian_minor_relaxation,
                        1.08 * minor_relaxation,
                    )
                minimum_minor_relaxation = min(
                    minimum_minor_relaxation, minor_relaxation
                )
                self.minimum_minor_relaxation_used = min(
                    self.minimum_minor_relaxation_used, minor_relaxation
                )
                minor_action = self._riemannian_minor_step(
                    minor_action, action_new, minor_relaxation
                )
            elif self.partial_observation_geometry and self.trust_region_enabled:
                minor_action = self._minor_trust_step(
                    minor_action, action_new, 0.48
                )
            else:
                minor_action = 0.52 * minor_action + 0.48 * action_new
            post_action_residual = float(np.max(np.abs(action_new - minor_action)))
            action_residual_trace.append(action_residual)
            if len(action_residual_trace) > 16:
                action_residual_trace.pop(0)
            max_policy_used = max(max_policy_used, policy_used)
            drift = np.clip(base_drift + control, -0.75, 0.75)
            # The forward equation always starts at the month-opening
            # distribution. Picard iterations reconcile the conjectured
            # endpoint mean field with that single monthly transition; they
            # must not advance calendar time repeatedly inside one month.
            if self.full_wfr_jko:
                evolved, candidate_engagement_mass, wfr_diagnostic = (
                    self.fp_solver.step_repeated_unbalanced(
                        opening_density, opening_engagement_mass,
                        drift * 0.075, self.diffusion * np.sqrt(0.075),
                        reaction_fitness, dt=1.0 / 3.0, repeats=3,
                    )
                )
                self.max_full_wfr_conditional_residual = max(
                    self.max_full_wfr_conditional_residual,
                    wfr_diagnostic.conditional_mass_residual,
                )
                self.max_full_wfr_stock_residual = max(
                    self.max_full_wfr_stock_residual,
                    wfr_diagnostic.stock_accounting_residual,
                )
                wfr_distance = max(
                    wfr_distance,
                    2.0 * float(np.max(np.abs(
                        np.sqrt(np.maximum(candidate_engagement_mass, 0.0))
                        - np.sqrt(np.maximum(opening_engagement_mass, 0.0))
                    ))) + np.sqrt(max(wfr_diagnostic.action, 0.0)),
                )
            else:
                evolved = self._fp_step(
                    opening_density, drift, self.diffusion, dt=0.025, repeats=3
                )
            old_mean, _ = self._moments(candidate)
            new_mean, _ = self._moments(evolved)
            mean_residual = float(np.max(np.abs(new_mean - old_mean)))
            density_by_population = np.sum(
                np.abs(evolved - candidate), axis=-1
            ) * self.dx
            density_residual = float(np.max(density_by_population))
            residual = max(
                mean_residual, density_residual,
                action_residual if self.advanced_graphon else 0.0,
                major_residual if self.advanced_graphon else 0.0,
            )
            if residual > 1.35 * previous_residual:
                accelerator.reset()
            if self.riemannian_anderson_enabled:
                accelerated = accelerator.step(candidate, evolved, self.dx)
                candidate = accelerated
            else:
                accelerated = accelerator.step(candidate, evolved)
                accelerated = np.maximum(accelerated, 0.0)
                accelerated /= np.maximum(
                    np.sum(accelerated, axis=-1, keepdims=True) * self.dx, 1e-12
                )
            if self.geometry_enabled and not self.riemannian_anderson_enabled:
                candidate, step_distance = self._fisher_rao_mix(candidate, accelerated)
                fisher_rao_distance = max(fisher_rao_distance, step_distance)
            elif not self.riemannian_anderson_enabled:
                candidate = accelerated
            previous_residual = residual
            previous_action_residual = action_residual
            fixed_point_iterations = iteration + 1
            if residual < tolerance:
                break

        if (
            self.advanced_graphon and self.geometry_enabled and self.wfr_strang
            and not self.full_wfr_jko
        ):
            candidate, closing_wfr_distance = self._apply_jump_generator(
                candidate, common_noise, jump_signal, network, dt=0.5
            )
            wfr_distance = max(wfr_distance, closing_wfr_distance)
            self.max_wfr_strang_mass_residual = max(
                self.max_wfr_strang_mass_residual,
                float(np.max(np.abs(np.sum(candidate, axis=-1) * self.dx - 1.0))),
            )
        if self.full_wfr_jko:
            self.engagement_mass = np.clip(
                candidate_engagement_mass, 0.05, 2.50
            )
            self.min_engagement_mass = min(
                self.min_engagement_mass, float(np.min(self.engagement_mass))
            )
            self.max_engagement_mass = max(
                self.max_engagement_mass, float(np.max(self.engagement_mass))
            )
            self.max_wfr_distance = max(self.max_wfr_distance, wfr_distance)
        self.density = candidate
        if self.riemannian_anderson_enabled:
            self.riemannian_anderson_diagnostics = accelerator.diagnostics
        self.value = value
        self.major_action = np.clip(major_action, -1.0, 1.0)
        if self.belief_system is not None:
            self.belief_system.register_information_action(self.major_action)
        self.previous_joint_action = np.clip(minor_action, 0.0, 1.0)
        self.previous_minor_action = np.clip(
            minor_action[..., ACTION_CHANNELS.index("collective_action")], 0.0, 1.0
        )
        self.max_major_fixed_point_residual = max(
            self.max_major_fixed_point_residual, major_residual
        )
        mean, positive = self._moments(self.density)
        actor_mean = np.sum(mean * self.age_weights[None, None, :], axis=2)
        actor_positive = np.sum(positive * self.age_weights[None, None, :], axis=2)
        mass_residual = float(np.max(np.abs(np.sum(self.density, axis=-1) * self.dx - 1.0)))
        self.max_mass_residual = max(self.max_mass_residual, mass_residual)
        self.last_residual = residual
        self.max_fixed_point_residual = max(self.max_fixed_point_residual, residual)
        self.max_density_residual = max(self.max_density_residual, density_residual)
        bellman_residual = float(np.max(bellman_residual_grid))
        self.max_bellman_residual = max(self.max_bellman_residual, bellman_residual)
        self.max_policy_iterations_used = max(
            self.max_policy_iterations_used, max_policy_used
        )
        self.max_fixed_point_iterations_used = max(
            self.max_fixed_point_iterations_used, fixed_point_iterations
        )
        self.anderson_accepted_paths += accelerator.diagnostics.accepted_paths
        self.anderson_rejected_paths += accelerator.diagnostics.rejected_paths
        self.max_fisher_rao_distance = max(
            self.max_fisher_rao_distance, fisher_rao_distance
        )
        if residual >= tolerance:
            action_gap = np.abs(action_new - minor_action)
            action_index = tuple(int(x) for x in np.unravel_index(
                int(np.argmax(action_gap)), action_gap.shape
            ))
            self.nonconverged_months += 1
            self.nonconverged_events.append({
                "month": int(self.month),
                "residual": float(residual),
                "mean_residual": float(mean_residual),
                "density_residual": float(density_residual),
                "action_residual": float(action_residual),
                "major_residual": float(major_residual),
                "post_action_residual": float(post_action_residual),
                "minor_relaxation": float(minor_relaxation),
                "minor_trust_radius": float(self.minor_riemannian_trust.radius),
                "action_index": action_index,
                "action_value": float(minor_action[action_index]),
                "action_target": float(action_new[action_index]),
                "action_residual_trace": [float(x) for x in action_residual_trace],
                "iterations": int(fixed_point_iterations),
            })
        support = np.clip(0.5 * (actor_mean + 1.0), 0.01, 0.99)
        fatigue_preference = np.clip(
            0.5 * (1.0 - actor_mean) * (
                0.35 + 0.65 * (damage + shortage + financial_stress) / 3.0
            ), 0.0, 1.0,
        )
        threshold_shift = np.clip(
            -0.10 * actor_mean[:, :, None] + 0.08 * fatigue_cost,
            -0.18, 0.18,
        )
        effort = np.sum(np.abs(control) * self.density, axis=-1) * self.dx
        actor_effort = np.sum(effort * self.age_weights[None, None, :], axis=2)
        type_mean_exploitability = (
            np.sum(bellman_residual_grid * self.density, axis=-1) * self.dx
            / max(self.discount, 1e-12)
        )
        type_worst_exploitability = (
            np.max(bellman_residual_grid, axis=-1) / max(self.discount, 1e-12)
        )
        major_deviation_gain = 0.5 * np.square(major_target - major_action)
        mean_exploitability = float(max(
            np.mean(type_mean_exploitability), np.mean(major_deviation_gain)
        ))
        worst_type_exploitability = float(max(
            np.max(type_worst_exploitability), np.max(major_deviation_gain)
        ))
        exploitability = worst_type_exploitability
        self.max_exploitability_bound = max(
            self.max_exploitability_bound, exploitability
        )
        self.sum_exploitability_bound += mean_exploitability
        self.exploitability_observations += 1
        self.max_mean_exploitability_bound = max(
            self.max_mean_exploitability_bound, mean_exploitability
        )
        self.max_worst_type_exploitability_bound = max(
            self.max_worst_type_exploitability_bound, worst_type_exploitability
        )
        actor_action_field = np.sum(
            network_action_field[..., ACTION_CHANNELS.index("collective_action")]
            * self.age_weights[None, None, :], axis=2
        )
        actor_joint_action_field = np.sum(
            network_action_field * self.age_weights[None, None, :, None], axis=2
        )
        finite_network_error = 0.0
        finite_network_payoff_deviation = 0.0
        if self.advanced_graphon and self.month % 12 == 0:
            _, _, finite_network_error = self._finite_network_fields(
                mean, self.previous_joint_action, kernel, layer_weights,
                intensity_multiplier,
            )
            self.max_finite_network_field_error = max(
                self.max_finite_network_field_error, finite_network_error
            )
            finite_network_payoff_deviation = 0.64 * finite_network_error
            self.max_finite_network_payoff_deviation = max(
                self.max_finite_network_payoff_deviation,
                finite_network_payoff_deviation,
            )
        sinkhorn_distance = 0.0
        if self.geometry_enabled and self.month % self.geomloss_interval == 0:
            measured_profile = self.information_geometry.sinkhorn_profile(
                calendar_opening_density, self.density
            )
            if np.all(np.isfinite(measured_profile)):
                self.cognitive_transport_cost = measured_profile
                sinkhorn_distance = float(np.mean(measured_profile))
                self.max_sinkhorn_distance = max(
                    self.max_sinkhorn_distance, sinkhorn_distance
                )
        self.month += 1
        return MFGFactors(
            support=support,
            mobilization_preference=actor_positive,
            fatigue_preference=fatigue_preference,
            threshold_shift=threshold_shift,
            control_effort=actor_effort,
            fixed_point_residual=residual,
            mean_residual=mean_residual,
            density_residual=density_residual,
            bellman_residual=bellman_residual,
            mass_residual=mass_residual,
            fisher_rao_distance=fisher_rao_distance,
            fixed_point_iterations=fixed_point_iterations,
            max_policy_iterations_used=max_policy_used,
            numerical_backend=self.numerical_backend,
            converged=residual < tolerance,
            network_action_field=actor_action_field,
            major_action=self.major_action.copy(),
            mean_exploitability_bound=mean_exploitability,
            worst_type_exploitability_bound=worst_type_exploitability,
            finite_network_field_error=finite_network_error,
            finite_network_payoff_deviation=finite_network_payoff_deviation,
            joint_action_field=actor_joint_action_field,
            engagement_mass=self.engagement_mass.copy(),
            sinkhorn_distance=sinkhorn_distance,
            wfr_distance=wfr_distance,
            fp_backend=self.fp_solver.backend,
            belief_truth_gap=(
                float(np.max(np.abs(
                    information_factors.self_belief - information_factors.self_truth
                ))) if information_factors is not None else 0.0
            ),
            observation_uncertainty=(
                float(np.mean(information_factors.observation_uncertainty))
                if information_factors is not None else 0.0
            ),
            bridge_cost=(
                float(np.mean(information_factors.bridge_cost))
                if information_factors is not None else 0.0
            ),
            network_bottleneck_risk=(
                float(np.mean(self.network_geometry.last_factors.bottleneck_risk))
                if self.network_geometry is not None else 0.0
            ),
            network_sheaf_consistency=(
                float(np.mean(self.network_geometry.last_factors.sheaf_consistency))
                if self.network_geometry is not None else 1.0
            ),
        )

    def diagnostics(self) -> dict:
        return {
            "grid_points": len(self.x),
            "last_fixed_point_residual": self.last_residual,
            "max_fixed_point_residual": self.max_fixed_point_residual,
            "max_density_residual": self.max_density_residual,
            "max_bellman_residual": self.max_bellman_residual,
            "max_mass_residual": self.max_mass_residual,
            "finsler_mobility_cost_active": self.finsler_mobility_cost_active,
            "minimum_mobility_cost": self.minimum_mobility_cost,
            "maximum_mobility_cost": self.maximum_mobility_cost,
            "mean_last_mobility_cost": float(np.mean(self.last_mobility_cost)),
            "maximum_control_cost_shift": self.maximum_control_cost_shift,
            "geometry_enabled": self.geometry_enabled,
            "wfr_strang": self.wfr_strang,
            "natural_gradient_major": self.natural_gradient_major,
            "partial_observation_geometry": self.partial_observation_geometry,
            "network_geometry_enabled": self.network_geometry_enabled,
            "riemannian_anderson_enabled": self.riemannian_anderson_enabled,
            "riemannian_fixed_point_iterations": self.riemannian_fixed_point_iterations,
            "riemannian_minor_relaxation": self.riemannian_minor_relaxation,
            "minimum_minor_relaxation": self.minimum_minor_relaxation_used,
            "riemannian_trust_region_enabled": self.riemannian_trust_region_enabled,
            "full_wfr_jko": self.full_wfr_jko,
            "fisher_trust_radius": self.fisher_trust_radius,
            "minor_policy_trust_radius": self.minor_policy_trust_radius,
            "trust_region_enabled": self.trust_region_enabled,
            "max_fisher_rao_distance": self.max_fisher_rao_distance,
            "max_natural_gradient_distance": self.max_natural_gradient_distance,
            "max_wfr_strang_mass_residual": self.max_wfr_strang_mass_residual,
            "max_full_wfr_conditional_residual": self.max_full_wfr_conditional_residual,
            "max_full_wfr_stock_residual": self.max_full_wfr_stock_residual,
            "max_major_trust_region_kl": self.max_major_trust_region_kl,
            "max_minor_trust_region_kl": self.max_minor_trust_region_kl,
            "major_trust_region_backtracks": self.major_trust_region_backtracks,
            "minor_trust_region_backtracks": self.minor_trust_region_backtracks,
            "max_policy_iterations_used": self.max_policy_iterations_used,
            "max_fixed_point_iterations_used": self.max_fixed_point_iterations_used,
            "anderson_accepted_paths": self.anderson_accepted_paths,
            "anderson_rejected_paths": self.anderson_rejected_paths,
            "riemannian_anderson": (
                {
                    "attempted": self.riemannian_anderson_diagnostics.attempted,
                    "accepted_paths": self.riemannian_anderson_diagnostics.accepted_paths,
                    "rejected_paths": self.riemannian_anderson_diagnostics.rejected_paths,
                    "resets": self.riemannian_anderson_diagnostics.resets,
                    "maximum_tangent_norm": self.riemannian_anderson_diagnostics.maximum_tangent_norm,
                    "maximum_transport_tangency_residual": self.riemannian_anderson_diagnostics.maximum_transport_tangency_residual,
                    "maximum_sphere_residual": self.riemannian_anderson_diagnostics.maximum_sphere_residual,
                    "maximum_regularized_condition": self.riemannian_anderson_diagnostics.maximum_regularized_condition,
                }
                if self.riemannian_anderson_diagnostics is not None else None
            ),
            "major_riemannian_trust_region": self.major_riemannian_trust.summary(),
            "minor_riemannian_trust_region": self.minor_riemannian_trust.summary(),
            "numerical_backend": backend_diagnostics(self.numerical_backend),
            "nonconverged_months": self.nonconverged_months,
            "nonconverged_events": self.nonconverged_events,
            "advanced_graphon": self.advanced_graphon,
            "graphon_mode": self.graphon_mode,
            "layers": list(GRAPHON_LAYERS),
            "major_types": list(MAJOR_TYPES),
            "action_channels": list(ACTION_CHANNELS),
            "max_kernel_row_residual": self.max_kernel_row_residual,
            "kernel_uncertainty": {
                "method": "pathwise_frozen_logistic_normal_scenario_prior",
                "log_scale": self.graphon_uncertainty_scale,
                "p05": float(np.quantile(self.path_layer_kernels, 0.05)),
                "p50": float(np.quantile(self.path_layer_kernels, 0.50)),
                "p95": float(np.quantile(self.path_layer_kernels, 0.95)),
            },
            "max_major_fixed_point_residual": self.max_major_fixed_point_residual,
            "max_exploitability_bound": self.max_exploitability_bound,
            "max_mean_exploitability_bound": self.max_mean_exploitability_bound,
            "max_worst_type_exploitability_bound": self.max_worst_type_exploitability_bound,
            "mean_exploitability_bound": (
                self.sum_exploitability_bound
                / max(self.exploitability_observations, 1)
            ),
            "max_finite_network_field_error": self.max_finite_network_field_error,
            "max_finite_network_payoff_deviation": self.max_finite_network_payoff_deviation,
            "layer_activity": self.layer_activity.tolist(),
            "minimum_engagement_mass": self.min_engagement_mass,
            "maximum_engagement_mass": self.max_engagement_mass,
            "max_wfr_distance": self.max_wfr_distance,
            "max_sinkhorn_distance": self.max_sinkhorn_distance,
            "fp_solver": self.fp_solver.diagnostics(),
            "information_geometry": self.information_geometry.diagnostics(),
            "partial_observation": (
                self.belief_system.diagnostics()
                if self.belief_system is not None else None
            ),
            "network_geometry": (
                self.network_geometry.diagnostics()
                if self.network_geometry is not None else None
            ),
        }

    def kernel_draw_score(self) -> np.ndarray:
        """Pathwise signed log-deviation from the documented block kernel."""
        if self.network_geometry is not None:
            return self.network_geometry.kernel_draw_score()
        baseline = np.maximum(self.layer_actor_kernels[None, ...], 1.0e-12)
        return np.mean(
            np.log(np.maximum(self.path_layer_kernels, 1.0e-12) / baseline),
            axis=(1, 2, 3),
        )
