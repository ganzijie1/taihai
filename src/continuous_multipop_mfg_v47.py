from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cognitive_geometry_v52 import CognitiveInformationGeometry
from fp_solvers_v52 import FokkerPlanckSolver
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
    ):
        if grid_points < 15 or grid_points % 2 == 0:
            raise ValueError("grid_points must be odd and at least 15")
        self.paths = paths
        self.numerical_backend = resolve_backend(numerical_backend)
        self.anderson_depth = max(int(anderson_depth), 0)
        self.geometry_enabled = bool(geometry_enabled)
        self.advanced_graphon = bool(advanced_graphon)
        self.graphon_mode = graphon_mode
        if graphon_mode not in {"none", "block", "low_rank", "multilayer", "finite"}:
            raise ValueError("unsupported graphon_mode")
        self.rng = np.random.default_rng(seed)
        self.x = np.linspace(-1.0, 1.0, grid_points)
        self.dx = float(self.x[1] - self.x[0])
        self.age_weights = np.array([0.18, 0.36, 0.27, 0.19])
        self.age_mobility = np.array([1.16, 1.04, 0.88, 0.70])[None, None, :, None]
        self.control_cost = np.array([0.72, 0.82, 0.94, 1.04])[None, None, :, None]
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
        major_bias = np.array([0.06, 0.10, -0.02, -0.05])[None, None, :]
        self.major_action = np.clip(
            (2.0 * initial_support - 1.0)[:, :, None] + major_bias, -0.90, 0.90
        )
        self.major_role_weights = np.array([0.34, 0.28, 0.20, 0.18])
        self.previous_minor_action = np.zeros((paths, len(ACTORS), len(AGES)))
        self.previous_joint_action = np.zeros(
            (paths, len(ACTORS), len(AGES), len(ACTION_CHANNELS))
        )
        self.engagement_mass = np.ones((paths, len(ACTORS), len(AGES)))
        self.cognitive_transport_cost = np.zeros((paths, len(ACTORS), len(AGES)))
        self.fp_solver = FokkerPlanckSolver(self.dx, backend=fp_backend)
        self.information_geometry = CognitiveInformationGeometry(
            self.x, device=self.fp_solver.device
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

    def _fisher_rao_mix(
        self, opening: np.ndarray, target: np.ndarray, weight: float = 0.78
    ) -> tuple[np.ndarray, float]:
        return self.information_geometry.fisher_rao_mix(opening, target, weight)

    def _moments(self, density: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mean = np.sum(density * self.x[None, None, None, :], axis=-1) * self.dx
        positive = np.sum(
            density * np.clip(self.x[None, None, None, :], 0.0, 1.0), axis=-1
        ) * self.dx
        return mean, positive

    def _apply_jump_generator(
        self, density: np.ndarray, common_noise: np.ndarray,
        jump_signal: np.ndarray, network: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """Apply the common-noise jump generator on the master event clock."""
        x = self.x[None, None, None, :]
        directional = 0.22 * common_noise[:, :, None, None] * x
        polarisation = 0.30 * jump_signal[:, :, None, None] * (np.abs(x) - 0.44)
        recovery = 0.035 * network[:, :, None, None] * (0.25 - np.abs(x))
        fitness = directional + polarisation + recovery
        evolved, mass, distance = self.information_geometry.wfr_reaction_step(
            density, self.engagement_mass, fitness, dt=1.0
        )
        attrition = np.exp(-0.10 * jump_signal[:, :, None])
        recovery_mass = 1.0 + 0.025 * network[:, :, None]
        self.engagement_mass = np.clip(mass * attrition * recovery_mass, 0.05, 2.50)
        self.min_engagement_mass = min(
            self.min_engagement_mass, float(np.min(self.engagement_mass))
        )
        self.max_engagement_mass = max(
            self.max_engagement_mass, float(np.max(self.engagement_mass))
        )
        self.max_wfr_distance = max(self.max_wfr_distance, distance)
        return evolved, distance

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
    ) -> tuple[np.ndarray, np.ndarray]:
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
        row_sum = np.sum(kernel, axis=3, keepdims=True)
        self.max_kernel_row_residual = max(
            self.max_kernel_row_residual,
            float(np.max(np.abs(np.sum(kernel / np.maximum(row_sum, 1e-12), axis=3) - 1.0))),
        )
        kernel /= np.maximum(row_sum, 1e-12)
        weights = np.broadcast_to(
            np.array([0.34, 0.24, 0.24, 0.18])[None, :],
            (self.paths, len(GRAPHON_LAYERS)),
        ).copy()
        weights[:, 0] += 0.08 * np.mean(common_noise, axis=1)
        weights[:, 3] += 0.08 * np.mean(jump_signal, axis=1)
        weights /= np.sum(weights, axis=1, keepdims=True)
        self.layer_activity += np.sum(weights, axis=0)
        return kernel, weights

    def _multilayer_fields(
        self, mean: np.ndarray, action: np.ndarray, kernel: np.ndarray,
        weights: np.ndarray, major_action: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        actor_state = np.einsum("plij,pjh->plih", kernel, mean)
        actor_action = np.einsum("plij,pjhc->plihc", kernel, action)
        age_state = np.einsum("pih,gh->pig", mean, self.age_kernel)
        age_action = np.einsum("pihc,gh->pigc", action, self.age_kernel)
        layer_state = 0.68 * actor_state + 0.32 * age_state[:, None, :, :]
        layer_action = 0.68 * actor_action + 0.32 * age_action[:, None, :, :, :]
        layer_major_by_role = np.einsum("plij,pjm->plim", kernel, major_action)
        layer_major = np.einsum(
            "plim,m->pli", layer_major_by_role, self.major_role_weights
        )
        state_field = np.einsum("pl,plih->pih", weights, layer_state)
        action_field = np.einsum("pl,plihc->pihc", weights, layer_action)
        major_field = np.einsum("pl,pli->pi", weights, layer_major)
        return state_field, action_field, major_field

    def _finite_network_fields(
        self, mean: np.ndarray, action: np.ndarray, kernel: np.ndarray,
        weights: np.ndarray,
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
            state_result[path] = np.einsum("l,lih->ih", weights[path], layer_state)
            action_result[path] = np.einsum(
                "l,lihc->ihc", weights[path], layer_action
            )
        graphon_state, graphon_action, _ = self._multilayer_fields(
            mean[:checked], action[:checked], kernel[:checked], weights[:checked],
            self.major_action[:checked],
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

    def _fp_step(self, density: np.ndarray, drift: np.ndarray, sigma: np.ndarray, dt: float) -> np.ndarray:
        evolved, _ = self.fp_solver.step(density, drift, sigma, dt)
        return evolved

    def _solve_hjb(
        self, reward: np.ndarray, base_drift: np.ndarray, initial_value: np.ndarray,
        max_policy_iterations: int = 18, policy_tolerance: float = 2.0e-5,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """Solve the discounted HJB with vectorized tridiagonal policy iteration."""
        value = initial_value.copy()
        control = np.zeros_like(value)
        sigma2 = np.broadcast_to(self.diffusion ** 2, value.shape)
        compiled = batched_hjb_policy(
            reward, base_drift, value,
            np.broadcast_to(self.control_cost, value.shape), sigma2,
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
                proposed = np.clip(value_x / self.control_cost, -0.55, 0.55)
                control = 0.35 * control + 0.65 * proposed
                drift = np.clip(base_drift + control, -0.75, 0.75)
                left = np.maximum(-drift, 0.0) / self.dx + 0.5 * sigma2 / self.dx ** 2
                right = np.maximum(drift, 0.0) / self.dx + 0.5 * sigma2 / self.dx ** 2
                left[..., 0] = 0.0
                right[..., -1] = 0.0
                diagonal = self.discount + left + right
                lower = -left[..., 1:]
                upper = -right[..., :-1]
                rhs = reward - 0.5 * self.control_cost * control ** 2
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
            control = np.clip(value_x / self.control_cost, -0.55, 0.55)
        drift = np.clip(base_drift + control, -0.75, 0.75)
        left = np.maximum(-drift, 0.0) / self.dx + 0.5 * sigma2 / self.dx ** 2
        right = np.maximum(drift, 0.0) / self.dx + 0.5 * sigma2 / self.dx ** 2
        left[..., 0] = 0.0
        right[..., -1] = 0.0
        diagonal = self.discount + left + right
        lower = -left[..., 1:]
        upper = -right[..., :-1]
        rhs = reward - 0.5 * self.control_cost * control ** 2
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
    ) -> MFGFactors:
        common_noise = (
            np.zeros_like(damage) if common_noise is None
            else np.clip(common_noise, -1.0, 1.0)
        )
        jump_signal = (
            np.zeros_like(damage) if jump_signal is None
            else np.clip(jump_signal, 0.0, 1.0)
        )
        age_sensitivity = np.array([1.12, 1.04, 0.94, 0.84])[None, None, :]
        benefit = (
            0.70 * progress - 0.55 * damage - 0.48 * shortage
            - 0.34 * inflation - 0.30 * financial_stress
            + 0.28 * command + 0.18 * network
            + 0.34 * contagion_mobilization - 0.42 * contagion_fatigue
        )[:, :, None] * age_sensitivity - 0.06 * self.cognitive_transport_cost
        fatigue_cost = (
            0.44 * damage + 0.38 * shortage + 0.30 * inflation
            + 0.28 * financial_stress + 0.30 * contagion_fatigue
        )[:, :, None]

        calendar_opening_density = self.density.copy()
        if self.advanced_graphon and self.geometry_enabled:
            opening_density, wfr_distance = self._apply_jump_generator(
                calendar_opening_density, common_noise, jump_signal, network
            )
        else:
            opening_density = calendar_opening_density
            wfr_distance = 0.0
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
        kernel, layer_weights = self._effective_graphon(
            damage, shortage, network, common_noise, jump_signal
        )
        accelerator = PathwiseAnderson(depth=self.anderson_depth, damping=0.38)
        previous_residual = np.inf
        max_policy_used = 0
        fixed_point_iterations = 0
        fisher_rao_distance = 0.0
        for iteration in range(max_fixed_point_iterations):
            mean, _ = self._moments(candidate)
            actor_mean = np.sum(mean * self.age_weights[None, None, :], axis=2)
            if self.advanced_graphon:
                state_field, network_action_field, major_field = self._multilayer_fields(
                    mean, minor_action, kernel, layer_weights, major_action
                )
                if self.graphon_mode == "finite":
                    state_field, network_action_field, finite_error = self._finite_network_fields(
                        mean, minor_action, kernel, layer_weights
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
                utility, base_drift, value
            )
            action_new = self._joint_action_response(
                candidate, control, damage, shortage, inflation, network
            )
            action_residual = float(np.max(np.abs(action_new - minor_action)))
            minor_action = 0.52 * minor_action + 0.48 * action_new
            max_policy_used = max(max_policy_used, policy_used)
            drift = np.clip(base_drift + control, -0.75, 0.75)
            # The forward equation always starts at the month-opening
            # distribution. Picard iterations reconcile the conjectured
            # endpoint mean field with that single monthly transition; they
            # must not advance calendar time repeatedly inside one month.
            evolved = opening_density.copy()
            for _ in range(3):
                evolved = self._fp_step(evolved, drift, self.diffusion, dt=0.025)
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
            accelerated = accelerator.step(candidate, evolved)
            accelerated = np.maximum(accelerated, 0.0)
            accelerated /= np.maximum(
                np.sum(accelerated, axis=-1, keepdims=True) * self.dx, 1e-12
            )
            if self.geometry_enabled:
                candidate, step_distance = self._fisher_rao_mix(candidate, accelerated)
                fisher_rao_distance = max(fisher_rao_distance, step_distance)
            else:
                candidate = accelerated
            previous_residual = residual
            fixed_point_iterations = iteration + 1
            if residual < tolerance:
                break

        self.density = candidate
        self.value = value
        self.major_action = np.clip(major_action, -1.0, 1.0)
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
            self.nonconverged_months += 1
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
                mean, self.previous_joint_action, kernel, layer_weights
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
        )

    def diagnostics(self) -> dict:
        return {
            "grid_points": len(self.x),
            "last_fixed_point_residual": self.last_residual,
            "max_fixed_point_residual": self.max_fixed_point_residual,
            "max_density_residual": self.max_density_residual,
            "max_bellman_residual": self.max_bellman_residual,
            "max_mass_residual": self.max_mass_residual,
            "geometry_enabled": self.geometry_enabled,
            "max_fisher_rao_distance": self.max_fisher_rao_distance,
            "max_policy_iterations_used": self.max_policy_iterations_used,
            "max_fixed_point_iterations_used": self.max_fixed_point_iterations_used,
            "anderson_accepted_paths": self.anderson_accepted_paths,
            "anderson_rejected_paths": self.anderson_rejected_paths,
            "numerical_backend": backend_diagnostics(self.numerical_backend),
            "nonconverged_months": self.nonconverged_months,
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
        }

    def kernel_draw_score(self) -> np.ndarray:
        """Pathwise signed log-deviation from the documented block kernel."""
        baseline = np.maximum(self.layer_actor_kernels[None, ...], 1.0e-12)
        return np.mean(
            np.log(np.maximum(self.path_layer_kernels, 1.0e-12) / baseline),
            axis=(1, 2, 3),
        )
