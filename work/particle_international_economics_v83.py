from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from differential_manifold_mfg_v81 import DifferentialManifoldMFGV81
from nested_ces_cge_v77 import NestedCESCGEV77


ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "data" / "particle_trade_v83" / "structural_priors.json"


def _systematic_resample(weights: np.ndarray, uniforms: np.ndarray) -> np.ndarray:
    """Vectorized systematic resampling for rows of normalized weights."""
    cumulative = np.cumsum(weights, axis=-1)
    cumulative[..., -1] = 1.0
    positions = (uniforms[..., None] + np.arange(weights.shape[-1])) / weights.shape[-1]
    return np.sum(positions[..., :, None] > cumulative[..., None, :], axis=-1)


class LatentIntentParticleSystemV83:
    """Sequential Monte Carlo latent-intent estimator with collision mixing."""

    def __init__(self, paths: int, seed: int, initial_support: np.ndarray, config: dict):
        self.paths = paths
        self.rng = np.random.default_rng(seed)
        self.n = int(config["intent_particles"])
        center = np.clip(2.0 * initial_support - 1.0, -0.95, 0.95)
        self.particles = np.clip(
            center[..., None] + self.rng.normal(0.0, 0.12, (paths, 4, self.n)),
            -1.0, 1.0,
        )
        self.weights = np.full((paths, 4, self.n), 1.0 / self.n)
        self.compromise = np.asarray(config["opinion_compromise"], dtype=float)[None, :, None]
        self.process_noise = np.asarray(config["process_noise"], dtype=float)[None, :, None]
        self.observation_noise = np.asarray(config["observation_noise"], dtype=float)[None, :, None]
        self.ess_threshold = float(config["resampling_ess_fraction"]) * self.n
        self.months = 0
        self.resampling_events = 0
        self.minimum_ess = float(self.n)
        self.maximum_weight_residual = 0.0
        self.maximum_intent_feedback = 0.0
        self.last_mean = center.copy()
        self.last_variance = np.full((paths, 4), 0.12 ** 2)

    def update(self, *, progress: np.ndarray, damage: np.ndarray,
               shortage: np.ndarray, command: np.ndarray, network: np.ndarray,
               common_noise: np.ndarray, jump_signal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        partner = np.roll(self.particles, 1 + self.months % max(self.n - 1, 1), axis=-1)
        social_access = np.clip(network[..., None], 0.05, 1.0)
        drift = (
            self.compromise * social_access * (partner - self.particles)
            + 0.055 * progress[..., None]
            + 0.030 * command[..., None]
            - 0.045 * damage[..., None]
            - 0.040 * shortage[..., None]
        )
        noise_scale = self.process_noise * (
            1.0 + 0.40 * np.abs(common_noise[..., None])
            + 0.55 * jump_signal[..., None]
        )
        self.particles = np.clip(
            self.particles + drift
            + self.rng.normal(0.0, 1.0, self.particles.shape) * noise_scale,
            -1.0, 1.0,
        )
        observation = np.clip(
            0.48 * progress + 0.22 * command + 0.14 * network
            - 0.10 * damage - 0.08 * shortage,
            -1.0, 1.0,
        )
        log_likelihood = -0.5 * (
            (self.particles - observation[..., None]) / self.observation_noise
        ) ** 2
        log_weights = np.log(np.maximum(self.weights, 1.0e-300)) + log_likelihood
        log_weights -= np.max(log_weights, axis=-1, keepdims=True)
        self.weights = np.exp(log_weights)
        self.weights /= np.sum(self.weights, axis=-1, keepdims=True)
        ess = 1.0 / np.sum(self.weights ** 2, axis=-1)
        self.minimum_ess = min(self.minimum_ess, float(np.min(ess)))
        needs = ess < self.ess_threshold
        if np.any(needs):
            uniforms = self.rng.random((self.paths, 4))
            indices = _systematic_resample(self.weights, uniforms)
            resampled = np.take_along_axis(self.particles, indices, axis=-1)
            self.particles = np.where(needs[..., None], resampled, self.particles)
            self.weights = np.where(
                needs[..., None], 1.0 / self.n, self.weights
            )
            self.resampling_events += int(np.count_nonzero(needs))
        mean = np.sum(self.weights * self.particles, axis=-1)
        variance = np.sum(
            self.weights * (self.particles - mean[..., None]) ** 2, axis=-1
        )
        self.last_mean = mean
        self.last_variance = variance
        self.months += 1
        self.maximum_weight_residual = max(
            self.maximum_weight_residual,
            float(np.max(np.abs(np.sum(self.weights, axis=-1) - 1.0))),
        )
        return mean, variance

    def diagnostics(self) -> dict:
        return {
            "months": self.months,
            "particle_count": self.n,
            "resampling_events": self.resampling_events,
            "minimum_effective_sample_size": self.minimum_ess,
            "maximum_weight_normalization_residual": self.maximum_weight_residual,
            "mean_posterior_variance_by_actor": np.mean(self.last_variance, axis=0).tolist(),
            "maximum_intent_feedback": self.maximum_intent_feedback,
            "state_owner": "LatentIntentParticleSystemV83",
        }


class AdaptiveMultilevelSplittingV83:
    """Generic AMS estimator. It never edits simulated outcome probabilities."""

    def __init__(self, replicas: int = 128, kill_fraction: float = 0.10,
                 seed: int = 83_901):
        self.replicas = max(int(replicas), 32)
        self.kill = max(1, int(round(self.replicas * kill_fraction)))
        self.rng = np.random.default_rng(seed)

    def estimate(self, transition, score, threshold: float,
                 max_levels: int = 128) -> dict:
        states = transition(self.rng, None, self.replicas)
        levels = 0
        survival = 1.0
        while levels < max_levels:
            values = np.asarray(score(states), dtype=float)
            if np.min(values) >= threshold:
                break
            order = np.argsort(values)
            killed = order[: self.kill]
            survivors = order[self.kill :]
            if len(survivors) == 0:
                survival = 0.0
                break
            parents = self.rng.choice(survivors, size=self.kill, replace=True)
            states[killed] = transition(self.rng, states[parents], self.kill)
            survival *= 1.0 - self.kill / self.replicas
            levels += 1
        final_share = float(np.mean(np.asarray(score(states)) >= threshold))
        return {
            "probability": survival * final_share,
            "levels": levels,
            "replicas": self.replicas,
            "estimator_only": True,
        }


class ParticleValidatedMFGV83(DifferentialManifoldMFGV81):
    """V8.1 MFG with a filtered latent-intent state and particle validation."""

    def __init__(self, *args, **kwargs):
        particle_enabled = kwargs.pop("particle_intent_enabled", True)
        finite_validation_enabled = kwargs.pop("finite_particle_validation_enabled", True)
        config = json.loads(LEDGER.read_text(encoding="utf-8"))["particle"]
        intent_particles = kwargs.pop("intent_particles", None)
        if intent_particles is not None:
            config = dict(config)
            config["intent_particles"] = int(intent_particles)
        paths, seed, initial_support = args[:3]
        super().__init__(*args, **kwargs)
        self.particle_enabled = bool(particle_enabled)
        self.finite_validation_enabled = bool(finite_validation_enabled)
        self.particle_intent = LatentIntentParticleSystemV83(
            paths, seed + 83_003, initial_support, config
        )
        self.finite_n = int(config["finite_player_particles"])
        self.check_interval = int(config["finite_player_check_interval_months"])
        self.v83_months = 0
        self.particle_validation_checks = 0
        self.maximum_empirical_wasserstein = 0.0
        self.maximum_finite_support_error = 0.0
        self.maximum_intent_feedback = 0.0

    def _finite_particle_check(self, support: np.ndarray) -> None:
        quantiles = (np.arange(self.finite_n) + 0.5) / self.finite_n
        cdf = np.cumsum(self.density, axis=-1) * self.dx
        cdf /= np.maximum(cdf[..., -1:], 1.0e-12)
        indices = np.sum(cdf[..., None, :] < quantiles[None, None, None, :, None], axis=-1)
        indices = np.clip(indices, 0, len(self.x) - 1)
        samples = self.x[indices]
        empirical_mean = np.mean(samples, axis=-1)
        model_mean = np.sum(self.density * self.x[None, None, None, :], axis=-1) * self.dx
        actor_empirical = np.sum(empirical_mean * self.age_weights[None, None, :], axis=2)
        support_error = np.abs(0.5 * (actor_empirical + 1.0) - support)
        # On a common one-dimensional grid, the mean error is a conservative
        # lower-order finite-particle Wasserstein diagnostic.
        self.maximum_empirical_wasserstein = max(
            self.maximum_empirical_wasserstein,
            float(np.max(np.abs(empirical_mean - model_mean))),
        )
        self.maximum_finite_support_error = max(
            self.maximum_finite_support_error, float(np.max(support_error))
        )
        self.particle_validation_checks += 1

    def update(self, **kwargs):
        modified = dict(kwargs)
        mean, variance = self.particle_intent.update(
            progress=kwargs["progress"], damage=kwargs["damage"],
            shortage=kwargs["shortage"], command=kwargs["command"],
            network=kwargs["network"],
            common_noise=kwargs.get("common_noise", np.zeros_like(kwargs["damage"])),
            jump_signal=kwargs.get("jump_signal", np.zeros_like(kwargs["damage"])),
        )
        if self.particle_enabled:
            confidence = np.exp(-2.5 * variance)
            feedback = np.clip(0.10 * confidence * mean, -0.08, 0.08)
            modified["progress"] = np.clip(kwargs["progress"] + feedback, -1.0, 1.0)
            self.maximum_intent_feedback = max(
                self.maximum_intent_feedback, float(np.max(np.abs(feedback)))
            )
            self.particle_intent.maximum_intent_feedback = self.maximum_intent_feedback
        factors = super().update(**modified)
        self.v83_months += 1
        if self.finite_validation_enabled and self.v83_months % self.check_interval == 0:
            self._finite_particle_check(factors.support)
        return factors

    def diagnostics(self) -> dict:
        payload = super().diagnostics()
        payload["v83_particle_methods"] = self.particle_intent.diagnostics() | {
            "enabled": self.particle_enabled,
            "finite_particle_validation_enabled": self.finite_validation_enabled,
            "finite_particle_count": self.finite_n,
            "finite_particle_validation_checks": self.particle_validation_checks,
            "maximum_empirical_wasserstein_proxy": self.maximum_empirical_wasserstein,
            "maximum_finite_support_error": self.maximum_finite_support_error,
            "ams_backend_implemented": True,
            "ams_changes_outcomes": False,
            "terminal_probability_adjustment": False,
            "data_ledger": str(LEDGER),
        }
        return payload


class StructuralInternationalCGEV83(NestedCESCGEV77):
    """CGE with endogenous structural-gravity, EK and Melitz trade states."""

    def __init__(self, *args, structural_gravity_enabled: bool = True,
                 melitz_enabled: bool = True,
                 external_liquidity_enabled: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        cfg = json.loads(LEDGER.read_text(encoding="utf-8"))["trade"]
        self.theta_actor = np.asarray(cfg["trade_elasticity_theta"], dtype=float)
        self.border_cost = np.asarray(cfg["ppml_border_cost"], dtype=float)
        self.technology_actor = np.asarray(cfg["technology_level"], dtype=float)
        self.entry_cost = np.asarray(cfg["melitz_entry_cost"], dtype=float)
        self.exit_sensitivity = np.asarray(cfg["melitz_exit_sensitivity"], dtype=float)
        self.capital_controls = np.asarray(cfg["capital_control_strength"], dtype=float)
        self.rerouting_capacity = np.asarray(cfg["third_party_rerouting_capacity"], dtype=float)
        self.structural_gravity_enabled = bool(structural_gravity_enabled)
        self.melitz_enabled = bool(melitz_enabled)
        self.external_liquidity_enabled = bool(external_liquidity_enabled)
        self.external_liquidity = np.broadcast_to(
            np.asarray(cfg["initial_external_liquidity"], dtype=float),
            (self.paths, 4),
        ).copy()
        if not self.external_liquidity_enabled:
            self.external_liquidity.fill(1.0)
        self.firm_active_share = np.full((self.paths, 4), 0.84)
        self.trade_cost = np.ones((self.paths, 4, 4))
        self.source_share = np.full((self.paths, 4, 4), 0.25)
        self.trade_resilience = np.ones((self.paths, 4))
        self.months_v83 = 0
        self.max_source_share_residual = 0.0
        self.max_external_liquidity_violation = 0.0
        self.min_firm_active_share = 1.0
        self.max_structural_shock_shift = 0.0
        self._current_shock_modifier = np.ones((self.paths, self.n_nodes))

    def snapshot_state(self):
        state = super().snapshot_state()
        state.update({
            "v83_external_liquidity": self.external_liquidity.copy(),
            "v83_firm_active_share": self.firm_active_share.copy(),
            "v83_trade_cost": self.trade_cost.copy(),
            "v83_source_share": self.source_share.copy(),
            "v83_trade_resilience": self.trade_resilience.copy(),
            "v83_shock_modifier": self._current_shock_modifier.copy(),
            "v83_months": self.months_v83,
        })
        return state

    def restore_state(self, state):
        super().restore_state(state)
        if "v83_external_liquidity" in state:
            self.external_liquidity = state["v83_external_liquidity"].copy()
            self.firm_active_share = state["v83_firm_active_share"].copy()
            self.trade_cost = state["v83_trade_cost"].copy()
            self.source_share = state["v83_source_share"].copy()
            self.trade_resilience = state["v83_trade_resilience"].copy()
            self._current_shock_modifier = state["v83_shock_modifier"].copy()
            self.months_v83 = int(state["v83_months"])

    def _update_structural_trade(self, *, actor_damage, actor_logistics,
                                 actor_financial_stress, actor_payment,
                                 actor_panic, blockade_feedback):
        importer = np.arange(4)[None, :, None]
        exporter = np.arange(4)[None, None, :]
        bilateral_distance = np.array([
            [0.00, 0.24, 0.68, 0.46],
            [0.24, 0.00, 0.72, 0.39],
            [0.68, 0.72, 0.00, 0.52],
            [0.46, 0.39, 0.52, 0.00],
        ])[None, :, :]
        conflict_pair = ((importer == 0) & (exporter > 0)) | ((exporter == 0) & (importer > 0))
        blockade_wedge = blockade_feedback[:, None, None] * conflict_pair
        logistics_wedge = 0.42 * (1.0 - actor_logistics[:, None, :])
        finance_wedge = 0.24 * actor_financial_stress[:, :, None]
        damage_wedge = 0.30 * actor_damage[:, None, :]
        proposed_trade_cost = np.clip(
            1.0 + bilateral_distance
            + self.border_cost[None, :, None]
            + blockade_wedge + logistics_wedge + finance_wedge + damage_wedge,
            1.0, 4.5,
        )
        np.einsum("pii->pi", proposed_trade_cost)[:] = 1.0
        self.trade_cost = (
            proposed_trade_cost if self.structural_gravity_enabled
            else np.ones_like(proposed_trade_cost)
        )
        theta = self.theta_actor[None, :, None]
        technology = (
            self.technology_actor[None, None, :]
            * actor_logistics[:, None, :]
            * (1.0 - 0.55 * actor_damage[:, None, :])
        )
        ek_competitiveness = np.maximum(technology, 1.0e-4) ** theta
        ek_competitiveness *= self.trade_cost ** (-theta)
        proposed_source_share = ek_competitiveness / np.maximum(
            np.sum(ek_competitiveness, axis=2, keepdims=True), 1.0e-12
        )
        self.source_share = (
            proposed_source_share if self.structural_gravity_enabled
            else np.broadcast_to(np.eye(4)[None, :, :], proposed_source_share.shape).copy()
        )
        market_access = np.sum(self.source_share, axis=1)
        profit_signal = (
            0.46 * market_access + 0.24 * actor_payment
            + 0.18 * actor_logistics - 0.20 * actor_financial_stress
            - self.entry_cost[None, :] - self.exit_sensitivity[None, :] * actor_damage
        )
        active_target = 1.0 / (1.0 + np.exp(-4.0 * profit_signal))
        if self.melitz_enabled:
            self.firm_active_share += 0.08 * (active_target - self.firm_active_share)
            self.firm_active_share = np.clip(self.firm_active_share, 0.08, 0.98)
        foreign_share = 1.0 - self.source_share[:, np.arange(4), np.arange(4)]
        rerouting = np.clip(
            self.rerouting_capacity[None, :] * foreign_share
            * self.firm_active_share * self.external_liquidity,
            0.0, 0.85,
        )
        self.trade_resilience = np.clip(
            0.34 + 0.42 * rerouting + 0.18 * self.firm_active_share
            + 0.16 * actor_logistics - 0.16 * actor_panic,
            0.18, 1.08,
        )
        actor_modifier = np.clip(
            1.10 - 0.42 * self.trade_resilience
            + 0.20 * (1.0 - self.external_liquidity)
            + 0.16 * (1.0 - self.firm_active_share),
            0.58, 1.34,
        )
        if not (self.structural_gravity_enabled or self.melitz_enabled
                or self.external_liquidity_enabled):
            actor_modifier.fill(1.0)
        region_modifier = np.ones((self.paths, self.n_actors))
        region_modifier[:, :4] = actor_modifier
        region_modifier[:, 4:] = (
            1.0 + 0.24 * np.mean(actor_modifier - 1.0, axis=1, keepdims=True)
        )
        self._current_shock_modifier = region_modifier[:, self.actor_of_node]

    def _scenario_shock(self, month, blockade_feedback):
        shock, trade_loss = super()._scenario_shock(month, blockade_feedback)
        modified = np.clip(shock * self._current_shock_modifier, 0.0, 0.98)
        self.max_structural_shock_shift = max(
            self.max_structural_shock_shift,
            float(np.max(np.abs(modified - shock))),
        )
        actor_resilience = np.mean(self.trade_resilience, axis=1)
        return modified, np.clip(trade_loss * (1.12 - 0.44 * actor_resilience), 0.0, 0.92)

    def solve(self, **kwargs):
        self._update_structural_trade(
            actor_damage=kwargs["actor_damage"],
            actor_logistics=kwargs["actor_logistics"],
            actor_financial_stress=kwargs["actor_financial_stress"],
            actor_payment=kwargs["actor_payment"],
            actor_panic=kwargs["actor_panic"],
            blockade_feedback=kwargs["blockade_feedback"],
        )
        result = super().solve(**kwargs)
        liquidity_flow = (
            0.018 * result.trade_balance[:, :4]
            - 0.010 * kwargs["actor_financial_stress"]
            - 0.008 * kwargs["blockade_feedback"][:, None]
        )
        flow_buffer = 1.0 - 0.65 * self.capital_controls[None, :]
        if self.external_liquidity_enabled:
            self.external_liquidity = np.clip(
                0.985 * self.external_liquidity + 0.015
                + liquidity_flow * flow_buffer,
                0.05, 1.25,
            )
        self.months_v83 += 1
        self.max_source_share_residual = max(
            self.max_source_share_residual,
            float(np.max(np.abs(np.sum(self.source_share, axis=2) - 1.0))),
        )
        self.max_external_liquidity_violation = max(
            self.max_external_liquidity_violation,
            max(0.0, -float(np.min(self.external_liquidity)),
                float(np.max(self.external_liquidity)) - 1.25),
        )
        self.min_firm_active_share = min(
            self.min_firm_active_share, float(np.min(self.firm_active_share))
        )
        return result

    def structural_trade_diagnostics(self) -> dict:
        return {
            "months": self.months_v83,
            "structural_gravity_ek_enabled": self.structural_gravity_enabled,
            "melitz_enabled": self.melitz_enabled,
            "external_liquidity_enabled": self.external_liquidity_enabled,
            "maximum_source_share_residual": self.max_source_share_residual,
            "maximum_external_liquidity_bound_violation": self.max_external_liquidity_violation,
            "minimum_firm_active_share": self.min_firm_active_share,
            "maximum_structural_shock_shift": self.max_structural_shock_shift,
            "mean_trade_resilience_by_actor": np.mean(self.trade_resilience, axis=0).tolist(),
            "mean_external_liquidity_by_actor": np.mean(self.external_liquidity, axis=0).tolist(),
            "state_owners": {
                "bilateral_trade_cost_and_source_share": "StructuralInternationalCGEV83",
                "exporting_firm_active_share": "StructuralInternationalCGEV83",
                "external_liquidity_buffer": "StructuralInternationalCGEV83",
                "trade_balance": "DynamicMultiRegionalCGE",
                "inventories_and_contracts": "MRRDCGEDQSFCState"
            },
            "terminal_probability_adjustment": False,
            "data_ledger": str(LEDGER),
        }

    def extension_diagnostics(self) -> dict:
        parent = (
            super().extension_diagnostics()
            if hasattr(super(), "extension_diagnostics") else None
        )
        result = dict(parent or {})
        result["v83_structural_international_economics"] = (
            self.structural_trade_diagnostics()
        )
        return result
