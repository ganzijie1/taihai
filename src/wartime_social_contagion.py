from __future__ import annotations

import numpy as np


ACTORS = ("china", "taiwan", "united_states", "japan")
AGES = ("18_29", "30_49", "50_64", "65_plus")


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -35.0, 35.0)))


def _logit(x):
    x = np.clip(x, 1e-6, 1.0 - 1e-6)
    return np.log(x / (1.0 - x))


class WartimeSocialContagionSystem:
    """Age-stratified competing contagion coupled to one campaign path ensemble.

    State order is S, E_m, I_m, E_f, I_f, R. The class advances four weekly
    substeps for every monthly campaign step. All event intensities use current
    path states; no future campaign information is read.
    """

    def __init__(self, paths, seed, initial_support, features=None):
        self.paths = paths
        self.rng = np.random.default_rng(seed)
        self.features = {
            "levy": True,
            "hawkes": True,
            "age_structure": True,
            "complex_threshold": True,
        }
        if features:
            self.features.update(features)
        shape = (paths, 4, 4)
        support = np.clip(initial_support, 0.01, 0.99)
        self.state = np.zeros(shape + (6,))
        self.state[..., 1] = 0.018
        self.state[..., 2] = (0.045 + 0.085 * support)[:, :, None]
        self.state[..., 3] = 0.015
        self.state[..., 4] = (0.020 + 0.050 * (1.0 - support))[:, :, None]
        self.state[..., 5] = 0.030
        self.state[..., 0] = 1.0 - self.state[..., 1:].sum(axis=-1)
        self.h_m = np.zeros((paths, 4))
        self.h_f = np.zeros((paths, 4))
        self.age_weights = np.array([0.18, 0.36, 0.27, 0.19])
        self.last_net = self._weighted(self.state[..., 2] - self.state[..., 4])
        self.max_conservation_error = 0.0
        self.event_counts_m = np.zeros((paths, 4))
        self.event_counts_f = np.zeros((paths, 4))
        self.random_batches_generated = 0

        self.contact = np.array([
            [0.52, 0.31, 0.12, 0.05],
            [0.18, 0.48, 0.25, 0.09],
            [0.08, 0.28, 0.46, 0.18],
            [0.04, 0.14, 0.29, 0.53],
        ])
        if not self.features["age_structure"]:
            self.contact[:] = 0.25
        self.actor_kernel = np.array([
            [1.00, 0.22, 0.12, 0.12],
            [0.26, 1.00, 0.38, 0.30],
            [0.10, 0.30, 1.00, 0.42],
            [0.10, 0.28, 0.45, 1.00],
        ])

    def _weighted(self, values):
        return np.sum(values * self.age_weights[None, None, :], axis=2)

    def _flow(self, source, hazard, sigma=0.07, noise=None):
        probability = 1.0 - np.exp(-np.clip(hazard, 0.0, 6.0))
        if noise is None:
            noise = self.rng.lognormal(-0.5 * sigma * sigma, sigma, source.shape)
        return np.minimum(source, source * probability * noise)

    def update(
        self, *, prior_support, actor_damage, actor_shortage, actor_inflation,
        actor_backlash, actor_network, actor_norm, actor_resilience,
        progress_signal, command_integrity, conflict_intensity,
        mfg_threshold_shift=None, mfg_mobilization=None, mfg_fatigue=None,
    ):
        age_online = np.array([1.25, 1.08, 0.86, 0.62])[None, None, :]
        age_broadcast = np.array([0.70, 0.88, 1.08, 1.28])[None, None, :]
        theta_m = np.array([0.17, 0.16, 0.19, 0.23])[None, None, :]
        theta_f = np.array([0.15, 0.14, 0.17, 0.21])[None, None, :]
        if not self.features["age_structure"]:
            theta_m[:] = theta_m.mean()
            theta_f[:] = theta_f.mean()
        if mfg_threshold_shift is not None:
            threshold_shift = np.clip(mfg_threshold_shift, -0.18, 0.18)
            theta_m = np.clip(theta_m + threshold_shift, 0.04, 0.42)
            theta_f = np.clip(theta_f - 0.65 * threshold_shift, 0.04, 0.42)
        beta_m = np.array([0.22, 0.21, 0.16, 0.18])[None, :, None]
        beta_f = np.array([0.14, 0.19, 0.13, 0.14])[None, :, None]
        fact_check = np.array([0.30, 0.20, 0.38, 0.34])[None, :]
        intensity = np.clip(conflict_intensity, 0.0, 1.0)[:, None]

        # Generate one monthly batch on the path axis.  This preserves the
        # stochastic law while avoiding dozens of small RNG dispatches inside
        # the four weekly substeps.
        event_uniform_m = self.rng.random((4, self.paths, 4))
        event_uniform_f = self.rng.random((4, self.paths, 4))
        common_uniform = self.rng.random((4, self.paths))
        if self.features["levy"]:
            pareto_m = self.rng.pareto(2.8, (4, self.paths, 4))
            pareto_f = self.rng.pareto(2.55, (4, self.paths, 4))
            pareto_common = self.rng.pareto(2.8, (4, self.paths))
        else:
            pareto_m = pareto_f = pareto_common = None
        flow_sigmas = np.array([0.07, 0.07, 0.05, 0.05, 0.04, 0.04, 0.03])
        flow_standard_normal = self.rng.standard_normal(
            (4, 7, self.paths, 4, 4)
        )
        flow_noise = np.exp(
            -0.5 * flow_sigmas[None, :, None, None, None] ** 2
            + flow_sigmas[None, :, None, None, None] * flow_standard_normal
        )
        self.random_batches_generated += 1

        for substep in range(4):
            im = self.state[..., 2]
            iff = self.state[..., 4]
            network_m = np.einsum("pah,gh->pag", im, self.contact)
            network_f = np.einsum("pah,gh->pag", iff, self.contact)

            p_m = np.clip(
                0.002 + intensity * (
                    0.010 * np.clip(progress_signal, 0.0, 1.0)
                    + 0.007 * actor_backlash + 0.003 * actor_norm
                ), 0.0, 0.16,
            )
            p_f = np.clip(
                0.0025 + intensity * (
                    0.013 * actor_damage + 0.010 * actor_shortage
                    + 0.007 * np.clip(actor_inflation, 0.0, 1.0)
                    + 0.009 * np.clip(-progress_signal, 0.0, 1.0)
                    + 0.006 * (1.0 - command_integrity)
                ), 0.0, 0.18,
            )
            event_m = event_uniform_m[substep] < p_m
            event_f = event_uniform_f[substep] < p_f
            common_probability = np.clip(
                0.001 + 0.004 * np.max(actor_damage + actor_shortage, axis=1), 0.0, 0.025
            )
            common_event = common_uniform[substep] < common_probability
            if self.features["levy"]:
                mark_m = np.minimum(0.65, 0.040 * (1.0 + pareto_m[substep]))
                mark_f = np.minimum(0.75, 0.048 * (1.0 + pareto_f[substep]))
                common_mark = np.minimum(
                    0.55, 0.035 * (1.0 + pareto_common[substep])
                )
            else:
                mark_m = np.full((self.paths, 4), 0.063)
                mark_f = np.full((self.paths, 4), 0.079)
                common_mark = np.full(self.paths, 0.056)
            exo_m = event_m * mark_m + common_event[:, None] * common_mark[:, None] * 0.40
            exo_f = event_f * mark_f + common_event[:, None] * common_mark[:, None] * 0.65
            self.event_counts_m += event_m
            self.event_counts_f += event_f

            cross_m = exo_m @ self.actor_kernel.T
            cross_f = exo_f @ self.actor_kernel.T
            if self.features["hawkes"]:
                endogenous_m = 0.18 * self._weighted(im)
                endogenous_f = 0.21 * self._weighted(iff)
            else:
                endogenous_m = 0.0
                endogenous_f = 0.0
            self.h_m = 0.70 * self.h_m + cross_m + endogenous_m
            self.h_f = 0.66 * self.h_f + cross_f * (1.0 - fact_check) + endogenous_f

            broadcast_m = self.h_m[:, :, None] * (0.58 * age_online + 0.42 * age_broadcast)
            broadcast_f = self.h_f[:, :, None] * (0.64 * age_online + 0.36 * age_broadcast)
            exposure_m = network_m * actor_network[:, :, None] + broadcast_m
            exposure_f = network_f * actor_network[:, :, None] + broadcast_f
            if self.features["complex_threshold"]:
                response_m = np.maximum(_sigmoid(8.0 * (exposure_m - theta_m)) - _sigmoid(-8.0 * theta_m), 0.0)
                response_f = np.maximum(_sigmoid(8.4 * (exposure_f - theta_f)) - _sigmoid(-8.4 * theta_f), 0.0)
            else:
                response_m = np.clip(exposure_m, 0.0, 1.0)
                response_f = np.clip(exposure_f, 0.0, 1.0)

            hazard_m = beta_m * response_m * (
                0.60 + 0.40 * actor_norm[:, :, None]
            ) * np.clip(1.0 - 0.45 * iff, 0.45, 1.0)
            if mfg_mobilization is not None:
                hazard_m += 0.018 * np.clip(mfg_mobilization, 0.0, 1.0)[:, :, None]
            fatigue_pressure = (
                0.018 * actor_damage + 0.014 * actor_shortage
                + 0.010 * np.clip(actor_inflation, 0.0, 1.0)
                + 0.010 * (1.0 - command_integrity)
            )[:, :, None]
            hazard_f = (beta_f * response_f + fatigue_pressure) * np.clip(1.0 - 0.30 * im, 0.50, 1.0)
            if mfg_fatigue is not None:
                hazard_f += 0.018 * np.clip(mfg_fatigue, 0.0, 1.0)[:, :, None]

            susceptible = self.state[..., 0]
            flow_m = self._flow(
                susceptible, hazard_m, noise=flow_noise[substep, 0]
            )
            flow_f = self._flow(
                susceptible, hazard_f, noise=flow_noise[substep, 1]
            )
            scale = np.minimum(1.0, susceptible / np.maximum(flow_m + flow_f, 1e-12))
            flow_m *= scale
            flow_f *= scale
            em_to_im = self._flow(
                self.state[..., 1], 0.34, 0.05, flow_noise[substep, 2]
            )
            ef_to_if = self._flow(
                self.state[..., 3], 0.31, 0.05, flow_noise[substep, 3]
            )
            im_to_r = self._flow(
                self.state[..., 2], 0.035 + 0.070 * iff, 0.04,
                flow_noise[substep, 4],
            )
            if_to_r = self._flow(
                self.state[..., 4], 0.045 + 0.055 * im, 0.04,
                flow_noise[substep, 5],
            )
            r_to_s = self._flow(
                self.state[..., 5], 0.025, 0.03, flow_noise[substep, 6]
            )

            self.state[..., 0] += -flow_m - flow_f + r_to_s
            self.state[..., 1] += flow_m - em_to_im
            self.state[..., 2] += em_to_im - im_to_r
            self.state[..., 3] += flow_f - ef_to_if
            self.state[..., 4] += ef_to_if - if_to_r
            self.state[..., 5] += im_to_r + if_to_r - r_to_s
            self.state = np.clip(self.state, 0.0, 1.0)
            error = float(np.max(np.abs(self.state.sum(axis=-1) - 1.0)))
            self.max_conservation_error = max(self.max_conservation_error, error)

        mobilization = self._weighted(self.state[..., 2])
        fatigue = self._weighted(self.state[..., 4])
        net = mobilization - fatigue
        delta_net = net - self.last_net
        support = _sigmoid(_logit(prior_support) + 1.65 * delta_net + 0.025 * net)
        collective_action = np.clip(
            0.58 * fatigue + 0.22 * actor_shortage + 0.16 * actor_damage
            + 0.10 * np.clip(-progress_signal, 0.0, 1.0)
            - 0.20 * actor_resilience, 0.0, 1.0,
        )
        panic = np.clip(
            0.18 * fatigue + 0.12 * actor_damage + 0.08 * actor_shortage
            + 0.05 * self.h_f - 0.06 * mobilization, 0.0, 1.0,
        )
        norm = np.clip(
            actor_norm + 0.10 * mobilization - 0.13 * fatigue
            + 0.025 * np.clip(progress_signal, -1.0, 1.0), 0.0, 1.0,
        )
        self.last_net = net
        return {
            "support": support,
            "mobilization": mobilization,
            "fatigue": fatigue,
            "collective_action": collective_action,
            "panic": panic,
            "norm": norm,
            "conservation_error": self.max_conservation_error,
        }

    def summary(self):
        mobilization = self._weighted(self.state[..., 2])
        fatigue = self._weighted(self.state[..., 4])
        return {
            "features": self.features,
            "random_batching": {
                "weekly_substeps_per_batch": 4,
                "monthly_batches_generated": self.random_batches_generated,
            },
            "max_population_conservation_error": self.max_conservation_error,
            "terminal_mobilization_p10_p50_p90": {
                actor: [float(x) for x in np.quantile(mobilization[:, i], (0.1, 0.5, 0.9))]
                for i, actor in enumerate(ACTORS)
            },
            "terminal_fatigue_p10_p50_p90": {
                actor: [float(x) for x in np.quantile(fatigue[:, i], (0.1, 0.5, 0.9))]
                for i, actor in enumerate(ACTORS)
            },
            "mean_event_counts": {
                actor: {
                    "mobilization": float(np.mean(self.event_counts_m[:, i])),
                    "fatigue": float(np.mean(self.event_counts_f[:, i])),
                }
                for i, actor in enumerate(ACTORS)
            },
        }
