from __future__ import annotations

import numpy as np

from continuous_multipop_mfg_v47 import ACTION_CHANNELS
from particle_international_economics_v83 import ParticleValidatedMFGV83
from statistical_contact_v84 import StatisticalContactSystemV84


class StatisticalContactMFGV84(ParticleValidatedMFGV83):
    """V8.3 MFG plus causal statistical observation and contact dissipation."""

    def __init__(self, *args, **kwargs):
        enabled = {
            name: kwargs.pop(f"v84_{name}_enabled", True)
            for name in (
                "nonparametric", "multivariate", "time_series", "wavelet", "contact",
                "wavelet_temporal", "wavelet_coherence", "wavelet_graph", "wavelet_scattering",
                "contact_military", "contact_economic", "contact_financial", "contact_social",
            )
        }
        paths = args[0]
        super().__init__(*args, **kwargs)
        self.v84 = StatisticalContactSystemV84(
            paths,
            self.actor_kernel,
            nonparametric_enabled=enabled["nonparametric"],
            multivariate_enabled=enabled["multivariate"],
            time_series_enabled=enabled["time_series"],
            wavelet_enabled=enabled["wavelet"],
            contact_enabled=enabled["contact"],
            wavelet_temporal_enabled=enabled["wavelet_temporal"],
            wavelet_coherence_enabled=enabled["wavelet_coherence"],
            wavelet_graph_enabled=enabled["wavelet_graph"],
            wavelet_scattering_enabled=enabled["wavelet_scattering"],
            contact_military_enabled=enabled["contact_military"],
            contact_economic_enabled=enabled["contact_economic"],
            contact_financial_enabled=enabled["contact_financial"],
            contact_social_enabled=enabled["contact_social"],
        )
        self.maximum_input_shift = 0.0
        self.maximum_outer_shift = 0.0

    def update(self, **kwargs):
        state = self.v84.update(**kwargs)
        modified = dict(kwargs)
        latent_stress = np.clip(0.5 + 0.18 * state.latent_factor, 0.0, 1.0)
        multiscale_stress = np.clip(
            0.45 * state.wavelet_energy
            + 0.30 * state.wavelet_coherence
            + 0.25 * np.maximum(state.graph_wavelet_anomaly, 0.0),
            0.0, 1.0,
        )
        modified["damage"] = np.clip(
            kwargs["damage"]
            + 0.010 * state.contact_military_loss
            + 0.006 * state.change_probability,
            0.0, 1.5,
        )
        modified["shortage"] = np.clip(
            kwargs["shortage"]
            + 0.012 * state.contact_economic_loss
            + 0.006 * multiscale_stress,
            0.0, 1.5,
        )
        modified["financial_stress"] = np.clip(
            kwargs["financial_stress"]
            + 0.012 * state.contact_financial_loss
            + 0.006 * state.volatility,
            0.0, 1.5,
        )
        modified["command"] = np.clip(
            kwargs["command"]
            * (1.0 - 0.010 * state.contact_military_loss
               - 0.004 * state.change_probability),
            0.0, 1.0,
        )
        modified["network"] = np.clip(
            kwargs["network"]
            * (1.0 - 0.006 * np.abs(state.graph_wavelet_anomaly)
               - 0.008 * state.contact_social_loss),
            0.0, 1.0,
        )
        modified["contagion_fatigue"] = np.clip(
            kwargs["contagion_fatigue"]
            + 0.010 * state.contact_social_loss
            + 0.006 * state.regime_probability,
            0.0, 1.0,
        )
        modified["common_noise"] = np.clip(
            kwargs.get("common_noise", np.zeros_like(kwargs["damage"]))
            + 0.020 * state.graph_wavelet_anomaly,
            -1.0, 1.0,
        )
        self.maximum_input_shift = max(
            self.maximum_input_shift,
            float(max(
                np.max(np.abs(modified["damage"] - kwargs["damage"])),
                np.max(np.abs(modified["shortage"] - kwargs["shortage"])),
                np.max(np.abs(modified["financial_stress"] - kwargs["financial_stress"])),
                np.max(np.abs(modified["command"] - kwargs["command"])),
            )),
        )
        factors = super().update(**modified)
        factors.statistical_latent_factor = state.latent_factor.copy()
        factors.statistical_regime_probability = state.regime_probability.copy()
        factors.statistical_volatility = state.volatility.copy()
        factors.statistical_change_probability = state.change_probability.copy()
        factors.wavelet_energy = state.wavelet_energy.copy()
        factors.wavelet_coherence = state.wavelet_coherence.copy()
        factors.graph_wavelet_anomaly = state.graph_wavelet_anomaly.copy()
        factors.contact_military_loss = state.contact_military_loss.copy()
        factors.contact_economic_loss = state.contact_economic_loss.copy()
        factors.contact_financial_loss = state.contact_financial_loss.copy()
        factors.contact_social_loss = state.contact_social_loss.copy()
        factors.contact_recovery_need = state.contact_recovery_need.copy()

        # Current-month outer consumers receive bounded factors. They modify
        # owned stocks in the master loop; this wrapper owns no duplicate stock.
        factors.cascade_social_loss = np.clip(
            factors.cascade_social_loss
            + 0.055 * state.contact_social_loss
            + 0.025 * state.regime_probability,
            0.0, 1.0,
        )
        factors.cascade_financial_loss = np.clip(
            factors.cascade_financial_loss
            + 0.055 * state.contact_financial_loss
            + 0.020 * state.volatility,
            0.0, 1.0,
        )
        factors.cascade_supply_loss = np.clip(
            factors.cascade_supply_loss
            + 0.055 * state.contact_economic_loss
            + 0.020 * multiscale_stress,
            0.0, 1.0,
        )
        factors.cascade_command_loss = np.clip(
            factors.cascade_command_loss
            + 0.055 * state.contact_military_loss
            + 0.020 * state.change_probability,
            0.0, 1.0,
        )
        factors.cascade_recovery_need = np.clip(
            factors.cascade_recovery_need + 0.08 * state.contact_recovery_need,
            0.0, 1.0,
        )
        factors.threshold_shift = np.clip(
            factors.threshold_shift
            + 0.012 * (latent_stress - 0.5)[:, :, None]
            + 0.010 * state.regime_probability[:, :, None],
            -0.18, 0.18,
        )
        if factors.joint_action_field is not None:
            factors.joint_action_field[..., ACTION_CHANNELS.index("hoarding")] = np.clip(
                factors.joint_action_field[..., ACTION_CHANNELS.index("hoarding")]
                + 0.015 * multiscale_stress,
                0.0, 1.0,
            )
            factors.joint_action_field[..., ACTION_CHANNELS.index("service")] = np.clip(
                factors.joint_action_field[..., ACTION_CHANNELS.index("service")]
                - 0.012 * state.contact_recovery_need,
                0.0, 1.0,
            )
        outer = max(
            np.max(state.contact_military_loss), np.max(state.contact_economic_loss),
            np.max(state.contact_financial_loss), np.max(state.contact_social_loss),
        )
        self.maximum_outer_shift = max(self.maximum_outer_shift, float(0.055 * outer))
        self.v84.max_feedback = max(self.v84.max_feedback, self.maximum_input_shift,
                                    self.maximum_outer_shift)
        return factors

    def diagnostics(self) -> dict:
        payload = super().diagnostics()
        payload["v84_statistical_multiscale_contact"] = self.v84.diagnostics() | {
            "maximum_mfg_input_shift": self.maximum_input_shift,
            "maximum_current_outer_factor_shift": self.maximum_outer_shift,
            "same_path_consumers": [
                "continuous MFG", "social contagion", "DEDS next-month hazards",
                "financial and firm stress", "CGE/DSGE current-month inputs",
                "logistics and command", "competing terminal risks through owned states",
            ],
            "explicit_clock": "monthly causal observation; current outer feedback; no future information",
        }
        return payload
