from __future__ import annotations

import numpy as np

from continuous_multipop_mfg_v47 import (
    ACTION_CHANNELS,
    ACTORS,
    GRAPHON_LAYERS,
    ContinuousMultiPopulationMFG,
)
from optimal_transport_geometry_v62 import DynamicCausalOptimalTransportGeometry


class ContinuousMultiPopulationMFGV62(ContinuousMultiPopulationMFG):
    """V6.2 MFG with pathwise FGW and lagged joint state-action WFR."""

    def __init__(self, *args, **kwargs):
        paths, seed, initial_support = args[:3]
        super().__init__(*args, **kwargs)
        if self.partial_observation_geometry:
            self.belief_system = DynamicCausalOptimalTransportGeometry(
                paths, seed + 71_003, initial_support,
                self.actor_kernel, self.age_kernel,
                ambiguity_radius=kwargs.get("information_ambiguity_radius", 0.025),
                belief_filter_enabled=kwargs.get("belief_filter_enabled", True),
                credibility_hmm_enabled=kwargs.get("credibility_hmm_enabled", True),
                schrodinger_bridge_enabled=kwargs.get("schrodinger_bridge_enabled", True),
                fgw_alignment_enabled=kwargs.get("fgw_alignment_enabled", True),
                robust_information_enabled=kwargs.get("robust_information_enabled", True),
                information_value_enabled=kwargs.get("information_value_enabled", True),
            )
        self.max_joint_action_mass_residual = 0.0
        self.max_joint_action_reaction = 0.0

    def _effective_graphon(self, *args, fgw_alignment=None, **kwargs):
        if fgw_alignment is None or np.asarray(fgw_alignment).ndim == 2:
            return super()._effective_graphon(
                *args, fgw_alignment=fgw_alignment, **kwargs
            )
        alignment = np.asarray(fgw_alignment)
        saved = self.path_layer_kernels
        self.path_layer_kernels = saved * alignment[:, None, :, :]
        try:
            return super()._effective_graphon(
                *args, fgw_alignment=None, **kwargs
            )
        finally:
            self.path_layer_kernels = saved

    def _reaction_fitness(self, common_noise: np.ndarray,
                          jump_signal: np.ndarray,
                          network: np.ndarray) -> np.ndarray:
        base = super()._reaction_fitness(common_noise, jump_signal, network)
        action = np.clip(self.previous_joint_action, 0.0, 1.0)
        weights = np.array([0.30, 0.16, -0.24, -0.18, 0.22])
        joint_signal = np.einsum("pahc,c->pah", action, weights)
        x = self.x[None, None, None, :]
        reaction = 0.11 * joint_signal[..., None] * x
        self.max_joint_action_reaction = max(
            self.max_joint_action_reaction, float(np.max(np.abs(reaction)))
        )
        total = np.sum(action, axis=-1)
        self.max_joint_action_mass_residual = max(
            self.max_joint_action_mass_residual,
            float(np.max(np.maximum(total - len(ACTION_CHANNELS), 0.0))),
        )
        return np.clip(base + reaction, -0.80, 0.55)

    def diagnostics(self) -> dict:
        result = super().diagnostics()
        result["v62_joint_state_action_wfr"] = {
            "enabled": True,
            "action_channels": list(ACTION_CHANNELS),
            "state_support_points": len(self.x),
            "coupling_clock": "one-month-lagged action distribution enters WFR reaction",
            "max_joint_action_mass_residual": self.max_joint_action_mass_residual,
            "max_joint_action_reaction": self.max_joint_action_reaction,
            "hard_capacity_network_flow_owner": "specialized network-flow modules",
        }
        return result
