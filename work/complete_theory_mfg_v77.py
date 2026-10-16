from __future__ import annotations

import numpy as np

from complete_information_geometry_v77 import CompleteInformationGeometryV77
from complete_network_geometry_v77 import CompleteNetworkGeometryV77
from complete_online_tda_v77 import CompleteOnlineTDAV77
from coupled_oscillator_v77 import CoupledOscillatorV77
from scf_signed_potential_mfg_v76 import SignedPotentialHypergraphMFGV76


class CompleteTheoryMFGV77(SignedPotentialHypergraphMFGV76):
    """V7.7 closure for every useful formerly theory-only mechanism."""

    def __init__(self, *args, **kwargs):
        paths, seed, initial_support = args[:3]
        super().__init__(*args, **kwargs)
        if self.partial_observation_geometry:
            self.belief_system = CompleteInformationGeometryV77(
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
        if self.network_geometry is not None:
            self.network_geometry = CompleteNetworkGeometryV77(
                self.network_geometry.raw_kernel,
                spectral_interval=kwargs.get("network_geometry_spectral_interval", 12),
                intensity_transition_enabled=kwargs.get("intensity_transition_enabled", True),
                supra_laplacian_enabled=kwargs.get("supra_laplacian_enabled", True),
                spectral_geometry_enabled=kwargs.get("spectral_geometry_enabled", True),
                forman_curvature_enabled=kwargs.get("forman_curvature_enabled", True),
                sheaf_laplacian_enabled=kwargs.get("sheaf_laplacian_enabled", True),
            )
        self.online_tda = CompleteOnlineTDAV77(paths, interval=12)
        self.oscillator = CoupledOscillatorV77(paths, seed + 97_001)
        self.maximum_oscillator_kernel_change = 0.0
        self.minimum_oscillator_intensity_factor = 1.0

    def _update_relation_memory(self, actor_mean, damage, shortage, network, jump_signal):
        if hasattr(self, "oscillator"):
            jump_signal = np.clip(
                jump_signal + 0.25 * self.oscillator.last_desynchronization[:, None],
                0.0, 1.5,
            )
        return super()._update_relation_memory(
            actor_mean, damage, shortage, network, jump_signal
        )

    def _effective_graphon(self, *args, **kwargs):
        kernel, weights, intensity = super()._effective_graphon(*args, **kwargs)
        damage, shortage, network, common_noise, jump_signal = args[:5]
        factors = self.oscillator.update(
            magnitude=kernel,
            sign=self.current_sign,
            layer_weights=weights,
            damage=np.asarray(damage),
            shortage=np.asarray(shortage),
            network=np.asarray(network),
            jump_signal=np.asarray(jump_signal),
            common_noise=np.asarray(common_noise),
        )
        phase_weighted = kernel * factors.phase_compatibility
        row_mass = np.sum(phase_weighted, axis=3, keepdims=True)
        adjusted = phase_weighted / np.maximum(row_mass, 1.0e-12)
        strength = np.mean(factors.phase_compatibility, axis=3)
        intensity = intensity * np.clip(strength, 0.55, 1.0)
        self.maximum_oscillator_kernel_change = max(
            self.maximum_oscillator_kernel_change,
            float(np.max(np.abs(adjusted - kernel))),
        )
        self.minimum_oscillator_intensity_factor = min(
            self.minimum_oscillator_intensity_factor, float(np.min(strength))
        )
        return adjusted, weights, intensity

    def diagnostics(self) -> dict:
        payload = super().diagnostics()
        payload["v77_coupled_oscillator"] = self.oscillator.diagnostics() | {
            "maximum_graphon_kernel_change": self.maximum_oscillator_kernel_change,
            "minimum_graphon_intensity_factor": self.minimum_oscillator_intensity_factor,
            "bidirectional_feedback": (
                "signed Graphon/stress drive phase-frequency dynamics; phase compatibility "
                "changes current MFG fields; lagged desynchronization changes relation memory"
            ),
        }
        return payload
