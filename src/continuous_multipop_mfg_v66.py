from __future__ import annotations

from continuous_multipop_mfg_v63 import ContinuousMultiPopulationMFGV63
from deep_optimal_transport_v66 import DeepCausalOptimalTransportGeometryV66


class ContinuousMultiPopulationMFGV66(ContinuousMultiPopulationMFGV63):
    """V6.3 stability diagnostics plus V6.6 path-conditioned OT geometry."""

    def __init__(self, *args, **kwargs):
        paths, seed, initial_support = args[:3]
        super().__init__(*args, **kwargs)
        if self.partial_observation_geometry:
            self.belief_system = DeepCausalOptimalTransportGeometryV66(
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
