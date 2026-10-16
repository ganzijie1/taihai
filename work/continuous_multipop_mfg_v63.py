from __future__ import annotations

import numpy as np

from continuous_multipop_mfg_v62 import ContinuousMultiPopulationMFGV62
from stability_dynamics_v63 import OnlineSystemStabilityV63


class ContinuousMultiPopulationMFGV63(ContinuousMultiPopulationMFGV62):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stability_v63 = OnlineSystemStabilityV63(self.paths)

    def update(self, **kwargs):
        state = np.stack((
            kwargs["progress"], kwargs["damage"], kwargs["shortage"],
            kwargs["inflation"], kwargs["financial_stress"], kwargs["command"],
            kwargs["network"], kwargs["contagion_mobilization"],
            kwargs["contagion_fatigue"],
        ), axis=2)
        self.stability_v63.update(state, kwargs.get("jump_signal", np.zeros_like(kwargs["damage"])))
        return super().update(**kwargs)

    def diagnostics(self) -> dict:
        result = super().diagnostics()
        result["v63_system_stability"] = self.stability_v63.diagnostics()
        return result
