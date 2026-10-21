from __future__ import annotations

import os

import numpy as np

from differential_manifold_v81 import DifferentialManifoldConstraintV81
from statistical_field_mfg_v80 import StatisticalFieldMFGV80


class DifferentialManifoldMFGV81(StatisticalFieldMFGV80):
    """V8.0 with an intrinsic probability and hybrid-chart constraint layer."""

    def __init__(self, *args, **kwargs) -> None:
        enabled = lambda name: os.environ.get(name, "0") != "1"
        simplex_enabled = kwargs.pop(
            "manifold_simplex_enabled", enabled("TAIWAN_V81_DISABLE_SIMPLEX")
        )
        spd_enabled = kwargs.pop(
            "manifold_spd_enabled", enabled("TAIWAN_V81_DISABLE_SPD")
        )
        slow_enabled = kwargs.pop(
            "manifold_slow_enabled", enabled("TAIWAN_V81_DISABLE_SLOW")
        )
        hybrid_enabled = kwargs.pop(
            "manifold_hybrid_enabled", enabled("TAIWAN_V81_DISABLE_HYBRID")
        )
        basin_enabled = kwargs.pop(
            "manifold_basin_enabled", enabled("TAIWAN_V81_DISABLE_BASIN")
        )
        paths, _seed, initial_support = args[:3]
        super().__init__(*args, **kwargs)
        self.manifold = DifferentialManifoldConstraintV81(
            paths,
            initial_support,
            simplex_enabled=simplex_enabled,
            spd_enabled=spd_enabled,
            slow_manifold_enabled=slow_enabled,
            hybrid_chart_enabled=hybrid_enabled,
            basin_diagnostics_enabled=basin_enabled,
        )
        self.maximum_iteration_budget = 0
        self.maximum_support_output_shift = 0.0
        self.manifold_feedback_updates = 0

    def update(self, **kwargs):
        step = self.manifold.prepare(kwargs)
        modified = dict(kwargs)
        modified["max_fixed_point_iterations"] = max(
            int(kwargs.get("max_fixed_point_iterations", 48)),
            step.iteration_budget,
        )
        self.maximum_iteration_budget = max(
            self.maximum_iteration_budget,
            modified["max_fixed_point_iterations"],
        )
        factors = super().update(**modified)
        raw_support = factors.support.copy()
        factors.support = np.clip(
            self.manifold.retract_support(raw_support), 1.0e-8, 1.0 - 1.0e-8
        )
        self.maximum_support_output_shift = max(
            self.maximum_support_output_shift,
            float(np.max(np.abs(factors.support - raw_support))),
        )
        self.manifold_feedback_updates += 1
        return factors

    def diagnostics(self) -> dict:
        payload = super().diagnostics()
        payload["v81_differential_manifold"] = self.manifold.diagnostics() | {
            "feedback_updates": self.manifold_feedback_updates,
            "maximum_mfg_iteration_budget": self.maximum_iteration_budget,
            "maximum_support_output_shift": self.maximum_support_output_shift,
            "same_path_consumers": [
                "continuous MFG support output",
                "outer social support state",
                "next-month cognition and political fragility",
            ],
            "explicit_event_clock": "monthly; no future information",
        }
        return payload
