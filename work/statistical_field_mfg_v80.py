from __future__ import annotations

import os

import numpy as np

from analytical_mechanics_v80 import AnalyticalMechanicsV80
from continuous_multipop_mfg_v47 import ACTION_CHANNELS
from information_criticality_mfg_v79 import InformationCriticalityMFGV79
from statistical_field_soc_v80 import (
    NonEquilibriumStatisticalFieldV80,
    QuasicriticalCascadeV80,
)


class StatisticalFieldMFGV80(InformationCriticalityMFGV79):
    """V7.9 plus a lagged, non-equilibrium four-domain residual field.

    The field is deliberately residual: it does not replace the social,
    financial, supply, command, MFG, CGE or DSGE state owners.  Month-t field
    estimates affect the outer state consumers in month t and MFG optimisation
    only in month t+1.
    """

    def __init__(self, *args, **kwargs) -> None:
        self.statistical_field_enabled = kwargs.pop(
            "statistical_field_enabled",
            os.environ.get("TAIWAN_V81_DISABLE_STATISTICAL_FIELD", "0") != "1",
        )
        self.analytical_mechanics_enabled = kwargs.pop(
            "analytical_mechanics_enabled",
            os.environ.get("TAIWAN_V81_DISABLE_ANALYTICAL_MECHANICS", "0") != "1",
        )
        paths, seed, _initial_support = args[:3]
        super().__init__(*args, **kwargs)
        self.cascade = QuasicriticalCascadeV80(paths, self.actor_kernel)
        self.statistical_field = NonEquilibriumStatisticalFieldV80(
            paths, self.actor_kernel, self.age_weights, seed + 80_003
        )
        self.analytical_mechanics = AnalyticalMechanicsV80(paths)
        self.previous_field_feedback = np.zeros((paths, 4, 4))
        self.maximum_lagged_field_input_shift = 0.0
        self.maximum_current_outer_field_contribution = 0.0
        self.field_feedback_updates = 0

    def _mfg_mean(self) -> np.ndarray:
        return np.sum(
            self.density * self.x[None, None, None, :], axis=-1
        ) * self.dx

    def update(self, **kwargs):
        lagged = (
            self.previous_field_feedback
            if self.statistical_field_enabled
            else np.zeros_like(self.previous_field_feedback)
        )
        if self.analytical_mechanics_enabled:
            mechanics = self.analytical_mechanics.update(**kwargs)
        else:
            zero = np.zeros_like(kwargs["damage"])
            mechanics = type("DisabledMechanicsFactors", (), {
                "logistics": zero,
                "financial": zero,
                "command": zero,
                "social": zero,
            })()
        modified = dict(kwargs)
        modified["damage"] = np.clip(
            kwargs["damage"] + 0.025 * lagged[..., 0]
            + 0.020 * lagged[..., 3] + 0.012 * mechanics.social
            + 0.010 * mechanics.command,
            0.0, 1.5,
        )
        modified["shortage"] = np.clip(
            kwargs["shortage"] + 0.040 * lagged[..., 2]
            + 0.018 * mechanics.logistics, 0.0, 1.5
        )
        modified["financial_stress"] = np.clip(
            kwargs["financial_stress"] + 0.040 * lagged[..., 1], 0.0, 1.5
        )
        modified["financial_stress"] = np.clip(
            modified["financial_stress"] + 0.016 * mechanics.financial,
            0.0, 1.5,
        )
        modified["command"] = np.clip(
            kwargs["command"]
            * (1.0 - 0.030 * lagged[..., 3] - 0.014 * mechanics.command),
            0.0, 1.0
        )
        modified["network"] = np.clip(
            kwargs["network"]
            * (1.0 - 0.020 * lagged[..., 0] - 0.020 * lagged[..., 3]),
            0.0, 1.0,
        )
        modified["contagion_fatigue"] = np.clip(
            kwargs["contagion_fatigue"] + 0.035 * lagged[..., 0], 0.0, 1.0
        )
        self.maximum_lagged_field_input_shift = max(
            self.maximum_lagged_field_input_shift,
            float(max(
                np.max(np.abs(modified["damage"] - kwargs["damage"])),
                np.max(np.abs(modified["shortage"] - kwargs["shortage"])),
                np.max(np.abs(
                    modified["financial_stress"] - kwargs["financial_stress"]
                )),
                np.max(np.abs(modified["command"] - kwargs["command"])),
            )),
        )

        factors = super().update(**modified)
        cascade_loss = np.stack((
            factors.cascade_social_loss,
            factors.cascade_financial_loss,
            factors.cascade_supply_loss,
            factors.cascade_command_loss,
        ), axis=-1)
        if self.statistical_field_enabled:
            field = self.statistical_field.update(
                damage=modified["damage"],
                shortage=modified["shortage"],
                inflation=modified["inflation"],
                financial_stress=modified["financial_stress"],
                command=modified["command"],
                network=modified["network"],
                contagion_mobilization=modified["contagion_mobilization"],
                contagion_fatigue=modified["contagion_fatigue"],
                jump_signal=modified.get(
                    "jump_signal", np.zeros_like(modified["damage"])
                ),
                mfg_mean=self._mfg_mean(),
                cascade_loss=cascade_loss,
            )
            current = np.stack((
                field.social, field.financial, field.supply, field.command
            ), axis=-1)
            self.previous_field_feedback = current.copy()

            # These factors already have same-month outer consumers in the parent
            # simulation.  The residual contribution is bounded and separately
            # diagnosed; it never alters terminal probabilities after simulation.
            weight = 0.30
            factors.cascade_social_loss = np.clip(
                factors.cascade_social_loss + weight * field.social, 0.0, 1.0
            )
            factors.cascade_financial_loss = np.clip(
                factors.cascade_financial_loss + weight * field.financial, 0.0, 1.0
            )
            factors.cascade_supply_loss = np.clip(
                factors.cascade_supply_loss + weight * field.supply, 0.0, 1.0
            )
            factors.cascade_command_loss = np.clip(
                factors.cascade_command_loss + weight * field.command, 0.0, 1.0
            )
            factors.cascade_recovery_need = np.clip(
                factors.cascade_recovery_need + 0.24 * field.recovery_need,
                0.0, 1.0,
            )
            factors.fatigue_preference = np.clip(
                factors.fatigue_preference + 0.06 * field.social, 0.0, 1.0
            )
            factors.threshold_shift = np.clip(
                factors.threshold_shift
                + 0.035 * field.social[:, :, None]
                + 0.025 * field.financial[:, :, None],
                -0.18, 0.18,
            )
            if factors.joint_action_field is not None:
                factors.joint_action_field[..., ACTION_CHANNELS.index("hoarding")] = (
                    np.clip(
                        factors.joint_action_field[
                            ..., ACTION_CHANNELS.index("hoarding")
                        ] + 0.045 * field.supply,
                        0.0, 1.0,
                    )
                )
                factors.joint_action_field[..., ACTION_CHANNELS.index("migration")] = (
                    np.clip(
                        factors.joint_action_field[
                            ..., ACTION_CHANNELS.index("migration")
                        ] + 0.035 * field.social,
                        0.0, 1.0,
                    )
                )
            self.maximum_current_outer_field_contribution = max(
                self.maximum_current_outer_field_contribution,
                float(weight * np.max(current)),
            )
            self.field_feedback_updates += 1
        return factors

    def diagnostics(self) -> dict:
        payload = super().diagnostics()
        payload["v80_non_equilibrium_statistical_field"] = (
            self.statistical_field.diagnostics() | {
                "enabled": self.statistical_field_enabled,
                "feedback_updates": self.field_feedback_updates,
                "maximum_lagged_mfg_input_shift": (
                    self.maximum_lagged_field_input_shift
                ),
                "maximum_current_outer_factor_contribution": (
                    self.maximum_current_outer_field_contribution
                ),
                "same_path_outer_consumers": [
                    "social contagion", "financial stress", "supply shortage",
                    "network integrity", "command hierarchy",
                ],
            }
        )
        payload["v80_soc_evidence"] = self.cascade.diagnostics()[
            "v80_soc_evidence"
        ]
        payload["v80_analytical_mechanics"] = (
            self.analytical_mechanics.diagnostics() | {
                "enabled": self.analytical_mechanics_enabled,
            }
        )
        return payload
