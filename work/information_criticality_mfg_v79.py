from __future__ import annotations

import numpy as np

from continuous_multipop_mfg_v47 import ACTION_CHANNELS
from information_criticality_v79 import (
    ConditionalTransferEntropyV79,
    InformationTheoryV79,
    QuasicriticalCascadeV79,
)
from mixed_potential_nonpotential_mfg_v78 import MixedPotentialNonpotentialMFGV78
from renormalization_scale_audit_v79 import RenormalizationScaleAuditV79


class InformationCriticalityMFGV79(MixedPotentialNonpotentialMFGV78):
    """V7.8 plus information-theoretic and quasi-critical same-path feedback."""

    def __init__(self, *args, **kwargs):
        paths, seed, initial_support = args[:3]
        super().__init__(*args, **kwargs)
        if self.partial_observation_geometry:
            self.belief_system = InformationTheoryV79(
                paths, seed + 79_003, initial_support,
                self.actor_kernel, self.age_kernel,
                ambiguity_radius=kwargs.get("information_ambiguity_radius", 0.025),
                belief_filter_enabled=kwargs.get("belief_filter_enabled", True),
                credibility_hmm_enabled=kwargs.get("credibility_hmm_enabled", True),
                schrodinger_bridge_enabled=kwargs.get("schrodinger_bridge_enabled", True),
                fgw_alignment_enabled=kwargs.get("fgw_alignment_enabled", True),
                robust_information_enabled=kwargs.get("robust_information_enabled", True),
                information_value_enabled=kwargs.get("information_value_enabled", True),
            )
        self.information_flow = ConditionalTransferEntropyV79()
        self.cascade = QuasicriticalCascadeV79(paths, self.actor_kernel)
        self.scale_audit = RenormalizationScaleAuditV79(self.age_weights)
        self.maximum_transfer_entropy_kernel_shift = 0.0
        self.maximum_cascade_input_shift = 0.0
        self.cascade_feedback_updates = 0

    @staticmethod
    def _normalise_positive_rows(kernel: np.ndarray) -> np.ndarray:
        mass = np.sum(np.maximum(kernel, 0.0), axis=-1, keepdims=True)
        return np.maximum(kernel, 0.0) / np.maximum(mass, 1.0e-12)

    def _effective_graphon(self, *args, **kwargs):
        kernel, weights, intensity = super()._effective_graphon(*args, **kwargs)
        gain = self.information_flow.current_gain()
        adjusted = self._normalise_positive_rows(kernel * gain[None, ...])
        self.maximum_transfer_entropy_kernel_shift = max(
            self.maximum_transfer_entropy_kernel_shift,
            float(np.max(np.abs(adjusted - kernel))),
        )

        damage, shortage, network, common_noise, jump_signal, actor_signal = args[:6]
        layer_signals = np.stack((
            0.52 * actor_signal + 0.30 * network - 0.18 * common_noise,
            -0.46 * shortage - 0.32 * damage + 0.22 * network,
            0.66 * actor_signal + 0.20 * network - 0.14 * jump_signal,
            0.58 * damage + 0.30 * jump_signal + 0.12 * shortage,
        ), axis=1)
        # The estimator observes month t only after the month-(t-1) gain has
        # been applied. Its updated direction therefore first affects t+1.
        self.information_flow.observe(layer_signals)
        return adjusted, weights, intensity

    def update(self, **kwargs):
        service_index = ACTION_CHANNELS.index("service")
        service = np.sum(
            self.previous_joint_action[..., service_index]
            * self.age_weights[None, None, :],
            axis=2,
        )
        cascade = self.cascade.update(
            damage=kwargs["damage"],
            shortage=kwargs["shortage"],
            inflation=kwargs["inflation"],
            financial_stress=kwargs["financial_stress"],
            command=kwargs["command"],
            network=kwargs["network"],
            contagion_mobilization=kwargs["contagion_mobilization"],
            contagion_fatigue=kwargs["contagion_fatigue"],
            jump_signal=kwargs.get("jump_signal", np.zeros_like(kwargs["damage"])),
            service_action=service,
        )
        modified = dict(kwargs)
        modified["damage"] = np.clip(
            kwargs["damage"] + 0.08 * cascade.social
            + 0.10 * cascade.command,
            0.0, 1.5,
        )
        modified["shortage"] = np.clip(
            kwargs["shortage"] + 0.14 * cascade.supply,
            0.0, 1.5,
        )
        modified["financial_stress"] = np.clip(
            kwargs["financial_stress"] + 0.15 * cascade.financial,
            0.0, 1.5,
        )
        modified["command"] = np.clip(
            kwargs["command"] * (1.0 - 0.16 * cascade.command),
            0.0, 1.0,
        )
        modified["network"] = np.clip(
            kwargs["network"]
            * (1.0 - 0.10 * cascade.social - 0.10 * cascade.command),
            0.0, 1.0,
        )
        modified["contagion_fatigue"] = np.clip(
            kwargs["contagion_fatigue"] + 0.18 * cascade.social,
            0.0, 1.0,
        )
        self.maximum_cascade_input_shift = max(
            self.maximum_cascade_input_shift,
            float(max(
                np.max(np.abs(modified["shortage"] - kwargs["shortage"])),
                np.max(np.abs(
                    modified["financial_stress"] - kwargs["financial_stress"]
                )),
                np.max(np.abs(modified["command"] - kwargs["command"])),
            )),
        )

        factors = super().update(**modified)
        factors.cascade_social_loss = cascade.social.copy()
        factors.cascade_financial_loss = cascade.financial.copy()
        factors.cascade_supply_loss = cascade.supply.copy()
        factors.cascade_command_loss = cascade.command.copy()
        factors.cascade_recovery_need = cascade.recovery_need.copy()
        factors.fatigue_preference = np.clip(
            factors.fatigue_preference + 0.12 * cascade.social,
            0.0, 1.0,
        )
        factors.threshold_shift = np.clip(
            factors.threshold_shift
            + 0.08 * cascade.social[:, :, None]
            + 0.05 * cascade.financial[:, :, None],
            -0.18, 0.18,
        )
        if factors.joint_action_field is not None:
            factors.joint_action_field[..., ACTION_CHANNELS.index("hoarding")] = np.clip(
                factors.joint_action_field[..., ACTION_CHANNELS.index("hoarding")]
                + 0.10 * cascade.supply,
                0.0, 1.0,
            )
            factors.joint_action_field[..., ACTION_CHANNELS.index("migration")] = np.clip(
                factors.joint_action_field[..., ACTION_CHANNELS.index("migration")]
                + 0.08 * cascade.social,
                0.0, 1.0,
            )
        self.scale_audit.observe(self.density, self.x)
        self.cascade_feedback_updates += 1
        return factors

    def diagnostics(self) -> dict:
        payload = super().diagnostics()
        payload["v79_information_flow"] = self.information_flow.diagnostics() | {
            "maximum_graphon_kernel_shift": self.maximum_transfer_entropy_kernel_shift,
            "graphon_feedback": True,
        }
        payload["v79_quasicritical_cascade"] = self.cascade.diagnostics() | {
            "feedback_updates": self.cascade_feedback_updates,
            "maximum_mfg_input_shift": self.maximum_cascade_input_shift,
            "outer_state_consumers": [
                "social contagion", "firm distress", "debt stress",
                "network integrity", "command hierarchy",
            ],
        }
        payload["v79_renormalization_scale_audit"] = self.scale_audit.diagnostics()
        return payload
