from __future__ import annotations

from dataclasses import dataclass

import numpy as np


CASES = ("timely_full", "limited", "delayed", "japan_only", "taiwan_alone")


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -35.0, 35.0)))


@dataclass
class InterventionChoice:
    probabilities: np.ndarray
    us_direct_probability: np.ndarray
    japan_combat_probability: np.ndarray
    japan_base_support_probability: np.ndarray
    entropy: np.ndarray


class EndogenousInterventionChoiceModel:
    """Pathwise Stackelberg-discrete-choice model for five intervention regimes.

    The United States sends the first direct-intervention signal. Japan then
    chooses combat intensity independently, subject to the institutional floor
    that US direct intervention entails Japanese base, logistics, and support.
    The five reported regimes are a mutually exclusive reduced partition of the
    underlying sequential choices.
    """

    def __init__(self, paths: int, seed: int):
        self.paths = paths
        self.rng = np.random.default_rng(seed)
        self.last_choice: InterventionChoice | None = None

    def probabilities(
        self, *, actor_support: np.ndarray, taiwan_readiness: np.ndarray,
        leader_continuity: np.ndarray, arms_support_us: np.ndarray,
        arms_support_japan: np.ndarray, fiscal_capacity: np.ndarray,
        financial_stress: np.ndarray, aft_delay_months: np.ndarray | None = None,
    ) -> InterventionChoice:
        support = np.clip(actor_support, 0.01, 0.99)
        readiness = np.clip(taiwan_readiness, 0.45, 1.55)
        continuity = np.clip(leader_continuity, 0.10, 1.0)
        fiscal = np.clip(fiscal_capacity, 0.05, 1.0)
        stress = np.clip(financial_stress, 0.0, 1.0)

        common = self.rng.normal(0.0, 0.12, self.paths)
        us_latent = (
            -0.72 + 2.05 * (support[:, 2] - 0.42)
            + 0.88 * (support[:, 1] - 0.62)
            + 0.74 * (np.clip(arms_support_us, 0.01, 0.99) - 0.62)
            + 0.46 * np.log(readiness)
            + 0.58 * (continuity[:, 2] - 0.75)
            + 0.64 * (fiscal[:, 2] - 0.55) - 0.72 * stress[:, 2]
            + common
        )
        us_direct = _sigmoid(us_latent)

        japan_latent = (
            -1.62 + 2.20 * (support[:, 3] - 0.11)
            + 0.92 * (support[:, 1] - 0.62)
            + 1.05 * (np.clip(arms_support_japan, 0.01, 0.99) - 0.55)
            + 0.72 * us_direct + 0.52 * (continuity[:, 3] - 0.75)
            + 0.55 * (fiscal[:, 3] - 0.55) - 0.78 * stress[:, 3]
            + 0.55 * common
        )
        japan_combat = _sigmoid(japan_latent)
        # Treaty/base access is distinct from direct combat. It has a hard
        # floor conditional on US direct intervention.
        japan_base = np.maximum(
            0.92 * us_direct,
            _sigmoid(-0.10 + 1.10 * arms_support_japan + 0.55 * support[:, 3]),
        )

        structural_delay = _sigmoid(
            -0.65 + 0.80 * stress[:, 2] + 0.55 * (1.0 - fiscal[:, 2])
            + 0.45 * (1.0 - continuity[:, 2]) - 0.30 * readiness
        )
        if aft_delay_months is None:
            delay = structural_delay
        else:
            aft_delay = _sigmoid((np.asarray(aft_delay_months) - 1.0) / 0.55)
            delay = np.clip(0.35 * structural_delay + 0.65 * aft_delay, 0.01, 0.99)
        full_given_us = np.clip(
            0.34 + 0.38 * japan_combat + 0.18 * readiness
            + 0.15 * japan_base - 0.22 * delay,
            0.05, 0.92,
        )
        delayed = us_direct * delay
        timely = us_direct * (1.0 - delay) * full_given_us
        limited = us_direct * (1.0 - delay) * (1.0 - full_given_us)
        no_us = 1.0 - us_direct
        japan_only = no_us * japan_combat
        taiwan_alone = no_us * (1.0 - japan_combat)
        probs = np.column_stack((timely, limited, delayed, japan_only, taiwan_alone))
        probs /= np.maximum(probs.sum(axis=1, keepdims=True), 1e-12)
        entropy = -np.sum(probs * np.log(np.maximum(probs, 1e-12)), axis=1)
        self.last_choice = InterventionChoice(
            probabilities=probs,
            us_direct_probability=us_direct,
            japan_combat_probability=japan_combat,
            japan_base_support_probability=japan_base,
            entropy=entropy,
        )
        return self.last_choice

    def diagnostics(self) -> dict:
        if self.last_choice is None:
            return {"evaluated": False}
        p = self.last_choice.probabilities
        return {
            "evaluated": True,
            "simplex_residual": float(np.max(np.abs(np.sum(p, axis=1) - 1.0))),
            "minimum_probability": float(np.min(p)),
            "maximum_probability": float(np.max(p)),
            "us_direct_implies_japan_base_floor_residual": float(np.max(
                np.maximum(
                    0.92 * self.last_choice.us_direct_probability
                    - self.last_choice.japan_base_support_probability,
                    0.0,
                )
            )),
        }
