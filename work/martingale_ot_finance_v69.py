from __future__ import annotations

import numpy as np

from simulate_four_party_financial_v45 import FinancialFactors, FourPartyFinancialSystem


class MartingaleOTFinancialSystemV69(FourPartyFinancialSystem):
    """Risk-neutral entropy projection coupled to the physical financial system.

    The projection is deliberately confined to asset-pricing states. It does
    not impose a martingale on physical output, logistics, or public support.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_martingale_residual = 0.0
        self.max_probability_residual = 0.0
        self.max_negative_probability = 0.0
        self.mean_relative_entropy_sum = 0.0
        self.mean_downside_wedge_sum = 0.0
        self.projection_months = 0

    @staticmethod
    def _entropy_projection(
        prior: np.ndarray, states: np.ndarray, target: np.ndarray
    ) -> np.ndarray:
        """KL I-projection onto sum(q)=1 and E_q[state]=target."""
        lo = np.full(target.shape, -160.0)
        hi = np.full(target.shape, 160.0)
        for _ in range(72):
            lam = 0.5 * (lo + hi)
            logits = np.log(np.maximum(prior, 1.0e-15)) + lam[..., None] * states
            logits -= np.max(logits, axis=-1, keepdims=True)
            q = np.exp(logits)
            q /= np.sum(q, axis=-1, keepdims=True)
            mean = np.sum(q * states, axis=-1)
            lo = np.where(mean < target, lam, lo)
            hi = np.where(mean >= target, lam, hi)
        logits = np.log(np.maximum(prior, 1.0e-15)) + (0.5 * (lo + hi))[..., None] * states
        logits -= np.max(logits, axis=-1, keepdims=True)
        q = np.exp(logits)
        return q / np.sum(q, axis=-1, keepdims=True)

    def update(self, *args, **kwargs) -> FinancialFactors:
        opening_asset = self.asset_price.copy()
        factors = super().update(*args, **kwargs)
        dt = 1.0 / 12.0
        forward = opening_asset * np.exp(self.policy_rate * dt)
        volatility = np.sqrt(np.maximum(self.financial_variance * dt, 1.0e-8))
        width = np.clip(2.4 * volatility + 0.035, 0.04, 0.38)
        states = forward[..., None] * np.stack(
            (1.0 - width, np.ones_like(width), 1.0 + width), axis=-1
        )
        physical_signal = np.clip(
            (self.asset_price - forward) / np.maximum(forward * width, 1.0e-8),
            -2.0, 2.0,
        )
        prior_logits = np.stack(
            (-0.35 - physical_signal, np.zeros_like(physical_signal),
             -0.35 + physical_signal), axis=-1
        )
        prior_logits -= np.max(prior_logits, axis=-1, keepdims=True)
        prior = np.exp(prior_logits)
        prior /= np.sum(prior, axis=-1, keepdims=True)
        q = self._entropy_projection(prior, states, forward)

        probability_residual = np.abs(np.sum(q, axis=-1) - 1.0)
        martingale_residual = np.abs(np.sum(q * states, axis=-1) - forward)
        relative_entropy = np.sum(
            q * (np.log(np.maximum(q, 1.0e-15))
                 - np.log(np.maximum(prior, 1.0e-15))), axis=-1
        )
        downside = np.sum(q * np.maximum(forward[..., None] - states, 0.0), axis=-1)
        downside /= np.maximum(forward, 1.0e-10)
        risk_wedge = np.clip(0.55 * np.sqrt(np.maximum(relative_entropy, 0.0))
                             + 0.45 * downside, 0.0, 0.35)

        access_factor = np.exp(-0.10 * risk_wedge)
        credit = np.clip(factors.credit * np.exp(-0.12 * risk_wedge), 0.08, 1.0)
        fiscal = np.clip(factors.fiscal * np.exp(-0.06 * risk_wedge), 0.08, 1.0)
        production = np.clip(factors.production * np.exp(-0.07 * risk_wedge), 0.25, 1.0)
        stress = np.clip(factors.stress + 0.08 * risk_wedge, 0.0, 1.0)
        self.capital_access = np.clip(self.capital_access * access_factor, 0.04, 0.99)
        adjusted = FinancialFactors(
            production, credit, factors.payment, fiscal, factors.aid,
            stress, factors.inflation,
        )
        self.last_factors = adjusted
        self.max_martingale_residual = max(
            self.max_martingale_residual, float(np.max(martingale_residual))
        )
        self.max_probability_residual = max(
            self.max_probability_residual, float(np.max(probability_residual))
        )
        self.max_negative_probability = max(
            self.max_negative_probability, float(np.max(np.maximum(-q, 0.0)))
        )
        self.mean_relative_entropy_sum += float(np.mean(relative_entropy))
        self.mean_downside_wedge_sum += float(np.mean(downside))
        self.projection_months += 1
        return adjusted

    def summary(self) -> dict:
        payload = super().summary()
        payload["martingale_optimal_transport"] = {
            "scope": "risk-neutral asset-price distribution only",
            "method": "discrete KL I-projection with discounted martingale constraint",
            "same_path_financial_feedback": True,
            "projection_months": self.projection_months,
            "max_martingale_residual": self.max_martingale_residual,
            "max_probability_residual": self.max_probability_residual,
            "max_negative_probability": self.max_negative_probability,
            "mean_relative_entropy": self.mean_relative_entropy_sum / max(self.projection_months, 1),
            "mean_downside_wedge": self.mean_downside_wedge_sum / max(self.projection_months, 1),
        }
        return payload
