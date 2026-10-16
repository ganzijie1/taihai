from __future__ import annotations

import numpy as np

from martingale_ot_finance_v69 import MartingaleOTFinancialSystemV69
from simulate_four_party_financial_v45 import FinancialFactors


ACTORS = ("china", "taiwan", "united_states", "japan")
FINANCING_STATES = ("hedge", "speculative", "ponzi")


class QuantitativeFinancialInstabilityV75(MartingaleOTFinancialSystemV69):
    """Pathwise financial-instability closure for the four-party system.

    The class keeps the inherited accounting owner and martingale projection.
    It adds a Minsky financing-state process, aggregate Eisenberg-Noe clearing,
    endogenous fire sales, JCIR credit intensity, regime-switching variance,
    Hawkes clustering, term/FX pressure and a local CVaR-DRO wedge.  All
    adjustments enter the current FinancialFactors object; the resulting
    credit, stress, production and inflation therefore feed the same master
    path before the social, MFG and terminal transitions.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        n = self.paths
        initial = np.array([
            [0.70, 0.24, 0.06],
            [0.74, 0.21, 0.05],
            [0.72, 0.22, 0.06],
            [0.77, 0.19, 0.04],
        ])
        self.financing_share = np.broadcast_to(initial[None, :, :], (n, 4, 3)).copy()
        self.dscr = np.ones((n, 4)) * np.array([[1.55, 1.62, 1.48, 1.70]])
        self.credit_intensity = np.ones((n, 4)) * np.array([[0.010, 0.009, 0.012, 0.008]])
        self.hawkes_intensity = np.zeros((n, 4))
        self.liquidity_gap = np.zeros((n, 4))
        self.fire_sale_pressure = np.zeros((n, 4))
        self.fx_pressure = np.zeros((n, 4))
        self.term_spread = self.sovereign_rate - self.policy_rate
        self.regime_probability = np.zeros((n, 4, 3))
        self.regime_probability[..., 0] = 0.88
        self.regime_probability[..., 1] = 0.10
        self.regime_probability[..., 2] = 0.02
        self.last_clearing_fraction = np.ones((n, 4))
        self.last_cvar_wedge = np.zeros((n, 4))
        self.last_dro_wedge = np.zeros((n, 4))
        self.last_investment_option_wedge = np.zeros((n, 4))
        self.months_v75 = 0
        self.max_financing_mass_residual = 0.0
        self.max_clearing_residual = 0.0
        self.max_negative_payment = 0.0
        self.max_fire_sale_pressure = 0.0
        self.max_credit_intensity = float(np.max(self.credit_intensity))
        self.max_regime_mass_residual = 0.0
        self.max_tail_wedge = 0.0

        # Aggregate, public-data-level cross-border financial obligations.
        # Coefficients are scenario priors, not institution-level exposures.
        self.liability_kernel = np.array([
            [0.00, 0.08, 0.12, 0.09],
            [0.06, 0.00, 0.10, 0.08],
            [0.10, 0.14, 0.00, 0.13],
            [0.08, 0.11, 0.15, 0.00],
        ])
        self.actor_transition_scale = np.array([1.12, 1.06, 0.98, 0.92])[None, :]
        self.actor_backstop = np.array([0.78, 0.66, 0.92, 0.88])[None, :]

    @staticmethod
    def _sigmoid(value: np.ndarray) -> np.ndarray:
        value = np.clip(value, -40.0, 40.0)
        return 1.0 / (1.0 + np.exp(-value))

    @staticmethod
    def _normalize_probability(value: np.ndarray) -> np.ndarray:
        clipped = np.maximum(value, 0.0)
        return clipped / np.maximum(np.sum(clipped, axis=-1, keepdims=True), 1.0e-15)

    def _update_minsky_states(
        self,
        cash_flow: np.ndarray,
        debt_service: np.ndarray,
        distress: np.ndarray,
        shortage: np.ndarray,
        policy_support: np.ndarray,
    ) -> None:
        interest = np.maximum(self.avg_coupon * (self.debt + 0.45 * self.local_debt), 1.0e-5)
        principal = np.maximum(debt_service - interest, 0.0)
        self.dscr = cash_flow / np.maximum(interest + principal, 1.0e-6)
        interest_cover = cash_flow / np.maximum(interest, 1.0e-6)
        leverage = self.debt + 0.62 * self.local_debt
        fragility = np.clip(
            0.30 * distress + 0.20 * shortage + 0.18 * leverage
            + 0.18 * (1.0 - self.asset_price) + 0.14 * (1.0 - self.capital_access)
            - 0.16 * policy_support,
            0.0,
            2.5,
        )
        hs = self._sigmoid(-3.1 + 3.0 * fragility + 1.8 * np.maximum(1.35 - self.dscr, 0.0))
        sp = self._sigmoid(-3.5 + 3.4 * fragility + 2.2 * np.maximum(1.0 - interest_cover, 0.0))
        sh = self._sigmoid(-2.6 - 2.0 * fragility + 1.5 * np.maximum(self.dscr - 1.1, 0.0))
        ps = self._sigmoid(-2.9 - 1.7 * fragility + 1.2 * np.maximum(interest_cover - 1.0, 0.0))
        scale = self.actor_transition_scale
        dt = 1.0 / 12.0
        hedge, speculative, ponzi = np.moveaxis(self.financing_share, -1, 0)
        flow_hs = np.minimum(hedge, hedge * hs * scale * dt)
        flow_sp = np.minimum(speculative, speculative * sp * scale * dt)
        flow_sh = np.minimum(speculative - flow_sp, speculative * sh * dt)
        flow_ps = np.minimum(ponzi, ponzi * ps * dt)
        updated = np.stack((
            hedge - flow_hs + flow_sh,
            speculative + flow_hs - flow_sp - flow_sh + flow_ps,
            ponzi + flow_sp - flow_ps,
        ), axis=-1)
        self.financing_share = self._normalize_probability(updated)
        self.max_financing_mass_residual = max(
            self.max_financing_mass_residual,
            float(np.max(np.abs(np.sum(self.financing_share, axis=-1) - 1.0))),
        )

    def _eisenberg_noe_clearing(
        self, external_assets: np.ndarray, obligations: np.ndarray
    ) -> np.ndarray:
        liabilities = (
            obligations[:, :, None]
            * self.liability_kernel[None, :, :]
        )
        total = np.sum(liabilities, axis=2) + 0.55 * obligations
        relative = liabilities / np.maximum(total[:, :, None], 1.0e-12)
        payment = total.copy()
        for _ in range(80):
            incoming = np.einsum("pji,pj->pi", relative, payment)
            updated = np.minimum(total, np.maximum(external_assets + incoming, 0.0))
            if float(np.max(np.abs(updated - payment))) < 1.0e-11:
                payment = updated
                break
            payment = 0.62 * payment + 0.38 * updated
        incoming = np.einsum("pji,pj->pi", relative, payment)
        residual = np.abs(payment - np.minimum(total, np.maximum(external_assets + incoming, 0.0)))
        self.max_clearing_residual = max(self.max_clearing_residual, float(np.max(residual)))
        self.max_negative_payment = max(
            self.max_negative_payment, float(np.max(np.maximum(-payment, 0.0)))
        )
        return np.clip(payment / np.maximum(total, 1.0e-12), 0.0, 1.0)

    def _update_regime_and_intensity(
        self, base_stress: np.ndarray, clearing_fraction: np.ndarray
    ) -> None:
        ponzi = self.financing_share[..., 2]
        event = np.clip(
            0.55 * self.bank_crisis.astype(float)
            + 0.30 * (clearing_fraction < 0.92)
            + 0.15 * (base_stress > 0.65),
            0.0,
            1.0,
        )
        self.hawkes_intensity = np.clip(
            np.exp(-0.42) * self.hawkes_intensity + 0.20 * event,
            0.0,
            2.0,
        )
        theta = np.clip(
            0.006 + 0.10 * base_stress + 0.08 * ponzi
            + 0.06 * (1.0 - clearing_fraction),
            0.002,
            0.40,
        )
        dt = 1.0 / 12.0
        innovation = self.rng.normal(0.0, 1.0, self.credit_intensity.shape)
        self.credit_intensity += (
            1.15 * (theta - self.credit_intensity) * dt
            + 0.10 * np.sqrt(np.maximum(self.credit_intensity, 1.0e-8) * dt) * innovation
            + 0.06 * event
            + 0.04 * self.hawkes_intensity
        )
        self.credit_intensity = np.clip(self.credit_intensity, 1.0e-5, 1.5)
        self.max_credit_intensity = max(
            self.max_credit_intensity, float(np.max(self.credit_intensity))
        )

        calm, stress, crisis = np.moveaxis(self.regime_probability, -1, 0)
        to_stress = np.clip(0.03 + 0.28 * base_stress + 0.18 * ponzi, 0.0, 0.55)
        to_crisis = np.clip(0.005 + 0.18 * self.credit_intensity + 0.20 * event, 0.0, 0.45)
        recover = np.clip(0.10 * self.actor_backstop * (1.0 - base_stress), 0.0, 0.12)
        flow_cs = calm * to_stress
        flow_sc = stress * recover
        flow_sx = np.minimum(stress - flow_sc, stress * to_crisis)
        flow_xs = crisis * (0.5 * recover)
        updated = np.stack((
            calm - flow_cs + flow_sc,
            stress + flow_cs - flow_sc - flow_sx + flow_xs,
            crisis + flow_sx - flow_xs,
        ), axis=-1)
        self.regime_probability = self._normalize_probability(updated)
        self.max_regime_mass_residual = max(
            self.max_regime_mass_residual,
            float(np.max(np.abs(np.sum(self.regime_probability, axis=-1) - 1.0))),
        )

    def update(self, *args, **kwargs) -> FinancialFactors:
        factors = super().update(*args, **kwargs)
        output = np.mean(np.asarray(kwargs["sector_output"]), axis=2)
        distress = np.mean(np.asarray(kwargs["firm_distress"]), axis=2)
        shortage = np.asarray(kwargs["actor_shortage"])
        damage = np.asarray(kwargs["actor_damage"])
        trade_factor = np.asarray(kwargs["trade_gdp_factor"])
        cash_flow = np.clip(
            output * trade_factor * (1.0 - 0.45 * distress - 0.25 * damage),
            0.005,
            1.5,
        )
        debt_stock = self.debt + 0.62 * self.local_debt
        debt_service = np.maximum(
            self.avg_coupon * debt_stock + debt_stock / np.maximum(self.maturity * 12.0, 12.0),
            1.0e-5,
        )
        policy_support = np.clip(
            0.45 * factors.fiscal + 0.30 * factors.payment
            + 0.25 * self.actor_backstop,
            0.0,
            1.0,
        )
        self._update_minsky_states(
            cash_flow, debt_service, distress, shortage, policy_support
        )

        external_assets = np.clip(
            0.34 * self.asset_price * self.bank_assets
            + 0.26 * cash_flow + 0.18 * self.reserve
            + 0.12 * factors.credit + 0.10 * policy_support,
            0.0,
            None,
        )
        obligations = np.clip(
            0.18 * self.bank_assets + 0.36 * debt_service
            + 0.12 * self.financing_share[..., 2],
            1.0e-6,
            None,
        )
        clearing_fraction = self._eisenberg_noe_clearing(external_assets, obligations)
        self.last_clearing_fraction = clearing_fraction

        available_liquidity = (
            0.22 * factors.credit + 0.18 * self.lcr + 0.14 * self.reserve
            + 0.16 * policy_support + 0.10 * clearing_fraction
        )
        self.liquidity_gap = np.maximum(
            debt_service + 0.20 * distress + 0.14 * shortage
            - cash_flow - available_liquidity,
            0.0,
        )
        collateral = np.maximum((1.0 - np.clip(0.12 + self.financial_variance, 0.0, 0.72)) * self.asset_price, 0.05)
        sale = np.clip(self.liquidity_gap / collateral, 0.0, 2.0)
        self.fire_sale_pressure = 0.72 * self.fire_sale_pressure + 0.28 * sale
        self.max_fire_sale_pressure = max(
            self.max_fire_sale_pressure, float(np.max(self.fire_sale_pressure))
        )
        price_impact = np.array([[0.10, 0.13, 0.08, 0.09]])
        self.asset_price = np.clip(
            self.asset_price * np.exp(-price_impact * self.fire_sale_pressure),
            0.08,
            1.20,
        )

        base_stress = np.clip(
            factors.stress + 0.22 * self.financing_share[..., 2]
            + 0.16 * (1.0 - clearing_fraction)
            + 0.12 * self.fire_sale_pressure,
            0.0,
            1.0,
        )
        self._update_regime_and_intensity(base_stress, clearing_fraction)
        regime_multiplier = np.sum(
            self.regime_probability * np.array([1.0, 1.8, 3.2])[None, None, :],
            axis=-1,
        )
        self.financial_variance = np.clip(
            self.financial_variance * (0.94 + 0.06 * regime_multiplier),
            0.002,
            0.50,
        )

        self.term_spread = self.sovereign_rate - self.policy_rate
        external_deficit = np.maximum(-np.asarray(kwargs["macro_trade_balance"]), 0.0)
        self.fx_pressure = np.clip(
            0.55 * self.fx_pressure + 0.18 * external_deficit
            + 0.16 * (1.0 - self.capital_access)
            + 0.12 * base_stress - 0.10 * self.reserve,
            0.0,
            1.5,
        )
        self.last_investment_option_wedge = np.clip(
            0.20 * np.sqrt(np.maximum(self.financial_variance, 0.0))
            + 0.16 * np.abs(self.term_spread) + 0.12 * self.fx_pressure,
            0.0,
            0.55,
        )
        local_loss_mean = np.clip(
            0.30 * base_stress + 0.22 * self.fire_sale_pressure
            + 0.18 * (1.0 - clearing_fraction),
            0.0,
            1.5,
        )
        local_loss_scale = np.sqrt(np.maximum(self.financial_variance, 1.0e-8))
        self.last_cvar_wedge = np.clip(
            local_loss_mean + 2.06 * local_loss_scale, 0.0, 1.5
        )
        self.last_dro_wedge = np.clip(
            0.55 * self.last_cvar_wedge
            + 0.25 * self.hawkes_intensity
            + 0.20 * self.credit_intensity,
            0.0,
            1.5,
        )
        tail_wedge = np.clip(
            0.28 * self.last_dro_wedge
            + 0.22 * self.last_investment_option_wedge,
            0.0,
            0.70,
        )
        self.max_tail_wedge = max(self.max_tail_wedge, float(np.max(tail_wedge)))

        credit = np.clip(
            factors.credit * clearing_fraction
            * np.exp(-0.18 * self.financing_share[..., 2] - 0.16 * tail_wedge),
            0.04,
            1.0,
        )
        payment = np.clip(factors.payment * clearing_fraction, 0.05, 1.0)
        stress = np.clip(base_stress + 0.18 * tail_wedge + 0.08 * self.fx_pressure, 0.0, 1.0)
        production = np.clip(
            factors.production * np.exp(
                -0.12 * (1.0 - credit) - 0.10 * self.last_investment_option_wedge
            ),
            0.20,
            1.0,
        )
        fiscal = np.clip(
            factors.fiscal * np.exp(-0.08 * tail_wedge), 0.05, 1.0
        )
        aid = np.clip(factors.aid * np.exp(-0.05 * tail_wedge), 0.05, 1.0)
        inflation = np.clip(
            factors.inflation + 0.035 * self.fx_pressure
            + 0.025 * self.fire_sale_pressure,
            -0.02,
            0.90,
        )
        adjusted = FinancialFactors(
            production, credit, payment, fiscal, aid, stress, inflation
        )
        self.last_factors = adjusted
        self.months_v75 += 1
        return adjusted

    def summary(self) -> dict:
        payload = super().summary()
        payload["v75_quantitative_financial_instability"] = {
            "classification": "active same-path dynamic financial state",
            "months": self.months_v75,
            "financing_states": list(FINANCING_STATES),
            "mean_financing_share": {
                name: float(np.mean(self.financing_share[..., index]))
                for index, name in enumerate(FINANCING_STATES)
            },
            "mean_dscr": float(np.mean(self.dscr)),
            "mean_clearing_fraction": float(np.mean(self.last_clearing_fraction)),
            "mean_credit_intensity": float(np.mean(self.credit_intensity)),
            "mean_hawkes_intensity": float(np.mean(self.hawkes_intensity)),
            "mean_fire_sale_pressure": float(np.mean(self.fire_sale_pressure)),
            "mean_fx_pressure": float(np.mean(self.fx_pressure)),
            "max_financing_mass_residual": self.max_financing_mass_residual,
            "max_regime_mass_residual": self.max_regime_mass_residual,
            "max_clearing_residual": self.max_clearing_residual,
            "max_negative_payment": self.max_negative_payment,
            "max_fire_sale_pressure": self.max_fire_sale_pressure,
            "max_credit_intensity": self.max_credit_intensity,
            "max_tail_wedge": self.max_tail_wedge,
            "components": [
                "Minsky three-state financing",
                "aggregate Eisenberg-Noe clearing",
                "endogenous liquidity gap and fire sale",
                "JCIR credit intensity with Hawkes excitation",
                "three-regime stochastic variance",
                "term-spread and FX pressure",
                "path-local CVaR and distributionally robust wedge",
                "inherited martingale optimal-transport pricing projection",
            ],
            "institution_level_claim": False,
            "terminal_probability_adjustment": False,
        }
        return payload
