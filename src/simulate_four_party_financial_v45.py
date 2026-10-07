from __future__ import annotations

from dataclasses import dataclass

import numpy as np


ACTORS = ("china", "taiwan", "united_states", "japan")
HORIZON_MONTHS = (12, 36, 60, 120, 180, 240, 360)


# Point observations anchor initial conditions. Ratios that are not published on
# one harmonized perimeter are represented by distributions around these anchors.
PUBLIC_CALIBRATION = {
    "china": {
        "cet1": 0.1087, "capital_adequacy": 0.1536, "npl": 0.0152, "lcr": 1.4973,
        "debt_to_gdp": 1.17, "local_debt_to_gdp": 0.62, "reserve_buffer": 0.17,
        "bank_assets_to_gdp": 3.20, "tax_revenue_to_gdp": 0.21,
        "source": "NFRA 2025Q3; PBOC Financial Stability Report; IMF 2025 Article IV",
    },
    "taiwan": {
        "cet1": 0.1260, "capital_adequacy": 0.1563, "npl": 0.0014, "lcr": 1.35,
        "debt_to_gdp": 0.28, "local_debt_to_gdp": 0.08, "reserve_buffer": 0.76,
        "bank_assets_to_gdp": 2.2739, "tax_revenue_to_gdp": 0.14,
        "source": "Taiwan FSC statistics; CBC 2025 annual report and reserve data",
    },
    "united_states": {
        "cet1": 0.1300, "capital_adequacy": 0.1500, "npl": 0.0100, "lcr": 1.25,
        "debt_to_gdp": 1.239, "local_debt_to_gdp": 0.16, "reserve_buffer": 0.02,
        "bank_assets_to_gdp": 0.88, "tax_revenue_to_gdp": 0.27,
        "source": "Federal Reserve 2025 banking conditions; IMF 2026 Article IV; US Treasury reserves",
    },
    "japan": {
        "cet1": 0.1250, "capital_adequacy": 0.1550, "npl": 0.0120, "lcr": 1.35,
        "debt_to_gdp": 2.49, "local_debt_to_gdp": 0.34, "reserve_buffer": 0.31,
        "bank_assets_to_gdp": 2.05, "tax_revenue_to_gdp": 0.34,
        "source": "BOJ Financial System Report 2026; Japan MOF debt and reserve statistics",
    },
}

SOURCE_URLS = {
    "china_banks": "https://big5.nfra.gov.cn/cn/view/pages/ItemDetail.html?docId=1233335&itemId=915",
    "china_stability": "https://www.pbc.gov.cn/goutongjiaoliu/113456/113469/2025122616592613805/2025122616590775273.pdf",
    "china_debt": "https://www.elibrary.imf.org/view/journals/002/2026/044/article-A001-en.xml",
    "taiwan_banks": "https://stat.fsc.gov.tw/",
    "taiwan_reserves": "https://www.cbc.gov.tw/en/np-699-2.html",
    "us_banks": "https://www.federalreserve.gov/publications/2025-december-supervision-and-regulation-report-banking-system-conditions.htm",
    "us_debt": "https://www.imf.org/en/news/articles/2026/04/01/pr-26102-usa-imf-executive-board-concludes-2026-article-iv-consult",
    "us_reserves": "https://home.treasury.gov/data/us-international-reserve-position/12122025",
    "japan_banks": "https://www.boj.or.jp/en/research/brp/fsr/data/fsr260421a.pdf",
    "japan_reserves": "https://www.mof.go.jp/english/policy/international_policy/reference/official_reserve_assets/e0708.html",
}


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -35.0, 35.0)))


def quantiles(values: np.ndarray) -> list[float]:
    return [float(x) for x in np.quantile(values, (0.10, 0.50, 0.90))]


@dataclass
class FinancialFactors:
    production: np.ndarray
    credit: np.ndarray
    payment: np.ndarray
    fiscal: np.ndarray
    aid: np.ndarray
    stress: np.ndarray
    inflation: np.ndarray


class FourPartyFinancialSystem:
    """Monthly balance-sheet network coupled to the V4.2 battlefield state."""

    def __init__(self, paths: int, months: int, seed: int, case: str, scenario: str):
        self.paths = paths
        self.months = months
        self.case = case
        self.scenario = scenario
        self.rng = np.random.default_rng(seed)
        n = paths
        anchors = [PUBLIC_CALIBRATION[a] for a in ACTORS]
        def initial(key, sigma=0.0):
            base = np.array([a[key] for a in anchors], dtype=float)[None, :]
            if sigma == 0:
                return np.repeat(base, n, axis=0)
            return base * self.rng.lognormal(0.0, sigma, (n, 4))

        self.cet1 = initial("cet1", 0.035)
        self.npl = initial("npl", 0.08)
        self.lcr = initial("lcr", 0.035)
        self.bank_assets = initial("bank_assets_to_gdp")
        self.debt = initial("debt_to_gdp", 0.035)
        self.local_debt = initial("local_debt_to_gdp", 0.06)
        self.reserve = initial("reserve_buffer", 0.035)
        self.tax_revenue = initial("tax_revenue_to_gdp")
        self.base_tax_revenue = self.tax_revenue.copy()
        self.tax_rates = np.repeat(np.array([
            [0.105, 0.105, 0.055], [0.075, 0.095, 0.060],
            [0.070, 0.145, 0.055], [0.105, 0.175, 0.070],
        ])[None, :, :], n, axis=0)
        self.base_tax_rates = self.tax_rates.copy()
        self.policy_rate = np.repeat(np.array([[0.015, 0.020, 0.036, 0.005]]), n, axis=0)
        self.sovereign_rate = np.repeat(np.array([[0.022, 0.024, 0.038, 0.015]]), n, axis=0)
        self.avg_coupon = np.repeat(np.array([[0.029, 0.025, 0.034, 0.014]]), n, axis=0)
        self.maturity = np.array([[7.8, 7.0, 6.1, 9.4]])
        self.capital_access = np.repeat(np.array([[0.55, 0.74, 0.98, 0.90]]), n, axis=0)
        self.payment = np.full((n, 4), 0.995)
        self.deposit_stability = np.repeat(np.array([[0.93, 0.91, 0.90, 0.94]]), n, axis=0)
        self.asset_price = np.ones((n, 4))
        self.financial_variance = np.repeat(np.array([[0.022, 0.026, 0.016, 0.019]]), n, axis=0)
        self.money_base = np.repeat(np.array([[0.34, 0.46, 0.18, 1.05]]), n, axis=0)
        self.expected_inflation = np.repeat(np.array([[0.020, 0.020, 0.025, 0.020]]), n, axis=0)
        self.bank_crisis = np.zeros((n, 4), dtype=bool)
        self.sovereign_restructured = np.zeros((n, 4), dtype=bool)
        self.payment_event = np.zeros((n, 4), dtype=bool)
        self.cumulative_recap = np.zeros((n, 4))
        self.cumulative_monetization = np.zeros((n, 4))
        self.cumulative_reserve_use = np.zeros((n, 4))
        self.pending_aid_out = np.zeros((n, 4))
        self.pending_aid_in = np.zeros((n, 4))
        self.pending_external_aid = np.zeros(n)
        self.cumulative_actual_aid_out = np.zeros((n, 4))
        self.cumulative_actual_aid_in = np.zeros((n, 4))
        self.max_bilateral_aid_residual = 0.0
        self.snapshots: dict[int, dict[str, np.ndarray]] = {}
        self.last_factors = None

        # Failure of a source country's financial rails impairs destinations.
        self.payment_network = np.array([
            [0.00, 0.10, 0.07, 0.10],
            [0.05, 0.00, 0.04, 0.05],
            [0.20, 0.26, 0.00, 0.22],
            [0.06, 0.13, 0.07, 0.00],
        ])

    def update(
        self,
        month: int,
        actor_damage: np.ndarray,
        actor_shortage: np.ndarray,
        actor_mobilization: np.ndarray,
        firm_distress: np.ndarray,
        sector_output: np.ndarray,
        trade_gdp_factor: np.ndarray,
        trade_industry_factor: np.ndarray,
        trade_inflation: np.ndarray,
        trade_loss: np.ndarray,
        macro_trade_balance: np.ndarray,
        macro_government_balance: np.ndarray,
        us_level: float,
        japan_level: float,
    ) -> FinancialFactors:
        dt = 1.0 / 12.0
        n = self.paths
        us_path = np.broadcast_to(np.asarray(us_level, dtype=float), (n,))
        japan_path = np.broadcast_to(np.asarray(japan_level, dtype=float), (n,))
        participation = np.column_stack((np.ones(n), np.ones(n), us_path, japan_path))
        output = np.mean(sector_output, axis=2)
        distress = np.mean(firm_distress, axis=2)
        gdp_loss = np.clip(1.0 - trade_gdp_factor + 0.16 * (1.0 - output), 0.0, 0.95)
        inflation_input = np.clip(trade_inflation, -0.02, 0.50)
        common_trade = np.clip(trade_loss[:, None], 0.0, 0.90)
        external_deficit = np.maximum(-macro_trade_balance, 0.0)
        external_surplus = np.maximum(macro_trade_balance, 0.0)
        government_gap = np.maximum(-macro_government_balance, 0.0)
        government_surplus = np.maximum(macro_government_balance, 0.0)

        # Common settlement/cyber shocks and actor-specific freezes are rare
        # jump events. They can be intercepted or repaired, so they are not
        # absorbing states.
        common_event = self.rng.random(n) < (0.0007 + 0.0020 * trade_loss)
        common_jump = common_event[:, None] * self.rng.lognormal(-2.7, 0.55, (n, 1))
        idio_prob = np.array([0.0018, 0.0026, 0.0008, 0.0011])[None, :] * participation
        idio_event = self.rng.random((n, 4)) < idio_prob
        idio_jump = idio_event * self.rng.lognormal(-3.0, 0.55, (n, 4))
        direct_payment_shock = common_jump * np.array([[0.65, 0.90, 0.45, 0.58]]) + idio_jump
        network_shock = direct_payment_shock @ self.payment_network
        payment_shock = np.clip(direct_payment_shock + network_shock, 0.0, 0.60)
        self.payment_event |= payment_shock > 0.08

        sanction = np.clip(
            common_trade * np.array([[0.78, 0.72, 0.16, 0.28]])
            + actor_damage * np.array([[0.18, 0.25, 0.08, 0.12]]), 0.0, 1.0
        )
        target_access = np.clip(
            np.column_stack((
                np.full(n, 0.55),
                0.42 + 0.46 * (0.55 * us_path + 0.45 * japan_path),
                np.full(n, 0.98),
                np.full(n, 0.90),
            )) - sanction * np.array([[0.42, 0.28, 0.10, 0.16]]),
            0.06,
            0.99,
        )
        target_access = np.clip(
            target_access + 0.10 * external_surplus - 0.16 * external_deficit
            + 0.06 * government_surplus - 0.10 * government_gap,
            0.04, 0.99,
        )
        self.capital_access += 0.10 * (target_access - self.capital_access)

        variance_noise = self.rng.normal(0.0, 1.0, (n, 4))
        self.financial_variance += 0.11 * (np.array([[0.022, 0.026, 0.016, 0.019]]) - self.financial_variance)
        self.financial_variance += 0.035 * np.sqrt(np.maximum(self.financial_variance, 1e-6) * dt) * variance_noise
        self.financial_variance += 0.08 * payment_shock
        self.financial_variance = np.clip(self.financial_variance, 0.002, 0.35)

        target_asset = np.clip(
            1.0 - 0.72 * gdp_loss - 0.28 * distress - 0.22 * sanction
            - 0.15 * actor_damage - 0.18 * payment_shock, 0.18, 1.10
        )
        asset_noise = np.sqrt(self.financial_variance * dt) * self.rng.normal(0.0, 0.06, (n, 4))
        self.asset_price = np.clip(self.asset_price + 0.08 * (target_asset - self.asset_price) + asset_noise, 0.12, 1.20)

        npl_target = np.clip(
            np.array([[0.0152, 0.0014, 0.0100, 0.0120]])
            + 0.13 * gdp_loss + 0.09 * distress + 0.05 * actor_shortage
            + 0.04 * actor_damage + 0.03 * (1.0 - self.asset_price), 0.001, 0.45
        )
        old_npl = self.npl.copy()
        self.npl += 0.07 * (npl_target - self.npl)
        new_bad_loans = np.maximum(self.npl - old_npl, 0.0)

        consolidated_debt = self.debt + np.array([[0.62, 0.24, 0.28, 0.34]]) * self.local_debt
        dollar_privilege = np.array([[0.00, 0.00, 0.48, 0.10]])
        reserve_capacity = np.clip(self.reserve + dollar_privilege, 0.0, 1.2)
        fiscal_limit = np.array([[2.25, 1.50, 3.60, 4.10]]) * (
            0.72 + 0.24 * self.capital_access + 0.18 * reserve_capacity
        )
        debt_pressure = consolidated_debt / np.maximum(fiscal_limit, 0.35)
        spread = np.clip(
            0.004 + 0.026 * debt_pressure ** 2 + 0.020 * (1.0 - self.capital_access)
            + 0.035 * sanction + 0.018 * (1.0 - self.asset_price), 0.002, 0.22
        )
        policy_target = np.clip(
            np.array([[0.015, 0.020, 0.032, 0.008]])
            + 0.42 * np.maximum(inflation_input - 0.025, 0.0)
            - 0.10 * gdp_loss, 0.0, 0.20
        )
        self.policy_rate += 0.08 * (policy_target - self.policy_rate)
        self.sovereign_rate += 0.10 * (self.policy_rate + spread - self.sovereign_rate)
        self.avg_coupon += (self.sovereign_rate - self.avg_coupon) / np.maximum(self.maturity * 12.0, 12.0)

        duration = np.array([[4.2, 5.0, 5.4, 7.1]])
        rate_loss = np.maximum(self.sovereign_rate - self.avg_coupon, 0.0) * duration
        credit_loss = 0.46 * new_bad_loans * self.bank_assets
        market_loss = 0.012 * np.maximum(1.0 - self.asset_price, 0.0) + 0.010 * rate_loss
        pre_recap_capital = self.cet1 + dt * (0.012 - 0.030 * distress) - credit_loss - market_loss
        recap_need = np.maximum(0.075 - pre_recap_capital, 0.0)
        public_recap = recap_need * np.array([[0.78, 0.62, 0.55, 0.68]]) * np.clip(1.15 - debt_pressure, 0.10, 1.0)
        self.cumulative_recap += public_recap
        self.cet1 = np.clip(pre_recap_capital + public_recap, 0.01, 0.22)

        run_pressure = np.clip(
            0.50 * np.maximum(0.09 - self.cet1, 0.0) / 0.09
            + 0.25 * payment_shock + 0.20 * sanction + 0.18 * (1.0 - self.asset_price), 0.0, 1.0
        )
        capital_control = np.array([[0.72, 0.28, 0.08, 0.16]])
        lender_support = np.array([[0.78, 0.66, 0.92, 0.88]])
        deposit_target = np.clip(1.0 - run_pressure * (1.0 - 0.62 * capital_control), 0.25, 1.0)
        self.deposit_stability += 0.12 * (deposit_target - self.deposit_stability)
        lcr_target = np.clip(
            np.array([[1.50, 1.35, 1.25, 1.35]]) - 0.70 * run_pressure
            + 0.32 * lender_support - 0.25 * payment_shock, 0.35, 2.2
        )
        self.lcr += 0.10 * (lcr_target - self.lcr)
        payment_target = np.clip(
            0.995 - 0.48 * payment_shock - 0.22 * run_pressure
            + 0.10 * lender_support - 0.10 * sanction, 0.18, 0.999
        )
        self.payment += 0.18 * (payment_target - self.payment)

        credit = sigmoid(
            22.0 * (self.cet1 - 0.065) + 2.2 * (self.lcr - 0.85)
            - 5.0 * self.npl + 3.0 * (self.payment - 0.75)
        )
        credit = np.clip(credit, 0.08, 1.0)

        war_surcharge = (1.0 - np.exp(-(month + 1) / 30.0)) * np.array([[0.045, 0.040, 0.055, 0.050]])
        bohn = 0.024 * np.maximum(debt_pressure - 0.55, 0.0)
        allocation = np.array([
            [0.42, 0.40, 0.18], [0.33, 0.45, 0.22],
            [0.25, 0.48, 0.27], [0.38, 0.43, 0.19],
        ])[None, :, :]
        tax_target = self.base_tax_rates + (war_surcharge * participation + bohn)[:, :, None] * allocation
        self.tax_rates += 0.045 * (tax_target - self.tax_rates)
        tax_elasticity = np.array([0.90, 1.25, 1.65])[None, None, :]
        tax_mix = np.array([
            [0.38, 0.42, 0.20], [0.32, 0.47, 0.21],
            [0.25, 0.50, 0.25], [0.36, 0.46, 0.18],
        ])[None, :, :]
        collection = np.clip(self.payment * credit * (1.0 - 0.32 * gdp_loss), 0.25, 1.0)
        raw_tax = np.sum(self.tax_rates * np.exp(-tax_elasticity * self.tax_rates) * tax_mix, axis=2)
        raw_base = np.sum(self.base_tax_rates * np.exp(-tax_elasticity * self.base_tax_rates) * tax_mix, axis=2)
        self.tax_revenue = (
            self.base_tax_revenue * (raw_tax / np.maximum(raw_base, 1e-6)) * collection
            + 0.10 * government_surplus - 0.04 * government_gap
        )
        self.tax_revenue = np.clip(self.tax_revenue, 0.005, 0.50)
        tax_distortion = np.sum(tax_mix * tax_elasticity * (self.tax_rates - self.base_tax_rates) ** 2, axis=2)

        war_spending = np.array([[0.075, 0.105, 0.060, 0.070]]) * participation * (0.45 + 0.55 * actor_mobilization)
        bank_bailout = public_recap / dt
        base_primary = np.array([[0.055, 0.012, 0.059, 0.035]])
        aid_out = self.pending_aid_out.copy()
        aid_in = self.pending_aid_in.copy()
        self.pending_aid_out.fill(0.0)
        self.pending_aid_in.fill(0.0)
        self.pending_external_aid.fill(0.0)
        tax_effort = np.maximum(self.tax_revenue - self.base_tax_revenue * collection, 0.0)
        primary_deficit = np.clip(
            base_primary + war_spending + bank_bailout + aid_out
            + 0.45 * government_gap + 0.16 * external_deficit
            - 0.35 * aid_in - tax_effort - 0.20 * government_surplus,
            0.0, 0.55,
        )
        bond_capacity = np.clip(self.capital_access * np.exp(-0.65 * debt_pressure), 0.05, 0.90)
        repression = np.array([[0.30, 0.20, 0.05, 0.22]]) * np.clip(self.deposit_stability, 0.2, 1.0)
        monetization_share = np.clip(1.0 - bond_capacity - repression, 0.03, 0.72)
        shares = bond_capacity + repression + monetization_share
        monetization = primary_deficit * monetization_share / shares
        self.cumulative_monetization += monetization * dt
        self.money_base = np.clip(self.money_base + monetization * dt - 0.02 * (self.money_base - np.array([[0.34, 0.46, 0.18, 1.05]])) * dt, 0.04, 2.8)

        reserve_need = np.clip(
            0.10 * common_trade + 0.08 * run_pressure + 0.06 * sanction
            + 0.12 * external_deficit - 0.04 * external_surplus,
            0.0, 0.25,
        )
        reserve_use = np.minimum(self.reserve, reserve_need * dt * np.array([[1.0, 1.0, 0.15, 0.75]]))
        self.reserve -= reserve_use
        self.cumulative_reserve_use += reserve_use
        reserve_rebuild = np.array([[0.006, 0.008, 0.001, 0.006]]) * trade_gdp_factor * dt
        self.reserve = np.clip(self.reserve + reserve_rebuild, 0.0, 1.2)

        nominal_growth = np.clip(0.018 + inflation_input - 0.08 * gdp_loss - 0.05 * tax_distortion, -0.20, 0.30)
        self.debt = np.clip(
            (self.debt * (1.0 + self.avg_coupon * dt) + primary_deficit * dt)
            / np.maximum(1.0 + nominal_growth * dt, 0.85), 0.05, 6.0
        )
        local_gap = np.maximum(
            self.local_debt * np.array([[0.033, 0.040, 0.043, 0.032]])
            + 0.012 * actor_mobilization - 0.055 * collection * trade_gdp_factor, 0.0
        )
        self.local_debt = np.clip(
            (self.local_debt * (1.0 + 0.03 * dt) + local_gap * dt)
            / np.maximum(1.0 + nominal_growth * dt, 0.85), 0.0, 3.0
        )

        effective_inflation = np.clip(
            inflation_input + 0.32 * monetization + 0.06 * common_trade
            + 0.04 * run_pressure + 0.05 * (1.0 - self.payment), -0.02, 0.80
        )
        self.expected_inflation = 0.82 * self.expected_inflation + 0.18 * effective_inflation

        crisis_hazard = sigmoid(
            -9.2 + 48.0 * np.maximum(0.075 - self.cet1, 0.0)
            + 4.5 * np.maximum(0.90 - self.lcr, 0.0) + 5.0 * run_pressure
            + 3.0 * payment_shock
        )
        new_crisis = self.rng.random((n, 4)) < crisis_hazard * dt
        self.bank_crisis |= new_crisis
        resolution = self.bank_crisis & (self.rng.random((n, 4)) < (0.025 + 0.08 * lender_support) * dt)
        self.bank_crisis &= ~resolution
        self.cet1 = np.where(resolution, np.maximum(self.cet1, 0.085), self.cet1)
        self.payment = np.where(resolution, np.maximum(self.payment, 0.78), self.payment)

        sovereign_hazard = sigmoid(-12.0 + 5.2 * debt_pressure + 2.0 * effective_inflation + 1.4 * sanction)
        new_restructure = (~self.sovereign_restructured) & (self.rng.random((n, 4)) < sovereign_hazard * dt)
        self.sovereign_restructured |= new_restructure
        self.debt = np.where(new_restructure, self.debt * np.array([[0.82, 0.84, 0.90, 0.90]]), self.debt)
        self.capital_access = np.where(new_restructure, self.capital_access * 0.78, self.capital_access)

        consolidated_debt = self.debt + np.array([[0.62, 0.24, 0.28, 0.34]]) * self.local_debt
        debt_pressure = consolidated_debt / np.maximum(fiscal_limit, 0.35)
        stress = np.clip(
            0.18 * np.clip((0.10 - self.cet1) / 0.10, 0.0, 1.0)
            + 0.15 * np.clip((self.npl - 0.01) / 0.15, 0.0, 1.0)
            + 0.14 * (1.0 - credit) + 0.14 * (1.0 - self.payment)
            + 0.15 * np.clip(debt_pressure, 0.0, 1.5) / 1.5
            + 0.10 * (1.0 - self.capital_access) + 0.08 * run_pressure
            + 0.06 * self.bank_crisis, 0.0, 1.0
        )
        fiscal = np.clip(1.0 - 0.48 * debt_pressure - 0.18 * primary_deficit + 0.12 * reserve_capacity, 0.08, 1.0)
        production = np.clip(
            1.0 - 0.28 * (1.0 - credit) - 0.24 * (1.0 - self.payment)
            - 0.16 * stress - 0.08 * tax_distortion, 0.25, 1.0
        )
        aid = np.clip(0.40 * fiscal + 0.30 * self.payment + 0.30 * self.capital_access, 0.10, 1.0)
        factors = FinancialFactors(production, credit, self.payment.copy(), fiscal, aid, stress, effective_inflation)
        self.last_factors = factors

        elapsed = month + 1
        if elapsed in HORIZON_MONTHS:
            self.snapshots[elapsed] = {
                "cet1": self.cet1.copy(), "npl": self.npl.copy(), "lcr": self.lcr.copy(),
                "credit": credit.copy(), "payment": self.payment.copy(), "asset_price": self.asset_price.copy(),
                "debt": self.debt.copy(), "local_debt": self.local_debt.copy(),
                "debt_pressure": debt_pressure.copy(), "capital_access": self.capital_access.copy(),
                "reserve": self.reserve.copy(), "tax_revenue": self.tax_revenue.copy(),
                "policy_rate": self.policy_rate.copy(), "sovereign_rate": self.sovereign_rate.copy(),
                "money_base": self.money_base.copy(), "inflation": effective_inflation.copy(),
                "stress": stress.copy(), "fiscal": fiscal.copy(), "production": production.copy(),
                "bank_crisis": self.bank_crisis.copy(), "sovereign_restructured": self.sovereign_restructured.copy(),
            }
        return factors

    def register_realized_aid(
        self, us_to_taiwan: np.ndarray, japan_to_taiwan: np.ndarray,
        external_minor_to_taiwan: np.ndarray,
    ) -> None:
        """Book one cleared aid flow with an explicit one-month settlement lag."""
        us = np.clip(np.asarray(us_to_taiwan, dtype=float), 0.0, None)
        japan = np.clip(np.asarray(japan_to_taiwan, dtype=float), 0.0, None)
        external = np.clip(np.asarray(external_minor_to_taiwan, dtype=float), 0.0, None)
        self.pending_aid_out[:, 2] += us
        self.pending_aid_out[:, 3] += japan
        self.pending_aid_in[:, 1] += us + japan + external
        self.pending_external_aid += external
        self.cumulative_actual_aid_out[:, 2] += us
        self.cumulative_actual_aid_out[:, 3] += japan
        self.cumulative_actual_aid_in[:, 1] += us + japan + external
        residual = np.max(np.abs(
            self.pending_aid_in[:, 1]
            - self.pending_aid_out[:, 2] - self.pending_aid_out[:, 3]
            - self.pending_external_aid
        ))
        self.max_bilateral_aid_residual = max(self.max_bilateral_aid_residual, float(residual))

    def summary(self) -> dict:
        payload = {}
        for month, snap in self.snapshots.items():
            payload[str(month // 12)] = {}
            for i, actor in enumerate(ACTORS):
                payload[str(month // 12)][actor] = {
                    "cet1_p10_p50_p90": quantiles(snap["cet1"][:, i]),
                    "npl_p10_p50_p90": quantiles(snap["npl"][:, i]),
                    "lcr_p10_p50_p90": quantiles(snap["lcr"][:, i]),
                    "credit_supply_p10_p50_p90": quantiles(snap["credit"][:, i]),
                    "payment_integrity_p10_p50_p90": quantiles(snap["payment"][:, i]),
                    "asset_price_index_p10_p50_p90": quantiles(snap["asset_price"][:, i]),
                    "central_debt_to_gdp_p10_p50_p90": quantiles(snap["debt"][:, i]),
                    "local_debt_to_gdp_p10_p50_p90": quantiles(snap["local_debt"][:, i]),
                    "debt_pressure_p10_p50_p90": quantiles(snap["debt_pressure"][:, i]),
                    "capital_market_access_p10_p50_p90": quantiles(snap["capital_access"][:, i]),
                    "reserve_buffer_p10_p50_p90": quantiles(snap["reserve"][:, i]),
                    "tax_revenue_to_gdp_p10_p50_p90": quantiles(snap["tax_revenue"][:, i]),
                    "policy_rate_p10_p50_p90": quantiles(snap["policy_rate"][:, i]),
                    "sovereign_rate_p10_p50_p90": quantiles(snap["sovereign_rate"][:, i]),
                    "money_base_to_gdp_p10_p50_p90": quantiles(snap["money_base"][:, i]),
                    "effective_inflation_p10_p50_p90": quantiles(snap["inflation"][:, i]),
                    "financial_stress_p10_p50_p90": quantiles(snap["stress"][:, i]),
                    "fiscal_capacity_p10_p50_p90": quantiles(snap["fiscal"][:, i]),
                    "production_finance_factor_p10_p50_p90": quantiles(snap["production"][:, i]),
                    "bank_crisis_active_share": float(np.mean(snap["bank_crisis"][:, i])),
                    "sovereign_restructured_share": float(np.mean(snap["sovereign_restructured"][:, i])),
                }
        payload["cumulative"] = {
            actor: {
                "bank_recapitalization_median": float(np.median(self.cumulative_recap[:, i])),
                "monetization_median": float(np.median(self.cumulative_monetization[:, i])),
                "reserve_use_median": float(np.median(self.cumulative_reserve_use[:, i])),
                "actual_aid_out_median": float(np.median(self.cumulative_actual_aid_out[:, i])),
                "actual_aid_in_median": float(np.median(self.cumulative_actual_aid_in[:, i])),
                "payment_event_share": float(np.mean(self.payment_event[:, i])),
            }
            for i, actor in enumerate(ACTORS)
        }
        return payload
