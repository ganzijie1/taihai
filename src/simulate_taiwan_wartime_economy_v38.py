import csv
import json
import math
import os
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
SEED = 20261007
PATHS = int(os.environ.get("TAIWAN_V38_PATHS", "8000"))
QUARTERS = 40
DT = 0.25
ACTORS = ("china", "united_states", "japan", "taiwan")
SECTORS = ("aircraft", "naval", "munitions_spares")
GROUPS = ("low_income_households", "formal_workers", "firms", "asset_holders", "military_procurement")
# HANK-style group coordinates: liquid wealth, hedgeable assets and skill or
# bargaining power. They map common shocks into heterogeneous behavior.
GROUP_LIQUIDITY = np.array([0.05, 0.28, 0.48, 0.82, 0.18])
GROUP_ASSETS = np.array([0.03, 0.22, 0.58, 0.92, 0.08])
GROUP_SKILL = np.array([0.20, 0.62, 0.74, 0.68, 0.82])
GROUP_WEIGHTS = np.array([0.34, 0.38, 0.12, 0.10, 0.06])
YEARS = (2030, 2050, 2075, 2100)
CASES = ("timely_full", "limited", "delayed", "taiwan_alone")


CASE_PARAMETERS = {
    "timely_full": {"us": 0.92, "japan": 0.76, "taiwan_aid": 0.90, "delay_q": 0},
    "limited": {"us": 0.38, "japan": 0.18, "taiwan_aid": 0.42, "delay_q": 1},
    "delayed": {"us": 0.76, "japan": 0.55, "taiwan_aid": 0.72, "delay_q": 4},
    "taiwan_alone": {"us": 0.0, "japan": 0.0, "taiwan_aid": 0.06, "delay_q": 99},
}


SCENARIOS = {
    "central": {"china_support": 0.12, "coalition_surge": 1.0, "blockade": 1.0, "mobilization": 1.0},
    "china_external_support_cut": {"china_support": 0.02, "coalition_surge": 1.0, "blockade": 1.05, "mobilization": 1.0},
    "coalition_industrial_surge": {"china_support": 0.12, "coalition_surge": 1.30, "blockade": 1.0, "mobilization": 1.0},
    "prolonged_maritime_disruption": {"china_support": 0.08, "coalition_surge": 0.92, "blockade": 1.35, "mobilization": 1.0},
    "full_mobilization_all": {"china_support": 0.14, "coalition_surge": 1.25, "blockade": 1.0, "mobilization": 1.28},
}


CALIBRATION_REGISTRY = {
    "direct_public_observations": {
        "cross_economy_public_debt": "IMF Global Debt Database and Fiscal Monitor; BIS general-government credit statistics",
        "china_official_local_debt": "PRC Ministry of Finance local-government debt balance, issuance, maturity and coupon reports",
        "united_states_fiscal_monetary": "US Treasury FiscalData, CBO, Federal Reserve Financial Accounts and FRED",
        "japan_fiscal_monetary": "Japan Ministry of Finance debt statistics and Bank of Japan Flow of Funds/JGB holdings",
        "taiwan_fiscal_external": "Taiwan MOF debt statistics, DGBAS national accounts and CBC reserves/international investment position",
        "tax_and_trade": "OECD Revenue Statistics, IMF GFS/IFS/DOTS, World Bank WDI and official customs statistics",
    },
    "derived_or_reconciled": {
        "china_general_government_perimeter": "Reconcile official central/local debt with IMF general-government and LGFV-adjusted perimeter; report both bounds",
        "subnational_guarantee_weight": "Estimated from historical bailouts, debt swaps, bank exposure and intergovernmental transfers",
        "effective_tax_bases": "Convert statutory and cash-revenue series into consumption, labour and capital effective rates",
        "average_coupon_and_maturity": "Debt-stock weighted average from issuance, maturity and holder distributions",
    },
    "structural_priors_not_directly_observed": {
        "wartime_fiscal_limit": "Nonlinear stochastic prior informed by historical restructurings; never treated as an observed threshold",
        "wartime_mobilization_ceiling": "Scenario prior constrained by industrial inputs, lead times and historical mobilization cases",
        "sanction_and_blockade_jump_size": "Scenario distribution because the event has no modern four-party observational analogue",
        "wartime_policy_reaction": "Regime-dependent prior; peacetime Taylor-rule estimates do not identify total-war policy",
    },
    "public_source_urls": [
        "https://www.imf.org/external/datamapper/GDD/2025",
        "https://www.bis.org/statistics/totcredit.htm",
        "https://zwgls.mof.gov.cn/tjsj/",
        "https://fiscaldata.treasury.gov/",
        "https://www.federalreserve.gov/releases/z1/",
        "https://www.mof.go.jp/english/policy/jgbs/",
        "https://www.boj.or.jp/en/statistics/sj/",
        "https://www.cbc.gov.tw/en/np-699-2.html",
        "https://www.dgbas.gov.tw/",
        "https://stats.oecd.org/",
    ],
}


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -35.0, 35.0)))


def logit(p):
    p = np.clip(p, 1e-5, 1.0 - 1e-5)
    return np.log(p / (1.0 - p))


def q(values):
    return [float(v) for v in np.quantile(values, (0.10, 0.50, 0.90))]


def load_campaign():
    payload = json.loads((OUT / "台海V3.6_GIS走廊多层备件搜索与军事经济_仿真摘要.json").read_text(encoding="utf-8"))
    return {(r["conflict_year"], r["case"]): r for r in payload["scenario_envelopes"]}


def simulate_one(year, case_key, scenario_name, scenario, campaign_row, seed):
    rng = np.random.default_rng(seed)
    n = PATHS
    case = CASE_PARAMETERS[case_key]

    # Indexed initial stocks and quarterly peacetime output. Sector detail is
    # retained because ships, aircraft and munitions have different lead times.
    initial_inventory = np.array([
        [1.00, 1.00, 1.00],
        [1.18, 1.25, 1.05],
        [0.58, 0.64, 0.52],
        [0.31, 0.22, 0.42],
    ])
    base_output = np.array([
        [0.024, 0.010, 0.062],
        [0.028, 0.014, 0.072],
        [0.012, 0.008, 0.026],
        [0.004, 0.002, 0.014],
    ])
    inventory = np.tile(initial_inventory, (n, 1, 1)) * rng.lognormal(0.0, 0.10, (n, 4, 3))
    capacity = np.ones((n, 4, 3)) * rng.lognormal(0.0, 0.08, (n, 4, 3))
    cumulative_output = np.full((n, 4, 3), 0.10)

    # Mobilization ceilings are multiples of peacetime output, not instant
    # production. Naval construction receives the lowest ramp rate.
    ceilings = np.array([
        [3.3, 2.2, 4.2],
        [4.3, 2.8, 5.0],
        [2.7, 2.1, 3.2],
        [1.8, 1.5, 2.4],
    ]) * scenario["mobilization"]
    ramp = np.array([
        [0.11, 0.050, 0.16],
        [0.14, 0.060, 0.19],
        [0.09, 0.045, 0.12],
        [0.075, 0.035, 0.10],
    ])
    ramp[:, 0] *= np.array([1.0, scenario["coalition_surge"], scenario["coalition_surge"], scenario["coalition_surge"]])
    ramp[:, 2] *= np.array([1.0, scenario["coalition_surge"], scenario["coalition_surge"], scenario["coalition_surge"]])

    domestic_inputs = np.array([0.78, 0.82, 0.28, 0.16])
    import_component = 1.0 - domestic_inputs
    trade_access = np.tile(np.array([0.82, 0.96, 0.92, 0.78]), (n, 1))
    gdp = np.ones((n, 4))
    inflation = np.tile(np.array([0.018, 0.026, 0.024, 0.021]), (n, 1))
    expected_inflation = inflation.copy()
    effective_inflation = inflation.copy()
    inflation_overhang = np.zeros((n, 4))
    shortage_index = np.zeros((n, 4))
    sanction_state = np.tile(np.array([0.15, 0.03, 0.05, 0.15]), (n, 1))
    deanchored = np.zeros((n, 4), dtype=bool)
    inflation_regime = np.zeros((n, 4), dtype=np.int8)
    group_expectations = np.repeat(inflation[:, :, None], len(GROUPS), axis=2)
    group_price_index = np.ones((n, 4, len(GROUPS)))
    group_real_position = np.ones((n, 4, len(GROUPS)))
    debt_ratio = np.tile(np.array([0.88, 1.23, 2.45, 0.34]), (n, 1))
    # Separate subnational balance sheets.  For China this includes the
    # official local-government layer plus a stochastic LGFV/contingent-debt
    # perimeter; it is not asserted to be one observed point estimate.
    local_debt_ratio = np.tile(np.array([0.62, 0.16, 0.34, 0.08]), (n, 1))
    local_debt_ratio *= rng.lognormal(0.0, np.array([0.16, 0.08, 0.10, 0.08]), (n, 4))
    local_coupon_rate = np.tile(np.array([0.033, 0.043, 0.032, 0.040]), (n, 1))
    capital_market_access = np.tile(np.array([0.42, 0.96, 0.88, 0.68]), (n, 1))
    sovereign_rate = np.tile(np.array([0.035, 0.040, 0.035, 0.042]), (n, 1))
    policy_rate = np.tile(np.array([0.018, 0.042, 0.012, 0.020]), (n, 1))
    coupon_rate = np.tile(np.array([0.029, 0.034, 0.014, 0.025]), (n, 1))
    debt_maturity = np.array([7.8, 6.1, 9.4, 7.0])[None, :]
    money_base_ratio = np.tile(np.array([0.34, 0.18, 1.05, 0.46]), (n, 1))
    fiscal_limit_base = rng.lognormal(
        np.log(np.array([2.25, 3.60, 4.10, 1.50])),
        np.array([0.12, 0.10, 0.10, 0.12]),
        (n, 4),
    )
    fiscal_limit = fiscal_limit_base.copy()
    debt_pressure = np.zeros((n, 4))
    fiscal_dominance = np.zeros((n, 4))
    qe_ycc_intensity = np.zeros((n, 4))
    money_growth = np.zeros((n, 4))
    seigniorage_gap = np.zeros((n, 4))
    tax_base_rates = np.array([
        [0.105, 0.105, 0.055],  # China: consumption, labour/social, capital
        [0.070, 0.145, 0.055],  # United States
        [0.105, 0.175, 0.070],  # Japan
        [0.075, 0.095, 0.060],  # Taiwan
    ])
    tax_rates = np.tile(tax_base_rates[None, :, :], (n, 1, 1))
    tax_revenue = np.zeros((n, 4))
    tax_distortion = np.zeros((n, 4))
    central_bailout = np.zeros((n, 4))
    restructured = np.zeros((n, 4), dtype=bool)
    local_restructured = np.zeros((n, 4), dtype=bool)
    defense_wacc = np.tile(np.array([0.070, 0.065, 0.060, 0.075]), (n, 1))
    mm_friction_wedge = np.zeros((n, 4))
    financial_variance = np.tile(np.array([0.025, 0.015, 0.018, 0.028]), (n, 1))
    credit_default_hazard = np.zeros((n, 4))
    capm_equity_cost = np.zeros((n, 4))
    apt_equity_cost = np.zeros((n, 4))
    derivative_hedge_ratio = np.zeros((n, 4))
    real_option_exercise = np.ones((n, 4, 3))
    cumulative_deficit = np.zeros((n, 4))
    real_war_spending = np.zeros((n, 4))
    destroyed_capacity = np.zeros((n, 4, 3))
    cumulative_consumption = np.zeros((n, 4, 3))
    cumulative_production = np.zeros((n, 4, 3))
    cumulative_financing = np.zeros((n, 4, 4))  # bonds, repression, money, external relief
    shock_variance = np.zeros((n, 4, 5))  # sanction, supply, fiscal, FX and demand

    # Initial combat intensity and loss demand. Allied demand ramps after the
    # case-specific entry delay; Taiwan and China begin at once.
    base_consumption = np.array([
        [0.034, 0.019, 0.100],
        [0.026, 0.016, 0.074],
        [0.014, 0.010, 0.035],
        [0.025, 0.017, 0.068],
    ])
    participation = np.array([1.0, case["us"], case["japan"], 1.0])
    snapshots = {}
    readiness_history = []
    baseline_readiness_history = []

    for quarter in range(QUARTERS):
        conflict_years = quarter * DT
        ally_active = 1.0 if quarter >= case["delay_q"] else 0.12
        active_participation = participation.copy()
        active_participation[1:3] *= ally_active
        intensity_decay = 0.72 + 0.28 * math.exp(-conflict_years / 4.0)
        tempo_noise = rng.lognormal(0.0, 0.10, (n, 4, 3))

        # Recursive SVAR impact ordering: sanctions are contemporaneously
        # predetermined; supply, fiscal, FX and demand shocks can transmit to
        # downstream prices within the quarter. This is a structural scenario
        # map, not an empirical Taiwan-war SVAR estimate.
        structural_innovations = rng.normal(0.0, 1.0, (n, 4, 5))
        svar_impact = np.array([
            [1.00, 0.00, 0.00, 0.00, 0.00],
            [0.28, 1.00, 0.00, 0.00, 0.00],
            [0.12, 0.08, 1.00, 0.00, 0.00],
            [0.22, 0.18, 0.25, 1.00, 0.00],
            [0.16, 0.24, 0.22, 0.30, 1.00],
        ])
        svar_shocks = structural_innovations @ svar_impact.T
        shock_variance += np.square(svar_shocks)

        # Sanctions arrive as correlated jumps and then diffuse through the
        # trade-finance network. Reciprocal effects reach coalition economies,
        # while Taiwan remains directly exposed to maritime disruption.
        common_event = rng.random(n) < (0.030 + 0.014 * scenario["blockade"])
        common_size = common_event * rng.lognormal(-1.9 + 0.08 * svar_shocks[:, 0, 0], 0.45, n)
        idiosyncratic = rng.random((n, 4)) < np.array([0.030, 0.012, 0.016, 0.025])[None, :]
        idiosyncratic = idiosyncratic * rng.lognormal(-2.4, 0.50, (n, 4))
        direct_jump = common_size[:, None] * np.array([1.00, 0.20, 0.38, 0.72])[None, :] + idiosyncratic
        diffusion = direct_jump @ np.array([
            [0.00, 0.10, 0.16, 0.12],
            [0.05, 0.00, 0.14, 0.18],
            [0.06, 0.12, 0.00, 0.14],
            [0.08, 0.15, 0.16, 0.00],
        ])
        sanction_state = np.clip(0.92 * sanction_state + direct_jump + diffusion, 0.0, 1.5)

        # Global capital-market access moves much faster for the coalition.
        # Taiwan receives financial guarantees when allies participate, while
        # physical delivery remains separately constrained by blockade.
        access_target = np.column_stack((
            np.clip(0.42 - 0.22 * sanction_state[:, 0], 0.08, 0.55),
            np.clip(0.97 - 0.025 * sanction_state[:, 1], 0.80, 0.99),
            np.clip(0.90 - 0.055 * sanction_state[:, 2] + 0.04 * case["us"] * ally_active, 0.68, 0.96),
            np.clip(0.34 - 0.10 * sanction_state[:, 3] + 0.58 * case["taiwan_aid"] * ally_active, 0.12, 0.94),
        ))
        access_speed = np.array([0.055, 0.20, 0.16, 0.15])[None, :]
        capital_market_access += access_speed * (access_target - capital_market_access)
        capital_market_access = np.clip(capital_market_access, 0.05, 0.99)

        # Square-root stochastic volatility with sanction jumps. It controls
        # the tail of financing costs without allowing negative variance.
        variance_shock = rng.normal(0.0, 1.0, (n, 4))
        variance_jump = common_event[:, None] * np.array([0.028, 0.010, 0.015, 0.024])[None, :]
        financial_variance += 1.15 * (np.array([0.025, 0.015, 0.018, 0.028])[None, :] - financial_variance) * DT
        financial_variance += 0.11 * np.sqrt(np.maximum(financial_variance, 1e-6) * DT) * variance_shock + variance_jump
        financial_variance = np.clip(financial_variance, 0.002, 0.30)

        # Nonlinear fiscal limits use consolidated exposure rather than central
        # debt alone.  Guarantee weights encode how much subnational debt is
        # expected to migrate to the sovereign balance sheet in a crisis.
        guarantee_weight = np.array([0.62, 0.28, 0.34, 0.24])[None, :]
        consolidated_debt = debt_ratio + guarantee_weight * local_debt_ratio
        reserve_capacity = np.array([0.00, 0.28, 0.13, 0.18])[None, :]
        fiscal_limit = fiscal_limit_base * (
            0.80 + 0.32 * capital_market_access + reserve_capacity
            - 0.11 * sanction_state - 0.08 * shortage_index
        )
        fiscal_limit = np.clip(fiscal_limit, 0.65, 5.5)
        debt_pressure = consolidated_debt / fiscal_limit
        fiscal_stress_lag = np.maximum(debt_pressure - 0.52, 0.0)

        # Monetary regimes differ.  A Taylor-rule target is capped by the rate
        # the fiscal authority can service.  Binding caps generate fiscal
        # dominance and QE/YCC rather than assuming identical central banks.
        natural_rate = np.array([0.014, 0.021, 0.004, 0.013])[None, :] + 0.006 * (
            consolidated_debt - np.array([1.15, 1.35, 2.55, 0.45])[None, :]
        )
        desired_policy = natural_rate + 1.45 * (expected_inflation - np.array([0.025, 0.030, 0.028, 0.030])[None, :])
        desired_policy += 0.20 * np.maximum(gdp - 1.0, -0.25)
        affordable_rate = np.clip(0.095 - 0.023 * consolidated_debt + 0.030 * capital_market_access, 0.008, 0.11)
        fiscal_dominance = sigmoid(18.0 * (desired_policy - affordable_rate))
        policy_target = np.minimum(desired_policy, affordable_rate)
        policy_rate = np.clip(0.76 * policy_rate + 0.24 * policy_target, -0.005, 0.18)
        qe_ycc_intensity = fiscal_dominance * np.array([0.68, 0.45, 0.82, 0.52])[None, :]

        # MM frictionless benchmark plus wartime wedges. Sovereign funding cost
        # feeds the WACC of defense suppliers; reserve-currency and alliance
        # guarantees lower the wedge but cannot create missing physical inputs.
        risk_free = policy_rate + np.array([0.006, 0.004, 0.003, 0.005])[None, :]
        reserve_advantage = np.array([0.000, 0.018, 0.007, 0.004])[None, :]
        credit_default_hazard = sigmoid(
            -5.2 + 1.7 * fiscal_stress_lag + 4.0 * expected_inflation
            + 0.85 * sanction_state + 1.35 * (1.0 - capital_market_access)
            + 1.8 * np.sqrt(financial_variance)
        )
        recovery_rate = np.array([0.42, 0.62, 0.58, 0.48])[None, :]
        credit_spread = credit_default_hazard * (1.0 - recovery_rate)
        liquidity_spread = 0.030 * (1.0 - capital_market_access) + 0.12 * financial_variance
        sovereign_rate = risk_free + credit_spread + liquidity_spread + 0.10 * expected_inflation - reserve_advantage
        sovereign_rate -= 0.020 * qe_ycc_intensity
        sovereign_rate = np.clip(sovereign_rate, 0.005, 0.45)
        coupon_rate += (sovereign_rate - coupon_rate) * DT / debt_maturity
        coupon_rate = np.clip(coupon_rate, 0.002, 0.30)
        local_spread = np.array([0.018, 0.010, 0.008, 0.011])[None, :] + 0.035 * np.maximum(
            debt_pressure - 0.55, 0.0
        )
        local_coupon_rate += (sovereign_rate + local_spread - local_coupon_rate) * DT / np.array([5.2, 5.8, 7.6, 5.5])[None, :]
        debt_weight = np.array([0.66, 0.72, 0.63, 0.58])[None, :]
        market_beta = np.array([1.00, 0.92, 0.88, 1.08])[None, :]
        market_premium = np.array([0.060, 0.050, 0.052, 0.065])[None, :]
        capm_equity_cost = risk_free + market_beta * market_premium
        # APT adds war, sanction, supply-chain and volatility factors.
        apt_equity_cost = capm_equity_cost + 0.030 * sanction_state
        apt_equity_cost += 0.022 * (1.0 - capital_market_access) + 0.16 * financial_variance
        apt_equity_cost += 0.018 * import_component[None, :] * (1.0 - trade_access)
        equity_cost = apt_equity_cost
        tax_shield = np.array([0.18, 0.21, 0.23, 0.20])[None, :]
        frictionless_mm = debt_weight * risk_free * (1.0 - tax_shield) + (1.0 - debt_weight) * capm_equity_cost
        defense_wacc = debt_weight * sovereign_rate * (1.0 - tax_shield) + (1.0 - debt_weight) * equity_cost
        mm_friction_wedge = np.maximum(defense_wacc - frictionless_mm, 0.0)
        derivative_hedge_ratio = np.array([0.28, 0.78, 0.70, 0.58])[None, :] * capital_market_access
        derivative_hedge_ratio *= np.exp(-1.8 * financial_variance - 0.20 * sanction_state)
        derivative_hedge_ratio = np.clip(derivative_hedge_ratio, 0.0, 0.82)

        # Trade access combines sanctions, maritime disruption and each actor's
        # ability to reroute. It never becomes a binary on/off switch.
        target_trade = np.array([
            max(0.18, 0.48 - 0.18 * scenario["blockade"]),
            0.88 - 0.04 * scenario["blockade"],
            max(0.42, 0.78 - 0.15 * scenario["blockade"] * active_participation[2]),
            max(0.08, 0.50 - 0.30 * scenario["blockade"]),
        ])
        target_trade = target_trade[None, :] - sanction_state * np.array([0.16, 0.06, 0.10, 0.18])[None, :]
        target_trade = np.clip(target_trade, 0.04, 0.96)
        trade_speed = np.array([0.09, 0.035, 0.065, 0.13])
        trade_access += trade_speed * (target_trade - trade_access)
        trade_access += rng.normal(0.0, 0.006, trade_access.shape) - 0.003 * svar_shocks[:, :, 1]
        trade_access = np.clip(trade_access, 0.04, 1.0)

        # External resource and equipment support. China has a narrow support
        # set; Taiwan's support depends strongly on the alliance regime.
        external_input = np.zeros((n, 4))
        external_input[:, 0] = scenario["china_support"] * rng.uniform(0.75, 1.25, n)
        external_input[:, 1] = 0.12 * scenario["coalition_surge"]
        external_input[:, 2] = 0.20 * case["us"] * scenario["coalition_surge"] * ally_active
        external_input[:, 3] = 0.34 * case["taiwan_aid"] * scenario["coalition_surge"] * ally_active
        input_availability = domestic_inputs[None, :] + import_component[None, :] * trade_access + external_input
        input_availability = np.clip(input_availability, 0.08, 1.12)

        mobilization_target = ceilings[None, :, :] * np.array([1.0, active_participation[1], active_participation[2], 1.0])[None, :, None]
        mobilization_target[:, 1:3, :] += 1.0 - np.array([active_participation[1], active_participation[2]])[None, :, None]
        fiscal_real_factor = np.exp(-1.6 * np.maximum(effective_inflation - 0.08, 0.0))
        fiscal_real_factor *= np.exp(-0.22 * np.maximum(debt_pressure - 0.72, 0.0))
        fiscal_real_factor *= np.exp(-2.2 * mm_friction_wedge)
        fiscal_real_factor *= np.take(np.array([1.00, 0.98, 0.92, 0.82, 0.65]), inflation_regime)
        # Real-option exercise: uncertainty raises the value of waiting, while
        # inventory scarcity, state guarantees and expected demand induce
        # lumpy capacity investment. Ships face the highest option threshold.
        inventory_gap = 1.0 - np.exp(np.mean(np.log(np.clip(inventory / initial_inventory[None, :, :], 0.02, 3.0)), axis=2))
        state_guarantee = np.array([0.68, 0.58, 0.56, 0.46])[None, :]
        state_guarantee[:, 1:4] += 0.14 * case["us"] * ally_active
        shadow_value = 0.65 + 0.70 * np.maximum(inventory_gap, 0.0) + 0.25 * active_participation[None, :]
        option_threshold = 0.55 + 1.8 * defense_wacc + 0.90 * np.sqrt(financial_variance) - 0.30 * state_guarantee
        sector_threshold = option_threshold[:, :, None] + np.array([0.00, 0.18, -0.08])[None, None, :]
        real_option_exercise = sigmoid(5.0 * (shadow_value[:, :, None] - sector_threshold))
        capacity += ramp[None, :, :] * real_option_exercise * (mobilization_target - capacity) * fiscal_real_factor[:, :, None]

        # Strategic strikes and disruption damage production plant. Taiwan is
        # the most exposed; continental US the least exposed.
        strike_exposure = np.array([0.0028, 0.00035, 0.0016, 0.0085])
        strike_exposure *= np.array([1.0, active_participation[1], active_participation[2], 1.0])
        repair_fraction = np.array([0.050, 0.075, 0.065, 0.045])
        damage = strike_exposure[None, :, None] * rng.lognormal(0.0, 0.30, (n, 4, 3))
        damage *= np.array([0.8, 1.25, 1.0])[None, None, :]
        destroyed_capacity += damage
        capacity *= 1.0 - np.minimum(damage, 0.08)
        repaired = repair_fraction[None, :, None] * destroyed_capacity
        destroyed_capacity -= repaired
        capacity += repaired
        capacity = np.clip(capacity, 0.18, ceilings[None, :, :] * 1.08)

        learning = np.clip(np.power(1.0 + cumulative_output, 0.085), 1.0, 1.32)
        bottleneck = input_availability[:, :, None]
        output = base_output[None, :, :] * capacity * learning * bottleneck * fiscal_real_factor[:, :, None]
        output *= rng.lognormal(0.0, 0.055, output.shape)

        # Direct equipment aid is additional to raw-material support and is
        # concentrated in munitions/spares rather than major ships.
        equipment_aid = np.zeros_like(output)
        equipment_aid[:, 0, 2] = 0.006 * scenario["china_support"]
        equipment_aid[:, 2, :] = np.array([0.0015, 0.0004, 0.0040]) * case["us"] * ally_active
        equipment_aid[:, 3, :] = np.array([0.0020, 0.0003, 0.0080]) * case["taiwan_aid"] * ally_active

        consumption = base_consumption[None, :, :] * active_participation[None, :, None] * intensity_decay * tempo_noise
        consumption[:, 0, :] *= 1.0 + 0.10 * scenario["blockade"]
        consumption[:, 3, :] *= 1.08
        available = inventory + output + equipment_aid
        actual_consumption = np.minimum(consumption, available)
        inventory = np.maximum(available - actual_consumption, 0.0)
        cumulative_consumption += actual_consumption
        cumulative_production += output + equipment_aid
        cumulative_output += output

        # Fiscal-inflation block. A hybrid Phillips curve is augmented with
        # fiscal dominance, exchange-rate/import pass-through and wartime price
        # controls. Controls split observed inflation from a latent monetary
        # overhang and shortages; they do not remove the real resource gap.
        mobilization_cost = np.sum(output / np.maximum(base_output[None, :, :], 1e-8), axis=2) / 3.0
        war_spending_share = np.array([0.075, 0.060, 0.070, 0.105])[None, :] * active_participation[None, :]
        war_spending_share = war_spending_share * np.sqrt(np.clip(mobilization_cost, 0.45, 4.5))
        diversion_cost = np.array([0.018, 0.006, 0.012, 0.025])[None, :]
        civilian_diversion = diversion_cost * np.maximum(mobilization_cost - 1.0, 0.0)
        aid_fiscal_relief = external_input * np.array([0.10, 0.05, 0.10, 0.18])[None, :]
        # Barro-style tax smoothing with a Bohn response to debt.  Distinct
        # consumption, labour and capital bases allow each political economy
        # to finance war differently and make the Laffer distortion explicit.
        war_surcharge = np.array([0.050, 0.060, 0.050, 0.045])[None, :]
        war_surcharge *= (1.0 - np.exp(-conflict_years / 2.5))
        bohn_response = 0.026 * np.maximum(debt_pressure - 0.58, 0.0)
        tax_allocation = np.array([
            [0.42, 0.40, 0.18], [0.25, 0.48, 0.27],
            [0.38, 0.43, 0.19], [0.33, 0.45, 0.22],
        ])
        tax_target = tax_base_rates[None, :, :] + (war_surcharge + bohn_response)[:, :, None] * tax_allocation[None, :, :]
        tax_rates += 0.16 * (tax_target - tax_rates)
        tax_rates = np.clip(tax_rates, 0.01, 0.55)
        tax_elasticity = np.array([0.90, 1.25, 1.65])[None, None, :]
        tax_mix = np.array([
            [0.38, 0.42, 0.20], [0.25, 0.50, 0.25],
            [0.36, 0.46, 0.18], [0.32, 0.47, 0.21],
        ])[None, :, :]
        collection_capacity = np.array([0.87, 0.94, 0.93, 0.92])[None, :] * np.clip(
            1.0 - 0.24 * shortage_index - 0.12 * sanction_state, 0.48, 1.0
        )
        tax_components = tax_rates * np.exp(-tax_elasticity * tax_rates) * collection_capacity[:, :, None]
        revenue_scale = np.array([1.85, 1.90, 2.05, 1.85])[None, :]
        tax_revenue = revenue_scale * np.sum(tax_components * tax_mix, axis=2)
        baseline_tax_components = tax_base_rates[None, :, :] * np.exp(-tax_elasticity * tax_base_rates[None, :, :])
        baseline_tax_revenue = revenue_scale * np.sum(baseline_tax_components * tax_mix, axis=2) * collection_capacity
        tax_effort = np.maximum(tax_revenue - baseline_tax_revenue, 0.0)
        tax_distortion = np.sum(tax_mix * tax_elasticity * np.square(tax_rates - tax_base_rates[None, :, :]), axis=2)

        # Subnational debt service is funded by local taxes and, in China's
        # case, a land/property-related revenue proxy that contracts under a
        # trade and output shock.  Unfunded gaps are split among rollover,
        # financial repression and central-government bailouts.
        land_revenue_index = np.clip(gdp * trade_access * np.exp(-0.09 * conflict_years), 0.08, 1.25)
        local_revenue_capacity = np.array([0.075, 0.040, 0.055, 0.035])[None, :]
        local_revenue = local_revenue_capacity * land_revenue_index
        local_service = local_coupon_rate * local_debt_ratio
        local_war_burden = np.array([0.018, 0.006, 0.010, 0.012])[None, :] * active_participation[None, :]
        local_gap = np.maximum(local_service + local_war_burden - local_revenue, 0.0)
        bailout_propensity = np.array([0.58, 0.28, 0.38, 0.32])[None, :]
        central_bailout = bailout_propensity * local_gap * sigmoid(8.0 * (debt_pressure - 0.52))
        primary_deficit = np.maximum(war_spending_share + central_bailout - tax_effort - aid_fiscal_relief, 0.0)
        import_gap = 1.0 - trade_access
        inflation_target = np.array([0.025, 0.030, 0.028, 0.030])[None, :]
        monetary_credibility = np.array([0.62, 0.86, 0.80, 0.72])[None, :]
        fiscal_stress = np.maximum(debt_pressure - 0.52, 0.0)
        fiscal_stress_effective = fiscal_stress * np.array([1.0, 0.48, 0.38, 0.90])[None, :]
        bond_share = np.array([0.48, 0.78, 0.66, 0.58])[None, :] * capital_market_access
        bond_share *= np.exp(-0.45 * fiscal_stress_effective - 2.0 * np.maximum(sovereign_rate - 0.06, 0.0))
        repression_share = np.array([0.34, 0.08, 0.25, 0.28])[None, :] * np.exp(-0.20 * shortage_index)
        monetization_share = np.clip(1.0 - bond_share - repression_share, 0.04, 0.75)
        financing_total = bond_share + repression_share + monetization_share
        bond_finance = primary_deficit * bond_share / financing_total
        repression_finance = primary_deficit * repression_share / financing_total
        monetization = primary_deficit * monetization_share / financing_total
        # Cagan money demand: once desired monetization exceeds seigniorage
        # capacity, the gap feeds expected inflation nonlinearly.
        real_money_demand = np.maximum(money_base_ratio * np.exp(-2.8 * expected_inflation), 0.025)
        money_growth = monetization / real_money_demand + 0.04 * qe_ycc_intensity
        seigniorage_capacity = np.maximum(effective_inflation, 0.0) * real_money_demand
        seigniorage_gap = np.maximum(monetization - seigniorage_capacity, 0.0)
        money_base_ratio = np.clip(money_base_ratio + monetization * DT - 0.08 * money_base_ratio * DT, 0.04, 2.5)
        cumulative_financing[:, :, 0] += bond_finance * DT
        cumulative_financing[:, :, 1] += repression_finance * DT
        cumulative_financing[:, :, 2] += monetization * DT
        cumulative_financing[:, :, 3] += aid_fiscal_relief * DT
        credibility_now = np.clip(
            monetary_credibility - 0.50 * fiscal_stress_effective
            - 1.10 * monetization - 0.18 * fiscal_dominance,
            0.12, 0.92,
        )
        expected_inflation = credibility_now * inflation_target + (1.0 - credibility_now) * (
            0.72 * effective_inflation + 0.28 * expected_inflation
        )
        # Heterogeneous expectations create wage, markup, hoarding and capital-
        # flight feedback instead of assuming one representative price setter.
        signal = expected_inflation[:, :, None]
        signal = signal + 0.065 * import_gap[:, :, None] * (1.0 - GROUP_LIQUIDITY)[None, None, :]
        signal = signal + 0.060 * sanction_state[:, :, None] * (1.0 - GROUP_ASSETS)[None, None, :]
        signal = signal + 0.055 * shortage_index[:, :, None] * (1.0 - GROUP_LIQUIDITY)[None, None, :]
        signal[:, :, 2] += 0.035 * fiscal_stress_effective
        signal[:, :, 3] += 0.050 * fiscal_stress_effective
        signal[:, :, 4] += 0.070 * (1.0 - input_availability)
        signal += deanchored[:, :, None] * np.array([0.035, 0.030, 0.045, 0.060, 0.050])[None, None, :]
        group_expectations = 0.62 * group_expectations + 0.38 * signal
        mean_field_expectation = np.sum(group_expectations * GROUP_WEIGHTS[None, None, :], axis=2)
        mpc = 1.0 - GROUP_LIQUIDITY
        hoarding = 0.020 * np.sum(
            GROUP_WEIGHTS[None, None, :] * mpc[None, None, :]
            * sigmoid(18.0 * (group_expectations - 0.08)), axis=2
        )
        wage_pressure = 0.08 * GROUP_SKILL[1] * np.maximum(group_expectations[:, :, 1] - 0.05, 0.0)
        markup_pressure = 0.10 * np.maximum(group_expectations[:, :, 2] - 0.04, 0.0)
        capital_flight = 0.07 * sigmoid(16.0 * (group_expectations[:, :, 3] - 0.07)) * (0.35 + sanction_state)
        demand_gap = np.maximum(war_spending_share - tax_effort - 0.35 * civilian_diversion + hoarding, 0.0)
        trade_dislocation = np.maximum(trade_access - target_trade, 0.0)
        fx_depreciation = 0.75 * monetization + 0.055 * import_gap + 0.035 * fiscal_stress
        fx_depreciation += 0.10 * np.maximum(sovereign_rate - policy_rate, 0.0)
        fx_depreciation -= 0.025 * external_input
        hedgeable_fx = 0.20 * trade_dislocation + 0.75 * monetization + 0.003 * svar_shocks[:, :, 3]
        fx_depreciation = fx_depreciation - 0.75 * monetization
        fx_depreciation += hedgeable_fx * (1.0 - derivative_hedge_ratio) + capital_flight
        supply_cost = 0.025 * import_gap + 0.020 * shortage_index + 0.025 * sanction_state
        fiscal_price_level = (
            0.040 * fiscal_stress_effective + 0.32 * monetization
            + 0.065 * money_growth + 0.14 * seigniorage_gap
            + 0.025 * qe_ycc_intensity + 0.004 * svar_shocks[:, :, 2]
        )
        free_inflation = (
            0.40 * effective_inflation + 0.17 * expected_inflation + 0.08 * mean_field_expectation
            + 0.10 * demand_gap + 0.18 * fx_depreciation
            + supply_cost + fiscal_price_level + wage_pressure + markup_pressure
        )
        free_inflation += 0.004 * svar_shocks[:, :, 4] + 0.003 * svar_shocks[:, :, 3] * (1.0 - derivative_hedge_ratio)
        free_inflation += rng.normal(0.0, 0.0035, free_inflation.shape)
        free_inflation = np.clip(free_inflation, -0.01, 0.90)

        control_capacity = np.array([0.58, 0.32, 0.42, 0.45])[None, :]
        control_decay = np.exp(-0.065 * conflict_years)
        control_strength = control_capacity * control_decay * np.exp(-1.8 * shortage_index)
        suppressed = control_strength * np.maximum(free_inflation - inflation_target, 0.0)
        inflation_overhang += suppressed * DT
        release = np.minimum(inflation_overhang / max(DT, 1e-6), 0.12 * shortage_index + 0.035 * conflict_years)
        inflation_overhang = np.maximum(inflation_overhang - release * DT, 0.0)
        inflation = np.clip(free_inflation - suppressed + release, -0.01, 0.70)
        shortage_index += DT * (0.30 * suppressed + 0.08 * import_gap - 0.25 * shortage_index)
        shortage_index = np.clip(shortage_index, 0.0, 1.0)
        effective_inflation = np.clip(inflation + 0.12 * shortage_index + 0.15 * inflation_overhang, -0.01, 0.90)
        deanchor_hazard = sigmoid(-5.6 + 8.0 * expected_inflation + 1.2 * fiscal_stress_effective + 0.9 * sanction_state + 0.8 * shortage_index)
        deanchored |= rng.random((n, 4)) < deanchor_hazard
        stable = (effective_inflation < 0.055) & (fiscal_stress_effective < 0.10)
        deanchored &= ~((rng.random((n, 4)) < 0.015) & stable)

        # DEDS regime transition with hysteresis: 0 anchored, 1 controlled
        # inflation, 2 shortage/overhang, 3 expectations deanchored, 4 fiscal
        # inflation crisis. The state affects next-quarter real procurement.
        next_regime = inflation_regime.copy()
        next_regime = np.where(effective_inflation > 0.055, np.maximum(next_regime, 1), next_regime)
        next_regime = np.where((shortage_index > 0.12) | (inflation_overhang > 0.035), np.maximum(next_regime, 2), next_regime)
        next_regime = np.where(deanchored & (mean_field_expectation > 0.085), np.maximum(next_regime, 3), next_regime)
        next_regime = np.where((effective_inflation > 0.35) | (fiscal_stress_effective > 0.75), 4, next_regime)
        recovery = (effective_inflation < 0.045) & (shortage_index < 0.06) & (rng.random((n, 4)) < 0.04)
        inflation_regime = np.where(recovery, np.maximum(next_regime - 1, 0), next_regime).astype(np.int8)

        group_inflation = inflation[:, :, None]
        group_inflation = group_inflation + 0.060 * import_gap[:, :, None] * (1.0 - GROUP_LIQUIDITY)[None, None, :]
        group_inflation = group_inflation + 0.080 * shortage_index[:, :, None] * (1.0 - GROUP_LIQUIDITY)[None, None, :]
        group_inflation = group_inflation + 0.070 * sanction_state[:, :, None] * (1.0 - GROUP_ASSETS)[None, None, :]
        group_inflation[:, :, 2] += 0.04 * import_gap
        group_inflation[:, :, 4] += 0.12 * (1.0 - input_availability)
        group_inflation = np.clip(group_inflation, -0.01, 1.20)
        income_growth = inflation[:, :, None] * (0.25 + 0.55 * GROUP_SKILL)[None, None, :]
        income_growth -= civilian_diversion[:, :, None] * (1.0 - 0.45 * GROUP_SKILL)[None, None, :]
        # Tax incidence differs by household/firm balance sheet instead of
        # subtracting one representative-agent tax rate.
        tax_incidence = np.stack((
            0.70 * tax_rates[:, :, 0] + 0.35 * tax_rates[:, :, 1],
            0.28 * tax_rates[:, :, 0] + 0.72 * tax_rates[:, :, 1],
            0.20 * tax_rates[:, :, 0] + 0.78 * tax_rates[:, :, 2],
            0.12 * tax_rates[:, :, 0] + 0.62 * tax_rates[:, :, 2],
            0.10 * tax_rates[:, :, 0] + 0.10 * tax_rates[:, :, 1],
        ), axis=2)
        baseline_incidence = np.stack((
            0.70 * tax_base_rates[None, :, 0] + 0.35 * tax_base_rates[None, :, 1],
            0.28 * tax_base_rates[None, :, 0] + 0.72 * tax_base_rates[None, :, 1],
            0.20 * tax_base_rates[None, :, 0] + 0.78 * tax_base_rates[None, :, 2],
            0.12 * tax_base_rates[None, :, 0] + 0.62 * tax_base_rates[None, :, 2],
            0.10 * tax_base_rates[None, :, 0] + 0.10 * tax_base_rates[None, :, 1],
        ), axis=2)
        income_growth -= 0.22 * np.maximum(tax_incidence - baseline_incidence, 0.0)
        income_growth[:, :, 2] += markup_pressure - 0.06 * sanction_state
        income_growth[:, :, 3] += 0.035 + 0.35 * capital_flight
        income_growth[:, :, 4] += 0.10 * war_spending_share
        group_price_index *= 1.0 + group_inflation * DT
        group_real_position *= (1.0 + income_growth * DT) / np.maximum(1.0 + group_inflation * DT, 0.2)

        damage_drag = np.mean(destroyed_capacity, axis=2) * 0.035
        trade_drag = np.array([0.035, 0.022, 0.045, 0.075])[None, :] * import_gap
        inflation_drag = 0.12 * np.square(np.maximum(effective_inflation - 0.06, 0.0))
        learning_spillover = 0.005 * np.log1p(np.mean(cumulative_output, axis=2))
        annual_growth = np.array([0.025, 0.018, 0.010, 0.018])[None, :]
        annual_growth = annual_growth - civilian_diversion - damage_drag - trade_drag - inflation_drag
        annual_growth = annual_growth + learning_spillover - 0.14 * tax_distortion
        gdp *= np.maximum(0.85, 1.0 + annual_growth * DT)
        growth_denominator = np.maximum(1.0 + annual_growth * DT, 0.82)
        debt_ratio = (debt_ratio * (1.0 + coupon_rate * DT) + primary_deficit * DT) / growth_denominator
        retained_local_gap = np.maximum(local_gap - central_bailout, 0.0)
        local_debt_ratio = (
            local_debt_ratio * (1.0 + local_coupon_rate * DT) + retained_local_gap * DT
        ) / growth_denominator

        # Competing central/local restructuring events provide a finite fiscal
        # endpoint.  Haircuts preserve path dependence through lost market
        # access and an inflation overhang rather than resetting the economy.
        restructure_hazard = sigmoid(-12.0 + 5.0 * debt_pressure + 2.2 * effective_inflation)
        new_restructure = (~restructured) & (rng.random((n, 4)) < restructure_hazard * DT)
        local_hazard = sigmoid(-11.0 + 4.5 * local_debt_ratio + 2.0 * (1.0 - land_revenue_index))
        new_local_restructure = (~local_restructured) & (rng.random((n, 4)) < local_hazard * DT)
        debt_ratio = np.where(new_restructure, debt_ratio * np.array([0.82, 0.88, 0.90, 0.84])[None, :], debt_ratio)
        local_debt_ratio = np.where(new_local_restructure, local_debt_ratio * 0.76, local_debt_ratio)
        capital_market_access *= np.where(new_restructure | new_local_restructure, 0.82, 1.0)
        inflation_overhang += 0.025 * new_restructure + 0.012 * new_local_restructure
        restructured |= new_restructure
        local_restructured |= new_local_restructure
        consolidated_debt = debt_ratio + guarantee_weight * local_debt_ratio
        debt_pressure = consolidated_debt / np.maximum(fiscal_limit, 0.2)
        cumulative_deficit += primary_deficit * DT
        real_war_spending += war_spending_share * gdp * DT / np.maximum(1.0 + effective_inflation, 0.5)

        readiness = np.exp(np.mean(np.log(np.clip(inventory / initial_inventory[None, :, :], 0.02, 3.0)), axis=2))
        baseline_inventory = np.maximum(initial_inventory[None, :, :] + (quarter + 1) * base_output[None, :, :] - cumulative_consumption, 0.01)
        baseline_readiness = np.exp(np.mean(np.log(np.clip(baseline_inventory / initial_inventory[None, :, :], 0.02, 3.0)), axis=2))
        readiness_history.append(readiness)
        baseline_readiness_history.append(baseline_readiness)

        if quarter + 1 in (4, 20, 40):
            snapshots[(quarter + 1) // 4] = {
                "gdp": gdp.copy(), "inflation": inflation.copy(), "effective_inflation": effective_inflation.copy(),
                "inflation_overhang": inflation_overhang.copy(), "shortage": shortage_index.copy(), "debt": debt_ratio.copy(),
                "local_debt": local_debt_ratio.copy(), "consolidated_debt": consolidated_debt.copy(),
                "fiscal_limit": fiscal_limit.copy(), "debt_pressure": debt_pressure.copy(),
                "trade": trade_access.copy(), "sanctions": sanction_state.copy(), "deanchored": deanchored.copy(),
                "capital_access": capital_market_access.copy(), "sovereign_rate": sovereign_rate.copy(),
                "policy_rate": policy_rate.copy(), "coupon_rate": coupon_rate.copy(),
                "local_coupon_rate": local_coupon_rate.copy(), "fiscal_dominance": fiscal_dominance.copy(),
                "qe_ycc": qe_ycc_intensity.copy(), "money_base": money_base_ratio.copy(),
                "money_growth": money_growth.copy(), "seigniorage_gap": seigniorage_gap.copy(),
                "tax_rates": tax_rates.copy(), "tax_revenue": tax_revenue.copy(),
                "tax_distortion": tax_distortion.copy(), "central_bailout": central_bailout.copy(),
                "restructured": restructured.copy(), "local_restructured": local_restructured.copy(),
                "defense_wacc": defense_wacc.copy(), "mm_wedge": mm_friction_wedge.copy(),
                "credit_hazard": credit_default_hazard.copy(), "financial_variance": financial_variance.copy(),
                "capm_equity": capm_equity_cost.copy(), "apt_equity": apt_equity_cost.copy(),
                "hedge_ratio": derivative_hedge_ratio.copy(), "option_exercise": real_option_exercise.copy(),
                "inflation_regime": inflation_regime.copy(),
                "group_price": group_price_index.copy(), "group_real_position": group_real_position.copy(),
                "readiness": readiness.copy(),
                "capacity": capacity.copy(), "inventory": inventory.copy(),
            }

    readiness_history = np.stack(readiness_history, axis=0)
    baseline_readiness_history = np.stack(baseline_readiness_history, axis=0)
    first_five = slice(0, 20)
    attacker_sustain = np.mean(readiness_history[first_five, :, 0], axis=0)
    attacker_baseline = np.mean(baseline_readiness_history[first_five, :, 0], axis=0)
    coalition_weights = np.array([case["us"], case["japan"], 1.0])
    coalition_weights /= coalition_weights.sum()
    defender_sustain = np.sum(np.mean(readiness_history[first_five, :, 1:4], axis=0) * coalition_weights[None, :], axis=1)
    defender_baseline = np.sum(np.mean(baseline_readiness_history[first_five, :, 1:4], axis=0) * coalition_weights[None, :], axis=1)
    relative_sustainment_gain = np.log(np.maximum(attacker_sustain / attacker_baseline, 0.05))
    relative_sustainment_gain -= np.log(np.maximum(defender_sustain / defender_baseline, 0.05))

    base_prob = campaign_row["broad_control_central"]
    adjusted_prob_paths = sigmoid(logit(base_prob) + 0.88 * relative_sustainment_gain)
    adjusted_probability = float(adjusted_prob_paths.mean())

    actor_metrics = {}
    for actor_index, actor in enumerate(ACTORS):
        actor_metrics[actor] = {}
        for horizon, snap in snapshots.items():
            production_multiplier = np.mean(snap["capacity"][:, actor_index, :], axis=1)
            inventory_ratio = np.exp(np.mean(np.log(np.clip(snap["inventory"][:, actor_index, :] / initial_inventory[None, actor_index, :], 0.01, 5.0)), axis=1))
            actor_metrics[actor][str(horizon)] = {
                "production_capacity_multiplier_p10_p50_p90": q(production_multiplier),
                "inventory_ratio_p10_p50_p90": q(inventory_ratio),
                "gdp_index_p10_p50_p90": q(snap["gdp"][:, actor_index]),
                "annual_inflation_p10_p50_p90": q(snap["inflation"][:, actor_index]),
                "effective_inflation_including_shortages_p10_p50_p90": q(snap["effective_inflation"][:, actor_index]),
                "inflation_overhang_p10_p50_p90": q(snap["inflation_overhang"][:, actor_index]),
                "shortage_index_p10_p50_p90": q(snap["shortage"][:, actor_index]),
                "debt_to_gdp_p10_p50_p90": q(snap["debt"][:, actor_index]),
                "subnational_debt_to_gdp_p10_p50_p90": q(snap["local_debt"][:, actor_index]),
                "consolidated_debt_to_gdp_p10_p50_p90": q(snap["consolidated_debt"][:, actor_index]),
                "fiscal_limit_to_gdp_p10_p50_p90": q(snap["fiscal_limit"][:, actor_index]),
                "debt_pressure_ratio_p10_p50_p90": q(snap["debt_pressure"][:, actor_index]),
                "trade_access_p10_p50_p90": q(snap["trade"][:, actor_index]),
                "sanction_intensity_p10_p50_p90": q(snap["sanctions"][:, actor_index]),
                "global_capital_market_access_p10_p50_p90": q(snap["capital_access"][:, actor_index]),
                "sovereign_bond_rate_p10_p50_p90": q(snap["sovereign_rate"][:, actor_index]),
                "policy_rate_p10_p50_p90": q(snap["policy_rate"][:, actor_index]),
                "average_coupon_rate_p10_p50_p90": q(snap["coupon_rate"][:, actor_index]),
                "subnational_coupon_rate_p10_p50_p90": q(snap["local_coupon_rate"][:, actor_index]),
                "fiscal_dominance_share_p10_p50_p90": q(snap["fiscal_dominance"][:, actor_index]),
                "qe_ycc_intensity_p10_p50_p90": q(snap["qe_ycc"][:, actor_index]),
                "monetary_base_to_gdp_p10_p50_p90": q(snap["money_base"][:, actor_index]),
                "money_growth_p10_p50_p90": q(snap["money_growth"][:, actor_index]),
                "seigniorage_gap_p10_p50_p90": q(snap["seigniorage_gap"][:, actor_index]),
                "tax_revenue_to_gdp_p10_p50_p90": q(snap["tax_revenue"][:, actor_index]),
                "tax_distortion_p10_p50_p90": q(snap["tax_distortion"][:, actor_index]),
                "central_bailout_flow_p10_p50_p90": q(snap["central_bailout"][:, actor_index]),
                "central_restructuring_share": float(snap["restructured"][:, actor_index].mean()),
                "subnational_restructuring_share": float(snap["local_restructured"][:, actor_index].mean()),
                "tax_rates_median": {
                    tax: float(np.median(snap["tax_rates"][:, actor_index, tax_index]))
                    for tax_index, tax in enumerate(("consumption", "labour", "capital"))
                },
                "defense_industry_wacc_p10_p50_p90": q(snap["defense_wacc"][:, actor_index]),
                "mm_friction_wedge_p10_p50_p90": q(snap["mm_wedge"][:, actor_index]),
                "credit_default_hazard_p10_p50_p90": q(snap["credit_hazard"][:, actor_index]),
                "financial_variance_p10_p50_p90": q(snap["financial_variance"][:, actor_index]),
                "capm_equity_cost_p10_p50_p90": q(snap["capm_equity"][:, actor_index]),
                "apt_equity_cost_p10_p50_p90": q(snap["apt_equity"][:, actor_index]),
                "derivative_hedge_ratio_p10_p50_p90": q(snap["hedge_ratio"][:, actor_index]),
                "real_option_exercise_share_p10_p50_p90": q(np.mean(snap["option_exercise"][:, actor_index, :], axis=1)),
                "expectations_deanchored_share": float(snap["deanchored"][:, actor_index].mean()),
                "inflation_regime_shares": {
                    str(state): float(np.mean(snap["inflation_regime"][:, actor_index] == state))
                    for state in range(5)
                },
                "group_cost_index_median": {
                    group: float(np.median(snap["group_price"][:, actor_index, group_index]))
                    for group_index, group in enumerate(GROUPS)
                },
                "group_real_position_median": {
                    group: float(np.median(snap["group_real_position"][:, actor_index, group_index]))
                    for group_index, group in enumerate(GROUPS)
                },
            }

    return {
        "conflict_year": year,
        "case": case_key,
        "scenario": scenario_name,
        "v36_broad_control_central": float(base_prob),
        "v38_broad_control_with_wartime_economy": adjusted_probability,
        "change_percentage_points": 100.0 * (adjusted_probability - base_prob),
        "relative_sustainment_log_gain_p10_p50_p90": q(relative_sustainment_gain),
        "actor_metrics": actor_metrics,
        "resource_accounting": {
            "china_production_to_consumption_ratio_p10_p50_p90": q(np.sum(cumulative_production[:, 0, :], axis=1) / np.maximum(np.sum(cumulative_consumption[:, 0, :], axis=1), 1e-8)),
            "taiwan_production_and_aid_to_consumption_ratio_p10_p50_p90": q(np.sum(cumulative_production[:, 3, :], axis=1) / np.maximum(np.sum(cumulative_consumption[:, 3, :], axis=1), 1e-8)),
            "china_cumulative_real_war_spending_index_p10_p50_p90": q(real_war_spending[:, 0]),
            "taiwan_cumulative_real_war_spending_index_p10_p50_p90": q(real_war_spending[:, 3]),
            "cumulative_war_financing_median_by_actor": {
                actor: {
                    source: float(np.median(cumulative_financing[:, actor_index, source_index]))
                    for source_index, source in enumerate(("domestic_bonds", "financial_repression", "monetization", "external_relief"))
                }
                for actor_index, actor in enumerate(ACTORS)
            },
            "svar_reduced_form_variance_share": {
                actor: {
                    shock: float(value)
                    for shock, value in zip(
                        ("sanction", "supply", "fiscal", "fx", "demand"),
                        (shock_variance[:, actor_index, :].sum(axis=0) / np.maximum(shock_variance[:, actor_index, :].sum(), 1e-9)),
                    )
                }
                for actor_index, actor in enumerate(ACTORS)
            },
        },
    }


def validate(results):
    errors = []
    for row in results:
        if not 0.0 <= row["v38_broad_control_with_wartime_economy"] <= 1.0:
            errors.append("probability outside [0,1]")
        for actor in ACTORS:
            for horizon in ("1", "5", "10"):
                metrics = row["actor_metrics"][actor][horizon]
                for key, values in metrics.items():
                    if not isinstance(values, list) or len(values) != 3:
                        continue
                    if not values[0] <= values[1] <= values[2]:
                        errors.append(f"unordered quantiles {actor} {horizon} {key}")
                    if key != "annual_inflation_p10_p50_p90" and values[0] < 0:
                        errors.append(f"negative stock metric {actor} {horizon} {key}")
    central = {(r["conflict_year"], r["case"]): r for r in results if r["scenario"] == "central"}
    surge = {(r["conflict_year"], r["case"]): r for r in results if r["scenario"] == "coalition_industrial_surge"}
    for key in set(central) & set(surge):
        if surge[key]["v38_broad_control_with_wartime_economy"] > central[key]["v38_broad_control_with_wartime_economy"] + 0.015:
            errors.append(f"coalition surge direction failed {key}")
    if errors:
        raise AssertionError("; ".join(errors[:12]))
    return "TAIWAN_WARTIME_ECONOMY_V38_VERIFICATION: PASS"


def write_csv(results):
    path = OUT / "台海V3.8战时经济与补产仿真.csv"
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["scenario", "conflict_year", "case", "v36_broad_control", "v38_broad_control", "change_pp", "china_capacity_5y", "china_inflation_5y", "china_gdp_5y", "taiwan_capacity_5y", "taiwan_inflation_5y", "taiwan_gdp_5y"])
        for row in results:
            chn = row["actor_metrics"]["china"]["5"]
            twn = row["actor_metrics"]["taiwan"]["5"]
            writer.writerow([
                row["scenario"], row["conflict_year"], row["case"], row["v36_broad_control_central"],
                row["v38_broad_control_with_wartime_economy"], row["change_percentage_points"],
                chn["production_capacity_multiplier_p10_p50_p90"][1], chn["annual_inflation_p10_p50_p90"][1],
                chn["gdp_index_p10_p50_p90"][1], twn["production_capacity_multiplier_p10_p50_p90"][1],
                twn["annual_inflation_p10_p50_p90"][1], twn["gdp_index_p10_p50_p90"][1],
            ])
    return path


def alliance_weighted_results(results):
    v37 = json.loads((OUT / "台海V3.7因果_演化博弈_竞争风险_联合仿真摘要.json").read_text(encoding="utf-8"))
    central = next(row for row in v37["results"] if row["scenario"] == "central")
    strategy_by_year = {row["year"]: row["alliance_strategy_mean"] for row in central["years"]}
    output = []
    for scenario in sorted({row["scenario"] for row in results}):
        for year in YEARS:
            shares = strategy_by_year[year]
            weights = {
                "taiwan_alone": shares["noncombat"],
                "limited": shares["limited"],
                "delayed": shares["delayed_full"],
                "timely_full": shares["timely_full"],
            }
            rows = {r["case"]: r for r in results if r["scenario"] == scenario and r["conflict_year"] == year}
            old = sum(weights[case] * rows[case]["v36_broad_control_central"] for case in CASES)
            new = sum(weights[case] * rows[case]["v38_broad_control_with_wartime_economy"] for case in CASES)
            output.append({
                "scenario": scenario,
                "conflict_year": year,
                "alliance_strategy_weights_from_v37": weights,
                "v36_weighted_conditional_broad_control": float(old),
                "v38_weighted_conditional_broad_control": float(new),
                "change_percentage_points": float(100.0 * (new - old)),
            })
    return output


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    aggregate_only = os.environ.get("TAIWAN_V38_AGGREGATE_ONLY") == "1"
    json_path = OUT / "台海V3.8战时经济生产财政通胀_仿真摘要.json"
    if aggregate_only:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        payload["alliance_weighted_results"] = alliance_weighted_results(payload["results"])
        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(payload["alliance_weighted_results"], ensure_ascii=False, indent=2))
        return
    campaign = load_campaign()
    results = []
    run = 0
    selected = os.environ.get("TAIWAN_V38_SCENARIOS")
    scenario_items = [(name, cfg) for name, cfg in SCENARIOS.items() if not selected or name in selected.split(",")]
    for scenario_name, scenario in scenario_items:
        for year in YEARS:
            for case in CASES:
                run += 1
                results.append(simulate_one(year, case, scenario_name, scenario, campaign[(year, case)], SEED + 97 * run))
    verification = validate(results)
    payload = {
        "model": "V3.8 endogenous wartime production, depletion, resource bottlenecks, trade, external support, fiscal dominance and inflation overlay",
        "paths_per_cell": PATHS,
        "quarters": QUARTERS,
        "actors": ACTORS,
        "sectors": SECTORS,
        "scenarios": SCENARIOS,
        "calibration_registry": CALIBRATION_REGISTRY,
        "identification_warning": "Mobilization ceilings, fiscal responses and wartime trade losses are broad structural priors. Historical cases calibrate mechanisms and ranges, not a Taiwan-war point forecast.",
        "results": results,
        "alliance_weighted_results": alliance_weighted_results(results),
        "verification": verification,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    csv_path = write_csv(results)
    central = [r for r in results if r["scenario"] == "central" and r["conflict_year"] == 2030]
    print(json.dumps({"verification": verification, "central_2030": central, "json": str(json_path), "csv": str(csv_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
