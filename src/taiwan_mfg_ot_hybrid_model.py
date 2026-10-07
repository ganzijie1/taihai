import csv
import json
import math
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

SEED = 20261001
PATHS = 12000
START_YEAR = 2026
END_YEAR = 2100
MONTHS = (END_YEAR - START_YEAR) * 12
SNAPSHOT_YEARS = {2030, 2035, 2040, 2050, 2075, 2100}

PRESSURE_STATES = ["常态威慑", "持续施压", "急性危机"]
MFG_ACTIONS = ["缓和交流", "政治疏离", "韧性维持现状", "防御动员", "升级民族主义"]
OT_TARGETS = ["监测预警", "基础设施韧性", "信息核验", "经济缓冲", "危机外交", "社会应急"]
TERMINALS = ["force", "peace", "separation", "status_quo"]
POST_FORCE_OUTCOMES = [
    "halted_or_withdrawn", "negotiated_ceasefire", "limited_contested_control",
    "broad_control", "regional_war_expansion", "protracted_stalemate_60m",
]
ALLIANCE_LEVELS = [
    "no_major_external_intervention", "indirect_or_limited_support",
    "us_direct_japan_limited", "us_japan_direct",
]

GOVERNANCE_SCENARIOS = {
    "突袭进入与信息分层": {
        "control": 0.72, "resistance": 0.32, "governance": 0.24,
        "service": 0.34, "legitimacy": 0.22, "information": 0.28,
        "network_adapt": 0.76, "coercion": 0.62, "governance_invest": 0.40,
        "service_invest": 0.45, "damage": 0.58, "friction": 0.62,
        "arrival_days": 42, "arrival_uncertainty": 0.58, "location_entropy": 0.82,
        "elite_awareness": 0.34, "local_diffusion": 0.28, "public_diffusion": 0.16,
        "leader_flight_hazard": 0.030, "leader_death_hazard": 0.010,
        "leader_disconnect_hazard": 0.025,
    },
    "多层网络分散抵抗": {
        "control": 0.58, "resistance": 0.58, "governance": 0.20,
        "service": 0.38, "legitimacy": 0.20, "information": 0.54,
        "network_adapt": 0.96, "coercion": 0.56, "governance_invest": 0.36,
        "service_invest": 0.42, "damage": 0.52, "friction": 0.82,
        "arrival_days": 35, "arrival_uncertainty": 0.44, "location_entropy": 0.64,
        "elite_awareness": 0.56, "local_diffusion": 0.58, "public_diffusion": 0.42,
        "leader_flight_hazard": 0.014, "leader_death_hazard": 0.022,
        "leader_disconnect_hazard": 0.020,
    },
    "城市基础设施冲击": {
        "control": 0.64, "resistance": 0.42, "governance": 0.22,
        "service": 0.20, "legitimacy": 0.18, "information": 0.42,
        "network_adapt": 0.82, "coercion": 0.60, "governance_invest": 0.42,
        "service_invest": 0.58, "damage": 0.86, "friction": 0.88,
        "arrival_days": 38, "arrival_uncertainty": 0.48, "location_entropy": 0.70,
        "elite_awareness": 0.50, "local_diffusion": 0.44, "public_diffusion": 0.34,
        "leader_flight_hazard": 0.018, "leader_death_hazard": 0.014,
        "leader_disconnect_hazard": 0.045,
    },
    "治理与公共服务优先": {
        "control": 0.56, "resistance": 0.44, "governance": 0.36,
        "service": 0.52, "legitimacy": 0.32, "information": 0.58,
        "network_adapt": 0.62, "coercion": 0.24, "governance_invest": 0.78,
        "service_invest": 0.84, "damage": 0.34, "friction": 0.48,
        "arrival_days": 48, "arrival_uncertainty": 0.34, "location_entropy": 0.48,
        "elite_awareness": 0.72, "local_diffusion": 0.76, "public_diffusion": 0.66,
        "leader_flight_hazard": 0.008, "leader_death_hazard": 0.006,
        "leader_disconnect_hazard": 0.012,
    },
}

# Estimated from the existing 2020-09 to 2026-09 monthly pressure series.
BASE_TRANSITION = np.array([
    [0.8475992633, 0.1152668430, 0.0371338937],
    [0.0142083939, 0.9443913347, 0.0414002714],
    [0.0233899982, 0.0803093399, 0.8963006619],
])

SCENARIOS = {
    "stabilization": {
        "pressure_drift": -0.28,
        "trade_2100": 0.20,
        "identity_drift": 0.00005,
        "dialogue": 0.32,
        "external_friction": -0.18,
        "shock_rate": 0.004,
        "leader_jump": 0.007,
        "ot_capacity": 1.10,
    },
    "baseline": {
        "pressure_drift": 0.20,
        "trade_2100": 0.12,
        "identity_drift": 0.00014,
        "dialogue": 0.10,
        "external_friction": 0.08,
        "shock_rate": 0.007,
        "leader_jump": 0.012,
        "ot_capacity": 1.00,
    },
    "adverse": {
        "pressure_drift": 0.65,
        "trade_2100": 0.06,
        "identity_drift": 0.00024,
        "dialogue": -0.08,
        "external_friction": 0.35,
        "shock_rate": 0.012,
        "leader_jump": 0.022,
        "ot_capacity": 0.88,
    },
}


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def softmax(values, temperature=1.0):
    scaled = values / max(temperature, 1e-6)
    scaled = scaled - np.max(scaled, axis=-1, keepdims=True)
    exp_values = np.exp(np.clip(scaled, -40.0, 40.0))
    return exp_values / exp_values.sum(axis=-1, keepdims=True)


def sample_rows(rng, probabilities):
    draws = rng.random(probabilities.shape[0])
    return (draws[:, None] > np.cumsum(probabilities, axis=1)).sum(axis=1)


def percentile(values, q):
    return float(np.quantile(values, q))


def solve_mfg(threat, trust, polarization, economic_cost, identity_distance,
              success_belief, death_salience, kinship):
    """Aggregate Logit MFG fixed point for five public response strategies."""
    m = np.array([0.08, 0.10, 0.57, 0.21, 0.04], dtype=float)
    for iteration in range(200):
        utilities = np.array([
            0.90 * kinship + 0.65 * (1.0 - threat) - 0.55 * identity_distance,
            0.55 * (1.0 - trust) + 0.30 * economic_cost - 0.25 * threat,
            1.25 * trust + 0.55 * (1.0 - economic_cost) - 0.30 * polarization,
            1.15 * threat + 0.55 * success_belief + 0.30 * trust - 0.70 * death_salience,
            1.05 * polarization + 0.75 * threat + 0.35 * identity_distance
            - 0.85 * death_salience * (1.0 - success_belief),
        ])
        response = softmax((utilities - 0.78 * m)[None, :], temperature=0.46)[0]
        new_m = 0.62 * m + 0.38 * response
        residual = float(np.abs(new_m - m).sum())
        m = new_m
        if residual < 1e-10:
            break
    return m, residual, iteration + 1


def solve_unbalanced_ot(pressure, economic_cost, misinformation, infrastructure_damage,
                        diplomacy_gap, scenario_capacity):
    """Entropic unbalanced OT from four abstract resource pools to six resilience uses."""
    supply = np.array([0.36, 0.20, 0.19, 0.25])
    supply *= scenario_capacity
    desired = np.array([
        0.14 + 0.20 * pressure,
        0.18 + 0.22 * infrastructure_damage,
        0.14 + 0.24 * misinformation,
        0.18 + 0.24 * economic_cost,
        0.16 + 0.20 * diplomacy_gap,
        0.14 + 0.18 * pressure,
    ])
    desired = desired / desired.sum()
    cost = np.array([
        [0.18, 0.26, 0.30, 0.55, 0.44, 0.36],
        [0.40, 0.34, 0.18, 0.42, 0.28, 0.20],
        [0.24, 0.48, 0.36, 0.38, 0.16, 0.44],
        [0.46, 0.25, 0.31, 0.16, 0.40, 0.28],
    ])
    epsilon = 0.22
    relaxation = 0.78
    kernel = np.exp(-cost / epsilon)
    u = np.ones(4)
    v = np.ones(6)
    for iteration in range(300):
        old_v = v.copy()
        u = (supply / np.maximum(kernel @ v, 1e-12)) ** relaxation
        v = (desired / np.maximum(kernel.T @ u, 1e-12)) ** relaxation
        if np.max(np.abs(v - old_v)) < 1e-11:
            break
    plan = (u[:, None] * kernel) * v[None, :]
    total = plan.sum()
    target_share = plan.sum(axis=0) / total
    mean_cost = float((plan * cost).sum() / total)
    coverage = float(np.clip(total / desired.sum(), 0.0, 1.25))
    return target_share, mean_cost, coverage, iteration + 1


def adjusted_transition(base, escalation_delta):
    matrices = np.empty((len(escalation_delta), 3, 3))
    for origin in range(3):
        direction = np.sign(np.arange(3) - origin)
        values = base[origin][None, :] * np.exp(escalation_delta[:, None] * direction[None, :])
        matrices[:, origin, :] = values / values.sum(axis=1, keepdims=True)
    return matrices


def simulate_post_force_campaign(features, seed):
    """High-level 60-month campaign/alliance model without operational targeting."""
    rng = np.random.default_rng(seed)
    n = len(features["resilience"])
    if n == 0:
        return {
            "force_paths": 0,
            "alliance_intervention_conditional": dict.fromkeys(ALLIANCE_LEVELS, 0.0),
            "outcome_conditional_on_force": dict.fromkeys(POST_FORCE_OUTCOMES, 0.0),
            "joint_defense_halted_or_withdrawn": 0.0,
        }

    support = features["external_support"]
    resilience = features["resilience"]
    identity = features["identity"]
    economic_link = features["economic_link"]
    death_salience = features["death_salience"]
    pressure = features["pressure_state"] / 2.0
    crisis_memory = features["crisis_memory"]

    alliance_scores = np.column_stack([
        0.05 - 0.75 * support + 0.35 * economic_link,
        0.48 + 0.25 * support + 0.20 * death_salience,
        -0.02 + 0.92 * support + 0.28 * pressure,
        -0.42 + 1.28 * support + 0.42 * pressure - 0.24 * death_salience,
    ])
    alliance_scores += rng.normal(0.0, 0.24, alliance_scores.shape)
    alliance = sample_rows(rng, softmax(alliance_scores, temperature=0.72)).astype(np.int8)
    alliance_strength = np.choose(alliance, [0.05, 0.34, 0.67, 0.92])

    defender_capacity = np.clip(
        0.48 * resilience + 0.38 * alliance_strength + 0.16 * support
        + rng.normal(0.0, 0.06, n), 0.05, 1.25,
    )
    attacker_momentum = np.clip(
        0.50 + 0.18 * pressure + 0.12 * crisis_memory
        - 0.12 * economic_link + rng.normal(0.0, 0.08, n), 0.08, 1.30,
    )
    political_tolerance = np.clip(
        0.62 - 0.42 * death_salience + 0.15 * crisis_memory
        + rng.normal(0.0, 0.06, n), 0.08, 0.95,
    )

    outcome = np.full(n, -1, dtype=np.int8)
    event_month = np.full(n, -1, dtype=np.int16)
    for month in range(1, 61):
        active = outcome < 0
        if not active.any():
            break
        idx = np.where(active)[0]
        duration = month / 60.0
        defense = defender_capacity[idx]
        momentum = attacker_momentum[idx]
        alliance_now = alliance_strength[idx]
        tolerance = political_tolerance[idx]

        attacker_momentum[idx] = np.clip(
            momentum + 0.010 * (momentum - defense) - 0.008 * alliance_now
            - 0.006 * duration + rng.normal(0.0, 0.012, len(idx)), 0.02, 1.35,
        )
        defender_capacity[idx] = np.clip(
            defense + 0.008 * alliance_now - 0.008 * momentum
            + 0.004 * resilience[idx] + rng.normal(0.0, 0.010, len(idx)), 0.02, 1.35,
        )
        political_tolerance[idx] = np.clip(
            tolerance - 0.008 * death_salience[idx] - 0.005 * duration
            + 0.003 * (momentum > defense), 0.02, 0.98,
        )

        defense = defender_capacity[idx]
        momentum = attacker_momentum[idx]
        tolerance = political_tolerance[idx]
        log_hazards = np.column_stack([
            -4.25 + 1.70 * defense + 0.75 * alliance_now - 1.25 * momentum + 0.35 * duration,
            -4.20 + 0.70 * (1.0 - tolerance) + 0.62 * death_salience[idx]
            + 0.48 * economic_link[idx] + 0.42 * duration,
            -4.85 + 1.08 * momentum - 0.72 * defense + 0.38 * identity[idx] + 0.30 * duration,
            -5.75 + 1.55 * momentum - 1.35 * defense - 0.62 * alliance_now - 0.42 * identity[idx],
            -5.25 + 0.92 * alliance_now + 0.72 * pressure[idx]
            + 0.46 * crisis_memory[idx] + 0.30 * duration,
        ])
        hazards = np.exp(np.clip(log_hazards, -12.0, -1.0))
        total = hazards.sum(axis=1)
        event = rng.random(len(idx)) < (1.0 - np.exp(-total))
        if event.any():
            chosen = sample_rows(rng, hazards[event] / total[event, None])
            event_idx = idx[event]
            outcome[event_idx] = chosen.astype(np.int8)
            event_month[event_idx] = month

    outcome[outcome < 0] = 5
    return {
        "force_paths": int(n),
        "horizon_months": 60,
        "alliance_intervention_conditional": {
            key: float(np.mean(alliance == i)) for i, key in enumerate(ALLIANCE_LEVELS)
        },
        "outcome_conditional_on_force": {
            key: float(np.mean(outcome == i)) for i, key in enumerate(POST_FORCE_OUTCOMES)
        },
        "joint_defense_halted_or_withdrawn": float(np.mean((outcome == 0) & (alliance >= 2))),
        "us_japan_direct_and_halted_or_withdrawn": float(np.mean((outcome == 0) & (alliance == 3))),
        "event_month_conditional": {
            key: None if i == 5 or not np.any(outcome == i) else {
                "median": percentile(event_month[outcome == i], 0.50),
                "p10": percentile(event_month[outcome == i], 0.10),
                "p90": percentile(event_month[outcome == i], 0.90),
            }
            for i, key in enumerate(POST_FORCE_OUTCOMES)
        },
        "identification_warning": (
            "Conditional structural-prior simulation, not an estimated probability of victory or defeat."
        ),
    }


def strategy_probabilities(state, identity, economic_link, resilience, external_support,
                           commitment, public_field, dialogue, external_friction):
    threat = state / 2.0
    status, defensive, escalatory = public_field[2], public_field[3], public_field[4]

    mainland_intensity = np.array([0.0, 0.35, 0.70, 1.0])
    m_utility = (
        mainland_intensity[None, :] * (
            0.75 * identity[:, None] + 0.32 * threat[:, None]
            + 0.22 * np.maximum(commitment[:, None], 0.0)
            + 0.14 * external_friction
        )
        - mainland_intensity[None, :] ** 2 * (
            0.55 * economic_link[:, None] + 0.62 * resilience[:, None]
            + 0.58 * external_support[:, None] + 0.20 * defensive
        )
        + dialogue * (1.0 - mainland_intensity[None, :])
    )

    # dialogue, status quo management, resilience investment, sovereignty push
    taiwan_escalation = np.array([-0.30, 0.0, 0.08, 0.50])
    t_utility = np.column_stack([
        0.78 * economic_link + 0.50 * (1.0 - identity) + dialogue,
        1.05 * status + 0.40 * (1.0 - threat),
        1.10 * threat + 0.72 * defensive + 0.55 * external_support - 0.30 * economic_link,
        0.88 * identity + 0.72 * escalatory + 0.25 * external_support - 0.72 * threat,
    ])

    external_intensity = np.array([0.0, 0.30, 0.65, 1.0])
    e_utility = (
        external_intensity[None, :] * (0.58 * threat[:, None] + 0.36 * identity[:, None])
        - external_intensity[None, :] ** 2 * (0.66 + 0.34 * economic_link[:, None])
        + external_friction * external_intensity[None, :]
    )
    return (
        softmax(m_utility, 0.38),
        softmax(t_utility, 0.40),
        softmax(e_utility, 0.42),
        mainland_intensity,
        taiwan_escalation,
        external_intensity,
    )


def simulate_scenario(name, cfg, seed):
    rng = np.random.default_rng(seed)
    n = PATHS
    terminal = np.full(n, -1, dtype=np.int8)
    terminal_month = np.full(n, -1, dtype=np.int16)
    state = rng.choice(3, n, p=[0.10, 0.62, 0.28]).astype(np.int8)
    force_features = {
        key: np.full(n, np.nan) for key in (
            "resilience", "external_support", "identity", "economic_link",
            "death_salience", "pressure_state", "crisis_memory",
        )
    }

    identity = np.clip(rng.normal(0.70, 0.05, n), 0.45, 0.90)
    economic_link = np.clip(rng.normal(0.266, 0.025, n), 0.15, 0.38)
    resilience = np.clip(rng.normal(0.56, 0.05, n), 0.35, 0.75)
    external_support = np.clip(rng.normal(0.55, 0.06, n), 0.30, 0.78)
    trust = np.clip(rng.normal(0.64, 0.05, n), 0.40, 0.82)
    polarization = np.clip(rng.normal(0.36, 0.06, n), 0.18, 0.60)
    success_belief = np.clip(rng.normal(0.58, 0.07, n), 0.30, 0.80)
    death_salience = np.clip(rng.normal(0.15, 0.04, n), 0.04, 0.32)
    kinship = np.clip(rng.normal(0.48, 0.08, n), 0.20, 0.75)
    commitment = rng.normal(-1.05, 0.22, n)
    leader_phase = np.zeros(n, dtype=np.int8)
    leader_entered = np.zeros(n, dtype=bool)
    leader_executed = np.zeros(n, dtype=bool)
    leader_intercepted = np.zeros(n, dtype=bool)
    crisis_memory = np.zeros(n)

    mfg_residual_max = 0.0
    ot_cost_sum = 0.0
    ot_iterations_max = 0
    mfg_iterations_max = 0
    trajectory_rows = []
    snapshots = {}

    for month in range(1, MONTHS + 1):
        year = START_YEAR + month / 12.0
        progress = min(month / MONTHS, 1.0)
        active = terminal < 0
        active_count = int(active.sum())
        if active_count == 0:
            break

        target_trade = 0.266 + progress * (cfg["trade_2100"] - 0.266)
        economic_link[active] += 0.018 * (target_trade - economic_link[active])
        economic_cost = np.clip(1.0 - economic_link[active] / 0.40, 0.0, 1.0)
        threat_mean = float(np.mean(state[active] / 2.0))
        trust_mean = float(np.mean(trust[active]))
        polarization_mean = float(np.mean(polarization[active]))
        economic_cost_mean = float(np.mean(economic_cost))
        identity_mean = float(np.mean(identity[active]))
        success_mean = float(np.mean(success_belief[active]))
        death_mean = float(np.mean(death_salience[active]))
        kinship_mean = float(np.mean(kinship[active]))

        public_field, mfg_residual, mfg_iterations = solve_mfg(
            threat_mean, trust_mean, polarization_mean, economic_cost_mean,
            identity_mean, success_mean, death_mean, kinship_mean,
        )
        mfg_residual_max = max(mfg_residual_max, mfg_residual)
        mfg_iterations_max = max(mfg_iterations_max, mfg_iterations)

        misinformation = np.clip(polarization_mean * (1.0 - trust_mean), 0.0, 1.0)
        infrastructure_damage = np.clip(0.10 + 0.38 * threat_mean + 0.15 * crisis_memory[active].mean(), 0.0, 1.0)
        diplomacy_gap = np.clip(0.48 + cfg["external_friction"] - cfg["dialogue"], 0.0, 1.0)
        ot_share, ot_cost, ot_coverage, ot_iterations = solve_unbalanced_ot(
            threat_mean, economic_cost_mean, misinformation,
            infrastructure_damage, diplomacy_gap, cfg["ot_capacity"],
        )
        ot_cost_sum += ot_cost
        ot_iterations_max = max(ot_iterations_max, ot_iterations)

        idx = np.where(active)[0]
        probs_m, probs_t, probs_e, m_intensity, t_escalation, e_intensity = strategy_probabilities(
            state[idx], identity[idx], economic_link[idx], resilience[idx],
            external_support[idx], commitment[idx], public_field,
            cfg["dialogue"], cfg["external_friction"],
        )
        action_m = sample_rows(rng, probs_m)
        action_t = sample_rows(rng, probs_t)
        action_e = sample_rows(rng, probs_e)
        m_level = m_intensity[action_m]
        t_level = t_escalation[action_t]
        e_level = e_intensity[action_e]

        info_gain = ot_share[2] * ot_coverage
        infrastructure_gain = ot_share[1] * ot_coverage
        economic_buffer = ot_share[3] * ot_coverage
        diplomacy_gain = ot_share[4] * ot_coverage
        civil_gain = ot_share[5] * ot_coverage
        monitoring_gain = ot_share[0] * ot_coverage

        rare_shock = rng.random(active_count) < cfg["shock_rate"] * (0.55 + state[idx] / 2.0)
        shock_size = rare_shock * np.clip(np.abs(rng.standard_t(3, active_count)) * 0.18, 0.0, 1.4)

        identity[idx] = np.clip(
            identity[idx] + cfg["identity_drift"] + 0.00018 * state[idx]
            + 0.00020 * m_level - 0.00022 * cfg["dialogue"] - 0.00010 * kinship[idx],
            0.35, 0.98,
        )
        resilience[idx] = np.clip(
            resilience[idx] + 0.0032 * (infrastructure_gain + civil_gain + monitoring_gain)
            + 0.0012 * (action_t == 2) - 0.0028 * shock_size - 0.0012 * state[idx],
            0.15, 0.95,
        )
        external_support[idx] = np.clip(
            external_support[idx] + 0.0015 * e_level + 0.0008 * diplomacy_gain
            - 0.0010 * economic_cost - 0.0008 * (action_t == 3), 0.10, 0.95,
        )
        trust[idx] = np.clip(
            trust[idx] + 0.0025 * info_gain + 0.0015 * diplomacy_gain
            - 0.0025 * polarization[idx] * state[idx] - 0.006 * shock_size
            + rng.normal(0.0, 0.0025, active_count), 0.05, 0.95,
        )
        polarization[idx] = np.clip(
            polarization[idx] + 0.0022 * state[idx] + 0.0020 * m_level
            + 0.0045 * shock_size - 0.0028 * info_gain - 0.0015 * diplomacy_gain,
            0.05, 0.98,
        )
        death_salience[idx] = np.clip(
            0.985 * death_salience[idx] + 0.0030 * state[idx] + 0.0040 * shock_size
            - 0.0015 * civil_gain, 0.02, 0.95,
        )
        success_belief[idx] = np.clip(
            success_belief[idx] + 0.0015 * resilience[idx] + 0.0010 * external_support[idx]
            - 0.0030 * state[idx] - 0.0020 * economic_cost, 0.05, 0.95,
        )

        bounded_commitment = np.clip(commitment[idx], -3.0, 3.0)
        structural_pressure = (
            0.10 + 0.20 * state[idx] + 0.22 * identity[idx]
            + 0.18 * public_field[4] + 0.16 * cfg["external_friction"]
            - 0.24 * economic_link[idx] - 0.28 * resilience[idx]
            - 0.18 * public_field[0]
        )
        leader_jump = (
            rng.random(active_count) < cfg["leader_jump"] * (0.55 + state[idx] / 2.0)
        ) * np.clip(0.35 + np.abs(rng.standard_t(3, active_count)) * 0.20, 0.0, 1.8)
        commitment[idx] = np.clip(
            commitment[idx] + 0.065 * (-bounded_commitment ** 3 + 1.02 * bounded_commitment + structural_pressure)
            + rng.normal(0.0, 0.035, active_count) + leader_jump, -3.0, 3.0,
        )

        phase = leader_phase[idx]
        can_enter = phase == 0
        enter = can_enter & (rng.random(active_count) < 0.006 * sigmoid(2.2 * (commitment[idx] - 0.25)))
        phase[enter] = 1
        leader_entered[idx[enter]] = True
        in_chain = (phase >= 1) & (phase <= 4)
        barrier = np.clip(
            0.22 + 0.35 * resilience[idx] + 0.22 * trust[idx]
            + 0.18 * info_gain + 0.12 * diplomacy_gain - 0.22 * polarization[idx],
            0.04, 0.92,
        )
        advance_h = np.clip(0.05 + 0.16 * sigmoid(2.0 * commitment[idx]) + 0.04 * phase, 0.0, 0.42)
        intercept_h = np.clip(0.03 + 0.25 * barrier, 0.0, 0.34)
        abandon_h = np.clip(0.04 + 0.14 * trust[idx] + 0.10 * economic_cost, 0.02, 0.28)
        total_h = advance_h + intercept_h + abandon_h
        event = in_chain & (rng.random(active_count) < 1.0 - np.exp(-total_h))
        choice = rng.random(active_count) * total_h
        advance = event & (choice < advance_h)
        intercepted = event & ~advance & (choice < advance_h + intercept_h)
        abandoned = event & ~advance & ~intercepted
        phase[intercepted | abandoned] = 0
        leader_intercepted[idx[intercepted]] = True
        phase[advance] += 1
        executed = advance & (phase >= 5)
        leader_executed[idx[executed]] = True
        phase[executed] = 0
        leader_phase[idx] = phase

        crisis_memory[idx] = np.clip(
            0.92 * crisis_memory[idx] + 0.18 * (state[idx] == 2)
            + 0.24 * executed + 0.12 * shock_size, 0.0, 2.5,
        )

        escalation_delta = (
            cfg["pressure_drift"] * progress + 0.42 * m_level + 0.28 * np.maximum(t_level, 0.0)
            + 0.22 * e_level + 0.30 * crisis_memory[idx] + 0.30 * public_field[4]
            - 0.44 * resilience[idx] - 0.26 * diplomacy_gain - 0.28 * public_field[0]
            - 0.18 * np.maximum(-t_level, 0.0)
        )
        matrices = adjusted_transition(BASE_TRANSITION, escalation_delta)
        row_probs = matrices[np.arange(active_count), state[idx], :]
        state[idx] = sample_rows(rng, row_probs).astype(np.int8)

        # Competing terminal hazards. These are transparent scenario priors because
        # none of the terminal events exists in the historical estimation sample.
        force_baseline_by_stratum = np.array([-10.75, -9.50, -8.25])
        peace_baseline_by_stratum = np.array([-8.75, -9.20, -9.65])
        separation_baseline_by_stratum = np.array([-9.65, -10.35, -11.05])
        force_linear = (
            force_baseline_by_stratum[state[idx]] + 0.95 * m_level + 0.62 * crisis_memory[idx]
            + 0.85 * executed + 0.55 * public_field[4] + 0.42 * progress
            - 1.05 * resilience[idx] - 0.78 * external_support[idx]
            - 0.58 * economic_link[idx] - 0.48 * death_salience[idx]
        )
        peace_linear = (
            peace_baseline_by_stratum[state[idx]] + 1.15 * public_field[0] + 1.05 * economic_link[idx]
            + 0.85 * diplomacy_gain + 0.55 * trust[idx] - 1.15 * identity[idx]
            - 0.35 * progress
        )
        separation_linear = (
            separation_baseline_by_stratum[state[idx]] + 1.05 * identity[idx]
            + 0.62 * progress + 0.42 * external_support[idx]
            + 0.40 * public_field[2] - 0.35 * m_level
        )
        h_force = np.exp(np.clip(force_linear, -15.0, -3.0))
        h_peace = np.exp(np.clip(peace_linear, -15.0, -3.0))
        h_separation = np.exp(np.clip(separation_linear, -15.0, -3.0))
        hazards = np.column_stack([h_force, h_peace, h_separation])
        total_hazard = hazards.sum(axis=1)
        exit_probability = 1.0 - np.exp(-total_hazard)
        exits = rng.random(active_count) < exit_probability
        if exits.any():
            conditional = hazards[exits] / total_hazard[exits, None]
            outcome = sample_rows(rng, conditional)
            exit_idx = idx[exits]
            terminal[exit_idx] = outcome.astype(np.int8)
            terminal_month[exit_idx] = month
            force_exit_idx = exit_idx[outcome == 0]
            if len(force_exit_idx):
                force_features["resilience"][force_exit_idx] = resilience[force_exit_idx]
                force_features["external_support"][force_exit_idx] = external_support[force_exit_idx]
                force_features["identity"][force_exit_idx] = identity[force_exit_idx]
                force_features["economic_link"][force_exit_idx] = economic_link[force_exit_idx]
                force_features["death_salience"][force_exit_idx] = death_salience[force_exit_idx]
                force_features["pressure_state"][force_exit_idx] = state[force_exit_idx]
                force_features["crisis_memory"][force_exit_idx] = crisis_memory[force_exit_idx]

        rounded_year = int(round(year))
        if abs(year - rounded_year) < 1e-9 and rounded_year in SNAPSHOT_YEARS:
            dist = [float(np.mean(terminal == outcome)) for outcome in range(3)]
            status = float(np.mean(terminal < 0))
            current_active = terminal < 0
            latent = [float(np.mean(state[current_active] == s)) if current_active.any() else 0.0 for s in range(3)]
            snapshots[str(rounded_year)] = {
                "force": dist[0], "peace": dist[1], "separation": dist[2], "status_quo": status,
                "latent_normal_given_status": latent[0],
                "latent_pressure_given_status": latent[1],
                "latent_crisis_given_status": latent[2],
                "mean_identity_distance": float(identity[current_active].mean()) if current_active.any() else 0.0,
                "mean_economic_link": float(economic_link[current_active].mean()) if current_active.any() else 0.0,
                "mean_resilience": float(resilience[current_active].mean()) if current_active.any() else 0.0,
                "mfg": dict(zip(MFG_ACTIONS, map(float, public_field))),
                "ot": dict(zip(OT_TARGETS, map(float, ot_share))),
                "ot_mean_cost": ot_cost,
            }
            trajectory_rows.append({
                "scenario": name, "year": rounded_year,
                **{key: value for key, value in zip(TERMINALS, dist + [status])},
                **{f"latent_{s}": latent[i] for i, s in enumerate(("normal", "pressure", "crisis"))},
                **{f"mfg_{i}": float(public_field[i]) for i in range(5)},
                **{f"ot_{i}": float(ot_share[i]) for i in range(6)},
                "identity_distance": snapshots[str(rounded_year)]["mean_identity_distance"],
                "economic_link": snapshots[str(rounded_year)]["mean_economic_link"],
                "resilience": snapshots[str(rounded_year)]["mean_resilience"],
            })

    terminal_counts = [float(np.mean(terminal == outcome)) for outcome in range(3)]
    status = float(np.mean(terminal < 0))
    final = dict(zip(TERMINALS, terminal_counts + [status]))
    event_times = {}
    for outcome, key in enumerate(TERMINALS[:3]):
        months = terminal_month[terminal == outcome]
        event_times[key] = None if len(months) == 0 else {
            "median_year": START_YEAR + percentile(months, 0.50) / 12.0,
            "p10_year": START_YEAR + percentile(months, 0.10) / 12.0,
            "p90_year": START_YEAR + percentile(months, 0.90) / 12.0,
        }

    force_mask = terminal == 0
    post_force = simulate_post_force_campaign(
        {key: values[force_mask] for key, values in force_features.items()},
        seed + 700001,
    )
    post_force["unconditional_by_2100_force_and_outcome"] = {
        key: final["force"] * value
        for key, value in post_force["outcome_conditional_on_force"].items()
    }

    return {
        "scenario": name,
        "paths": n,
        "months": MONTHS,
        "snapshots": snapshots,
        "final_2100": final,
        "event_time_conditional": event_times,
        "post_force_campaign": post_force,
        "leader_process": {
            "entered": float(leader_entered.mean()),
            "executed": float(leader_executed.mean()),
            "intercepted_at_least_once": float(leader_intercepted.mean()),
        },
        "solver_diagnostics": {
            "max_mfg_l1_residual": mfg_residual_max,
            "max_mfg_iterations": mfg_iterations_max,
            "max_ot_iterations": ot_iterations_max,
            "mean_ot_cost": ot_cost_sum / MONTHS,
        },
        "trajectory_rows": trajectory_rows,
    }


def simulate_governance_stress(name, cfg, seed, paths=30000, months=360):
    """Non-operational post-entry governance stress test inspired by Yiwei mechanisms."""
    rng = np.random.default_rng(seed)

    # Pre-entry information stage: known strategic danger does not imply knowing
    # the realized time, location, or whether local organizations are prepared.
    arrival_days = np.clip(
        rng.lognormal(math.log(cfg["arrival_days"]), cfg["arrival_uncertainty"], paths),
        5.0, 180.0,
    )
    location_entropy = np.clip(
        rng.normal(cfg["location_entropy"], 0.08, paths), 0.01, 0.99,
    )
    elite_awareness = np.clip(
        rng.beta(8.0 * cfg["elite_awareness"], 8.0 * (1.0 - cfg["elite_awareness"]), paths),
        0.01, 0.99,
    )
    credibility = 1.0 - 0.68 * location_entropy
    local_awareness = np.clip(
        cfg["local_diffusion"] * credibility * (0.35 + 0.65 * elite_awareness)
        * (1.0 - np.exp(-arrival_days / 28.0)), 0.0, 0.99,
    )
    public_awareness = np.clip(
        cfg["public_diffusion"] * credibility * (0.25 + 0.75 * local_awareness)
        * (1.0 - np.exp(-arrival_days / 40.0)), 0.0, 0.99,
    )
    local_preparedness = np.clip(
        0.12 + 0.62 * local_awareness + 0.20 * public_awareness
        - 0.20 * location_entropy, 0.0, 0.99,
    )
    surprise_index = np.clip(
        (1.0 - elite_awareness) * (1.0 - local_awareness)
        * (1.0 - local_preparedness) * (0.45 + 0.55 * location_entropy),
        0.0, 1.0,
    )

    control = np.clip(rng.normal(cfg["control"], 0.045, paths), 0.02, 0.98)
    resistance = np.clip(rng.normal(cfg["resistance"], 0.055, paths), 0.02, 0.98)
    governance = np.clip(rng.normal(cfg["governance"], 0.040, paths), 0.01, 0.95)
    service = np.clip(rng.normal(cfg["service"], 0.045, paths), 0.01, 0.95)
    legitimacy = np.clip(rng.normal(cfg["legitimacy"], 0.045, paths), 0.01, 0.95)
    information = np.clip(rng.normal(cfg["information"], 0.055, paths), 0.01, 0.98)
    social_network = np.clip(rng.normal(0.44 + 0.18 * cfg["network_adapt"], 0.05, paths), 0.05, 0.98)
    grievance = np.clip(rng.normal(0.30 + 0.20 * cfg["coercion"], 0.05, paths), 0.02, 0.95)
    control = np.clip(control + 0.15 * surprise_index, 0.0, 1.0)
    resistance = np.clip(resistance - 0.08 * surprise_index + 0.06 * local_preparedness, 0.0, 1.0)

    # 0 active, 1 fled, 2 killed, 3 disconnected. This is an organizational
    # coordination layer, not a model of any named contemporary leader.
    leader_status = np.zeros(paths, dtype=np.int8)
    leadership_coordination = np.clip(rng.normal(0.72, 0.06, paths), 0.35, 0.92)
    martyr_memory = np.zeros(paths)
    coercion_memory = np.full(paths, 0.18 + 0.24 * cfg["coercion"])
    identity_distance = np.clip(rng.normal(0.72, 0.07, paths), 0.38, 0.94)
    external_social_link = np.clip(rng.normal(0.58, 0.08, paths), 0.24, 0.88)
    cumulative_cost = np.zeros(paths)
    fiscal_gap = np.zeros(paths)
    humanitarian_pressure = np.zeros(paths)
    stable_streak = np.zeros(paths, dtype=np.int16)
    stable_month = np.full(paths, -1, dtype=np.int16)
    snapshots = {}

    for month in range(1, months + 1):
        leader_active = leader_status == 0
        if month <= 12 and leader_active.any():
            active_idx = np.where(leader_active)[0]
            flight_h = cfg["leader_flight_hazard"] * (
                0.55 + 0.70 * (1.0 - control[active_idx]) + 0.35 * surprise_index[active_idx]
            )
            death_h = cfg["leader_death_hazard"] * (
                0.45 + 0.70 * resistance[active_idx] + 0.30 * cfg["damage"]
            )
            disconnect_h = cfg["leader_disconnect_hazard"] * (
                0.50 + 0.65 * cfg["damage"] + 0.35 * (1.0 - information[active_idx])
            )
            hazards = np.column_stack([flight_h, death_h, disconnect_h])
            total_h = hazards.sum(axis=1)
            event = rng.random(len(active_idx)) < 1.0 - np.exp(-total_h)
            if event.any():
                conditional = hazards[event] / total_h[event, None]
                outcome = sample_rows(rng, conditional) + 1
                event_idx = active_idx[event]
                leader_status[event_idx] = outcome.astype(np.int8)
                leadership_coordination[event_idx] *= np.choose(
                    outcome - 1, [0.52, 0.42, 0.66]
                )
                martyr_memory[event_idx[outcome == 2]] += 0.55

        martyr_memory *= 0.975
        coercion_memory = np.clip(
            0.995 * coercion_memory + 0.010 * cfg["coercion"]
            + 0.006 * humanitarian_pressure - 0.006 * legitimacy, 0.0, 1.5,
        )
        effective_coordination = np.clip(
            leadership_coordination
            + 0.42 * social_network * (1.0 - leadership_coordination), 0.0, 1.0,
        )
        displacement = np.clip(
            cfg["damage"] * (1.0 - service) + 0.45 * resistance + rng.normal(0, 0.025, paths),
            0.0, 1.0,
        )
        social_network = np.clip(
            0.955 * social_network
            + 0.026 * cfg["network_adapt"] * (resistance + grievance + displacement)
            + 0.010 * martyr_memory - 0.032 * governance * service, 0.02, 1.0,
        )
        information = np.clip(
            information + 0.024 * governance * (1.0 - information)
            + 0.012 * social_network * (1.0 - information)
            - 0.018 * cfg["friction"] * information, 0.01, 0.99,
        )
        grievance = np.clip(
            0.94 * grievance + 0.030 * cfg["coercion"] * control
            + 0.024 * displacement - 0.040 * service - 0.025 * legitimacy,
            0.01, 0.99,
        )
        mobilization_pressure = (
            0.020 * social_network + 0.017 * grievance
            + 0.009 * (1.0 - information)
            + 0.010 * effective_coordination * social_network
            + 0.009 * coercion_memory * identity_distance
            + 0.005 * external_social_link
        )
        demobilization_pressure = (
            0.038 * governance * service + 0.026 * legitimacy + 0.015 * control
        )
        resistance = np.clip(
            resistance + mobilization_pressure * (1.0 - resistance)
            - demobilization_pressure * resistance
            + rng.normal(0, 0.007, paths), 0.0, 1.0,
        )
        control = np.clip(
            control + 0.038 * cfg["coercion"] * (1.0 - control)
            + 0.024 * governance - 0.034 * resistance * control
            - 0.016 * cfg["friction"] * (1.0 - service)
            + rng.normal(0, 0.006, paths), 0.0, 1.0,
        )
        governance = np.clip(
            governance + 0.030 * cfg["governance_invest"] * control * (1.0 - governance)
            + 0.012 * information - 0.026 * resistance * governance
            - 0.014 * cfg["friction"] * (1.0 - control), 0.0, 1.0,
        )
        service = np.clip(
            service + 0.034 * cfg["service_invest"] * governance * (1.0 - service)
            - 0.022 * cfg["damage"] * (1.0 - control)
            - 0.018 * resistance * service, 0.0, 1.0,
        )
        humanitarian_pressure = np.clip(
            0.92 * humanitarian_pressure + 0.055 * (1.0 - service)
            + 0.035 * displacement + 0.022 * resistance, 0.0, 2.5,
        )
        legitimacy = np.clip(
            legitimacy + 0.020 * service + 0.016 * governance
            - 0.028 * cfg["coercion"] - 0.018 * humanitarian_pressure
            - 0.014 * cfg["damage"] + rng.normal(0, 0.004, paths), 0.0, 1.0,
        )
        identity_distance = np.clip(
            identity_distance + 0.0012 * coercion_memory + 0.0008 * humanitarian_pressure
            + 0.0005 * external_social_link - 0.0014 * legitimacy * service
            - 0.0006 * governance, 0.10, 1.0,
        )
        external_social_link = np.clip(
            0.998 * external_social_link + 0.0012 * identity_distance
            - 0.0009 * legitimacy * governance, 0.05, 0.98,
        )
        monthly_cost = (
            0.70 * cfg["coercion"] + 0.85 * (1.0 - control)
            + 0.75 * resistance + 0.65 * cfg["friction"] * (1.0 - service)
            + 0.55 * cfg["governance_invest"] + 0.60 * cfg["service_invest"]
            + 0.70 * humanitarian_pressure
        )
        cumulative_cost += monthly_cost
        local_revenue = 1.65 * control * governance * service * (0.35 + 0.65 * legitimacy)
        fiscal_gap += monthly_cost - local_revenue

        stable = (
            (control > 0.75) & (governance > 0.65) & (service > 0.65)
            & (resistance < 0.25) & (legitimacy > 0.45)
            & (local_revenue >= monthly_cost)
        )
        stable_streak = np.where(stable, stable_streak + 1, 0)
        newly_stable = (stable_month < 0) & (stable_streak >= 24)
        stable_month[newly_stable] = month

        if month in (24, 60, 120, 240, 360):
            fragile_control = (
                (control > 0.70)
                & ((governance < 0.60) | (service < 0.60) | (legitimacy < 0.40) | (resistance > 0.30))
            )
            snapshots[str(month // 12)] = {
                "stable_control_probability": float(np.mean((stable_month > 0) & (stable_month <= month))),
                "fragile_control_probability": float(np.mean(fragile_control)),
                "mean_control": float(control.mean()),
                "mean_governance": float(governance.mean()),
                "mean_service": float(service.mean()),
                "mean_legitimacy": float(legitimacy.mean()),
                "mean_resistance": float(resistance.mean()),
                "mean_information_reach": float(information.mean()),
                "mean_effective_coordination": float(effective_coordination.mean()),
                "mean_identity_distance": float(identity_distance.mean()),
                "mean_coercion_memory": float(coercion_memory.mean()),
                "mean_humanitarian_pressure": float(humanitarian_pressure.mean()),
                "cumulative_cost_median": percentile(cumulative_cost, 0.50),
                "fiscal_gap_median": percentile(fiscal_gap, 0.50),
            }

    return {
        "scenario": name,
        "paths": paths,
        "months": months,
        "snapshots": snapshots,
        "pre_entry_information": {
            "arrival_days_p10_p50_p90": [
                percentile(arrival_days, 0.10), percentile(arrival_days, 0.50), percentile(arrival_days, 0.90)
            ],
            "mean_location_entropy": float(location_entropy.mean()),
            "mean_elite_awareness": float(elite_awareness.mean()),
            "mean_local_awareness": float(local_awareness.mean()),
            "mean_public_awareness": float(public_awareness.mean()),
            "mean_surprise_index": float(surprise_index.mean()),
        },
        "leadership_exit": {
            "still_active": float(np.mean(leader_status == 0)),
            "fled": float(np.mean(leader_status == 1)),
            "killed": float(np.mean(leader_status == 2)),
            "disconnected": float(np.mean(leader_status == 3)),
            "mean_effective_coordination_30y": float(effective_coordination.mean()),
        },
        "stable_control_by_30y": float(np.mean(stable_month > 0)),
        "stable_time_conditional": None if not np.any(stable_month > 0) else {
            "median_years": percentile(stable_month[stable_month > 0], 0.50) / 12.0,
            "p10_years": percentile(stable_month[stable_month > 0], 0.10) / 12.0,
            "p90_years": percentile(stable_month[stable_month > 0], 0.90) / 12.0,
        },
        "final_resistance_p10_p50_p90": [
            percentile(resistance, 0.10), percentile(resistance, 0.50), percentile(resistance, 0.90)
        ],
        "cumulative_cost_p10_p50_p90": [
            percentile(cumulative_cost, 0.10), percentile(cumulative_cost, 0.50), percentile(cumulative_cost, 0.90)
        ],
        "interpretation": (
            "Conditional mechanism stress test after a force event. Historical Yiwei parameters are not "
            "transferred; only information layering, adaptive networks, governance lag and service costs are reused."
        ),
    }


def validate(results, governance_results):
    by_name = {result["scenario"]: result for result in results}
    for result in results:
        previous_terminal = -1.0
        for snapshot in result["snapshots"].values():
            total = sum(snapshot[key] for key in TERMINALS)
            assert abs(total - 1.0) < 1e-12
            terminal_mass = 1.0 - snapshot["status_quo"]
            assert terminal_mass + 1e-12 >= previous_terminal
            previous_terminal = terminal_mass
            assert abs(sum(snapshot["mfg"].values()) - 1.0) < 1e-10
            assert abs(sum(snapshot["ot"].values()) - 1.0) < 1e-10
        assert result["solver_diagnostics"]["max_mfg_l1_residual"] < 1e-8
        assert result["leader_process"]["executed"] <= result["leader_process"]["entered"] + 1e-12
        assert abs(sum(result["post_force_campaign"]["outcome_conditional_on_force"].values()) - 1.0) < 1e-12
        assert abs(sum(result["post_force_campaign"]["alliance_intervention_conditional"].values()) - 1.0) < 1e-12
    assert by_name["adverse"]["final_2100"]["force"] > by_name["baseline"]["final_2100"]["force"]
    assert by_name["baseline"]["final_2100"]["force"] > by_name["stabilization"]["final_2100"]["force"]
    assert by_name["stabilization"]["final_2100"]["peace"] > by_name["baseline"]["final_2100"]["peace"]
    governance_by_name = {result["scenario"]: result for result in governance_results}
    assert governance_by_name["治理与公共服务优先"]["stable_control_by_30y"] > governance_by_name["多层网络分散抵抗"]["stable_control_by_30y"]
    for result in governance_results:
        for snapshot in result["snapshots"].values():
            for key, value in snapshot.items():
                assert math.isfinite(value), (result["scenario"], key, value)


def write_outputs(results, governance_results):
    summary = {
        "model": "reversible hidden pressure + three-player stochastic response + aggregate Logit MFG + unbalanced entropic OT + finite leader jump gates + competing terminal risks + post-force campaign/alliance submodel",
        "version": "V3",
        "start_year": START_YEAR,
        "end_year": END_YEAR,
        "paths_per_scenario": PATHS,
        "historical_identification_warning": (
            "Only reversible pressure dynamics are estimated from 2020-2026 observations. "
            "Terminal hazards are transparent scenario priors, not historically identified frequencies."
        ),
        "survival_model": {
            "family": "hidden-pressure-state-stratified cause-specific Cox competing-risks model with time-varying covariates",
            "strata": ["normal", "pressure", "crisis"],
            "causes": ["force", "peace", "separation"],
            "survival_state": "status_quo",
            "discrete_time_mapping": "P(any event in month t)=1-exp(-sum_k h_k(t)); P(k|event)=h_k(t)/sum_j h_j(t)",
            "cumulative_incidence": "F_k(t)=integral_0^t S(u-) h_k(u) du",
            "identification": "baseline hazards and terminal-event coefficients are scenario priors; only the 2020-2026 reversible pressure process is estimated",
        },
        "post_force_scope": (
            "A standardized 60-month strategic campaign and alliance-intervention simulation begins after each force-onset event. "
            "It models high-level outcomes only and contains no targeting or operational planning."
        ),
        "official_2026_anchors": {
            "taiwan_defense_budget_nato_share_gdp": 0.0332,
            "taiwan_mnd_planned_budget_ntd_billion": 806.0,
            "broad_status_quo_support_2025": 0.864,
            "reject_one_country_two_systems_2025": 0.837,
            "export_share_cn_hk_2025_local_series": 0.266,
        },
        "results": [{key: value for key, value in result.items() if key != "trajectory_rows"} for result in results],
        "post_force_governance_stress_tests": governance_results,
        "verification": "TAIWAN_MFG_OT_HYBRID_VERIFICATION: PASS",
    }
    summary_path = OUT / "台海MFG_OT混合模型_仿真摘要.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    forecast_path = OUT / "台海MFG_OT混合模型_长期预测.csv"
    with forecast_path.open("w", newline="", encoding="utf-8-sig") as handle:
        fieldnames = [
            "scenario", "year", "force", "peace", "separation", "status_quo",
            "latent_normal", "latent_pressure", "latent_crisis",
            "identity_distance", "economic_link", "resilience",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            for year, snapshot in result["snapshots"].items():
                writer.writerow({
                    "scenario": result["scenario"], "year": year,
                    "force": snapshot["force"], "peace": snapshot["peace"],
                    "separation": snapshot["separation"], "status_quo": snapshot["status_quo"],
                    "latent_normal": snapshot["latent_normal_given_status"],
                    "latent_pressure": snapshot["latent_pressure_given_status"],
                    "latent_crisis": snapshot["latent_crisis_given_status"],
                    "identity_distance": snapshot["mean_identity_distance"],
                    "economic_link": snapshot["mean_economic_link"],
                    "resilience": snapshot["mean_resilience"],
                })

    allocation_path = OUT / "台海MFG_OT混合模型_策略与资源分布.csv"
    allocation_fields = ["scenario", "year"] + [f"mfg_{i}" for i in range(5)] + [f"ot_{i}" for i in range(6)]
    with allocation_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=allocation_fields)
        writer.writeheader()
        for result in results:
            for row in result["trajectory_rows"]:
                writer.writerow({key: row[key] for key in allocation_fields})

    governance_path = OUT / "台海MFG_OT混合模型_乙未机制压力测试.csv"
    governance_fields = [
        "scenario", "year", "stable_control_probability", "mean_control",
        "fragile_control_probability",
        "mean_governance", "mean_service", "mean_legitimacy", "mean_resistance",
        "mean_information_reach", "mean_humanitarian_pressure", "cumulative_cost_median",
        "mean_effective_coordination", "mean_identity_distance", "mean_coercion_memory",
        "fiscal_gap_median",
    ]
    with governance_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=governance_fields)
        writer.writeheader()
        for result in governance_results:
            for year, snapshot in result["snapshots"].items():
                writer.writerow({"scenario": result["scenario"], "year": year, **snapshot})

    post_force_path = OUT / "台海MFG_OT混合模型_冲突后战役与联盟介入.csv"
    post_force_fields = [
        "scenario", "force_incidence_by_2100", *ALLIANCE_LEVELS,
        *POST_FORCE_OUTCOMES, "joint_defense_halted_or_withdrawn",
        "us_japan_direct_and_halted_or_withdrawn",
    ]
    with post_force_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=post_force_fields)
        writer.writeheader()
        for result in results:
            post = result["post_force_campaign"]
            writer.writerow({
                "scenario": result["scenario"],
                "force_incidence_by_2100": result["final_2100"]["force"],
                **post["alliance_intervention_conditional"],
                **post["outcome_conditional_on_force"],
                "joint_defense_halted_or_withdrawn": post["joint_defense_halted_or_withdrawn"],
                "us_japan_direct_and_halted_or_withdrawn": post["us_japan_direct_and_halted_or_withdrawn"],
            })
    return summary_path, forecast_path, allocation_path, governance_path, post_force_path


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    results = [
        simulate_scenario(name, cfg, SEED + i * 1009)
        for i, (name, cfg) in enumerate(SCENARIOS.items())
    ]
    governance_results = [
        simulate_governance_stress(name, cfg, SEED + 10000 + i * 1013)
        for i, (name, cfg) in enumerate(GOVERNANCE_SCENARIOS.items())
    ]
    validate(results, governance_results)
    paths = write_outputs(results, governance_results)
    compact = {
        result["scenario"]: {
            "final_2100": result["final_2100"],
            "event_time_conditional": result["event_time_conditional"],
            "leader_process": result["leader_process"],
            "solver_diagnostics": result["solver_diagnostics"],
        }
        for result in results
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2))
    print(json.dumps({r["scenario"]: {
        "stable_control_by_30y": r["stable_control_by_30y"],
        "final_resistance_p10_p50_p90": r["final_resistance_p10_p50_p90"],
        "cumulative_cost_p10_p50_p90": r["cumulative_cost_p10_p50_p90"],
    } for r in governance_results}, ensure_ascii=False, indent=2))
    for path in paths:
        print(path)
    print("TAIWAN_MFG_OT_HYBRID_VERIFICATION: PASS")


if __name__ == "__main__":
    main()
