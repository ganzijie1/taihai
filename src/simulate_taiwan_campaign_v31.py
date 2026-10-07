import csv
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

GIS_SUMMARY_PATH = OUT / "台海V3.6_GIS机动走廊摘要.json"
GRAPHON_MFG_PATH = OUT / "台海V3.6_Graphon平均场走廊韧性.json"

SEED = 20261006
PATHS = int(os.environ.get("TAIWAN_PATHS", "6000"))
WEEKS = 780
HORIZON_WEEKS = (260, 520, 780)
YEARS = {2030: 0.80, 2050: 1.00, 2075: 1.17, 2100: 1.26}
OUTCOMES = [
    "defense_restored_or_withdrawal",
    "negotiated_ceasefire",
    "limited_contested_control",
    "broad_control",
    "frozen_stalemate",
]

# Discrete-event modes used by the hybrid DEDS layer.
FACILITY_OPERATIONAL = 0
FACILITY_DEGRADED = 1
FACILITY_DISABLED = 2
ALLY_NOT_ARRIVED = 0
ALLY_RAMPING = 1
ALLY_DEPLOYED = 2
PORT_UNAVAILABLE = 0
PORT_CONVERTING = 1
PORT_OPERATIONAL = 2


@dataclass(frozen=True)
class AllianceCase:
    label: str
    median_delay: float
    delay_sigma: float
    air_strength: float
    sea_strength: float
    resupply: float
    base_access: float
    japan_direct: float


ALLIANCE_CASES = {
    "timely_full": AllianceCase(
        "美日及时全面介入", 1.5, 0.28, 0.94, 0.92, 0.90, 0.92, 0.90
    ),
    "limited": AllianceCase(
        "有限介入", 3.0, 0.42, 0.38, 0.32, 0.42, 0.48, 0.20
    ),
    "delayed": AllianceCase(
        "延迟介入", 12.0, 0.48, 0.78, 0.74, 0.72, 0.76, 0.58
    ),
    "taiwan_alone": AllianceCase(
        "台湾单独作战", 999.0, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0
    ),
}


STRUCTURES = {
    "defender_favorable": {
        "combat_shift": -0.16,
        "defender_repair": 1.18,
        "attacker_repair": 0.88,
        "warning": 0.12,
        "attacker_stock": 0.82,
        "defender_stock": 1.18,
        "lift": 0.88,
    },
    "central": {
        "combat_shift": 0.0,
        "defender_repair": 1.0,
        "attacker_repair": 1.0,
        "warning": 0.0,
        "attacker_stock": 1.0,
        "defender_stock": 1.0,
        "lift": 1.0,
    },
    "attacker_favorable": {
        "combat_shift": 0.18,
        "defender_repair": 0.82,
        "attacker_repair": 1.14,
        "warning": -0.12,
        "attacker_stock": 1.20,
        "defender_stock": 0.82,
        "lift": 1.15,
    },
}


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -35.0, 35.0)))


def safe_ratio(a, b):
    return a / np.maximum(b, 1e-6)


def load_gis_calibration():
    """Load public-data regional mobility calibration, with a declared fallback."""
    if GIS_SUMMARY_PATH.exists():
        payload = json.loads(GIS_SUMMARY_PATH.read_text(encoding="utf-8"))
        aggregate = payload["aggregate"]
        calibration = {
            "source": "public GIS preprocessing",
            "mobility": float(aggregate["mobility_index"]),
            "friction": float(aggregate["terrain_friction_index"]),
            "capacity": float(aggregate["bottleneck_capacity"]),
            "redundancy": float(aggregate["redundancy"]),
        }
        if GRAPHON_MFG_PATH.exists():
            mfg = json.loads(GRAPHON_MFG_PATH.read_text(encoding="utf-8"))
            calibration["dynamic_resilience"] = float(mfg["mean_average_mobility"])
            calibration["source"] += " + major-minor graphon MFG"
        else:
            calibration["dynamic_resilience"] = 0.80
        return calibration
    return {
        "source": "declared fallback pending public GIS preprocessing",
        "mobility": 0.58,
        "friction": 0.42,
        "capacity": 0.60,
        "redundancy": 0.45,
        "dynamic_resilience": 0.80,
    }


GIS = load_gis_calibration()


def optimal_search_allocation(prior, detection_rate, budget):
    """KKT water filling for max sum p_i(1-exp(-a_i e_i)), sum e_i <= B."""
    prior = np.maximum(prior, 1e-8)
    detection_rate = np.maximum(detection_rate, 1e-5)
    budget = np.maximum(np.asarray(budget), 0.0)
    active = np.ones_like(prior, dtype=bool)
    effort = np.zeros_like(prior)
    log_pa = np.log(prior * detection_rate)
    for _ in range(prior.shape[1]):
        inv_rate = np.where(active, 1.0 / detection_rate, 0.0)
        denominator = np.maximum(inv_rate.sum(axis=1), 1e-9)
        log_lambda = (
            (np.where(active, log_pa / detection_rate, 0.0)).sum(axis=1)
            - budget
        ) / denominator
        candidate = (log_pa - log_lambda[:, None]) / detection_rate
        new_active = active & (candidate > 0.0)
        if np.array_equal(new_active, active):
            effort = np.where(active, candidate, 0.0)
            break
        active = new_active
        effort = np.where(active, candidate, 0.0)
    effort_sum = effort.sum(axis=1, keepdims=True)
    effort *= budget[:, None] / np.maximum(effort_sum, 1e-9)
    detection = 1.0 - np.exp(-detection_rate * effort)
    return effort, detection


def q(values, probability):
    return float(np.quantile(values, probability))


def facility_mode(availability):
    return np.where(
        availability < 0.20,
        FACILITY_DISABLED,
        np.where(availability < 0.65, FACILITY_DEGRADED, FACILITY_OPERATIONAL),
    ).astype(np.int8)


def allied_logistics_max_flow(forward_base, rear_base, port, fuel, c4isr, dispersion):
    """Closed-form max flow for the small series-parallel allied logistics graph."""
    source_to_rear = np.minimum(
        0.58 * rear_base + 0.42 * port,
        0.62 * fuel + 0.38 * c4isr,
    )
    rear_to_air = np.minimum(
        0.72 * rear_base + 0.28 * c4isr,
        0.62 * fuel + 0.38 * dispersion,
    )
    air_to_theater = np.minimum(
        0.70 * forward_base + 0.30 * dispersion,
        0.55 * fuel + 0.45 * c4isr,
    )
    rear_to_sea = np.minimum(
        0.68 * port + 0.32 * rear_base,
        0.62 * fuel + 0.38 * c4isr,
    )
    sea_to_theater = np.minimum(
        0.78 * port + 0.22 * dispersion,
        0.66 * fuel + 0.34 * c4isr,
    )
    air_flow = np.minimum(source_to_rear, np.minimum(rear_to_air, air_to_theater))
    sea_flow = np.minimum(source_to_rear, np.minimum(rear_to_sea, sea_to_theater))
    parallel_sink = 0.38 * air_flow + 0.62 * sea_flow
    total_flow = np.minimum(source_to_rear, parallel_sink)
    return (
        np.clip(air_flow, 0.0, 1.0),
        np.clip(sea_flow, 0.0, 1.0),
        np.clip(total_flow, 0.0, 1.0),
    )


def event_week_quantiles(weeks):
    observed = weeks[weeks > 0]
    if not len(observed):
        return None
    return [q(observed, 0.10), q(observed, 0.50), q(observed, 0.90)]


def simulate_case(case_key, case, conflict_year, capability_ratio, structure_key, structure, seed):
    rng = np.random.default_rng(seed)
    n = PATHS

    preparedness = np.clip(rng.beta(6.0, 3.0, n) + structure["warning"], 0.22, 0.98)
    attacker_quality = np.clip(rng.lognormal(0.0, 0.11, n), 0.70, 1.35)
    defender_quality = np.clip(rng.lognormal(0.0, 0.13, n), 0.68, 1.35)
    # Public GIS supplies the regional mean; scenario draws retain unobserved
    # bridge, tunnel, traffic and damage uncertainty.
    geography_friction = np.clip(
        0.55 * rng.beta(6.5, 2.8, n) + 0.45 * GIS["friction"], 0.24, 0.94
    )
    corridor_mobility = np.clip(
        GIS["mobility"]
        * GIS["dynamic_resilience"]
        * rng.lognormal(0.0, 0.12, n),
        0.16,
        0.96,
    )
    weather_memory = rng.beta(5.0, 3.0, n)

    delay = np.full(n, 999.0)
    if case_key != "taiwan_alone":
        mu = np.log(case.median_delay)
        delay = np.clip(rng.lognormal(mu, case.delay_sigma, n), 0.5, 26.0)

    attacker_air = np.clip(0.92 * capability_ratio * attacker_quality, 0.45, 1.55)
    attacker_sea = np.clip(0.86 * capability_ratio * attacker_quality, 0.40, 1.50)
    defender_air = np.clip(0.46 * defender_quality * (0.68 + 0.42 * preparedness), 0.18, 0.82)
    defender_sea = np.clip(0.31 * defender_quality * (0.70 + 0.38 * preparedness), 0.12, 0.65)
    amphibious_lift = np.clip(
        rng.normal(0.62 * capability_ratio * structure["lift"], 0.08, n), 0.24, 1.18
    )

    attacker_precision = np.clip(
        rng.normal(0.90 * capability_ratio * structure["attacker_stock"], 0.10, n), 0.32, 1.45
    )
    attacker_general = np.clip(
        rng.normal(1.0 * structure["attacker_stock"], 0.08, n), 0.42, 1.45
    )
    defender_precision = np.clip(
        rng.normal((0.62 + 0.18 * preparedness) * structure["defender_stock"], 0.09, n), 0.22, 1.28
    )
    defender_general = np.clip(
        rng.normal(0.82 * structure["defender_stock"], 0.09, n), 0.32, 1.30
    )
    attacker_fuel = np.clip(rng.normal(0.94, 0.06, n), 0.62, 1.10)
    defender_fuel = np.clip(rng.normal(0.68 + 0.18 * preparedness, 0.08, n), 0.30, 1.05)

    defender_port = np.clip(0.72 + 0.24 * preparedness, 0.45, 0.98)
    defender_airfield = np.clip(0.68 + 0.27 * preparedness, 0.40, 0.98)
    attacker_base = np.clip(rng.normal(0.94, 0.025, n), 0.82, 0.99)
    captured_port = np.zeros(n)
    beachhead = np.zeros(n)
    control = np.zeros(n)
    admin_control = np.zeros(n)
    command = np.clip(0.64 + 0.24 * preparedness, 0.42, 0.94)
    defender_org = np.clip(0.64 + 0.23 * preparedness, 0.42, 0.94)
    attacker_logistics = np.clip(rng.normal(0.78, 0.06, n), 0.48, 0.96)
    attacker_tolerance = np.clip(rng.normal(0.76, 0.09, n), 0.42, 0.96)
    defender_tolerance = np.clip(rng.normal(0.78, 0.10, n), 0.38, 0.98)

    outcome = np.full(n, -1, dtype=np.int8)
    event_week = np.full(n, -1, dtype=np.int16)
    broad_streak = np.zeros(n, dtype=np.int16)
    expansion = np.zeros(n, dtype=bool)
    peak_control = np.zeros(n)
    min_def_port = defender_port.copy()
    min_def_airfield = defender_airfield.copy()
    delivered_mass = np.zeros(n)
    attacker_ammo_shortage_week = np.full(n, -1, dtype=np.int16)
    defender_ammo_shortage_week = np.full(n, -1, dtype=np.int16)

    # V3.4 reciprocal long-range strike, finite allied inventories and queues.
    attacker_theater_missiles = np.clip(
        rng.normal(1.02 * capability_ratio * structure["attacker_stock"], 0.12, n), 0.35, 1.65
    )
    attacker_long_range_missiles = np.clip(
        rng.normal(0.72 * capability_ratio * structure["attacker_stock"], 0.11, n), 0.18, 1.35
    )
    attacker_isr = np.clip(rng.normal(0.82, 0.08, n), 0.45, 0.98)
    attacker_launcher_survival = np.clip(rng.normal(0.90, 0.05, n), 0.64, 0.99)

    allied_forward_base = np.clip(rng.normal(0.92, 0.035, n), 0.78, 0.99)
    allied_rear_base = np.clip(rng.normal(0.96, 0.020, n), 0.86, 1.0)
    allied_port = np.clip(rng.normal(0.94, 0.030, n), 0.80, 0.99)
    allied_fuel = np.clip(rng.normal(0.91, 0.040, n), 0.74, 0.99)
    allied_c4isr = np.clip(rng.normal(0.93, 0.030, n), 0.80, 0.99)
    allied_bmd = np.clip(rng.normal(0.72 * structure["defender_stock"], 0.10, n), 0.30, 1.15)
    allied_dispersion = np.clip(
        rng.beta(5.5, 2.8, n) + 0.18 * structure["warning"], 0.28, 0.96
    )
    allied_repair_capacity = np.clip(
        rng.normal(0.70 * structure["defender_repair"], 0.08, n), 0.35, 1.05
    )

    # Ready stocks, repair queues and cumulative arrivals are separate.  The
    # alliance begins outside the theater and arrives through a finite pipeline.
    allied_air_target = np.clip(
        case.air_strength * case.base_access * rng.lognormal(0.0, 0.10, n), 0.0, 1.25
    )
    allied_sea_target = np.clip(
        case.sea_strength * case.base_access * rng.lognormal(0.0, 0.11, n), 0.0, 1.20
    )
    allied_enabler_target = np.clip(
        (0.18 + 0.62 * case.air_strength) * case.base_access
        * rng.lognormal(0.0, 0.12, n), 0.0, 0.95
    )
    allied_air_ready = np.zeros(n)
    allied_sea_ready = np.zeros(n)
    allied_enablers = np.zeros(n)
    allied_air_arrived = np.zeros(n)
    allied_sea_arrived = np.zeros(n)
    allied_enablers_arrived = np.zeros(n)
    allied_air_repair_queue = np.zeros(n)
    allied_sea_repair_queue = np.zeros(n)
    allied_air_turnaround_queue = np.zeros(n)
    allied_sea_service_queue = np.zeros(n)
    allied_precision_stock = np.zeros(n)
    allied_general_stock = np.zeros(n)
    allied_operational_fuel = np.zeros(n)
    allied_spare_stock = np.zeros(n)
    # Three-echelon METRIC-style inventory: central depot, rear theater depot,
    # and the forward stock consumed by repair shops. Pipelines impose lead time.
    allied_spare_central = np.clip(
        rng.normal(0.86 * case.resupply, 0.08, n), 0.0, 1.25
    )
    allied_spare_rear = np.clip(
        rng.normal(0.24 * case.resupply, 0.04, n), 0.0, 0.48
    )
    allied_spare_pipeline_c2r = np.zeros((6, n))
    allied_spare_pipeline_r2f = np.zeros((3, n))
    allied_precision_arrived = np.zeros(n)
    allied_general_arrived = np.zeros(n)
    allied_operational_fuel_arrived = np.zeros(n)
    allied_spare_arrived = np.zeros(n)
    allied_air_destroyed = np.zeros(n)
    allied_sea_destroyed = np.zeros(n)
    allied_enabler_destroyed = np.zeros(n)
    allied_precision_shortage_week = np.full(n, -1, dtype=np.int16)
    allied_spare_shortage_week = np.full(n, -1, dtype=np.int16)
    total_allied_cargo_capacity = np.zeros(n)
    total_allied_cargo_used = np.zeros(n)
    max_allied_air_repair_queue = np.zeros(n)
    max_allied_sea_repair_queue = np.zeros(n)
    max_allied_air_turnaround_queue = np.zeros(n)
    max_allied_sea_service_queue = np.zeros(n)
    total_allied_air_demand = np.zeros(n)
    total_allied_air_served = np.zeros(n)
    total_allied_sea_demand = np.zeros(n)
    total_allied_sea_served = np.zeros(n)
    total_search_effort = np.zeros((n, 3))
    total_search_detection = np.zeros((n, 3))
    industrial_allocation = np.zeros((n, 4))
    industrial_budget_spent = np.zeros(n)
    allied_industrial_capacity = np.clip(
        rng.normal(0.58 + 0.32 * case.resupply, 0.07, n), 0.28, 1.10
    )

    attacker_air_repair_queue = np.zeros(n)
    attacker_sea_repair_queue = np.zeros(n)
    defender_air_repair_queue = np.zeros(n)
    defender_sea_repair_queue = np.zeros(n)
    attacker_air_destroyed = np.zeros(n)
    attacker_sea_destroyed = np.zeros(n)
    defender_air_destroyed = np.zeros(n)
    defender_sea_destroyed = np.zeros(n)
    attacker_spares = np.clip(rng.normal(0.72 * structure["attacker_stock"], 0.08, n), 0.28, 1.15)
    defender_spares = np.clip(rng.normal(0.54 * structure["defender_stock"], 0.08, n), 0.18, 1.05)

    min_allied_forward_base = allied_forward_base.copy()
    min_allied_rear_base = allied_rear_base.copy()
    min_allied_port = allied_port.copy()
    min_allied_fuel = allied_fuel.copy()
    min_allied_c4isr = allied_c4isr.copy()
    min_allied_logistics_flow = np.ones(n)
    min_attacker_logistics_flow = np.ones(n)
    allied_forward_disabled_week = np.full(n, -1, dtype=np.int16)
    allied_rear_disabled_week = np.full(n, -1, dtype=np.int16)
    allied_port_disabled_week = np.full(n, -1, dtype=np.int16)
    allied_network_mode = facility_mode(allied_forward_base)
    total_long_range_salvo = np.zeros(n)
    total_counterstrike = np.zeros(n)
    interdiction_effort = np.zeros((n, 5))
    min_allied_spare_fill_rate = np.ones(n)
    ally_mode = np.full(n, ALLY_NOT_ARRIVED, dtype=np.int8)
    defender_port_mode = facility_mode(defender_port)
    defender_airfield_mode = facility_mode(defender_airfield)
    captured_port_mode = np.full(n, PORT_UNAVAILABLE, dtype=np.int8)
    captured_port_ready_week = np.full(n, -1, dtype=np.int16)
    ally_arrival_event_week = np.full(n, -1, dtype=np.int16)
    ally_deployed_event_week = np.full(n, -1, dtype=np.int16)
    port_disabled_event_week = np.full(n, -1, dtype=np.int16)
    airfield_disabled_event_week = np.full(n, -1, dtype=np.int16)
    port_conversion_event_week = np.full(n, -1, dtype=np.int16)
    port_operational_event_week = np.full(n, -1, dtype=np.int16)
    expansion_event_week = np.full(n, -1, dtype=np.int16)
    event_transition_count = np.zeros(n, dtype=np.int16)
    horizon_snapshots = {}

    for week in range(1, WEEKS + 1):
        active = outcome < 0
        if not active.any():
            break
        idx = np.where(active)[0]

        weather_memory[idx] = np.clip(
            0.72 * weather_memory[idx] + 0.28 * rng.beta(5.0, 3.0, len(idx)), 0.08, 0.98
        )
        weather = weather_memory[idx]
        arrived = week >= np.ceil(delay[idx])
        newly_arrived = arrived & (ally_mode[idx] == ALLY_NOT_ARRIVED)
        ally_mode[idx[newly_arrived]] = ALLY_RAMPING
        ally_arrival_event_week[idx[newly_arrived]] = week
        event_transition_count[idx[newly_arrived]] += 1
        fully_deployed = arrived & (week >= np.ceil(delay[idx]) + 4)
        newly_deployed = fully_deployed & (ally_mode[idx] == ALLY_RAMPING)
        ally_mode[idx[newly_deployed]] = ALLY_DEPLOYED
        ally_deployed_event_week[idx[newly_deployed]] = week
        event_transition_count[idx[newly_deployed]] += 1
        ramp = np.where(arrived, np.minimum((week - np.ceil(delay[idx]) + 1.0) / 5.0, 1.0), 0.0)

        # Long-range fires are allocated between Taiwan targets and the allied
        # base network. More extensive intervention attracts more fire, but
        # also diverts launchers, ISR and weapons from the Taiwan strike plan.
        intervention_signal = case.air_strength + case.sea_strength + case.resupply
        if case_key == "taiwan_alone":
            allied_target_share = np.zeros(len(idx))
        else:
            preemption = np.where(week <= 12, 0.20, 0.06)
            allied_target_share = np.clip(
                0.08 + 0.16 * intervention_signal + preemption + 0.16 * ramp,
                0.0, 0.58,
            )

        # Weekly production/reconstitution is slow compared with opening
        # salvos but matters over the 15-year horizon.
        attacker_long_range_missiles[idx] = np.clip(
            attacker_long_range_missiles[idx]
            + 0.0022 * capability_ratio * attacker_launcher_survival[idx],
            0.0, 1.55,
        )
        attacker_theater_missiles[idx] = np.clip(
            attacker_theater_missiles[idx]
            + 0.0030 * capability_ratio * attacker_launcher_survival[idx],
            0.0, 1.75,
        )
        allied_bmd[idx] = np.clip(
            allied_bmd[idx] + 0.0018 * case.resupply * ramp * allied_port[idx],
            0.0, 1.25,
        )

        opening_salvo = 0.052 if week <= 16 else 0.014
        requested_salvo = (
            opening_salvo * allied_target_share * attacker_isr[idx]
            * attacker_launcher_survival[idx]
        )
        long_range_salvo = np.minimum(attacker_long_range_missiles[idx], requested_salvo)
        saturation = long_range_salvo / np.maximum(0.012 + 0.050 * allied_bmd[idx], 1e-5)
        penetration = sigmoid(
            -0.55 + 1.65 * saturation - 0.75 * allied_dispersion[idx]
        )
        strike_damage = (
            6.20 * np.exp(1.4 * structure["combat_shift"])
            * long_range_salvo * penetration * attacker_isr[idx]
            * rng.lognormal(0.0, 0.22, len(idx))
        )
        total_long_range_salvo[idx] += long_range_salvo
        attacker_long_range_missiles[idx] = np.clip(
            attacker_long_range_missiles[idx] - long_range_salvo, 0.0, 1.55
        )
        allied_bmd[idx] = np.clip(
            allied_bmd[idx] - 0.62 * long_range_salvo * (1.0 - penetration), 0.0, 1.25
        )

        # One-step network interdiction: estimate each component's marginal
        # effect on maximum logistics flow, then use a soft allocation so that
        # uncertainty and target saturation prevent a deterministic solution.
        _, _, baseline_network_flow = allied_logistics_max_flow(
            allied_forward_base[idx], allied_rear_base[idx], allied_port[idx],
            allied_fuel[idx], allied_c4isr[idx], allied_dispersion[idx],
        )
        components = [
            allied_forward_base[idx], allied_rear_base[idx], allied_port[idx],
            allied_fuel[idx], allied_c4isr[idx],
        ]
        marginal_flow_loss = []
        perturbation = 0.03
        for component_index in range(5):
            perturbed = [value.copy() for value in components]
            perturbed[component_index] = np.clip(
                perturbed[component_index] - perturbation, 0.0, 1.0
            )
            _, _, perturbed_flow = allied_logistics_max_flow(
                perturbed[0], perturbed[1], perturbed[2], perturbed[3],
                perturbed[4], allied_dispersion[idx],
            )
            marginal_flow_loss.append(
                np.maximum(0.0, baseline_network_flow - perturbed_flow) / perturbation
            )
        marginal_flow_loss = np.column_stack(marginal_flow_loss)
        direct_operational_value = np.column_stack((
            np.full(len(idx), 0.24 + 0.14 * case.air_strength),
            np.full(len(idx), 0.12),
            np.full(len(idx), 0.20 + 0.14 * case.sea_strength),
            np.full(len(idx), 0.24 + 0.08 * (case.air_strength + case.sea_strength)),
            np.full(len(idx), 0.20),
        ))
        target_scores = (
            0.68 * marginal_flow_loss + 0.32 * direct_operational_value
        ) * np.column_stack(components)
        target_scores -= target_scores.max(axis=1, keepdims=True)
        target_weights = np.exp(3.2 * target_scores)
        target_weights /= target_weights.sum(axis=1, keepdims=True)
        interdiction_effort[idx, :] += target_weights * long_range_salvo[:, None]
        allied_forward_base[idx] = np.clip(
            allied_forward_base[idx] - strike_damage * target_weights[:, 0]
            + 0.020 * allied_repair_capacity[idx] * (1.0 - allied_forward_base[idx]), 0.0, 1.0
        )
        allied_rear_base[idx] = np.clip(
            allied_rear_base[idx] - strike_damage * target_weights[:, 1]
            + 0.013 * allied_repair_capacity[idx] * (1.0 - allied_rear_base[idx]), 0.0, 1.0
        )
        allied_port[idx] = np.clip(
            allied_port[idx] - strike_damage * target_weights[:, 2]
            + 0.015 * allied_repair_capacity[idx] * (1.0 - allied_port[idx]), 0.0, 1.0
        )
        allied_fuel[idx] = np.clip(
            allied_fuel[idx] - strike_damage * target_weights[:, 3]
            + 0.012 * allied_repair_capacity[idx] * (1.0 - allied_fuel[idx]), 0.0, 1.0
        )
        allied_c4isr[idx] = np.clip(
            allied_c4isr[idx] - strike_damage * target_weights[:, 4]
            + 0.017 * allied_repair_capacity[idx] * (1.0 - allied_c4isr[idx]), 0.0, 1.0
        )

        # Aircraft caught on the ground, ships in port and scarce enabling
        # platforms are not represented by base availability alone.
        ground_air_loss = np.minimum(
            allied_air_ready[idx],
            0.070 * strike_damage * (1.0 - 0.72 * allied_dispersion[idx]),
        )
        port_sea_loss = np.minimum(
            allied_sea_ready[idx],
            0.040 * strike_damage * (1.0 - 0.55 * allied_dispersion[idx]),
        )
        enabler_loss = np.minimum(
            allied_enablers[idx],
            0.052 * strike_damage * (1.0 - 0.64 * allied_dispersion[idx]),
        )
        allied_air_ready[idx] -= ground_air_loss
        allied_sea_ready[idx] -= port_sea_loss
        allied_enablers[idx] -= enabler_loss
        allied_air_repair_queue[idx] += 0.52 * ground_air_loss
        allied_sea_repair_queue[idx] += 0.58 * port_sea_loss
        allied_air_destroyed[idx] += 0.48 * ground_air_loss
        allied_sea_destroyed[idx] += 0.42 * port_sea_loss
        allied_enabler_destroyed[idx] += enabler_loss

        new_allied_mode = facility_mode(allied_forward_base[idx])
        allied_mode_changed = new_allied_mode != allied_network_mode[idx]
        event_transition_count[idx[allied_mode_changed]] += 1
        newly_forward_disabled = (
            (new_allied_mode == FACILITY_DISABLED)
            & (allied_forward_disabled_week[idx] < 0)
        )
        allied_forward_disabled_week[idx[newly_forward_disabled]] = week
        allied_network_mode[idx] = new_allied_mode
        newly_rear_disabled = (
            (allied_rear_base[idx] < 0.20) & (allied_rear_disabled_week[idx] < 0)
        )
        newly_allied_port_disabled = (
            (allied_port[idx] < 0.20) & (allied_port_disabled_week[idx] < 0)
        )
        allied_rear_disabled_week[idx[newly_rear_disabled]] = week
        allied_port_disabled_week[idx[newly_allied_port_disabled]] = week

        # ACE dispersion preserves a floor, while runway, fuel, ports and
        # C4ISR form a multiplicative bottleneck rather than perfect substitutes.
        air_network = np.clip(
            np.power(np.maximum(allied_dispersion[idx], 1e-4), 0.10)
            * np.power(np.maximum(allied_forward_base[idx], 1e-4), 0.35)
            * np.power(np.maximum(allied_rear_base[idx], 1e-4), 0.10)
            * np.power(np.maximum(allied_fuel[idx], 1e-4), 0.25)
            * np.power(np.maximum(allied_c4isr[idx], 1e-4), 0.20) / 0.89,
            0.12 * allied_dispersion[idx], 1.08,
        )
        sea_network = np.clip(
            np.power(np.maximum(allied_dispersion[idx], 1e-4), 0.08)
            * np.power(np.maximum(allied_forward_base[idx], 1e-4), 0.18)
            * np.power(np.maximum(allied_rear_base[idx], 1e-4), 0.12)
            * np.power(np.maximum(allied_port[idx], 1e-4), 0.34)
            * np.power(np.maximum(allied_fuel[idx], 1e-4), 0.28) / 0.90,
            0.10 * allied_dispersion[idx], 1.08,
        )
        resupply_network = np.clip(
            np.power(np.maximum(allied_port[idx], 1e-4), 0.34)
            * np.power(np.maximum(allied_rear_base[idx], 1e-4), 0.20)
            * np.power(np.maximum(allied_fuel[idx], 1e-4), 0.30)
            * np.power(np.maximum(allied_c4isr[idx], 1e-4), 0.16) / 0.93,
            0.08, 1.08,
        )
        air_route_flow, sea_route_flow, logistics_max_flow = allied_logistics_max_flow(
            allied_forward_base[idx],
            allied_rear_base[idx],
            allied_port[idx],
            allied_fuel[idx],
            allied_c4isr[idx],
            allied_dispersion[idx],
        )
        min_allied_logistics_flow[idx] = np.minimum(
            min_allied_logistics_flow[idx], logistics_max_flow
        )
        # Reinforcement pipeline: the initial force arrives according to the
        # deployment ramp; replacements are much slower and remain finite.
        planned_air = allied_air_target[idx] * ramp
        planned_sea = allied_sea_target[idx] * ramp
        planned_enablers = allied_enabler_target[idx] * ramp
        air_arrival = (
            np.maximum(0.0, planned_air - allied_air_arrived[idx]) * air_route_flow
        )
        sea_arrival = (
            np.maximum(0.0, planned_sea - allied_sea_arrived[idx]) * sea_route_flow
        )
        enabler_arrival = (
            np.maximum(0.0, planned_enablers - allied_enablers_arrived[idx])
            * air_route_flow
        )
        replacement_gate = fully_deployed.astype(float)
        air_arrival += (
            0.0012 * allied_air_target[idx] * case.resupply * air_route_flow
            * replacement_gate
        )
        sea_arrival += (
            0.00034 * allied_sea_target[idx] * case.resupply * sea_route_flow
            * replacement_gate
        )
        enabler_arrival += (
            0.00055 * allied_enabler_target[idx] * case.resupply * air_route_flow
            * replacement_gate
        )
        allied_air_arrived[idx] += air_arrival
        allied_sea_arrived[idx] += sea_arrival
        allied_enablers_arrived[idx] += enabler_arrival
        allied_air_ready[idx] += air_arrival
        allied_sea_ready[idx] += sea_arrival
        allied_enablers[idx] += enabler_arrival

        # Military-economics layer: a finite weekly mobilization budget is
        # allocated by marginal readiness return. The four sectors are air and
        # missile defense, spare production, repair plant, and transport.
        scarcity = np.column_stack((
            np.maximum(0.0, 0.82 - allied_bmd[idx]),
            np.maximum(
                0.0,
                0.62 * case.resupply
                - allied_spare_central[idx] - allied_spare_rear[idx]
                - allied_spare_stock[idx],
            ),
            allied_air_repair_queue[idx] + 1.6 * allied_sea_repair_queue[idx],
            np.maximum(0.0, 0.76 - logistics_max_flow),
        ))
        sector_productivity = np.array([0.92, 1.08, 0.86, 0.78])
        sector_scores = 2.6 * scarcity * sector_productivity
        sector_scores -= sector_scores.max(axis=1, keepdims=True)
        sector_shares = np.exp(sector_scores)
        sector_shares /= sector_shares.sum(axis=1, keepdims=True)
        mobilization_ramp = 1.0 - np.exp(-week / 52.0)
        weekly_industrial_budget = (
            0.0065 * allied_industrial_capacity[idx] * case.resupply
            * mobilization_ramp * (0.72 + 0.28 * allied_rear_base[idx])
        )
        sector_output = weekly_industrial_budget[:, None] * sector_shares
        industrial_allocation[idx, :] += sector_output
        industrial_budget_spent[idx] += weekly_industrial_budget
        allied_bmd[idx] = np.clip(allied_bmd[idx] + 0.28 * sector_output[:, 0], 0.0, 1.25)
        allied_spare_central[idx] = np.clip(
            allied_spare_central[idx] + 1.15 * sector_output[:, 1], 0.0, 1.45
        )
        allied_repair_capacity[idx] = np.clip(
            allied_repair_capacity[idx] + 0.035 * sector_output[:, 2], 0.0, 1.15
        )
        transport_investment = 0.55 * sector_output[:, 3]

        # Receive outstanding echelon orders before calculating new orders.
        c2r_slot = week % allied_spare_pipeline_c2r.shape[0]
        r2f_slot = week % allied_spare_pipeline_r2f.shape[0]
        allied_spare_rear[idx] += allied_spare_pipeline_c2r[c2r_slot, idx]
        allied_spare_stock[idx] += allied_spare_pipeline_r2f[r2f_slot, idx]
        allied_spare_pipeline_c2r[c2r_slot, idx] = 0.0
        allied_spare_pipeline_r2f[r2f_slot, idx] = 0.0
        rear_inventory_position = (
            allied_spare_rear[idx]
            + allied_spare_pipeline_c2r[:, idx].sum(axis=0)
        )
        rear_base_stock = 0.52 * case.resupply
        central_to_rear_order = np.minimum(
            allied_spare_central[idx],
            np.maximum(0.0, rear_base_stock - rear_inventory_position),
        )
        central_to_rear_order = np.minimum(
            central_to_rear_order,
            0.018 * case.resupply * (0.45 * allied_rear_base[idx] + 0.55 * allied_port[idx]),
        )
        allied_spare_central[idx] -= central_to_rear_order
        allied_spare_pipeline_c2r[(week + 5) % 6, idx] += central_to_rear_order

        # Precision weapons, general stores, fuel and forward-bound spares
        # compete for one multi-commodity transport capacity.
        # multi-commodity transport capacity; no commodity receives the full
        # max-flow capacity independently.
        precision_target = 0.78 * case.resupply * ramp
        general_target = 0.92 * case.resupply * ramp
        fuel_target = 0.88 * case.resupply * ramp
        spare_target = 0.42 * case.resupply * ramp
        precision_request = np.maximum(
            0.0, precision_target - allied_precision_arrived[idx]
        ) + 0.0065 * case.resupply * replacement_gate
        general_request = np.maximum(
            0.0, general_target - allied_general_arrived[idx]
        ) + 0.0080 * case.resupply * replacement_gate
        fuel_request = np.maximum(
            0.0, fuel_target - allied_operational_fuel_arrived[idx]
        ) + 0.0100 * case.resupply * replacement_gate
        forward_inventory_position = (
            allied_spare_stock[idx]
            + allied_spare_pipeline_r2f[:, idx].sum(axis=0)
        )
        spare_request = np.minimum(
            allied_spare_rear[idx],
            np.maximum(0.0, spare_target - forward_inventory_position)
            + 0.0045 * case.resupply * replacement_gate,
        )
        cargo_requests = np.column_stack((
            precision_request, general_request, fuel_request, spare_request,
        ))
        deployment_surge = np.where(arrived & (~fully_deployed), 0.78, 0.045)
        cargo_capacity = (
            case.resupply * logistics_max_flow * deployment_surge
            + transport_investment
        )
        priority_weight = np.array([1.28, 0.86, 1.15, 0.78])
        weighted_requests = cargo_requests * priority_weight
        first_allocation = np.minimum(
            cargo_requests,
            cargo_capacity[:, None] * weighted_requests
            / np.maximum(weighted_requests.sum(axis=1, keepdims=True), 1e-9),
        )
        remaining_capacity = np.maximum(
            0.0, cargo_capacity - first_allocation.sum(axis=1)
        )
        unmet_request = np.maximum(0.0, cargo_requests - first_allocation)
        second_allocation = np.minimum(
            unmet_request,
            remaining_capacity[:, None] * unmet_request
            / np.maximum(unmet_request.sum(axis=1, keepdims=True), 1e-9),
        )
        cargo_allocation = first_allocation + second_allocation
        precision_arrival = cargo_allocation[:, 0]
        general_arrival = cargo_allocation[:, 1]
        fuel_arrival = cargo_allocation[:, 2]
        spare_arrival = cargo_allocation[:, 3]
        total_allied_cargo_capacity[idx] += cargo_capacity
        total_allied_cargo_used[idx] += cargo_allocation.sum(axis=1)
        allied_precision_arrived[idx] += precision_arrival
        allied_general_arrived[idx] += general_arrival
        allied_operational_fuel_arrived[idx] += fuel_arrival
        allied_spare_rear[idx] = np.maximum(0.0, allied_spare_rear[idx] - spare_arrival)
        allied_spare_pipeline_r2f[(week + 2) % 3, idx] += spare_arrival
        allied_spare_arrived[idx] += spare_arrival
        allied_precision_stock[idx] = np.clip(
            allied_precision_stock[idx] + precision_arrival, 0.0, 1.35
        )
        allied_general_stock[idx] = np.clip(
            allied_general_stock[idx] + general_arrival, 0.0, 1.55
        )
        allied_operational_fuel[idx] = np.clip(
            allied_operational_fuel[idx] + fuel_arrival, 0.0, 1.55
        )
        allied_spare_stock[idx] = np.clip(allied_spare_stock[idx], 0.0, 1.10)

        # Finite-server repair queues.  Completion cannot exceed the damaged
        # backlog or the weekly capacity of depots and ports.
        pending_repair_demand = (
            allied_air_repair_queue[idx] + 1.6 * allied_sea_repair_queue[idx]
        )
        spare_fill_rate = np.minimum(
            1.0,
            allied_spare_stock[idx] / np.maximum(0.018 + 2.8 * pending_repair_demand, 0.018),
        )
        min_allied_spare_fill_rate[idx] = np.minimum(
            min_allied_spare_fill_rate[idx],
            np.where(pending_repair_demand > 0.005, spare_fill_rate, 1.0),
        )
        air_repair_capacity = (
            0.0105 * allied_repair_capacity[idx] * air_network * air_route_flow
            * (0.45 + 0.55 * allied_enablers[idx] / np.maximum(allied_enabler_target[idx], 0.05))
            * spare_fill_rate
        )
        sea_repair_capacity = (
            0.0040 * allied_repair_capacity[idx] * sea_network * sea_route_flow
            * spare_fill_rate
        )
        air_repaired = np.minimum(allied_air_repair_queue[idx], air_repair_capacity)
        sea_repaired = np.minimum(allied_sea_repair_queue[idx], sea_repair_capacity)
        allied_air_repair_queue[idx] -= air_repaired
        allied_sea_repair_queue[idx] -= sea_repaired
        allied_air_ready[idx] += air_repaired
        allied_sea_ready[idx] += sea_repaired
        allied_spare_stock[idx] = np.clip(
            allied_spare_stock[idx] - 0.55 * air_repaired - 1.10 * sea_repaired,
            0.0,
            1.10,
        )

        # Fluid queue approximation for sortie turnaround and naval servicing.
        enabler_factor = np.minimum(
            1.0,
            allied_enablers[idx] / np.maximum(0.72 * allied_enabler_target[idx], 0.04),
        )
        precision_factor = np.minimum(
            1.0,
            allied_precision_stock[idx] / max(0.16 * case.resupply, 0.03),
        )
        general_factor = np.minimum(
            1.0,
            allied_general_stock[idx] / max(0.18 * case.resupply, 0.03),
        )
        operational_fuel_factor = np.minimum(
            1.0,
            allied_operational_fuel[idx] / max(0.20 * case.resupply, 0.03),
        )
        air_demand = 0.155 * allied_air_ready[idx]
        sea_demand = 0.082 * allied_sea_ready[idx]
        air_service_capacity = (
            0.145 * np.maximum(allied_air_target[idx], 0.02) * air_network
            * enabler_factor * precision_factor * operational_fuel_factor
        )
        sea_service_capacity = (
            0.075 * np.maximum(allied_sea_target[idx], 0.02) * sea_network
            * general_factor * operational_fuel_factor
        )
        air_work = allied_air_turnaround_queue[idx] + air_demand
        sea_work = allied_sea_service_queue[idx] + sea_demand
        air_served = np.minimum(air_work, air_service_capacity)
        sea_served = np.minimum(sea_work, sea_service_capacity)
        total_allied_air_demand[idx] += air_demand
        total_allied_air_served[idx] += np.minimum(air_served, air_demand)
        total_allied_sea_demand[idx] += sea_demand
        total_allied_sea_served[idx] += np.minimum(sea_served, sea_demand)
        allied_air_turnaround_queue[idx] = 0.86 * np.maximum(0.0, air_work - air_served)
        allied_sea_service_queue[idx] = 0.91 * np.maximum(0.0, sea_work - sea_served)
        air_queue_penalty = 1.0 / (
            1.0 + allied_air_turnaround_queue[idx] / np.maximum(air_service_capacity, 0.01)
        )
        sea_queue_penalty = 1.0 / (
            1.0 + allied_sea_service_queue[idx] / np.maximum(sea_service_capacity, 0.01)
        )
        air_service_ratio = np.clip(
            air_served / np.maximum(air_demand, 0.01), 0.0, 1.0
        ) * air_queue_penalty
        sea_service_ratio = np.clip(
            sea_served / np.maximum(sea_demand, 0.01), 0.0, 1.0
        ) * sea_queue_penalty

        ally_air = allied_air_ready[idx] * air_service_ratio
        ally_sea = allied_sea_ready[idx] * sea_service_ratio
        ally_resupply = (
            case.resupply * ramp * logistics_max_flow
            * np.minimum(1.0, allied_general_stock[idx] / 0.22)
            * (0.72 + 0.28 * corridor_mobility[idx])
        )
        allied_precision_stock[idx] = np.clip(
            allied_precision_stock[idx] - 0.017 * ally_air, 0.0, 1.35
        )
        allied_general_stock[idx] = np.clip(
            allied_general_stock[idx] - 0.010 * ally_sea - 0.005 * ally_resupply,
            0.0, 1.55,
        )
        allied_operational_fuel[idx] = np.clip(
            allied_operational_fuel[idx] - 0.013 * ally_air - 0.009 * ally_sea,
            0.0, 1.55,
        )
        allied_short = (
            arrived
            & (allied_precision_stock[idx] < max(0.10 * case.resupply, 0.025))
        )
        allied_new_short = allied_short & (allied_precision_shortage_week[idx] < 0)
        allied_precision_shortage_week[idx[allied_new_short]] = week
        allied_spare_short = arrived & (pending_repair_demand > 0.012) & (spare_fill_rate < 0.72)
        allied_new_spare_short = (
            allied_spare_short & (allied_spare_shortage_week[idx] < 0)
        )
        allied_spare_shortage_week[idx[allied_new_spare_short]] = week

        max_allied_air_repair_queue[idx] = np.maximum(
            max_allied_air_repair_queue[idx], allied_air_repair_queue[idx]
        )
        max_allied_sea_repair_queue[idx] = np.maximum(
            max_allied_sea_repair_queue[idx], allied_sea_repair_queue[idx]
        )
        max_allied_air_turnaround_queue[idx] = np.maximum(
            max_allied_air_turnaround_queue[idx], allied_air_turnaround_queue[idx]
        )
        max_allied_sea_service_queue[idx] = np.maximum(
            max_allied_sea_service_queue[idx], allied_sea_service_queue[idx]
        )

        # Search theory is separated from firepower. ISR effort is allocated
        # among mobile launchers, fleet units and lift assets by the KKT rule;
        # fire can only exploit the resulting detection probability.
        search_prior = np.column_stack((
            0.42 * attacker_launcher_survival[idx],
            0.33 * attacker_sea[idx],
            0.25 * amphibious_lift[idx],
        ))
        search_rate = np.column_stack((
            0.72 + 0.48 * allied_c4isr[idx],
            0.88 + 0.42 * allied_enablers[idx],
            0.80 + 0.36 * allied_enablers[idx],
        ))
        search_budget = (
            0.20 * ramp * (0.42 * ally_air + 0.38 * ally_sea + 0.20 * allied_enablers[idx])
            * allied_c4isr[idx]
        )
        flexible_effort, _ = optimal_search_allocation(
            search_prior, search_rate, 0.75 * search_budget
        )
        mission_floor = search_budget[:, None] * np.array([0.10, 0.08, 0.07])
        search_effort = flexible_effort + mission_floor
        search_detection = 1.0 - np.exp(-search_rate * search_effort)
        total_search_effort[idx, :] += search_effort
        total_search_detection[idx, :] += search_detection
        detected_value = (
            (search_prior * search_detection).sum(axis=1)
            / np.maximum(search_prior.sum(axis=1), 1e-9)
        )

        # Reciprocal strikes degrade launchers, sensors, missile stocks and
        # mainland base availability. Dispersed mobile launchers prevent a
        # deterministic disarming strike.
        counterstrike = (
            0.010 * ramp * (0.55 * ally_air + 0.45 * ally_sea)
            * allied_c4isr[idx] * (0.48 + 1.35 * detected_value)
            * rng.lognormal(0.0, 0.18, len(idx))
        )
        total_counterstrike[idx] += counterstrike
        attacker_launcher_survival[idx] = np.clip(
            attacker_launcher_survival[idx] - 0.22 * counterstrike
            + 0.0020 * structure["attacker_repair"] * (1.0 - attacker_launcher_survival[idx]),
            0.28, 1.0,
        )
        attacker_isr[idx] = np.clip(
            attacker_isr[idx] - 0.28 * counterstrike
            + 0.0024 * structure["attacker_repair"] * (1.0 - attacker_isr[idx]),
            0.24, 1.0,
        )
        attacker_long_range_missiles[idx] = np.clip(
            attacker_long_range_missiles[idx] - 0.18 * counterstrike, 0.0, 1.55
        )
        attacker_theater_missiles[idx] = np.clip(
            attacker_theater_missiles[idx] - 0.14 * counterstrike, 0.0, 1.75
        )
        attacker_sea[idx] = np.clip(
            attacker_sea[idx] - 0.026 * counterstrike * search_detection[:, 1],
            0.0,
            1.50,
        )
        amphibious_lift[idx] = np.clip(
            amphibious_lift[idx] - 0.030 * counterstrike * search_detection[:, 2],
            0.0,
            1.18,
        )

        theater_missile_factor = np.minimum(1.0, attacker_theater_missiles[idx] / 0.24)
        air_naval_precision_factor = np.minimum(1.0, attacker_precision[idx] / 0.22)
        weapon_mix = 0.65 * air_naval_precision_factor + 0.35 * theater_missile_factor
        kill_chain_factor = 0.72 + 0.28 * np.sqrt(
            attacker_isr[idx] * attacker_launcher_survival[idx]
        )
        a_precision_factor = weapon_mix * kill_chain_factor
        d_precision_factor = np.minimum(1.0, defender_precision[idx] / 0.18)
        a_fuel_factor = np.minimum(1.0, attacker_fuel[idx] / 0.24)
        d_fuel_factor = np.minimum(1.0, defender_fuel[idx] / 0.20)

        a_air_eff = attacker_air[idx] * attacker_base[idx] * a_precision_factor * a_fuel_factor
        d_air_eff = (
            defender_air[idx] * defender_airfield[idx] * d_precision_factor * d_fuel_factor
            + ally_air
        )
        air_control = sigmoid(
            2.35 * np.log(safe_ratio(a_air_eff + 0.03, d_air_eff + 0.03))
            + structure["combat_shift"]
        )

        a_sea_eff = attacker_sea[idx] * a_precision_factor * attacker_logistics[idx]
        d_sea_eff = defender_sea[idx] * d_precision_factor + ally_sea
        sea_control = sigmoid(
            2.15 * np.log(safe_ratio(a_sea_eff + 0.03, d_sea_eff + 0.03))
            + structure["combat_shift"]
        )

        taiwan_target_share = 1.0 - 0.20 * allied_target_share
        strike_intensity = np.minimum(
            1.0,
            (0.55 * a_precision_factor + 0.45 * air_control) * taiwan_target_share,
        )
        defender_port[idx] = np.clip(
            defender_port[idx]
            - 0.042 * strike_intensity * air_control * rng.lognormal(0.0, 0.16, len(idx))
            + 0.020 * structure["defender_repair"] * (1.0 - defender_port[idx])
            * (0.45 + 0.55 * command[idx]),
            0.0, 1.0,
        )
        defender_airfield[idx] = np.clip(
            defender_airfield[idx]
            - 0.052 * strike_intensity * air_control * rng.lognormal(0.0, 0.15, len(idx))
            + 0.026 * structure["defender_repair"] * (1.0 - defender_airfield[idx])
            * (0.42 + 0.58 * command[idx]),
            0.0, 1.0,
        )

        new_port_mode = facility_mode(defender_port[idx])
        new_airfield_mode = facility_mode(defender_airfield[idx])
        port_changed = new_port_mode != defender_port_mode[idx]
        airfield_changed = new_airfield_mode != defender_airfield_mode[idx]
        event_transition_count[idx[port_changed]] += 1
        event_transition_count[idx[airfield_changed]] += 1
        newly_port_disabled = (
            (new_port_mode == FACILITY_DISABLED)
            & (port_disabled_event_week[idx] < 0)
        )
        newly_airfield_disabled = (
            (new_airfield_mode == FACILITY_DISABLED)
            & (airfield_disabled_event_week[idx] < 0)
        )
        port_disabled_event_week[idx[newly_port_disabled]] = week
        airfield_disabled_event_week[idx[newly_airfield_disabled]] = week
        defender_port_mode[idx] = new_port_mode
        defender_airfield_mode[idx] = new_airfield_mode
        attacker_base[idx] = np.clip(
            attacker_base[idx]
            - 0.010 * d_precision_factor * (1.0 - air_control) * (0.25 + ally_air)
            + 0.018 * structure["attacker_repair"] * (1.0 - attacker_base[idx]),
            0.25, 1.0,
        )

        # Symmetric stock-flow attrition.  Gross losses leave the ready stock;
        # only the repairable share enters a finite-capacity repair queue.
        attacker_air_loss = np.minimum(
            attacker_air[idx],
            0.0090 * attacker_air[idx] * (1.0 - air_control)
            * (0.45 + d_precision_factor + 0.55 * ally_air),
        )
        defender_air_loss = np.minimum(
            defender_air[idx],
            0.0110 * defender_air[idx] * air_control * (0.60 + a_precision_factor),
        )
        allied_air_loss = np.minimum(
            allied_air_ready[idx],
            0.0100 * allied_air_ready[idx] * air_control
            * (0.50 + a_precision_factor) * (0.35 + 0.65 * air_service_ratio),
        )
        attacker_sea_loss = np.minimum(
            attacker_sea[idx],
            0.0075 * attacker_sea[idx] * (1.0 - sea_control)
            * (0.50 + d_precision_factor + 0.65 * ally_sea),
        )
        defender_sea_loss = np.minimum(
            defender_sea[idx],
            0.0090 * defender_sea[idx] * sea_control * (0.55 + a_precision_factor),
        )
        allied_sea_loss = np.minimum(
            allied_sea_ready[idx],
            0.0082 * allied_sea_ready[idx] * sea_control
            * (0.52 + a_precision_factor) * (0.40 + 0.60 * sea_service_ratio),
        )

        attacker_air[idx] -= attacker_air_loss
        defender_air[idx] -= defender_air_loss
        allied_air_ready[idx] -= allied_air_loss
        attacker_sea[idx] -= attacker_sea_loss
        defender_sea[idx] -= defender_sea_loss
        allied_sea_ready[idx] -= allied_sea_loss

        attacker_air_repair_queue[idx] += 0.40 * attacker_air_loss
        defender_air_repair_queue[idx] += 0.43 * defender_air_loss
        allied_air_repair_queue[idx] += 0.42 * allied_air_loss
        attacker_sea_repair_queue[idx] += 0.48 * attacker_sea_loss
        defender_sea_repair_queue[idx] += 0.52 * defender_sea_loss
        allied_sea_repair_queue[idx] += 0.50 * allied_sea_loss
        attacker_air_destroyed[idx] += 0.60 * attacker_air_loss
        defender_air_destroyed[idx] += 0.57 * defender_air_loss
        allied_air_destroyed[idx] += 0.58 * allied_air_loss
        attacker_sea_destroyed[idx] += 0.52 * attacker_sea_loss
        defender_sea_destroyed[idx] += 0.48 * defender_sea_loss
        allied_sea_destroyed[idx] += 0.50 * allied_sea_loss

        attacker_spares[idx] = np.clip(
            attacker_spares[idx] + 0.0034 * capability_ratio, 0.0, 1.25
        )
        defender_spares[idx] = np.clip(
            defender_spares[idx] + 0.0015 + 0.0032 * ally_resupply,
            0.0,
            1.15,
        )
        attacker_repair_demand = (
            attacker_air_repair_queue[idx] + 1.5 * attacker_sea_repair_queue[idx]
        )
        defender_repair_demand = (
            defender_air_repair_queue[idx] + 1.5 * defender_sea_repair_queue[idx]
        )
        attacker_spare_fill = np.minimum(
            1.0,
            attacker_spares[idx] / np.maximum(0.018 + 2.6 * attacker_repair_demand, 0.018),
        )
        defender_spare_fill = np.minimum(
            1.0,
            defender_spares[idx] / np.maximum(0.018 + 2.6 * defender_repair_demand, 0.018),
        )
        attacker_air_repaired = np.minimum(
            attacker_air_repair_queue[idx],
            0.0022 * structure["attacker_repair"] * attacker_base[idx] * attacker_spare_fill,
        )
        defender_air_repaired = np.minimum(
            defender_air_repair_queue[idx],
            0.0017 * structure["defender_repair"] * defender_airfield[idx] * defender_spare_fill,
        )
        attacker_sea_repaired = np.minimum(
            attacker_sea_repair_queue[idx],
            0.00125 * structure["attacker_repair"] * attacker_base[idx] * attacker_spare_fill,
        )
        defender_sea_repaired = np.minimum(
            defender_sea_repair_queue[idx],
            0.00085 * structure["defender_repair"] * defender_port[idx] * defender_spare_fill,
        )
        attacker_air_repair_queue[idx] -= attacker_air_repaired
        defender_air_repair_queue[idx] -= defender_air_repaired
        attacker_sea_repair_queue[idx] -= attacker_sea_repaired
        defender_sea_repair_queue[idx] -= defender_sea_repaired
        attacker_spares[idx] = np.clip(
            attacker_spares[idx] - 0.55 * attacker_air_repaired - 1.05 * attacker_sea_repaired,
            0.0,
            1.25,
        )
        defender_spares[idx] = np.clip(
            defender_spares[idx] - 0.55 * defender_air_repaired - 1.05 * defender_sea_repaired,
            0.0,
            1.15,
        )

        attacker_air[idx] = np.clip(
            attacker_air[idx] + attacker_air_repaired + 0.00055 * capability_ratio,
            0.0, 1.70,
        )
        defender_air[idx] = np.clip(
            defender_air[idx] + defender_air_repaired + 0.00020 + 0.00038 * ally_resupply,
            0.0, 1.0,
        )
        attacker_sea[idx] = np.clip(
            attacker_sea[idx] + attacker_sea_repaired + 0.00024 * capability_ratio,
            0.0, 1.65,
        )
        defender_sea[idx] = np.clip(
            defender_sea[idx] + defender_sea_repaired + 0.00010 + 0.00016 * ally_resupply,
            0.0, 0.90,
        )

        # Tankers, AEW and other enabling aircraft have their own exposure and
        # cannot be recreated by ordinary fighter replacement flows.
        enabler_combat_loss = np.minimum(
            allied_enablers[idx],
            0.0035 * allied_enablers[idx] * air_control
            * (0.35 + 0.65 * a_precision_factor),
        )
        allied_enablers[idx] -= enabler_combat_loss
        allied_enabler_destroyed[idx] += enabler_combat_loss

        attacker_precision[idx] = np.clip(
            attacker_precision[idx] - 0.026 * (0.55 + air_control) + 0.015 * capability_ratio,
            0.0, 1.35,
        )
        attacker_theater_missiles[idx] = np.clip(
            attacker_theater_missiles[idx]
            - 0.020 * taiwan_target_share * (0.55 + air_control),
            0.0, 1.75,
        )
        attacker_general[idx] = np.clip(
            attacker_general[idx] - 0.016 * (0.55 + beachhead[idx]) + 0.013,
            0.0, 1.35,
        )
        defender_precision[idx] = np.clip(
            defender_precision[idx] - 0.022 * (0.55 + 1.0 - air_control)
            + 0.018 * ally_resupply * defender_port[idx],
            0.0, 1.20,
        )
        defender_general[idx] = np.clip(
            defender_general[idx] - 0.014 * (0.60 + defender_org[idx])
            + 0.015 * ally_resupply * defender_port[idx],
            0.0, 1.25,
        )
        attacker_fuel[idx] = np.clip(
            attacker_fuel[idx] - 0.020 * (0.55 + air_control + sea_control) + 0.022 * attacker_logistics[idx],
            0.0, 1.20,
        )
        defender_fuel[idx] = np.clip(
            defender_fuel[idx] - 0.018 * (0.65 + 1.0 - air_control)
            + 0.020 * ally_resupply * defender_port[idx],
            0.0, 1.15,
        )

        a_short = (attacker_precision[idx] < 0.22) | (attacker_general[idx] < 0.20)
        d_short = (defender_precision[idx] < 0.18) | (defender_general[idx] < 0.18)
        a_new = a_short & (attacker_ammo_shortage_week[idx] < 0)
        d_new = d_short & (defender_ammo_shortage_week[idx] < 0)
        attacker_ammo_shortage_week[idx[a_new]] = week
        defender_ammo_shortage_week[idx[d_new]] = week

        lift_loss = 0.020 * amphibious_lift[idx] * (1.0 - sea_control) * (
            0.55 + d_precision_factor + 0.80 * ally_sea
        )
        amphibious_lift[idx] = np.clip(amphibious_lift[idx] - lift_loss, 0.0, 1.15)

        assault_window = week <= 20
        beach_throughput = (
            amphibious_lift[idx] * sea_control * weather * attacker_logistics[idx]
            * np.minimum(1.0, attacker_general[idx] / 0.22)
        )
        opposition = defender_org[idx] * command[idx] * (0.40 + 0.60 * (1.0 - air_control))
        beach_delta = np.where(
            assault_window,
            0.150 * beach_throughput * (1.0 - beachhead[idx]) - 0.042 * opposition * beachhead[idx],
            0.032 * captured_port[idx] * attacker_logistics[idx] * (1.0 - beachhead[idx])
            - 0.028 * opposition * beachhead[idx],
        )
        beachhead[idx] = np.clip(beachhead[idx] + beach_delta, 0.0, 1.0)
        delivered_mass[idx] += beach_throughput

        port_capture_prob = sigmoid(
            -4.1 + 5.0 * beachhead[idx] + 2.0 * control[idx]
            + 1.0 * air_control + 0.8 * sea_control - 1.4 * defender_org[idx]
        )
        captured_port[idx] = np.clip(
            captured_port[idx]
            + 0.078 * port_capture_prob * (1.0 - captured_port[idx])
            - 0.014 * (1.0 - sea_control) * captured_port[idx],
            0.0, 1.0,
        )

        conversion_start = (
            (captured_port_mode[idx] == PORT_UNAVAILABLE)
            & (captured_port[idx] >= 0.35)
        )
        conversion_idx = idx[conversion_start]
        if len(conversion_idx):
            conversion_delay = np.ceil(
                np.clip(rng.lognormal(np.log(8.0), 0.42, len(conversion_idx)), 3.0, 26.0)
            ).astype(np.int16)
            captured_port_mode[conversion_idx] = PORT_CONVERTING
            captured_port_ready_week[conversion_idx] = week + conversion_delay
            port_conversion_event_week[conversion_idx] = week
            event_transition_count[conversion_idx] += 1

        conversion_ready = (
            (captured_port_mode[idx] == PORT_CONVERTING)
            & (week >= captured_port_ready_week[idx])
            & (captured_port[idx] >= 0.25)
        )
        ready_idx = idx[conversion_ready]
        captured_port_mode[ready_idx] = PORT_OPERATIONAL
        port_operational_event_week[ready_idx] = week
        event_transition_count[ready_idx] += 1

        port_lost = (
            (captured_port_mode[idx] == PORT_OPERATIONAL)
            & (captured_port[idx] < 0.12)
        )
        lost_idx = idx[port_lost]
        captured_port_mode[lost_idx] = PORT_UNAVAILABLE
        captured_port_ready_week[lost_idx] = -1
        event_transition_count[lost_idx] += 1

        converted_port = captured_port[idx] * np.where(
            captured_port_mode[idx] == PORT_OPERATIONAL,
            1.0,
            np.where(captured_port_mode[idx] == PORT_CONVERTING, 0.55, 0.25),
        )

        attacker_source_capacity = np.minimum(
            attacker_base[idx],
            0.58 * np.minimum(1.0, attacker_fuel[idx] / 0.25)
            + 0.42 * np.minimum(1.0, attacker_general[idx] / 0.24),
        )
        attacker_cross_strait_capacity = np.minimum(
            0.32 + 0.68 * sea_control,
            0.38 + 0.62 * np.minimum(1.0, amphibious_lift[idx] / 0.34),
        )
        corridor_service = np.clip(
            corridor_mobility[idx]
            * (0.68 + 0.32 * converted_port)
            * (0.78 + 0.22 * attacker_logistics[idx]),
            0.10,
            0.96,
        )
        attacker_sink_capacity = np.clip(
            (0.42 + 0.18 * beachhead[idx] + 0.40 * converted_port)
            * (0.76 + 0.24 * corridor_service),
            0.02,
            1.0,
        )
        attacker_logistics_flow = np.minimum(
            attacker_source_capacity,
            np.minimum(attacker_cross_strait_capacity, attacker_sink_capacity),
        )
        min_attacker_logistics_flow[idx] = np.minimum(
            min_attacker_logistics_flow[idx], attacker_logistics_flow
        )
        attacker_legacy_logistics_target = (
            0.22 + 0.34 * sea_control + 0.24 * converted_port
            + 0.20 * np.minimum(1.0, attacker_fuel[idx] / 0.25)
        )
        attacker_flow_weight = 0.05 + 0.15 * converted_port
        attacker_logistics_target = (
            (1.0 - attacker_flow_weight) * attacker_legacy_logistics_target
            + attacker_flow_weight * attacker_logistics_flow
        )
        attacker_logistics[idx] = np.clip(
            0.70 * attacker_logistics[idx] + 0.30 * attacker_logistics_target,
            0.02,
            1.0,
        )
        ground_advantage = sigmoid(
            -1.0 + 2.1 * beachhead[idx] + 1.4 * attacker_logistics[idx]
            + 0.8 * air_control + 0.7 * sea_control
            - 1.7 * defender_org[idx] - 1.0 * command[idx]
            - 0.9 * geography_friction[idx]
            + 0.34 * (corridor_service - 0.50)
            + structure["combat_shift"]
        )
        control[idx] = np.clip(
            control[idx]
            + 0.058 * ground_advantage * beachhead[idx] * (1.0 - control[idx])
            - 0.021 * (1.0 - ground_advantage) * defender_org[idx] * control[idx],
            0.0, 1.0,
        )
        admin_control[idx] = np.clip(
            admin_control[idx]
            + 0.042 * control[idx] * converted_port * (1.0 - admin_control[idx])
            - 0.014 * defender_org[idx] * admin_control[idx],
            0.0, 1.0,
        )
        command[idx] = np.clip(
            command[idx]
            - 0.020 * air_control * strike_intensity
            - 0.018 * control[idx]
            + 0.012 * preparedness[idx] * (1.0 - control[idx]),
            0.02, 1.0,
        )
        defender_org[idx] = np.clip(
            defender_org[idx]
            - 0.038 * ground_advantage * (0.25 + beachhead[idx])
            - 0.012 * (1.0 - command[idx])
            + 0.013 * preparedness[idx] * (1.0 - control[idx]),
            0.01, 1.0,
        )

        attacker_tolerance[idx] = np.clip(
            attacker_tolerance[idx]
            - 0.006 * (1.0 - ground_advantage) - 0.004 * (1.0 - attacker_logistics[idx])
            + 0.002 * (control[idx] > 0.45),
            0.0, 1.0,
        )
        defender_tolerance[idx] = np.clip(
            defender_tolerance[idx]
            - 0.005 * ground_advantage - 0.004 * (1.0 - defender_port[idx])
            + 0.002 * (1.0 - control[idx]),
            0.0, 1.0,
        )

        expansion_prob = sigmoid(
            -7.0 + 2.0 * ally_air + 1.8 * ally_sea + 1.2 * air_control
            + 0.8 * sea_control + 0.010 * week
        )
        expansion_event = (~expansion[idx]) & (rng.random(len(idx)) < 0.018 * expansion_prob)
        expansion[idx] |= expansion_event
        expansion_event_week[idx[expansion_event]] = week
        event_transition_count[idx[expansion_event]] += 1

        broad_now = (control[idx] >= 0.70) & (admin_control[idx] >= 0.58) & (defender_org[idx] <= 0.28)
        broad_streak[idx] = np.where(broad_now, broad_streak[idx] + 1, 0)
        broad = broad_streak[idx] >= 8

        operational_collapse = (
            (week >= 18)
            & (beachhead[idx] < 0.018) & (amphibious_lift[idx] < 0.09)
        )
        long_war_hazard_scale = 1.0 / np.sqrt(1.0 + week / 104.0)
        withdrawal_hazard = 0.016 * long_war_hazard_scale * sigmoid(
            -2.0 + 4.0 * (0.22 - attacker_tolerance[idx])
            + 2.2 * (0.35 - control[idx]) + 1.5 * (0.30 - attacker_logistics[idx])
        )
        withdrawal = operational_collapse | (
            (week >= 26) & (rng.random(len(idx)) < withdrawal_hazard)
        )
        ceasefire_hazard = 0.0025 * sigmoid(
            -1.0 + 2.2 * (1.0 - attacker_tolerance[idx])
            + 1.6 * (1.0 - defender_tolerance[idx])
            + 0.8 * expansion[idx] + 0.006 * week
        )
        ceasefire = (
            (week >= 16) & (~broad) & (~withdrawal)
            & (attacker_tolerance[idx] < 0.62) & (defender_tolerance[idx] < 0.62)
            & (rng.random(len(idx)) < ceasefire_hazard)
        )

        chosen = np.full(len(idx), -1, dtype=np.int8)
        chosen[withdrawal] = 0
        chosen[ceasefire] = 1
        chosen[broad] = 3
        event = chosen >= 0
        outcome[idx[event]] = chosen[event]
        event_week[idx[event]] = week

        peak_control[idx] = np.maximum(peak_control[idx], control[idx])
        min_def_port[idx] = np.minimum(min_def_port[idx], defender_port[idx])
        min_def_airfield[idx] = np.minimum(min_def_airfield[idx], defender_airfield[idx])
        min_allied_forward_base[idx] = np.minimum(
            min_allied_forward_base[idx], allied_forward_base[idx]
        )
        min_allied_rear_base[idx] = np.minimum(
            min_allied_rear_base[idx], allied_rear_base[idx]
        )
        min_allied_port[idx] = np.minimum(min_allied_port[idx], allied_port[idx])
        min_allied_fuel[idx] = np.minimum(min_allied_fuel[idx], allied_fuel[idx])
        min_allied_c4isr[idx] = np.minimum(min_allied_c4isr[idx], allied_c4isr[idx])

        if week in HORIZON_WEEKS:
            still_active = outcome < 0
            horizon_snapshots[str(week)] = {
                **{name: float(np.mean(outcome == i)) for i, name in enumerate(OUTCOMES[:4])},
                "active_contested": float(np.mean(still_active & ((control >= 0.22) | (beachhead >= 0.18)))),
                "active_low_control": float(np.mean(still_active & (control < 0.22) & (beachhead < 0.18))),
            }

    for checkpoint in HORIZON_WEEKS:
        key = str(checkpoint)
        if key not in horizon_snapshots:
            still_active = outcome < 0
            horizon_snapshots[key] = {
                **{name: float(np.mean(outcome == i)) for i, name in enumerate(OUTCOMES[:4])},
                "active_contested": float(np.mean(still_active & ((control >= 0.22) | (beachhead >= 0.18)))),
                "active_low_control": float(np.mean(still_active & (control < 0.22) & (beachhead < 0.18))),
            }

    unresolved = outcome < 0
    contested = unresolved & ((control >= 0.22) | (beachhead >= 0.18))
    outcome[contested] = 2
    outcome[unresolved & ~contested] = 4
    event_week[outcome >= 0] = np.where(event_week[outcome >= 0] < 0, WEEKS, event_week[outcome >= 0])

    assert np.isin(ally_mode, [ALLY_NOT_ARRIVED, ALLY_RAMPING, ALLY_DEPLOYED]).all()
    assert np.isin(defender_port_mode, [FACILITY_OPERATIONAL, FACILITY_DEGRADED, FACILITY_DISABLED]).all()
    assert np.isin(defender_airfield_mode, [FACILITY_OPERATIONAL, FACILITY_DEGRADED, FACILITY_DISABLED]).all()
    assert np.isin(captured_port_mode, [PORT_UNAVAILABLE, PORT_CONVERTING, PORT_OPERATIONAL]).all()
    assert np.isin(allied_network_mode, [FACILITY_OPERATIONAL, FACILITY_DEGRADED, FACILITY_DISABLED]).all()
    for state in (
        attacker_theater_missiles, attacker_long_range_missiles, attacker_isr,
        attacker_launcher_survival, allied_forward_base, allied_rear_base,
        allied_port, allied_fuel, allied_c4isr, allied_bmd,
        allied_air_ready, allied_sea_ready, allied_enablers,
        allied_air_repair_queue, allied_sea_repair_queue,
        allied_air_turnaround_queue, allied_sea_service_queue,
        allied_precision_stock, allied_general_stock, allied_operational_fuel,
        allied_spare_stock, allied_spare_central, allied_spare_rear,
        allied_spare_pipeline_c2r, allied_spare_pipeline_r2f,
        attacker_spares, defender_spares,
        attacker_air_repair_queue, attacker_sea_repair_queue,
        defender_air_repair_queue, defender_sea_repair_queue,
    ):
        assert np.isfinite(state).all()
        assert np.all(state >= 0.0)
    completed_conversion = port_operational_event_week > 0
    assert np.all(port_conversion_event_week[completed_conversion] > 0)
    assert np.all(
        port_operational_event_week[completed_conversion]
        > port_conversion_event_week[completed_conversion]
    )
    deployed = ally_deployed_event_week > 0
    assert np.all(ally_arrival_event_week[deployed] > 0)
    assert np.all(ally_deployed_event_week[deployed] > ally_arrival_event_week[deployed])

    result = {
        "case": case_key,
        "case_label": case.label,
        "conflict_year": conflict_year,
        "capability_ratio": capability_ratio,
        "structure": structure_key,
        "paths": n,
        "horizon_snapshots": horizon_snapshots,
        "outcomes": {name: float(np.mean(outcome == i)) for i, name in enumerate(OUTCOMES)},
        "regional_expansion_overlay": float(np.mean(expansion)),
        "metrics": {
            "intervention_delay_weeks_p10_p50_p90": None if case_key == "taiwan_alone" else [
                q(delay, 0.10), q(delay, 0.50), q(delay, 0.90)
            ],
            "peak_population_control_p10_p50_p90": [q(peak_control, 0.10), q(peak_control, 0.50), q(peak_control, 0.90)],
            "minimum_port_availability_p10_p50_p90": [q(min_def_port, 0.10), q(min_def_port, 0.50), q(min_def_port, 0.90)],
            "minimum_airfield_availability_p10_p50_p90": [q(min_def_airfield, 0.10), q(min_def_airfield, 0.50), q(min_def_airfield, 0.90)],
            "minimum_allied_forward_base_p10_p50_p90": [q(min_allied_forward_base, 0.10), q(min_allied_forward_base, 0.50), q(min_allied_forward_base, 0.90)],
            "minimum_allied_rear_base_p10_p50_p90": [q(min_allied_rear_base, 0.10), q(min_allied_rear_base, 0.50), q(min_allied_rear_base, 0.90)],
            "minimum_allied_port_p10_p50_p90": [q(min_allied_port, 0.10), q(min_allied_port, 0.50), q(min_allied_port, 0.90)],
            "minimum_allied_fuel_p10_p50_p90": [q(min_allied_fuel, 0.10), q(min_allied_fuel, 0.50), q(min_allied_fuel, 0.90)],
            "minimum_allied_c4isr_p10_p50_p90": [q(min_allied_c4isr, 0.10), q(min_allied_c4isr, 0.50), q(min_allied_c4isr, 0.90)],
            "minimum_allied_logistics_max_flow_p10_p50_p90": [q(min_allied_logistics_flow, 0.10), q(min_allied_logistics_flow, 0.50), q(min_allied_logistics_flow, 0.90)],
            "minimum_attacker_logistics_max_flow_p10_p50_p90": [q(min_attacker_logistics_flow, 0.10), q(min_attacker_logistics_flow, 0.50), q(min_attacker_logistics_flow, 0.90)],
            "remaining_long_range_missiles_p10_p50_p90": [q(attacker_long_range_missiles, 0.10), q(attacker_long_range_missiles, 0.50), q(attacker_long_range_missiles, 0.90)],
            "remaining_theater_missiles_p10_p50_p90": [q(attacker_theater_missiles, 0.10), q(attacker_theater_missiles, 0.50), q(attacker_theater_missiles, 0.90)],
            "total_long_range_salvo_p10_p50_p90": [q(total_long_range_salvo, 0.10), q(total_long_range_salvo, 0.50), q(total_long_range_salvo, 0.90)],
            "total_counterstrike_p10_p50_p90": [q(total_counterstrike, 0.10), q(total_counterstrike, 0.50), q(total_counterstrike, 0.90)],
            "delivered_mass_p10_p50_p90": [q(delivered_mass, 0.10), q(delivered_mass, 0.50), q(delivered_mass, 0.90)],
            "allied_air_ready_final_p10_p50_p90": [q(allied_air_ready, 0.10), q(allied_air_ready, 0.50), q(allied_air_ready, 0.90)],
            "allied_sea_ready_final_p10_p50_p90": [q(allied_sea_ready, 0.10), q(allied_sea_ready, 0.50), q(allied_sea_ready, 0.90)],
            "allied_enablers_final_p10_p50_p90": [q(allied_enablers, 0.10), q(allied_enablers, 0.50), q(allied_enablers, 0.90)],
            "allied_spares_final_p10_p50_p90": [q(allied_spare_stock, 0.10), q(allied_spare_stock, 0.50), q(allied_spare_stock, 0.90)],
            "allied_spares_rear_final_p10_p50_p90": [q(allied_spare_rear, 0.10), q(allied_spare_rear, 0.50), q(allied_spare_rear, 0.90)],
            "allied_spares_central_final_p10_p50_p90": [q(allied_spare_central, 0.10), q(allied_spare_central, 0.50), q(allied_spare_central, 0.90)],
            "allied_spares_pipeline_final_p10_p50_p90": [
                q(allied_spare_pipeline_c2r.sum(axis=0) + allied_spare_pipeline_r2f.sum(axis=0), 0.10),
                q(allied_spare_pipeline_c2r.sum(axis=0) + allied_spare_pipeline_r2f.sum(axis=0), 0.50),
                q(allied_spare_pipeline_c2r.sum(axis=0) + allied_spare_pipeline_r2f.sum(axis=0), 0.90),
            ],
            "minimum_allied_spare_fill_rate_p10_p50_p90": [q(min_allied_spare_fill_rate, 0.10), q(min_allied_spare_fill_rate, 0.50), q(min_allied_spare_fill_rate, 0.90)],
            "gis_calibration": GIS,
            "corridor_mobility_p10_p50_p90": [q(corridor_mobility, 0.10), q(corridor_mobility, 0.50), q(corridor_mobility, 0.90)],
            "search_effort_shares": {
                name: float(total_search_effort[:, i].sum() / max(total_search_effort.sum(), 1e-9))
                for i, name in enumerate(("mobile_launchers", "fleet", "lift"))
            },
            "mean_weekly_search_detection_by_target": {
                name: float(np.mean(total_search_detection[:, i] / WEEKS))
                for i, name in enumerate(("mobile_launchers", "fleet", "lift"))
            },
            "military_economics_allocation_shares": {
                name: float(industrial_allocation[:, i].sum() / max(industrial_allocation.sum(), 1e-9))
                for i, name in enumerate(("air_missile_defense", "spares", "repair_capacity", "transport"))
            },
            "industrial_budget_spent_p10_p50_p90": [
                q(industrial_budget_spent, 0.10), q(industrial_budget_spent, 0.50), q(industrial_budget_spent, 0.90)
            ],
            "allied_cargo_capacity_utilization_p10_p50_p90": [
                q(total_allied_cargo_used / np.maximum(total_allied_cargo_capacity, 1e-8), 0.10),
                q(total_allied_cargo_used / np.maximum(total_allied_cargo_capacity, 1e-8), 0.50),
                q(total_allied_cargo_used / np.maximum(total_allied_cargo_capacity, 1e-8), 0.90),
            ],
            "interdiction_target_effort_shares": {
                name: float(interdiction_effort[:, target_index].sum() / max(interdiction_effort.sum(), 1e-9))
                for target_index, name in enumerate((
                    "forward_base", "rear_base", "port", "fuel", "c4isr"
                ))
            },
            "allied_air_destroyed_p10_p50_p90": [q(allied_air_destroyed, 0.10), q(allied_air_destroyed, 0.50), q(allied_air_destroyed, 0.90)],
            "allied_sea_destroyed_p10_p50_p90": [q(allied_sea_destroyed, 0.10), q(allied_sea_destroyed, 0.50), q(allied_sea_destroyed, 0.90)],
            "allied_enabler_destroyed_p10_p50_p90": [q(allied_enabler_destroyed, 0.10), q(allied_enabler_destroyed, 0.50), q(allied_enabler_destroyed, 0.90)],
            "maximum_allied_air_repair_queue_p10_p50_p90": [q(max_allied_air_repair_queue, 0.10), q(max_allied_air_repair_queue, 0.50), q(max_allied_air_repair_queue, 0.90)],
            "maximum_allied_sea_repair_queue_p10_p50_p90": [q(max_allied_sea_repair_queue, 0.10), q(max_allied_sea_repair_queue, 0.50), q(max_allied_sea_repair_queue, 0.90)],
            "maximum_allied_air_turnaround_queue_p10_p50_p90": [q(max_allied_air_turnaround_queue, 0.10), q(max_allied_air_turnaround_queue, 0.50), q(max_allied_air_turnaround_queue, 0.90)],
            "maximum_allied_sea_service_queue_p10_p50_p90": [q(max_allied_sea_service_queue, 0.10), q(max_allied_sea_service_queue, 0.50), q(max_allied_sea_service_queue, 0.90)],
            "allied_air_service_fraction_p10_p50_p90": [
                q(total_allied_air_served / np.maximum(total_allied_air_demand, 1e-8), 0.10),
                q(total_allied_air_served / np.maximum(total_allied_air_demand, 1e-8), 0.50),
                q(total_allied_air_served / np.maximum(total_allied_air_demand, 1e-8), 0.90),
            ],
            "allied_sea_service_fraction_p10_p50_p90": [
                q(total_allied_sea_served / np.maximum(total_allied_sea_demand, 1e-8), 0.10),
                q(total_allied_sea_served / np.maximum(total_allied_sea_demand, 1e-8), 0.50),
                q(total_allied_sea_served / np.maximum(total_allied_sea_demand, 1e-8), 0.90),
            ],
            "attacker_precision_shortage_by_12w": float(np.mean((attacker_ammo_shortage_week > 0) & (attacker_ammo_shortage_week <= 12))),
            "defender_precision_shortage_by_12w": float(np.mean((defender_ammo_shortage_week > 0) & (defender_ammo_shortage_week <= 12))),
            "attacker_precision_shortage_by_26w": float(np.mean((attacker_ammo_shortage_week > 0) & (attacker_ammo_shortage_week <= 26))),
            "defender_precision_shortage_by_26w": float(np.mean((defender_ammo_shortage_week > 0) & (defender_ammo_shortage_week <= 26))),
            "attacker_precision_shortage_by_52w": float(np.mean((attacker_ammo_shortage_week > 0) & (attacker_ammo_shortage_week <= 52))),
            "defender_precision_shortage_by_52w": float(np.mean((defender_ammo_shortage_week > 0) & (defender_ammo_shortage_week <= 52))),
            "allied_precision_shortage_by_12w": float(np.mean((allied_precision_shortage_week > 0) & (allied_precision_shortage_week <= 12))),
            "allied_precision_shortage_by_26w": float(np.mean((allied_precision_shortage_week > 0) & (allied_precision_shortage_week <= 26))),
            "allied_precision_shortage_by_52w": float(np.mean((allied_precision_shortage_week > 0) & (allied_precision_shortage_week <= 52))),
            "allied_spare_shortage_by_12w": float(np.mean((allied_spare_shortage_week > 0) & (allied_spare_shortage_week <= 12))),
            "allied_spare_shortage_by_26w": float(np.mean((allied_spare_shortage_week > 0) & (allied_spare_shortage_week <= 26))),
            "allied_spare_shortage_by_52w": float(np.mean((allied_spare_shortage_week > 0) & (allied_spare_shortage_week <= 52))),
            "broad_control_week_p10_p50_p90": None if not np.any(outcome == 3) else [
                q(event_week[outcome == 3], 0.10), q(event_week[outcome == 3], 0.50), q(event_week[outcome == 3], 0.90)
            ],
            "deds_event_metrics": {
                "ally_arrival_week_p10_p50_p90": event_week_quantiles(ally_arrival_event_week),
                "ally_full_deployment_week_p10_p50_p90": event_week_quantiles(ally_deployed_event_week),
                "defender_port_disabled_share": float(np.mean(port_disabled_event_week > 0)),
                "defender_port_first_disabled_week_p10_p50_p90": event_week_quantiles(port_disabled_event_week),
                "defender_airfield_disabled_share": float(np.mean(airfield_disabled_event_week > 0)),
                "defender_airfield_first_disabled_week_p10_p50_p90": event_week_quantiles(airfield_disabled_event_week),
                "captured_port_conversion_share": float(np.mean(port_conversion_event_week > 0)),
                "captured_port_operational_share": float(np.mean(port_operational_event_week > 0)),
                "captured_port_operational_week_p10_p50_p90": event_week_quantiles(port_operational_event_week),
                "regional_expansion_week_p10_p50_p90": event_week_quantiles(expansion_event_week),
                "allied_forward_base_disabled_share": float(np.mean(allied_forward_disabled_week > 0)),
                "allied_forward_base_first_disabled_week_p10_p50_p90": event_week_quantiles(allied_forward_disabled_week),
                "allied_rear_base_disabled_share": float(np.mean(allied_rear_disabled_week > 0)),
                "allied_port_disabled_share": float(np.mean(allied_port_disabled_week > 0)),
                "event_transitions_p10_p50_p90": [
                    q(event_transition_count, 0.10),
                    q(event_transition_count, 0.50),
                    q(event_transition_count, 0.90),
                ],
            },
        },
    }
    return result


def aggregate(results):
    grouped = {}
    for row in results:
        key = (row["case"], row["conflict_year"])
        grouped.setdefault(key, []).append(row)

    table = []
    for (case_key, year), rows in grouped.items():
        central = next(row for row in rows if row["structure"] == "central")
        record = {
            "case": case_key,
            "case_label": central["case_label"],
            "conflict_year": year,
            "capability_ratio": central["capability_ratio"],
        }
        for outcome in OUTCOMES:
            values = [row["outcomes"][outcome] for row in rows]
            record[f"{outcome}_central"] = central["outcomes"][outcome]
            record[f"{outcome}_min"] = min(values)
            record[f"{outcome}_max"] = max(values)
        expansion = [row["regional_expansion_overlay"] for row in rows]
        record["regional_expansion_central"] = central["regional_expansion_overlay"]
        record["regional_expansion_min"] = min(expansion)
        record["regional_expansion_max"] = max(expansion)
        record["central_metrics"] = central["metrics"]
        table.append(record)
    return sorted(table, key=lambda x: (x["conflict_year"], list(ALLIANCE_CASES).index(x["case"])))


def verify(results, table):
    for row in results:
        assert abs(sum(row["outcomes"].values()) - 1.0) < 1e-12
        assert all(0.0 <= value <= 1.0 for value in row["outcomes"].values())
    by_year = {}
    for row in table:
        by_year.setdefault(row["conflict_year"], {})[row["case"]] = row
    for year, rows in by_year.items():
        timely = rows["timely_full"]["broad_control_central"]
        limited = rows["limited"]["broad_control_central"]
        delayed = rows["delayed"]["broad_control_central"]
        alone = rows["taiwan_alone"]["broad_control_central"]
        assert timely <= min(limited, delayed) + 0.01, (year, timely, limited, delayed)
        assert alone + 0.01 >= max(limited, delayed), (year, limited, delayed, alone)
    for case_key in ALLIANCE_CASES:
        sequence = [by_year[y][case_key]["broad_control_central"] for y in sorted(YEARS)]
        assert all(a <= b + 0.015 for a, b in zip(sequence, sequence[1:]))


def write_outputs(results, table):
    summary = {
        "model": "V3.6 military OR campaign with GIS corridors, search theory, multi-echelon spares and military-economics allocation",
        "seed": SEED,
        "paths_per_cell": PATHS,
        "weeks": WEEKS,
        "horizon_weeks": HORIZON_WEEKS,
        "conflict_year_capability_ratio": YEARS,
        "alliance_cases": {key: vars(value) for key, value in ALLIANCE_CASES.items()},
        "structural_adjudicators": STRUCTURES,
        "results": results,
        "scenario_envelopes": table,
        "interpretation": "Central frequencies and structural envelopes are conditional model outputs, not empirical probabilities. Public wargames are used as directional soft constraints, not iid observations.",
        "verification": "TAIWAN_CAMPAIGN_V36_VERIFICATION: PASS",
    }
    json_path = OUT / "台海V3.6_GIS走廊多层备件搜索与军事经济_仿真摘要.json"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    csv_path = OUT / "台海V3.6四类联盟条件_结果.csv"
    fields = [
        "conflict_year", "case", "case_label", "capability_ratio",
        "defense_restored_or_withdrawal_central", "negotiated_ceasefire_central",
        "limited_contested_control_central", "broad_control_central",
        "broad_control_min", "broad_control_max", "frozen_stalemate_central",
        "regional_expansion_central",
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in table:
            writer.writerow({key: row[key] for key in fields})
    return json_path, csv_path


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    results = []
    run = 0
    for year, capability_ratio in YEARS.items():
        for case_key, case in ALLIANCE_CASES.items():
            for structure_key, structure in STRUCTURES.items():
                run += 1
                results.append(simulate_case(
                    case_key, case, year, capability_ratio, structure_key, structure,
                    SEED + 1009 * run,
                ))
    table = aggregate(results)
    json_path, csv_path = write_outputs(results, table)
    verify(results, table)
    print("TAIWAN_CAMPAIGN_V36_VERIFICATION: PASS")
    print(json_path)
    print(csv_path)
    for row in table:
        print(
            row["conflict_year"], row["case_label"],
            f"broad={100 * row['broad_control_central']:.2f}%",
            f"range={100 * row['broad_control_min']:.2f}-{100 * row['broad_control_max']:.2f}%",
            f"stalemate={100 * row['frozen_stalemate_central']:.2f}%",
        )


if __name__ == "__main__":
    main()
