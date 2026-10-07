from __future__ import annotations

from dataclasses import dataclass

import numpy as np


PREWAR_SCENARIOS = {
    "central": {},
    "growth_rebalancing_and_stabilization": {
        "growth": [0.15, 0.15, 0.10, 0.0], "alliance_credibility": 0.05,
        "interdependence": 0.30, "political_pressure": -0.25,
    },
    "china_high_growth_rearmament": {
        "growth": [0.65, 0.0, 0.0, 0.0], "china_burden": 0.35,
        "political_pressure": 0.20,
    },
    "china_structural_slowdown": {
        "growth": [-0.85, 0.0, 0.0, 0.0], "china_burden": -0.10,
        "political_pressure": 0.15,
    },
    "us_japan_taiwan_coordination": {
        "growth": [0.0, 0.10, 0.10, 0.15], "allied_burden": 0.40,
        "alliance_credibility": 0.55,
    },
    "technology_trade_decoupling": {
        "growth": [-0.35, -0.25, 0.0, 0.0], "interdependence": -0.55,
        "political_pressure": 0.30, "decoupling": 0.60,
    },
    "closing_window_pressure": {
        "growth": [-0.25, 0.0, 0.0, 0.0], "allied_burden": 0.35,
        "alliance_credibility": 0.40, "political_pressure": 0.65,
        "decoupling": 0.25,
    },
}


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values, axis=-1, keepdims=True)
    weights = np.exp(np.clip(shifted, -35.0, 35.0))
    return weights / np.maximum(np.sum(weights, axis=-1, keepdims=True), 1e-12)


@dataclass
class PrewarFactors:
    actor_support: np.ndarray
    taiwan_readiness: np.ndarray
    arms_support_us: np.ndarray
    arms_support_japan: np.ndarray
    leader_continuity: np.ndarray
    pressure_belief: np.ndarray
    military_capital: np.ndarray
    economic_capital: np.ndarray
    strategy_distribution: np.ndarray
    ot_allocation: np.ndarray
    diagnostics: dict


class ClosedPrewarEvolutionSystem:
    """Same-path monthly prewar HMM, games, arms, capital, OT, and leader gates."""

    def __init__(
        self, paths: int, seed: int, observed_anchor: dict,
        causal_pressure_effect: float = 0.0, scenario: str = "central",
    ):
        if scenario not in PREWAR_SCENARIOS:
            raise ValueError(f"unknown prewar scenario: {scenario}")
        self.paths = paths
        self.rng = np.random.default_rng(seed)
        self.anchor = observed_anchor
        self.causal_pressure_effect = float(causal_pressure_effect)
        raw_scenario = PREWAR_SCENARIOS[scenario]
        self.scenario = scenario
        self.scenario_parameters = {
            "growth": np.asarray(raw_scenario.get("growth", [0.0] * 4), dtype=float),
            "china_burden": float(raw_scenario.get("china_burden", 0.0)),
            "allied_burden": float(raw_scenario.get("allied_burden", 0.0)),
            "alliance_credibility": float(raw_scenario.get("alliance_credibility", 0.0)),
            "interdependence": float(raw_scenario.get("interdependence", 0.0)),
            "political_pressure": float(raw_scenario.get("political_pressure", 0.0)),
            "decoupling": float(raw_scenario.get("decoupling", 0.0)),
        }
        self.interdependence = np.ones(paths)
        self.support = np.clip(
            0.86 * observed_anchor["actor_support"]
            + 0.14 * np.array([[0.66, 0.61, 0.39, 0.10]]), 0.01, 0.99
        )
        group_shift = np.array([0.14, 0.03, -0.08, -0.02])[None, None, :]
        base_logit = np.log(self.support / np.maximum(1.0 - self.support, 1e-12))
        self.population_latent = base_logit[:, :, None] + group_shift
        self.population_weights = np.array([0.24, 0.31, 0.27, 0.18])
        self.graphon = np.array([
            [0.50, 0.28, 0.14, 0.08], [0.22, 0.46, 0.23, 0.09],
            [0.11, 0.25, 0.47, 0.17], [0.08, 0.14, 0.27, 0.51],
        ])
        self.filtered_poll = self.support.copy()
        self.party_state = self.rng.random((paths, 4)) < self.support
        self.pressure_belief = np.repeat(np.array([[0.66, 0.27, 0.07]]), paths, axis=0)
        self.strategy = np.repeat(
            np.array([[
                [0.48, 0.40, 0.12],
                [0.30, 0.58, 0.12],
                [0.36, 0.45, 0.19],
            ]]), paths, axis=0,
        )
        self.economic_capital = np.ones((paths, 4))
        self.military_capital = np.ones((paths, 4))
        self.growth_rate = np.repeat(
            np.array([[4.40, 6.455, 2.30, 0.80]]), paths, axis=0
        )
        self.leader_stage = np.zeros((paths, 4), dtype=np.int8)
        self.leader_continuity = np.ones((paths, 4))
        self.pipeline = np.zeros((paths, 2, 5, 6))
        self.pipeline[:, :, :, 0] = self.rng.lognormal(-2.0, 0.22, (paths, 2, 5))
        self.delivered_capability = np.zeros((paths, 5))
        self.delivered_by_donor = np.zeros((paths, 2))
        self.last_absorbed = np.zeros((paths, 5))
        self.ot_allocation = np.full((paths, 4, 6), 1.0 / 6.0)
        self.max_probability_residual = 0.0
        self.max_resource_residual = 0.0
        self.max_pipeline_residual = 0.0

    def _resource_ot(self, demand: np.ndarray) -> None:
        cost = np.array([
            [0.10, 0.36, 0.32, 0.50, 0.65, 0.42],
            [0.42, 0.18, 0.26, 0.48, 0.54, 0.36],
            [0.55, 0.38, 0.12, 0.34, 0.48, 0.30],
            [0.62, 0.44, 0.30, 0.14, 0.36, 0.20],
        ])
        kernel = np.exp(-cost / 0.24)[None, :, :]
        supply = np.clip(
            np.stack((
                self.economic_capital[:, 0], self.economic_capital[:, 1],
                self.economic_capital[:, 2], self.economic_capital[:, 3],
            ), axis=1), 0.10, 1.40
        )
        u = np.ones_like(supply)
        v = np.ones_like(demand)
        relaxation = 0.72
        for _ in range(35):
            u = (supply / np.maximum(np.sum(kernel * v[:, None, :], axis=2), 1e-12)) ** relaxation
            v = (demand / np.maximum(np.sum(kernel * u[:, :, None], axis=1), 1e-12)) ** relaxation
        plan = kernel * u[:, :, None] * v[:, None, :]
        source_scale = np.minimum(1.0, supply / np.maximum(plan.sum(axis=2), 1e-12))
        plan *= source_scale[:, :, None]
        self.ot_allocation = plan / np.maximum(plan.sum(axis=(1, 2), keepdims=True), 1e-12)
        self.max_resource_residual = max(
            self.max_resource_residual,
            float(np.max(np.maximum(plan.sum(axis=2) - supply, 0.0))),
        )

    def _update_pipeline(self, donor_support: np.ndarray, pressure: np.ndarray) -> None:
        opening = self.pipeline.sum(axis=(1, 2, 3)) + self.delivered_capability.sum(axis=1)
        inflow = np.zeros((self.paths, 2, 5))
        category = np.array([0.026, 0.023, 0.018, 0.021, 0.025])[None, None, :]
        inflow[:, 0, :] = category[:, 0, :] * donor_support[:, 0, None] * (0.70 + 0.30 * pressure[:, None])
        inflow[:, 1, :] = 0.62 * category[:, 0, :] * donor_support[:, 1, None] * (0.75 + 0.25 * pressure[:, None])
        self.pipeline[:, :, :, 0] += inflow
        progress_rates = np.array([0.18, 0.12, 0.08, 0.16, 0.20])
        losses = np.zeros(self.paths)
        for stage in range(4, -1, -1):
            flow = self.pipeline[:, :, :, stage] * progress_rates[stage]
            loss = 0.004 * flow * (1.0 + pressure[:, None, None])
            self.pipeline[:, :, :, stage] -= flow
            self.pipeline[:, :, :, stage + 1] += flow - loss
            losses += loss.sum(axis=(1, 2))
        absorbed = self.pipeline[:, :, :, 5] * 0.14
        self.pipeline[:, :, :, 5] -= absorbed
        self.last_absorbed = absorbed.sum(axis=1)
        self.delivered_capability += self.last_absorbed
        self.delivered_by_donor += absorbed.sum(axis=2)
        closing = self.pipeline.sum(axis=(1, 2, 3)) + self.delivered_capability.sum(axis=1)
        residual = np.max(np.abs(closing - opening - inflow.sum(axis=(1, 2)) + losses))
        self.max_pipeline_residual = max(self.max_pipeline_residual, float(residual))

    def update(self, month: int) -> None:
        pressure = self.pressure_belief[:, 1] + self.pressure_belief[:, 2]
        scenario = self.scenario_parameters
        china_payoff = np.stack((
            0.62 - 0.24 * pressure,
            0.44 + 0.50 * pressure - 0.18 * self.support[:, 2],
            -0.30 + 0.72 * pressure + 0.35 * self.support[:, 0]
            - 0.42 * self.support[:, 2] - 0.25 * self.military_capital[:, 1],
        ), axis=1)
        china_payoff[:, 1] += 0.10 * scenario["political_pressure"]
        china_payoff[:, 2] += 0.18 * scenario["political_pressure"] + 0.08 * scenario["decoupling"]
        taiwan_payoff = np.stack((
            0.46 + 0.24 * self.economic_capital[:, 1] - 0.30 * pressure,
            0.38 + 0.48 * pressure + 0.22 * self.support[:, 2],
            -0.20 + 0.36 * (1.0 - self.support[:, 0]) + 0.26 * self.support[:, 2]
            - 0.50 * pressure,
        ), axis=1)
        alliance_payoff = np.stack((
            0.50 - 0.56 * pressure,
            0.34 + 0.42 * pressure - 0.14 * (1.0 - self.economic_capital[:, 2]),
            0.08 + 0.70 * pressure + 0.25 * self.support[:, 1]
            - 0.28 * (1.0 - self.economic_capital[:, 2]),
        ), axis=1)
        alliance_payoff[:, 1:] += scenario["alliance_credibility"] * np.array([0.10, 0.18])[None, :]
        response = np.stack((
            _softmax(china_payoff / 0.20),
            _softmax(taiwan_payoff / 0.20),
            _softmax(alliance_payoff / 0.20),
        ), axis=1)
        average_payoff = np.stack((china_payoff, taiwan_payoff, alliance_payoff), axis=1)
        replicator = self.strategy * (
            average_payoff - np.sum(self.strategy * average_payoff, axis=2, keepdims=True)
        )
        self.strategy = np.clip(
            self.strategy + 0.045 * replicator + 0.16 * (response - self.strategy),
            1e-8, None,
        )
        self.strategy /= self.strategy.sum(axis=2, keepdims=True)
        self.max_probability_residual = max(
            self.max_probability_residual,
            float(np.max(np.abs(self.strategy.sum(axis=2) - 1.0))),
        )

        aggressive = self.strategy[:, 0, 1] + self.strategy[:, 0, 2]
        deterrence = self.strategy[:, 2, 1] + self.strategy[:, 2, 2]
        transition = np.zeros((self.paths, 3, 3))
        transition[:, 0, :] = np.stack((
            0.91 - 0.16 * aggressive, 0.075 + 0.12 * aggressive, 0.015 + 0.04 * aggressive
        ), axis=1)
        transition[:, 1, :] = np.stack((
            0.10 + 0.08 * deterrence, 0.79 - 0.06 * aggressive, 0.11 + 0.06 * aggressive - 0.08 * deterrence
        ), axis=1)
        transition[:, 2, :] = np.stack((
            0.025 + 0.05 * deterrence, 0.20 + 0.10 * deterrence, 0.775 - 0.15 * deterrence
        ), axis=1)
        transition = np.clip(transition, 1e-5, None)
        transition /= transition.sum(axis=2, keepdims=True)
        predicted = np.einsum("pi,pij->pj", self.pressure_belief, transition)
        latent_pressure = predicted[:, 1] + 2.0 * predicted[:, 2]
        local_shock = self.rng.random(self.paths) < 0.035
        observation = (
            latent_pressure + local_shock * self.causal_pressure_effect
            + 0.10 * scenario["political_pressure"] + 0.08 * scenario["decoupling"]
            + self.rng.normal(0.0, 0.18, self.paths)
        )
        centers = np.array([0.12, 0.92, 1.72])[None, :]
        likelihood = np.exp(-0.5 * ((observation[:, None] - centers) / 0.28) ** 2)
        self.pressure_belief = predicted * likelihood
        self.pressure_belief /= np.maximum(self.pressure_belief.sum(axis=1, keepdims=True), 1e-12)

        crisis = self.pressure_belief[:, 2]
        demand = np.stack((
            0.35 + 0.55 * crisis, 0.42 + 0.40 * pressure,
            0.34 + 0.32 * pressure, 0.38 + 0.30 * pressure,
            0.28 + 0.45 * crisis, 0.32 + 0.28 * pressure,
        ), axis=1)
        self._resource_ot(demand)
        resilience_flow = self.ot_allocation[:, :, 1].sum(axis=1)
        information_flow = self.ot_allocation[:, :, 2].sum(axis=1)
        economic_buffer = self.ot_allocation[:, :, 3].sum(axis=1)

        leader_hazard = np.clip(
            0.002 + 0.012 * crisis[:, None] + 0.006 * self.strategy[:, 0, 2, None],
            0.0, 0.06,
        )
        entering = (self.leader_stage == 0) & (self.rng.random((self.paths, 4)) < leader_hazard)
        self.leader_stage[entering] = 1
        intercept = np.clip(
            0.30 + 0.32 * self.leader_continuity + 0.18 * information_flow[:, None]
            + 0.10 * resilience_flow[:, None], 0.05, 0.92
        )
        active = (self.leader_stage > 0) & (self.leader_stage < 5)
        blocked = active & (self.rng.random((self.paths, 4)) < intercept * 0.22)
        advance = active & (~blocked) & (self.rng.random((self.paths, 4)) < (0.18 + 0.22 * crisis[:, None]))
        self.leader_stage[blocked] = 0
        self.leader_stage[advance] += 1
        implemented = self.leader_stage == 5
        self.leader_continuity = np.clip(
            self.leader_continuity - 0.16 * implemented
            + 0.025 * (1.0 - self.leader_continuity), 0.12, 1.0
        )
        self.leader_stage[implemented & (self.rng.random((self.paths, 4)) < 0.35)] = 0

        donor_support = np.column_stack((
            np.clip(0.55 * self.support[:, 2] + 0.45 * self.strategy[:, 2, 1:].sum(axis=1), 0.01, 0.99),
            np.clip(0.58 * self.support[:, 3] + 0.42 * self.strategy[:, 2, 1:].sum(axis=1), 0.01, 0.99),
        ))
        donor_support = np.clip(
            donor_support + 0.12 * scenario["alliance_credibility"], 0.01, 0.99
        )
        self._update_pipeline(donor_support, pressure)

        year = 2026.0 + month / 12.0
        knot_years = np.array([2026.0, 2035.0, 2050.0, 2075.0, 2100.0])
        target_paths = np.array([
            [4.40, 3.40, 2.30, 1.70, 1.40],
            [6.455, 2.50, 1.90, 1.40, 1.10],
            [2.30, 2.00, 1.80, 1.70, 1.60],
            [0.80, 0.90, 0.70, 0.60, 0.50],
        ])
        target_growth = np.array([
            np.interp(year, knot_years, target_paths[actor]) for actor in range(4)
        ])[None, :] + scenario["growth"][None, :]
        innovation_scale = np.array([1.855, 2.751, 1.791, 1.891])[None, :] / np.sqrt(12.0)
        common_growth = self.rng.normal(0.0, 0.10, (self.paths, 1))
        self.growth_rate = np.clip(
            target_growth + 0.92 * (self.growth_rate - target_growth)
            + 0.16 * innovation_scale * self.rng.normal(size=(self.paths, 4))
            + common_growth * np.array([[0.55, 0.70, 0.45, 0.60]]),
            -8.0, 10.0,
        )
        self.interdependence = np.clip(
            0.992 * self.interdependence
            + 0.008 * (1.0 + scenario["interdependence"] - scenario["decoupling"])
            - 0.002 * pressure,
            0.15, 1.35,
        )
        pressure_drag = 0.18 * pressure[:, None] * np.array([[0.70, 0.80, 0.25, 0.45]])
        pressure_drag[:, [0, 1]] += 0.16 * scenario["decoupling"]
        monthly_growth = (self.growth_rate - pressure_drag) / 1200.0
        self.economic_capital = np.clip(
            self.economic_capital * (1.0 + monthly_growth)
            + 0.0008 * economic_buffer[:, None], 0.20, 8.0
        )
        burden_offset = np.array([
            scenario["china_burden"], scenario["allied_burden"],
            scenario["allied_burden"], scenario["allied_burden"],
        ])[None, :] / 100.0
        defense_effort = (
            np.array([0.0155, 0.0332, 0.0305, 0.0200])[None, :] + burden_offset
        ) / 12.0
        self.military_capital = np.clip(
            (1.0 - 0.004) * self.military_capital
            + defense_effort * self.economic_capital
            + np.column_stack((
                np.zeros(self.paths), 0.06 * self.last_absorbed.mean(axis=1),
                np.zeros(self.paths), np.zeros(self.paths),
            )), 0.35, 5.0
        )
        observed_poll = np.clip(
            self.support + self.rng.normal(0.0, 0.035, self.support.shape), 0.01, 0.99
        )
        self.filtered_poll = 0.76 * self.filtered_poll + 0.24 * observed_poll
        if (month + 1) % 48 == 0:
            incumbent_score = (
                1.10 * (self.filtered_poll - 0.50)
                + 0.55 * (self.economic_capital - 1.0)
                - 0.45 * pressure[:, None]
                + self.rng.normal(0.0, 0.16, (self.paths, 4))
            )
            election_probability = 1.0 / (1.0 + np.exp(-incumbent_score))
            self.party_state = self.rng.random((self.paths, 4)) < election_probability
        population_support = 1.0 / (1.0 + np.exp(-self.population_latent))
        graphon_mean = population_support @ self.graphon.T
        type_threat = np.array([1.10, 0.96, 0.82, 1.16])[None, None, :]
        type_cost = np.array([0.72, 0.92, 1.18, 0.80])[None, None, :]
        population_drift = (
            0.30 * (graphon_mean - population_support)
            + 0.20 * (self.party_state[:, :, None] - 0.50)
            + 0.18 * pressure[:, None, None] * type_threat
            - 0.16 * np.maximum(1.0 - self.economic_capital, 0.0)[:, :, None] * type_cost
            - 0.10 * (1.0 - self.leader_continuity)[:, :, None]
        )
        anchor_logit = np.log(
            np.clip(self.anchor["actor_support"], 0.01, 0.99)
            / np.clip(1.0 - self.anchor["actor_support"], 0.01, 0.99)
        )[:, :, None]
        self.population_latent = np.clip(
            self.population_latent + 0.08 * (anchor_logit - self.population_latent)
            + population_drift + self.rng.normal(0.0, 0.045, self.population_latent.shape),
            -5.0, 5.0,
        )
        population_support = 1.0 / (1.0 + np.exp(-self.population_latent))
        self.support = np.clip(
            np.sum(population_support * self.population_weights[None, None, :], axis=2),
            0.01, 0.99,
        )

    def run(self, months: int = 60) -> PrewarFactors:
        for month in range(months):
            self.update(month)
        delivered = self.delivered_capability.mean(axis=1)
        readiness = np.clip(
            0.72 * self.anchor["taiwan_readiness"]
            + 0.18 * self.military_capital[:, 1] + 0.10 * (1.0 + delivered),
            0.55, 1.65,
        )
        donor_pipeline = self.pipeline.sum(axis=(2, 3))
        donor_total = donor_pipeline + self.delivered_by_donor
        donor_support = donor_total / np.maximum(donor_total + 0.55, 1e-12)
        diagnostics = {
            "pressure_simplex_residual": float(np.max(np.abs(self.pressure_belief.sum(axis=1) - 1.0))),
            "strategy_simplex_residual": self.max_probability_residual,
            "ot_supply_residual": self.max_resource_residual,
            "arms_pipeline_stock_residual": self.max_pipeline_residual,
            "minimum_pipeline_stock": float(np.min(self.pipeline)),
            "filtered_poll_range": [
                float(np.min(self.filtered_poll)), float(np.max(self.filtered_poll))
            ],
        }
        return PrewarFactors(
            actor_support=self.support.copy(),
            taiwan_readiness=readiness,
            arms_support_us=np.clip(donor_support[:, 0], 0.01, 0.99),
            arms_support_japan=np.clip(donor_support[:, 1], 0.01, 0.99),
            leader_continuity=self.leader_continuity.copy(),
            pressure_belief=self.pressure_belief.copy(),
            military_capital=self.military_capital.copy(),
            economic_capital=self.economic_capital.copy(),
            strategy_distribution=self.strategy.copy(),
            ot_allocation=self.ot_allocation.copy(),
            diagnostics=diagnostics,
        )

    def run_trajectory(self, months: int = 900) -> dict[str, np.ndarray | dict]:
        """Advance once per month and retain states needed at an event onset."""
        trajectory = {
            "actor_support": np.empty((months, self.paths, 4)),
            "taiwan_readiness": np.empty((months, self.paths)),
            "arms_support_us": np.empty((months, self.paths)),
            "arms_support_japan": np.empty((months, self.paths)),
            "leader_continuity": np.empty((months, self.paths, 4)),
            "pressure_belief": np.empty((months, self.paths, 3)),
            "military_capital": np.empty((months, self.paths, 4)),
            "economic_capital": np.empty((months, self.paths, 4)),
            "interdependence": np.empty((months, self.paths)),
        }
        for month in range(months):
            self.update(month)
            delivered = self.delivered_capability.mean(axis=1)
            readiness = np.clip(
                0.72 * self.anchor["taiwan_readiness"]
                + 0.18 * self.military_capital[:, 1] + 0.10 * (1.0 + delivered),
                0.55, 1.65,
            )
            donor_pipeline = self.pipeline.sum(axis=(2, 3))
            donor_total = donor_pipeline + self.delivered_by_donor
            donor_support = donor_total / np.maximum(donor_total + 0.55, 1e-12)
            trajectory["actor_support"][month] = self.support
            trajectory["taiwan_readiness"][month] = readiness
            trajectory["arms_support_us"][month] = np.clip(donor_support[:, 0], 0.01, 0.99)
            trajectory["arms_support_japan"][month] = np.clip(donor_support[:, 1], 0.01, 0.99)
            trajectory["leader_continuity"][month] = self.leader_continuity
            trajectory["pressure_belief"][month] = self.pressure_belief
            trajectory["military_capital"][month] = self.military_capital
            trajectory["economic_capital"][month] = self.economic_capital
            trajectory["interdependence"][month] = self.interdependence
        trajectory["diagnostics"] = {
            "pressure_simplex_residual": float(np.max(np.abs(self.pressure_belief.sum(axis=1) - 1.0))),
            "strategy_simplex_residual": self.max_probability_residual,
            "ot_supply_residual": self.max_resource_residual,
            "arms_pipeline_stock_residual": self.max_pipeline_residual,
            "minimum_pipeline_stock": float(np.min(self.pipeline)),
            "growth_rate_range": [float(np.min(self.growth_rate)), float(np.max(self.growth_rate))],
            "economic_capital_range": [float(np.min(self.economic_capital)), float(np.max(self.economic_capital))],
            "scenario": self.scenario,
            "scenario_parameters": {
                key: value.tolist() if isinstance(value, np.ndarray) else value
                for key, value in self.scenario_parameters.items()
            },
        }
        return trajectory
