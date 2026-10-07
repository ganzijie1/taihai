import json
import math
import os
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
PATHS = int(os.environ.get("TAIWAN_V40_PATHS", "6000"))
MONTHS = 180
DT = 1.0 / 12.0
SEED = 20261004
YEARS = (2030, 2050, 2075, 2100)
CASES = ("timely_full", "limited", "delayed", "japan_only", "taiwan_alone")
SIDES = ("china", "united_states", "japan", "taiwan")
NODES = ("bases", "ports", "airfields", "fuel", "munitions", "transport", "c4isr", "industry_repair", "political")
STOCKS = ("precision_munitions", "fuel", "spares")
REGIMES = (
    "neutral", "us_systemic", "japan_systemic", "china_systemic",
    "external_joint_systemic", "bilateral_systemic", "taiwan_political",
    "global_coalition_surge",
)

TARGET_REGIME_PROB = np.array([0.76, 0.03, 0.03, 0.05, 0.04, 0.06, 0.02, 0.01])
PROPOSAL_REGIME_PROB = np.array([0.22, 0.14, 0.14, 0.14, 0.12, 0.16, 0.05, 0.03])
AUX_GROUPS = (
    "indo_pacific_treaty", "european_allies", "regional_access",
    "intelligence_industrial", "sanctions_only_partners",
)
AUX_GROUP_WEIGHTS = np.array([.18, .22, .18, .17, .25])
AUX_GRAPHON = np.array([
    [.48, .14, .20, .12, .06],
    [.13, .52, .08, .17, .10],
    [.20, .08, .50, .12, .10],
    [.14, .18, .12, .48, .08],
    [.08, .14, .12, .10, .56],
])


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -35.0, 35.0)))


def logit(p):
    p = np.clip(p, 1e-8, 1.0 - 1e-8)
    return np.log(p / (1.0 - p))


def weighted_mean(x, w):
    return float(np.sum(x * w) / np.sum(w))


def weighted_quantile(x, w, probs=(0.10, 0.50, 0.90)):
    order = np.argsort(x)
    xs = x[order]
    ws = w[order]
    cdf = (np.cumsum(ws) - 0.5 * ws) / np.sum(ws)
    return [float(np.interp(p, cdf, xs)) for p in probs]


def weighted_cvar(x, w, alpha):
    threshold = weighted_quantile(x, w, (alpha,))[0]
    tail = x >= threshold
    return weighted_mean(x[tail], w[tail]) if np.any(tail) else float(threshold)


def load_inputs():
    v38 = json.loads((OUT / "台海V3.8战时经济生产财政通胀_仿真摘要.json").read_text(encoding="utf-8"))
    v39 = json.loads((OUT / "台海V3.9四方民意平均场与战前军售_仿真摘要.json").read_text(encoding="utf-8"))
    rows38 = {
        (r["conflict_year"], r["case"]): r
        for r in v38["results"] if r["scenario"] == "central"
    }
    # Japan-only intervention was not a V3.8 case. Its baseline is a transparent
    # log-odds interpolation between limited allied intervention and Taiwan
    # alone; V4.0 then evolves US and Japan physical/political states separately.
    for year in YEARS:
        limited = rows38[(year, "limited")]
        alone = rows38[(year, "taiwan_alone")]
        synthetic = dict(limited)
        synthetic["case"] = "japan_only"
        synthetic["v38_broad_control_with_wartime_economy"] = float(sigmoid(
            .35 * logit(limited["v38_broad_control_with_wartime_economy"])
            + .65 * logit(alone["v38_broad_control_with_wartime_economy"])
        ))
        rows38[(year, "japan_only")] = synthetic
    readiness39 = {
        r["year"]: r["taiwan_prewar_readiness_multiplier_p10_p50_p90"][1]
        for r in v39["results"]
    }
    return rows38, readiness39


def simulate_post_conflict_governance(rng, final_nodes, final_opinion, path_probability, importance_weight):
    # This aggregate twenty-year module is conditional on broad control. It
    # models local political intermediaries without equating party identity
    # with willingness to serve or popular acceptance of a sovereignty change.
    conditional_weight = importance_weight * np.clip(path_probability, 1e-8, 1.0)
    scenarios = {
        "credible_autonomy_commitment": (.74, .08),
        "partial_credibility": (.46, .16),
        "low_credibility": (.22, .28),
    }
    results = {}
    for scenario, (initial_credibility, breach_base) in scenarios.items():
        credibility = np.clip(
            initial_credibility + rng.normal(0.0, .045, path_probability.size), .03, .97
        )
        service = np.clip(.45 * final_nodes[:, 3, 7] + .35 * final_nodes[:, 3, 5] + .20, .08, .95)
        support = final_opinion[:, 3, 0]
        cautious = final_opinion[:, 3, 1]
        oppose = final_opinion[:, 3, 2]
        panic = final_opinion[:, 3, 3]
        resistance = np.clip(.28 + .48 * oppose + .24 * cautious + .35 * panic - .12 * support, .08, .92)
        latent_grievance = resistance.copy()
        social_trust = np.clip(.52 - .18 * resistance, .12, .75)
        # Behavioral states: committed cooperation, opportunistic compliance,
        # dual positioning, and exit/refusal. These are not party labels.
        agent_distribution = np.column_stack((
            .08 + .12 * credibility,
            .48 + .04 * credibility,
            .10 + .03 * (1.0 - credibility),
            .34 - .19 * credibility,
        ))
        agent_distribution = np.clip(agent_distribution, .01, None)
        agent_distribution /= agent_distribution.sum(axis=1, keepdims=True)
        snapshots = {}
        for governance_year in range(1, 21):
            representation_gap = np.clip(.62 - .25 * support + .22 * resistance, .18, .86)
            contract_utility = np.column_stack((
                -0.15 + 1.25 * credibility + .45 * service - .55 * resistance,
                .52 + .42 * credibility + .25 * service - .18 * resistance,
                .08 + .48 * resistance + .30 * (1.0 - credibility),
                -.10 + .82 * resistance + .62 * (1.0 - credibility),
            ))
            contract_utility += .34 * np.log(np.clip(agent_distribution, 1e-5, 1.0))
            response = np.exp(contract_utility - contract_utility.max(axis=1, keepdims=True))
            response /= response.sum(axis=1, keepdims=True)
            agent_distribution = .72 * agent_distribution + .28 * response
            agent_effort = (
                agent_distribution[:, 0] + .62 * agent_distribution[:, 1]
                + .18 * agent_distribution[:, 2]
            )
            information_distortion = np.clip(
                .12 + .42 * agent_distribution[:, 1] + .65 * agent_distribution[:, 2]
                + .28 * representation_gap, .05, .85
            )
            household_compliance = np.clip(
                1.0 - resistance - .24 * representation_gap + .20 * credibility, .03, .95
            )
            central_capacity = np.clip(.58 + .30 * service, .10, .95)
            governance_delivery = np.power(
                np.clip(central_capacity * agent_effort * household_compliance, 1e-6, 1.0), 1.0 / 3.0
            ) * np.exp(-.42 * information_distortion)
            breach_probability = np.clip(
                breach_base + .12 * resistance + .07 * (1.0 - service), .01, .65
            )
            policy_breach = rng.random(path_probability.size) < breach_probability
            credibility = np.clip(
                credibility + .055 * (~policy_breach) * governance_delivery
                - .16 * policy_breach - .05 * representation_gap, .01, .99
            )
            coercion = np.clip(.20 + .42 * resistance - .28 * credibility, .02, .72)
            service = np.clip(
                service + .075 * governance_delivery - .055 * resistance
                - .035 * coercion + rng.normal(0.0, .012, path_probability.size), .03, .98
            )
            resistance = np.clip(
                resistance + .10 * representation_gap + .14 * coercion
                + .18 * policy_breach + .08 * information_distortion
                - .20 * governance_delivery - .12 * service - .10 * credibility,
                .01, .99
            )
            social_trust = np.clip(
                social_trust + .06 * service + .05 * credibility
                - .08 * coercion - .08 * information_distortion, .02, .95
            )
            latent_grievance = np.clip(
                latent_grievance + .10 * representation_gap + .12 * policy_breach
                + .06 * coercion + .05 * information_distortion
                - .07 * service - .06 * credibility - .04 * social_trust,
                .01, .99
            )
            if governance_year in (5, 10, 20):
                snapshots[str(governance_year)] = {
                    "resistance_p10_p50_p90": weighted_quantile(resistance, conditional_weight),
                    "latent_grievance_p10_p50_p90": weighted_quantile(
                        latent_grievance, conditional_weight
                    ),
                    "governance_delivery_p10_p50_p90": weighted_quantile(
                        governance_delivery, conditional_weight
                    ),
                    "commitment_credibility_p10_p50_p90": weighted_quantile(
                        credibility, conditional_weight
                    ),
                    "social_trust_p10_p50_p90": weighted_quantile(social_trust, conditional_weight),
                }
        stable = (
            (resistance < .30) & (latent_grievance < .38)
            & (governance_delivery > .50) & (credibility > .40)
        )
        results[scenario] = {
            "conditional_stable_governance_share": weighted_mean(stable.astype(float), conditional_weight),
            "year_snapshots": snapshots,
            "terminal_agent_behavior_mean": {
                state: weighted_mean(agent_distribution[:, state_index], conditional_weight)
                for state_index, state in enumerate(
                    ("committed_cooperation", "opportunistic_compliance", "dual_positioning", "exit_or_refusal")
                )
            },
        }
    return results


def dependency_matrix(side_index):
    # Row i depends on column j. Political state is handled separately but can
    # still receive civilian-system cascade pressure from the physical network.
    w = np.zeros((9, 9))
    w[0, [3, 4, 5, 6, 7]] = [.15, .18, .12, .22, .13]
    w[1, [3, 5, 6, 7]] = [.12, .22, .10, .18]
    w[2, [3, 4, 5, 6, 7]] = [.16, .16, .10, .23, .14]
    w[3, [1, 5, 6, 7]] = [.18, .21, .08, .12]
    w[4, [1, 5, 6, 7]] = [.12, .18, .12, .20]
    w[5, [1, 3, 6, 7]] = [.16, .18, .15, .12]
    w[6, [0, 2, 3, 5, 7]] = [.08, .08, .12, .12, .16]
    w[7, [1, 3, 5, 6]] = [.18, .17, .17, .15]
    w[8, [1, 3, 4, 5, 6]] = [.06, .08, .08, .06, .12]
    if side_index == 0:  # continental depth and land rerouting
        w[:, 1] *= .72
        w[:, 5] *= .80
        w[:, 7] *= .88
    elif side_index == 1:  # US: global depth, but long logistics and access
        w[:, 5] *= 1.18
        w[:, 8] *= 1.10
        w[:, 1] *= .85
    elif side_index == 2:  # Japan: dense bases, ports and domestic politics
        w[:, 1] *= 1.08
        w[:, 3] *= 1.12
        w[:, 8] *= 1.16
    else:  # Taiwan: short internal lines, concentrated infrastructure
        w[:, 1] *= 1.20
        w[:, 2] *= 1.18
        w[:, 6] *= 1.15
    return w


DEPENDENCY = np.stack([dependency_matrix(i) for i in range(4)])


def case_participation(case, month):
    if case == "timely_full":
        return 1.0, 1.0, .18
    if case == "limited":
        return .46, .54, .08
    if case == "delayed":
        return (.08, .15, .03) if month < 3 else (.82, .72, .12)
    if case == "japan_only":
        return .02, .72, .04
    return 0.02, 0.02, .01


def regime_bias(regime_index):
    # Bias is applied to latent shock severity, not directly to outcomes.
    bias = np.zeros((regime_index.size, 4))
    bias[regime_index == 1, 1] = 2.00
    bias[regime_index == 2, 2] = 2.00
    bias[regime_index == 3, 0] = 2.00
    bias[regime_index == 4, 1:3] = 1.70
    bias[regime_index == 5, :] = 1.65
    bias[regime_index == 6, 3] = 0.80
    return bias


def simulate_cell(year, case, base_row, taiwan_prewar_readiness, seed):
    rng = np.random.default_rng(seed)
    n = PATHS
    regime = rng.choice(len(REGIMES), n, p=PROPOSAL_REGIME_PROB)
    importance_weight = TARGET_REGIME_PROB[regime] / PROPOSAL_REGIME_PROB[regime]
    tail_bias = regime_bias(regime)

    initial_nodes = np.array([
        [.92, .91, .91, .90, .88, .93, .91, .90, .82],
        [.93, .90, .92, .91, .91, .84, .94, .92, .80],
        [.89, .87, .89, .84, .86, .88, .90, .87, .77],
        [.86, .80, .82, .78, .78, .86, .84, .77, .78],
    ])
    nodes = np.tile(initial_nodes[None, :, :], (n, 1, 1))
    nodes *= rng.lognormal(0.0, np.array([.035, .038, .043, .050])[None, :, None], nodes.shape)
    nodes[:, 3, :8] *= np.clip(taiwan_prewar_readiness, .85, 1.35)
    nodes = np.clip(nodes, .25, .99)
    repair_queue = np.zeros_like(nodes)

    inventories = np.tile(np.array([
        [1.05, 1.15, 1.00],
        [1.22, 1.32, 1.18],
        [.92, 1.02, .90],
        [.72, .76, .70],
    ])[None, :, :], (n, 1, 1))
    inventories[:, 3, :] *= np.clip(taiwan_prewar_readiness, .85, 1.35)

    production = np.array([
        [.040, .075, .030],
        [.052, .090, .040],
        [.030, .052, .028],
        [.010, .022, .014],
    ])
    baseline_lambda = np.array([.16, .12, .13, .18])[None, :]
    hawkes_lambda = np.repeat(baseline_lambda, n, axis=0)
    hawkes_memory = np.zeros((n, 4))
    cumulative_loss = np.zeros((n, 4))
    cumulative_outage = np.zeros((n, 4))
    min_nodes = nodes.copy()
    max_repair_queue = np.zeros_like(nodes)
    months_systemic = np.zeros((n, 4))
    common_evt_exceedances = np.zeros(n, dtype=np.int16)
    side_evt_exceedances = np.zeros((n, 4), dtype=np.int16)
    political_absorbed = np.zeros((n, 4), dtype=bool)
    first_absorption_month = np.full((n, 4), -1, dtype=np.int16)
    leader_disruption_remaining = np.zeros((n, 4), dtype=np.int16)
    leader_event_count = np.zeros((n, 4), dtype=np.int16)
    leader_type_count = np.zeros((n, 4, 3), dtype=np.int16)
    leader_attempt_count = np.zeros((n, 4), dtype=np.int16)
    leader_attempt_success_count = np.zeros((n, 4), dtype=np.int16)
    leader_voluntary_exit_count = np.zeros((n, 4), dtype=np.int16)
    opinion_initial = np.array([
        [.62, .25, .10, .03], [.42, .34, .20, .04],
        [.45, .34, .17, .04], [.58, .27, .11, .04],
    ])
    opinion_distribution = np.tile(opinion_initial[None, :, :], (n, 1, 1))
    opinion_distribution += rng.normal(0.0, .008, opinion_distribution.shape)
    opinion_distribution = np.clip(opinion_distribution, .002, None)
    opinion_distribution /= opinion_distribution.sum(axis=2, keepdims=True)
    initial_morale = (
        opinion_initial[:, 0] + .35 * opinion_initial[:, 1]
        - opinion_initial[:, 2] - 1.40 * opinion_initial[:, 3]
    )
    opinion_mfg_residual = []
    wartime_proxy_distribution = np.tile(
        np.array([.015, .060, .070, .855])[None, :], (n, 1)
    )
    wartime_proxy_effect = np.zeros(n)
    wartime_local_access = np.zeros(n)
    wartime_proxy_residual = []
    # Partial-Stackelberg three-major-player MFG. The US sends a leading
    # commitment signal; Japan chooses intervention/rescue and Taiwan chooses
    # defensive mobilization as independent major players. Graphon-coupled
    # minor groups react to all three major players and their peers.
    minor_distribution = np.tile(np.array([.42, .36, .19, .03]), (n, len(AUX_GROUPS), 1))
    minor_distribution += rng.normal(0.0, .012, minor_distribution.shape)
    minor_distribution = np.clip(minor_distribution, .002, None)
    minor_distribution /= minor_distribution.sum(axis=2, keepdims=True)
    minor_mfg_residual = []

    # Five-year macro medians from V3.8 inform political-economic pressure.
    actor_metrics = base_row["actor_metrics"]
    inflation_target = np.column_stack((
        np.full(n, actor_metrics["china"]["5"]["effective_inflation_including_shortages_p10_p50_p90"][1]),
        np.full(n, actor_metrics["united_states"]["5"]["effective_inflation_including_shortages_p10_p50_p90"][1]),
        np.full(n, actor_metrics["japan"]["5"]["effective_inflation_including_shortages_p10_p50_p90"][1]),
        np.full(n, actor_metrics["taiwan"]["5"]["effective_inflation_including_shortages_p10_p50_p90"][1]),
    ))
    gdp_target = np.column_stack((
        np.full(n, actor_metrics["china"]["5"]["gdp_index_p10_p50_p90"][1]),
        np.full(n, actor_metrics["united_states"]["5"]["gdp_index_p10_p50_p90"][1]),
        np.full(n, actor_metrics["japan"]["5"]["gdp_index_p10_p50_p90"][1]),
        np.full(n, actor_metrics["taiwan"]["5"]["gdp_index_p10_p50_p90"][1]),
    ))

    # Common, side and node exposures create t-copula-like tail dependence.
    common_loading = np.array([.45, .40, .44, .52])[None, :, None]
    side_loading = np.array([.42, .45, .46, .40])[None, :, None]
    node_exposure = np.array([
        [.82, .70, .76, .58, .64, .50, .70, .44, .12],
        [.68, .58, .68, .52, .50, .68, .62, .46, .14],
        [.76, .70, .76, .68, .60, .64, .68, .54, .16],
        [.88, .86, .90, .82, .78, .74, .86, .70, .18],
    ])[None, :, :]
    cascade_strength = np.array([.080, .086, .095, .120])[None, :, None]
    repair_base = np.array([.014, .018, .015, .012])[None, :, None]

    for month in range(MONTHS):
        base_us_participation, base_japan_participation, auxiliary_signal = case_participation(case, month)
        us_retention = sigmoid(
            3.20 - 1.30 * cumulative_loss[:, 1] - .95 * cumulative_outage[:, 1]
            - 1.15 * political_absorbed[:, 1]
            - .55 * (leader_disruption_remaining[:, 1] > 0)
        ) / sigmoid(3.20)
        us_participation = np.clip(
            base_us_participation * (.75 + .25 * us_retention), .01, 1.0
        )
        taiwan_threat = 1.0 - np.exp(np.mean(np.log(np.clip(nodes[:, 3, :8], .03, 1.0)), axis=1))
        japan_best_response = sigmoid(
            -1.05 + 1.10 * base_japan_participation + .78 * us_participation
            + 1.05 * taiwan_threat - 1.35 * cumulative_loss[:, 2]
            - .80 * cumulative_outage[:, 2] + .60 * (nodes[:, 2, 8] - .55)
            - .60 * (leader_disruption_remaining[:, 2] > 0)
        )
        japan_participation = np.clip(
            .68 * base_japan_participation + .32 * japan_best_response, .01, 1.0
        )
        taiwan_effort = sigmoid(
            1.55 + .42 * us_participation + .30 * japan_participation
            + 1.18 * taiwan_threat - 1.35 * cumulative_loss[:, 3]
            - .88 * cumulative_outage[:, 3] + .72 * (nodes[:, 3, 8] - .55)
            - .72 * (leader_disruption_remaining[:, 3] > 0)
            - .65 * wartime_proxy_effect
        )
        major_signal = (
            .54 * us_participation + .25 * japan_participation
            + .13 * taiwan_effort + auxiliary_signal
        )
        group_major_exposure = np.array([1.12, .92, .84, 1.02, .62])[None, :, None]
        peer_field = np.einsum("gh,nhs->ngs", AUX_GRAPHON, minor_distribution)
        utilities = np.zeros((n, len(AUX_GROUPS), 4))
        utilities[:, :, 0] = .28 - .42 * major_signal[:, None]
        utilities[:, :, 1] = .42 + .70 * major_signal[:, None]
        utilities[:, :, 2] = .10 + 1.05 * major_signal[:, None]
        utilities[:, :, 3] = -.95 + 1.42 * major_signal[:, None]
        utilities += .34 * np.log(np.clip(peer_field, 1e-5, 1.0))
        utilities += group_major_exposure * major_signal[:, None, None] * np.array([-.16, .10, .22, .30])[None, None, :]
        group_cost = np.array([.88, .95, .82, .90, 1.18])[None, :, None]
        utilities -= group_cost * np.array([.00, .08, .22, .62])[None, None, :]
        utilities[:, :, 3] += 1.15 * (regime == 7)[:, None]
        utilities[:, :, 1:3] += .30 * (regime == 7)[:, None, None]
        utilities -= rng.normal(0.0, .06, utilities.shape)
        response = np.exp(utilities - utilities.max(axis=2, keepdims=True))
        response /= response.sum(axis=2, keepdims=True)
        previous_minor = minor_distribution.copy()
        minor_distribution = .78 * minor_distribution + .22 * response
        minor_mfg_residual.append(float(np.mean(np.abs(minor_distribution - previous_minor))))
        aggregate_minor = np.sum(minor_distribution * AUX_GROUP_WEIGHTS[None, :, None], axis=1)
        auxiliary_participation = aggregate_minor @ np.array([.00, .04, .18, .42])
        auxiliary_sanctions = aggregate_minor @ np.array([.00, .65, .82, .95])
        global_sanction_pressure = np.clip(
            .22 * us_participation + .18 * japan_participation
            + .56 * auxiliary_sanctions, 0.0, .88
        )
        political_active = 1.0 - political_absorbed.astype(float)

        stock_fill = np.minimum(inventories, 1.0)
        operational = np.exp(np.mean(np.log(np.clip(nodes[:, :, :8], .03, 1.0)), axis=2))
        stock_readiness = np.exp(np.mean(np.log(np.clip(stock_fill, .03, 1.0)), axis=2))
        readiness = operational * stock_readiness * political_active

        # Wartime contested-governance network. Aggregate local access stands
        # for already controlled or administratively reachable districts; no
        # geographic route or target is represented. Local intermediaries can
        # cooperate, comply opportunistically, dual-position, or refuse/exit.
        relative_local_control = readiness[:, 0] - readiness[:, 3]
        wartime_local_access = sigmoid(
            -3.10 + 1.75 * relative_local_control
            + 1.30 * (1.0 - operational[:, 3]) + .55 * (1.0 - nodes[:, 3, 8])
        )
        wartime_commitment_credibility = np.clip(
            .46 + .18 * nodes[:, 0, 8] - .12 * cumulative_loss[:, 0], .12, .72
        )
        proxy_utility = np.column_stack((
            -1.65 + 1.25 * wartime_local_access + 1.05 * wartime_commitment_credibility,
            .20 + .72 * wartime_local_access + .36 * wartime_commitment_credibility,
            .05 + .48 * wartime_local_access + .52 * (1.0 - wartime_commitment_credibility),
            .92 - .82 * wartime_local_access - .48 * wartime_commitment_credibility,
        ))
        proxy_utility += .30 * np.log(np.clip(wartime_proxy_distribution, 1e-5, 1.0))
        proxy_response = np.exp(proxy_utility - proxy_utility.max(axis=1, keepdims=True))
        proxy_response /= proxy_response.sum(axis=1, keepdims=True)
        previous_proxy = wartime_proxy_distribution.copy()
        wartime_proxy_distribution = .84 * wartime_proxy_distribution + .16 * proxy_response
        wartime_proxy_residual.append(float(np.mean(np.abs(wartime_proxy_distribution - previous_proxy))))
        wartime_proxy_effect = wartime_local_access * (
            wartime_proxy_distribution[:, 0] + .48 * wartime_proxy_distribution[:, 1]
        )
        readiness[:, 0] *= 1.0 + .05 * wartime_proxy_effect
        readiness[:, 3] *= 1.0 - .14 * wartime_proxy_effect

        attack_power = np.zeros((n, 4))
        attack_power[:, 0] = readiness[:, 0]
        attack_power[:, 1] = readiness[:, 1] * us_participation
        attack_power[:, 2] = readiness[:, 2] * japan_participation
        attack_power[:, 3] = readiness[:, 3] * taiwan_effort

        # Hawkes events: attacks and successful disruptions raise short-run
        # intensity; depleted stocks and political absorption suppress it.
        event_probability = 1.0 - np.exp(-np.clip(hawkes_lambda * attack_power, 0.0, 1.8))
        events = rng.random((n, 4)) < event_probability
        hawkes_memory = .68 * hawkes_memory + events
        hawkes_lambda = baseline_lambda + .12 * hawkes_memory

        common_t = rng.standard_t(4, (n, 1, 1))
        side_t = rng.standard_t(5, (n, 4, 1))
        idio_t = rng.standard_t(6, (n, 4, 9))
        latent_severity = common_loading * common_t + side_loading * side_t + .48 * idio_t
        latent_severity += tail_bias[:, :, None]
        severity = np.log1p(np.exp(latent_severity)) / 2.2
        # Peaks-over-threshold EVT marks: Hawkes determines clustered arrival
        # times; a generalized Pareto tail determines exceedance magnitude.
        # A common mark creates cross-node tail dependence, while side marks
        # retain asymmetric local extremes.
        common_tail_probability = .012 + .025 * (regime == 5) + .008 * (regime != 0)
        common_tail = rng.random(n) < common_tail_probability
        u_common = np.clip(rng.random(n), 1e-9, 1.0 - 1e-9)
        xi_common, beta_common = .24, .34
        gpd_common = beta_common / xi_common * (np.power(1.0 - u_common, -xi_common) - 1.0)
        side_tail_probability = .009 + .016 * (tail_bias > .4)
        side_tail = rng.random((n, 4)) < side_tail_probability
        u_side = np.clip(rng.random((n, 4)), 1e-9, 1.0 - 1e-9)
        xi_side, beta_side = .20, .28
        gpd_side = beta_side / xi_side * (np.power(1.0 - u_side, -xi_side) - 1.0)
        severity += common_tail[:, None, None] * gpd_common[:, None, None]
        severity += side_tail[:, :, None] * gpd_side[:, :, None]
        severity = np.clip(severity, .02, 3.8)
        common_evt_exceedances += common_tail
        side_evt_exceedances += side_tail

        incoming_power = np.zeros((n, 4))
        incoming_power[:, 0] = (
            .50 * attack_power[:, 1] + .24 * attack_power[:, 2]
            + .22 * attack_power[:, 3] + .04 * auxiliary_participation
        )
        incoming_power[:, 1] = attack_power[:, 0] * (.28 + .72 * us_participation)
        incoming_power[:, 2] = attack_power[:, 0] * (.34 + .66 * japan_participation)
        incoming_power[:, 3] = 1.12 * attack_power[:, 0]
        incoming_event = np.zeros((n, 4))
        incoming_event[:, 0] = np.maximum(np.maximum(events[:, 1], events[:, 2]), events[:, 3])
        incoming_event[:, 1] = events[:, 0]
        incoming_event[:, 2] = events[:, 0]
        incoming_event[:, 3] = events[:, 0]

        direct_damage = (
            .055 * incoming_event[:, :, None] * incoming_power[:, :, None]
            * node_exposure * severity
        )
        evt_direct_damage = (
            .050 * common_tail[:, None, None] * (.45 + gpd_common[:, None, None])
            + .115 * side_tail[:, :, None] * (.40 + gpd_side[:, :, None])
        ) * node_exposure * (.55 + .45 * incoming_power[:, :, None])
        direct_damage += evt_direct_damage
        # Common natural/cyber/space failures can affect both sides even when a
        # particular strike event is absent.
        common_failure = .0080 * (np.abs(common_t) > 2.6) * severity
        direct_damage += common_failure * np.array([.45, .38, .46, .58])[None, :, None]

        nodes[:, :, :8] = np.clip(nodes[:, :, :8] - direct_damage[:, :, :8], .01, 1.0)
        repair_queue[:, :, :8] += direct_damage[:, :, :8]

        # A directed leadership-disruption action is a two-stage event stream:
        # first choose whether to attempt, then realize success or failure.
        # Voluntary exit is a separate competing risk. Neither branch directly
        # fixes the terminal outcome because succession can restore continuity.
        eligible_leader = (leader_disruption_remaining == 0) & (~political_absorbed)
        target_offset = np.array([.20, .00, .05, .75])[None, :]
        attempt_probability = sigmoid(
            -7.25 + 1.10 * incoming_power + .90 * incoming_event
            + .70 * side_tail + .80 * (1.0 - nodes[:, :, 6]) + target_offset
        )
        leader_attempt = eligible_leader & (rng.random((n, 4)) < attempt_probability)
        success_probability = sigmoid(
            -2.65 + .90 * incoming_power + .70 * incoming_event
            + 1.20 * (1.0 - nodes[:, :, 6]) + .50 * side_tail
            + np.array([.05, .00, .05, .30])[None, :]
        )
        leader_attempt_success = leader_attempt & (rng.random((n, 4)) < success_probability)
        leader_attempt_failed = leader_attempt & (~leader_attempt_success)
        voluntary_exit_probability = sigmoid(
            -9.20 + 2.30 * cumulative_loss + 1.70 * cumulative_outage
            + 1.40 * (1.0 - nodes[:, :, 8])
            + np.array([.00, -.20, -.10, .45])[None, :]
        )
        voluntary_exit = eligible_leader & (~leader_attempt_success) & (
            rng.random((n, 4)) < voluntary_exit_probability
        )
        new_leader_event = leader_attempt_success | voluntary_exit
        leader_attempt_count += leader_attempt
        leader_attempt_success_count += leader_attempt_success
        leader_voluntary_exit_count += voluntary_exit
        type_uniform = rng.random((n, 4))
        leader_type = np.where(
            voluntary_exit, 1,
            np.where(type_uniform < .28, 0, 2)
        )
        duration = np.where(
            leader_type == 0, rng.integers(1, 3, (n, 4)),
            np.where(leader_type == 1, rng.integers(2, 6, (n, 4)), rng.integers(2, 5, (n, 4)))
        )
        leader_disruption_remaining = np.where(
            new_leader_event, duration, np.maximum(leader_disruption_remaining - 1, 0)
        )
        leader_event_count += new_leader_event
        for leader_kind in range(3):
            leader_type_count[:, :, leader_kind] += new_leader_event & (leader_type == leader_kind)
        political_jump = np.choose(leader_type, (.045, .135, .105)) * new_leader_event
        command_jump = np.choose(leader_type, (.085, .065, .120)) * new_leader_event
        nodes[:, :, 8] = np.clip(nodes[:, :, 8] - political_jump, .01, 1.0)
        nodes[:, :, 6] = np.clip(nodes[:, :, 6] - command_jump, .01, 1.0)
        repair_queue[:, :, 6] += command_jump

        # Interdependent-network cascades. This is applied symmetrically, but
        # each side has its own topology and dependency coefficients.
        gap = 1.0 - nodes
        cascade = np.empty_like(nodes)
        for side in range(4):
            cascade[:, side, :] = gap[:, side, :] @ DEPENDENCY[side].T
        cascade_damage = cascade_strength * np.maximum(cascade - .16, 0.0)
        nodes[:, :, :8] = np.clip(nodes[:, :, :8] - cascade_damage[:, :, :8], .01, 1.0)
        repair_queue[:, :, :8] += cascade_damage[:, :, :8]

        # Production, consumption, imports and repair use the same stock-flow
        # equations for all sides. Parameter differences encode depth and access.
        industry = nodes[:, :, 7]
        transport = nodes[:, :, 5]
        port = nodes[:, :, 1]
        prod_factor = industry * (.45 + .55 * transport) * (.55 + .45 * port)
        production_flow = production[None, :, :] * prod_factor[:, :, None]
        production_flow[:, 0, :] *= (1.0 - .22 * global_sanction_pressure[:, None])
        production_flow[:, 1, :] *= (1.0 + .08 * auxiliary_participation[:, None])
        inventories += production_flow
        aid = np.zeros_like(inventories)
        aid[:, 1, :] = .020 * us_participation[:, None] * nodes[:, 1, 1, None]
        aid[:, 1, :] += .008 * auxiliary_participation[:, None]
        aid[:, 2, :] = .014 * japan_participation[:, None] * nodes[:, 2, 1, None]
        aid[:, 3, :] = (
            .016 * us_participation + .011 * japan_participation
            + .006 * auxiliary_participation
        )[:, None] * nodes[:, 3, 1, None]
        inventories += aid
        demand = np.tile(np.array([
            [.052, .058, .035], [.045, .063, .039], [.038, .052, .032], [.033, .040, .025]
        ])[None, :, :], (n, 1, 1))
        demand[:, 1, :] *= us_participation[:, None]
        demand[:, 2, :] *= japan_participation[:, None]
        demand[:, 3, :] *= taiwan_effort[:, None]
        demand = demand * attack_power[:, :, None] * (.80 + .20 * severity[:, :, :3])
        inventories = np.clip(inventories - np.minimum(inventories, demand), .001, 3.0)

        repair_capacity = repair_base * nodes[:, :, 7, None] * (.35 + .65 * stock_fill[:, :, 2, None])
        repair_capacity *= (.45 + .55 * nodes[:, :, 5, None])
        repair_service = np.minimum(repair_queue, repair_capacity)
        repair_queue -= repair_service
        nodes[:, :, :8] = np.clip(nodes[:, :, :8] + repair_service[:, :, :8], .01, 1.0)

        # Casualties and political competing risks affect every side. A major
        # common shock can produce bilateral fatigue rather than one-sided collapse.
        loss_increment = .0022 * incoming_power * incoming_event * np.mean(severity[:, :, :6], axis=2)
        loss_increment += .0012 * np.maximum(.55 - readiness, 0.0)
        cumulative_loss += loss_increment
        physical_outage = np.mean(1.0 - nodes[:, :, :8], axis=2)
        cumulative_outage += physical_outage * DT
        macro_phase = min(month / 60.0, 1.0)
        macro_stress = macro_phase * (inflation_target + np.maximum(1.0 - gdp_target, 0.0))
        attacker_rally = np.zeros((n, 4))
        attacker_discouragement = np.zeros((n, 4))
        dominant_attacker = (1, 0, 0, 0)
        for target_side, attacker_side in enumerate(dominant_attacker):
            attacker_rally[:, attacker_side] += (
                .90 * leader_attempt_success[:, target_side]
                + .45 * voluntary_exit[:, target_side]
            )
            attacker_discouragement[:, attacker_side] += .55 * leader_attempt_failed[:, target_side]
        # Mean-field opinion and morale states: support, cautious, oppose, panic.
        # Failed attempts can rally support; successful disruption and exit can
        # split the population between rallying around succession and panic.
        opinion_utility = np.log(np.clip(opinion_initial, 1e-5, 1.0))[None, :, :]
        opinion_utility = np.tile(opinion_utility, (n, 1, 1))
        succession_credibility = nodes[:, :, 8] * nodes[:, :, 6]
        opinion_utility[:, :, 0] += (
            .55 * nodes[:, :, 8] - .90 * cumulative_loss - .35 * macro_stress
            + .30 * leader_attempt_failed
            + .36 * leader_attempt_success * succession_credibility
            + .10 * voluntary_exit * succession_credibility
            + .28 * attacker_rally - .20 * attacker_discouragement
        )
        opinion_utility[:, :, 1] += .22 * macro_stress + .22 * physical_outage
        opinion_utility[:, :, 2] += (
            .72 * cumulative_loss + .48 * macro_stress
            + .35 * leader_attempt_success * (1.0 - succession_credibility)
            + .52 * voluntary_exit
            + .16 * attacker_discouragement
        )
        opinion_utility[:, :, 3] += (
            1.05 * physical_outage + .82 * cumulative_loss
            + .95 * leader_attempt_success * (1.0 - succession_credibility)
            + 1.20 * voluntary_exit
        )
        opinion_utility[:, 3, 0] -= .26 * wartime_proxy_effect
        opinion_utility[:, 3, 1] += .18 * wartime_proxy_effect
        opinion_utility[:, 3, 3] -= .10 * wartime_proxy_effect
        opinion_utility += .30 * np.log(np.clip(opinion_distribution, 1e-5, 1.0))
        opinion_response = np.exp(opinion_utility - opinion_utility.max(axis=2, keepdims=True))
        opinion_response /= opinion_response.sum(axis=2, keepdims=True)
        previous_opinion = opinion_distribution.copy()
        opinion_distribution = .86 * opinion_distribution + .14 * opinion_response
        opinion_mfg_residual.append(float(np.mean(np.abs(opinion_distribution - previous_opinion))))
        morale_index = (
            opinion_distribution[:, :, 0] + .35 * opinion_distribution[:, :, 1]
            - opinion_distribution[:, :, 2] - 1.40 * opinion_distribution[:, :, 3]
        )
        morale_delta = morale_index - initial_morale[None, :]
        mobilization_support = np.array([.010, .004, .005, .008])[None, :]
        political_drift = mobilization_support - .055 * loss_increment - .006 * physical_outage - .0025 * macro_stress
        political_drift -= .003 * (regime[:, None] == 5)
        political_drift[:, 3] -= .014 * (regime == 6)
        political_drift -= .0028 * (leader_disruption_remaining > 0)
        political_drift += .0050 * morale_delta
        political_drift[:, 3] -= .0040 * wartime_proxy_effect
        political_drift += rng.normal(0.0, .0035, (n, 4))
        nodes[:, :, 8] = np.clip(nodes[:, :, 8] + political_drift, .01, .99)

        collapse_hazard = sigmoid(
            -8.4 + 5.0 * cumulative_loss + 3.2 * physical_outage
            + 2.0 * macro_stress + 2.2 * (1.0 - nodes[:, :, 8])
        )
        collapse_hazard *= np.array([.70, .78, .86, 1.00])[None, :]
        new_absorption = (~political_absorbed) & (rng.random((n, 4)) < collapse_hazard / 12.0)
        first_absorption_month = np.where(new_absorption, month, first_absorption_month)
        political_absorbed |= new_absorption

        systemic = (operational < .48) | (nodes[:, :, 8] < .35) | (stock_readiness < .30)
        months_systemic += systemic
        min_nodes = np.minimum(min_nodes, nodes)
        max_repair_queue = np.maximum(max_repair_queue, repair_queue)

    final_stock_fill = np.minimum(inventories, 1.0)
    physical_readiness = np.exp(np.mean(np.log(np.clip(nodes[:, :, :8], .01, 1.0)), axis=2))
    inventory_readiness = np.exp(np.mean(np.log(np.clip(final_stock_fill, .01, 1.0)), axis=2))
    final_readiness = physical_readiness * inventory_readiness * nodes[:, :, 8]
    persistent_failure = months_systemic >= 6

    china_failure = persistent_failure[:, 0] | (final_readiness[:, 0] < .40)
    us_failure = persistent_failure[:, 1] | (final_readiness[:, 1] < .40) | political_absorbed[:, 1]
    japan_failure = persistent_failure[:, 2] | (final_readiness[:, 2] < .40) | political_absorbed[:, 2]
    taiwan_failure = persistent_failure[:, 3] | (final_readiness[:, 3] < .38) | political_absorbed[:, 3]
    final_us_participation = us_participation
    final_japan_participation = japan_participation
    final_taiwan_effort = taiwan_effort
    final_aggregate_minor = np.sum(minor_distribution * AUX_GROUP_WEIGHTS[None, :, None], axis=1)
    final_auxiliary_path = final_aggregate_minor @ np.array([.00, .04, .18, .42])
    external_failure = (
        (us_failure & (final_us_participation > .15))
        | (japan_failure & (final_japan_participation > .15))
    )
    joint_category = np.full(n, "neither_systemic", dtype=object)
    joint_category[us_failure & ~japan_failure & ~china_failure] = "us_only_failure"
    joint_category[japan_failure & ~us_failure & ~china_failure] = "japan_only_failure"
    joint_category[us_failure & japan_failure & ~china_failure] = "us_japan_failure"
    joint_category[china_failure & ~external_failure] = "china_only_failure"
    joint_category[china_failure & external_failure] = "china_and_external_failure"

    external_weight_sum = (
        .55 * final_us_participation + .28 * final_japan_participation
        + .17 * final_taiwan_effort
    )
    external_effective = (
        .55 * final_us_participation * final_readiness[:, 1]
        + .28 * final_japan_participation * final_readiness[:, 2]
        + .17 * final_taiwan_effort * final_readiness[:, 3]
    ) / external_weight_sum
    tail_advantage = (1.0 - external_effective) - (1.0 - final_readiness[:, 0])
    tail_advantage += .45 * taiwan_failure - .42 * china_failure
    tail_advantage += .14 * us_failure * final_us_participation + .13 * japan_failure * final_japan_participation
    tail_advantage -= .10 * final_auxiliary_path
    tail_advantage += .16 * (nodes[:, 0, 8] - .72) - .12 * (nodes[:, 3, 8] - .68)
    centered_advantage = tail_advantage - weighted_mean(tail_advantage, importance_weight)

    base_probability = float(base_row["v38_broad_control_with_wartime_economy"])
    path_probability = sigmoid(logit(base_probability) + 3.25 * centered_advantage)
    outcome_draw = rng.random(n) < path_probability
    estimate = weighted_mean(path_probability, importance_weight)
    realized = weighted_mean(outcome_draw.astype(float), importance_weight)
    ess = float(np.square(np.sum(importance_weight)) / np.sum(np.square(importance_weight)))
    se = math.sqrt(max(realized * (1.0 - realized), 1e-9) / ess)
    post_conflict_governance = simulate_post_conflict_governance(
        rng, nodes, opinion_distribution, path_probability, importance_weight
    )

    category_metrics = {}
    for category in (
        "neither_systemic", "us_only_failure", "japan_only_failure",
        "us_japan_failure", "china_only_failure", "china_and_external_failure",
    ):
        mask = joint_category == category
        if not np.any(mask):
            category_metrics[category] = {"target_weight_share": 0.0, "broad_control_probability": None}
            continue
        category_metrics[category] = {
            "target_weight_share": float(np.sum(importance_weight[mask]) / np.sum(importance_weight)),
            "broad_control_probability": weighted_mean(path_probability[mask], importance_weight[mask]),
            "china_readiness_median": weighted_quantile(final_readiness[mask, 0], importance_weight[mask], (.50,))[0],
            "us_readiness_median": weighted_quantile(final_readiness[mask, 1], importance_weight[mask], (.50,))[0],
            "japan_readiness_median": weighted_quantile(final_readiness[mask, 2], importance_weight[mask], (.50,))[0],
            "taiwan_readiness_median": weighted_quantile(final_readiness[mask, 3], importance_weight[mask], (.50,))[0],
        }

    regime_metrics = {}
    for r, name in enumerate(REGIMES):
        mask = regime == r
        regime_metrics[name] = {
            "target_prior": float(TARGET_REGIME_PROB[r]),
            "proposal_share": float(np.mean(mask)),
            "broad_control_probability": float(np.mean(path_probability[mask])),
            "realized_event_rate": float(np.mean(outcome_draw[mask])),
        }

    leadership_metrics = {}
    for side_index, side in enumerate(SIDES):
        weighted_attempts = float(np.sum(importance_weight * leader_attempt_count[:, side_index]))
        weighted_successes = float(np.sum(importance_weight * leader_attempt_success_count[:, side_index]))
        leadership_metrics[side] = {
            "any_directed_attempt_share": weighted_mean(
                (leader_attempt_count[:, side_index] > 0).astype(float), importance_weight
            ),
            "any_successful_attempt_share": weighted_mean(
                (leader_attempt_success_count[:, side_index] > 0).astype(float), importance_weight
            ),
            "conditional_success_per_attempt": (
                weighted_successes / weighted_attempts if weighted_attempts > 0 else None
            ),
            "any_voluntary_exit_share": weighted_mean(
                (leader_voluntary_exit_count[:, side_index] > 0).astype(float), importance_weight
            ),
            "event_count_p10_p50_p90": weighted_quantile(
                leader_event_count[:, side_index].astype(float), importance_weight
            ),
            "event_type_mean": {
                event_type: weighted_mean(
                    leader_type_count[:, side_index, event_type_index].astype(float), importance_weight
                )
                for event_type_index, event_type in enumerate(
                    ("temporary_disconnection", "flight_or_exit", "incapacitation_or_death")
                )
            },
        }

    return {
        "conflict_year": year,
        "case": case,
        "base_v38_probability": base_probability,
        "v40_importance_weighted_probability": estimate,
        "importance_weighted_realized_rate": realized,
        "approx_95pct_monte_carlo_interval": [max(0.0, realized - 1.96 * se), min(1.0, realized + 1.96 * se)],
        "effective_sample_size": ess,
        "path_probability_p10_p50_p90": weighted_quantile(path_probability, importance_weight),
        "cvar95_path_probability": weighted_cvar(path_probability, importance_weight, .95),
        "cvar99_path_probability": weighted_cvar(path_probability, importance_weight, .99),
        "wasserstein_lipschitz_upper_epsilon_0_02": min(1.0, estimate + .02),
        "joint_failure_categories": category_metrics,
        "rare_event_regimes": regime_metrics,
        "intervention_intensity": {
            "conditional_case_baseline": {
                "united_states": float(case_participation(case, MONTHS)[0]),
                "japan": float(case_participation(case, MONTHS)[1]),
            },
            "united_states_endogenous_p10_p50_p90": weighted_quantile(
                final_us_participation, importance_weight
            ),
            "japan_partial_stackelberg_response_p10_p50_p90": weighted_quantile(
                final_japan_participation, importance_weight
            ),
            "taiwan_major_player_defense_effort_p10_p50_p90": weighted_quantile(
                final_taiwan_effort, importance_weight
            ),
            "auxiliary_allies_p10_p50_p90": weighted_quantile(final_auxiliary_path, importance_weight),
            "auxiliary_minor_state_mean": {
                state: weighted_mean(final_aggregate_minor[:, state_index], importance_weight)
                for state_index, state in enumerate(("nonparticipation", "sanctions", "logistics_bases", "limited_combat"))
            },
            "graphon_group_limited_combat_mean": {
                group: weighted_mean(minor_distribution[:, group_index, 3], importance_weight)
                for group_index, group in enumerate(AUX_GROUPS)
            },
            "minor_mfg_last_ten_step_residual": float(np.mean(minor_mfg_residual[-10:])),
        },
        "terminal_readiness_p10_p50_p90": {
            side: weighted_quantile(final_readiness[:, i], importance_weight)
            for i, side in enumerate(SIDES)
        },
        "minimum_node_p10_p50_p90": {
            side: {
                node: weighted_quantile(min_nodes[:, i, j], importance_weight)
                for j, node in enumerate(NODES)
            }
            for i, side in enumerate(SIDES)
        },
        "maximum_repair_queue_p10_p50_p90": {
            side: weighted_quantile(np.max(max_repair_queue[:, i, :8], axis=1), importance_weight)
            for i, side in enumerate(SIDES)
        },
        "political_absorption_share": {
            side: weighted_mean(political_absorbed[:, i].astype(float), importance_weight)
            for i, side in enumerate(SIDES)
        },
        "leadership_continuity_event_stream": leadership_metrics,
        "opinion_morale_mean_field": {
            side: {
                "terminal_state_mean": {
                    state: weighted_mean(opinion_distribution[:, side_index, state_index], importance_weight)
                    for state_index, state in enumerate(("support", "cautious", "oppose", "panic"))
                },
                "terminal_morale_p10_p50_p90": weighted_quantile(
                    morale_index[:, side_index], importance_weight
                ),
            }
            for side_index, side in enumerate(SIDES)
        },
        "opinion_mfg_last_ten_step_residual": float(np.mean(opinion_mfg_residual[-10:])),
        "post_conflict_multilevel_principal_agent_governance": post_conflict_governance,
        "wartime_contested_governance_proxy_network": {
            "local_access_share_p10_p50_p90": weighted_quantile(
                wartime_local_access, importance_weight
            ),
            "conditional_on_broad_control_local_access_p10_p50_p90": weighted_quantile(
                wartime_local_access, importance_weight * np.clip(path_probability, 1e-8, 1.0)
            ),
            "effective_proxy_delivery_p10_p50_p90": weighted_quantile(
                wartime_proxy_effect, importance_weight
            ),
            "conditional_on_broad_control_proxy_delivery_p10_p50_p90": weighted_quantile(
                wartime_proxy_effect, importance_weight * np.clip(path_probability, 1e-8, 1.0)
            ),
            "terminal_behavior_mean": {
                state: weighted_mean(wartime_proxy_distribution[:, state_index], importance_weight)
                for state_index, state in enumerate(
                    ("committed_cooperation", "opportunistic_compliance", "dual_positioning", "refusal_or_exit")
                )
            },
            "last_ten_step_residual": float(np.mean(wartime_proxy_residual[-10:])),
            "identification_warning": "Political camp identity is not treated as willingness to accept appointment or as popular acceptance.",
        },
        "evt_exceedance_count_p10_p50_p90": {
            "common": weighted_quantile(common_evt_exceedances.astype(float), importance_weight),
            **{
                side: weighted_quantile(side_evt_exceedances[:, i].astype(float), importance_weight)
                for i, side in enumerate(SIDES)
            },
        },
        "first_political_absorption_month_p50": {
            side: (
                weighted_quantile(first_absorption_month[first_absorption_month[:, i] >= 0, i].astype(float),
                                  importance_weight[first_absorption_month[:, i] >= 0], (.50,))[0]
                if np.any(first_absorption_month[:, i] >= 0) else None
            ) for i, side in enumerate(SIDES)
        },
    }


def validate(results):
    errors = []
    for row in results:
        for key in ("base_v38_probability", "v40_importance_weighted_probability", "importance_weighted_realized_rate",
                    "cvar95_path_probability", "cvar99_path_probability", "wasserstein_lipschitz_upper_epsilon_0_02"):
            if not 0.0 <= row[key] <= 1.0:
                errors.append(f"probability range {row['conflict_year']} {row['case']} {key}")
        shares = sum(x["target_weight_share"] for x in row["joint_failure_categories"].values())
        if abs(shares - 1.0) > 1e-8:
            errors.append(f"joint categories do not sum to one {row['conflict_year']} {row['case']}")
        if row["cvar99_path_probability"] + 1e-12 < row["cvar95_path_probability"]:
            errors.append("CVaR ordering")
        if row["effective_sample_size"] < .25 * PATHS:
            errors.append("importance sampling ESS too low")
    if errors:
        raise AssertionError("; ".join(errors[:12]))
    return "TAIWAN_SYMMETRIC_TAIL_V40_VERIFICATION: PASS"


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    rows38, readiness39 = load_inputs()
    results = []
    run = 0
    selected_year = os.environ.get("TAIWAN_V40_YEAR")
    selected_case = os.environ.get("TAIWAN_V40_CASE")
    for year in YEARS:
        if selected_year and year != int(selected_year):
            continue
        for case in CASES:
            if selected_case and case != selected_case:
                continue
            run += 1
            results.append(simulate_cell(
                year, case, rows38[(year, case)], readiness39[year], SEED + 101 * run
            ))
    verification = validate(results)
    payload = {
        "model": "V4.0 symmetric interdependent networks, t-copula dependence, Hawkes jumps, POT-GPD EVT marks, partial-Stackelberg three-major-player Graphon MFG, opinion-morale MFG, leadership event streams, post-conflict multilevel principal-agent governance, stock-flow repair queues, political competing risks and rare-event importance sampling",
        "paths_per_cell": PATHS,
        "months": MONTHS,
        "sides": SIDES,
        "nodes": NODES,
        "stocks": STOCKS,
        "target_regime_probabilities": dict(zip(REGIMES, TARGET_REGIME_PROB.tolist())),
        "proposal_regime_probabilities": dict(zip(REGIMES, PROPOSAL_REGIME_PROB.tolist())),
        "identification_warning": "Rare-regime target weights are transparent structural priors, not observed frequencies. Equations are symmetric; side-specific topology and parameters are not assumed equal.",
        "results": results,
        "verification": verification,
    }
    out = OUT / "台海V4.0双方对称相关失效与稀有事件_仿真摘要.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"verification": verification, "results": results, "output": str(out)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
