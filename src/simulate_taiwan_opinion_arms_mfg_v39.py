import json
import math
import os
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
PATHS = int(os.environ.get("TAIWAN_V39_PATHS", "20000"))
SEED = 20261003
START_YEAR = 2026
END_YEAR = 2100
TARGET_YEARS = (2030, 2050, 2075, 2100)
ACTORS = ("china", "united_states", "japan", "taiwan")
CASES = ("timely_full", "limited", "delayed", "taiwan_alone")
CAPABILITIES = ("air_defense", "anti_ship", "long_range_fires", "isr_c2", "platforms_spares")


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -35.0, 35.0)))


def logit(p):
    p = np.clip(p, 1e-5, 1.0 - 1e-5)
    return np.log(p / (1.0 - p))


def quantiles(x):
    return [float(v) for v in np.quantile(x, (0.10, 0.50, 0.90))]


def softmax(scores):
    scores = scores - np.max(scores, axis=1, keepdims=True)
    exp_scores = np.exp(scores)
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)


def load_inputs():
    v37 = json.loads((OUT / "台海V3.7因果_演化博弈_竞争风险_联合仿真摘要.json").read_text(encoding="utf-8"))
    v38 = json.loads((OUT / "台海V3.8战时经济生产财政通胀_仿真摘要.json").read_text(encoding="utf-8"))
    central37 = next(row for row in v37["results"] if row["scenario"] == "central")
    years37 = {row["year"]: row for row in central37["years"]}
    central38 = {
        (row["conflict_year"], row["case"]): row
        for row in v38["results"] if row["scenario"] == "central"
    }
    return years37, central38


def nearest_v37(years37, year):
    return years37[min(years37, key=lambda y: abs(y - year))]


def simulate():
    rng = np.random.default_rng(SEED)
    years37, central38 = load_inputs()
    n = PATHS

    # Four types per actor.  The entries are priors on support for direct
    # wartime action, not approval of deterrence, arms aid or threat perception.
    type_names = {
        "china": ("nationalist", "pragmatic_urban", "economically_vulnerable", "security_linked"),
        "united_states": ("republican", "democratic", "independent", "defense_industry_regions"),
        "japan": ("ldp_conservative", "centrist_opposition", "okinawa_frontline", "business_export"),
        "taiwan": ("dpp_leaning", "kmt_leaning", "nonaligned_tpp", "military_age_high_exposure"),
    }
    weights = np.array([
        [0.28, 0.30, 0.27, 0.15],
        [0.31, 0.31, 0.28, 0.10],
        [0.32, 0.31, 0.17, 0.20],
        [0.34, 0.27, 0.24, 0.15],
    ])
    direct_priors = np.array([
        [0.76, 0.57, 0.44, 0.70],
        [0.48, 0.40, 0.37, 0.58],
        [0.18, 0.10, 0.07, 0.15],
        [0.74, 0.50, 0.58, 0.68],
    ])
    latent = np.tile(logit(direct_priors)[None, :, :], (n, 1, 1))
    latent += rng.normal(0.0, 0.32, latent.shape)
    baseline_latent = logit(direct_priors)[None, :, :]

    # Graphon blocks approximate within-actor media and social-network exposure.
    graphons = np.array([
        [[0.48, .18, .12, .22], [.15, .48, .25, .12], [.10, .28, .52, .10], [.25, .10, .08, .57]],
        [[.46, .14, .22, .18], [.13, .48, .24, .15], [.20, .22, .48, .10], [.20, .16, .10, .54]],
        [[.50, .20, .12, .18], [.20, .46, .20, .14], [.16, .22, .52, .10], [.18, .16, .08, .58]],
        [[.52, .12, .22, .14], [.14, .52, .24, .10], [.23, .22, .46, .09], [.16, .12, .10, .62]],
    ])

    threat = np.tile(np.array([0.62, 0.48, 0.68, 0.72]), (n, 1))
    economic_anxiety = np.tile(np.array([0.38, 0.34, 0.40, 0.42]), (n, 1))
    filtered_poll = np.sum(sigmoid(latent) * weights[None, :, :], axis=2)
    party_state = rng.binomial(1, np.array([0.88, 0.50, 0.70, 0.55]), (n, 4))
    military_industry_pressure = np.tile(np.array([0.44, 0.70, 0.48, 0.36]), (n, 1))

    # Arms support is broader than willingness to enter combat.  The US direct
    # pipeline and Japan indirect/dual-use pipeline are explicitly separated.
    us_pipeline = np.zeros((n, 6, len(CAPABILITIES)))
    jp_pipeline = np.zeros_like(us_pipeline)
    us_pipeline[:, :, :] = np.array([0.12, 0.13, 0.11, 0.09, 0.08])[None, None, :]
    jp_pipeline[:, :, :] = np.array([0.010, 0.008, 0.006, 0.025, 0.030])[None, None, :]
    taiwan_capability_stock = np.ones((n, len(CAPABILITIES)))
    absorption = np.tile(np.array([0.72, 0.74, 0.68, 0.78, 0.70]), (n, 1))
    arms_backlog = np.zeros((n, len(CAPABILITIES)))

    snapshots = {}
    equilibrium_residuals = []
    for year in range(START_YEAR, END_YEAR + 1):
        support = sigmoid(latent)
        mean_support = np.sum(support * weights[None, :, :], axis=2)

        common_security = rng.normal(0.0, 0.15, (n, 1))
        regional_security = rng.normal(0.0, 0.10, (n, 1))
        crisis_jump = rng.random((n, 1)) < (0.035 + 0.00035 * (year - START_YEAR))
        crisis_size = crisis_jump * rng.gamma(1.8, 0.16, (n, 1))
        threat_target = np.array([0.62, 0.50, 0.70, 0.72])[None, :]
        threat += 0.16 * (threat_target - threat) + 0.08 * common_security
        threat += crisis_size * np.array([0.62, 0.52, 0.70, 0.78])[None, :]
        threat += regional_security * np.array([0.15, 0.10, 0.24, 0.28])[None, :]
        threat = np.clip(threat, 0.05, 1.40)

        global_cycle = rng.normal(0.0, 0.10, (n, 1))
        economic_anxiety += 0.20 * (np.array([.40, .34, .39, .42])[None, :] - economic_anxiety)
        economic_anxiety += 0.06 * global_cycle + 0.05 * crisis_size
        economic_anxiety = np.clip(economic_anxiety, 0.05, 1.20)

        # Elections update major-player control; parties and governments are
        # major players, while citizens are the minor-player populations.
        if (year - START_YEAR) % 4 == 0 and year > START_YEAR:
            incumbent_score = 0.55 * filtered_poll - 0.45 * economic_anxiety + rng.normal(0.0, .18, (n, 4))
            persistence = np.array([1.10, .55, .92, .64])[None, :]
            party_state = rng.random((n, 4)) < sigmoid(logit(.50) + persistence * (party_state - .5) + incumbent_score)

        major_stance = np.column_stack((
            0.62 + 0.24 * party_state[:, 0] + 0.16 * threat[:, 0],
            0.36 + 0.18 * party_state[:, 1] + 0.22 * military_industry_pressure[:, 1] + 0.18 * threat[:, 1],
            0.18 + 0.20 * party_state[:, 2] + 0.24 * threat[:, 2] - 0.12 * economic_anxiety[:, 2],
            0.42 + 0.25 * party_state[:, 3] + 0.24 * threat[:, 3] - 0.10 * economic_anxiety[:, 3],
        ))

        # Noisy polls observe different manifestations of a latent disposition.
        observed_poll = np.clip(mean_support + rng.normal(0.0, .035, mean_support.shape), .01, .99)
        filtered_poll = 0.72 * filtered_poll + 0.28 * observed_poll

        previous_support = support.copy()
        for actor in range(4):
            local_mean = support[:, actor, :] @ graphons[actor].T
            peer_gap = local_mean - support[:, actor, :]
            type_threat = np.array([1.08, .92, .80, 1.18])[None, :]
            type_cost = np.array([.72, .92, 1.22, .76])[None, :]
            drift = 0.14 * (baseline_latent[:, actor, :] - latent[:, actor, :])
            drift += 0.52 * peer_gap
            drift += 0.34 * (major_stance[:, actor, None] - .50)
            drift += 0.28 * (threat[:, actor, None] - .50) * type_threat
            drift -= 0.30 * economic_anxiety[:, actor, None] * type_cost
            if actor == 0:
                drift += np.array([.12, .02, -.04, .10])[None, :]
            elif actor == 1:
                drift += np.array([.02, -.01, -.04, .11])[None, :]
            elif actor == 2:
                drift += np.array([.06, -.02, -.10, -.03])[None, :]
            else:
                drift += np.array([.10, -.05, .00, -.02])[None, :]
            latent[:, actor, :] += drift + rng.normal(0.0, .11, (n, 4)) + 0.08 * common_security
        latent = np.clip(latent, -4.8, 4.8)
        equilibrium_residuals.append(float(np.mean(np.abs(sigmoid(latent) - previous_support))))

        direct_support = np.sum(sigmoid(latent) * weights[None, :, :], axis=2)
        arms_support_us = sigmoid(logit(np.clip(direct_support[:, 1], .01, .99)) + 0.82)
        arms_support_jp = sigmoid(logit(np.clip(direct_support[:, 2], .01, .99)) + 2.30)

        # Notifications/appropriations enter stage 0.  Each later stage has a
        # category-specific delivery hazard.  Undelivered quantities remain in
        # a backlog rather than vanishing.
        demand_mix = np.array([1.20, 1.25, 1.08, 1.10, .82])[None, :]
        us_commitment = 0.055 * arms_support_us[:, None] * (0.65 + threat[:, 1, None]) * demand_mix
        us_commitment *= 0.75 + 0.38 * military_industry_pressure[:, 1, None]
        jp_commitment = (
            0.012 * arms_support_jp[:, None] * (0.60 + threat[:, 2, None])
            * np.array([.45, .30, .25, 1.20, 1.35])[None, :]
        )
        us_pipeline[:, 0, :] += us_commitment
        jp_pipeline[:, 0, :] += jp_commitment

        stage_hazard_us = np.array([.82, .70, .55, .47, .42])[None, :]
        stage_hazard_jp = np.array([.68, .58, .50, .62, .66])[None, :]
        delivered_us = us_pipeline[:, -1, :] * stage_hazard_us
        delivered_jp = jp_pipeline[:, -1, :] * stage_hazard_jp
        arms_backlog += us_pipeline[:, -1, :] * (1.0 - stage_hazard_us)
        arms_backlog += jp_pipeline[:, -1, :] * (1.0 - stage_hazard_jp)
        for stage in range(5, 0, -1):
            us_pipeline[:, stage, :] = us_pipeline[:, stage - 1, :]
            jp_pipeline[:, stage, :] = jp_pipeline[:, stage - 1, :]
        us_pipeline[:, 0, :] = 0.0
        jp_pipeline[:, 0, :] = 0.0
        backlog_release = 0.16 * arms_backlog * (0.75 + 0.25 * arms_support_us[:, None])
        arms_backlog -= backlog_release
        delivered = delivered_us + delivered_jp + backlog_release
        training_gain = 0.10 * (1.0 - absorption) + 0.05 * direct_support[:, 3, None]
        absorption = np.clip(absorption + training_gain, .45, .96)
        taiwan_capability_stock *= np.array([.965, .960, .955, .950, .945])[None, :]
        taiwan_capability_stock += delivered * absorption
        taiwan_capability_stock = np.clip(taiwan_capability_stock, .35, 2.40)

        if year in TARGET_YEARS:
            capability_weights = np.array([.25, .26, .17, .20, .12])
            readiness = np.exp(np.sum(np.log(taiwan_capability_stock) * capability_weights[None, :], axis=1))
            snapshots[year] = {
                "direct_support": direct_support.copy(),
                "arms_support_us": arms_support_us.copy(),
                "arms_support_jp": arms_support_jp.copy(),
                "filtered_poll": filtered_poll.copy(),
                "threat": threat.copy(),
                "economic_anxiety": economic_anxiety.copy(),
                "readiness": readiness,
                "capability_stock": taiwan_capability_stock.copy(),
                "backlog": arms_backlog.copy(),
            }

    results = []
    for year in TARGET_YEARS:
        snap = snapshots[year]
        support = snap["direct_support"]
        old = nearest_v37(years37, year)
        old_weights = np.array([
            old["alliance_strategy_mean"]["timely_full"],
            old["alliance_strategy_mean"]["limited"],
            old["alliance_strategy_mean"]["delayed_full"],
            old["alliance_strategy_mean"]["noncombat"],
        ])
        scores = np.log(np.maximum(old_weights[None, :], 1e-5))
        scores = np.repeat(scores, n, axis=0)
        scores[:, 0] += 1.75 * (support[:, 1] - .43) + 1.55 * (support[:, 2] - .11)
        scores[:, 1] += 1.20 * (snap["arms_support_us"] - .63) + .55 * (snap["arms_support_jp"] - .55)
        scores[:, 2] += 1.10 * (support[:, 1] - .43) + .85 * (support[:, 2] - .11)
        scores[:, 3] -= .85 * (support[:, 1] - .43) + .60 * (support[:, 2] - .11)
        alliance_weights = softmax(scores)

        case_probabilities = np.zeros((n, 4))
        for case_index, case in enumerate(CASES):
            base = central38[(year, case)]["v38_broad_control_with_wartime_economy"]
            political_shift = 0.32 * (support[:, 0] - .60) - 0.48 * (support[:, 3] - .60)
            if case == "timely_full":
                political_shift -= .36 * (support[:, 1] - .43) + .30 * (support[:, 2] - .11)
            elif case == "delayed":
                political_shift -= .24 * (support[:, 1] - .43) + .18 * (support[:, 2] - .11)
            elif case == "limited":
                political_shift -= .12 * (snap["arms_support_us"] - .63)
            political_shift -= .58 * np.log(np.maximum(snap["readiness"], .35))
            case_probabilities[:, case_index] = sigmoid(logit(base) + political_shift)

        weighted_conditional = np.sum(alliance_weights * case_probabilities, axis=1)
        force_onset = float(old["force_onset"])
        results.append({
            "year": year,
            "force_onset_from_v37": force_onset,
            "old_v38_weighted_conditional_broad_control": float(sum(
                old_weights[i] * central38[(year, case)]["v38_broad_control_with_wartime_economy"]
                for i, case in enumerate(CASES)
            )),
            "v39_weighted_conditional_broad_control_p10_p50_p90": quantiles(weighted_conditional),
            "v39_unconditional_broad_control_p10_p50_p90": quantiles(force_onset * weighted_conditional),
            "alliance_case_probability_mean": {
                case: float(np.mean(alliance_weights[:, i])) for i, case in enumerate(CASES)
            },
            "direct_war_support_p10_p50_p90": {
                actor: quantiles(support[:, i]) for i, actor in enumerate(ACTORS)
            },
            "arms_or_indirect_support_p10_p50_p90": {
                "united_states": quantiles(snap["arms_support_us"]),
                "japan": quantiles(snap["arms_support_jp"]),
            },
            "taiwan_prewar_readiness_multiplier_p10_p50_p90": quantiles(snap["readiness"]),
            "taiwan_capability_stock_median": {
                cap: float(np.median(snap["capability_stock"][:, i])) for i, cap in enumerate(CAPABILITIES)
            },
            "arms_backlog_index_median": {
                cap: float(np.median(snap["backlog"][:, i])) for i, cap in enumerate(CAPABILITIES)
            },
        })

    payload = {
        "model": "V3.9 multi-population major-minor graphon partially observed MFG-of-controls with common noise, optimal stopping proxy, and prewar arms-delivery pipeline",
        "paths": PATHS,
        "years": [START_YEAR, END_YEAR],
        "actors": ACTORS,
        "population_types": type_names,
        "capabilities": CAPABILITIES,
        "poll_measurement_warning": "Threat perception, arms-aid support, willingness to fight, and direct-intervention support are distinct observables. Survey house effects and hypothetical-bias corrections remain uncertain.",
        "china_poll_warning": "Open, repeated, nationally representative direct-war-willingness data are not comparable to US/Japan/Taiwan series; China priors therefore remain wider and are linked to the pre-existing trust/casualty model.",
        "japan_sales_warning": "The baseline treats Japan as indirect logistics, components, dual-use and training support; it does not assert a historical large direct arms-sales channel to Taiwan.",
        "public_data_sources": {
            "taiwan_identity_party_status_quo": "NCCU Election Study Center core political attitude trends",
            "taiwan_willingness_and_alliance_expectations": "Taiwan INDSR survey waves and item wording",
            "us_troops_arms_sanctions": "Chicago Council Taiwan survey series; question-specific estimates",
            "japan_threat_and_response": "Japan Cabinet Office/MOD defense surveys and response-option polls",
            "us_arms_pipeline": "DSCA Major Arms Sales Library and Historical Sales Book, contracts and delivery reports",
            "military_industry": "US budget, contract-award and production-capacity public series",
        },
        "equilibrium_diagnostics": {
            "mean_one_step_population_residual": float(np.mean(equilibrium_residuals[-10:])),
            "max_one_step_population_residual": float(np.max(equilibrium_residuals[-10:])),
            "note": "Particle quantal-response approximation; these are population consistency diagnostics, not a proof of an exact HJB-FP equilibrium.",
        },
        "results": results,
        "verification": "TAIWAN_OPINION_ARMS_MFG_V39: PASS",
    }
    return payload


def validate(payload):
    for row in payload["results"]:
        weights = row["alliance_case_probability_mean"]
        if abs(sum(weights.values()) - 1.0) > 1e-8:
            raise AssertionError("alliance probabilities do not sum to one")
        for key in ("v39_weighted_conditional_broad_control_p10_p50_p90", "v39_unconditional_broad_control_p10_p50_p90"):
            q = row[key]
            if not (0.0 <= q[0] <= q[1] <= q[2] <= 1.0):
                raise AssertionError(f"invalid probability quantiles: {key}")
        if min(row["taiwan_prewar_readiness_multiplier_p10_p50_p90"]) <= 0.0:
            raise AssertionError("nonpositive readiness")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    payload = simulate()
    validate(payload)
    path = OUT / "台海V3.9四方民意平均场与战前军售_仿真摘要.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"verification": payload["verification"], "results": payload["results"], "path": str(path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
