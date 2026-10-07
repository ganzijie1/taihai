import csv
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
CAMPAIGN_PATH = OUT / "台海V3.6_GIS走廊多层备件搜索与军事经济_仿真摘要.json"
ONSET_PATH = OUT / "台海经济军力联合模型_仿真摘要.json"
JSON_PATH = OUT / "台海V3.6联盟响应权重与期望结果.json"
CSV_PATH = OUT / "台海V3.6四类介入概率与期望结果.csv"

SEED = 20261005
PARAMETER_DRAWS = 3000
CRISES_PER_DRAW = 400
YEARS = (2030, 2050, 2075, 2100)
CASES = ("timely_full", "limited", "delayed", "taiwan_alone")


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -35.0, 35.0)))


def quantiles(x):
    return [float(np.quantile(x, p)) for p in (0.05, 0.50, 0.95)]


def load_campaign():
    return json.loads(CAMPAIGN_PATH.read_text(encoding="utf-8"))


def load_onset():
    return json.loads(ONSET_PATH.read_text(encoding="utf-8"))


def campaign_lookup(campaign):
    return {
        (row["conflict_year"], row["case"]): row
        for row in campaign["scenario_envelopes"]
    }


def simulate_year(rng, year, capability_ratio, campaign_rows):
    # First dimension is epistemic parameter uncertainty; the second is the
    # unresolved crisis realization conditional on a major invasion attempt.
    m, k = PARAMETER_DRAWS, CRISES_PER_DRAW

    # Common coalition posture makes participation, strength and delay
    # correlated. It is deliberately broad because no repeated Taiwan-war
    # sample identifies these coefficients.
    posture = rng.normal(0.0, 0.52, (m, 1))
    beta_hit_us = rng.normal(1.30, 0.25, (m, 1))
    beta_hit_jp = rng.normal(0.72, 0.20, (m, 1))
    beta_hold = rng.normal(0.82, 0.20, (m, 1))
    beta_cred = rng.normal(1.10, 0.24, (m, 1))
    beta_nuclear = rng.normal(-1.05, 0.25, (m, 1))
    beta_distraction = rng.normal(-0.82, 0.22, (m, 1))
    participation_intercept = rng.normal(-0.05, 0.42, (m, 1))

    warning = rng.beta(5.0, 2.8, (m, k))
    taiwan_holds = rng.beta(5.0, 2.4 + 0.45 * max(capability_ratio - 0.8, 0.0), (m, k))
    alliance_credibility = rng.beta(7.0, 4.2, (m, k))
    domestic_support = rng.beta(5.0, 3.6, (m, k))
    nuclear_escalation = rng.beta(3.0 + 0.35 * capability_ratio, 4.2, (m, k))
    global_distraction = rng.beta(2.2, 5.0, (m, k))
    # Alliance defense is a joint product: deterrence is partly public, while
    # territorial defense and base protection generate ally-specific benefits.
    # Burden concentration creates a free-riding wedge.
    ally_specific_benefit = rng.beta(4.2, 3.4, (m, k))
    burden_concentration = rng.beta(2.8, 4.6, (m, k))

    # Whether the opening campaign attacks US forces or Japanese territory is
    # endogenous to the attack plan and strongly changes treaty/authorization
    # pathways. These are uncertain scenario nodes, not historical frequencies.
    hit_us_prob = rng.beta(2.2, 5.0, (m, 1))
    hit_japan_prob = rng.beta(1.7, 6.0, (m, 1))
    hit_us = rng.random((m, k)) < hit_us_prob
    hit_japan = rng.random((m, k)) < hit_japan_prob

    capability_cost = max(capability_ratio - 0.8, 0.0)
    eta_participate = (
        participation_intercept
        + posture
        + 0.95  # major invasion rather than a gray-zone episode
        + beta_hit_us * hit_us
        + beta_hit_jp * hit_japan
        + beta_hold * (taiwan_holds - 0.50)
        + beta_cred * (alliance_credibility - 0.50)
        + 0.62 * (domestic_support - 0.50)
        + beta_nuclear * nuclear_escalation
        + beta_distraction * global_distraction
        - 0.28 * capability_cost
        + 0.58 * (ally_specific_benefit - 0.50)
        - 0.46 * burden_concentration
    )
    p_participate = sigmoid(eta_participate)
    participate = rng.random((m, k)) < p_participate

    # Japanese base access and direct combat are separate decisions. The
    # Security Treaty makes US base access institutionally different from a
    # Japanese determination that the event threatens Japan's survival.
    p_base_access = sigmoid(
        0.62 + 0.72 * posture + 1.05 * hit_japan + 0.58 * hit_us
        + 0.72 * (alliance_credibility - 0.50) - 0.72 * nuclear_escalation
    )
    base_access = rng.random((m, k)) < p_base_access
    p_japan_direct = sigmoid(
        -0.92 + 0.70 * posture + 1.65 * hit_japan + 0.48 * hit_us
        + 0.62 * (domestic_support - 0.50) - 0.82 * nuclear_escalation
        + 0.72 * (ally_specific_benefit - 0.50)
        - 0.34 * burden_concentration
    )
    japan_direct = rng.random((m, k)) < p_japan_direct

    # AFT response-time equation. Authorization friction is higher when US
    # forces and Japanese territory have not been attacked.
    delay_noise = rng.normal(0.0, 0.58, (m, k))
    log_delay = (
        np.log(7.0)
        - 0.62 * posture
        - 0.78 * warning
        - 0.72 * hit_us
        - 0.42 * hit_japan
        - 0.34 * base_access
        + 0.55 * global_distraction
        + 0.42 * nuclear_escalation
        + delay_noise
    )
    delay_weeks = np.clip(np.exp(log_delay), 0.5, 52.0)

    intensity_noise = rng.normal(0.0, 0.62, (m, k))
    intensity = sigmoid(
        -0.18 + 0.82 * posture + 0.90 * base_access + 0.58 * japan_direct
        + 0.60 * hit_us + 0.42 * taiwan_holds + 0.45 * domestic_support
        - 0.72 * nuclear_escalation - 0.25 * global_distraction
        + 0.38 * ally_specific_benefit - 0.30 * burden_concentration
        + intensity_noise
    )

    full = participate & base_access & (intensity >= 0.68)
    timely_full = full & (delay_weeks <= 4.0)
    delayed = full & (delay_weeks > 4.0)
    limited = participate & ~full
    taiwan_alone = ~participate

    regimes = np.stack((timely_full, limited, delayed, taiwan_alone), axis=2)
    assert np.all(regimes.sum(axis=2) == 1)
    weights = regimes.mean(axis=1)
    assert np.allclose(weights.sum(axis=1), 1.0)

    central_campaign = np.array([
        campaign_rows[(year, case)]["broad_control_central"] for case in CASES
    ])
    low_campaign = np.array([
        campaign_rows[(year, case)]["broad_control_min"] for case in CASES
    ])
    high_campaign = np.array([
        campaign_rows[(year, case)]["broad_control_max"] for case in CASES
    ])

    expected_central = weights @ central_campaign
    # Draw one coherent adjudication position per parameter draw. A triangular
    # distribution puts more mass near the central campaign calibration while
    # retaining both structural extremes.
    u = rng.triangular(-1.0, 0.0, 1.0, m)
    campaign_draw = np.where(
        u[:, None] < 0.0,
        central_campaign + (-u[:, None]) * (low_campaign - central_campaign),
        central_campaign + u[:, None] * (high_campaign - central_campaign),
    )
    expected_joint = np.sum(weights * campaign_draw, axis=1)

    mean_weights = weights.mean(axis=0)
    central_from_mean = float(mean_weights @ central_campaign)
    assert abs(central_from_mean - float(expected_central.mean())) < 1e-12

    return {
        "conflict_year": year,
        "capability_ratio": capability_ratio,
        "regime_probabilities": {
            case: {
                "mean": float(mean_weights[i]),
                "p05_p50_p95": quantiles(weights[:, i]),
            }
            for i, case in enumerate(CASES)
        },
        "campaign_broad_control_central": {
            case: float(central_campaign[i]) for i, case in enumerate(CASES)
        },
        "expected_broad_control_central_campaign": {
            "mean": float(expected_central.mean()),
            "p05_p50_p95_from_regime_uncertainty": quantiles(expected_central),
        },
        "expected_broad_control_joint_uncertainty": {
            "mean": float(expected_joint.mean()),
            "p05_p50_p95": quantiles(expected_joint),
        },
        "diagnostics": {
            "mean_direct_participation": float(participate.mean()),
            "mean_japan_base_access": float(base_access.mean()),
            "mean_japan_direct_action": float(japan_direct.mean()),
            "mean_ally_specific_benefit": float(ally_specific_benefit.mean()),
            "mean_burden_concentration": float(burden_concentration.mean()),
            "participating_delay_weeks_p05_p50_p95": quantiles(delay_weeks[participate])
            if participate.any() else None,
            "participating_intensity_p05_p50_p95": quantiles(intensity[participate])
            if participate.any() else None,
        },
    }


def interpolate_result(results, year, field, quantile_index=None):
    xs = np.array([row["conflict_year"] for row in results], dtype=float)
    if quantile_index is None:
        ys = np.array([row[field]["mean"] for row in results])
    else:
        ys = np.array([row[field]["p05_p50_p95"][quantile_index] for row in results])
    return float(np.interp(year, xs, ys))


def integrate_onset_and_campaign(results, onset):
    central = next(row for row in onset["results"] if row["scenario"] == "central")
    force_points = {row["year"]: row["force"] for row in central["years"]}
    checkpoints = [2030, 2035, 2040, 2050, 2075, 2100]
    previous_year = 2025
    previous_force = 0.0
    cumulative_mean = 0.0
    cumulative_low = 0.0
    cumulative_high = 0.0
    output = []
    for year in checkpoints:
        force = force_points[year]
        incidence = max(force - previous_force, 0.0)
        midpoint = 2030.0 if year == 2030 else (previous_year + 1.0 + year) / 2.0
        conditional_mean = interpolate_result(
            results, midpoint, "expected_broad_control_joint_uncertainty"
        )
        conditional_low = interpolate_result(
            results, midpoint, "expected_broad_control_joint_uncertainty", 0
        )
        conditional_high = interpolate_result(
            results, midpoint, "expected_broad_control_joint_uncertainty", 2
        )
        cumulative_mean += incidence * conditional_mean
        cumulative_low += incidence * conditional_low
        cumulative_high += incidence * conditional_high
        output.append({
            "horizon": year,
            "cumulative_force_onset": force,
            "new_force_incidence_since_previous_checkpoint": incidence,
            "representative_onset_year": midpoint,
            "eventual_broad_control_within_15y_after_onset_mean": cumulative_mean,
            "structural_low_high_approximation": [cumulative_low, cumulative_high],
        })
        previous_year = year
        previous_force = force
    return output


def write_outputs(results, campaign, onset, unconditional):
    summary = {
        "model": "V3.6 Bayesian alliance response layer + GIS corridors, search, multi-echelon spares and military-economics campaign mixture",
        "conditioning": "major invasion attempt has begun",
        "seed": SEED,
        "parameter_draws": PARAMETER_DRAWS,
        "crises_per_parameter_draw": CRISES_PER_DRAW,
        "regime_partition": {
            "timely_full": "direct participation, Japanese base access, intensity >= 0.68, delay <= 4 weeks",
            "limited": "direct participation but full-intervention conditions not jointly met",
            "delayed": "full-intervention conditions met but delay > 4 weeks",
            "taiwan_alone": "no direct combat participation",
        },
        "identification_warning": (
            "No repeated Taiwan-war outcome sample exists. Coefficients are broad structural priors "
            "anchored to legal decision gates and public studies, not an empirically identified posterior."
        ),
        "results": results,
        "unconditional_eventual_broad_control_by_horizon": unconditional,
        "campaign_input_seed": campaign["seed"],
        "onset_input_seed": onset["seed"],
        "verification": "TAIWAN_ALLIANCE_RESPONSE_V36_VERIFICATION: PASS",
    }
    JSON_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    with CSV_PATH.open("w", encoding="utf-8-sig", newline="") as handle:
        fields = [
            "conflict_year", "timely_full_mean", "limited_mean", "delayed_mean",
            "taiwan_alone_mean", "expected_broad_control_central_mean",
            "expected_broad_control_joint_mean", "expected_broad_control_joint_p05",
            "expected_broad_control_joint_p50", "expected_broad_control_joint_p95",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in results:
            joint = row["expected_broad_control_joint_uncertainty"]
            writer.writerow({
                "conflict_year": row["conflict_year"],
                **{f"{case}_mean": row["regime_probabilities"][case]["mean"] for case in CASES},
                "expected_broad_control_central_mean": row["expected_broad_control_central_campaign"]["mean"],
                "expected_broad_control_joint_mean": joint["mean"],
                "expected_broad_control_joint_p05": joint["p05_p50_p95"][0],
                "expected_broad_control_joint_p50": joint["p05_p50_p95"][1],
                "expected_broad_control_joint_p95": joint["p05_p50_p95"][2],
            })


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    campaign = load_campaign()
    onset = load_onset()
    rows = campaign_lookup(campaign)
    rng = np.random.default_rng(SEED)
    results = []
    for year in YEARS:
        capability_ratio = next(
            row["capability_ratio"] for row in campaign["scenario_envelopes"]
            if row["conflict_year"] == year
        )
        results.append(simulate_year(rng, year, capability_ratio, rows))
    unconditional = integrate_onset_and_campaign(results, onset)
    write_outputs(results, campaign, onset, unconditional)
    print("TAIWAN_ALLIANCE_RESPONSE_V36_VERIFICATION: PASS")
    for row in results:
        w = row["regime_probabilities"]
        joint = row["expected_broad_control_joint_uncertainty"]
        print(
            row["conflict_year"],
            "weights=", ", ".join(f"{case}:{100*w[case]['mean']:.2f}%" for case in CASES),
            f"expected={100*joint['mean']:.2f}%",
            f"p05-p95={100*joint['p05_p50_p95'][0]:.2f}-{100*joint['p05_p50_p95'][2]:.2f}%",
        )
    for row in unconditional:
        print(
            f"by {row['horizon']} unconditional eventual broad control="
            f"{100*row['eventual_broad_control_within_15y_after_onset_mean']:.2f}%"
        )


if __name__ == "__main__":
    main()
