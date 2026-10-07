import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

N = 100_000
WEEKS = 26
STATE_NAMES = [
    "gray_zone", "isolation_blockade", "limited_clash", "regional_escalation",
    "negotiated_deescalation", "prolonged_stalemate", "humanitarian_economic_instability",
]
LEADER_OUTCOMES = ["execution", "intercepted", "abandoned", "expired"]

SCENARIOS = {
    "baseline_resilience": dict(influence=0.42, verification=0.76, info_resilience=0.78,
                                jump_rate=0.012, jump_scale=0.42, isolation=0.32,
                                barrier=0.68),
    "cognitive_pressure": dict(influence=0.92, verification=0.42, info_resilience=0.48,
                               jump_rate=0.016, jump_scale=0.48, isolation=0.48,
                               barrier=0.60),
    "leader_tail_jump": dict(influence=0.42, verification=0.76, info_resilience=0.78,
                             jump_rate=0.065, jump_scale=0.82, isolation=0.72,
                             barrier=0.42),
    "combined_stress": dict(influence=0.96, verification=0.34, info_resilience=0.38,
                            jump_rate=0.070, jump_scale=0.88, isolation=0.80,
                            barrier=0.34),
    "combined_with_strong_barriers": dict(influence=0.96, verification=0.80,
                                          info_resilience=0.82, jump_rate=0.070,
                                          jump_scale=0.88, isolation=0.42,
                                          barrier=0.86),
    "limited_named_loss_memorial": dict(influence=0.55, verification=0.70,
                                        info_resilience=0.72, jump_rate=0.022,
                                        jump_scale=0.52, isolation=0.42, barrier=0.64,
                                        casualty_mode="limited_memorial"),
    "sustained_losses_falling_success": dict(influence=0.66, verification=0.62,
                                             info_resilience=0.62, jump_rate=0.032,
                                             jump_scale=0.60, isolation=0.55, barrier=0.55,
                                             casualty_mode="sustained"),
    "kinship_salient_civilian_losses": dict(influence=0.62, verification=0.68,
                                            info_resilience=0.66, jump_rate=0.028,
                                            jump_scale=0.56, isolation=0.48, barrier=0.60,
                                            casualty_mode="kinship_civilian"),
    "polarized_nationalist_casualties": dict(influence=0.88, verification=0.45,
                                             info_resilience=0.50, jump_rate=0.048,
                                             jump_scale=0.72, isolation=0.68, barrier=0.43,
                                             casualty_mode="polarized"),
    "casualty_shock_with_resilience": dict(influence=0.72, verification=0.82,
                                           info_resilience=0.84, jump_rate=0.032,
                                           jump_scale=0.60, isolation=0.40, barrier=0.82,
                                           casualty_mode="protected"),
}


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def categorical(rng, probabilities):
    u = rng.random(probabilities.shape[0])
    return (u[:, None] > np.cumsum(probabilities, axis=1)).sum(axis=1)


def simulate(name, p, seed):
    rng = np.random.default_rng(seed)
    state = np.full(N, 1, dtype=np.int8)

    false_intensity = np.full(N, 0.18)
    verified_intensity = np.full(N, 0.30)
    trust = np.full(N, 0.66)
    polarization = np.full(N, 0.31)
    morale = np.full(N, 0.62)
    information_service = np.full(N, p["info_resilience"])

    commitment = rng.normal(-1.10, 0.20, N)
    phase = np.zeros(N, dtype=np.int8)
    leader_outcome = np.full(N, -1, dtype=np.int8)
    execution_pulse = np.zeros(N)
    entered_chain = np.zeros(N, dtype=bool)

    casualty_mode = p.get("casualty_mode", "none")
    casualty_enabled = casualty_mode != "none"
    war_experience = rng.beta(1.8, 5.5, N) if casualty_enabled else np.full(N, 0.25)
    casualty_sensitivity = 0.65 + 0.70 * (1.0 - war_experience)
    death_salience = np.zeros(N)
    defensive_nationalism = np.clip(0.24 + 0.08 * (1.0 - war_experience), 0.0, 1.0)
    revenge_nationalism = np.clip(0.11 + 0.06 * (1.0 - war_experience), 0.0, 1.0)
    kinship = rng.beta(2.2, 2.2, N) if casualty_enabled else np.full(N, 0.50)
    success_belief = np.full(N, 0.62)
    war_support = np.full(N, 0.55)
    cumulative_loss = np.zeros(N)

    weekly = []
    for week in range(WEEKS):
        active_crisis = state < 3
        military_contact = (state == 2).astype(float)

        own_military_loss = np.zeros(N)
        own_civilian_loss = np.zeros(N)
        other_civilian_loss = np.zeros(N)
        memorial = np.zeros(N)
        identity_frame = np.zeros(N)
        if casualty_enabled:
            heterogeneity = rng.lognormal(mean=-0.03, sigma=0.30, size=N)
            if casualty_mode == "limited_memorial" and week == 4:
                own_military_loss = 0.16 * heterogeneity
                memorial[:] = 0.92
            elif casualty_mode == "sustained" and week >= 4:
                own_military_loss = 0.052 * heterogeneity
                own_civilian_loss = 0.018 * heterogeneity * (week >= 8)
                memorial[:] = 0.48
            elif casualty_mode == "kinship_civilian" and week in (6, 7, 8):
                own_civilian_loss = 0.055 * heterogeneity
                other_civilian_loss = 0.095 * heterogeneity
                memorial[:] = 0.35
            elif casualty_mode == "polarized" and week in (4, 9, 14):
                own_military_loss = 0.090 * heterogeneity
                other_civilian_loss = 0.040 * heterogeneity
                memorial[:] = 0.90
                identity_frame[:] = 0.88
            elif casualty_mode == "protected" and week >= 5 and week <= 12:
                own_military_loss = 0.032 * heterogeneity
                own_civilian_loss = 0.012 * heterogeneity
                other_civilian_loss = 0.020 * heterogeneity
                memorial[:] = 0.42
                identity_frame[:] = 0.18

            total_loss = own_military_loss + own_civilian_loss + other_civilian_loss
            cumulative_loss += total_loss
            visibility = np.clip(0.45 + 0.48 * memorial + 0.20 * verified_intensity, 0.0, 1.0)
            personal_cost = own_military_loss + 1.35 * own_civilian_loss + kinship * other_civilian_loss
            death_salience = np.clip(
                0.82 * death_salience + casualty_sensitivity * visibility * personal_cost,
                0.0, 1.0)
            defensive_nationalism = np.clip(
                0.90 * defensive_nationalism
                + 0.34 * memorial * (own_military_loss + 0.65 * own_civilian_loss)
                * (1.0 - defensive_nationalism), 0.0, 1.0)
            revenge_nationalism = np.clip(
                0.84 * revenge_nationalism
                + 0.42 * memorial * (own_military_loss + identity_frame * other_civilian_loss)
                * (1.0 - revenge_nationalism)
                - 0.30 * kinship * other_civilian_loss, 0.0, 1.0)
            success_belief = np.clip(
                success_belief + 0.025 * verified_intensity
                - 0.20 * own_military_loss - 0.27 * own_civilian_loss
                - 0.018 * cumulative_loss + rng.normal(0, 0.006, N), 0.0, 1.0)
            war_support = sigmoid(
                -0.55 + 1.10 * defensive_nationalism + 0.82 * revenge_nationalism
                + 1.15 * success_belief
                - 1.35 * death_salience * (1.0 - success_belief)
                - 1.05 * kinship * other_civilian_loss
                + rng.normal(0, 0.12, N))

        false_shock = rng.gamma(1.4, 0.045, N) * p["influence"]
        event_amplification = 0.10 * military_contact + 0.06 * execution_pulse
        false_intensity = np.clip(
            0.68 * false_intensity + false_shock + event_amplification
            + 0.16 * false_intensity * polarization, 0.0, 1.5)
        verified_intensity = np.clip(
            0.62 * verified_intensity + 0.13 * p["verification"] * information_service
            + rng.normal(0, 0.012, N), 0.0, 1.2)

        disruption = 0.035 * p["influence"] * (0.5 + false_intensity)
        repair = 0.055 * p["info_resilience"] * (1.0 - information_service)
        information_service = np.clip(
            information_service - disruption + repair + rng.normal(0, 0.006, N), 0.05, 1.0)

        evidence_balance = verified_intensity - false_intensity
        trust = np.clip(
            trust + 0.045 * evidence_balance * information_service
            - 0.025 * military_contact + rng.normal(0, 0.008, N), 0.0, 1.0)
        polarization = np.clip(
            polarization + 0.050 * false_intensity * (1.0 - polarization)
            - 0.042 * verified_intensity * polarization + rng.normal(0, 0.007, N), 0.0, 1.0)
        morale = np.clip(
            morale + 0.030 * trust + 0.025 * verified_intensity
            - 0.050 * false_intensity - 0.035 * military_contact
            + rng.normal(0, 0.008, N), 0.0, 1.0)
        coordination = np.clip(
            information_service * (0.42 * trust + 0.34 * morale + 0.24 * (1.0 - polarization)),
            0.0, 1.0)

        bounded = np.clip(commitment, -3.0, 3.0)
        cognitive_distortion = false_intensity * polarization * (1.0 - trust)
        casualty_rally = revenge_nationalism * war_support if casualty_enabled else 0.0
        casualty_aversion = death_salience * (1.0 - success_belief) if casualty_enabled else 0.0
        structural_pressure = (0.13 + 0.22 * military_contact + 0.30 * cognitive_distortion
                               + 0.24 * casualty_rally - 0.20 * casualty_aversion)
        jump = (rng.random(N) < p["jump_rate"] * active_crisis) * np.maximum(
            0.0, rng.standard_t(3, N) * 0.18 + p["jump_scale"])
        commitment += 0.10 * (-bounded ** 3 + 1.08 * bounded + structural_pressure)
        commitment += rng.normal(0, 0.055, N) + jump
        commitment = np.clip(commitment, -3.0, 3.0)

        can_enter = active_crisis & (phase == 0) & (leader_outcome < 0)
        enter = can_enter & (rng.random(N) < 0.20 * sigmoid(2.4 * (commitment - 0.20)))
        phase[enter] = 1
        entered_chain |= enter

        in_chain = active_crisis & (phase >= 1) & (phase <= 4) & (leader_outcome < 0)
        barrier = np.clip(
            p["barrier"] * (0.48 + 0.32 * coordination + 0.20 * verified_intensity)
            * (1.0 - 0.58 * p["isolation"])
            * (1.0 + 0.18 * kinship * other_civilian_loss
               - 0.12 * revenge_nationalism * war_support), 0.01, 0.95)
        advance_hazard = np.clip(
            0.08 + 0.22 * sigmoid(2.2 * (commitment - 0.15))
            + 0.10 * cognitive_distortion + 0.04 * phase, 0.0, 0.62)
        intercept_hazard = np.clip(0.025 + 0.28 * barrier, 0.0, 0.45)
        abandon_hazard = np.clip(
            0.035 + 0.12 * verified_intensity + 0.08 * coordination
            - 0.06 * sigmoid(commitment), 0.01, 0.30)
        total = advance_hazard + intercept_hazard + abandon_hazard
        event_probability = 1.0 - np.exp(-total)
        draw_event = rng.random(N)
        event = in_chain & (draw_event < event_probability)
        choice = rng.random(N) * total
        advance = event & (choice < advance_hazard)
        intercepted = event & ~advance & (choice < advance_hazard + intercept_hazard)
        abandoned = event & ~advance & ~intercepted
        leader_outcome[intercepted] = 1
        leader_outcome[abandoned] = 2
        phase[intercepted | abandoned] = 0
        phase[advance] += 1
        executed = advance & (phase >= 5)
        leader_outcome[executed] = 0
        execution_pulse = np.where(executed, 1.0, 0.62 * execution_pulse)

        for current in (0, 1, 2):
            idx = np.where(state == current)[0]
            if idx.size == 0:
                continue
            if current == 0:
                base = np.array([0.915, 0.025, 0.005, 0.0, 0.045, 0.010, 0.0])
            elif current == 1:
                base = np.array([0.0, 0.780, 0.070, 0.015, 0.080, 0.045, 0.010])
            else:
                base = np.array([0.0, 0.080, 0.660, 0.045, 0.100, 0.080, 0.035])
            probs = np.tile(base, (idx.size, 1))
            risk = np.clip(
                1.60 * false_intensity[idx] * polarization[idx]
                + 1.20 * (0.55 - coordination[idx])
                + 1.25 * execution_pulse[idx]
                + 0.72 * revenge_nationalism[idx] * war_support[idx]
                - 0.88 * death_salience[idx] * (1.0 - success_belief[idx])
                - 0.55 * kinship[idx] * other_civilian_loss[idx], 0.0, 3.0)
            stabilizer = np.clip(
                verified_intensity[idx] * trust[idx] * information_service[idx], 0.0, 1.0)
            probs[:, 2] *= np.exp(0.52 * risk - 0.35 * stabilizer)
            probs[:, 3] *= np.exp(0.75 * risk - 0.55 * stabilizer)
            probs[:, 4] *= np.exp(-0.42 * risk + 0.48 * stabilizer)
            probs[:, 5] *= np.exp(0.18 * risk)
            probs[:, 4] *= np.exp(0.72 * death_salience[idx] * (1.0 - success_belief[idx])
                                     + 0.45 * kinship[idx] * other_civilian_loss[idx])
            probs[:, 6] *= np.exp(0.55 * risk + 0.45 * (1.0 - information_service[idx])
                                     + 0.80 * own_civilian_loss[idx] + 0.20 * cumulative_loss[idx])
            probs /= probs.sum(axis=1, keepdims=True)
            state[idx] = categorical(rng, probs)

        weekly.append({
            "week": week + 1,
            "trust": float(trust.mean()),
            "polarization": float(polarization.mean()),
            "morale": float(morale.mean()),
            "information_service": float(information_service.mean()),
            "coordination": float(coordination.mean()),
            "false_intensity": float(false_intensity.mean()),
            "verified_intensity": float(verified_intensity.mean()),
            "active_leader_chain": float(np.mean((phase >= 1) & (phase <= 4))),
            "death_salience": float(death_salience.mean()),
            "defensive_nationalism": float(defensive_nationalism.mean()),
            "revenge_nationalism": float(revenge_nationalism.mean()),
            "success_belief": float(success_belief.mean()),
            "war_support": float(war_support.mean()),
            "cumulative_loss_index": float(cumulative_loss.mean()),
            "war_experience": float(war_experience.mean()),
            "casualty_sensitivity": float(casualty_sensitivity.mean()),
        })

    leader_outcome[leader_outcome < 0] = 3
    state_dist = np.bincount(state, minlength=len(STATE_NAMES)) / N
    leader_dist = np.bincount(leader_outcome, minlength=len(LEADER_OUTCOMES)) / N
    return {
        "scenario": name,
        "trajectories": N,
        "weeks": WEEKS,
        "state_distribution": dict(zip(STATE_NAMES, state_dist.tolist())),
        "leader_process": dict(zip(LEADER_OUTCOMES, leader_dist.tolist())),
        "entered_leader_chain": float(entered_chain.mean()),
        "final_cognitive_state": weekly[-1],
        "weekly": weekly,
    }


def main():
    results = [simulate(name, p, 20260930 + i) for i, (name, p) in enumerate(SCENARIOS.items())]
    rows = []
    for result in results:
        row = {"scenario": result["scenario"]}
        row.update(result["state_distribution"])
        row.update({f"leader_{k}": v for k, v in result["leader_process"].items()})
        row.update({f"final_{k}": v for k, v in result["final_cognitive_state"].items() if k != "week"})
        rows.append(row)
    with (OUT / "台海认知战与领导突变_情景结果.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "model": "multilayer cognitive-information dynamics with finite leader jump gates",
        "warning": "Synthetic conditional stress test; not a forecast or estimate of any current leader.",
        "results": results,
    }
    (OUT / "台海认知战与领导突变_仿真摘要.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    by_name = {r["scenario"]: r for r in results}
    base = by_name["baseline_resilience"]
    combined = by_name["combined_stress"]
    protected = by_name["combined_with_strong_barriers"]
    memorial = by_name["limited_named_loss_memorial"]
    sustained = by_name["sustained_losses_falling_success"]
    kinship_case = by_name["kinship_salient_civilian_losses"]
    polarized = by_name["polarized_nationalist_casualties"]
    casualty_protected = by_name["casualty_shock_with_resilience"]
    assert abs(sum(base["state_distribution"].values()) - 1.0) < 1e-12
    assert abs(sum(combined["leader_process"].values()) - 1.0) < 1e-12
    assert combined["final_cognitive_state"]["trust"] < base["final_cognitive_state"]["trust"]
    assert combined["final_cognitive_state"]["polarization"] > base["final_cognitive_state"]["polarization"]
    assert combined["state_distribution"]["regional_escalation"] > base["state_distribution"]["regional_escalation"]
    assert protected["leader_process"]["execution"] < combined["leader_process"]["execution"]
    assert protected["state_distribution"]["regional_escalation"] < combined["state_distribution"]["regional_escalation"]
    assert memorial["weekly"][4]["revenge_nationalism"] > memorial["weekly"][3]["revenge_nationalism"]
    assert sustained["final_cognitive_state"]["success_belief"] < memorial["final_cognitive_state"]["success_belief"]
    assert kinship_case["state_distribution"]["negotiated_deescalation"] > polarized["state_distribution"]["negotiated_deescalation"]
    assert casualty_protected["state_distribution"]["regional_escalation"] < polarized["state_distribution"]["regional_escalation"]
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("TAIWAN_COGNITIVE_JUMP_VERIFICATION: PASS")


if __name__ == "__main__":
    main()
