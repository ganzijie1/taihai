from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from public_choice_v85 import FourPartyPublicChoiceSystemV85


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "taiwan_v85_diagnostic_only.json"


def inputs(paths: int, month: int, rng: np.random.Generator) -> dict:
    trend = month / 72.0
    shortage = np.clip(
        0.04 + 0.32 * trend + rng.normal(0.0, 0.012, (paths, 4)), 0.0, 0.8
    )
    damage = np.clip(
        0.01 + 0.20 * trend + rng.normal(0.0, 0.008, (paths, 4)), 0.0, 0.6
    )
    support = np.clip(0.68 - 0.20 * shortage - 0.16 * damage, 0.05, 0.95)
    return {
        "month": month,
        "actor_support": support,
        "actor_damage": damage,
        "actor_shortage": shortage,
        "actor_output": np.clip(0.96 - 0.45 * shortage - 0.30 * damage, 0.1, 1.1),
        "financial_capacity": np.clip(0.88 - 0.30 * shortage, 0.1, 1.0),
        "debt_stress": np.clip(0.12 + 0.50 * shortage, 0.0, 1.2),
        "leader_continuity": np.clip(0.94 - 0.18 * damage, 0.2, 1.0),
        "political_resilience": np.clip(0.82 - 0.25 * shortage, 0.1, 1.0),
        "actor_network": np.clip(0.90 - 0.22 * damage, 0.1, 1.0),
        "actor_panic": np.clip(0.06 + 0.40 * damage, 0.0, 1.0),
        "actor_action": np.clip(0.04 + 0.30 * shortage, 0.0, 1.0),
        "actor_inflation": np.clip(0.02 + 0.16 * shortage, -0.05, 0.8),
        "conflict_intensity": np.full(paths, min(1.0, 0.25 + trend)),
        "intervention_ceiling": np.array([0.85, 0.72]),
        "macro_government_balance": np.clip(-0.05 - 0.25 * shortage, -1.0, 1.0),
    }


def main() -> None:
    paths = 32
    rng = np.random.default_rng(85_003)
    initial_support = np.clip(rng.normal(0.66, 0.035, (paths, 4)), 0.05, 0.95)
    system = FourPartyPublicChoiceSystemV85(paths, 85_007, initial_support)
    neutral = FourPartyPublicChoiceSystemV85(
        paths, 85_007, initial_support, features={"feedback_enabled": False}
    )
    finite = True
    simplex = True
    neutral_exact = True
    actor_heterogeneity = False
    last = None
    for month in range(72):
        state = inputs(paths, month, rng)
        last = system.update(**state)
        neutral_last = neutral.update(**state)
        finite &= all(
            np.all(np.isfinite(value))
            for value in (
                last.budget_shares, last.policy_feasibility,
                last.bureaucratic_execution, last.procurement_factor,
                last.alliance_multiplier, last.aid_authorization,
                last.fiscal_drag,
            )
        )
        simplex &= bool(
            np.max(np.abs(last.budget_shares.sum(axis=2) - 1.0)) <= 1.0e-12
        )
        neutral_exact &= bool(
            np.max(np.abs(neutral_last.bureaucratic_execution - 1.0)) == 0.0
            and np.max(np.abs(neutral_last.procurement_factor - 1.0)) == 0.0
            and np.max(np.abs(neutral_last.alliance_multiplier - 1.0)) == 0.0
            and np.max(np.abs(neutral_last.support_shift)) == 0.0
        )
        actor_heterogeneity |= bool(
            np.max(np.ptp(last.policy_feasibility, axis=1)) > 1.0e-4
        )

    diagnostic = system.diagnostics()
    checks = {
        "monthly_clock_complete": diagnostic["months"] == 72,
        "finite_outputs": finite,
        "budget_simplex_conserved": simplex,
        "probability_bounds": diagnostic["maximum_probability_bound_violation"] <= 1.0e-12,
        "bounded_state_jump": diagnostic["maximum_state_jump"] <= 0.20,
        "positive_execution": diagnostic["minimum_bureaucratic_execution"] > 0.0,
        "feedback_active": diagnostic["maximum_feedback_shift"] > 0.0,
        "actor_heterogeneity": actor_heterogeneity,
        "neutral_ablation_exact": neutral_exact,
        "all_features_registered": all(diagnostic["enabled"].values()),
        "no_future_information": not diagnostic["future_information_used"],
        "no_terminal_probability_adjustment": not diagnostic["terminal_probability_adjustment"],
        "no_duplicate_structural_state": not diagnostic["structural_state_duplication"],
    }
    payload = {
        "classification": "DIAGNOSTIC_ONLY",
        "model_version": "V8.5-FOUR-PARTY-PUBLIC-CHOICE",
        "status": "PASS" if all(checks.values()) else "BLOCKED",
        "checks": checks,
        "diagnostics": diagnostic,
        "substantive_outcomes_generated": False,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("TAIWAN_V85_DIAGNOSTIC_ONLY:", payload["status"])
    if payload["status"] != "PASS":
        raise RuntimeError(str({key: value for key, value in checks.items() if not value}))


if __name__ == "__main__":
    main()
