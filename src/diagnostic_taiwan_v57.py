"""DIAGNOSTIC_ONLY: invariant and directional checks for V5.7."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from wartime_collective_action_regime_v57 import (
    WartimeCollectiveActionRegimeSystem,
)


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "outputs" / "taiwan_v57_social_regime_diagnostic.json"


def inputs(paths: int, severity: float) -> dict:
    shape = (paths, 4)
    return {
        "support": np.full(shape, 0.68 - 0.12 * severity),
        "action_seed": np.full(shape, 0.08 + 0.18 * severity),
        "network_action": np.full(shape, 0.09 + 0.16 * severity),
        "actor_network": np.full(shape, 0.87 - 0.12 * severity),
        "actor_norm": np.full(shape, 0.76 - 0.16 * severity),
        "actor_panic": np.full(shape, 0.08 + 0.25 * severity),
        "shortage": np.full(shape, 0.08 + 0.55 * severity),
        "inflation": np.full(shape, 0.025 + 0.18 * severity),
        "output": np.full(shape, 0.96 - 0.50 * severity),
        "damage": np.full(shape, 0.02 + 0.38 * severity),
        "financial_stress": np.full(shape, 0.10 + 0.58 * severity),
        "fiscal_capacity": np.full(shape, 0.90 - 0.48 * severity),
        "progress": np.full(shape, 0.08 - 0.50 * severity),
        "command": np.full(shape, 0.92 - 0.42 * severity),
        "leader_continuity": np.full(shape, 0.94 - 0.34 * severity),
        "social_mobilization": np.full(shape, 0.10 + 0.22 * severity),
        "social_fatigue": np.full(shape, 0.08 + 0.38 * severity),
        "conflict_intensity": np.full(paths, 0.40 + 0.50 * severity),
        "war_intervention": np.broadcast_to(
            np.array([1.0, 1.0, 0.75, 0.65]), shape
        ).copy(),
    }


def run_fixture(severity: float) -> tuple[WartimeCollectiveActionRegimeSystem, object]:
    paths = 32
    system = WartimeCollectiveActionRegimeSystem(
        paths, 57_101, np.full((paths, 4), 0.68)
    )
    factors = None
    fixture = inputs(paths, severity)
    for month in range(24):
        factors = system.update(month=month, **fixture)
    return system, factors


def main() -> None:
    inherited = json.loads(
        (ROOT / "outputs" / "taiwan_v56_geometry_diagnostic.json").read_text(
            encoding="utf-8"
        )
    )
    mild_system, mild = run_fixture(0.15)
    severe_system, severe = run_fixture(0.85)
    mild_diag = mild_system.diagnostics()
    severe_diag = severe_system.diagnostics()
    checks = {
        "inherited_v56_pass": inherited.get("status") == "PASS",
        "loyalty_simplex": max(
            mild_diag["max_loyalty_simplex_residual"],
            severe_diag["max_loyalty_simplex_residual"],
        ) < 1.0e-12,
        "event_simplex": max(
            mild_diag["max_event_probability_sum"],
            severe_diag["max_event_probability_sum"],
        ) <= 1.0 + 1.0e-12,
        "state_bounds": bool(
            np.all((severe.household_pressure >= 0.0) & (severe.household_pressure <= 1.0))
            and np.all((severe.common_knowledge >= 0.0) & (severe.common_knowledge <= 1.0))
            and np.all((severe.organization_capital >= 0.0) & (severe.organization_capital <= 1.0))
            and np.all((severe.collective_action >= 0.0) & (severe.collective_action <= 1.0))
        ),
        "control_bounds": bool(
            np.all((severe.government_controls >= 0.0) & (severe.government_controls <= 0.85))
        ),
        "feedback_bounds": bool(
            np.all((severe.labor_factor >= 0.35) & (severe.labor_factor <= 1.0))
            and np.all((severe.logistics_factor >= 0.45) & (severe.logistics_factor <= 1.0))
            and np.all((severe.morale_factor >= 0.35) & (severe.morale_factor <= 1.02))
            and np.all((severe.fiscal_drag >= 0.0) & (severe.fiscal_drag <= 0.12))
        ),
        "severe_pressure_direction": float(np.mean(severe.household_pressure))
        > float(np.mean(mild.household_pressure)),
        "severe_fragility_direction": float(np.mean(severe.political_fragility))
        > float(np.mean(mild.political_fragility)),
        "severe_compliance_direction": float(np.mean(severe.compliance))
        < float(np.mean(mild.compliance)),
        "first_event_exclusive": bool(
            np.all((severe_system.first_event >= 0) & (severe_system.first_event <= 7))
        ),
        "regime_state_valid": bool(
            np.all((severe_system.regime >= 0) & (severe_system.regime <= 4))
        ),
        "revolution_resolution_exclusive": bool(
            np.all(
                (severe_system.revolution_resolution >= 0)
                & (severe_system.revolution_resolution <= 2)
            )
        ),
        "revolution_not_terminal": bool(
            np.all(
                (severe.war_continuation >= 0.0)
                & (severe.war_continuation <= 1.0)
                & (severe.negotiation_pressure >= 0.0)
                & (severe.negotiation_pressure <= 1.0)
            )
        ),
        "war_talk_complementarity": bool(
            np.max(np.abs(
                severe.war_continuation + severe.negotiation_pressure - 1.0
            )) <= 1e-12
        ),
        "resolution_signals_mutually_exclusive": bool(
            not np.any(severe.government_replaced & severe.revolution_suppressed)
        ),
    }
    checks = {key: bool(value) for key, value in checks.items()}
    payload = {
        "classification": "DIAGNOSTIC_ONLY; no campaign outcome probabilities",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "mild": mild_diag,
        "severe": severe_diag,
        "directional_statistics": {
            "mild_pressure_mean": float(np.mean(mild.household_pressure)),
            "severe_pressure_mean": float(np.mean(severe.household_pressure)),
            "mild_fragility_mean": float(np.mean(mild.political_fragility)),
            "severe_fragility_mean": float(np.mean(severe.political_fragility)),
            "mild_compliance_mean": float(np.mean(mild.compliance)),
            "severe_compliance_mean": float(np.mean(severe.compliance)),
        },
    }
    TARGET.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"TAIWAN_V57_SOCIAL_DIAGNOSTIC: {payload['status']}")
    if payload["status"] != "PASS":
        print(json.dumps({k: v for k, v in checks.items() if not v}, indent=2))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
