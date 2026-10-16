"""DIAGNOSTIC_ONLY: conformal DRO and financial martingale OT; no outcomes."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import simulate_taiwan_fully_closed_v60 as v60
from continuous_multipop_mfg_v66 import ContinuousMultiPopulationMFGV66
from martingale_ot_finance_v69 import MartingaleOTFinancialSystemV69
from simulate_four_party_financial_v45 import FourPartyFinancialSystem


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "outputs" / "taiwan_v69_ot_symplectic_active_diagnostic.json"


def finance_inputs(paths: int, month: int) -> dict:
    phase = (month + 1) / 18.0
    damage = np.clip(phase * np.array([[0.12, 0.24, 0.08, 0.10]]), 0.0, 0.7)
    damage = np.repeat(damage, paths, axis=0)
    return {
        "month": month,
        "actor_damage": damage,
        "actor_shortage": np.clip(0.04 + 0.55 * damage, 0.0, 1.0),
        "actor_mobilization": np.clip(0.15 + damage, 0.0, 1.0),
        "firm_distress": np.repeat(damage[:, :, None], 3, axis=2),
        "sector_output": np.repeat((1.0 - 0.35 * damage)[:, :, None], 3, axis=2),
        "trade_gdp_factor": np.clip(1.0 - 0.25 * damage, 0.1, 1.0),
        "trade_industry_factor": np.clip(1.0 - 0.30 * damage, 0.1, 1.0),
        "trade_inflation": 0.02 + 0.12 * damage,
        "trade_loss": np.full(paths, 0.12 * phase),
        "macro_trade_balance": -0.04 * damage,
        "macro_government_balance": -0.06 * damage,
        "us_level": 0.8,
        "japan_level": 0.7,
    }


def main() -> None:
    paths = 8
    active = MartingaleOTFinancialSystemV69(paths, 18, 69_101, "full", "baseline")
    ablated = FourPartyFinancialSystem(paths, 18, 69_101, "full", "baseline")
    for month in range(18):
        inputs = finance_inputs(paths, month)
        active_factor = active.update(**inputs)
        base_factor = ablated.update(**inputs)
    initial = np.repeat(np.array([[0.62, 0.48, 0.55, 0.52]]), paths, axis=0)
    mfg = ContinuousMultiPopulationMFGV66(paths, 69_201, initial, **v60.MFG_CONFIG)
    for month in range(14):
        damage = np.clip((month + 1) / 14.0 * np.array([[0.10, 0.22, 0.08, 0.12]]), 0, 1)
        damage = np.repeat(damage, paths, axis=0)
        mfg.update(
            progress=0.1 - damage, damage=damage,
            shortage=0.05 + 0.4 * damage, inflation=0.02 + 0.2 * damage,
            financial_stress=0.04 + 0.3 * damage,
            command=np.clip(0.96 - 0.3 * damage, 0.05, 1),
            network=np.clip(0.95 - 0.35 * damage, 0.05, 1),
            contagion_mobilization=0.12 + 0.2 * damage,
            contagion_fatigue=0.08 + 0.3 * damage,
            common_noise=np.zeros((paths, 4)),
            jump_signal=np.zeros((paths, 4)),
            max_fixed_point_iterations=48, tolerance=5.0e-4,
        )
    fd = active.summary()["martingale_optimal_transport"]
    cd = mfg.diagnostics()["partial_observation"]["v66_deep_causal_transport"]["conformal_dro"]
    ablation_delta = float(np.max(np.abs(active_factor.credit - base_factor.credit)))
    checks = {
        "martingale_projection_active": fd["projection_months"] == 18,
        "martingale_residual": fd["max_martingale_residual"] < 1.0e-10,
        "probability_simplex": fd["max_probability_residual"] < 1.0e-12,
        "nonnegative_probabilities": fd["max_negative_probability"] < 1.0e-14,
        "financial_feedback_ablation": ablation_delta > 1.0e-8,
        "conformal_radius_active": cd["updates"] > 0,
        "conformal_components_nonnegative": min(
            cd["observation_radius"], cd["parameter_radius"], cd["structure_radius"]
        ) >= 0.0,
    }
    payload = {
        "classification": "DIAGNOSTIC_ONLY; no outcome probabilities",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "martingale_ot": fd,
        "conformal_dro": cd,
        "credit_ablation_max_abs_delta": ablation_delta,
        "rom_selection": {
            "status": "excluded_not_candidate",
            "reason": "full-order symplectic core has only 3 q + 3 p states per actor; reduction does not address the measured CGE/DSGE/MFG bottleneck",
            "active_replacement": "full-order Stormer-Verlet and SO(3) Lie-group update",
        },
    }
    TARGET.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"TAIWAN_V69_DIAGNOSTIC: {payload['status']}")
    if payload["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
