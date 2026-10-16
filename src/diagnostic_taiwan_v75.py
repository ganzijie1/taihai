from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from hypergraph_mfg_v75 import DirectedMultilayerHypergraphMFGV75
from quantitative_financial_instability_v75 import QuantitativeFinancialInstabilityV75


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
TARGET = OUT / "taiwan_v75_diagnostic_only.json"


def hypergraph_fixture() -> dict:
    paths = 12
    support = np.tile(np.array([
        [0.62, 0.48, 0.55, 0.57],
        [0.58, 0.52, 0.51, 0.54],
    ]), (paths // 2, 1))
    system = DirectedMultilayerHypergraphMFGV75(
        paths,
        750_013,
        support,
        advanced_graphon=True,
        graphon_mode="multilayer",
        partial_observation_geometry=False,
        network_geometry_enabled=False,
    )
    shape = (paths, 4)
    damage = np.tile(np.array([[0.18, 0.28, 0.09, 0.11], [0.22, 0.25, 0.10, 0.12]]), (paths // 2, 1))
    shortage = np.tile(np.array([[0.24, 0.31, 0.12, 0.14], [0.26, 0.29, 0.13, 0.15]]), (paths // 2, 1))
    network = np.tile(np.array([[0.72, 0.66, 0.78, 0.75], [0.69, 0.64, 0.76, 0.73]]), (paths // 2, 1))
    common_noise = np.tile(np.array([[0.08, 0.05, 0.03, 0.04], [0.06, 0.07, 0.02, 0.03]]), (paths // 2, 1))
    jump = np.tile(np.array([[0.05, 0.12, 0.03, 0.04], [0.07, 0.10, 0.02, 0.05]]), (paths // 2, 1))
    signal = np.tile(np.array([[0.25, -0.12, 0.08, 0.10], [0.20, -0.08, 0.06, 0.09]]), (paths // 2, 1))
    kernel, weights, intensity = system._effective_graphon(
        damage, shortage, network, common_noise, jump, signal
    )
    mean = np.linspace(-0.52, 0.68, paths * 4 * 4).reshape(paths, 4, 4)
    action = np.linspace(0.05, 0.85, paths * 4 * 4 * 5).reshape(paths, 4, 4, 5)
    state, action_field, major = system._multilayer_fields(
        mean, action, kernel, weights, system.major_action, intensity
    )
    finite_state, finite_action, finite_error = system._finite_network_fields(
        mean, action, kernel, weights, intensity
    )
    diag = system.diagnostics()["v75_directed_multilayer_hypergraph_mfg"]
    checks = {
        "state_shape": state.shape == mean.shape,
        "action_shape": action_field.shape == action.shape,
        "major_shape": major.shape == shape,
        "finite_fields": bool(
            np.all(np.isfinite(state))
            and np.all(np.isfinite(action_field))
            and np.all(np.isfinite(major))
        ),
        "finite_rollback_shapes": (
            finite_state.shape == mean[:8].shape
            and finite_action.shape == action[:8].shape
        ),
        "finite_subsample_exercised": finite_state.shape[0] < mean.shape[0],
        "finite_rollback_error": bool(np.isfinite(finite_error) and finite_error >= 0.0),
        "kernel_rows": float(np.max(np.abs(np.sum(kernel, axis=3) - 1.0))) <= 1.0e-12,
        "positive_head_coverage": diag["max_incidence_residual"] <= 1.0e-12,
        "higher_order_active": diag["max_higher_order_contribution"] > 0.0,
        "not_clique_equivalent": diag["max_clique_rollback_error"] > 0.0,
        "no_terminal_adjustment": not diag["terminal_probability_adjustment"],
    }
    return {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks, "diagnostics": diag}


def finance_fixture() -> dict:
    system = QuantitativeFinancialInstabilityV75(
        3, 24, 750_029, "limited_intervention", "peace_baseline"
    )
    cash_flow = np.array([
        [0.90, 0.82, 1.02, 0.98],
        [0.76, 0.70, 0.94, 0.91],
        [0.84, 0.74, 0.97, 0.95],
    ])
    debt_service = np.array([
        [0.42, 0.39, 0.48, 0.44],
        [0.51, 0.47, 0.52, 0.49],
        [0.46, 0.45, 0.50, 0.47],
    ])
    distress = np.array([
        [0.22, 0.28, 0.18, 0.16],
        [0.38, 0.42, 0.25, 0.23],
        [0.30, 0.36, 0.21, 0.20],
    ])
    shortage = 0.7 * distress
    support = np.broadcast_to(np.array([[0.78, 0.66, 0.92, 0.88]]), distress.shape)
    system._update_minsky_states(cash_flow, debt_service, distress, shortage, support)
    obligations = np.array([
        [0.42, 0.38, 0.45, 0.40],
        [0.48, 0.44, 0.49, 0.46],
        [0.45, 0.41, 0.47, 0.43],
    ])
    external_assets = obligations * np.array([
        [1.20, 1.08, 1.28, 1.22],
        [0.18, 0.24, 0.36, 0.30],
        [0.72, 0.64, 0.84, 0.76],
    ])
    clearing = system._eisenberg_noe_clearing(external_assets, obligations)
    system._update_regime_and_intensity(distress, clearing)
    checks = {
        "financing_mass": float(np.max(np.abs(np.sum(system.financing_share, axis=-1) - 1.0))) <= 1.0e-12,
        "nonnegative_financing_share": float(np.min(system.financing_share)) >= 0.0,
        "finite_dscr": bool(np.all(np.isfinite(system.dscr))),
        "clearing_bounds": bool(np.all((clearing >= 0.0) & (clearing <= 1.0))),
        "partial_clearing_exercised": bool(np.min(clearing) < 0.95),
        "clearing_residual": system.max_clearing_residual <= 1.0e-8,
        "regime_mass": float(np.max(np.abs(np.sum(system.regime_probability, axis=-1) - 1.0))) <= 1.0e-12,
        "positive_credit_intensity": float(np.min(system.credit_intensity)) > 0.0,
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "diagnostics": {
            "max_financing_mass_residual": system.max_financing_mass_residual,
            "max_clearing_residual": system.max_clearing_residual,
            "max_regime_mass_residual": system.max_regime_mass_residual,
            "mean_dscr": float(np.mean(system.dscr)),
            "mean_clearing_fraction": float(np.mean(clearing)),
        },
    }


def main() -> None:
    hypergraph = hypergraph_fixture()
    finance = finance_fixture()
    payload = {
        "model_version": "V7.5-HYPERGRAPH-MINSKY-FINANCE",
        "classification": "DIAGNOSTIC_ONLY; no Monte Carlo aggregation, scenario ranking or outcome probabilities",
        "status": "PASS" if hypergraph["status"] == finance["status"] == "PASS" else "FAIL",
        "hypergraph_mfg": hypergraph,
        "quantitative_finance": finance,
    }
    TARGET.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"TAIWAN_V75_DIAGNOSTIC_ONLY: {payload['status']}")
    if payload["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
