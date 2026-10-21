from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from statistical_contact_v84 import (
    ContactGeometryToolkitV84,
    MultivariateToolkitV84,
    NonparametricToolkitV84,
    StatisticalContactSystemV84,
    TimeSeriesToolkitV84,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "taiwan_v84_diagnostic_only.json"


def main() -> None:
    rng = np.random.default_rng(84_003)
    paths = 24
    kernel = np.array([
        [0.58, 0.10, 0.16, 0.16],
        [0.12, 0.54, 0.18, 0.16],
        [0.12, 0.16, 0.58, 0.14],
        [0.14, 0.18, 0.18, 0.50],
    ])
    system = StatisticalContactSystemV84(paths, kernel)
    previous_contact = np.zeros((paths, 4, 4))
    monotone = True
    finite = True
    last = None
    for month in range(48):
        damage = np.clip(0.02 + month / 180.0 + rng.normal(0.0, 0.01, (paths, 4)), 0.0, 0.5)
        shortage = np.clip(0.04 + month / 220.0 + rng.normal(0.0, 0.015, (paths, 4)), 0.0, 0.6)
        zero = np.zeros((paths, 4))
        last = system.update(
            progress=rng.normal(0.0, 0.1, (paths, 4)), damage=damage,
            shortage=shortage, inflation=0.02 + 0.12 * shortage,
            financial_stress=np.clip(0.08 + 0.30 * shortage, 0.0, 1.0),
            command=np.clip(0.88 - 0.45 * damage, 0.1, 1.0),
            network=np.clip(0.90 - 0.35 * damage, 0.1, 1.0),
            contagion_mobilization=zero,
            contagion_fatigue=np.clip(0.10 + 0.25 * damage, 0.0, 1.0),
            common_noise=zero,
            jump_signal=(rng.random((paths, 4)) < 0.02).astype(float),
        )
        current = system.contact_state.copy()
        monotone &= bool(np.all(current >= previous_contact - 1.0e-14))
        finite &= bool(np.all(np.isfinite(current)))
        previous_contact = current

    x = np.linspace(-1.0, 1.0, 64)
    y = x * x + 0.05 * rng.normal(size=len(x))
    kde = NonparametricToolkitV84.kde(x, x)
    local = NonparametricToolkitV84.local_linear(x, y, x[::8], 0.25)
    isotonic = NonparametricToolkitV84.isotonic(np.array([0.0, 0.2, 0.15, 0.5]))
    sample = rng.normal(size=(80, 5))
    covariance = MultivariateToolkitV84.shrinkage_covariance(sample)
    scores, vectors, values, _, _ = MultivariateToolkitV84.pca(sample, 2)
    mixture = MultivariateToolkitV84.gaussian_mixture(sample[:, :2], 2, 8)
    series = rng.normal(size=(72, 3)).cumsum(axis=0)
    bvar_beta, bvar_residual, bvar_covariance = TimeSeriesToolkitV84.bvar(series)
    dcc = TimeSeriesToolkitV84.dcc(bvar_residual)
    change = TimeSeriesToolkitV84.bayesian_change_probability(series[:, 0])
    quantile_beta = NonparametricToolkitV84.quantile_regression(
        np.column_stack((np.ones(len(x)), x)), y, 0.75, 12
    )
    forest = NonparametricToolkitV84.survival_forest(
        np.arange(1, 65), (np.arange(64) % 5 == 0).astype(float),
        np.column_stack((x, x * x)), np.column_stack((x[::8], x[::8] ** 2)),
        trees=8,
    )
    q_next, p_next, burden_next = ContactGeometryToolkitV84.contact_step(
        np.zeros((4, 2)), np.ones((4, 2)) * 0.1, np.zeros(4),
        np.zeros((4, 2)), np.ones((4, 2)) * 0.05, np.ones((4, 2)) * 0.02,
    )
    diagnostic = system.diagnostics()
    checks = {
        "all_five_layers_enabled": all(system.flags.values()),
        "finite_dynamic_state": finite,
        "contact_loss_monotone": monotone,
        "contact_state_bounded": bool(np.max(system.contact_state) <= system.contact_cap + 1.0e-12),
        "probability_bounds": bool(np.min(last.regime_probability) >= 0.0 and np.max(last.regime_probability) <= 1.0),
        "kde_nonnegative": bool(np.min(kde) >= 0.0),
        "local_linear_finite": bool(np.all(np.isfinite(local))),
        "isotonic_monotone": bool(np.all(np.diff(isotonic) >= -1.0e-12)),
        "covariance_spd": bool(np.min(np.linalg.eigvalsh(covariance)) > 0.0),
        "pca_shapes": scores.shape == (80, 2) and vectors.shape == (5, 2) and values.shape == (2,),
        "mixture_normalized": bool(np.max(np.abs(mixture.sum(axis=1) - 1.0)) <= 1.0e-12),
        "bvar_finite_spd": bool(np.all(np.isfinite(bvar_beta)) and np.min(np.linalg.eigvalsh(bvar_covariance)) >= -1.0e-12),
        "dcc_bounded": bool(np.max(np.abs(dcc)) <= 1.0 + 1.0e-12),
        "change_probability_bounded": bool(np.min(change) >= 0.0 and np.max(change) <= 1.0),
        "quantile_backend_finite": bool(np.all(np.isfinite(quantile_beta))),
        "survival_forest_nonnegative": bool(np.min(forest) >= 0.0),
        "contact_integrator_finite": bool(np.all(np.isfinite(q_next)) and np.all(np.isfinite(p_next)) and np.all(np.isfinite(burden_next))),
        "no_future_information": diagnostic["no_future_information"],
        "no_terminal_probability_adjustment": not diagnostic["terminal_probability_adjustment"],
        "no_duplicate_structural_state": not diagnostic["structural_state_duplication"],
    }
    payload = {
        "classification": "DIAGNOSTIC_ONLY",
        "model_version": "V8.4-STATISTICAL-MULTISCALE-CONTACT",
        "status": "PASS" if all(checks.values()) else "BLOCKED",
        "checks": checks,
        "diagnostics": diagnostic,
        "substantive_outcomes_generated": False,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("TAIWAN_V84_DIAGNOSTIC_ONLY:", payload["status"])
    if payload["status"] != "PASS":
        raise RuntimeError(str({key: value for key, value in checks.items() if not value}))


if __name__ == "__main__":
    main()
