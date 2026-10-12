"""DIAGNOSTIC_ONLY: invariant checks for the V5.8 information geometry."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from continuous_multipop_mfg_v47 import ContinuousMultiPopulationMFG


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
TARGET = OUT / "taiwan_v58_information_geometry_diagnostic.json"
LEDGER = OUT / "taiwan_v58_information_geometry_source_ledger.json"


def fixture(paths: int, seed: int, severity: float = 0.35) -> dict:
    rng = np.random.default_rng(seed)
    shape = (paths, 4)
    return {
        "progress": rng.uniform(-0.18, 0.18, shape) - 0.08 * severity,
        "damage": rng.uniform(0.01, 0.25, shape) * (0.5 + severity),
        "shortage": rng.uniform(0.02, 0.30, shape) * (0.5 + severity),
        "inflation": rng.uniform(0.01, 0.12, shape) * (0.6 + severity),
        "financial_stress": rng.uniform(0.04, 0.32, shape) * (0.5 + severity),
        "command": rng.uniform(0.58, 0.94, shape),
        "network": rng.uniform(0.42, 0.90, shape),
        "contagion_mobilization": rng.uniform(0.08, 0.55, shape),
        "contagion_fatigue": rng.uniform(0.06, 0.48, shape),
        "common_noise": rng.uniform(-0.30, 0.30, shape),
        "jump_signal": rng.uniform(0.0, 0.24, shape) * severity,
    }


def numeric_difference(left, right) -> float:
    values = []
    for field in left.__dataclass_fields__:
        a = getattr(left, field)
        b = getattr(right, field)
        if isinstance(a, np.ndarray):
            values.append(float(np.max(np.abs(a.astype(float) - b.astype(float)))))
        elif isinstance(a, (float, int, np.floating, np.integer)):
            values.append(abs(float(a) - float(b)))
    return max(values, default=0.0)


def main() -> None:
    paths = 4
    support = np.full((paths, 4), 0.56)
    baseline_a = ContinuousMultiPopulationMFG(
        paths, 58_101, support, advanced_graphon=True,
        graphon_mode="multilayer", fp_backend="numpy_fvm",
        geometry_enabled=True, wfr_strang=True,
        natural_gradient_major=True, geomloss_interval=99,
    )
    baseline_b = ContinuousMultiPopulationMFG(
        paths, 58_101, support, advanced_graphon=True,
        graphon_mode="multilayer", fp_backend="numpy_fvm",
        geometry_enabled=True, wfr_strang=True,
        natural_gradient_major=True, geomloss_interval=99,
    )
    base_inputs = fixture(paths, 58_201)
    baseline_difference = numeric_difference(
        baseline_a.update(**base_inputs), baseline_b.update(**base_inputs)
    )

    advanced = ContinuousMultiPopulationMFG(
        paths, 58_301, support, advanced_graphon=True,
        graphon_mode="multilayer", fp_backend="torch_fvm",
        geometry_enabled=True, wfr_strang=False,
        natural_gradient_major=True, geomloss_interval=99,
        partial_observation_geometry=True, full_wfr_jko=True,
        fisher_trust_radius=0.035, minor_policy_trust_radius=0.025,
        information_ambiguity_radius=0.025,
    )
    factor = None
    for month in range(6):
        factor = advanced.update(**fixture(paths, 58_401 + month, 0.20 + 0.10 * month))
    diagnostic = advanced.diagnostics()
    belief = diagnostic["partial_observation"]
    fp = diagnostic["fp_solver"]
    alignment = np.asarray(belief["fgw_alignment"])
    checks = {
        "legacy_default_regression": baseline_difference == 0.0,
        "mfg_converged": factor is not None and factor.converged,
        "posterior_simplex": belief["max_posterior_simplex_residual"] <= 1.0e-12,
        "credibility_simplex": belief["max_reliability_simplex_residual"] <= 1.0e-12,
        "schrodinger_marginals": belief["max_bridge_marginal_residual"] <= 1.0e-7,
        "fgw_rows": bool(np.max(np.abs(np.sum(alignment, axis=1) - 1.0)) <= 1.0e-12),
        "fgw_nonnegative": bool(np.all(alignment >= 0.0)),
        "full_wfr_conditional_mass": diagnostic["max_full_wfr_conditional_residual"] <= 1.0e-12,
        "full_wfr_stock_accounting": diagnostic["max_full_wfr_stock_residual"] <= 1.0e-12,
        "full_wfr_nonnegative": fp["minimum_wfr_measure"] >= 0.0,
        "major_trust_region": diagnostic["max_major_trust_region_kl"] <= 0.035 + 1.0e-12,
        "minor_trust_region": diagnostic["max_minor_trust_region_kl"] <= 0.025 + 1.0e-12,
        "scores_finite": bool(
            np.isfinite(belief["mean_brier_score"])
            and np.isfinite(belief["mean_log_score"])
        ),
        "beliefs_nontrivial": belief["max_belief_truth_gap"] > 0.0,
        "ambiguity_bounded": 0.0 <= belief["max_ambiguity_radius"] <= 0.18,
        "same_path_actions_registered": bool(
            np.any(np.abs(advanced.belief_system.lagged_information_action) > 0.0)
        ),
    }
    checks = {key: bool(value) for key, value in checks.items()}
    payload = {
        "classification": "DIAGNOSTIC_ONLY; no outcome probabilities",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "legacy_default_max_difference": baseline_difference,
        "mfg": diagnostic,
        "last_factor": {
            "fixed_point_residual": factor.fixed_point_residual,
            "mass_residual": factor.mass_residual,
            "belief_truth_gap": factor.belief_truth_gap,
            "observation_uncertainty": factor.observation_uncertainty,
            "bridge_cost": factor.bridge_cost,
        },
    }
    TARGET.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    ledger = {
        "model_version": "V5.8",
        "calibration_status": "structural priors; public-data calibration remains explicitly pending",
        "sources": [
            {
                "source": "https://my.feishu.cn/docx/M6t9dmRhkoMhPrxJnV3ciYJXn2c",
                "publisher": "project model specification",
                "observation_date": "2026-10-09",
                "unit": "equation family and implementation contract",
                "identification": "scenario_prior",
            },
            {
                "source": "https://epubs.siam.org/doi/abs/10.1137/20M1339982",
                "publisher": "SIAM Review",
                "observation_date": "2021",
                "unit": "Schrodinger bridge method",
                "identification": "derived",
            },
            {
                "source": "https://arxiv.org/abs/1508.05216",
                "publisher": "Journal of Functional Analysis preprint",
                "observation_date": "2015-2018",
                "unit": "unbalanced optimal transport formulation",
                "identification": "derived",
            },
            {
                "source": "https://arxiv.org/abs/1805.09114",
                "publisher": "research preprint",
                "observation_date": "2018",
                "unit": "fused Gromov-Wasserstein network alignment method",
                "identification": "derived",
            },
            {
                "source": "https://proceedings.mlr.press/v139/wang21j.html",
                "publisher": "PMLR",
                "observation_date": "2021",
                "unit": "mean-field policy-gradient method",
                "identification": "derived",
            },
        ],
        "not_observed": [
            "wartime source credibility transitions",
            "wartime information-channel distortion",
            "future four-party belief distributions",
        ],
    }
    LEDGER.write_text(json.dumps(ledger, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"TAIWAN_V58_INFORMATION_GEOMETRY_DIAGNOSTIC: {payload['status']}")
    if payload["status"] != "PASS":
        print(json.dumps({key: value for key, value in checks.items() if not value}, indent=2))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
