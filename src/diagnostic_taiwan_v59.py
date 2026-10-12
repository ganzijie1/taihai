"""DIAGNOSTIC_ONLY: invariant checks for V5.9 network geometry."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from continuous_multipop_mfg_v47 import ContinuousMultiPopulationMFG
from diagnostic_taiwan_v58 import fixture, numeric_difference


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
TARGET = OUT / "taiwan_v59_network_geometry_diagnostic.json"
LEDGER = OUT / "taiwan_v59_network_geometry_source_ledger.json"


def main() -> None:
    paths = 4
    support = np.full((paths, 4), 0.56)
    baseline_a = ContinuousMultiPopulationMFG(
        paths, 59_101, support, advanced_graphon=True,
        graphon_mode="multilayer", fp_backend="numpy_fvm",
        geometry_enabled=True, wfr_strang=False,
        natural_gradient_major=True, geomloss_interval=99,
        partial_observation_geometry=True, full_wfr_jko=True,
    )
    baseline_b = ContinuousMultiPopulationMFG(
        paths, 59_101, support, advanced_graphon=True,
        graphon_mode="multilayer", fp_backend="numpy_fvm",
        geometry_enabled=True, wfr_strang=False,
        natural_gradient_major=True, geomloss_interval=99,
        partial_observation_geometry=True, full_wfr_jko=True,
        network_geometry_enabled=False,
    )
    base_inputs = fixture(paths, 59_201)
    legacy_difference = numeric_difference(
        baseline_a.update(**base_inputs), baseline_b.update(**base_inputs)
    )

    advanced = ContinuousMultiPopulationMFG(
        paths, 59_301, support, advanced_graphon=True,
        graphon_mode="multilayer", fp_backend="numpy_fvm",
        geometry_enabled=True, wfr_strang=False,
        natural_gradient_major=True, geomloss_interval=99,
        partial_observation_geometry=True, full_wfr_jko=True,
        network_geometry_enabled=True,
        network_geometry_spectral_interval=3,
    )
    factor = None
    for month in range(6):
        factor = advanced.update(**fixture(paths, 59_401 + month, 0.20 + 0.10 * month))
    diagnostic = advanced.diagnostics()
    geometry = diagnostic["network_geometry"]
    checks = {
        "v58_default_regression": legacy_difference == 0.0,
        "mfg_converged": factor is not None and factor.converged,
        "transition_rows": geometry["max_transition_row_residual"] <= 1.0e-12,
        "positive_intensity": geometry["minimum_intensity"] > 0.0,
        "finite_intensity": np.isfinite(geometry["maximum_intensity"]),
        "actor_spectral_gap_nonnegative": geometry["minimum_actor_spectral_gap"] >= -1.0e-12,
        "supra_spectral_gap_nonnegative": geometry["minimum_supra_spectral_gap"] >= -1.0e-12,
        "conductance_bounded": 0.0 <= geometry["minimum_conductance"] <= 1.0,
        "sheaf_psd": geometry["minimum_sheaf_eigenvalue"] >= -1.0e-12,
        "temporal_step_bounded": geometry["maximum_temporal_log_step"] <= 0.18 + 1.0e-12,
        "bottleneck_risk_bounded": 0.0 <= geometry["maximum_bottleneck_risk"] <= 1.0,
        "supra_refresh_clock": geometry["supra_refreshes"] == 2,
        "same_path_feedback_nontrivial": bool(
            factor.network_bottleneck_risk > 0.0
            and factor.network_sheaf_consistency < 1.0
        ),
    }
    checks = {key: bool(value) for key, value in checks.items()}
    payload = {
        "classification": "DIAGNOSTIC_ONLY; no outcome probabilities",
        "model_version": "V5.9",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "v58_default_max_difference": legacy_difference,
        "network_geometry": geometry,
        "last_factor": {
            "fixed_point_residual": factor.fixed_point_residual,
            "mass_residual": factor.mass_residual,
            "network_bottleneck_risk": factor.network_bottleneck_risk,
            "network_sheaf_consistency": factor.network_sheaf_consistency,
        },
    }
    TARGET.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    ledger = {
        "model_version": "V5.9",
        "calibration_status": "structural priors; network parameters require public-data calibration",
        "sources": [
            {
                "source": "https://my.feishu.cn/docx/Ou3NdGj7CoGUqwxfht5clSdqnxd",
                "publisher": "project network-geometry specification",
                "observation_date": "2026-10-09",
                "unit": "equation family and implementation contract",
                "identification": "scenario_prior",
            },
            {
                "source": "https://arxiv.org/abs/2008.10216",
                "publisher": "research preprint",
                "observation_date": "2020",
                "unit": "Graphon mean-field-game method",
                "identification": "derived",
            },
            {
                "source": "https://arxiv.org/abs/2005.12798",
                "publisher": "research preprint",
                "observation_date": "2020",
                "unit": "discourse-sheaf opinion dynamics method",
                "identification": "derived",
            },
            {
                "source": "https://epubs.siam.org/doi/10.1137/20M1339246",
                "publisher": "SIAM Journal on Applied Mathematics",
                "observation_date": "2021",
                "unit": "dense-graph and graphon random-walk method",
                "identification": "derived",
            },
        ],
        "not_observed": [
            "future wartime multilayer edge strengths",
            "future cross-layer coupling coefficients",
            "future network collapse thresholds",
        ],
    }
    LEDGER.write_text(json.dumps(ledger, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"TAIWAN_V59_NETWORK_GEOMETRY_DIAGNOSTIC: {payload['status']}")
    if payload["status"] != "PASS":
        print(json.dumps({key: value for key, value in checks.items() if not value}, indent=2))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
