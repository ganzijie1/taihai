"""DIAGNOSTIC_ONLY functional-analysis checks on analytic fixtures."""

from __future__ import annotations

import json
from pathlib import Path

from functional_analysis_v68 import analyse_cell


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "outputs" / "taiwan_v68_functional_diagnostic.json"


def main() -> None:
    fixture = {
        "mfg": {
            "max_mass_residual": 2.0e-16,
            "last_fixed_point_residual": 1.5e-4,
            "max_fixed_point_residual": 4.5e-4,
            "max_kernel_row_residual": 2.0e-16,
            "fp_solver": {
                "max_mass_residual_before_normalisation": 3.0e-16,
                "minimum_density_before_projection": 1.0e-8,
            },
        },
        "macro": {
            "max_outer_residual": 2.5e-3,
            "max_cge_solver_residual": 1.9e-3,
            "max_dsge_solver_residual": 7.5e-4,
        },
        "large_deviation": {
            "max_likelihood_identity_residual": 2.0e-16,
            "max_empirical_measure_mass_residual": 0.0,
        },
    }
    report = analyse_cell(fixture)
    if not report.passed:
        raise AssertionError("functional-analysis fixture did not pass")
    payload = {
        "scope": "DIAGNOSTIC_ONLY; no probabilities or scenario results",
        "status": "PASS",
        "report": report.as_dict(),
    }
    TARGET.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("DIAGNOSTIC_ONLY_TAIWAN_V68_FUNCTIONAL_ANALYSIS: PASS")


if __name__ == "__main__":
    main()

