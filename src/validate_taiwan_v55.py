from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
TARGET = OUT / "taiwan_v55_validation_and_v54_comparison.json"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def indexed(rows):
    return {(row["trade_scenario"], row["horizon_years"]): row for row in rows}


def main() -> None:
    v54 = json.loads((OUT / "taiwan_v54_windows_accelerated_simulation.json").read_text(encoding="utf-8"))
    v55_path = OUT / "taiwan_v55_econometric_simulation.json"
    v55 = json.loads(v55_path.read_text(encoding="utf-8"))
    v54_aggregate = indexed(v54["aggregate"])
    v55_aggregate = indexed(v55["aggregate"])
    differences = []
    for key, current in v55_aggregate.items():
        old = v54_aggregate[key]
        differences.append({
            "trade_scenario": key[0], "horizon_years": key[1],
            "conditional_v54": old["conditional_mean"],
            "conditional_v55": current["conditional_mean"],
            "conditional_change_pp": 100.0 * (current["conditional_mean"] - old["conditional_mean"]),
            "unconditional_v54": old["unconditional_mean"],
            "unconditional_v55": current["unconditional_mean"],
            "unconditional_change_pp": 100.0 * (current["unconditional_mean"] - old["unconditional_mean"]),
        })
    diagnostics = list(v55["module_diagnostics"].values())
    residuals = {
        "macro_outer": max(item["macro"]["max_outer_residual"] for item in diagnostics),
        "cge": max(item["macro"]["max_cge_solver_residual"] for item in diagnostics),
        "dsge": max(item["macro"]["max_dsge_solver_residual"] for item in diagnostics),
        "mfg": max(item["mfg"]["max_fixed_point_residual"] for item in diagnostics),
        "macro_nonconverged_months": max(item["macro"]["nonconverged_months"] for item in diagnostics),
        "mfg_nonconverged_months": max(item["mfg"]["nonconverged_months"] for item in diagnostics),
    }
    decomposition_residual = max(abs(
        row["peace_arrangement_cif"] + row["legal_separation_cif"]
        + row["status_quo_survival"] + row["joint_force_and_broad_control_by_2100"]
        + row["force_without_broad_control_by_2100"] - 1.0
    ) for row in v55["verification"]["calendar_2100"])
    csv_path = OUT / "taiwan_v55_econometric_results.csv"
    checks = {
        "csv_hash_matches": digest(csv_path) == v55["verification"]["result_sha256"],
        "runtime_validation_pass": v55["verification"]["runtime_validation"].endswith("PASS"),
        "calendar_decomposition_residual": decomposition_residual,
        "macro_nonconverged_months": residuals["macro_nonconverged_months"],
        "mfg_nonconverged_months": residuals["mfg_nonconverged_months"],
        "all_probabilities_bounded": all(
            0.0 <= row["conditional_broad_control_rate"] <= 1.0
            for row in v55["conditional"]
        ),
    }
    pass_status = (
        all(value for key, value in checks.items() if isinstance(value, bool))
        and decomposition_residual < 1e-9
        and residuals["macro_nonconverged_months"] == 0
        and residuals["mfg_nonconverged_months"] == 0
        and residuals["macro_outer"] <= 3.0e-3
        and residuals["cge"] <= 2.0e-3
        and residuals["dsge"] <= 8.0e-4
        and residuals["mfg"] <= 5.0e-4
    )
    payload = {
        "model_version": "V5.5", "status": "PASS" if pass_status else "FAIL",
        "checks": checks, "maximum_residuals": residuals,
        "v54_v55_differences": differences,
        "calendar_2100": v55["verification"]["calendar_2100"],
        "intervention_probability_means": v55["verification"]["intervention_probability_means"],
        "force_onset_mean": v55["verification"]["force_onset_mean"],
    }
    TARGET.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"TAIWAN_V55_VALIDATION: {payload['status']}")
    if not pass_status:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
