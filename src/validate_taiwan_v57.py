from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
TARGET = OUT / "taiwan_v57_validation_and_v56_comparison.json"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def indexed(rows: list[dict]) -> dict[tuple, dict]:
    return {(row["trade_scenario"], row["horizon_years"]): row for row in rows}


def maximum(diagnostics: list[dict], section: str, key: str) -> float:
    return max(float(item[section].get(key, 0.0)) for item in diagnostics)


def minimum(diagnostics: list[dict], section: str, key: str) -> float:
    return min(float(item[section][key]) for item in diagnostics)


def main() -> None:
    old = json.loads((OUT / "taiwan_v56_geometry_simulation.json").read_text(encoding="utf-8"))
    current_path = OUT / "taiwan_v57_social_regime_simulation.json"
    current = json.loads(current_path.read_text(encoding="utf-8"))
    old_aggregate = indexed(old["aggregate"])
    differences = []
    for row in current["aggregate"]:
        key = (row["trade_scenario"], row["horizon_years"])
        baseline = old_aggregate[key]
        differences.append({
            "trade_scenario": key[0], "horizon_years": key[1],
            "conditional_v56": baseline["conditional_mean"],
            "conditional_v57": row["conditional_mean"],
            "conditional_change_pp": 100.0 * (row["conditional_mean"] - baseline["conditional_mean"]),
            "unconditional_v56": baseline["unconditional_mean"],
            "unconditional_v57": row["unconditional_mean"],
            "unconditional_change_pp": 100.0 * (row["unconditional_mean"] - baseline["unconditional_mean"]),
        })

    diagnostics = list(current["module_diagnostics"].values())
    political = [item["political"] for item in diagnostics if item.get("political")]
    residuals = {
        "macro_outer": maximum(diagnostics, "macro", "max_outer_residual"),
        "cge": maximum(diagnostics, "macro", "max_cge_solver_residual"),
        "dsge": maximum(diagnostics, "macro", "max_dsge_solver_residual"),
        "mfg": maximum(diagnostics, "mfg", "max_fixed_point_residual"),
        "macro_nonconverged_months": maximum(diagnostics, "macro", "nonconverged_months"),
        "mfg_nonconverged_months": maximum(diagnostics, "mfg", "nonconverged_months"),
        "randers_norm": maximum(diagnostics, "operations", "max_randers_one_form_norm"),
        "so3_orthogonality": maximum(diagnostics, "operations", "max_so3_orthogonality_residual"),
        "so3_determinant": maximum(diagnostics, "operations", "max_so3_determinant_residual"),
        "pnt_minimum_eigenvalue": minimum(diagnostics, "pnt", "minimum_covariance_eigenvalue"),
        "joseph_symmetry": maximum(diagnostics, "pnt", "max_joseph_symmetry_residual"),
        "dynamic_finsler_row": maximum(diagnostics, "spatial", "max_dynamic_transition_row_residual"),
        "port_hamiltonian_dae": maximum(diagnostics, "energy", "max_port_hamiltonian_dae_residual"),
        "dissipation_violation": maximum(diagnostics, "energy", "max_dissipation_inequality_violation"),
        "manifold_tangent": maximum(diagnostics, "geometry", "max_manifold_tangent_residual"),
        "wfr_mass": maximum(diagnostics, "mfg", "max_wfr_strang_mass_residual"),
        "saltation_finite_difference": maximum(diagnostics, "hybrid", "max_saltation_finite_difference_residual"),
        "saltation_nonfinite_count": maximum(diagnostics, "hybrid", "saltation_nonfinite_count"),
        "loyalty_simplex": max(float(item["max_loyalty_simplex_residual"]) for item in political),
        "event_probability_sum": max(float(item["max_event_probability_sum"]) for item in political),
        "minimum_social_feedback_factor": min(float(item["minimum_feedback_factor"]) for item in political),
    }
    decomposition = max(abs(
        row["peace_arrangement_cif"] + row["legal_separation_cif"]
        + row["status_quo_survival"] + row["joint_force_and_broad_control_by_2100"]
        + row["force_without_broad_control_by_2100"] - 1.0
    ) for row in current["verification"]["calendar_2100"])
    resolution_shares = [
        share
        for item in political
        for actor in item["revolution_resolution_shares"].values()
        for share in actor.values()
    ]
    csv_path = OUT / "taiwan_v57_social_regime_results.csv"
    checks = {
        "csv_hash_matches": digest(csv_path) == current["verification"]["result_sha256"],
        "runtime_validation_pass": current["verification"]["runtime_validation"].endswith("PASS"),
        "calendar_decomposition": decomposition < 1e-9,
        "probabilities_bounded": all(
            0.0 <= row["conditional_broad_control_rate"] <= 1.0
            for row in current["conditional"]
        ),
        "macro_converged": residuals["macro_nonconverged_months"] == 0,
        "mfg_converged": residuals["mfg_nonconverged_months"] == 0,
        "macro_outer": residuals["macro_outer"] <= 3e-3,
        "cge": residuals["cge"] <= 2e-3,
        "dsge": residuals["dsge"] <= 8e-4,
        "mfg": residuals["mfg"] <= 5e-4,
        "randers": residuals["randers_norm"] < 1.0,
        "so3": max(residuals["so3_orthogonality"], residuals["so3_determinant"]) <= 1e-12,
        "pnt_spd_joseph": residuals["pnt_minimum_eigenvalue"] > 0.0 and residuals["joseph_symmetry"] <= 1e-12,
        "dynamic_finsler": residuals["dynamic_finsler_row"] <= 1e-10,
        "port_hamiltonian": max(residuals["port_hamiltonian_dae"], residuals["dissipation_violation"]) <= 1e-10,
        "slow_manifold": residuals["manifold_tangent"] <= 1e-12,
        "wfr": residuals["wfr_mass"] <= 1e-10,
        "saltation": residuals["saltation_finite_difference"] <= 1e-12 and residuals["saltation_nonfinite_count"] == 0,
        "loyalty_simplex": residuals["loyalty_simplex"] <= 1e-12,
        "event_probability": residuals["event_probability_sum"] <= 1.0,
        "feedback_floor": residuals["minimum_social_feedback_factor"] >= 0.35,
        "resolution_shares_bounded": all(0.0 <= value <= 1.0 for value in resolution_shares),
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    payload = {
        "model_version": "V5.7", "status": status, "checks": checks,
        "calendar_decomposition_residual": decomposition,
        "maximum_residuals": residuals, "v56_v57_differences": differences,
        "calendar_2100": current["verification"]["calendar_2100"],
        "intervention_probability_means": current["verification"]["intervention_probability_means"],
        "force_onset_mean": current["verification"]["force_onset_mean"],
    }
    TARGET.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"TAIWAN_V57_VALIDATION: {status}")
    if status != "PASS":
        print(json.dumps({key: value for key, value in checks.items() if not value}, indent=2))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
