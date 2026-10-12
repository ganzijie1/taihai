from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
TARGET = OUT / "taiwan_v59_validation_and_v58_comparison.json"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def indexed(rows: list[dict]) -> dict[tuple, dict]:
    return {(row["trade_scenario"], row["horizon_years"]): row for row in rows}


def main() -> None:
    old = json.loads((OUT / "taiwan_v58_information_geometry_simulation.json").read_text(encoding="utf-8"))
    current_path = OUT / "taiwan_v59_network_geometry_simulation.json"
    current = json.loads(current_path.read_text(encoding="utf-8"))
    ablation = json.loads((OUT / "taiwan_v59_network_geometry_ablations.json").read_text(encoding="utf-8"))
    old_aggregate = indexed(old["aggregate"])
    differences = []
    for row in current["aggregate"]:
        key = (row["trade_scenario"], row["horizon_years"])
        baseline = old_aggregate[key]
        differences.append({
            "trade_scenario": key[0],
            "horizon_years": key[1],
            "conditional_v58": baseline["conditional_mean"],
            "conditional_v59": row["conditional_mean"],
            "conditional_change_pp": 100.0 * (row["conditional_mean"] - baseline["conditional_mean"]),
            "unconditional_v58": baseline["unconditional_mean"],
            "unconditional_v59": row["unconditional_mean"],
            "unconditional_change_pp": 100.0 * (row["unconditional_mean"] - baseline["unconditional_mean"]),
        })

    diagnostics = list(current["module_diagnostics"].values())
    geometries = [item["mfg"]["network_geometry"] for item in diagnostics]
    residuals = {
        "macro_outer": max(item["macro"]["max_outer_residual"] for item in diagnostics),
        "cge": max(item["macro"]["max_cge_solver_residual"] for item in diagnostics),
        "dsge": max(item["macro"]["max_dsge_solver_residual"] for item in diagnostics),
        "mfg": max(item["mfg"]["max_fixed_point_residual"] for item in diagnostics),
        "macro_nonconverged_months": max(item["macro"]["nonconverged_months"] for item in diagnostics),
        "mfg_nonconverged_months": max(item["mfg"]["nonconverged_months"] for item in diagnostics),
        "transition_row": max(item["max_transition_row_residual"] for item in geometries),
        "minimum_intensity": min(item["minimum_intensity"] for item in geometries),
        "minimum_actor_spectral_gap": min(item["minimum_actor_spectral_gap"] for item in geometries),
        "minimum_supra_spectral_gap": min(item["minimum_supra_spectral_gap"] for item in geometries),
        "minimum_conductance": min(item["minimum_conductance"] for item in geometries),
        "minimum_sheaf_eigenvalue": min(item["minimum_sheaf_eigenvalue"] for item in geometries),
        "maximum_temporal_log_step": max(item["maximum_temporal_log_step"] for item in geometries),
        "maximum_bottleneck_risk": max(item["maximum_bottleneck_risk"] for item in geometries),
    }
    decomposition = max(abs(
        row["peace_arrangement_cif"] + row["legal_separation_cif"]
        + row["status_quo_survival"] + row["joint_force_and_broad_control_by_2100"]
        + row["force_without_broad_control_by_2100"] - 1.0
    ) for row in current["verification"]["calendar_2100"])
    variants = ablation.get("variants", {})
    expected_variants = {
        "complete_v59", "without_intensity_transition", "without_supra_laplacian",
        "without_spectral_geometry", "without_forman_curvature",
        "without_sheaf_laplacian", "without_all_v59_network_geometry",
    }
    ablation_converged = all(
        value["diagnostics"]["macro_nonconverged_months"] == 0
        and value["diagnostics"]["mfg_nonconverged_months"] == 0
        and value["diagnostics"]["max_macro_residual"] <= 3.0e-3
        and value["diagnostics"]["max_cge_residual"] <= 2.0e-3
        and value["diagnostics"]["max_dsge_residual"] <= 8.0e-4
        and value["diagnostics"]["max_mfg_residual"] <= 5.0e-4
        for value in variants.values()
    )
    csv_path = OUT / "taiwan_v59_network_geometry_results.csv"
    checks = {
        "csv_hash_matches": digest(csv_path) == current["verification"]["result_sha256"],
        "runtime_validation_pass": current["verification"]["runtime_validation"].endswith("PASS"),
        "calendar_decomposition": decomposition <= 1.0e-9,
        "probabilities_bounded": all(
            0.0 <= row["conditional_broad_control_rate"] <= 1.0
            for row in current["conditional"]
        ),
        "macro_converged": residuals["macro_nonconverged_months"] == 0,
        "mfg_converged": residuals["mfg_nonconverged_months"] == 0,
        "macro_outer": residuals["macro_outer"] <= 3.0e-3,
        "cge": residuals["cge"] <= 2.0e-3,
        "dsge": residuals["dsge"] <= 8.0e-4,
        "mfg": residuals["mfg"] <= 5.0e-4,
        "transition_rows": residuals["transition_row"] <= 1.0e-12,
        "positive_intensity": residuals["minimum_intensity"] > 0.0,
        "actor_spectral_gap": residuals["minimum_actor_spectral_gap"] >= -1.0e-12,
        "supra_spectral_gap": residuals["minimum_supra_spectral_gap"] >= -1.0e-12,
        "conductance": 0.0 <= residuals["minimum_conductance"] <= 1.0,
        "sheaf_psd": residuals["minimum_sheaf_eigenvalue"] >= -1.0e-12,
        "temporal_step": residuals["maximum_temporal_log_step"] <= 0.18 + 1.0e-12,
        "bottleneck_risk": 0.0 <= residuals["maximum_bottleneck_risk"] <= 1.0,
        "ablation_complete": set(variants) == expected_variants and ablation.get("status") == "PASS",
        "ablation_converged": ablation_converged,
    }
    checks = {key: bool(value) for key, value in checks.items()}
    payload = {
        "model_version": "V5.9",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "calendar_decomposition_residual": decomposition,
        "maximum_residuals_and_bounds": residuals,
        "v58_v59_differences": differences,
        "calendar_2100": current["verification"]["calendar_2100"],
        "intervention_probability_means": current["verification"]["intervention_probability_means"],
        "force_onset_mean": current["verification"]["force_onset_mean"],
        "ablation_path_count": ablation["paths"],
        "ablation_variant_count": len(variants),
    }
    TARGET.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"TAIWAN_V59_VALIDATION: {payload['status']}")
    if payload["status"] != "PASS":
        print(json.dumps({key: value for key, value in checks.items() if not value}, indent=2))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
