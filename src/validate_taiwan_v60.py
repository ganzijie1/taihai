from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
CURRENT = OUT / "taiwan_v60_riemannian_simulation.json"
BASELINE = OUT / "taiwan_v59_network_geometry_simulation.json"
ABLATION = OUT / "taiwan_v60_riemannian_ablations.json"
TARGET = OUT / "taiwan_v60_validation_and_v59_comparison.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main() -> None:
    current = json.loads(CURRENT.read_text(encoding="utf-8"))
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    ablation = json.loads(ABLATION.read_text(encoding="utf-8"))
    diagnostics = list(current["module_diagnostics"].values())

    residuals = {
        "macro_outer": max(item["macro"]["max_outer_residual"] for item in diagnostics),
        "cge": max(item["macro"]["max_cge_solver_residual"] for item in diagnostics),
        "dsge": max(item["macro"]["max_dsge_solver_residual"] for item in diagnostics),
        "mfg": max(item["mfg"]["max_fixed_point_residual"] for item in diagnostics),
        "macro_nonconverged_months": sum(item["macro"]["nonconverged_months"] for item in diagnostics),
        "mfg_nonconverged_months": sum(item["mfg"]["nonconverged_months"] for item in diagnostics),
        "ram_tangency": max(
            item["mfg"]["riemannian_anderson"]["maximum_transport_tangency_residual"]
            for item in diagnostics
        ),
        "ram_sphere": max(
            item["mfg"]["riemannian_anderson"]["maximum_sphere_residual"]
            for item in diagnostics
        ),
        "pnt_barycenter": max(item["pnt"]["max_barycenter_residual"] for item in diagnostics),
        "pnt_source_weight": max(item["pnt"]["max_source_weight_residual"] for item in diagnostics),
        "pnt_minimum_eigenvalue": min(item["pnt"]["minimum_covariance_eigenvalue"] for item in diagnostics),
        "product_metric_minimum_eigenvalue": min(
            item["geometry"]["minimum_product_metric_eigenvalue"] for item in diagnostics
        ),
        "regime_transition_count": sum(
            item["geometry"]["regime_transition_count"] for item in diagnostics
        ),
        "maximum_geodesic_reconstruction_error": max(
            item["geometry"]["maximum_geodesic_reconstruction_error"]
            for item in diagnostics
        ),
    }
    calendar_residual = max(
        abs(
            item["force_onset_cif"] + item["peace_arrangement_cif"]
            + item["legal_separation_cif"] + item["status_quo_survival"] - 1.0
        ) for item in current["verification"]["calendar_2100"]
    )

    old = {
        (item["trade_scenario"], item["horizon_years"]): item
        for item in baseline["aggregate"]
    }
    differences = []
    for item in current["aggregate"]:
        prior = old[(item["trade_scenario"], item["horizon_years"])]
        differences.append({
            "trade_scenario": item["trade_scenario"],
            "horizon_years": item["horizon_years"],
            "v60_conditional": item["conditional_mean"],
            "v59_conditional": prior["conditional_mean"],
            "conditional_difference": item["conditional_mean"] - prior["conditional_mean"],
            "v60_unconditional": item["unconditional_mean"],
            "v59_unconditional": prior["unconditional_mean"],
            "unconditional_difference": item["unconditional_mean"] - prior["unconditional_mean"],
        })

    checks = {
        "runtime_validation": current["verification"].get("runtime_validation") == "TAIWAN_V60_RUNTIME_VALIDATION: PASS",
        "formal_cell_count": len(current["module_diagnostics"]) == 30,
        "aggregate_count": len(current["aggregate"]) == 30,
        "macro_outer": residuals["macro_outer"] <= 0.003,
        "cge": residuals["cge"] <= 0.002,
        "dsge": residuals["dsge"] <= 0.0008,
        "mfg": residuals["mfg"] <= 0.0005,
        "no_nonconvergence": (
            residuals["macro_nonconverged_months"] == 0
            and residuals["mfg_nonconverged_months"] == 0
        ),
        "ram_tangency": residuals["ram_tangency"] <= 1.0e-10,
        "ram_sphere": residuals["ram_sphere"] <= 1.0e-10,
        "pnt_barycenter": residuals["pnt_barycenter"] <= 5.0e-7,
        "pnt_source_weights": residuals["pnt_source_weight"] <= 1.0e-12,
        "pnt_spd": residuals["pnt_minimum_eigenvalue"] > 0.0,
        "product_metric_positive": residuals["product_metric_minimum_eigenvalue"] > 0.0,
        "regime_pod_active": residuals["regime_transition_count"] > 0,
        "geodesic_reconstruction_finite": residuals["maximum_geodesic_reconstruction_error"] < float("inf"),
        "calendar_decomposition": calendar_residual <= 1.0e-10,
        "ablation_complete": (
            ablation.get("status") == "PASS" and len(ablation.get("variants", [])) == 8
        ),
        "ablation_converged": all(
            item["mfg_nonconverged_months"] == 0
            and item["calendar_decomposition_residual"] <= 1.0e-10
            for item in ablation.get("variants", [])
        ),
    }
    checks = {key: bool(value) for key, value in checks.items()}
    payload = {
        "model_version": "V6.0",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "maximum_residuals_and_bounds": residuals,
        "calendar_decomposition_residual": calendar_residual,
        "v59_v60_differences": differences,
        "maximum_absolute_conditional_difference": max(
            abs(item["conditional_difference"]) for item in differences
        ),
        "maximum_absolute_unconditional_difference": max(
            abs(item["unconditional_difference"]) for item in differences
        ),
        "calendar_2100": current["verification"]["calendar_2100"],
        "result_sha256": sha256(CURRENT),
        "ablation_sha256": sha256(ABLATION),
    }
    TARGET.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"TAIWAN_V60_VALIDATION: {payload['status']}")
    if payload["status"] != "PASS":
        print(json.dumps({key: value for key, value in checks.items() if not value}, indent=2))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
