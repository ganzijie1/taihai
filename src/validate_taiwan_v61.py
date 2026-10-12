from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
CURRENT = OUT / "taiwan_v61_sectoral_simulation.json"
BASELINE = OUT / "taiwan_v60_riemannian_simulation.json"
ABLATION = OUT / "taiwan_v61_sectoral_ablations.json"
TARGET = OUT / "taiwan_v61_validation_and_v60_comparison.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main() -> None:
    current = json.loads(CURRENT.read_text(encoding="utf-8"))
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    ablation = json.loads(ABLATION.read_text(encoding="utf-8"))
    diagnostics = list(current["module_diagnostics"].values())
    macro = [item["macro"] for item in diagnostics]
    sector = [item["three_sector_physical_economy"] for item in macro]
    mfg = [item["mfg"] for item in diagnostics]
    residuals = {
        "macro_outer": max(item["max_outer_residual"] for item in macro),
        "cge": max(item["max_cge_solver_residual"] for item in macro),
        "dsge": max(item["max_dsge_solver_residual"] for item in macro),
        "mfg": max(item["max_fixed_point_residual"] for item in mfg),
        "macro_nonconverged_months": sum(item["nonconverged_months"] for item in macro),
        "mfg_nonconverged_months": sum(item["nonconverged_months"] for item in mfg),
        "labor_share": max(item["max_labor_share_residual"] for item in sector),
        "land_share": max(item["max_land_share_residual"] for item in sector),
        "service_stock": max(item["max_service_stock_residual"] for item in sector),
        "node_bound": max(item["max_node_bound_violation"] for item in sector),
        "sector_accounting": max(item["max_sector_accounting_residual"] for item in sector),
        "minimum_water": min(item["minimum_water"] for item in sector),
        "minimum_soil": min(item["minimum_soil"] for item in sector),
        "minimum_firm_mass": min(item["minimum_firm_mass"] for item in sector),
        "minimum_public_service_capacity": min(item["minimum_public_service_capacity"] for item in sector),
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
            "v61_conditional": item["conditional_mean"],
            "v60_conditional": prior["conditional_mean"],
            "conditional_difference": item["conditional_mean"] - prior["conditional_mean"],
            "v61_unconditional": item["unconditional_mean"],
            "v60_unconditional": prior["unconditional_mean"],
            "unconditional_difference": item["unconditional_mean"] - prior["unconditional_mean"],
        })
    variants = ablation.get("variants", [])
    expected = {
        "complete_v61", "without_agriculture", "without_industry",
        "without_services", "without_labor_reallocation",
        "without_structural_transformation", "without_all_v61_sectoral",
    }
    ablation_converged = all(
        item["macro_nonconverged_months"] == 0
        and item["max_macro_residual"] <= 0.003
        and item["max_cge_residual"] <= 0.002
        and item["max_dsge_residual"] <= 0.0008
        and item["calendar_decomposition_residual"] <= 1.0e-10
        for item in variants
    )
    checks = {
        "runtime_validation": current["verification"].get("runtime_validation") == "TAIWAN_V61_RUNTIME_VALIDATION: PASS",
        "formal_cell_count": len(diagnostics) == 30,
        "aggregate_count": len(current["aggregate"]) == 30,
        "sector_clock_complete": all(item["months"] == 360 for item in sector),
        "macro_outer": residuals["macro_outer"] <= 0.003,
        "cge": residuals["cge"] <= 0.002,
        "dsge": residuals["dsge"] <= 0.0008,
        "mfg": residuals["mfg"] <= 0.0005,
        "no_nonconvergence": residuals["macro_nonconverged_months"] == 0 and residuals["mfg_nonconverged_months"] == 0,
        "labor_simplex": residuals["labor_share"] <= 1.0e-12,
        "land_simplex": residuals["land_share"] <= 1.0e-12,
        "service_nonstorable": residuals["service_stock"] <= 1.0e-12,
        "node_bounds": residuals["node_bound"] <= 1.0e-12,
        "sector_accounting": residuals["sector_accounting"] <= 1.0e-10,
        "physical_state_bounds": (
            residuals["minimum_water"] >= 0.12 - 1.0e-12
            and residuals["minimum_soil"] >= 0.35 - 1.0e-12
            and residuals["minimum_firm_mass"] >= 0.15 - 1.0e-12
            and residuals["minimum_public_service_capacity"] >= 0.10 - 1.0e-12
        ),
        "calendar_decomposition": calendar_residual <= 1.0e-10,
        "ablation_complete": ablation.get("status") == "PASS" and {item["variant"] for item in variants} == expected,
        "ablation_converged": ablation_converged,
    }
    checks = {key: bool(value) for key, value in checks.items()}
    payload = {
        "model_version": "V6.1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "maximum_residuals_and_bounds": residuals,
        "calendar_decomposition_residual": calendar_residual,
        "v60_v61_differences": differences,
        "maximum_absolute_conditional_difference": max(abs(item["conditional_difference"]) for item in differences),
        "maximum_absolute_unconditional_difference": max(abs(item["unconditional_difference"]) for item in differences),
        "calendar_2100": current["verification"]["calendar_2100"],
        "result_sha256": sha256(CURRENT),
        "ablation_sha256": sha256(ABLATION),
    }
    TARGET.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"TAIWAN_V61_VALIDATION: {payload['status']}")
    if payload["status"] != "PASS":
        print(json.dumps({key: value for key, value in checks.items() if not value}, indent=2))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
