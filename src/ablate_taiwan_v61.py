from __future__ import annotations

import json
import os
from pathlib import Path

import simulate_taiwan_fully_closed_v61 as model


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
TARGET = OUT / "taiwan_v61_sectoral_ablations.json"
CHECKPOINT = OUT / "taiwan_v61_sectoral_ablations.checkpoint.json"

VARIANTS = {
    "complete_v61": {},
    "without_agriculture": {"agriculture_enabled": False},
    "without_industry": {"industry_enabled": False},
    "without_services": {"services_enabled": False},
    "without_labor_reallocation": {"labor_reallocation_enabled": False},
    "without_structural_transformation": {"structural_transformation_enabled": False},
    "without_all_v61_sectoral": {"sectoral_layer_enabled": False},
}


def main() -> None:
    if os.environ.get("TAIWAN_ALLOW_COMPLETE_RUN") != "1":
        raise RuntimeError("V6.1 ablation requires the complete-model gate")
    paths = int(os.environ.get("TAIWAN_V61_ABLATION_PATHS", "16"))
    months = 360
    baseline_scenario = list(model.v60.v59.v58.core.trade.v43.SCENARIOS)[0]
    results = []
    for name, flags in VARIANTS.items():
        rows, aggregate, diagnostics, verification = model.run(
            paths=paths,
            months=months,
            scenario_names=[baseline_scenario],
            sector_flags=flags,
        )
        aggregate_30 = next(
            item for item in aggregate if item["horizon_years"] == 30
        )
        macro = [item["macro"] for item in diagnostics.values()]
        sector = [item["three_sector_physical_economy"] for item in macro]
        results.append({
            "variant": name,
            "flags": flags,
            "conditional_30y": aggregate_30["conditional_mean"],
            "unconditional_30y": aggregate_30["unconditional_mean"],
            "max_macro_residual": max(item["max_outer_residual"] for item in macro),
            "max_cge_residual": max(item["max_cge_solver_residual"] for item in macro),
            "max_dsge_residual": max(item["max_dsge_solver_residual"] for item in macro),
            "macro_nonconverged_months": sum(item["nonconverged_months"] for item in macro),
            "max_labor_share_residual": max(item["max_labor_share_residual"] for item in sector),
            "max_land_share_residual": max(item["max_land_share_residual"] for item in sector),
            "max_service_stock_residual": max(item["max_service_stock_residual"] for item in sector),
            "max_sector_accounting_residual": max(item["max_sector_accounting_residual"] for item in sector),
            "calendar_decomposition_residual": max(
                abs(
                    item["force_onset_cif"] + item["peace_arrangement_cif"]
                    + item["legal_separation_cif"] + item["status_quo_survival"] - 1.0
                ) for item in verification["calendar_2100"]
            ),
        })
        checkpoint_payload = {
            "model_version": "V6.1",
            "status": "IN_PROGRESS",
            "paths": paths,
            "months": months,
            "trade_scenarios": ["baseline"],
            "intervention_cases": 5,
            "common_random_numbers": True,
            "completed_variants": len(results),
            "variants": results,
        }
        checkpoint_tmp = CHECKPOINT.with_suffix(".tmp")
        checkpoint_tmp.write_text(
            json.dumps(checkpoint_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        checkpoint_tmp.replace(CHECKPOINT)
        print(f"TAIWAN_V61_ABLATION_CELL: {name}", flush=True)
    payload = {
        "model_version": "V6.1",
        "status": "PASS",
        "paths": paths,
        "months": months,
        "trade_scenarios": ["baseline"],
        "intervention_cases": 5,
        "common_random_numbers": True,
        "variants": results,
    }
    TARGET.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    CHECKPOINT.unlink(missing_ok=True)
    print("TAIWAN_V61_ABLATION: PASS")


if __name__ == "__main__":
    main()
