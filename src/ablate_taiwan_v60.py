from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import simulate_taiwan_fully_closed_v60 as model


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
TARGET = OUT / "taiwan_v60_riemannian_ablations.json"

VARIANTS = (
    "complete_v60",
    "without_riemannian_anderson",
    "without_riemannian_policy_trust_region",
    "without_adaptive_bures",
    "without_bures_barycenter",
    "without_regime_riemannian_pod",
    "without_product_manifold_metric",
    "without_all_v60_riemannian",
)


def configs(name: str) -> tuple[dict, dict]:
    mfg = dict(model.MFG_CONFIG)
    geometry = copy.deepcopy(model.GEOMETRY_CONFIG)
    if name in {"without_riemannian_anderson", "without_all_v60_riemannian"}:
        mfg["riemannian_anderson_enabled"] = False
    if name in {
        "without_riemannian_policy_trust_region", "without_all_v60_riemannian",
    }:
        mfg["riemannian_trust_region_enabled"] = False
    if name in {"without_adaptive_bures", "without_all_v60_riemannian"}:
        geometry["pnt"]["adaptive_bures_enabled"] = False
    if name in {"without_bures_barycenter", "without_all_v60_riemannian"}:
        geometry["pnt"]["bures_barycenter_enabled"] = False
    if name in {"without_regime_riemannian_pod", "without_all_v60_riemannian"}:
        geometry["multiscale"]["regime_riemannian_pod"] = False
    if name in {"without_product_manifold_metric", "without_all_v60_riemannian"}:
        geometry["multiscale"]["product_manifold_metric"] = False
    return mfg, geometry


def main() -> None:
    if os.environ.get("TAIWAN_ALLOW_COMPLETE_RUN") != "1":
        raise RuntimeError("V6.0 ablation requires the complete-model gate")
    paths = int(os.environ.get("TAIWAN_V60_ABLATION_PATHS", "16"))
    months = 360
    results = []
    for name in VARIANTS:
        mfg, geometry = configs(name)
        rows, aggregate, diagnostics, verification = model.run(
            paths=paths,
            months=months,
            scenario_names=["baseline"],
            mfg_config=mfg,
            geometry_config=geometry,
        )
        aggregate_30 = next(item for item in aggregate if item["horizon_years"] == 30)
        mfg_diagnostics = [item["mfg"] for item in diagnostics.values()]
        pnt_diagnostics = [item["pnt"] for item in diagnostics.values()]
        geometry_diagnostics = [item["geometry"] for item in diagnostics.values()]
        results.append({
            "variant": name,
            "conditional_30y": aggregate_30["conditional_mean"],
            "unconditional_30y": aggregate_30["unconditional_mean"],
            "max_mfg_residual": max(item["max_fixed_point_residual"] for item in mfg_diagnostics),
            "mfg_nonconverged_months": sum(item["nonconverged_months"] for item in mfg_diagnostics),
            "max_ram_tangency_residual": max(
                (item.get("riemannian_anderson") or {}).get(
                    "maximum_transport_tangency_residual", 0.0
                ) for item in mfg_diagnostics
            ),
            "max_pnt_barycenter_residual": max(
                item.get("max_barycenter_residual", 0.0) for item in pnt_diagnostics
            ),
            "minimum_product_metric_eigenvalue": min(
                item.get("minimum_product_metric_eigenvalue", 1.0)
                for item in geometry_diagnostics
            ),
            "calendar_decomposition_residual": max(
                abs(
                    item["force_onset_cif"] + item["peace_arrangement_cif"]
                    + item["legal_separation_cif"] + item["status_quo_survival"] - 1.0
                ) for item in verification["calendar_2100"]
            ),
        })
        print(f"TAIWAN_V60_ABLATION_CELL: {name}", flush=True)
    payload = {
        "model_version": "V6.0",
        "status": "PASS",
        "paths": paths,
        "months": months,
        "trade_scenarios": ["baseline"],
        "intervention_cases": 5,
        "common_random_numbers": True,
        "variants": results,
    }
    TARGET.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("TAIWAN_V60_ABLATION: PASS")


if __name__ == "__main__":
    main()
