from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path

import simulate_taiwan_fully_closed_v59 as model


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
TARGET = OUT / "taiwan_v59_network_geometry_ablations.json"

VARIANTS = {
    "complete_v59": {},
    "without_intensity_transition": {"intensity_transition_enabled": False},
    "without_supra_laplacian": {"supra_laplacian_enabled": False},
    "without_spectral_geometry": {"spectral_geometry_enabled": False},
    "without_forman_curvature": {"forman_curvature_enabled": False},
    "without_sheaf_laplacian": {"sheaf_laplacian_enabled": False},
    "without_all_v59_network_geometry": {"network_geometry_enabled": False},
}


def selected_diagnostics(diagnostics: dict) -> dict:
    values = list(diagnostics.values())
    enabled = [item["mfg"].get("network_geometry") for item in values]
    enabled = [item for item in enabled if item is not None]
    return {
        "max_macro_residual": max(item["macro"]["max_outer_residual"] for item in values),
        "max_cge_residual": max(item["macro"]["max_cge_solver_residual"] for item in values),
        "max_dsge_residual": max(item["macro"]["max_dsge_solver_residual"] for item in values),
        "max_mfg_residual": max(item["mfg"]["max_fixed_point_residual"] for item in values),
        "macro_nonconverged_months": max(item["macro"]["nonconverged_months"] for item in values),
        "mfg_nonconverged_months": max(item["mfg"]["nonconverged_months"] for item in values),
        "max_transition_row_residual": max(
            (item["max_transition_row_residual"] for item in enabled), default=0.0
        ),
    }


def main() -> None:
    if os.environ.get("TAIWAN_ALLOW_COMPLETE_RUN") != "1":
        raise RuntimeError("V5.9 ablation blocked until complete-model gate passes")
    paths = int(os.environ.get("TAIWAN_V59_ABLATION_PATHS", "16"))
    output = {
        "classification": "post-gate common-random-number ablation; not a replacement for 128-path level estimate",
        "paths": paths,
        "months": 360,
        "trade_scenarios": ["baseline"],
        "intervention_cases": "all five",
        "variants": {},
    }
    baseline_scenario = list(model.v58.core.trade.v43.SCENARIOS)[0]
    for name, changes in VARIANTS.items():
        config = deepcopy(model.MFG_CONFIG)
        config.update(changes)
        frozen_draws = model.v58.load_draws(128)
        draws = {key: value[:paths].copy() for key, value in frozen_draws.items()}
        rows, aggregate, diagnostics, verification = model.v58.core.run(
            paths=paths,
            months=360,
            scenario_names=[baseline_scenario],
            mechanism_enabled=True,
            energy_enabled=True,
            geometry_enabled=True,
            hybrid_enabled=True,
            topology_enabled=True,
            mfg_config=config,
            geometry_config=model.v58.GEOMETRY_CONFIG,
            macro_factory=model.v58.CoupledMRRDCGEDQSFCEconomicSystem,
            political_factory=model.v58.WartimeCollectiveActionRegimeSystem,
            econometric_draws=draws,
            traceability_path=model.TRACE,
            coupling_audit_path=model.AUDIT,
        )
        output["variants"][name] = {
            "changes": changes,
            "aggregate": aggregate,
            "conditional": rows,
            "diagnostics": selected_diagnostics(diagnostics),
            "calendar_2100": verification["calendar_2100"],
        }
        TARGET.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"TAIWAN_V59_ABLATION_VARIANT: {name} PASS", flush=True)
    output["status"] = "PASS"
    TARGET.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print("TAIWAN_V59_ABLATION: PASS")


if __name__ == "__main__":
    main()
