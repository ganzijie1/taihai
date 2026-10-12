from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path

import simulate_taiwan_fully_closed_v58 as model


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
TARGET = OUT / "taiwan_v58_information_geometry_ablations.json"


VARIANTS = {
    "complete_v58": {},
    "without_belief_filter": {"belief_filter_enabled": False},
    "without_credibility_hmm": {"credibility_hmm_enabled": False},
    "without_fisher_kl_trust_region": {"trust_region_enabled": False},
    "without_joint_wfr": {"full_wfr_jko": False, "wfr_strang": True},
    "without_schrodinger_bridge": {"schrodinger_bridge_enabled": False},
    "without_fgw_alignment": {"fgw_alignment_enabled": False},
    "without_divergence_robustness": {"robust_information_enabled": False},
    "without_information_value": {"information_value_enabled": False},
    "without_all_v58_information_geometry": {
        "partial_observation_geometry": False,
        "full_wfr_jko": False,
        "wfr_strang": True,
        "trust_region_enabled": False,
    },
}


def selected_diagnostics(diagnostics: dict) -> dict:
    values = list(diagnostics.values())
    return {
        "max_macro_residual": max(item["macro"]["max_outer_residual"] for item in values),
        "max_cge_residual": max(item["macro"]["max_cge_solver_residual"] for item in values),
        "max_dsge_residual": max(item["macro"]["max_dsge_solver_residual"] for item in values),
        "max_mfg_residual": max(item["mfg"]["max_fixed_point_residual"] for item in values),
        "macro_nonconverged_months": max(item["macro"]["nonconverged_months"] for item in values),
        "mfg_nonconverged_months": max(item["mfg"]["nonconverged_months"] for item in values),
    }


def main() -> None:
    if os.environ.get("TAIWAN_ALLOW_COMPLETE_RUN") != "1":
        raise RuntimeError("V5.8 ablation blocked until the complete-model gate passes")
    paths = int(os.environ.get("TAIWAN_V58_ABLATION_PATHS", "16"))
    output = {
        "classification": "post-gate common-random-number ablation; not a replacement for the 128-path level estimate",
        "paths": paths,
        "months": 360,
        "trade_scenarios": ["baseline"],
        "intervention_cases": "all five",
        "draw_subset_contract": "deterministic first paths of the 128 frozen V5.5 draws",
        "variants": {},
    }
    baseline_scenario = list(model.core.trade.v43.SCENARIOS)[0]
    for name, changes in VARIANTS.items():
        config = deepcopy(model.MFG_CONFIG)
        config.update(changes)
        frozen_draws = model.load_draws(128)
        draws = {key: value[:paths].copy() for key, value in frozen_draws.items()}
        rows, aggregate, diagnostics, verification = model.core.run(
            paths=paths,
            months=360,
            scenario_names=[baseline_scenario],
            mechanism_enabled=True,
            energy_enabled=True,
            geometry_enabled=True,
            hybrid_enabled=True,
            topology_enabled=True,
            mfg_config=config,
            geometry_config=model.GEOMETRY_CONFIG,
            macro_factory=model.CoupledMRRDCGEDQSFCEconomicSystem,
            political_factory=model.WartimeCollectiveActionRegimeSystem,
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
        TARGET.write_text(
            json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"TAIWAN_V58_ABLATION_VARIANT: {name} PASS", flush=True)
    output["status"] = "PASS"
    TARGET.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print("TAIWAN_V58_ABLATION: PASS")


if __name__ == "__main__":
    main()
