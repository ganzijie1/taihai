from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

import simulate_taiwan_fully_closed_v47 as core
import simulate_trade_war_v43 as trade
from dynamic_macro_dqsfc_v53 import CoupledMRRDCGEDQSFCEconomicSystem
from econometric_calibration_v55 import load_draws
from simulate_taiwan_fully_closed_v52 import MFG_CONFIG


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
TRACE = OUT / "taiwan_v55_traceability.json"
AUDIT = OUT / "taiwan_v55_static_coupling_audit.json"
SPEC = OUT / "taiwan_v55_ablation_specification.json"
TARGET = OUT / "taiwan_v55_econometric_ablations.json"


def scenario_key(code: str) -> str:
    return next(key for key, value in trade.SCENARIOS.items() if value == code)


def draw_variants(base: dict[str, np.ndarray]):
    variants = {}
    current = {key: value.copy() for key, value in base.items()}
    current["trade_shock_scale"] = np.ones_like(current["trade_shock_scale"])
    variants["remove_ppml_trade_uncertainty"] = current

    current = {key: value.copy() for key, value in base.items() if key != "hmm_transition"}
    variants["remove_hmm_transition_posterior"] = current

    current = {key: value.copy() for key, value in base.items() if key != "growth_rate"}
    variants["remove_growth_calibration"] = current

    current = {key: value.copy() for key, value in base.items()}
    current["macro_output_scale"] = np.ones_like(current["macro_output_scale"])
    current["macro_inflation_scale"] = np.ones_like(current["macro_inflation_scale"])
    variants["remove_macro_lp_var_posterior"] = current

    current = {key: value.copy() for key, value in base.items()}
    for key in ("force_beta_scale", "peace_beta_scale", "separation_beta_scale"):
        current[key] = np.ones_like(current[key])
    for key in ("us_intervention_shift", "japan_intervention_shift"):
        current[key] = np.zeros_like(current[key])
    variants["remove_terminal_and_alliance_prior_uncertainty"] = current
    return variants


def diagnostic_maxima(diagnostics: dict) -> dict:
    values = list(diagnostics.values())
    return {
        "macro_outer_residual": max(x["macro"]["max_outer_residual"] for x in values),
        "cge_residual": max(x["macro"]["max_cge_solver_residual"] for x in values),
        "dsge_residual": max(x["macro"]["max_dsge_solver_residual"] for x in values),
        "mfg_residual": max(x["mfg"]["max_fixed_point_residual"] for x in values),
        "macro_nonconverged_months": max(x["macro"]["nonconverged_months"] for x in values),
        "mfg_nonconverged_months": max(x["mfg"]["nonconverged_months"] for x in values),
    }


def main() -> None:
    if os.environ.get("TAIWAN_ALLOW_COMPLETE_RUN") != "1":
        raise RuntimeError("V5.5 ablations require the passed complete-model gate")
    specification = json.loads(SPEC.read_text(encoding="utf-8"))
    if specification.get("status") != "FROZEN":
        raise RuntimeError("V5.5 ablation specification is not frozen")
    paths = int(specification["paths"])
    months = int(specification["months"])
    scenarios = [scenario_key(code) for code in specification["trade_scenarios"]]
    payload = {"model_version": "V5.5", "scope": "validation ablations", "ablations": {}}
    for name, draws in draw_variants(load_draws(paths)).items():
        rows, aggregate, diagnostics, verification = core.run(
            paths=paths, months=months, scenario_names=scenarios,
            mechanism_enabled=True, energy_enabled=True, geometry_enabled=True,
            hybrid_enabled=True, topology_enabled=True, mfg_config=MFG_CONFIG,
            macro_factory=CoupledMRRDCGEDQSFCEconomicSystem,
            econometric_draws=draws, traceability_path=TRACE,
            coupling_audit_path=AUDIT,
        )
        payload["ablations"][name] = {
            "aggregate": aggregate,
            "calendar_2100": verification["calendar_2100"],
            "force_onset_mean": verification["force_onset_mean"],
            "intervention_probability_means": verification["intervention_probability_means"],
            "maximum_residuals": diagnostic_maxima(diagnostics),
        }
        TARGET.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"TAIWAN_V55_ABLATION_COMPLETE {name}", flush=True)
    payload["status"] = "PASS"
    TARGET.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("TAIWAN_V55_ABLATIONS: PASS")


if __name__ == "__main__":
    main()
