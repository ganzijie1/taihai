from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from closed_prewar_system_v47 import ClosedPrewarEvolutionSystem
from competing_risk_survival_v47 import StratifiedCompetingRiskSurvival
from dynamic_macro_dqsfc_v53 import CoupledMRRDCGEDQSFCEconomicSystem
from econometric_calibration_v55 import DRAWS, load_draws
from endogenous_intervention_v47 import EndogenousInterventionChoiceModel


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "taiwan_v55_econometric_integration_diagnostic.json"


def main() -> None:
    paths = 4
    all_draws = np.load(DRAWS)
    draws = {key: all_draws[key][:paths].copy() for key in all_draws.files}
    anchor = {
        "actor_support": np.full((paths, 4), 0.50),
        "taiwan_readiness": np.ones(paths),
    }
    prewar = ClosedPrewarEvolutionSystem(
        paths, 55, anchor, econometric_draws=draws
    )
    prewar_growth_draw_identity = bool(
        np.allclose(prewar.growth_rate, np.clip(draws["growth_rate"], -2.5, 8.5))
    )
    prewar.update(0)
    survival = StratifiedCompetingRiskSurvival(paths, 56, econometric_draws=draws)
    intervention = EndogenousInterventionChoiceModel(paths, 57, econometric_draws=draws)
    macro = CoupledMRRDCGEDQSFCEconomicSystem(
        paths, 58, "central", econometric_draws=draws
    )
    zeros = np.zeros((paths, 4))
    ones = np.ones((paths, 4))
    macro_result = macro.update(
        month=0,
        actor_damage=zeros,
        actor_logistics=ones,
        actor_labor=ones,
        actor_financial_stress=zeros,
        actor_payment=ones,
        actor_fiscal=ones,
        actor_social_action=zeros,
        actor_panic=zeros,
        actor_mobilization=zeros,
        blockade=np.zeros(paths),
    )
    checks = {
        "draw_file_path_count": int(load_draws(128)["growth_rate"].shape[0]),
        "prewar_probability_residual": float(
            np.max(np.abs(prewar.pressure_belief.sum(axis=1) - 1.0))
        ),
        "prewar_growth_draw_identity": prewar_growth_draw_identity,
        "survival_draw_identity": bool(
            np.allclose(survival.beta_scales["force"], draws["force_beta_scale"])
        ),
        "intervention_draw_identity": bool(
            np.allclose(intervention.us_intercept_shift, draws["us_intervention_shift"])
        ),
        "macro_draw_identity": bool(
            np.allclose(macro.cge.trade_shock_scale, draws["trade_shock_scale"])
        ),
        "macro_zero_input_finite": bool(
            np.all(np.isfinite(macro_result.gdp_factor))
            and np.all(np.isfinite(macro_result.inflation))
        ),
        "macro_zero_input_converged": bool(macro_result.converged),
        "macro_zero_input_accounting_residual": float(
            max(
                macro_result.cge_goods_residual,
                macro_result.cge_factor_residual,
                macro_result.dsge_residual,
                macro_result.external_account_residual,
            )
        ),
    }
    failures = [
        key for key, value in checks.items()
        if (isinstance(value, bool) and not value)
        or (key == "prewar_probability_residual" and value > 1e-10)
    ]
    payload = {
        "scope": "DIAGNOSTIC_ONLY; no campaign outcomes or probabilities",
        "checks": checks,
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if failures:
        raise RuntimeError("V5.5 econometric integration diagnostic failed: " + ", ".join(failures))
    print("V5.5 ECONOMETRIC INTEGRATION: DIAGNOSTIC_ONLY PASS")


if __name__ == "__main__":
    main()
