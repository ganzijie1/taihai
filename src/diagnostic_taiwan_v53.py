from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "work"))

from dynamic_macro_dqsfc_v53 import CoupledMRRDCGEDQSFCEconomicSystem
from dynamic_macro_equilibrium_v47 import CoupledMacroEconomicSystem


OUT = ROOT / "outputs" / "taiwan_v53_diagnostic_only.json"


def inputs(paths: int, shock: bool = False) -> dict:
    ones = np.ones((paths, 4))
    zeros = np.zeros((paths, 4))
    damage = zeros.copy()
    logistics = ones.copy()
    financial = zeros.copy()
    panic = zeros.copy()
    blockade = np.zeros(paths)
    if shock:
        damage[:, 1] = 0.24
        logistics[:, 1] = 0.48
        financial[:, 1] = 0.22
        panic[:, 1] = 0.18
        blockade[:] = 0.36
    return {
        "actor_damage": damage,
        "actor_logistics": logistics,
        "actor_labor": ones.copy(),
        "actor_financial_stress": financial,
        "actor_payment": ones.copy(),
        "actor_fiscal": ones.copy(),
        "actor_social_action": zeros.copy(),
        "actor_panic": panic,
        "actor_mobilization": zeros.copy(),
        "blockade": blockade,
    }


def run_path(model, months: int, shock: bool) -> list:
    states = []
    for month in range(months):
        result = model.update(month=month, **inputs(model.paths, shock=shock))
        states.append(result.gdp_factor.copy())
    return states


def run() -> dict:
    seed = 530_001
    neutral = CoupledMRRDCGEDQSFCEconomicSystem(2, seed, "和平基线")
    neutral_states = run_path(neutral, 4, False)
    neutral_diag = neutral.diagnostics()

    replay_a = CoupledMRRDCGEDQSFCEconomicSystem(2, seed + 1, "封锁_制裁_多节点同步冲击")
    replay_b = CoupledMRRDCGEDQSFCEconomicSystem(2, seed + 1, "封锁_制裁_多节点同步冲击")
    state_a = run_path(replay_a, 4, True)
    state_b = run_path(replay_b, 4, True)
    deterministic_error = float(max(
        np.max(np.abs(a - b)) for a, b in zip(state_a, state_b)
    ))
    shock_diag = replay_a.diagnostics()

    disabled = CoupledMRRDCGEDQSFCEconomicSystem(
        2, seed, "和平基线", dqsfc_enabled=False
    )
    disabled_states = run_path(disabled, 4, False)
    legacy = CoupledMacroEconomicSystem(2, seed, "和平基线")
    legacy_states = run_path(legacy, 4, False)
    zero_shock_difference = float(np.max(np.abs(
        disabled_states[-1] - legacy_states[-1]
    )))
    enabled_structural_difference = float(np.max(np.abs(
        neutral_states[-1] - legacy_states[-1]
    )))
    dq = shock_diag["mr_rd_cge_dq_sfc"]
    neutral_dq = neutral_diag["mr_rd_cge_dq_sfc"]
    sam = dq["social_accounting_matrix"]
    assertions = {
        "deterministic_seed_replay": deterministic_error < 1.0e-13,
        "material_conservation": dq["max_material_residual"] < 1.0e-10,
        "sfc_double_entry": dq["max_sfc_residual"] < 1.0e-10,
        "sam_has_full_account_set": sam["account_count"] == 19,
        "sam_benchmark_balanced": sam["benchmark_balance_residual"] < 1.0e-8,
        "sam_ras_converged": sam["benchmark_ras_residual"] < 1.0e-8,
        "sam_dynamic_accounts_balanced": sam["max_dynamic_balance_residual"] < 1.0e-9,
        "sam_saving_investment_closed": sam["max_saving_investment_residual"] < 1.0e-9,
        "sam_nonnegative": (
            sam["benchmark_minimum_cell"] >= -1.0e-12
            and sam["minimum_transaction"] >= -1.0e-12
        ),
        "capital_queue_conservation": dq["max_capital_queue_residual"] < 1.0e-10,
        "inventory_nonnegative": dq["minimum_inventory"] >= -1.0e-12,
        "backlog_nonnegative": dq["minimum_backlog"] >= -1.0e-12,
        "employment_bounded": 0.35 <= dq["minimum_employment"] <= 1.0,
        "age_complementarity": dq["max_age_complementarity_residual"] < 5.0e-3,
        "jorgenson_adding_up": dq["max_jorgenson_share_residual"] < 1.0e-12,
        "hssw_expenditure_identity": dq["max_hssw_identity_residual"] < 1.0e-10,
        "dq_inner_iteration_finite": dq["max_inner_residual"] < 5.0e-4,
        "zero_shock_regression": zero_shock_difference < 1.0e-13,
        "explicit_lag_declared": dq["feedback_lag"].startswith("one explicit month"),
        "no_duplicate_bank_debt_owner": (
            dq["state_owners"]["bank_capital_public_debt_reserves"]
            == "FourPartyFinancialSystem"
        ),
        "sam_unique_owner": (
            dq["state_owners"]["sam_transactions_and_institutional_distribution"]
            == "FourPartySocialAccountingMatrix"
        ),
        "neutral_macro_converged": neutral_diag["nonconverged_months"] == 0,
        "shock_macro_converged": shock_diag["nonconverged_months"] == 0,
        "shock_creates_disequilibrium_state": dq["maximum_backlog"] > 0.0,
        "neutral_dq_exercised": neutral_dq["months"] == 4,
    }
    result = {
        "label": "DIAGNOSTIC_ONLY",
        "substantive_outcomes_generated": False,
        "status": "PASS" if all(assertions.values()) else "FAIL",
        "assertions": {key: bool(value) for key, value in assertions.items()},
        "deterministic_error": deterministic_error,
        "zero_shock_difference": zero_shock_difference,
        "enabled_structural_difference": enabled_structural_difference,
        "neutral": neutral_diag,
        "shock": shock_diag,
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    report = run()
    print(f"TAIWAN_V53_DIAGNOSTIC_ONLY: {report['status']}")
    if report["status"] != "PASS":
        print(json.dumps(report["assertions"], ensure_ascii=False, indent=2))
        raise SystemExit(1)
