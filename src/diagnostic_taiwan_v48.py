"""DIAGNOSTIC_ONLY: mechanism constraints, perturbations, and all-path coupling."""

from __future__ import annotations

import numpy as np

import diagnostic_taiwan_v47
from dynamic_mechanism_design_v48 import RobustDynamicMechanismDesigner


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def inputs(paths: int, stress: float = 0.35) -> dict:
    return {
        "month": 0,
        "actor_support": np.full((paths, 4), 0.58),
        "actor_damage": np.full((paths, 4), stress * 0.45),
        "actor_shortage": np.full((paths, 4), stress),
        "actor_output": np.full((paths, 4), 0.82 - 0.25 * stress),
        "financial_capacity": np.full((paths, 4), 0.72 - 0.20 * stress),
        "debt_stress": np.full((paths, 4), 0.22 + 0.35 * stress),
        "leader_continuity": np.full((paths, 4), 0.86),
        "political_resilience": np.full((paths, 4), 0.74 - 0.15 * stress),
        "conflict_regime": np.zeros(paths, dtype=np.int8),
        "regime_duration": np.full(paths, 12),
        "land_control": np.full(paths, 0.24),
        "proxy_capacity": np.full(paths, 0.12),
        "proxy_leakage": np.full(paths, 0.30),
        "trade_loss": np.full(paths, stress * 0.65),
        "prior_blockade": np.full(paths, stress),
        "us_ceiling": 0.80,
        "japan_ceiling": 0.66,
    }


def main() -> None:
    # Includes the one-month full V4.7 path with the mechanism object attached.
    diagnostic_taiwan_v47.main()
    paths = 12
    base = RobustDynamicMechanismDesigner(paths, 401)
    low = base.update(**inputs(paths, 0.15))
    stressed = RobustDynamicMechanismDesigner(paths, 401)
    high = stressed.update(**inputs(paths, 0.75))
    require(
        np.mean(high.responsibility[:, 1:].sum(axis=1))
        > np.mean(low.responsibility[:, 1:].sum(axis=1)),
        "robust burden target must respond to stress",
    )
    require(
        np.mean(high.insurance_transfer[:, 1])
        >= np.mean(low.insurance_transfer[:, 1]),
        "indexed transfer must respond to insured loss",
    )
    require(np.max(np.abs(high.insurance_transfer.sum(axis=1))) < 1e-12, "insurance clearing")
    require(np.max(np.abs(high.transfer.sum(axis=1))) < 1e-12, "VCG transfer clearing")
    ceasefire_inputs = inputs(paths, 0.55)
    ceasefire_inputs["conflict_regime"] = np.full(paths, 2, dtype=np.int8)
    ceasefire = RobustDynamicMechanismDesigner(paths, 401).update(**ceasefire_inputs)
    high_conflict = RobustDynamicMechanismDesigner(paths, 401).update(**inputs(paths, 0.55))
    require(
        np.mean(ceasefire.sanction_relief) > np.mean(high_conflict.sanction_relief),
        "sanction relief must increase in ceasefire state",
    )
    diagnostic = stressed.diagnostics()
    for key in (
        "max_incentive_compatibility_residual",
        "max_participation_residual",
        "max_budget_residual",
        "max_resource_residual",
        "max_limited_liability_residual",
        "max_dynamic_commitment_residual",
        "max_transfer_balance_residual",
    ):
        require(diagnostic[key] < 1e-9, key)

    # Every documented submechanism must have a neutral, testable off-state.
    for module in RobustDynamicMechanismDesigner.MODULES:
        off = RobustDynamicMechanismDesigner(paths, 401, {module: False})
        factor = off.update(**inputs(paths, 0.55))
        if module == "alliance_vcg":
            require(np.allclose(factor.alliance_multiplier, 1.0), "alliance neutralization")
        elif module == "aid_contract":
            require(np.allclose(factor.aid_multiplier, 1.0), "aid neutralization")
        elif module == "procurement":
            require(np.allclose(factor.procurement_factor, 1.0), "procurement neutralization")
        elif module == "information_design":
            require(np.allclose(factor.information_support_shift, 0.0), "information neutralization")
        elif module == "ceasefire_contract":
            require(np.allclose(factor.ceasefire_break_multiplier, 1.0), "ceasefire neutralization")
        elif module == "governance_contract":
            require(np.allclose(factor.governance_trust_bonus, 0.0), "governance neutralization")
        elif module == "trade_insurance":
            require(np.allclose(factor.insurance_transfer, 0.0), "insurance neutralization")
    print("DIAGNOSTIC_ONLY_TAIWAN_V48: PASS")


if __name__ == "__main__":
    main()
