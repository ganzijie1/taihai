"""DIAGNOSTIC_ONLY: invariants for the V6.1 three-sector upgrade."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import simulate_trade_war_v43 as trade
from sectoral_economy_v61 import CoupledSectoralEconomicSystem


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "outputs" / "taiwan_v61_sectoral_diagnostic.json"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def inputs(paths: int, month: int) -> dict:
    intensity = np.array([0.05, 0.35, 0.65, 0.95])[:, None]
    phase = min(month / 8.0, 1.0)
    damage = np.clip(intensity * phase * np.array([[0.90, 1.05, 0.72, 0.82]]), 0.0, 0.75)
    financial = np.clip(0.04 + 0.48 * damage, 0.0, 0.75)
    panic = np.clip(0.05 + 0.42 * damage, 0.0, 0.80)
    social = np.clip(0.04 + 0.32 * damage, 0.0, 0.65)
    return {
        "month": month,
        "actor_damage": damage,
        "actor_logistics": np.clip(1.0 - 0.62 * damage, 0.18, 1.0),
        "actor_labor": np.clip(1.0 - 0.35 * damage, 0.25, 1.0),
        "actor_financial_stress": financial,
        "actor_payment": np.clip(1.0 - 0.46 * financial, 0.25, 1.0),
        "actor_fiscal": np.clip(0.94 - 0.36 * financial, 0.25, 1.0),
        "actor_social_action": social,
        "actor_panic": panic,
        "actor_mobilization": np.clip(0.08 + 0.52 * damage, 0.0, 0.75),
        "blockade": np.clip(np.linspace(0.0, 0.58, paths) * phase, 0.0, 0.85),
        "max_outer_iterations": 18,
        "tolerance": 3.0e-3,
    }


def run_model(**switches):
    paths = 4
    scenario = list(trade.SCENARIOS.values())[0]
    model = CoupledSectoralEconomicSystem(
        paths, 61_001, scenario, econometric_draws=None, **switches
    )
    factors = []
    for month in range(18):
        factors.append(model.update(**inputs(paths, month)))
    return model, factors


def main() -> None:
    model, factors = run_model()
    disabled, disabled_factors = run_model(
        agriculture_enabled=False,
        industry_enabled=False,
        services_enabled=False,
        labor_reallocation_enabled=False,
    )
    diag = model.diagnostics()
    sector = diag["three_sector_physical_economy"]
    dq = diag["mr_rd_cge_dq_sfc"]
    macro = {key: value for key, value in diag.items() if key != "mr_rd_cge_dq_sfc"}

    final = factors[-1]
    disabled_final = disabled_factors[-1]
    finite = all(
        np.all(np.isfinite(getattr(final, field)))
        for field in final.__dataclass_fields__
        if isinstance(getattr(final, field), np.ndarray)
    )
    switch_difference = max(
        float(np.max(np.abs(final.gdp_factor - disabled_final.gdp_factor))),
        float(np.max(np.abs(final.industry_factor - disabled_final.industry_factor))),
        float(np.max(np.abs(final.inflation - disabled_final.inflation))),
    )
    service_stock = float(np.max(np.abs(
        model.dqsfc.inventory[:, model.sectoral.service_node_mask]
    )))
    checks = {
        "finite_macro_outputs": finite,
        "sector_clock_advanced": sector["months"] == 18,
        "labor_simplex": sector["max_labor_share_residual"] < 1.0e-12,
        "service_is_nonstorable": service_stock < 1.0e-12,
        "service_stock_residual": sector["max_service_stock_residual"] < 1.0e-12,
        "node_realization_bounds": sector["max_node_bound_violation"] < 1.0e-12,
        "sector_actor_accounting": sector["max_sector_accounting_residual"] < 1.0e-10,
        "water_bounds": 0.12 <= sector["minimum_water"] <= 1.12,
        "soil_bounds": 0.35 <= sector["minimum_soil"] <= 1.05,
        "industrial_operability_bounds": 0.08 <= sector["minimum_industrial_operability"] <= 1.10,
        "land_simplex": sector["max_land_share_residual"] < 1.0e-12,
        "firm_mass_positive": sector["minimum_firm_mass"] >= 0.15,
        "industrial_quality_positive": sector["minimum_industrial_quality"] >= 0.55,
        "digital_adoption_bounds": (
            0.05 <= sector["minimum_digital_adoption"]
            <= sector["maximum_digital_adoption"] <= 0.96
        ),
        "public_service_capacity_positive": (
            sector["minimum_public_service_capacity"] >= 0.10
        ),
        "baumol_pressure_bounded": (
            0.0 <= sector["maximum_baumol_cost_pressure"] <= 1.50
        ),
        "service_queue_exercised": sector["maximum_service_wait"] > 0.0,
        "service_expiry_exercised": sector["maximum_expired_service"] >= 0.0,
        "dq_material_balance": dq["max_material_residual"] < 1.0e-10,
        "sfc_balance": dq["max_sfc_residual"] < 1.0e-10,
        "sam_balance": (
            dq["social_accounting_matrix"]["max_dynamic_balance_residual"]
            < 1.0e-10
        ),
        "module_switch_changes_outcome": switch_difference > 1.0e-6,
    }
    checks = {key: bool(value) for key, value in checks.items()}
    payload = {
        "classification": "DIAGNOSTIC_ONLY; no outcome probabilities",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "switch_difference": switch_difference,
        "service_inventory_max": service_stock,
        "macro": macro,
        "dq_sfc": dq,
        "three_sector": sector,
        "disabled_three_sector": disabled.sectoral.diagnostics(),
    }
    TARGET.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"TAIWAN_V61_SECTORAL_DIAGNOSTIC: {payload['status']}")
    if payload["status"] != "PASS":
        print(json.dumps({key: value for key, value in checks.items() if not value}, indent=2))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
