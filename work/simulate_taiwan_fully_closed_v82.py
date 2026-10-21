from __future__ import annotations

import csv
import hashlib
import json
import os
from functools import partial
from pathlib import Path

import simulate_taiwan_fully_closed_v81 as v81
from sectoral_development_labor_v82 import ActorSpecificEconomicSystemV82


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
TRACE = OUT / "taiwan_v82_traceability.json"
AUDIT = OUT / "taiwan_v82_static_coupling_audit.json"
FREEZE = OUT / "taiwan_v82_frozen_specification.json"
OUT_CSV = OUT / "taiwan_v82_results.csv"
OUT_JSON = OUT / "taiwan_v82_simulation.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def complete_model_gate() -> tuple[bool, list[str]]:
    if not TRACE.exists() or not AUDIT.exists() or not FREEZE.exists():
        return False, ["V8.2 traceability, audit or frozen specification missing"]
    trace = json.loads(TRACE.read_text(encoding="utf-8"))
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    gaps = list(trace.get("blocking_gaps", []))
    passed = (
        trace.get("gate_status") == "PASS"
        and audit.get("simulation_authorized") is True
        and freeze.get("status") == "FROZEN"
        and not gaps
    )
    return passed, gaps


def validate_v82(diagnostics: dict, months: int) -> None:
    v81.validate_v81(diagnostics, months)
    errors: list[str] = []
    for cell, item in diagnostics.items():
        layer = item.get("macro", {}).get("three_sector_physical_economy", {}).get(
            "v82_actor_specific_economic_deepening"
        )
        if layer is None:
            errors.append(f"{cell}: missing V8.2 actor-specific layer")
            continue
        checks = {
            "clock": layer["months"] >= months,
            "enterprise simplex": layer["maximum_enterprise_share_residual"] <= 1.0e-12,
            "enterprise bounds": layer["maximum_probability_bound_violation"] <= 1.0e-12,
            "positive entitlement": layer["minimum_entitlement_access"] > 0.0,
            "bounded fatigue": 0.0 <= layer["maximum_fatigue"] <= 0.70 + 1.0e-12,
            "positive matching": layer["minimum_matching_wedge"] > 0.0,
            "no posthoc": not layer["terminal_probability_adjustment"],
        }
        errors.extend(f"{cell}: {name}" for name, ok in checks.items() if not ok)
    if errors:
        raise RuntimeError("V8.2 runtime validation failed: " + "; ".join(errors))


def run(paths: int = 128, months: int = 360, *,
        agriculture_deepening_enabled: bool = True,
        industry_deepening_enabled: bool = True,
        development_deepening_enabled: bool = True,
        labor_deepening_enabled: bool = True, **kwargs):
    gate, gaps = complete_model_gate()
    if not gate:
        raise RuntimeError(
            "V8.2 complete-model gate failed: "
            + "; ".join(gaps or ["gate not passed"])
        )
    v70 = v81.v80.v78.v77.v76.v75.v74.v73.v72.v71.v70
    old_macro = v70.AsymmetricAdjustmentMacroV70
    old_paths = (v81.TRACE, v81.AUDIT, v81.FREEZE)
    v70.AsymmetricAdjustmentMacroV70 = partial(
        ActorSpecificEconomicSystemV82,
        agriculture_deepening_enabled=agriculture_deepening_enabled,
        industry_deepening_enabled=industry_deepening_enabled,
        development_deepening_enabled=development_deepening_enabled,
        labor_deepening_enabled=labor_deepening_enabled,
    )
    v81.TRACE, v81.AUDIT, v81.FREEZE = TRACE, AUDIT, FREEZE
    try:
        result = v81.run(paths=paths, months=months, **kwargs)
    finally:
        v70.AsymmetricAdjustmentMacroV70 = old_macro
        v81.TRACE, v81.AUDIT, v81.FREEZE = old_paths
    validate_v82(result[2], months)
    return result


def write(rows, aggregate, diagnostics, verification) -> None:
    with OUT_CSV.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    verification["model_version"] = "V8.2-ACTOR-SPECIFIC-SECTOR-DEVELOPMENT-LABOR"
    verification["runtime_validation"] = "TAIWAN_V82_RUNTIME_VALIDATION: PASS"
    verification["result_sha256"] = sha256(OUT_CSV)
    verification["v82_upgrade"] = {
        "actor_specific_agriculture": True,
        "actor_specific_industry": True,
        "development_dynamics": True,
        "labor_search_skill_health": True,
        "existing_inventory_capital_and_employment_owners_preserved": True,
        "posthoc_probability_adjustment": False,
        "frozen_specification_sha256": sha256(FREEZE),
    }
    OUT_JSON.write_text(json.dumps({
        "verification": verification,
        "aggregate": aggregate,
        "conditional": rows,
        "module_diagnostics": diagnostics,
    }, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    if os.environ.get("TAIWAN_ALLOW_COMPLETE_RUN") != "1":
        raise RuntimeError("V8.2 outcome simulation requires complete-run authorization")
    result = run(
        paths=int(os.environ.get("TAIWAN_V82_PATHS", "128")),
        months=int(os.environ.get("TAIWAN_V82_MONTHS", "360")),
    )
    write(*result)
    print("TAIWAN_V82_COMPLETE_RUN: PASS")


if __name__ == "__main__":
    main()
