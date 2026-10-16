from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path

import simulate_taiwan_fully_closed_v47 as core
import simulate_taiwan_fully_closed_v68 as v68
from martingale_ot_finance_v69 import MartingaleOTFinancialSystemV69


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
TRACE = OUT / "taiwan_v69_traceability.json"
AUDIT = OUT / "taiwan_v69_static_coupling_audit.json"
FREEZE = OUT / "taiwan_v69_frozen_specification.json"
OUT_CSV = OUT / "taiwan_v69_results.csv"
OUT_JSON = OUT / "taiwan_v69_simulation.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def validate_v69(diagnostics: dict, months: int) -> None:
    errors: list[str] = []
    for cell, item in diagnostics.items():
        finance = item.get("finance", {}).get("martingale_optimal_transport")
        if finance is None:
            errors.append(f"{cell}: missing martingale OT finance")
        else:
            if finance["projection_months"] != months:
                errors.append(f"{cell}: finance projection clock mismatch")
            if finance["max_martingale_residual"] > 1.0e-10:
                errors.append(f"{cell}: martingale residual")
            if finance["max_probability_residual"] > 1.0e-12:
                errors.append(f"{cell}: probability residual")
            if finance["max_negative_probability"] > 1.0e-14:
                errors.append(f"{cell}: negative risk-neutral probability")
        deep = item.get("mfg", {}).get("partial_observation", {}).get(
            "v66_deep_causal_transport", {}
        )
        conformal = deep.get("conformal_dro")
        if conformal is None or conformal.get("updates", 0) <= 0:
            errors.append(f"{cell}: inactive conformal DRO decomposition")
    if errors:
        raise RuntimeError("V6.9 runtime validation failed: " + "; ".join(errors))


def run(paths: int = 128, months: int = 360,
        scenario_names: list[str] | None = None,
        case_names: list[str] | None = None, **kwargs):
    old_paths = (v68.TRACE, v68.AUDIT, v68.FREEZE)
    old_finance = core.FourPartyFinancialSystem
    v68.TRACE, v68.AUDIT, v68.FREEZE = TRACE, AUDIT, FREEZE
    core.FourPartyFinancialSystem = MartingaleOTFinancialSystemV69
    try:
        result = v68.run(
            paths=paths, months=months, scenario_names=scenario_names,
            case_names=case_names, **kwargs,
        )
    finally:
        core.FourPartyFinancialSystem = old_finance
        v68.TRACE, v68.AUDIT, v68.FREEZE = old_paths
    validate_v69(result[2], months)
    result[3]["functional_analysis"]["same_outcomes_as_v67_by_construction"] = False
    return result


def write(rows, aggregate, diagnostics, verification) -> None:
    with OUT_CSV.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    verification["model_version"] = "V6.9-OT-SYM-ACTIVE"
    verification["runtime_validation"] = "TAIWAN_V69_RUNTIME_VALIDATION: PASS"
    verification["result_sha256"] = sha256(OUT_CSV)
    verification["v69_upgrade"] = {
        "conformal_dro_radius_decomposition": True,
        "martingale_ot_financial_projection": True,
        "same_path_financial_feedback": True,
        "structure_preserving_full_order_core": True,
        "rom_excluded_due_to_six_dimensional_state": True,
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
        raise RuntimeError("V6.9 outcome simulation blocked until the complete-model gate passes")
    frozen = json.loads(FREEZE.read_text(encoding="utf-8"))
    if frozen.get("status") != "FROZEN":
        raise RuntimeError("V6.9 frozen specification is missing or invalid")
    result = run(
        paths=int(os.environ.get("TAIWAN_V69_PATHS", "128")),
        months=int(os.environ.get("TAIWAN_V69_MONTHS", "360")),
    )
    write(*result)
    print("TAIWAN_V69_COMPLETE_RUN: PASS")


if __name__ == "__main__":
    main()
