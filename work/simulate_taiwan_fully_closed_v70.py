from __future__ import annotations

import csv
import hashlib
import json
import os
from functools import partial
from pathlib import Path

import simulate_taiwan_fully_closed_v69 as v69
from tinbergen_algebraic_v70 import (
    AsymmetricAdjustmentMacroV70,
    TinbergenAlgebraicRollingGameV70,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
TRACE = OUT / "taiwan_v70_traceability.json"
AUDIT = OUT / "taiwan_v70_static_coupling_audit.json"
FREEZE = OUT / "taiwan_v70_frozen_specification.json"
OUT_CSV = OUT / "taiwan_v70_results.csv"
OUT_JSON = OUT / "taiwan_v70_simulation.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def validate_v70(diagnostics: dict, months: int) -> None:
    errors: list[str] = []
    for cell, item in diagnostics.items():
        strategy = item.get("strategy", {}).get("tinbergen_algebraic_v70")
        macro = item.get("macro", {}).get("tinbergen_asymmetric_adjustment_v70")
        if strategy is None:
            errors.append(f"{cell}: missing Tinbergen/algebraic strategy layer")
        else:
            if strategy["months"] != months:
                errors.append(f"{cell}: strategy clock mismatch")
            if strategy["reachability_enabled"] and strategy["reachability_gate_months"] != months:
                errors.append(f"{cell}: inactive reachability gate")
            if strategy["minimum_local_response_rank"] < 4:
                errors.append(f"{cell}: target-tool rank deficiency")
            if strategy["maximum_simplex_residual"] > 1.0e-12:
                errors.append(f"{cell}: strategy simplex residual")
            if strategy["maximum_barrier_violation"] > 1.0e-9:
                errors.append(f"{cell}: certifiable barrier violation")
            if strategy["identification_rank"] <= 0:
                errors.append(f"{cell}: absent algebraic-identification diagnostic")
            if strategy["tropical_enabled"]:
                if strategy["maximum_tropical_min_identity_residual"] > 1.0e-12:
                    errors.append(f"{cell}: tropical min-plus identity")
                if abs(sum(strategy["tropical_type_shares"]) - 1.0) > 1.0e-12:
                    errors.append(f"{cell}: tropical bottleneck shares")
            if strategy["terminal_probability_adjustment"]:
                errors.append(f"{cell}: forbidden terminal adjustment")
        if macro is None:
            errors.append(f"{cell}: missing asymmetric macro adjustment layer")
        else:
            if macro["months"] != months:
                errors.append(f"{cell}: macro adjustment clock mismatch")
            if macro["enabled"] and macro["maximum_real_input_wedge"] <= 0.0:
                errors.append(f"{cell}: inactive macro adjustment cost")
            if macro["terminal_probability_adjustment"]:
                errors.append(f"{cell}: forbidden macro terminal adjustment")
    if errors:
        raise RuntimeError("V7.0 runtime validation failed: " + "; ".join(errors))


def run(paths: int = 128, months: int = 360,
        scenario_names: list[str] | None = None,
        case_names: list[str] | None = None, *,
        reachability_enabled: bool = True,
        algebraic_filter_enabled: bool = True,
        asymmetric_adjustment_enabled: bool = True,
        tropical_enabled: bool = True,
        **kwargs):
    v62 = v69.v68.v67.v66.v64.v63.v62
    core = v69.core
    old_paths = (v69.TRACE, v69.AUDIT, v69.FREEZE)
    old_macro = v62.CoupledSectoralEconomicSystemV62
    old_strategy = core.RollingHorizonDifferentialGame
    v69.TRACE, v69.AUDIT, v69.FREEZE = TRACE, AUDIT, FREEZE
    v62.CoupledSectoralEconomicSystemV62 = partial(
        AsymmetricAdjustmentMacroV70,
        adjustment_enabled=asymmetric_adjustment_enabled,
    )
    core.RollingHorizonDifferentialGame = partial(
        TinbergenAlgebraicRollingGameV70,
        reachability_enabled=reachability_enabled,
        algebraic_filter_enabled=algebraic_filter_enabled,
        tropical_enabled=tropical_enabled,
    )
    try:
        result = v69.run(
            paths=paths, months=months, scenario_names=scenario_names,
            case_names=case_names, **kwargs,
        )
    finally:
        v69.TRACE, v69.AUDIT, v69.FREEZE = old_paths
        v62.CoupledSectoralEconomicSystemV62 = old_macro
        core.RollingHorizonDifferentialGame = old_strategy
    validate_v70(result[2], months)
    return result


def write(rows, aggregate, diagnostics, verification) -> None:
    with OUT_CSV.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    verification["model_version"] = "V7.0-TINBERGEN-ALGEBRAIC"
    verification["runtime_validation"] = "TAIWAN_V70_RUNTIME_VALIDATION: PASS"
    verification["result_sha256"] = sha256(OUT_CSV)
    verification["v70_upgrade"] = {
        "target_tool_reachability_gate": True,
        "asymmetric_real_adjustment_cost": True,
        "polynomial_barrier_filter": True,
        "algebraic_identifiability_diagnostic": True,
        "cusp_multiple_equilibrium_hysteresis": True,
        "tropical_bottleneck_control": True,
        "same_path_pre_transition_feedback": True,
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
        raise RuntimeError("V7.0 outcome simulation blocked until the complete-model gate passes")
    frozen = json.loads(FREEZE.read_text(encoding="utf-8"))
    if frozen.get("status") != "FROZEN":
        raise RuntimeError("V7.0 frozen specification is missing or invalid")
    result = run(
        paths=int(os.environ.get("TAIWAN_V70_PATHS", "128")),
        months=int(os.environ.get("TAIWAN_V70_MONTHS", "360")),
    )
    write(*result)
    print("TAIWAN_V70_COMPLETE_RUN: PASS")


if __name__ == "__main__":
    main()
