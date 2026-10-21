from __future__ import annotations

import csv
import hashlib
import json
import os
from functools import partial
from pathlib import Path

import simulate_taiwan_fully_closed_v63 as v63
from stochastic_geometry_v64 import DynamicStochasticGeometryV64


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
TRACE = OUT / "taiwan_v64_traceability.json"
AUDIT = OUT / "taiwan_v64_static_coupling_audit.json"
FREEZE = OUT / "taiwan_v64_frozen_specification.json"
OUT_CSV = OUT / "taiwan_v64_results.csv"
OUT_JSON = OUT / "taiwan_v64_simulation.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def validate_v64(diagnostics: dict, months: int) -> None:
    errors = []
    for cell, item in diagnostics.items():
        sg = item.get("stochastic_geometry")
        if sg is None:
            errors.append(f"{cell}: missing stochastic geometry")
            continue
        if sg["months"] != months:
            errors.append(f"{cell}: stochastic geometry clock mismatch")
        if sg["max_probability_residual"] > 1e-12:
            errors.append(f"{cell}: coverage probability bounds")
        if sg["max_service_capacity_violation"] > 1e-12:
            errors.append(f"{cell}: service exceeds capacity")
        if sg["max_causal_clock_violation"] > 1e-12 or sg["future_information_used"]:
            errors.append(f"{cell}: stochastic geometry uses future information")
        if sg["dynamic_failures"] <= 0 or sg["dynamic_recoveries"] <= 0:
            errors.append(f"{cell}: dynamic random closed set inactive")
        macro = item["macro"]
        if macro["max_cge_accounting_residual"] > 1e-10:
            errors.append(f"{cell}: CGE delivery-rationing accounting identity")
        if not (0.0 <= macro["max_cge_unmet_demand_share"] <= 2.0):
            errors.append(f"{cell}: invalid CGE unmet-demand state")
    if errors:
        raise RuntimeError("V6.4 runtime validation failed: " + "; ".join(errors))


def run(paths: int = 128, months: int = 360,
        scenario_names: list[str] | None = None,
        case_names: list[str] | None = None, *,
        random_field_enabled: bool = True,
        cascade_enabled: bool = True,
        feedback_enabled: bool = True,
        transport_symplectic_factory=None,
        rare_event_factory=None, path_group_names=None,
        paths_per_group: int | None = None):
    old = (v63.TRACE, v63.AUDIT, v63.FREEZE)
    v63.TRACE, v63.AUDIT, v63.FREEZE = TRACE, AUDIT, FREEZE
    factory = partial(
        DynamicStochasticGeometryV64,
        random_field_enabled=random_field_enabled,
        cascade_enabled=cascade_enabled,
        feedback_enabled=feedback_enabled,
    )
    try:
        result = v63.run(
            paths=paths, months=months, scenario_names=scenario_names,
            case_names=case_names, stochastic_geometry_factory=factory,
            transport_symplectic_factory=transport_symplectic_factory,
            rare_event_factory=rare_event_factory,
            path_group_names=path_group_names,
            paths_per_group=paths_per_group,
        )
    finally:
        v63.TRACE, v63.AUDIT, v63.FREEZE = old
    validate_v64(result[2], months)
    return result


def write(rows, aggregate, diagnostics, verification) -> None:
    with OUT_CSV.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    verification["model_version"] = "V6.4-SG"
    verification["runtime_validation"] = "TAIWAN_V64_RUNTIME_VALIDATION: PASS"
    verification["result_sha256"] = sha256(OUT_CSV)
    verification["v64_stochastic_geometry_upgrade"] = {
        "real_gis_conditioning": True,
        "marked_log_gaussian_cox": True,
        "dynamic_random_closed_set": True,
        "coverage_meta_distribution": True,
        "capacity_service_tessellation": True,
        "correlated_failure_and_recovery": True,
        "same_path_bidirectional_feedback": True,
        "cge_rationing_identity_explicit": True,
        "no_posthoc_probability_adjustment": True,
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
        raise RuntimeError("V6.4 outcome simulation blocked until the complete-model gate passes")
    frozen = json.loads(FREEZE.read_text(encoding="utf-8"))
    if frozen.get("status") != "FROZEN":
        raise RuntimeError("V6.4 frozen specification is missing or invalid")
    paths = int(os.environ.get("TAIWAN_V64_PATHS", "128"))
    months = int(os.environ.get("TAIWAN_V64_MONTHS", "360"))
    result = run(paths=paths, months=months)
    write(*result)
    print("TAIWAN_V64_COMPLETE_RUN: PASS")


if __name__ == "__main__":
    main()
