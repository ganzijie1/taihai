from __future__ import annotations

import csv
import hashlib
import json
import os
from functools import partial
from pathlib import Path

import numpy as np

import simulate_taiwan_fully_closed_v84 as v84
from public_choice_v85 import FourPartyPublicChoiceSystemV85


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
TRACE = OUT / "taiwan_v85_traceability.json"
AUDIT = OUT / "taiwan_v85_static_coupling_audit.json"
FREEZE = OUT / "taiwan_v85_frozen_specification.json"
OUT_CSV = OUT / "taiwan_v85_results.csv"
OUT_JSON = OUT / "taiwan_v85_simulation.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def complete_model_gate() -> tuple[bool, list[str]]:
    if not TRACE.exists() or not AUDIT.exists() or not FREEZE.exists():
        return False, ["V8.5 traceability, audit or frozen specification missing"]
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


def validate_v85(diagnostics: dict, months: int) -> None:
    v84.validate_v84(diagnostics, months)
    errors: list[str] = []
    required = {
        "electoral_competition", "selectorate_coalition",
        "bureaucratic_execution", "interest_group_rent",
        "fiscal_illusion_common_pool", "veto_points",
        "commitment_reputation", "alliance_two_level", "feedback_enabled",
    }
    for cell, item in diagnostics.items():
        layer = item.get("public_choice")
        if layer is None:
            errors.append(f"{cell}: missing public-choice diagnostics")
            continue
        checks = {
            "clock": layer["months"] >= months,
            "budget simplex": layer["maximum_budget_simplex_residual"] <= 1.0e-12,
            "probability bounds": layer["maximum_probability_bound_violation"] <= 1.0e-12,
            "state jump": layer["maximum_state_jump"] <= 0.20,
            "execution positive": layer["minimum_bureaucratic_execution"] > 0.0,
            "feedback active": layer["maximum_feedback_shift"] > 0.0,
            "features": required <= set(layer["enabled"]),
            "no future": not layer["future_information_used"],
            "no posthoc": not layer["terminal_probability_adjustment"],
            "no duplicate state": not layer["structural_state_duplication"],
        }
        errors.extend(f"{cell}: {name}" for name, ok in checks.items() if not ok)
    if errors:
        raise RuntimeError("V8.5 runtime validation failed: " + "; ".join(errors))


def run(
    paths: int = 128,
    months: int = 360,
    *,
    electoral_competition_enabled: bool = True,
    selectorate_coalition_enabled: bool = True,
    bureaucratic_execution_enabled: bool = True,
    interest_group_rent_enabled: bool = True,
    fiscal_illusion_common_pool_enabled: bool = True,
    veto_points_enabled: bool = True,
    commitment_reputation_enabled: bool = True,
    alliance_two_level_enabled: bool = True,
    public_choice_feedback_enabled: bool = True,
    **kwargs,
):
    gate, gaps = complete_model_gate()
    if not gate:
        raise RuntimeError(
            "V8.5 complete-model gate failed: " + "; ".join(gaps or ["gate not passed"])
        )
    features = {
        "electoral_competition": electoral_competition_enabled,
        "selectorate_coalition": selectorate_coalition_enabled,
        "bureaucratic_execution": bureaucratic_execution_enabled,
        "interest_group_rent": interest_group_rent_enabled,
        "fiscal_illusion_common_pool": fiscal_illusion_common_pool_enabled,
        "veto_points": veto_points_enabled,
        "commitment_reputation": commitment_reputation_enabled,
        "alliance_two_level": alliance_two_level_enabled,
        "feedback_enabled": public_choice_feedback_enabled,
    }
    factory = partial(FourPartyPublicChoiceSystemV85, features=features)
    old_paths = (v84.TRACE, v84.AUDIT, v84.FREEZE)
    v84.TRACE, v84.AUDIT, v84.FREEZE = TRACE, AUDIT, FREEZE
    try:
        result = v84.run(
            paths=paths,
            months=months,
            public_choice_factory=factory,
            **kwargs,
        )
    finally:
        v84.TRACE, v84.AUDIT, v84.FREEZE = old_paths
    validate_v85(result[2], months)
    return result


def write(rows, aggregate, diagnostics, verification) -> None:
    with OUT_CSV.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    verification["model_version"] = "V8.5-FOUR-PARTY-PUBLIC-CHOICE"
    verification["runtime_validation"] = "TAIWAN_V85_RUNTIME_VALIDATION: PASS"
    verification["result_sha256"] = sha256(OUT_CSV)
    verification["v85_upgrade"] = {
        "public_choice_same_path_feedback": True,
        "four_party_heterogeneity": True,
        "budget_simplex_enforced": True,
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
        raise RuntimeError("V8.5 outcome simulation requires complete-run authorization")
    result = run(
        paths=int(os.environ.get("TAIWAN_V85_PATHS", "128")),
        months=int(os.environ.get("TAIWAN_V85_MONTHS", "360")),
    )
    write(*result)
    print("TAIWAN_V85_COMPLETE_RUN: PASS")


if __name__ == "__main__":
    main()
