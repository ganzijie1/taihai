from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np

import simulate_taiwan_fully_closed_v66 as v66
from large_deviation_v67 import HybridLargeDeviationSystemV67


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
TRACE = OUT / "taiwan_v67_traceability.json"
AUDIT = OUT / "taiwan_v67_static_coupling_audit.json"
FREEZE = OUT / "taiwan_v67_frozen_specification.json"
OUT_CSV = OUT / "taiwan_v67_results.csv"
OUT_JSON = OUT / "taiwan_v67_simulation.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def validate_v67(diagnostics: dict, months: int) -> None:
    errors: list[str] = []
    for cell, item in diagnostics.items():
        ld = item.get("large_deviation")
        if ld is None:
            errors.append(f"{cell}: missing V6.7 large-deviation module")
            continue
        if ld["months"] != months:
            errors.append(f"{cell}: large-deviation clock mismatch")
        if ld["future_information_used"]:
            errors.append(f"{cell}: future information used")
        if ld["max_causal_clock_violation"] > 0.0:
            errors.append(f"{cell}: causal clock violation")
        if ld["max_likelihood_identity_residual"] > 1.0e-12:
            errors.append(f"{cell}: analytic likelihood identity")
        if ld["sample_likelihood_normalization_residual"] > 0.35:
            errors.append(f"{cell}: sample likelihood normalization")
        if ld["effective_sample_size_ratio"] < 0.25:
            errors.append(f"{cell}: importance-sampling ESS")
        if ld["max_empirical_measure_mass_residual"] > 1.0e-12:
            errors.append(f"{cell}: empirical-measure mass")
        for key in (
            "mean_continuous_action", "mean_jump_action",
            "mean_field_empirical_action",
        ):
            value = ld[key]
            if not np.isfinite(value) or value < -1.0e-12:
                errors.append(f"{cell}: invalid {key}")
    if errors:
        raise RuntimeError("V6.7 runtime validation failed: " + "; ".join(errors))


def run(
    paths: int = 128,
    months: int = 360,
    scenario_names: list[str] | None = None,
    case_names: list[str] | None = None,
    rare_event_factory=HybridLargeDeviationSystemV67,
    **kwargs,
):
    old_paths = (v66.TRACE, v66.AUDIT, v66.FREEZE)
    v66.TRACE, v66.AUDIT, v66.FREEZE = TRACE, AUDIT, FREEZE
    try:
        result = v66.run(
            paths=paths,
            months=months,
            scenario_names=scenario_names,
            case_names=case_names,
            rare_event_factory=rare_event_factory,
            **kwargs,
        )
    finally:
        v66.TRACE, v66.AUDIT, v66.FREEZE = old_paths
    validate_v67(result[2], months)
    return result


def write(rows, aggregate, diagnostics, verification) -> None:
    with OUT_CSV.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    verification["model_version"] = "V6.7-LD"
    verification["runtime_validation"] = "TAIWAN_V67_RUNTIME_VALIDATION: PASS"
    verification["result_sha256"] = sha256(OUT_CSV)
    verification["v67_large_deviation_upgrade"] = {
        "hybrid_diffusion_jump_action_ledger": True,
        "mfg_empirical_measure_action": True,
        "four_pre_registered_rare_sets": True,
        "defensive_multimodal_mixture": True,
        "exact_radon_nikodym_weights": True,
        "schrodinger_bridge_interpretation": "finite latent path-mode bridge",
        "same_path_bidirectional_feedback": True,
        "asymptotic_action_used_as_probability": False,
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
        raise RuntimeError("V6.7 outcome simulation blocked until the complete-model gate passes")
    frozen = json.loads(FREEZE.read_text(encoding="utf-8"))
    if frozen.get("status") != "FROZEN":
        raise RuntimeError("V6.7 frozen specification is missing or invalid")
    result = run(
        paths=int(os.environ.get("TAIWAN_V67_PATHS", "128")),
        months=int(os.environ.get("TAIWAN_V67_MONTHS", "360")),
    )
    write(*result)
    print("TAIWAN_V67_COMPLETE_RUN: PASS")


if __name__ == "__main__":
    main()
