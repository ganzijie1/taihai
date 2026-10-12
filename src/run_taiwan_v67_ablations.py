from __future__ import annotations

from functools import partial
import json
import os
from pathlib import Path

import numpy as np

import simulate_taiwan_fully_closed_v67 as v67
from large_deviation_v67 import (
    HybridLargeDeviationSystemV67,
    TARGET_MODE_PROBABILITY,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
TARGET = OUT / "taiwan_v67_large_deviation_ablations.json"


def main() -> None:
    if os.environ.get("TAIWAN_ALLOW_COMPLETE_RUN") != "1":
        raise RuntimeError("V6.7 ablations require the complete-run authorization flag")
    scenario = list(
        v67.v66.v64.v63.v62.v61.v60.v59.v58.core.trade.v43.SCENARIOS
    )[0]
    joint_extreme_tilt = np.array([0.40, 0.10, 0.10, 0.10, 0.10, 0.20])
    variants = {
        "ordinary_target_sampling": partial(
            HybridLargeDeviationSystemV67,
            proposal_mode_probability=TARGET_MODE_PROBABILITY,
        ),
        "single_joint_extreme_tilt": partial(
            HybridLargeDeviationSystemV67,
            proposal_mode_probability=joint_extreme_tilt,
        ),
        "multimodal_defensive_proposal": HybridLargeDeviationSystemV67,
        "multimodal_without_action_ledger": partial(
            HybridLargeDeviationSystemV67,
            action_enabled=False,
        ),
    }
    payload = {
        "model_version": "V6.7-LD",
        "classification": (
            "pre-registered structural integration ablation; 16 paths, 60 months, "
            "one complete-model cell; not a published probability estimate"
        ),
        "common_random_numbers": True,
        "paths": 16,
        "months": 60,
        "case": "taiwan_alone",
        "trade_scenario": scenario,
        "variants": {},
    }
    for name, factory in variants.items():
        rows, aggregate, diagnostics, verification = v67.run(
            paths=16,
            months=60,
            scenario_names=[scenario],
            case_names=["taiwan_alone"],
            rare_event_factory=factory,
        )
        cell = next(iter(diagnostics.values()))
        payload["variants"][name] = {
            "conditional_rows": rows,
            "aggregate": aggregate,
            "large_deviation": cell["large_deviation"],
            "runtime_validation": verification["runtime_validation"],
        }
    with_action = payload["variants"]["multimodal_defensive_proposal"]["conditional_rows"]
    without_action = payload["variants"]["multimodal_without_action_ledger"]["conditional_rows"]
    action_difference = max(
        abs(a["conditional_broad_control_rate"] - b["conditional_broad_control_rate"])
        for a, b in zip(with_action, without_action)
    )
    checks = {
        "all_runtime_validations_pass": all(
            item["runtime_validation"] == "TAIWAN_V51_RUNTIME_VALIDATION: PASS"
            for item in payload["variants"].values()
        ),
        "ordinary_sampling_unit_weights": (
            payload["variants"]["ordinary_target_sampling"]["large_deviation"]
            ["effective_sample_size_ratio"] > 0.999999
        ),
        "multimodal_ess_above_floor": (
            payload["variants"]["multimodal_defensive_proposal"]["large_deviation"]
            ["effective_sample_size_ratio"] >= 0.25
        ),
        "action_ledger_is_read_only": action_difference <= 1.0e-12,
    }
    payload["checks"] = checks
    payload["action_ledger_probability_difference"] = action_difference
    payload["status"] = "PASS" if all(checks.values()) else "FAIL"
    TARGET.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if payload["status"] != "PASS":
        raise RuntimeError("V6.7 ablation failed: " + ", ".join(
            name for name, passed in checks.items() if not passed
        ))
    print("TAIWAN_V67_LARGE_DEVIATION_ABLATIONS: PASS")


if __name__ == "__main__":
    main()
