from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
TARGET = OUT / "taiwan_v79_code_coupling_audit.json"


def main() -> None:
    files = {
        name: (ROOT / "work" / name).read_text(encoding="utf-8")
        for name in (
            "information_criticality_v79.py",
            "renormalization_scale_audit_v79.py",
            "information_criticality_mfg_v79.py",
            "simulate_taiwan_fully_closed_v79.py",
            "simulate_taiwan_multidomain_v42.py",
            "continuous_multipop_mfg_v47.py",
        )
    }
    info = files["information_criticality_v79.py"]
    wrapper = files["information_criticality_mfg_v79.py"]
    outer = files["simulate_taiwan_multidomain_v42.py"]
    runner = files["simulate_taiwan_fully_closed_v79.py"]
    checks = {
        "inherits_v78": "MixedPotentialNonpotentialMFGV78" in wrapper,
        "correlated_source_likelihood": "effective_sample_size" in info
        and "source_exposure" in info,
        "asymmetric_rate_distortion": "false_negative_cost" in info
        and "decision_risk" in info,
        "derived_channel_capacity": "effective_capacity" in info,
        "conditional_transfer_entropy": "ConditionalTransferEntropyV79" in info
        and "current_gain" in wrapper,
        "explicit_information_lag": "first affects t+1" in wrapper,
        "quasicritical_fast_clock": "QuasicriticalCascadeV79" in info
        and "maximum_microsteps_allowed" in info,
        "hawkes_trigger_reused": '"trigger_owner": "existing jump and Hawkes event signals"' in info,
        "outer_social_feedback": 's["actor_panic"]' in outer
        and "cascade_social" in outer,
        "outer_financial_feedback": 's["actor_debt_stress"]' in outer
        and "cascade_financial" in outer,
        "outer_production_feedback": 's["actor_firm_distress"]' in outer
        and "cascade_supply" in outer,
        "outer_command_feedback": 's["actor_command"]' in outer
        and "cascade_command" in outer,
        "scale_read_only": "terminal_probability_adjustment" in files[
            "renormalization_scale_audit_v79.py"
        ],
        "mfg_contract_extended": "cascade_social_loss" in files[
            "continuous_multipop_mfg_v47.py"
        ],
        "guarded_runner": "complete_model_gate" in runner
        and "TAIWAN_ALLOW_COMPLETE_RUN" in runner,
        "no_posthoc_probability_patch": (
            '"posthoc_probability_adjustment": False' in runner
        ),
    }
    payload = {
        "model_version": "V7.9-INFORMATION-CRITICALITY-SCALE",
        "classification": "STATIC_CODE_COUPLING_AUDIT",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "blocking_gaps": [name for name, value in checks.items() if not value],
        "outcome_simulation_executed": False,
    }
    TARGET.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"TAIWAN_V79_CODE_COUPLING_AUDIT: {payload['status']}")
    if payload["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
