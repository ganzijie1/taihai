from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
TARGET = OUT / "taiwan_v80_code_coupling_audit.json"


def main() -> None:
    names = (
        "statistical_field_soc_v80.py", "analytical_mechanics_v80.py",
        "statistical_field_mfg_v80.py", "simulate_taiwan_fully_closed_v80.py",
        "simulate_taiwan_v80_checkpointed.py",
        "simulate_taiwan_multidomain_v42.py",
    )
    files = {
        name: (ROOT / "work" / name).read_text(encoding="utf-8")
        for name in names
    }
    field = files["statistical_field_soc_v80.py"]
    mechanics = files["analytical_mechanics_v80.py"]
    wrapper = files["statistical_field_mfg_v80.py"]
    outer = files["simulate_taiwan_multidomain_v42.py"]
    runner = files["simulate_taiwan_fully_closed_v80.py"]
    checkpointed = files["simulate_taiwan_v80_checkpointed.py"]
    checks = {
        "inherits_v79": "InformationCriticalityMFGV79" in wrapper,
        "dmft_memory_fitted": "np.linalg.solve(gram" in field,
        "colored_noise_active": "colored_noise" in field,
        "msrjd_diagnostic": "maximum_msrjd_action" in field,
        "binder_multiscale": "binder_cumulant_by_resolution" in field,
        "cavity_resolve": "_cavity_resolve" in field,
        "soc_event_evidence": "power_law_vs_exponential" in field,
        "soc_claim_boundary": "not confirmed SOC" in field,
        "dirac_skew_operator": "self.dirac" in mechanics,
        "lagrange_dalembert_step": "new_velocity" in mechanics
        and "new_q" in mechanics,
        "noether_diagnostics": "maximum_dirac_power_residual" in mechanics
        and "maximum_discrete_energy_balance_residual" in mechanics,
        "pmp_crosscheck": "pmp_action" in mechanics,
        "explicit_field_lag": "month t+1" in wrapper,
        "same_path_outer_contract": "cascade_social_loss" in wrapper
        and "cascade_social_loss" in outer,
        "guarded_runner": "complete_model_gate" in runner
        and "TAIWAN_ALLOW_COMPLETE_RUN" in runner,
        "atomic_checkpoint_runner": "temporary.replace(target)" in checkpointed
        and "freeze_sha256" in checkpointed
        and "atomic_per_scenario_case" in checkpointed,
        "no_posthoc_probability_patch": (
            '"posthoc_probability_adjustment": False' in runner
        ),
    }
    payload = {
        "model_version": "V8.0-STATISTICAL-FIELD-ANALYTICAL-MECHANICS",
        "classification": "STATIC_CODE_COUPLING_AUDIT",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "blocking_gaps": [key for key, value in checks.items() if not value],
        "outcome_simulation_executed": False,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    TARGET.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"TAIWAN_V80_CODE_COUPLING_AUDIT: {payload['status']}")
    if payload["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
