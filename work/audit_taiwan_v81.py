from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
TARGET = OUT / "taiwan_v81_code_coupling_audit.json"


def main() -> None:
    names = (
        "differential_manifold_v81.py",
        "differential_manifold_mfg_v81.py",
        "simulate_taiwan_fully_closed_v81.py",
        "simulate_taiwan_v81_checkpointed.py",
    )
    files = {
        name: (ROOT / "work" / name).read_text(encoding="utf-8")
        for name in names
    }
    manifold = files["differential_manifold_v81.py"]
    wrapper = files["differential_manifold_mfg_v81.py"]
    runner = files["simulate_taiwan_fully_closed_v81.py"]
    checkpointed = files["simulate_taiwan_v81_checkpointed.py"]
    checks = {
        "inherits_v80": "StatisticalFieldMFGV80" in wrapper,
        "simplex_retraction": "np.sqrt(self.support_simplex)" in manifold,
        "simplex_normalization": "np.sum(updated, axis=-1" in manifold,
        "spd_projection": "np.linalg.eigh" in manifold and "_project_spd" in manifold,
        "slow_manifold_state": "self.slow_state" in manifold and "normal_residual" in manifold,
        "hybrid_hysteresis": "low_to_mid" in manifold and "high_to_mid" in manifold,
        "jump_fast_exit": "fast_exit" in manifold,
        "attractor_basin_diagnostic": "ATTRACTORS" in manifold and "separatrix" in manifold,
        "same_path_feedback": "factors.support =" in wrapper and "manifold.prepare" in wrapper,
        "no_physical_state_ownership": '"physical_stocks": "inherited dedicated solvers"' in manifold,
        "guarded_runner": "complete_model_gate" in runner and "TAIWAN_ALLOW_COMPLETE_RUN" in runner,
        "atomic_checkpoint_runner": "temporary.replace(target)" in checkpointed and "freeze_sha256" in checkpointed,
        "aggregate_labels_preserved": 'item["trade_scenario"]' in checkpointed and 'item["case"]' in checkpointed,
        "no_posthoc_probability_patch": '"posthoc_probability_adjustment": False' in runner,
    }
    payload = {
        "model_version": "V8.1-DIFFERENTIAL-MANIFOLD-CONSTRAINTS",
        "classification": "STATIC_CODE_COUPLING_AUDIT",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "blocking_gaps": [key for key, value in checks.items() if not value],
        "outcome_simulation_executed": False,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    TARGET.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"TAIWAN_V81_CODE_COUPLING_AUDIT: {payload['status']}")
    if payload["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
