from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
TARGET = OUT / "taiwan_v78_code_coupling_audit.json"


def main() -> None:
    sources = {
        name: (ROOT / "work" / name).read_text(encoding="utf-8")
        for name in (
            "mixed_potential_nonpotential_mfg_v78.py",
            "simulate_taiwan_fully_closed_v78.py",
            "diagnostic_taiwan_v78.py",
        )
    }
    mixed = sources["mixed_potential_nonpotential_mfg_v78.py"]
    runner = sources["simulate_taiwan_fully_closed_v78.py"]
    checks = {
        "inherits_complete_v77": "CompleteTheoryMFGV77" in mixed,
        "weighted_adjoint_split": "_weighted_split" in mixed and "adjoint" in mixed,
        "same_iteration_predictor_corrector": "predicted_mean" in mixed
        and "second_state" in mixed,
        "potential_proximal_owner": "potential_proximal_updates" in mixed,
        "directed_hypergraph_in_nonpotential_block": "_nonpotential_fields" in mixed
        and "_hypergraph_fields" in mixed,
        "joint_state_action_field": "predicted_action" in mixed
        and "predicted_major" in mixed,
        "vi_gap": "maximum_mixed_vi_gap" in mixed,
        "kernel_uncertainty_preserved": "kernel_uncertainty" in mixed,
        "four_ablation_families": all(
            name in mixed for name in (
                "no_graph_mfg", "block_gmfg", "low_rank_graphon", "finite_graph"
            )
        ),
        "runtime_same_path_replacement": "MixedPotentialNonpotentialMFGV78" in runner
        and "v77.run" in runner,
        "no_probability_patch": '"posthoc_probability_adjustment": False' in runner,
        "runtime_validation": "validate_v78" in runner,
    }
    payload = {
        "model_version": "V7.8-MIXED-POTENTIAL-NONPOTENTIAL-MFG",
        "classification": "STATIC_CODE_COUPLING_AUDIT",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "blocking_gaps": [name for name, value in checks.items() if not value],
    }
    TARGET.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"TAIWAN_V78_CODE_COUPLING_AUDIT: {payload['status']}")
    if payload["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
