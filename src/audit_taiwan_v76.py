from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
TARGET = OUT / "taiwan_v76_code_coupling_audit.json"


def main() -> None:
    source = (ROOT / "work" / "scf_signed_potential_mfg_v76.py").read_text(encoding="utf-8")
    tda_source = (ROOT / "work" / "online_tda_v76.py").read_text(encoding="utf-8")
    wrapper = (ROOT / "work" / "simulate_taiwan_fully_closed_v76.py").read_text(encoding="utf-8")
    required = {
        "sign_magnitude_separation": "kernel * self.current_sign" in source,
        "opposing_state_flip": "kernel * (sign - 1.0)" in source,
        "potential_circulation_split": "0.5 * (signed_kernel + transpose)" in source and "0.5 * (signed_kernel - transpose)" in source,
        "signed_laplacian": "np.linalg.eigvalsh(laplacian)" in source,
        "adaptive_scf_trust": "nonpotential_cap" in source and "trust_contractions" in source,
        "multistart_branch_probe": "for start in (-0.82, 0.0, 0.82)" in source,
        "dynamic_positive_negative_graphon": "_update_relation_memory" in source and "formation_barrier" in source and "fracture_barrier" in source,
        "nonabsorbing_boundary_test": "minimum_zero_boundary_formation_rate" in source,
        "nonautonomous_spectrum": "_update_nonautonomous_spectrum" in source,
        "online_tda_feedback": "self.online_tda.update" in source and "self.tda_delayed_shock" in source,
        "tda_h0_persistence": "beta0_curve" in tda_source,
        "tda_bipersistence_rank_slices": "last_bipersistence_rank" in tda_source,
        "tda_mapper_nerve": "_mapper_nerve" in tda_source,
        "tda_zigzag": "zigzag_splits" in tda_source and "zigzag_merges" in tda_source,
        "tda_directed_flag": "directed_flags" in tda_source,
        "tda_relative_h0": "relative_bridge_gain" in tda_source,
        "runtime_replacement": "SignedPotentialHypergraphMFGV76" in wrapper and "AdaptiveTrustRegionSCFAnderson" in wrapper,
        "runtime_validation": "validate_v76" in wrapper,
        "no_probability_patch": "posthoc_probability_adjustment\": False" in wrapper,
    }
    payload = {
        "model_version": "V7.6-SIGNED-POTENTIAL-SCF",
        "classification": "STATIC_CODE_COUPLING_AUDIT",
        "status": "PASS" if all(required.values()) else "FAIL",
        "checks": required,
    }
    TARGET.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"TAIWAN_V76_CODE_COUPLING_AUDIT: {payload['status']}")
    if payload["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
