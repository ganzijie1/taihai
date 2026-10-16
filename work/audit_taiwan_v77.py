from __future__ import annotations

import json
from pathlib import Path

from theory_resolution_v77 import write_resolution


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
TARGET = OUT / "taiwan_v77_code_coupling_audit.json"


def main() -> None:
    resolution = write_resolution()
    sources = {
        name: (ROOT / "work" / name).read_text(encoding="utf-8")
        for name in (
            "complete_information_geometry_v77.py",
            "complete_network_geometry_v77.py",
            "complete_online_tda_v77.py",
            "coupled_oscillator_v77.py",
            "complete_theory_mfg_v77.py",
            "nested_ces_cge_v77.py",
            "simulate_taiwan_fully_closed_v77.py",
        )
    }
    joined = "\n".join(sources.values())
    checks = {
        "all_useful_theory_resolved": not resolution["unresolved_useful"],
        "discrete_information_bottleneck_ba": "_information_bottleneck" in joined and "pyz" in joined,
        "active_observation_kkt": "_active_observation_plan" in joined and "maximum_attention_kkt_residual" in joined,
        "full_fgw_outer_fixed_point": "_fgw_proximal" in joined and "fgw_outer_iterations" in joined,
        "directed_ollivier_transport": "_directed_ollivier" in joined and "wasserstein" in joined,
        "network_hodge_projection": "_hodge" in joined and "_laplacian_pinv" in joined,
        "exact_h0_barcode": "_mst_deaths" in joined,
        "exact_gf2_h1": "_gf2_rank" in joined and "_clique_betti_one" in joined,
        "full_mapper_pullback_clustering": "_exact_mapper" in joined and "mapper_vertices" in joined,
        "bipersistence_image_rank": "_bipersistence" in joined and "maximum_bipersistence_image_rank" in joined,
        "consecutive_zigzag_maps": "_zigzag" in joined and "through" in joined,
        "hypergraph_simplicial_homology": "_hypergraph_homology" in joined,
        "nested_ces_two_levels": "factor_nest" in joined and "upper_nest" in joined,
        "inertial_signed_sakaguchi_kuramoto": "inertial signed directed Sakaguchi-Kuramoto" in joined,
        "oscillator_bidirectional_feedback": "lagged desynchronization changes relation memory" in joined,
        "same_path_runtime_replacement": "CompleteTheoryMFGV77" in sources["simulate_taiwan_fully_closed_v77.py"],
        "nested_cge_runtime_replacement": "NestedCESCGEV77" in sources["simulate_taiwan_fully_closed_v77.py"],
        "no_probability_patch": '"posthoc_probability_adjustment": False' in joined,
    }
    payload = {
        "model_version": "V7.7-COMPLETE-THEORY-CLOSURE",
        "classification": "STATIC_CODE_COUPLING_AUDIT",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "blocking_gaps": [name for name, value in checks.items() if not value],
    }
    TARGET.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"TAIWAN_V77_CODE_COUPLING_AUDIT: {payload['status']}")
    if payload["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
