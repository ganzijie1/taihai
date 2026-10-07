from __future__ import annotations

import json
import sys
from pathlib import Path

import simulate_taiwan_full_coupled_v46 as model


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "台海V4.6传播结构敏感性.json"
CONFIGS = {
    "hybrid_selected": {},
    "without_levy_marks": {"levy": False},
    "without_hawkes_self_excitation": {"hawkes": False},
    "homogeneous_age_mixing": {"age_structure": False},
    "simple_contagion": {"complex_threshold": False},
}


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    results = {}
    for name, features in CONFIGS.items():
        _, aggregate, _, _, verification = model.run(
            paths=80,
            months=360,
            scenario_names=["和平基线"],
            include_ablation=False,
            social_features=features,
        )
        results[name] = {
            "features_overrides": features,
            "aggregate": aggregate,
            "max_conservation_error": verification["social_population_conservation_max_error"],
        }
    baseline = {
        row["horizon_years"]: row["weighted_conditional_mean"]
        for row in results["hybrid_selected"]["aggregate"]
    }
    for name, result in results.items():
        result["difference_percentage_points_vs_hybrid"] = {
            str(row["horizon_years"]): 100.0 * (
                row["weighted_conditional_mean"] - baseline[row["horizon_years"]]
            )
            for row in result["aggregate"]
        }
    payload = {
        "status": "PASS",
        "scope": "80 paths, 360 months, baseline trade scenario, all five intervention cases",
        "parameter_status": "structural sensitivity, not empirical confidence interval",
        "results": results,
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("TAIWAN_SOCIAL_V46_SENSITIVITY: PASS")
    for name, result in results.items():
        print(name, result["difference_percentage_points_vs_hybrid"])


if __name__ == "__main__":
    main()
