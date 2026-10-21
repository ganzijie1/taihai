from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
PLAN = OUT / "taiwan_v82_ablation_plan.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main() -> None:
    sources = [
        ROOT / "work" / "sectoral_development_labor_v82.py",
        ROOT / "work" / "simulate_taiwan_fully_closed_v82.py",
        ROOT / "work" / "simulate_taiwan_v82_checkpointed.py",
        ROOT / "data" / "sectoral_v82" / "actor_structural_priors.json",
        OUT / "taiwan_v82_frozen_specification.json",
    ]
    missing = [str(path) for path in sources if not path.exists()]
    if missing:
        raise RuntimeError("missing V8.2 ablation sources: " + "; ".join(missing))
    disabled = {
        "baseline": "none",
        "no_agriculture_deepening": "V8.2 agriculture deepening only",
        "no_industry_deepening": "V8.2 industry deepening only",
        "no_development_deepening": "V8.2 development deepening only",
        "no_labor_deepening": "V8.2 labor deepening only",
        "no_statistical_field": "V8.0 statistical-field layer only",
        "no_analytical_mechanics": "V8.0 analytical-mechanics layer only",
    }
    payload = {
        "model_version": "V8.2-ACTOR-SPECIFIC-SECTOR-DEVELOPMENT-LABOR",
        "status": "FROZEN",
        "paths": 128,
        "months": 360,
        "cells_per_variant": 30,
        "variants": {
            name: {"disabled_only": value} for name, value in disabled.items()
        },
        "source_hashes": {
            str(path.relative_to(ROOT)): sha256(path) for path in sources
        },
        "requirements": {
            "same_seeds_paths_horizon": True,
            "one_at_a_time": True,
            "full_module_validation_each_cell": True,
            "posthoc_probability_adjustment": False,
        },
    }
    PLAN.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("TAIWAN_V82_ABLATION_PLAN: FROZEN")


if __name__ == "__main__":
    main()
