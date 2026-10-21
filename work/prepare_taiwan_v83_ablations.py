from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
PLAN = OUT / "taiwan_v83_ablation_plan.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main() -> None:
    sources = [
        ROOT / "work" / "particle_international_economics_v83.py",
        ROOT / "work" / "simulate_taiwan_fully_closed_v83.py",
        ROOT / "work" / "simulate_taiwan_v83_checkpointed.py",
        ROOT / "work" / "audit_taiwan_v83_results.py",
        ROOT / "data" / "particle_trade_v83" / "structural_priors.json",
        OUT / "taiwan_v83_frozen_specification.json",
        OUT / "taiwan_v83_traceability.json",
        OUT / "taiwan_v83_static_coupling_audit.json",
    ]
    missing = [str(path) for path in sources if not path.exists()]
    if missing:
        raise RuntimeError("missing V8.3 formal-run sources: " + "; ".join(missing))
    disabled = {
        "baseline": "none",
        "no_particle_intent": "V8.3 particle-intent feedback only",
        "no_structural_gravity_ek": "V8.3 structural-gravity and EK feedback only",
        "no_melitz": "V8.3 Melitz firm entry-exit feedback only",
        "no_external_liquidity": "V8.3 external-liquidity feedback only",
        "no_statistical_field": "V8.0 statistical-field layer only",
        "no_analytical_mechanics": "V8.0 analytical-mechanics layer only",
        "particle_n32": "none; particle-count sensitivity N=32",
        "particle_n128": "none; particle-count sensitivity N=128",
    }
    payload = {
        "model_version": "V8.3-PARTICLE-INTERNATIONAL-ECONOMICS",
        "status": "FROZEN",
        "paths": 128,
        "months": 360,
        "cells_per_variant": 30,
        "total_cells": len(disabled) * 30,
        "variants": {
            name: {"disabled_only": value} for name, value in disabled.items()
        },
        "particle_convergence_counts": [32, 64, 128],
        "source_hashes": {
            str(path.relative_to(ROOT)): sha256(path) for path in sources
        },
        "requirements": {
            "same_seeds_paths_horizon": True,
            "one_at_a_time": True,
            "full_module_validation_each_cell": True,
            "particle_count_convergence": True,
            "manifest_audit_passed": True,
            "posthoc_probability_adjustment": False,
        },
    }
    PLAN.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("TAIWAN_V83_ABLATION_PLAN: FROZEN")


if __name__ == "__main__":
    main()
