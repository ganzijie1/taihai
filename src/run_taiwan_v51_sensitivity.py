from __future__ import annotations

import json
import os
from pathlib import Path

import simulate_taiwan_fully_closed_v51 as model


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
CONFIGS = {
    "low_threshold": {
        "hybrid_config": {"threshold_shift": -0.06, "control_capacity_scale": 1.0},
        "topology_enabled": True,
    },
    "high_threshold": {
        "hybrid_config": {"threshold_shift": 0.06, "control_capacity_scale": 1.0},
        "topology_enabled": True,
    },
    "weak_control": {
        "hybrid_config": {"threshold_shift": 0.0, "control_capacity_scale": 0.75},
        "topology_enabled": True,
    },
    "strong_control": {
        "hybrid_config": {"threshold_shift": 0.0, "control_capacity_scale": 1.25},
        "topology_enabled": True,
    },
    "topology_off": {
        "hybrid_config": {"threshold_shift": 0.0, "control_capacity_scale": 1.0},
        "topology_enabled": False,
    },
}


def main() -> None:
    if os.environ.get("TAIWAN_ALLOW_COMPLETE_RUN") != "1":
        raise RuntimeError("V5.1 sensitivity blocked until the complete-model gate passes")
    paths = int(os.environ.get("TAIWAN_V51_SENSITIVITY_PATHS", "24"))
    results = {}
    for name, config in CONFIGS.items():
        rows, aggregate, diagnostics, verification = model.run(
            paths=paths, months=360,
            hybrid_config=config["hybrid_config"],
            topology_enabled=config["topology_enabled"],
        )
        hybrid_summary = {
            cell: diag["hybrid"] for cell, diag in diagnostics.items()
        }
        results[name] = {
            "config": config,
            "paths": paths,
            "aggregate": aggregate,
            "conditional": rows,
            "hybrid_diagnostics": hybrid_summary,
            "topology_diagnostics": {
                cell: diag["topology"] for cell, diag in diagnostics.items()
            },
            "runtime_validation": verification["runtime_validation"],
        }
        (OUT / f"taiwan_v51_sensitivity_{name}.json").write_text(
            json.dumps(results[name], ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"TAIWAN_V51_SENSITIVITY_{name.upper()}: PASS", flush=True)
    (OUT / "taiwan_v51_sensitivity_all.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
