from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
PLAN = OUT / "taiwan_v81_inherited_ablation_plan.json"
BASELINE = OUT / "taiwan_v81_simulation.json"
BASELINE_CSV = OUT / "taiwan_v81_results.csv"

SOURCES = (
    ROOT / "work" / "statistical_field_mfg_v80.py",
    ROOT / "work" / "simulate_taiwan_fully_closed_v80.py",
    ROOT / "work" / "simulate_taiwan_fully_closed_v81.py",
    ROOT / "work" / "simulate_taiwan_v81_inherited_ablations.py",
    ROOT / "work" / "audit_taiwan_v81_inherited_ablations.py",
    ROOT / "work" / "audit_taiwan_v81_results.py",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main() -> None:
    baseline_audit = json.loads(
        (OUT / "taiwan_v81_result_audit.json").read_text(encoding="utf-8")
    )
    if baseline_audit.get("artifact_integrity_status") != "PASS":
        raise RuntimeError("V8.1 baseline artifact audit must pass before ablation")
    missing = [str(path) for path in SOURCES if not path.exists()]
    if missing:
        raise RuntimeError("missing ablation sources: " + "; ".join(missing))
    payload = {
        "status": "FROZEN",
        "purpose": "V8.0 inherited one-at-a-time terminal-outcome ablations",
        "baseline": {
            "model_version": "V8.1-DIFFERENTIAL-MANIFOLD-CONSTRAINTS",
            "paths": 128,
            "months": 360,
            "scenario_cells": 30,
            "result_json_sha256": sha256(BASELINE),
            "result_csv_sha256": sha256(BASELINE_CSV),
            "frozen_specification_sha256": sha256(
                OUT / "taiwan_v81_frozen_specification.json"
            ),
        },
        "ablations": {
            "no_statistical_field": {
                "disabled_only": ["v80_non_equilibrium_statistical_field"],
                "environment": {
                    "TAIWAN_V81_DISABLE_STATISTICAL_FIELD": "1",
                    "TAIWAN_V81_DISABLE_ANALYTICAL_MECHANICS": "0",
                },
            },
            "no_analytical_mechanics": {
                "disabled_only": ["v80_analytical_mechanics"],
                "environment": {
                    "TAIWAN_V81_DISABLE_STATISTICAL_FIELD": "0",
                    "TAIWAN_V81_DISABLE_ANALYTICAL_MECHANICS": "1",
                },
            },
        },
        "invariants": {
            "same_master_paths": True,
            "same_event_clock": True,
            "same_seeds": True,
            "same_trade_scenarios": True,
            "same_intervention_regimes": True,
            "same_terminal_classifier": True,
            "same_solver_tolerances": True,
            "posthoc_probability_adjustment": False,
        },
        "source_hashes": {
            str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path)
            for path in SOURCES
        },
    }
    PLAN.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("TAIWAN_V81_INHERITED_ABLATION_PLAN: FROZEN")


if __name__ == "__main__":
    main()
