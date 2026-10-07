from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "work"
OUT = ROOT / "outputs"
FILES = [
    "hybrid_filippov_impulse_v51.py", "topological_dynamics_v51.py",
    "dynamic_macro_equilibrium_v47.py",
    "simulate_taiwan_multidomain_v42.py",
    "simulate_taiwan_fully_closed_v47.py", "simulate_taiwan_fully_closed_v51.py",
    "audit_taiwan_v51.py", "diagnostic_taiwan_v51.py",
]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main() -> None:
    trace = json.loads((OUT / "taiwan_v51_traceability.json").read_text(encoding="utf-8"))
    audit = json.loads((OUT / "taiwan_v51_static_coupling_audit.json").read_text(encoding="utf-8"))
    if trace["gate_status"] != "PASS" or audit["status"] != "PASS":
        raise RuntimeError("cannot freeze V5.1 before the complete-model gate passes")
    frozen = {
        "model_version": "V5.1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "paths": 128,
        "campaign_months": 360,
        "intervention_cases": 5,
        "trade_conditions": 6,
        "seed_contract": "inherited V5.0 common-random-number schedule",
        "filippov": {
            "surfaces": ["logistics", "financial", "mobilization", "command"],
            "on_thresholds": [0.34, 0.78, 0.50, 0.44],
            "off_thresholds": [0.24, 0.62, 0.36, 0.31],
            "minimum_dwell_months": 2,
            "boundary_band": 0.025,
        },
        "impulses": ["homeland_island_damage", "leader_disruption", "coup", "collapse"],
        "topological_dynamics": {
            "symbolic_cells": 16,
            "partition": "four Filippov regime indicators",
            "diagnostics": [
                "pathwise transition accounting", "Morse SCC decomposition",
                "spectral-radius entropy proxy",
            ],
            "feedback": "bounded same-path command, financial, logistics, mobilization and panic factors",
        },
        "sensitivity_design": {
            "low_threshold": {"threshold_shift": -0.06, "control_capacity_scale": 1.0},
            "high_threshold": {"threshold_shift": 0.06, "control_capacity_scale": 1.0},
            "weak_control": {"threshold_shift": 0.0, "control_capacity_scale": 0.75},
            "strong_control": {"threshold_shift": 0.0, "control_capacity_scale": 1.25},
            "topology_off": {"topology_enabled": False},
            "paths": 24,
            "common_random_numbers": True,
        },
        "identification": "thresholds and control capacities are scenario_prior",
        "code_sha256": {name: digest(WORK / name) for name in FILES},
        "traceability_sha256": digest(OUT / "taiwan_v51_traceability.json"),
        "audit_sha256": digest(OUT / "taiwan_v51_static_coupling_audit.json"),
    }
    (OUT / "taiwan_v51_frozen_specification.json").write_text(
        json.dumps(frozen, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("TAIWAN_V51_SPECIFICATION_FROZEN: PASS")


if __name__ == "__main__":
    main()
