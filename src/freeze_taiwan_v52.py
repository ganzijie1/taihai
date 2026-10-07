from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import geomloss
import torch
import torchdiffeq


ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "work"
OUT = ROOT / "outputs"
FILES = [
    "fp_solvers_v52.py", "cognitive_geometry_v52.py",
    "continuous_multipop_mfg_v47.py", "simulate_taiwan_multidomain_v42.py",
    "simulate_taiwan_fully_closed_v47.py", "simulate_taiwan_fully_closed_v52.py",
    "audit_taiwan_v52.py", "diagnostic_taiwan_v52.py",
]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main() -> None:
    trace_path = OUT / "taiwan_v52_traceability.json"
    audit_path = OUT / "taiwan_v52_static_coupling_audit.json"
    diagnostic_path = OUT / "taiwan_v52_diagnostic_only.json"
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    diagnostic = json.loads(diagnostic_path.read_text(encoding="utf-8"))
    if not (
        trace["gate_status"] == "PASS"
        and audit["status"] == "PASS"
        and diagnostic["status"] == "PASS"
    ):
        raise RuntimeError("cannot freeze V5.2 before every implementation gate passes")
    frozen = {
        "model_version": "V5.2",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "paths": 128,
        "campaign_months": 360,
        "intervention_cases": 5,
        "trade_conditions": 6,
        "seed_contract": "inherited V5.1 common-random-number schedule",
        "mfg": {
            "graphon_layers": ["information", "economic", "identity", "casualty"],
            "graphon_mode": "multilayer",
            "kernel_uncertainty": {
                "distribution": "frozen pathwise logistic-normal scenario prior",
                "log_scale": 0.06,
                "identification": "scenario_prior, not posterior",
            },
            "major_types": ["government", "military", "major_party", "platform_media"],
            "minor_action_channels": [
                "collective_action", "service", "migration", "hoarding",
                "information_forward",
            ],
            "fixed_point_tolerance": 5.0e-4,
            "density_mass_tolerance": 1.0e-10,
            "finite_nodes_per_type": 3,
        },
        "fp": {
            "production_backend": "torch_fvm",
            "device": "cuda",
            "advection": "conservative upwind finite volume with CFL subcycling",
            "diffusion": "backward Euler batched tridiagonal solve",
            "boundary": "reflecting zero flux",
            "calibration_backend": "torchdiffeq RK4 method-of-lines",
        },
        "cognitive_geometry": {
            "fixed_point": "Fisher-Rao geodesic",
            "unbalanced_mass": "WFR reaction-transport splitting",
            "transport_loss": "GeomLoss tensorized debiased Sinkhorn",
            "blur": 0.08,
            "reach": 0.45,
            "refresh_months": 12,
        },
        "graph_ablations": ["none", "block", "low_rank", "multilayer", "finite"],
        "equilibrium_validation": [
            "joint fixed-point residual", "mean exploitability bound",
            "worst-type exploitability bound", "finite-network field error",
            "finite-network payoff-deviation bound",
        ],
        "runtime_thresholds": {
            "major_fixed_point": 5.0e-4,
            "mean_exploitability": 1.0e-2,
            "worst_type_exploitability": 5.0e-2,
            "finite_network_field_error": 2.5e-1,
            "finite_network_payoff_deviation": 2.0e-1,
            "fp_mass_residual": 1.0e-10,
            "fp_minimum_density": -1.0e-10,
            "geomloss_failures": 0,
        },
        "hard_capacity_network_flow": "unchanged operational_constraints_v47 solver",
        "dependencies": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "torchdiffeq": getattr(torchdiffeq, "__version__", "0.2.5"),
            "geomloss": getattr(geomloss, "__version__", "0.3.1"),
        },
        "identification": (
            "new Graphon, action and geometry coefficients are transparent scenario priors; "
            "formal results require kernel and graph-structure sensitivity reporting"
        ),
        "code_sha256": {name: digest(WORK / name) for name in FILES},
        "traceability_sha256": digest(trace_path),
        "audit_sha256": digest(audit_path),
        "diagnostic_sha256": digest(diagnostic_path),
    }
    (OUT / "taiwan_v52_frozen_specification.json").write_text(
        json.dumps(frozen, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("TAIWAN_V52_SPECIFICATION_FROZEN: PASS")


if __name__ == "__main__":
    main()
