from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path

import simulate_taiwan_fully_closed_v47 as core


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
TRACE = OUT / "taiwan_v52_traceability.json"
AUDIT = OUT / "taiwan_v52_static_coupling_audit.json"
OUT_CSV = OUT / "taiwan_v52_graphon_gmfg_results.csv"
OUT_JSON = OUT / "taiwan_v52_graphon_gmfg_simulation.json"


MFG_CONFIG = {
    "advanced_graphon": True,
    "graphon_mode": "multilayer",
    "graphon_uncertainty_scale": 0.06,
    "fp_backend": "torch_fvm",
    "geomloss_interval": 12,
    "finite_nodes_per_type": 3,
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def run(
    paths: int = 128, months: int = 360,
    scenario_names: list[str] | None = None,
    case_names: list[str] | None = None,
):
    return core.run(
        paths=paths, months=months,
        scenario_names=scenario_names, case_names=case_names,
        mechanism_enabled=True, energy_enabled=True, geometry_enabled=True,
        hybrid_enabled=True, topology_enabled=True,
        mfg_config=MFG_CONFIG,
        traceability_path=TRACE, coupling_audit_path=AUDIT,
    )


def write(rows, aggregate, diagnostics, verification) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    verification["result_sha256"] = sha256(OUT_CSV)
    verification["model_version"] = "V5.2"
    verification["graphon_gmfg"] = {
        "layers": ["information", "economic", "identity", "casualty"],
        "major_types": ["government", "military", "major_party", "platform_media"],
        "action_channels": [
            "collective_action", "service", "migration", "hoarding",
            "information_forward",
        ],
        "kernel_uncertainty": "pathwise frozen logistic-normal interval prior",
        "fp_backend": "CUDA conservative semi-implicit finite volume",
        "optional_calibration_backend": "torchdiffeq",
        "cognitive_geometry": "Fisher-Rao, WFR reaction split, GeomLoss Sinkhorn",
    }
    OUT_JSON.write_text(json.dumps({
        "verification": verification,
        "aggregate": aggregate,
        "conditional": rows,
        "module_diagnostics": diagnostics,
    }, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    if os.environ.get("TAIWAN_ALLOW_COMPLETE_RUN") != "1":
        raise RuntimeError(
            "V5.2 outcome simulation blocked until the complete-model gate passes"
        )
    paths = int(os.environ.get("TAIWAN_V52_PATHS", "128"))
    months = int(os.environ.get("TAIWAN_V52_MONTHS", "360"))
    rows, aggregate, diagnostics, verification = run(paths=paths, months=months)
    write(rows, aggregate, diagnostics, verification)
    print("TAIWAN_V52_COMPLETE_RUN: PASS")


if __name__ == "__main__":
    main()
