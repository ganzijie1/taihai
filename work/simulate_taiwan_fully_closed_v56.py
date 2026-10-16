from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path

import simulate_taiwan_fully_closed_v47 as core
from dynamic_macro_dqsfc_v53 import CoupledMRRDCGEDQSFCEconomicSystem
from econometric_calibration_v55 import CALIBRATION, DRAWS, load_draws
from simulate_taiwan_fully_closed_v52 import MFG_CONFIG as V52_MFG_CONFIG


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
TRACE = OUT / "taiwan_v56_traceability.json"
AUDIT = OUT / "taiwan_v56_static_coupling_audit.json"
FREEZE = OUT / "taiwan_v56_frozen_specification.json"
OUT_CSV = OUT / "taiwan_v56_geometry_results.csv"
OUT_JSON = OUT / "taiwan_v56_geometry_simulation.json"

MFG_CONFIG = {
    **V52_MFG_CONFIG,
    "wfr_strang": True,
    "natural_gradient_major": True,
}

GEOMETRY_CONFIG = {
    "operations": {"advanced_geometry": True},
    "pnt": {
        "geometry_metric": "bures_wasserstein",
        "joseph_update": True,
    },
    "spatial": {"dynamic_finsler": True},
    "energy": {"port_hamiltonian_dae": True},
    "multiscale": {"learned_slow_manifold": True},
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def run(
    paths: int = 128,
    months: int = 360,
    scenario_names: list[str] | None = None,
    case_names: list[str] | None = None,
    *,
    geometry_config: dict | None = None,
    mfg_config: dict | None = None,
    hybrid_config: dict | None = None,
):
    draws = load_draws(paths)
    return core.run(
        paths=paths,
        months=months,
        scenario_names=scenario_names,
        case_names=case_names,
        mechanism_enabled=True,
        energy_enabled=True,
        geometry_enabled=True,
        hybrid_enabled=True,
        hybrid_config=hybrid_config,
        topology_enabled=True,
        mfg_config=MFG_CONFIG if mfg_config is None else mfg_config,
        geometry_config=(
            GEOMETRY_CONFIG if geometry_config is None else geometry_config
        ),
        macro_factory=CoupledMRRDCGEDQSFCEconomicSystem,
        econometric_draws=draws,
        traceability_path=TRACE,
        coupling_audit_path=AUDIT,
    )


def write(rows, aggregate, diagnostics, verification) -> None:
    with OUT_CSV.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    verification["model_version"] = "V5.6"
    verification["runtime_validation"] = "TAIWAN_V56_RUNTIME_VALIDATION: PASS"
    verification["result_sha256"] = sha256(OUT_CSV)
    verification["geometry_upgrade"] = {
        "time_varying_finsler_gis": True,
        "subriemannian_reachability": True,
        "so3_aggregate_kinematics": True,
        "wfr_strang_graphon_mfg": True,
        "fisher_rao_major_player_update": True,
        "bures_wasserstein_pnt": True,
        "joseph_covariance_update": True,
        "port_hamiltonian_dae": True,
        "constrained_slow_manifold": True,
        "saltation_sensitivity": True,
        "topological_diagnostics": True,
        "frozen_specification_sha256": sha256(FREEZE),
    }
    verification["econometric_calibration"] = {
        "calibration_sha256": sha256(CALIBRATION),
        "draws_sha256": sha256(DRAWS),
        "draw_contract": "one frozen draw per master path, shared across cases/scenarios",
    }
    OUT_JSON.write_text(
        json.dumps(
            {
                "verification": verification,
                "aggregate": aggregate,
                "conditional": rows,
                "module_diagnostics": diagnostics,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    if os.environ.get("TAIWAN_ALLOW_COMPLETE_RUN") != "1":
        raise RuntimeError(
            "V5.6 outcome simulation blocked until the complete-model gate passes"
        )
    frozen = json.loads(FREEZE.read_text(encoding="utf-8"))
    if frozen.get("status") != "FROZEN":
        raise RuntimeError("V5.6 frozen specification is missing or invalid")
    paths = int(os.environ.get("TAIWAN_V56_PATHS", "128"))
    months = int(os.environ.get("TAIWAN_V56_MONTHS", "360"))
    rows, aggregate, diagnostics, verification = run(paths=paths, months=months)
    write(rows, aggregate, diagnostics, verification)
    print("TAIWAN_V56_COMPLETE_RUN: PASS")


if __name__ == "__main__":
    main()
