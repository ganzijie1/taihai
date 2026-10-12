from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path

import simulate_taiwan_fully_closed_v47 as core
from dynamic_macro_dqsfc_v53 import CoupledMRRDCGEDQSFCEconomicSystem
from econometric_calibration_v55 import CALIBRATION, DRAWS, load_draws
from simulate_taiwan_fully_closed_v52 import MFG_CONFIG


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
TRACE = OUT / "taiwan_v55_traceability.json"
AUDIT = OUT / "taiwan_v55_static_coupling_audit.json"
FREEZE = OUT / "taiwan_v55_frozen_specification.json"
OUT_CSV = OUT / "taiwan_v55_econometric_results.csv"
OUT_JSON = OUT / "taiwan_v55_econometric_simulation.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def run(
    paths: int = 128,
    months: int = 360,
    scenario_names: list[str] | None = None,
    case_names: list[str] | None = None,
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
        topology_enabled=True,
        mfg_config=MFG_CONFIG,
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
    verification["model_version"] = "V5.5"
    verification["runtime_validation"] = "TAIWAN_V55_RUNTIME_VALIDATION: PASS"
    verification["result_sha256"] = sha256(OUT_CSV)
    verification["econometric_calibration"] = {
        "calibration_sha256": sha256(CALIBRATION),
        "draws_sha256": sha256(DRAWS),
        "frozen_specification_sha256": sha256(FREEZE),
        "draw_contract": "one frozen draw per master path, shared across cases/scenarios",
        "estimated": ["PPML trade effects", "HMM transitions", "predictive macro VAR/local projections"],
        "scenario_prior": ["terminal hazards", "alliance intercepts", "wartime elasticities"],
    }
    OUT_JSON.write_text(
        json.dumps(
            {"verification": verification, "aggregate": aggregate,
             "conditional": rows, "module_diagnostics": diagnostics},
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    if os.environ.get("TAIWAN_ALLOW_COMPLETE_RUN") != "1":
        raise RuntimeError("V5.5 outcome simulation blocked until the complete-model gate passes")
    frozen = json.loads(FREEZE.read_text(encoding="utf-8"))
    if frozen.get("status") != "FROZEN":
        raise RuntimeError("V5.5 frozen specification is missing or invalid")
    paths = int(os.environ.get("TAIWAN_V55_PATHS", "128"))
    months = int(os.environ.get("TAIWAN_V55_MONTHS", "360"))
    rows, aggregate, diagnostics, verification = run(paths=paths, months=months)
    write(rows, aggregate, diagnostics, verification)
    print("TAIWAN_V55_COMPLETE_RUN: PASS")


if __name__ == "__main__":
    main()
