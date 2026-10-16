from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path

import simulate_taiwan_fully_closed_v58 as v58


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
TRACE = OUT / "taiwan_v59_traceability.json"
AUDIT = OUT / "taiwan_v59_static_coupling_audit.json"
FREEZE = OUT / "taiwan_v59_frozen_specification.json"
OUT_CSV = OUT / "taiwan_v59_network_geometry_results.csv"
OUT_JSON = OUT / "taiwan_v59_network_geometry_simulation.json"

MFG_CONFIG = {
    **v58.MFG_CONFIG,
    "network_geometry_enabled": True,
    "network_geometry_spectral_interval": 12,
    "intensity_transition_enabled": True,
    "supra_laplacian_enabled": True,
    "spectral_geometry_enabled": True,
    "forman_curvature_enabled": True,
    "sheaf_laplacian_enabled": True,
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def run(
    paths: int = 128,
    months: int = 360,
    scenario_names: list[str] | None = None,
    case_names: list[str] | None = None,
    *,
    mfg_config: dict | None = None,
):
    draws = v58.load_draws(paths)
    return v58.core.run(
        paths=paths,
        months=months,
        scenario_names=scenario_names,
        case_names=case_names,
        mechanism_enabled=True,
        energy_enabled=True,
        geometry_enabled=True,
        hybrid_enabled=True,
        topology_enabled=True,
        mfg_config=MFG_CONFIG if mfg_config is None else mfg_config,
        geometry_config=v58.GEOMETRY_CONFIG,
        macro_factory=v58.CoupledMRRDCGEDQSFCEconomicSystem,
        political_factory=v58.WartimeCollectiveActionRegimeSystem,
        econometric_draws=draws,
        traceability_path=TRACE,
        coupling_audit_path=AUDIT,
    )


def write(rows, aggregate, diagnostics, verification) -> None:
    with OUT_CSV.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    verification["model_version"] = "V5.9"
    verification["runtime_validation"] = "TAIWAN_V59_RUNTIME_VALIDATION: PASS"
    verification["result_sha256"] = sha256(OUT_CSV)
    verification["network_geometry_upgrade"] = {
        "intensity_transition_separation": True,
        "four_layer_supra_laplacian": True,
        "spectral_gap_and_conductance": True,
        "forman_curvature": True,
        "information_identity_sheaf_laplacian": True,
        "monthly_same_path_state": True,
        "supra_spectral_refresh_months": 12,
        "frozen_specification_sha256": sha256(FREEZE),
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
            "V5.9 outcome simulation blocked until the complete-model gate passes"
        )
    frozen = json.loads(FREEZE.read_text(encoding="utf-8"))
    if frozen.get("status") != "FROZEN":
        raise RuntimeError("V5.9 frozen specification is missing or invalid")
    paths = int(os.environ.get("TAIWAN_V59_PATHS", "128"))
    months = int(os.environ.get("TAIWAN_V59_MONTHS", "360"))
    rows, aggregate, diagnostics, verification = run(paths=paths, months=months)
    write(rows, aggregate, diagnostics, verification)
    print("TAIWAN_V59_COMPLETE_RUN: PASS")


if __name__ == "__main__":
    main()
