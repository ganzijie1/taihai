from __future__ import annotations

import csv
import hashlib
import json
import os
from functools import partial
from pathlib import Path

import numpy as np

import simulate_taiwan_fully_closed_v60 as v60
from sectoral_economy_v61 import CoupledSectoralEconomicSystem


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
TRACE = OUT / "taiwan_v61_traceability.json"
AUDIT = OUT / "taiwan_v61_static_coupling_audit.json"
FREEZE = OUT / "taiwan_v61_frozen_specification.json"
OUT_CSV = OUT / "taiwan_v61_sectoral_results.csv"
OUT_JSON = OUT / "taiwan_v61_sectoral_simulation.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def load_frozen_draws(paths: int) -> dict[str, np.ndarray]:
    with np.load(OUT / "taiwan_v55_econometric_draws.npz") as data:
        available = min(value.shape[0] for value in data.values())
        if paths > available:
            raise ValueError(
                f"requested {paths} paths but frozen posterior contains {available}"
            )
        return {key: data[key][:paths].copy() for key in data.files}


def validate_sector_diagnostics(diagnostics: dict) -> None:
    errors = []
    for cell, item in diagnostics.items():
        sector = item["macro"].get("three_sector_physical_economy")
        if sector is None:
            errors.append(f"{cell}: missing three-sector diagnostics")
            continue
        if not sector.get("layer_enabled", True):
            continue
        if sector["max_labor_share_residual"] > 1.0e-12:
            errors.append(f"{cell}: labor-share simplex residual")
        if sector["max_land_share_residual"] > 1.0e-12:
            errors.append(f"{cell}: land-share simplex residual")
        if sector["max_service_stock_residual"] > 1.0e-12:
            errors.append(f"{cell}: service stock is not zero")
        if sector["max_node_bound_violation"] > 1.0e-12:
            errors.append(f"{cell}: node realization bound")
        if sector["max_sector_accounting_residual"] > 1.0e-10:
            errors.append(f"{cell}: sector-to-actor accounting residual")
        if sector["minimum_water"] < 0.12 - 1.0e-12:
            errors.append(f"{cell}: water state lower bound")
        if sector["minimum_soil"] < 0.35 - 1.0e-12:
            errors.append(f"{cell}: soil state lower bound")
        if sector["minimum_firm_mass"] < 0.15 - 1.0e-12:
            errors.append(f"{cell}: firm mass lower bound")
        if sector["minimum_public_service_capacity"] < 0.10 - 1.0e-12:
            errors.append(f"{cell}: public service capacity lower bound")
    if errors:
        raise RuntimeError("V6.1 sectoral runtime validation failed: " + "; ".join(errors))


def run(
    paths: int = 128,
    months: int = 360,
    scenario_names: list[str] | None = None,
    case_names: list[str] | None = None,
    *,
    sector_flags: dict | None = None,
):
    draws = load_frozen_draws(paths)
    macro_factory = (
        CoupledSectoralEconomicSystem
        if not sector_flags
        else partial(CoupledSectoralEconomicSystem, **sector_flags)
    )
    result = v60.v59.v58.core.run(
        paths=paths,
        months=months,
        scenario_names=scenario_names,
        case_names=case_names,
        mechanism_enabled=True,
        energy_enabled=True,
        geometry_enabled=True,
        hybrid_enabled=True,
        topology_enabled=True,
        mfg_config=v60.MFG_CONFIG,
        geometry_config=v60.GEOMETRY_CONFIG,
        macro_factory=macro_factory,
        political_factory=v60.v59.v58.WartimeCollectiveActionRegimeSystem,
        econometric_draws=draws,
        traceability_path=TRACE,
        coupling_audit_path=AUDIT,
    )
    validate_sector_diagnostics(result[2])
    return result


def write(rows, aggregate, diagnostics, verification) -> None:
    with OUT_CSV.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    verification["model_version"] = "V6.1"
    verification["runtime_validation"] = "TAIWAN_V61_RUNTIME_VALIDATION: PASS"
    verification["result_sha256"] = sha256(OUT_CSV)
    verification["three_sector_upgrade"] = {
        "agriculture_biophysical_state": True,
        "industrial_operability_and_input_bottleneck": True,
        "nonstorable_service_queue": True,
        "sectoral_labor_reallocation": True,
        "io_sam_dq_sfc_bridge": True,
        "monthly_same_path_bidirectional_feedback": True,
        "no_posthoc_probability_adjustment": True,
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
            "V6.1 outcome simulation blocked until the complete-model gate passes"
        )
    frozen = json.loads(FREEZE.read_text(encoding="utf-8"))
    if frozen.get("status") != "FROZEN":
        raise RuntimeError("V6.1 frozen specification is missing or invalid")
    paths = int(os.environ.get("TAIWAN_V61_PATHS", "128"))
    months = int(os.environ.get("TAIWAN_V61_MONTHS", "360"))
    rows, aggregate, diagnostics, verification = run(paths=paths, months=months)
    write(rows, aggregate, diagnostics, verification)
    print("TAIWAN_V61_COMPLETE_RUN: PASS")


if __name__ == "__main__":
    main()
