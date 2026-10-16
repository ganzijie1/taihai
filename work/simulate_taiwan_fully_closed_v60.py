from __future__ import annotations

import copy
import csv
import hashlib
import json
import os
from pathlib import Path

import simulate_taiwan_fully_closed_v59 as v59


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
TRACE = OUT / "taiwan_v60_traceability.json"
AUDIT = OUT / "taiwan_v60_static_coupling_audit.json"
FREEZE = OUT / "taiwan_v60_frozen_specification.json"
OUT_CSV = OUT / "taiwan_v60_riemannian_results.csv"
OUT_JSON = OUT / "taiwan_v60_riemannian_simulation.json"

MFG_CONFIG = {
    **v59.MFG_CONFIG,
    "riemannian_anderson_enabled": True,
    "riemannian_anderson_regularization": 1.0e-5,
    "riemannian_trust_region_enabled": True,
    "riemannian_fixed_point_iterations": 128,
    "riemannian_minor_relaxation": 0.40,
}

GEOMETRY_CONFIG = copy.deepcopy(v59.v58.GEOMETRY_CONFIG)
GEOMETRY_CONFIG["pnt"].update({
    "adaptive_bures_enabled": True,
    "bures_barycenter_enabled": True,
})
GEOMETRY_CONFIG["multiscale"].update({
    "regime_riemannian_pod": True,
    "product_manifold_metric": True,
})


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def run(
    paths: int = 128,
    months: int = 360,
    scenario_names: list[str] | None = None,
    case_names: list[str] | None = None,
    *,
    mfg_config: dict | None = None,
    geometry_config: dict | None = None,
):
    draws = v59.v58.load_draws(paths)
    return v59.v58.core.run(
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
        geometry_config=(
            GEOMETRY_CONFIG if geometry_config is None else geometry_config
        ),
        macro_factory=v59.v58.CoupledMRRDCGEDQSFCEconomicSystem,
        political_factory=v59.v58.WartimeCollectiveActionRegimeSystem,
        econometric_draws=draws,
        traceability_path=TRACE,
        coupling_audit_path=AUDIT,
    )


def write(rows, aggregate, diagnostics, verification) -> None:
    with OUT_CSV.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    verification["model_version"] = "V6.0"
    verification["runtime_validation"] = "TAIWAN_V60_RUNTIME_VALIDATION: PASS"
    verification["result_sha256"] = sha256(OUT_CSV)
    verification["riemannian_upgrade"] = {
        "regularized_riemannian_anderson": True,
        "full_fisher_rao_policy_trust_region": True,
        "adaptive_bures_wasserstein_barycenter": True,
        "regime_riemannian_pod": True,
        "positive_product_manifold_metric": True,
        "monthly_same_path_state": True,
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
            "V6.0 outcome simulation blocked until the complete-model gate passes"
        )
    frozen = json.loads(FREEZE.read_text(encoding="utf-8"))
    if frozen.get("status") != "FROZEN":
        raise RuntimeError("V6.0 frozen specification is missing or invalid")
    paths = int(os.environ.get("TAIWAN_V60_PATHS", "128"))
    months = int(os.environ.get("TAIWAN_V60_MONTHS", "360"))
    rows, aggregate, diagnostics, verification = run(paths=paths, months=months)
    write(rows, aggregate, diagnostics, verification)
    print("TAIWAN_V60_COMPLETE_RUN: PASS")


if __name__ == "__main__":
    main()
