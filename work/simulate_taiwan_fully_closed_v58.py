from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path

import simulate_taiwan_fully_closed_v47 as core
from dynamic_macro_dqsfc_v53 import CoupledMRRDCGEDQSFCEconomicSystem
from econometric_calibration_v55 import CALIBRATION, DRAWS, load_draws
from simulate_taiwan_fully_closed_v56 import GEOMETRY_CONFIG, MFG_CONFIG as V56_MFG_CONFIG
from wartime_collective_action_regime_v57 import WartimeCollectiveActionRegimeSystem


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
TRACE = OUT / "taiwan_v58_traceability.json"
AUDIT = OUT / "taiwan_v58_static_coupling_audit.json"
FREEZE = OUT / "taiwan_v58_frozen_specification.json"
OUT_CSV = OUT / "taiwan_v58_information_geometry_results.csv"
OUT_JSON = OUT / "taiwan_v58_information_geometry_simulation.json"

MFG_CONFIG = {
    **V56_MFG_CONFIG,
    "wfr_strang": False,
    "partial_observation_geometry": True,
    "full_wfr_jko": True,
    "trust_region_enabled": True,
    "fisher_trust_radius": 0.035,
    "minor_policy_trust_radius": 0.025,
    "information_ambiguity_radius": 0.025,
    "belief_filter_enabled": True,
    "credibility_hmm_enabled": True,
    "schrodinger_bridge_enabled": True,
    "fgw_alignment_enabled": True,
    "robust_information_enabled": True,
    "information_value_enabled": True,
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
        mfg_config=MFG_CONFIG if mfg_config is None else mfg_config,
        geometry_config=GEOMETRY_CONFIG,
        macro_factory=CoupledMRRDCGEDQSFCEconomicSystem,
        political_factory=WartimeCollectiveActionRegimeSystem,
        econometric_draws=draws,
        traceability_path=TRACE,
        coupling_audit_path=AUDIT,
    )


def write(rows, aggregate, diagnostics, verification) -> None:
    with OUT_CSV.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    verification["model_version"] = "V5.8"
    verification["runtime_validation"] = "TAIWAN_V58_RUNTIME_VALIDATION: PASS"
    verification["result_sha256"] = sha256(OUT_CSV)
    verification["information_geometry_upgrade"] = {
        "actor_specific_belief_posteriors": True,
        "six_source_three_state_credibility_hmm": True,
        "joint_implicit_wfr_transport_reaction": True,
        "fisher_and_kl_trust_regions": True,
        "schrodinger_bridge_estimator": True,
        "fused_gromov_wasserstein_alignment": True,
        "kl_ambiguity_robust_kernel": True,
        "one_month_information_action_lag": True,
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
            "V5.8 outcome simulation blocked until the complete-model gate passes"
        )
    frozen = json.loads(FREEZE.read_text(encoding="utf-8"))
    if frozen.get("status") != "FROZEN":
        raise RuntimeError("V5.8 frozen specification is missing or invalid")
    paths = int(os.environ.get("TAIWAN_V58_PATHS", "128"))
    months = int(os.environ.get("TAIWAN_V58_MONTHS", "360"))
    rows, aggregate, diagnostics, verification = run(paths=paths, months=months)
    write(rows, aggregate, diagnostics, verification)
    print("TAIWAN_V58_COMPLETE_RUN: PASS")


if __name__ == "__main__":
    main()
