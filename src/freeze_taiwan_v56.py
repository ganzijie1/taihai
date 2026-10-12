from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "work"
OUT = ROOT / "outputs"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main() -> None:
    trace_path = OUT / "taiwan_v56_traceability.json"
    audit_path = OUT / "taiwan_v56_static_coupling_audit.json"
    diagnostic_path = OUT / "taiwan_v56_geometry_diagnostic.json"
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    diagnostic = json.loads(diagnostic_path.read_text(encoding="utf-8"))
    if not (
        trace.get("gate_status") == "PASS"
        and audit.get("status") == "PASS"
        and audit.get("simulation_authorized") is True
        and diagnostic.get("status") == "PASS"
    ):
        raise RuntimeError("V5.6 cannot be frozen before all gates pass")

    files = [
        WORK / "operational_constraints_v47.py",
        WORK / "spatial_control_network_v47.py",
        WORK / "continuous_multipop_mfg_v47.py",
        WORK / "four_party_energy_network_v49.py",
        WORK / "geometric_multiscale_v50.py",
        WORK / "hybrid_filippov_impulse_v51.py",
        WORK / "simulate_taiwan_fully_closed_v47.py",
        WORK / "simulate_taiwan_fully_closed_v56.py",
        WORK / "diagnostic_taiwan_v56.py",
        WORK / "audit_taiwan_v56.py",
        WORK / "validate_taiwan_v56.py",
        WORK / "ablate_taiwan_v56.py",
        OUT / "taiwan_differential_geometry_upgrade_feishu.md",
        OUT / "taiwan_v55_econometric_calibration.json",
        OUT / "taiwan_v55_econometric_draws.npz",
        OUT / "taiwan_v55_econometric_source_ledger.json",
        trace_path, audit_path, diagnostic_path,
    ]
    payload = {
        "model_version": "V5.6", "status": "FROZEN",
        "frozen_at": "2026-10-08 Asia/Shanghai",
        "method_document": "https://my.feishu.cn/docx/DkMsdAwMCoECTgxykCSc7D3Qneg",
        "simulation_contract": {
            "paths": 128, "months": 360, "intervention_cases": 5,
            "trade_scenarios": 6, "master_clock": "monthly same-path plus unique event clock",
            "draw_contract": "one frozen V5.5 econometric draw per master path, shared across cases/scenarios",
            "outcome_gate": "no V5.6 substantive result before trace, static audit, diagnostic and freeze all pass",
        },
        "ablation_contract": {
            "paths": 32, "months": 360, "trade_scenarios": ["baseline"],
            "intervention_cases": 5, "common_random_numbers": True,
            "one_at_a_time": ["finsler_lie", "wfr_information", "spd_joseph", "port_hamiltonian", "slow_manifold", "saltation_sensitivity"],
        },
        "acceptance_thresholds": {
            "macro_outer": 3e-3, "cge": 2e-3, "dsge": 8e-4, "mfg": 5e-4,
            "randers_norm_strictly_below": 1.0, "so3": 1e-12,
            "dynamic_finsler": 1e-10, "port_hamiltonian_dae": 1e-10,
            "wfr_mass": 1e-10, "saltation_finite_difference": 1e-12,
        },
        "identification_contract": {
            "estimated": ["inherited V5.5 PPML/HMM/predictive macro parameters"],
            "scenario_prior": ["geometric metric loadings", "slow-manifold structural design distribution", "dissipation coefficients"],
            "not_claimed": ["historically identified wartime manifold", "classified operational precision", "causal geometric coefficients"],
        },
        "files": {str(path.relative_to(ROOT)): digest(path) for path in files},
    }
    target = OUT / "taiwan_v56_frozen_specification.json"
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"TAIWAN_V56_FREEZE: PASS; sha256={digest(target)}")


if __name__ == "__main__":
    main()
