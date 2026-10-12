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
    trace_path = OUT / "taiwan_v59_traceability.json"
    audit_path = OUT / "taiwan_v59_static_coupling_audit.json"
    diagnostic_path = OUT / "taiwan_v59_network_geometry_diagnostic.json"
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    diagnostic = json.loads(diagnostic_path.read_text(encoding="utf-8"))
    if not (
        trace.get("gate_status") == "PASS"
        and audit.get("status") == "PASS"
        and audit.get("simulation_authorized") is True
        and diagnostic.get("status") == "PASS"
    ):
        raise RuntimeError("V5.9 cannot be frozen before all gates pass")
    files = [
        WORK / "network_geometry_v59.py",
        WORK / "continuous_multipop_mfg_v47.py",
        WORK / "simulate_taiwan_fully_closed_v59.py",
        WORK / "simulate_taiwan_v59_checkpointed.py",
        WORK / "diagnostic_taiwan_v59.py",
        WORK / "audit_taiwan_v59.py",
        WORK / "ablate_taiwan_v59.py",
        WORK / "validate_taiwan_v59.py",
        ROOT / "scripts" / "audit_manifest.py",
        OUT / "台海模型网络几何深化与V5.9升级路线_飞书LaTeX.md",
        OUT / "taiwan_v59_network_geometry_source_ledger.json",
        OUT / "taiwan_v55_econometric_calibration.json",
        OUT / "taiwan_v55_econometric_draws.npz",
        trace_path,
        audit_path,
        diagnostic_path,
    ]
    payload = {
        "model_version": "V5.9",
        "status": "FROZEN",
        "frozen_at": "2026-10-10 Asia/Shanghai",
        "method_document": "https://my.feishu.cn/docx/Ou3NdGj7CoGUqwxfht5clSdqnxd",
        "simulation_contract": {
            "paths": 128,
            "months": 360,
            "intervention_cases": 5,
            "trade_scenarios": 6,
            "master_clock": "monthly same-path plus unique event clock",
            "draw_contract": "one frozen V5.5 econometric draw per path shared across cases/scenarios",
            "outcome_gate": "manifest, static audit and diagnostics must pass before results",
        },
        "network_geometry_contract": {
            "active_scope": [
                "intensity-transition separation", "four-layer supra-Laplacian",
                "spectral gap and conductance", "weighted Forman curvature",
                "information-identity scalar-stalk sheaf Laplacian",
            ],
            "state_update": "monthly bounded proximal update on each master path",
            "supra_spectral_refresh_months": 12,
            "deferred_by_method_document_section_19": [
                "directed Ollivier curvature", "Hodge decomposition",
                "hypergraph and simplicial dynamics", "zigzag persistence",
                "full Gromov-Wasserstein proximal kernel estimation",
            ],
        },
        "acceptance_thresholds": {
            "macro_outer": 0.003,
            "cge": 0.002,
            "dsge": 0.0008,
            "mfg": 0.0005,
            "transition_row": 1.0e-12,
            "temporal_log_step": 0.18,
            "sheaf_minimum_eigenvalue": -1.0e-12,
            "bottleneck_risk_lower": 0.0,
            "bottleneck_risk_upper": 1.0,
        },
        "ablation_contract": {
            "paths": 16,
            "months": 360,
            "trade_scenarios": ["baseline"],
            "intervention_cases": 5,
            "common_random_numbers": True,
            "one_at_a_time": [
                "intensity_transition", "supra_laplacian", "spectral_geometry",
                "forman_curvature", "sheaf_laplacian", "all_network_geometry",
            ],
        },
        "files": {str(path.relative_to(ROOT)): digest(path) for path in files},
    }
    target = OUT / "taiwan_v59_frozen_specification.json"
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"TAIWAN_V59_FREEZE: PASS; sha256={digest(target)}")


if __name__ == "__main__":
    main()
