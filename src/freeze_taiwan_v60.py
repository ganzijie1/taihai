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
    trace_path = OUT / "taiwan_v60_traceability.json"
    audit_path = OUT / "taiwan_v60_static_coupling_audit.json"
    diagnostic_path = OUT / "taiwan_v60_riemannian_diagnostic.json"
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    diagnostic = json.loads(diagnostic_path.read_text(encoding="utf-8"))
    if not (
        trace.get("gate_status") == "PASS"
        and audit.get("status") == "PASS"
        and audit.get("simulation_authorized") is True
        and diagnostic.get("status") == "PASS"
    ):
        raise RuntimeError("V6.0 cannot be frozen before all gates pass")
    files = [
        WORK / "riemannian_geometry_v60.py",
        WORK / "continuous_multipop_mfg_v47.py",
        WORK / "operational_constraints_v47.py",
        WORK / "geometric_multiscale_v50.py",
        WORK / "simulate_taiwan_fully_closed_v47.py",
        WORK / "simulate_taiwan_fully_closed_v60.py",
        WORK / "simulate_taiwan_v60_checkpointed.py",
        WORK / "diagnostic_taiwan_v60.py",
        WORK / "audit_taiwan_v60.py",
        ROOT / "scripts" / "audit_manifest.py",
        OUT / "台海模型的黎曼几何深化与V6升级路线_飞书LaTeX.md",
        OUT / "taiwan_v60_riemannian_source_ledger.json",
        OUT / "taiwan_v55_econometric_calibration.json",
        OUT / "taiwan_v55_econometric_draws.npz",
        trace_path,
        audit_path,
        diagnostic_path,
    ]
    payload = {
        "model_version": "V6.0",
        "status": "FROZEN",
        "frozen_at": "2026-10-10 Asia/Shanghai",
        "method_document": "https://my.feishu.cn/docx/UzrzdYIE7o43byxTBgbc4wRbnZf",
        "simulation_contract": {
            "paths": 128,
            "months": 360,
            "intervention_cases": 5,
            "trade_scenarios": 6,
            "master_clock": "monthly same-path plus unique event clock",
            "draw_contract": "one frozen V5.5 econometric draw per path shared across cases/scenarios",
            "outcome_gate": "manifest, static audit and diagnostics must pass before results",
        },
        "riemannian_contract": {
            "active_scope": [
                "regularized Riemannian Anderson on Fisher square-root spheres",
                "adaptive Fisher-Rao policy trust regions",
                "adaptive Bures-Wasserstein four-source PNT barycenter",
                "normal-crisis-war-recovery tangent POD charts",
                "positive five-block product metric with bounded rank-one cross term",
            ],
            "state_update": "inner MFG fixed point and monthly same-path physical/economic state",
            "not_active": [
                "Kahler geometry", "semi-Riemannian optimization",
                "direct curvature adjustment of terminal probabilities",
                "global Levi-Civita connection without a downstream consumer",
            ],
        },
        "acceptance_thresholds": {
            "macro_outer": 0.003,
            "cge": 0.002,
            "dsge": 0.0008,
            "mfg": 0.0005,
            "ram_transport_tangency": 1.0e-10,
            "ram_sphere": 1.0e-10,
            "pnt_barycenter": 5.0e-7,
            "pnt_source_weight": 1.0e-12,
            "product_metric_minimum_eigenvalue": 0.0,
        },
        "ablation_contract": {
            "paths": 16,
            "months": 360,
            "trade_scenarios": ["baseline"],
            "intervention_cases": 5,
            "common_random_numbers": True,
            "one_at_a_time": [
                "riemannian_anderson", "riemannian_policy_trust_region",
                "adaptive_bures", "bures_barycenter",
                "regime_riemannian_pod", "product_manifold_metric",
                "all_v60_riemannian",
            ],
        },
        "files": {str(path.relative_to(ROOT)): digest(path) for path in files},
    }
    target = OUT / "taiwan_v60_frozen_specification.json"
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"TAIWAN_V60_FREEZE: PASS; sha256={digest(target)}")


if __name__ == "__main__":
    main()
