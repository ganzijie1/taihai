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
    trace_path = OUT / "taiwan_v57_traceability.json"
    audit_path = OUT / "taiwan_v57_static_coupling_audit.json"
    diagnostic_path = OUT / "taiwan_v57_social_regime_diagnostic.json"
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    diagnostic = json.loads(diagnostic_path.read_text(encoding="utf-8"))
    if not (
        trace.get("gate_status") == "PASS"
        and audit.get("status") == "PASS"
        and audit.get("simulation_authorized") is True
        and diagnostic.get("status") == "PASS"
    ):
        raise RuntimeError("V5.7 cannot be frozen before all gates pass")
    files = [
        WORK / "wartime_collective_action_regime_v57.py",
        WORK / "simulate_taiwan_multidomain_v42.py",
        WORK / "simulate_taiwan_fully_closed_v47.py",
        WORK / "simulate_taiwan_fully_closed_v57.py",
        WORK / "diagnostic_taiwan_v57.py",
        WORK / "audit_taiwan_v57.py",
        WORK / "ablate_taiwan_v57.py",
        ROOT / "scripts" / "audit_manifest.py",
        OUT / "wartime_collective_action_regime_transition_feishu.md",
        OUT / "taiwan_v57_social_source_ledger.json",
        OUT / "taiwan_v55_econometric_calibration.json",
        OUT / "taiwan_v55_econometric_draws.npz",
        trace_path,
        audit_path,
        diagnostic_path,
    ]
    payload = {
        "model_version": "V5.7",
        "status": "FROZEN",
        "frozen_at": "2026-10-09 Asia/Shanghai",
        "method_document": "https://my.feishu.cn/docx/E6mrdVdn7oyWZ8xogIqcRGLKn2d",
        "simulation_contract": {
            "paths": 128,
            "months": 360,
            "intervention_cases": 5,
            "trade_scenarios": 6,
            "master_clock": "monthly same-path plus unique event clock",
            "social_feedback_lag": "one explicit month",
            "draw_contract": "one frozen V5.5 econometric draw per master path, shared across cases/scenarios",
            "outcome_gate": "no V5.7 substantive result before trace, static audit, manifest audit, diagnostic and freeze all pass"
        },
        "ablation_contract": {
            "paths": 16,
            "months": 360,
            "trade_scenarios": ["baseline"],
            "intervention_cases": 5,
            "common_random_numbers": True,
            "one_at_a_time": [
                "common_knowledge_network", "organization_capital",
                "government_mpc", "loyalty_hmm", "competing_risks",
                "revolution_episode", "external_intervention_nationalism",
                "closed_feedback"
            ]
        },
        "acceptance_thresholds": {
            "macro_outer": 0.003,
            "cge": 0.002,
            "dsge": 0.0008,
            "mfg": 0.0005,
            "loyalty_simplex": 1e-12,
            "event_probability_sum": 1.0,
            "labor_factor_min": 0.35,
            "logistics_factor_min": 0.45,
            "morale_factor_min": 0.35,
            "fiscal_drag_max": 0.12
            ,"revolution_resolution_codes": [0, 1, 2]
            ,"war_talk_complement_residual": 1e-12
        },
        "identification_contract": {
            "observed_or_estimated": [
                "inherited V5.5 econometric calibration",
                "public macro, trade, fiscal, financial and social-unrest validation anchors"
            ],
            "scenario_prior": [
                "government policy archetypes", "security loyalty transitions",
                "organization accumulation", "cause-specific wartime social hazards"
                ,"directed foreign-support kernels", "external-first nationalism response"
            ],
            "not_claimed": [
                "identified probability of revolution in a future Taiwan conflict",
                "classified security-force behavior", "operational targeting precision"
            ]
        },
        "validation_plan": [
            "probability and HMM simplex",
            "bounded states and feedback factors",
            "macro/CGE/DSGE/MFG convergence",
            "V5.6 comparison",
            "one-at-a-time social module ablations",
            "revolution-resolution exclusivity and war-talk complementarity",
            "deterministic seed and file hashes"
        ],
        "files": {str(path.relative_to(ROOT)): digest(path) for path in files}
    }
    target = OUT / "taiwan_v57_frozen_specification.json"
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"TAIWAN_V57_FREEZE: PASS; sha256={digest(target)}")


if __name__ == "__main__":
    main()
