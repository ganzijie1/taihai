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
    trace_path = OUT / "taiwan_v58_traceability.json"
    audit_path = OUT / "taiwan_v58_static_coupling_audit.json"
    diagnostic_path = OUT / "taiwan_v58_information_geometry_diagnostic.json"
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    diagnostic = json.loads(diagnostic_path.read_text(encoding="utf-8"))
    if not (
        trace.get("gate_status") == "PASS"
        and audit.get("status") == "PASS"
        and audit.get("simulation_authorized") is True
        and diagnostic.get("status") == "PASS"
    ):
        raise RuntimeError("V5.8 cannot be frozen before all gates pass")
    files = [
        WORK / "information_geometry_v58.py",
        WORK / "continuous_multipop_mfg_v47.py",
        WORK / "fp_solvers_v52.py",
        WORK / "simulate_taiwan_multidomain_v42.py",
        WORK / "simulate_taiwan_fully_closed_v47.py",
        WORK / "simulate_taiwan_fully_closed_v58.py",
        WORK / "wartime_collective_action_regime_v57.py",
        WORK / "diagnostic_taiwan_v58.py",
        WORK / "audit_taiwan_v58.py",
        WORK / "ablate_taiwan_v58.py",
        WORK / "validate_taiwan_v58.py",
        ROOT / "scripts" / "audit_manifest.py",
        OUT / "台海模型信息几何深化与升级路线_飞书LaTeX.md",
        OUT / "taiwan_v58_information_geometry_source_ledger.json",
        OUT / "taiwan_v55_econometric_calibration.json",
        OUT / "taiwan_v55_econometric_draws.npz",
        trace_path,
        audit_path,
        diagnostic_path,
    ]
    payload = {
        "model_version": "V5.8",
        "status": "FROZEN",
        "frozen_at": "2026-10-09 Asia/Shanghai",
        "method_document": "https://my.feishu.cn/docx/M6t9dmRhkoMhPrxJnV3ciYJXn2c",
        "simulation_contract": {
            "paths": 128,
            "months": 360,
            "intervention_cases": 5,
            "trade_scenarios": 6,
            "master_clock": "monthly same-path plus unique event clock",
            "information_action_lag": "one explicit month",
            "draw_contract": "one frozen V5.5 econometric draw per master path, shared across cases/scenarios",
            "outcome_gate": "no V5.8 substantive result before manifest, static audit, diagnostic and freeze pass",
        },
        "information_geometry_contract": {
            "belief_grid_points": 9,
            "actors": 4,
            "sources": 6,
            "credibility_states": 3,
            "fisher_trust_radius": 0.035,
            "minor_kl_trust_radius": 0.025,
            "base_ambiguity_radius": 0.025,
            "wfr_solver": "joint implicit finite-volume advection-diffusion-reaction",
            "bridge_solver": "entropic IPFP with current and prior marginals only",
            "fgw_solver": "frozen alternating fused feature-structure alignment",
        },
        "ablation_contract": {
            "paths": 16,
            "months": 360,
            "trade_scenarios": ["baseline"],
            "intervention_cases": 5,
            "common_random_numbers": True,
            "one_at_a_time": [
                "belief_filter", "credibility_hmm", "fisher_kl_trust_region",
                "joint_wfr", "schrodinger_bridge", "fgw_alignment",
                "divergence_robustness", "information_value_allocation",
                "all_v58_information_geometry",
            ],
        },
        "acceptance_thresholds": {
            "macro_outer": 0.003,
            "cge": 0.002,
            "dsge": 0.0008,
            "mfg": 0.0005,
            "posterior_simplex": 1e-12,
            "credibility_simplex": 1e-12,
            "schrodinger_marginal": 1e-7,
            "wfr_conditional_mass": 1e-12,
            "wfr_stock_accounting": 1e-12,
            "major_kl": 0.035,
            "minor_kl": 0.025,
            "information_budget": 1e-12,
        },
        "identification_contract": {
            "observed_or_estimated": [
                "inherited V5.5 econometric calibration",
                "public macro, trade, fiscal, financial and social validation anchors",
            ],
            "scenario_prior": [
                "wartime information-channel noise",
                "source-reliability transitions",
                "KL ambiguity radius",
                "cross-society FGW feature weights",
            ],
            "not_claimed": [
                "empirically identified future wartime misinformation",
                "classified information operations",
                "operational targeting or platform-specific intervention",
            ],
        },
        "validation_plan": [
            "probability and credibility simplex",
            "bridge and FGW marginal constraints",
            "joint WFR nonnegativity and stock accounting",
            "Fisher/KL trust-region compliance",
            "macro/CGE/DSGE/MFG convergence",
            "V5.7 comparison",
            "one-at-a-time and grouped information-geometry ablations",
            "deterministic seed and file hashes",
        ],
        "files": {str(path.relative_to(ROOT)): digest(path) for path in files},
    }
    target = OUT / "taiwan_v58_frozen_specification.json"
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"TAIWAN_V58_FREEZE: PASS; sha256={digest(target)}")


if __name__ == "__main__":
    main()
