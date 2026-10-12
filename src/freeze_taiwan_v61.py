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
    trace_path = OUT / "taiwan_v61_traceability.json"
    audit_path = OUT / "taiwan_v61_static_coupling_audit.json"
    diagnostic_path = OUT / "taiwan_v61_sectoral_diagnostic.json"
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    diagnostic = json.loads(diagnostic_path.read_text(encoding="utf-8"))
    if not (
        trace.get("gate_status") == "PASS"
        and audit.get("status") == "PASS"
        and audit.get("simulation_authorized") is True
        and diagnostic.get("status") == "PASS"
    ):
        raise RuntimeError("V6.1 cannot be frozen before all gates pass")
    files = [
        WORK / "sectoral_economy_v61.py",
        WORK / "simulate_taiwan_fully_closed_v61.py",
        WORK / "simulate_taiwan_v61_checkpointed.py",
        WORK / "diagnostic_taiwan_v61.py",
        WORK / "audit_taiwan_v61.py",
        WORK / "ablate_taiwan_v61.py",
        WORK / "validate_taiwan_v61.py",
        OUT / "农业工业第三产业统一建模框架_飞书LaTeX.md",
        OUT / "taiwan_v61_sectoral_source_ledger.json",
        OUT / "taiwan_v60_frozen_specification.json",
        OUT / "taiwan_v55_econometric_calibration.json",
        OUT / "taiwan_v55_econometric_draws.npz",
        trace_path,
        audit_path,
        diagnostic_path,
    ]
    payload = {
        "model_version": "V6.1",
        "status": "FROZEN",
        "frozen_at": "2026-10-10 Asia/Shanghai",
        "method_document": "https://my.feishu.cn/docx/IPmBdZpRVoQPv0xpoqucDan2nTb",
        "inherited_specification_sha256": digest(OUT / "taiwan_v60_frozen_specification.json"),
        "simulation_contract": {
            "paths": 128,
            "months": 360,
            "intervention_cases": 5,
            "trade_scenarios": 6,
            "master_clock": "monthly same-path plus inherited unique event clock",
            "draw_contract": "one frozen V5.5 econometric draw per path shared across cases and scenarios",
            "outcome_gate": "manifest, static audit, diagnostic, ablation and validation must pass",
        },
        "sectoral_contract": {
            "agriculture": ["weather-water-soil yield", "land allocation", "food security and DQ-SFC inventory"],
            "industry": ["input bottleneck", "equipment operability and repair", "firm entry-exit", "quality and learning"],
            "services": ["non-storable queue", "digital adoption", "public service capacity", "Baumol pressure"],
            "structural_transformation": ["Engel demand", "sector labor mobility", "Kaldor-Verdoorn learning"],
            "accounting": "140-node IO plan -> sector physical realization -> DQ-SFC transactions -> SAM and next-month CGE/DSGE",
            "no_posthoc_probability_adjustment": True
        },
        "acceptance_thresholds": {
            "macro_outer": 0.003,
            "cge": 0.002,
            "dsge": 0.0008,
            "mfg": 0.0005,
            "labor_and_land_simplex": 1.0e-12,
            "service_stock": 1.0e-12,
            "sector_actor_accounting": 1.0e-10
        },
        "ablation_contract": {
            "paths": 16,
            "months": 360,
            "trade_scenarios": ["baseline"],
            "intervention_cases": 5,
            "common_random_numbers": True,
            "one_at_a_time": [
                "agriculture", "industry", "services", "labor_reallocation",
                "structural_transformation", "all_v61_sectoral"
            ]
        },
        "files": {str(path.relative_to(ROOT)): digest(path) for path in files}
    }
    target = OUT / "taiwan_v61_frozen_specification.json"
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"TAIWAN_V61_FREEZE: PASS; sha256={digest(target)}")


if __name__ == "__main__":
    main()
