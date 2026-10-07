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
    trace_path = OUT / "taiwan_v53_traceability.json"
    audit_path = OUT / "taiwan_v53_static_coupling_audit.json"
    diagnostic_path = OUT / "taiwan_v53_diagnostic_only.json"
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    diagnostic = json.loads(diagnostic_path.read_text(encoding="utf-8"))
    if not (
        trace.get("gate_status") == "PASS"
        and audit.get("status") == "PASS"
        and diagnostic.get("status") == "PASS"
    ):
        raise RuntimeError("V5.3 specification cannot be frozen before all gates pass")

    files = [
        WORK / "dynamic_macro_equilibrium_v47.py",
        WORK / "dynamic_macro_dqsfc_v53.py",
        WORK / "social_accounting_matrix_v53.py",
        WORK / "simulate_taiwan_fully_closed_v47.py",
        WORK / "simulate_taiwan_fully_closed_v53.py",
        WORK / "simulate_four_party_financial_v45.py",
        OUT / "DRCCGE与非均衡CGE升级调研_飞书LaTeX.md",
        trace_path,
        audit_path,
        diagnostic_path,
        ROOT / "data" / "oecd_icio_2025" / "icio_2022_8x10.npz",
        ROOT / "data" / "china_mrio" / "dual_circulation_2022_140nodes.npz",
    ]
    payload = {
        "model_version": "V5.3",
        "status": "FROZEN",
        "frozen_at": "2026-10-07 Asia/Shanghai",
        "simulation_contract": {
            "paths": 128,
            "months": 360,
            "master_clock": "monthly same-path",
            "economic_feedback_lag": "one explicit month",
            "all_trade_scenarios": True,
            "all_intervention_cases": True,
            "base_seed_owner": "simulate_taiwan_multidomain_v42.SEED",
        },
        "identification_boundary": {
            "public_IO_accounts": "observed/derived",
            "sam_household_groups_and_institutional_allocation": "structural_prior_balanced_to_observed_control_totals",
            "wartime_price_search_cancellation_parameters": "scenario_prior",
            "no_classified_precision_claimed": True,
        },
        "files": {str(path.relative_to(ROOT)): digest(path) for path in files},
    }
    target = OUT / "taiwan_v53_frozen_specification.json"
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"TAIWAN_V53_FREEZE: PASS; sha256={digest(target)}")


if __name__ == "__main__":
    main()
