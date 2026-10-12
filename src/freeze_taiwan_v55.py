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
    trace_path = OUT / "taiwan_v55_traceability.json"
    audit_path = OUT / "taiwan_v55_static_coupling_audit.json"
    calibration_diag = OUT / "taiwan_v55_econometric_diagnostic.json"
    integration_diag = OUT / "taiwan_v55_econometric_integration_diagnostic.json"
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    integration = json.loads(integration_diag.read_text(encoding="utf-8"))
    if not (
        trace.get("gate_status") == "PASS"
        and audit.get("status") == "PASS"
        and audit.get("simulation_authorized") is True
        and integration.get("status") == "PASS"
    ):
        raise RuntimeError("V5.5 cannot be frozen before all gates pass")
    files = [
        WORK / "econometric_calibration_v55.py",
        WORK / "diagnostic_econometric_integration_v55.py",
        WORK / "closed_prewar_system_v47.py",
        WORK / "competing_risk_survival_v47.py",
        WORK / "endogenous_intervention_v47.py",
        WORK / "dynamic_macro_equilibrium_v47.py",
        WORK / "dynamic_macro_dqsfc_v53.py",
        WORK / "simulate_taiwan_fully_closed_v47.py",
        WORK / "simulate_taiwan_fully_closed_v55.py",
        WORK / "validate_taiwan_v55.py",
        WORK / "finalize_taiwan_v55_metadata.py",
        OUT / "台海模型计量经济学识别与校准框架_飞书LaTeX.md",
        OUT / "taiwan_v55_econometric_calibration.json",
        OUT / "taiwan_v55_econometric_draws.npz",
        OUT / "taiwan_v55_econometric_source_ledger.json",
        trace_path, audit_path, calibration_diag, integration_diag,
        ROOT / "data" / "econometrics_v55" / "world_bank_macro_1960_2025.json",
    ] + [
        ROOT / "data" / "econometrics_v55" / f"icio_{year}_8x10.npz"
        for year in range(2016, 2023)
    ]
    payload = {
        "model_version": "V5.5",
        "status": "FROZEN",
        "frozen_at": "2026-10-08 Asia/Shanghai",
        "simulation_contract": {
            "paths": 128, "months": 360, "intervention_cases": 5,
            "trade_scenarios": 6, "master_clock": "monthly same-path",
            "posterior_draw_contract": "one frozen vector per path, reused for all months/cases/scenarios",
            "outcome_gate": "no V5.5 substantive run before this freeze",
        },
        "identification_contract": {
            "estimated": ["PPML pair/year effects", "HMM transitions", "predictive macro VAR/LP"],
            "scenario_prior": ["Armington elasticities", "terminal hazard coefficients", "alliance-choice coefficients", "Taiwan trend growth uncertainty"],
            "not_claimed": ["causal SVAR", "causal wartime elasticities", "classified operational precision"],
        },
        "files": {str(path.relative_to(ROOT)): digest(path) for path in files},
    }
    target = OUT / "taiwan_v55_frozen_specification.json"
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"TAIWAN_V55_FREEZE: PASS; sha256={digest(target)}")


if __name__ == "__main__":
    main()
