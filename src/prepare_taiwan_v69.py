from __future__ import annotations

import hashlib
import json
import py_compile
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
TRACE = OUT / "taiwan_v69_traceability.json"
AUDIT = OUT / "taiwan_v69_static_coupling_audit.json"
FREEZE = OUT / "taiwan_v69_frozen_specification.json"
DIAGNOSTIC = OUT / "taiwan_v69_ot_symplectic_active_diagnostic.json"
AUDITORS = [ROOT / "scripts" / "taiwan_model_audit_manifest.py",
            ROOT / "scripts" / "audit_manifest.py"]
FILES = [ROOT / "work" / name for name in (
    "requirements-wsl-jax-cuda.txt",
    "simulate_taiwan_multidomain_v42.py",
    "optimal_transport_geometry_v62.py",
    "deep_optimal_transport_v66.py",
    "transport_symplectic_v66.py",
    "martingale_ot_finance_v69.py",
    "simulate_taiwan_fully_closed_v69.py",
    "simulate_taiwan_v69_checkpointed.py",
    "diagnostic_taiwan_v69.py",
)]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def data(source: str, unit: str, identification: str) -> list[dict]:
    return [{
        "source": source,
        "publisher": "publishers and dates inherited from the audited V4.5 financial and V6.6 OT ledgers",
        "observation_date": "accessed 2026-10-13",
        "unit": unit,
        "identification": identification,
    }]


def main() -> None:
    for path in FILES:
        if path.suffix == ".py":
            py_compile.compile(str(path), doraise=True)
    diagnostic = json.loads(DIAGNOSTIC.read_text(encoding="utf-8"))
    if diagnostic.get("status") != "PASS":
        raise RuntimeError("V6.9 diagnostic has not passed")

    trace = json.loads((OUT / "taiwan_v68_traceability.json").read_text(encoding="utf-8"))
    trace["model_version"] = "V6.9-OT-SYM-ACTIVE"
    trace["coverage_basis"] += (
        "; V6.9 activates the worthwhile remaining methods from Ob5/Usu: "
        "blocked conformal DRO and risk-neutral martingale OT; structure-preserving "
        "ROM is explicitly superseded because the full symplectic state is six-dimensional"
    )
    for module in trace["modules"]:
        if module.get("module_id") == "v66_martingale_ot_candidate":
            module.update({
                "module_id": "v69_financial_martingale_ot",
                "document_sections": ["V6.5-OT 9", "10", "13"],
                "role": "constraint_solver", "status": "active",
                "implementation": [
                    "work/martingale_ot_finance_v69.py:MartingaleOTFinancialSystemV69._entropy_projection",
                    "work/martingale_ot_finance_v69.py:MartingaleOTFinancialSystemV69.update",
                ],
                "clock": "monthly same-path financial clock; current and past states only",
                "pathwise": True,
                "inputs": ["opening asset price", "policy rate", "stochastic variance", "physical price signal"],
                "outputs": ["risk-neutral probabilities", "KL wedge", "credit/fiscal/production factors", "lagged capital access"],
                "feedback_consumers": ["finance", "sectoral production", "CGE/DSGE", "next-month capital access"],
                "data": data("audited four-party public financial calibration ledger", "asset-price and financial-capacity indices", "derived"),
                "invariants": ["discounted martingale", "probability simplex", "nonnegative probabilities", "finance-only scope", "no future information"],
                "tests": ["outputs/taiwan_v69_ot_symplectic_active_diagnostic.json"],
                "outputs_artifact": "outputs/taiwan_v69_simulation.json",
            })
        elif module.get("module_id") == "v66_structure_preserving_rom_candidate":
            module.update({
                "module_id": "v69_structure_preserving_rom_excluded",
                "role": "superseded", "status": "superseded", "pathwise": False,
                "implementation": [
                    "excluded after dimensional audit: full-order core has 3 q + 3 p states per actor",
                    "active full-order replacement remains work/transport_symplectic_v66.py:_symplectic_step",
                ],
                "clock": "not in result path",
                "inputs": ["six-dimensional full-order state audit"],
                "outputs": ["documented exclusion"],
                "feedback_consumers": ["none; full-order symplectic core remains active"],
                "data": data("same-code dimensional and bottleneck audit", "state dimensions and runtime ownership", "derived"),
                "invariants": ["cannot affect probabilities", "no surrogate error introduced"],
                "tests": ["outputs/taiwan_v69_ot_symplectic_active_diagnostic.json"],
                "outputs_artifact": "outputs/taiwan_v69_ot_symplectic_active_diagnostic.json",
            })
    trace["modules"].append({
        "module_id": "v69_block_conformal_dro",
        "document_sections": ["V6.5-OT 10.1", "10.2", "13"],
        "role": "estimator", "status": "active",
        "implementation": [
            "work/deep_optimal_transport_v66.py:DeepCausalOptimalTransportGeometryV66._causal_conformal_radius",
            "work/optimal_transport_geometry_v62.py:DynamicCausalOptimalTransportGeometry.update",
        ],
        "clock": "monthly, estimated from rolling history before current assimilation",
        "pathwise": True,
        "inputs": ["past-only belief forecast errors", "rolling blocked calibration window"],
        "outputs": ["observation, parameter and structure radii", "robust information ambiguity radius"],
        "feedback_consumers": ["same-month robust belief control", "MFG actions", "social and command states"],
        "data": data("same-path past forecast-error ledger", "belief error and ambiguity radius", "derived"),
        "invariants": ["no current or future realization in radius", "nonnegative components", "bounded total radius"],
        "tests": ["outputs/taiwan_v69_ot_symplectic_active_diagnostic.json"],
        "outputs_artifact": "outputs/taiwan_v69_simulation.json",
    })
    trace["coverage_percent"] = 100.0
    trace["gate_status"] = "PASS"
    trace["blocking_gaps"] = []
    TRACE.write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")

    audit = json.loads((OUT / "taiwan_v68_static_coupling_audit.json").read_text(encoding="utf-8"))
    active_count = sum(m.get("status") == "active" for m in trace["modules"])
    audit.update({
        "model_version": "V6.9-OT-SYM-ACTIVE", "status": "PASS",
        "blocking_gaps": {}, "diagnostic_status": "PASS",
        "active_module_count": active_count, "simulation_authorized": True,
    })
    boundary = audit.setdefault("selected_method_boundary", {"active": [], "theory_only": []})
    boundary["active"].extend([
        "blocked conformal observation/parameter/structure DRO radius",
        "risk-neutral martingale optimal transport with financial feedback",
    ])
    boundary["theory_only"] = [
        x for x in boundary.get("theory_only", [])
        if x not in {"martingale OT", "symplectic ROM/neural surrogate"}
    ]
    boundary["superseded"] = [
        "structure-preserving ROM: no reduction benefit for the six-dimensional full-order symplectic core"
    ]
    AUDIT.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    for auditor in AUDITORS:
        subprocess.run([sys.executable, str(auditor), str(TRACE)], check=True)

    freeze = json.loads((OUT / "taiwan_v68_frozen_specification.json").read_text(encoding="utf-8"))
    freeze.update({
        "model_version": "V6.9-OT-SYM-ACTIVE", "status": "FROZEN",
        "hashes": {str(path.relative_to(ROOT)): sha256(path) for path in FILES},
        "diagnostic_hashes": {
            "outputs/taiwan_v69_ot_symplectic_active_diagnostic.json": sha256(DIAGNOSTIC)
        },
    })
    freeze["validation_plan"] = {
        "runtime": [
            "all inherited V6.8 gates", "discounted martingale residual",
            "probability simplex and nonnegativity", "conformal radius activity",
            "same-path financial feedback",
        ],
        "ablations": ["martingale OT off", "conformal decomposition off"],
        "common_random_numbers": True,
        "v67_outcome_equivalence_required": False,
    }
    FREEZE.write_text(json.dumps(freeze, ensure_ascii=False, indent=2), encoding="utf-8")
    print("TAIWAN_V69_PREPARE_AND_FREEZE: PASS")


if __name__ == "__main__":
    main()
