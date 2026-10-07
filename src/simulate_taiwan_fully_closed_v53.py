from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path

import simulate_taiwan_fully_closed_v47 as core
from dynamic_macro_dqsfc_v53 import CoupledMRRDCGEDQSFCEconomicSystem
from simulate_taiwan_fully_closed_v52 import MFG_CONFIG


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
TRACE = OUT / "taiwan_v53_traceability.json"
AUDIT = OUT / "taiwan_v53_static_coupling_audit.json"
OUT_CSV = OUT / "taiwan_v53_mr_rd_cge_dq_sfc_results.csv"
OUT_JSON = OUT / "taiwan_v53_mr_rd_cge_dq_sfc_simulation.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def run(
    paths: int = 128,
    months: int = 360,
    scenario_names: list[str] | None = None,
    case_names: list[str] | None = None,
):
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
        mfg_config=MFG_CONFIG,
        macro_factory=CoupledMRRDCGEDQSFCEconomicSystem,
        traceability_path=TRACE,
        coupling_audit_path=AUDIT,
    )


def write(rows, aggregate, diagnostics, verification) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    verification["result_sha256"] = sha256(OUT_CSV)
    verification["model_version"] = "V5.3"
    verification["economic_core"] = {
        "name": "MR-RD-CGE-DQ-SFC",
        "nodes": 140,
        "regional_accounts": 14,
        "sectors": 10,
        "disequilibrium_states": [
            "planned_demand", "actual_transactions", "inventory", "backlog",
            "contract_price", "unemployment", "vacancies",
        ],
        "capital": "old/new vintages plus three-stage construction queue",
        "short_term_sfc_clearing_accounts": [
            "households", "firms", "government", "banks", "external"
        ],
        "social_accounting_matrix": {
            "accounts": 19,
            "coverage": (
                "10 commodities, labor, capital factor, three household groups, "
                "enterprises, government, capital account, rest of world"
            ),
            "benchmark": "public IO control totals plus explicitly tagged structural priors",
            "dynamic_update": "monthly same-path realized transactions",
        },
        "feedback_lag": "one explicit month",
        "parameter_boundary": (
            "IO factor shares derived from public data; wartime adjustment "
            "speeds remain scenario priors"
        ),
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
            "V5.3 outcome simulation blocked until the complete-model gate passes"
        )
    paths = int(os.environ.get("TAIWAN_V53_PATHS", "128"))
    months = int(os.environ.get("TAIWAN_V53_MONTHS", "360"))
    rows, aggregate, diagnostics, verification = run(paths=paths, months=months)
    write(rows, aggregate, diagnostics, verification)
    print("TAIWAN_V53_COMPLETE_RUN: PASS")


if __name__ == "__main__":
    main()
