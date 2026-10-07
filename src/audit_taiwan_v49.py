from __future__ import annotations

import json
from pathlib import Path

import audit_taiwan_v48 as prior


ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "work"
OUT = ROOT / "outputs"

REQUIRED_EDGES = [
    ("campaign damage/trade/finance->energy", ["energy_system.update", "actor_damage=actor_damage", "trade_loss=trade_volume_loss", "financial_capacity=s[\"actor_financial_capacity\"]"]),
    ("energy->military production/logistics", ["production_cn *= energy_opening.production", "energy_opening.logistics"]),
    ("energy->firm production", ["production_function[:, :, 0] *= energy_opening.production", "production_function[:, :, 1:] *= energy_opening.civilian"]),
    ("energy->shortage/inflation", ["energy_opening.shortage", "energy_opening.inflation_impulse"]),
    ("energy->CGE-DSGE logistics", ["* (energy_opening.logistics if energy_opening is not None else 1.0)"]),
    ("energy diagnostics->runtime gate", ["energy.diagnostics", "max_flow_min_cut_residual", "energy delivery feasibility residual"]),
]


def source(name: str) -> str:
    return (WORK / name).read_text(encoding="utf-8")


def main() -> None:
    OUT.mkdir(exist_ok=True)
    prior.main()
    old = json.loads((OUT / "taiwan_v48_traceability.json").read_text(encoding="utf-8"))
    modules = [m for m in old["modules"] if m.get("status") == "active"]
    energy_source = source("four_party_energy_network_v49.py")
    battle_source = source("simulate_taiwan_multidomain_v42.py")
    runner_source = source("simulate_taiwan_fully_closed_v47.py")
    required_symbols = [
        "FourPartyEnergyNetwork", "_minimum_cut", "cut_masks", "inventory",
        "infrastructure", "bottleneck_cut_frequency", "SOURCE_URLS",
    ]
    missing_symbols = [symbol for symbol in required_symbols if symbol not in energy_source]
    blocking = []
    if missing_symbols:
        blocking.append({"module": "four_party_energy_maxflow_mincut", "missing_symbols": missing_symbols})
    module = {
        "module_id": "four_party_energy_maxflow_mincut",
        "document_sections": ["30.1-30.8"],
        "equations": ["multicarrier maximum flow", "minimum cut dual", "inventory balance", "damage-repair recursion", "energy-war feedback"],
        "role": "constraint_solver",
        "status": "active",
        "implementation_status": "PASS" if not missing_symbols else "FAIL",
        "implementation": ["work/four_party_energy_network_v49.py", "work/simulate_taiwan_multidomain_v42.py"],
        "symbols": required_symbols,
        "state_owner": "FourPartyEnergyNetwork inventory/infrastructure/cut state",
        "clock": "monthly; lagged energy availability consumed before current damage and repair update",
        "pathwise": True,
        "inputs": ["damage", "output", "mobilization", "finance", "trade loss", "blockade", "sanction relief", "alliance intervention"],
        "outputs": ["carrier availability", "production/logistics/civilian factors", "energy shortage", "inflation impulse", "minimum cut and bottleneck"],
        "feedback_consumers": ["campaign_deds_multidomain", "labor_firms_supply_collapse", "finance_fiscal_money_tax_debt_assets", "dynamic_mr_cge_mrio_gravity_chips", "open_dsge_hank_mundell_fleming"],
        "solver": "batched exact enumeration of all 16 cuts for each carrier and path, followed by capacity-feasible priority allocation",
        "convergence_criterion": "max-flow/min-cut and delivery feasibility residuals below 1e-9",
        "data": [{
            "source": "EIA country analyses and Taiwan MOEA energy statistics; URLs embedded in SOURCE_URLS",
            "publisher": "U.S. EIA and Taiwan energy authority",
            "observation_date": "accessed 2026-10-05",
            "unit": "normalized domestic/import capacity shares",
            "identification": "public structural anchor plus documented wartime scenario priors",
        }],
        "invariants": ["max flow equals minimum cut", "delivery does not exceed demand", "bounded inventory and infrastructure"],
        "tests": ["work/diagnostic_taiwan_v49.py", "outputs/taiwan_v49_static_coupling_audit.json"],
        "outputs_artifact": "outputs/taiwan_v49_energy_mechanism_simulation.json",
    }
    edge_rows = []
    for label, tokens in REQUIRED_EDGES:
        body = runner_source if "runtime gate" in label else battle_source
        missing = [token for token in tokens if token not in body]
        status = "PASS" if not missing else "FAIL"
        if missing:
            blocking.append({"edge": label, "missing_tokens": missing})
        edge_rows.append({"edge": label, "status": status})
    modules.append(module)
    passed = sum(m["implementation_status"] == "PASS" for m in modules)
    coverage = 100.0 * passed / len(modules)
    if old["gate_status"] != "PASS":
        blocking.append({"inherited_v48_gate": old["gate_status"]})
    status = "PASS" if coverage == 100.0 and not blocking else "FAIL"
    trace = {
        "model_version": "V4.9",
        "document_ids": ["UEnMdV2AGoaAuKxAsSwcg563nSb", "R7qkdxtxCoIgqQxs0lgcrSofnVM"],
        "coverage_basis": "all 31 V4.8 active equation families plus the four-party multicarrier energy network",
        "active_equation_family_count": len(modules),
        "passed_equation_family_count": passed,
        "coverage_percent": coverage,
        "gate_status": status,
        "modules": modules,
        "blocking_gaps": blocking,
    }
    audit = {
        "model_version": "V4.9",
        "status": status,
        "master_clock": "monthly same-path update with lagged energy service and current damage/repair/flow clearing",
        "required_edges": edge_rows,
        "blocking_gaps": blocking,
        "simulation_authorized": status == "PASS",
    }
    (OUT / "taiwan_v49_traceability.json").write_text(
        json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUT / "taiwan_v49_static_coupling_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"TAIWAN_V49_STATIC_AUDIT: {status}; coverage={coverage:.2f}%")
    if blocking:
        print(json.dumps(blocking, ensure_ascii=False, indent=2))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
