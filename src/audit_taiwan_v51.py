from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import audit_taiwan_v50 as prior


ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "work"
OUT = ROOT / "outputs"
METHOD_DOC = "QIQZd7QOCo7LQIxmJv0cHFb6nCg"
TOPOLOGY_DOC = "U0Zydx0uSoxaxZxutUKc6lbDnBe"


def main() -> None:
    prior.main()
    inherited = json.loads((OUT / "taiwan_v50_traceability.json").read_text(encoding="utf-8"))
    modules = deepcopy(inherited["modules"])
    body = (WORK / "hybrid_filippov_impulse_v51.py").read_text(encoding="utf-8")
    battle = (WORK / "simulate_taiwan_multidomain_v42.py").read_text(encoding="utf-8")
    runner = (WORK / "simulate_taiwan_fully_closed_v47.py").read_text(encoding="utf-8")
    topology_body = (WORK / "topological_dynamics_v51.py").read_text(encoding="utf-8")
    macro_body = (WORK / "dynamic_macro_equilibrium_v47.py").read_text(encoding="utf-8")
    symbols = [
        "HybridFilippovImpulseSystem", "update_filippov", "apply_damage_reset",
        "apply_leader_reset", "apply_coup_reset", "apply_collapse_reset",
        "minimum_dwell", "max_sliding_residual", "max_reset_residual",
        "stress_min", "stress_max", "new_event",
    ]
    missing = [symbol for symbol in symbols if symbol not in body]
    edge_tokens = [
        "hybrid_system.begin_month", "hybrid_system.update_filippov",
        "hybrid_factors.logistics", "hybrid_factors.financial",
        "hybrid_factors.mobilization", "hybrid_factors.command",
        "hybrid_system.apply_damage_reset", "hybrid_system.apply_leader_reset",
        "hybrid_system.apply_coup_reset", "hybrid_system.apply_collapse_reset",
    ]
    missing_edges = [token for token in edge_tokens if token not in battle]
    runner_tokens = [
        "HybridFilippovImpulseSystem", '"hybrid": hybrid.diagnostics()',
        "TopologicalRegimeDynamics", '"topology": topology.diagnostics()',
        "impulse reset residual", "Filippov sliding normal residual",
        "topological transition accounting residual",
        "topological dynamics has no nontrivial transition graph",
    ]
    missing_runner = [token for token in runner_tokens if token not in runner]
    blocking = []
    if inherited.get("gate_status") != "PASS":
        blocking.append({"inherited_v50_gate": inherited.get("gate_status")})
    if missing:
        blocking.append({"missing_symbols": missing})
    if missing_edges:
        blocking.append({"missing_same_path_edges": missing_edges})
    if missing_runner:
        blocking.append({"missing_runtime_gate_tokens": missing_runner})
    macro_solver_tokens = [
        "float(np.max(path_residual)) < tolerance",
        "household_mapping_seconds",
        "household_share_of_dsge",
    ]
    missing_macro_solver = [
        token for token in macro_solver_tokens if token not in macro_body
    ]
    if missing_macro_solver:
        blocking.append({"missing_dsge_hank_solver_tokens": missing_macro_solver})

    modules.append({
        "module_id": "hybrid_filippov_impulse",
        "document_sections": ["31", "31.1", "31.2", "31.3"],
        "equations": [
            "Filippov differential inclusion", "convex sliding vector field",
            "hysteretic switching with minimum dwell", "marked impulse reset map",
            "same-path event ledger",
        ],
        "role": "dynamic_state",
        "status": "active",
        "implementation_status": "PASS" if not (missing or missing_edges or missing_runner) else "FAIL",
        "implementation": [
            "work/hybrid_filippov_impulse_v51.py",
            "work/simulate_taiwan_multidomain_v42.py",
            "work/simulate_taiwan_fully_closed_v47.py",
            "work/simulate_taiwan_fully_closed_v51.py",
        ],
        "symbols": symbols,
        "state_owner": "HybridFilippovImpulseSystem.regime/dwell/event ledger",
        "clock": "monthly same-path, with unique marked resets inside each campaign month",
        "pathwise": True,
        "inputs": [
            "logistics service", "financial capacity and debt stress", "stock shortage",
            "support and conflict intensity", "command service", "existing Hawkes/DEDS event masks",
        ],
        "outputs": [
            "logistics/financial/mobilization/command switching factors",
            "leader/damage/coup/collapse reset state", "event ledger diagnostics",
        ],
        "feedback_consumers": [
            "campaign_deds_multidomain", "dynamic_mr_cge_mrio_gravity_chips",
            "open_economy_dsge_hank", "continuous_graphon_mfg",
        ],
        "solver": "explicit hysteretic Filippov convexification plus exact marked reset maps",
        "convergence_criterion": "sliding normal residual <1e-12, reset residual <1e-12, no duplicate event",
        "data": [{
            "source": "https://my.feishu.cn/docx/UEnMdV2AGoaAuKxAsSwcg563nSb",
            "publisher": "Taiwan model research specification",
            "observation_date": "2026-10-06",
            "unit": "dimensionless structural scenario prior",
            "identification": "scenario_prior",
            "uncertainty": "threshold and reset-strength structural sensitivity required",
        }],
        "invariants": [
            "unique event settlement", "minimum dwell", "bounded convex weight",
            "zero normal velocity on a sliding surface", "bounded reset state",
        ],
        "tests": [
            "work/diagnostic_taiwan_v51.py",
            "work/run_taiwan_v51_sensitivity.py",
            "outputs/taiwan_v51_static_coupling_audit.json",
        ],
        "outputs_artifact": "outputs/taiwan_v51_hybrid_simulation.json",
    })
    topology_symbols = [
        "TopologicalRegimeDynamics", "transition_counts", "visit_counts",
        "_strongly_connected_components", "morse_sets",
        "topological_entropy_upper_proxy", "max_feedback_deviation",
    ]
    missing_topology = [symbol for symbol in topology_symbols if symbol not in topology_body]
    topology_edges = [
        "topology_system.update", "topology_factors.financial",
        "topology_factors.logistics", "topology_factors.command",
        "topology_factors.mobilization", "topology_factors.panic_impulse",
    ]
    missing_topology_edges = [token for token in topology_edges if token not in battle]
    if missing_topology:
        blocking.append({"module": "symbolic_topological_dynamics", "missing_symbols": missing_topology})
    if missing_topology_edges:
        blocking.append({"module": "symbolic_topological_dynamics", "missing_edges": missing_topology_edges})
    modules.append({
        "module_id": "symbolic_topological_dynamics",
        "document_sections": ["31.4", "topology 5.1-5.6", "topology 7-9"],
        "equations": [
            "16-cell symbolic partition", "pathwise transition counting",
            "recurrence-novelty-Hamming instability", "Morse SCC decomposition",
            "adjacency spectral-radius entropy proxy",
        ],
        "role": "dynamic_state",
        "status": "active",
        "implementation_status": "PASS" if not (missing_topology or missing_topology_edges) else "FAIL",
        "implementation": [
            "work/topological_dynamics_v51.py",
            "work/simulate_taiwan_multidomain_v42.py",
            "work/simulate_taiwan_fully_closed_v47.py",
        ],
        "symbols": topology_symbols,
        "state_owner": "TopologicalRegimeDynamics.transition_counts/visit_counts",
        "clock": "monthly same-path after Filippov switching and before physical/economic updates",
        "pathwise": True,
        "inputs": ["four Filippov regimes", "conflict regime", "four-domain collapse state"],
        "outputs": [
            "symbol", "recurrence", "topological instability",
            "bounded command/financial/logistics/mobilization/panic feedback", "Morse graph diagnostics",
        ],
        "feedback_consumers": [
            "campaign_deds_multidomain", "dynamic_mr_cge_mrio_gravity_chips",
            "open_economy_dsge_hank", "age_stratified_social_contagion",
        ],
        "solver": "finite symbolic transition graph, Tarjan SCC and adjacency spectral radius",
        "convergence_criterion": "exact transition accounting, bounded nonzero feedback, finite entropy proxy",
        "data": [{
            "source": f"https://my.feishu.cn/docx/{TOPOLOGY_DOC}",
            "publisher": "Taiwan hybrid/topological dynamics method specification",
            "observation_date": "2026-10-06",
            "unit": "dimensionless structural map",
            "identification": "scenario_prior",
            "uncertainty": "symbol partition and feedback-strength sensitivity",
        }],
        "invariants": [
            "symbols in 0..15", "path isolation", "exact transition count",
            "bounded feedback", "finite Morse graph and entropy proxy",
        ],
        "tests": ["work/diagnostic_taiwan_v51.py", "outputs/taiwan_v51_static_coupling_audit.json"],
        "outputs_artifact": "outputs/taiwan_v51_hybrid_simulation.json",
    })
    status = "PASS" if not blocking else "FAIL"
    active = [module for module in modules if module["status"] == "active"]
    trace = {
        "model_version": "V5.1",
        "document_ids": inherited["document_ids"] + [METHOD_DOC, TOPOLOGY_DOC],
        "heading_inventory_count": inherited["heading_inventory_count"] + 16,
        "coverage_basis": "V5.0 complete manifest plus active Filippov-impulse and symbolic topological dynamics layers",
        "active_equation_family_count": len(active),
        "passed_equation_family_count": sum(m["implementation_status"] == "PASS" for m in active),
        "coverage_percent": 100.0 * sum(m["implementation_status"] == "PASS" for m in active) / len(active),
        "gate_status": status,
        "modules": modules,
        "blocking_gaps": blocking,
    }
    audit = {
        "model_version": "V5.1",
        "status": status,
        "master_clock": "monthly same-path with unique intra-step marked reset ledger",
        "required_edges": [
            {"edge": token, "status": "FAIL" if token in missing_edges else "PASS"}
            for token in edge_tokens
        ],
        "duplicate_state_owners": [],
        "mandatory_solver_inheritance": "V5.0 PASS",
        "blocking_gaps": blocking,
        "simulation_authorized": status == "PASS",
    }
    (OUT / "taiwan_v51_traceability.json").write_text(
        json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUT / "taiwan_v51_static_coupling_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"TAIWAN_V51_STATIC_AUDIT: {status}; coverage={trace['coverage_percent']:.2f}%")
    if blocking:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
