from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import audit_taiwan_v49 as prior


ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "work"
OUT = ROOT / "outputs"
GEOMETRY_DOC = "E3SDdy2reoTeVjxiiZhcszw8nve"


def source(name: str) -> str:
    return (WORK / name).read_text(encoding="utf-8")


def require_tokens(body: str, tokens: list[str]) -> list[str]:
    return [token for token in tokens if token not in body]


def geometry_data() -> list[dict]:
    return [{
        "source": f"https://my.feishu.cn/docx/{GEOMETRY_DOC}",
        "publisher": "Taiwan model geometry-method specification",
        "observation_date": "2026-10-06",
        "unit": "dimensionless structural map; public-data inputs inherited from owning module",
        "identification": "scenario_prior",
    }]


def main() -> None:
    OUT.mkdir(exist_ok=True)
    prior.main()
    inherited = json.loads(
        (OUT / "taiwan_v49_traceability.json").read_text(encoding="utf-8")
    )
    modules = deepcopy(inherited["modules"])
    allowed_identification = {"observed", "derived", "estimated", "scenario_prior"}
    for module in modules:
        for datum in module.get("data", []):
            if datum.get("identification") not in allowed_identification:
                datum["identification_note"] = datum.get("identification")
                datum["identification"] = "scenario_prior"
    by_id = {module["module_id"]: module for module in modules}
    blocking: list[dict] = []

    upgrades = {
        "continuous_graphon_mfg": {
            "sections": ["geometry 5.2", "geometry 11", "geometry 12"],
            "equations": ["Fisher-Rao geodesic density interpolation"],
            "implementation": ["work/continuous_multipop_mfg_v47.py"],
            "symbols": ["_fisher_rao_mix", "fisher_rao_distance", "geometry_enabled"],
            "invariants": ["Fisher-Rao simplex mass residual below 1e-10"],
        },
        "unbalanced_optimal_transport": {
            "sections": ["geometry 6", "geometry 16"],
            "equations": ["directed Finsler cost", "KL-relaxed echelon Sinkhorn"],
            "implementation": ["work/operational_constraints_v47.py"],
            "symbols": ["_unbalanced_transport", "finsler_cost", "front_priority"],
            "invariants": ["unbalanced Sinkhorn residual below 1e-5"],
        },
        "operations_network_flow_metric_search": {
            "sections": ["geometry 3", "geometry 16", "geometry 17"],
            "equations": ["sub-Riemannian echelon adjacency", "Finsler reachability"],
            "implementation": ["work/operational_constraints_v47.py"],
            "symbols": ["finsler_reachability", "corridor_adjacency"],
            "invariants": ["directed reachability remains in (0,1]"],
        },
        "geospatial_pnt": {
            "sections": ["geometry 5.3", "geometry 12"],
            "equations": ["affine-invariant SPD covariance geodesic"],
            "implementation": ["work/operational_constraints_v47.py"],
            "symbols": ["_affine_invariant_geodesic", "minimum_covariance_eigenvalue"],
            "invariants": ["covariance remains symmetric positive definite"],
        },
        "spatial_population_control": {
            "sections": ["geometry 3", "geometry 16", "geometry 17"],
            "equations": ["directed graph Finsler propagation", "spectral connectivity"],
            "implementation": ["work/spatial_control_network_v47.py"],
            "symbols": ["finsler_transition", "graph_spectral_gap", "attacker_pnt"],
            "invariants": ["positive graph spectral gap", "bounded reachability"],
        },
        "four_party_energy_maxflow_mincut": {
            "sections": ["geometry 9", "geometry 13", "geometry 15", "geometry 17"],
            "equations": ["port-Hamiltonian midpoint balance", "Dirac antisymmetry", "Hodge flow decomposition"],
            "implementation": ["work/four_party_energy_network_v49.py"],
            "symbols": ["_hodge_metrics", "max_port_hamiltonian_residual", "graph_resilience"],
            "invariants": ["Hodge divergence, Dirac and storage residuals below tolerance"],
        },
    }
    source_cache = {
        name: source(name) for name in (
            "continuous_multipop_mfg_v47.py", "operational_constraints_v47.py",
            "spatial_control_network_v47.py", "four_party_energy_network_v49.py",
        )
    }
    for module_id, upgrade in upgrades.items():
        module = by_id[module_id]
        module["document_sections"].extend(upgrade["sections"])
        module["equations"].extend(upgrade["equations"])
        for item in upgrade["implementation"]:
            if item not in module["implementation"]:
                module["implementation"].append(item)
        module["symbols"].extend(upgrade["symbols"])
        module["invariants"].extend(upgrade["invariants"])
        module["tests"] = ["work/diagnostic_taiwan_v50.py", "outputs/taiwan_v50_static_coupling_audit.json"]
        module["outputs_artifact"] = "outputs/taiwan_v50_geometry_simulation.json"
        module["data"].extend(geometry_data())
        body = "\n".join(source_cache[Path(item).name] for item in upgrade["implementation"])
        missing = require_tokens(body, upgrade["symbols"])
        if missing:
            module["implementation_status"] = "FAIL"
            blocking.append({"module": module_id, "missing_symbols": missing})

    multiscale_symbols = [
        "GeometricMultiscaleCoupler", "slow_state", "combat_factor",
        "macro_logistics_factor", "aggregation_residual",
    ]
    multiscale_source = source("geometric_multiscale_v50.py")
    missing = require_tokens(multiscale_source, multiscale_symbols)
    if missing:
        blocking.append({"module": "geometric_multiscale_coupling", "missing_symbols": missing})
    modules.append({
        "module_id": "geometric_multiscale_coupling",
        "document_sections": ["geometry 19", "geometry 23", "geometry 26"],
        "equations": ["fast-slow manifold decomposition", "geometric-mean reconstruction", "cross-scale consistency penalty"],
        "role": "constraint_solver",
        "status": "active",
        "implementation_status": "PASS" if not missing else "FAIL",
        "implementation": ["work/geometric_multiscale_v50.py", "work/simulate_taiwan_multidomain_v42.py"],
        "symbols": multiscale_symbols,
        "state_owner": "GeometricMultiscaleCoupler.slow_state",
        "clock": "monthly same-path after opening states and before combat/macro closing updates",
        "pathwise": True,
        "inputs": ["front/transit/depth integrity", "sector output", "financial capacity", "command", "energy logistics"],
        "outputs": ["combat consistency factor", "macro logistics factor", "fast/slow components"],
        "feedback_consumers": ["campaign_deds_multidomain", "dynamic_mr_cge_mrio_gravity_chips"],
        "solver": "explicit slow-manifold update and exact multiplicative reconstruction",
        "convergence_criterion": "aggregation residual below 1e-12",
        "data": geometry_data(),
        "invariants": ["same-path reconstruction", "bounded positive consistency factors"],
        "tests": ["work/diagnostic_taiwan_v50.py", "outputs/taiwan_v50_static_coupling_audit.json"],
        "outputs_artifact": "outputs/taiwan_v50_geometry_simulation.json",
    })

    theory = [
        ("lie_group_and_symplectic_fast_mechanics", ["geometry 4", "geometry 7"],
         "No aggregate state has an identified rigid-body or conservative phase-space contract; local six-DOF models remain outside the active macro solver."),
        ("contact_and_metriplectic_irreversibility", ["geometry 10"],
         "Dissipation is currently represented by explicit stock losses and port-Hamiltonian R terms; no independently identified contact variable is available."),
        ("tda_persistent_homology_candidate", ["geometry 18", "geometry 26.1"],
         "Conditional candidate only; activation requires rolling out-of-sample and permutation-test improvement."),
        ("kahler_geometry_not_adopted", ["geometry 20.1"],
         "No natural complex structure or testable Kähler constraint exists in the current state space."),
        ("category_theory_interface_language", ["geometry 20.2"],
         "Used only as typed-interface audit language and does not generate dynamics or probabilities."),
    ]
    for module_id, sections, rationale in theory:
        modules.append({
            "module_id": module_id,
            "document_sections": sections,
            "equations": [],
            "role": "theory_only",
            "status": "theory_only",
            "implementation_status": "PASS",
            "implementation": ["outputs/台海模型中的微分几何_辛几何_黎曼几何与哈密顿系统_飞书LaTeX.md"],
            "symbols": [],
            "state_owner": "none",
            "clock": "none",
            "pathwise": False,
            "inputs": [],
            "outputs": [],
            "feedback_consumers": [],
            "solver": "none",
            "convergence_criterion": rationale,
            "data": geometry_data(),
            "invariants": ["excluded from active probability path"],
            "tests": ["outputs/taiwan_v50_traceability.json"],
            "outputs_artifact": "outputs/taiwan_v50_traceability.json",
        })

    battle = source("simulate_taiwan_multidomain_v42.py")
    runner = source("simulate_taiwan_fully_closed_v47.py")
    edges = {
        "all geometry constructors enabled only on V5.0": ["geometry_enabled=geometry_enabled"],
        "multiscale to campaign tempo": ["command_eff * geometry_factors.combat_factor"],
        "multiscale to macro logistics": ["geometry_factors.macro_logistics_factor"],
        "PNT and blockade to spatial Finsler reachability": ["attacker_pnt=s[\"pnt_cn\"]", "blockade=blockade"],
        "geometry diagnostics to runtime gate": ["Fisher-Rao simplex mass residual", "Hodge divergence residual", "multiscale aggregation residual"],
    }
    edge_rows = []
    for label, tokens in edges.items():
        body = runner if "constructors" in label or "runtime gate" in label else battle
        missing_edge = require_tokens(body, tokens)
        edge_rows.append({"edge": label, "status": "PASS" if not missing_edge else "FAIL"})
        if missing_edge:
            blocking.append({"edge": label, "missing_tokens": missing_edge})

    active = [module for module in modules if module["status"] == "active"]
    passed = sum(module["implementation_status"] == "PASS" for module in active)
    coverage = 100.0 * passed / len(active)
    if inherited.get("gate_status") != "PASS":
        blocking.append({"inherited_v49_gate": inherited.get("gate_status")})
    status = "PASS" if coverage == 100.0 and not blocking else "FAIL"
    trace = {
        "model_version": "V5.0",
        "document_ids": ["UEnMdV2AGoaAuKxAsSwcg563nSb", GEOMETRY_DOC],
        "heading_inventory_count": 262,
        "coverage_basis": "all V4.9 active equation families plus selected structure-preserving geometry; conditional and rejected geometries explicitly theory-only",
        "active_equation_family_count": len(active),
        "passed_equation_family_count": passed,
        "coverage_percent": coverage,
        "gate_status": status,
        "modules": modules,
        "blocking_gaps": blocking,
    }
    audit = {
        "model_version": "V5.0",
        "status": status,
        "master_clock": "monthly same-path geometry, campaign, energy, finance, CGE-DSGE and social fixed-point path",
        "required_edges": edge_rows,
        "duplicate_state_owners": [],
        "theory_only_exclusions": [module_id for module_id, _, _ in theory],
        "blocking_gaps": blocking,
        "simulation_authorized": status == "PASS",
    }
    (OUT / "taiwan_v50_traceability.json").write_text(
        json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUT / "taiwan_v50_static_coupling_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"TAIWAN_V50_STATIC_AUDIT: {status}; coverage={coverage:.2f}%")
    if blocking:
        print(json.dumps(blocking, ensure_ascii=False, indent=2))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
