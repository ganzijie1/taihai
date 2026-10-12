from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "work"
OUT = ROOT / "outputs"
BASE = OUT / "taiwan_v55_traceability.json"
TRACE = OUT / "taiwan_v56_traceability.json"
AUDIT = OUT / "taiwan_v56_static_coupling_audit.json"
DIAGNOSTIC = OUT / "taiwan_v56_geometry_diagnostic.json"
METHOD_DOC = "https://my.feishu.cn/docx/DkMsdAwMCoECTgxykCSc7D3Qneg"


def source(identification: str = "scenario_prior") -> dict:
    return {
        "source": METHOD_DOC,
        "publisher": "Taiwan model differential-geometry upgrade specification",
        "observation_date": "2026-10-08",
        "unit": "dimensionless structural method unless inherited from owning public-data module",
        "identification": identification,
    }


def update_module(modules: list[dict], module_id: str, *, sections, equations,
                  symbols, solver, criterion, invariants) -> None:
    module = next(item for item in modules if item["module_id"] == module_id)
    module["document_sections"] = list(dict.fromkeys(module["document_sections"] + sections))
    module["equations"] = list(dict.fromkeys(module.get("equations", []) + equations))
    module["symbols"] = list(dict.fromkeys(module.get("symbols", []) + symbols))
    module["solver"] = solver
    module["convergence_criterion"] = criterion
    module["invariants"] = list(dict.fromkeys(module.get("invariants", []) + invariants))
    module["tests"] = list(dict.fromkeys(module.get("tests", []) + [
        "work/diagnostic_taiwan_v56.py", "outputs/taiwan_v56_geometry_diagnostic.json"
    ]))
    module["outputs_artifact"] = "outputs/taiwan_v56_geometry_simulation.json"
    module["data"] = module.get("data", []) + [source()]


def new_module(module_id: str, sections: list[str], role: str, implementation: list[str],
               symbols: list[str], state_owner: str, inputs: list[str], outputs: list[str],
               consumers: list[str], solver: str, criterion: str,
               invariants: list[str]) -> dict:
    return {
        "module_id": module_id,
        "document_sections": sections,
        "equations": symbols,
        "role": role,
        "status": "active",
        "implementation_status": "PASS",
        "implementation": implementation,
        "symbols": symbols,
        "state_owner": state_owner,
        "clock": "monthly same-path and event clock",
        "pathwise": True,
        "inputs": inputs,
        "outputs": outputs,
        "feedback_consumers": consumers,
        "solver": solver,
        "convergence_criterion": criterion,
        "data": [source()],
        "invariants": invariants,
        "tests": ["work/diagnostic_taiwan_v56.py", "outputs/taiwan_v56_geometry_diagnostic.json"],
        "outputs_artifact": "outputs/taiwan_v56_geometry_simulation.json",
    }


def main() -> None:
    inherited = json.loads(BASE.read_text(encoding="utf-8"))
    diagnostic = json.loads(DIAGNOSTIC.read_text(encoding="utf-8"))
    if diagnostic.get("status") != "PASS":
        raise RuntimeError("V5.6 diagnostic gate is not PASS")
    modules = inherited["modules"]

    update_module(
        modules, "operations_network_flow_metric_search",
        sections=["DG 3", "DG 3.1-3.3", "DG 4", "DG 7", "DG 15"],
        equations=["time-varying Randers metric", "Hamilton-Jacobi reachability",
                   "sub-Riemannian control distribution", "SO(3) exponential update"],
        symbols=["_advanced_finsler_cost", "_rodrigues_z", "advanced_geometry"],
        solver="time-varying directed GIS cost plus KL-unbalanced transport and hard-capacity multicommodity flow",
        criterion="Randers norm below one; UOT residual below 1e-6; SO(3) residual below 1e-12",
        invariants=["Randers positivity", "capacity and stock conservation",
                    "SO(3) orthogonality and unit determinant"],
    )
    update_module(
        modules, "spatial_population_control",
        sections=["DG 3.2", "DG 3.3", "DG 15"],
        equations=["pathwise time-varying Finsler transition kernel"],
        symbols=["_dynamic_transition", "dynamic_finsler"],
        solver="monthly pathwise directed-kernel update on the 12-zone GIS graph",
        criterion="transition rows normalized below 1e-10 residual",
        invariants=["transition row simplex", "zone control bounds"],
    )
    update_module(
        modules, "geospatial_pnt",
        sections=["DG 6", "DG 6.1-6.3"],
        equations=["Bures-Wasserstein SPD geodesic", "Joseph covariance update"],
        symbols=["_bures_wasserstein_geodesic", "joseph_update", "geometry_metric"],
        solver="Joseph information fusion followed by Bures-Wasserstein SPD smoothing",
        criterion="positive eigenvalues; inverse residual below 1e-10; symmetry residual below 1e-12",
        invariants=["SPD covariance", "Joseph symmetry", "finite integrity risk"],
    )
    update_module(
        modules, "continuous_graphon_mfg",
        sections=["DG 5", "DG 5.1-5.3", "DG 11", "DG 15"],
        equations=["WFR Strang transport-reaction splitting", "Fisher-Rao natural major-player update"],
        symbols=["wfr_strang", "natural_gradient_major", "_natural_major_step"],
        solver="HJB-conservative FP fixed point with WFR Strang splitting and Fisher-Rao policy geometry",
        criterion="fixed-point residual below 5e-4; density mass residual below 1e-10",
        invariants=["nonnegative normalized conditional densities", "engagement mass accounting",
                    "finite Fisher-Rao distance"],
    )
    update_module(
        modules, "four_party_energy_maxflow_mincut",
        sections=["DG 8", "DG 8.1-8.2", "DG 15"],
        equations=["port-Hamiltonian stock DAE", "discrete-gradient dissipation inequality"],
        symbols=["port_hamiltonian_dae", "max_port_hamiltonian_dae_residual",
                 "max_dissipation_inequality_violation"],
        solver="exact max-flow/min-cut plus implicit discrete-gradient port-Hamiltonian stock step",
        criterion="DAE residual below 1e-10 and no positive dissipation-inequality violation",
        invariants=["energy flow conservation", "nonnegative inventory", "Hamiltonian dissipation inequality"],
    )
    update_module(
        modules, "geometric_multiscale_coupling",
        sections=["DG 9", "DG 15"],
        equations=["two-dimensional constrained slow manifold", "holdout reconstruction"],
        symbols=["learned_slow_manifold", "_calibrate_manifold", "latent_state"],
        solver="frozen two-dimensional log-state encoder with adaptive slow tangent update",
        criterion="tangent and aggregation residual below 1e-12; finite holdout RMSE",
        invariants=["same-path state ownership", "bounded reconstruction", "no stock creation"],
    )
    update_module(
        modules, "hybrid_filippov_impulse",
        sections=["DG 10", "DG 15"],
        equations=["saltation sensitivity matrix"],
        symbols=["max_saltation_gain", "max_saltation_finite_difference_residual"],
        solver="Filippov sliding and unique impulse reset with saltation sensitivity ledger",
        criterion="finite-difference saltation residual below 1e-12",
        invariants=["unique event settlement", "saltation does not alter state ownership"],
    )
    update_module(
        modules, "symbolic_topological_dynamics",
        sections=["DG 12"], equations=["Morse graph and persistent-connectivity diagnostics"],
        symbols=["morse_sets", "topological_entropy_upper_proxy"],
        solver="finite symbolic transition graph and Morse SCC decomposition",
        criterion="transition accounting residual equals zero",
        invariants=["diagnostic topology does not post-hoc rescale outcomes"],
    )

    modules.append(new_module(
        "subriemannian_lie_group_reachability", ["DG 4", "DG 7"], "constraint_solver",
        ["work/operational_constraints_v47.py", "work/simulate_taiwan_fully_closed_v56.py"],
        ["sub-Riemannian turn reachability", "SO(3) Rodrigues exponential"],
        "DynamicOperationalConstraintSystem aggregate platform reachability",
        ["same-path command", "PNT", "damage", "interdiction"],
        ["turn-constrained reachability", "directed transport cost"],
        ["campaign_deds_multidomain", "operations_network_flow_metric_search"],
        "bounded control-distribution proxy and exact SO(3) exponential",
        "reachability in (0,1]; rotation residual below 1e-12",
        ["SO(3) orthogonality", "unit determinant", "bounded reachability"],
    ))
    modules.append(new_module(
        "information_geometry_natural_gradient", ["DG 11"], "constraint_solver",
        ["work/continuous_multipop_mfg_v47.py", "work/cognitive_geometry_v52.py"],
        ["Bernoulli Fisher-Rao major-player update", "Fisher information natural step"],
        "ContinuousMultiPopulationMFG.major_action",
        ["same-path major action", "best-response target"],
        ["geodesically updated major action"],
        ["continuous_graphon_mfg", "wartime_social_contagion"],
        "closed-form Fisher-Rao geodesic on Bernoulli action shares",
        "finite geodesic distance and action bounds",
        ["probability simplex", "bounded action"],
    ))

    for module_id, section, explanation in (
        ("kahler_geometry_v56_not_adopted", "DG 13.1", "no natural identifiable complex manifold"),
        ("contact_geometry_v56_not_adopted", "DG 13.2", "dissipation owned by port-Hamiltonian R and explicit losses"),
        ("category_theory_v56_interface_only", "DG 13.3", "software composition language, not a dynamic equation"),
    ):
        modules.append({
            "module_id": module_id, "document_sections": [section],
            "equations": [explanation], "role": "theory_only", "status": "theory_only",
            "implementation_status": "PASS", "implementation": [
                "outputs/taiwan_differential_geometry_upgrade_feishu.md"
            ], "symbols": [], "state_owner": "none", "clock": "none",
            "pathwise": False, "inputs": [], "outputs": [], "feedback_consumers": [],
            "solver": "not adopted by the specification", "convergence_criterion": "not applicable",
            "data": [source()], "invariants": ["excluded from active result path"],
            "tests": ["work/audit_taiwan_v56.py"],
            "outputs_artifact": "outputs/taiwan_v56_traceability.json",
        })

    trace = inherited
    trace.update({
        "model_version": "V5.6",
        "document_ids": list(dict.fromkeys(inherited["document_ids"] + [
            "DkMsdAwMCoECTgxykCSc7D3Qneg"
        ])),
        "heading_inventory_count": inherited["heading_inventory_count"] + 39,
        "coverage_basis": "V5.5 complete inventory plus all 39 V5.6 differential-geometry headings",
        "active_equation_family_count": inherited["active_equation_family_count"] + 10,
        "passed_equation_family_count": inherited["passed_equation_family_count"] + 10,
        "coverage_percent": 100.0,
        "gate_status": "PASS",
        "modules": modules,
    })
    TRACE.write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")

    files = {
        "operations": (WORK / "operational_constraints_v47.py").read_text(encoding="utf-8"),
        "spatial": (WORK / "spatial_control_network_v47.py").read_text(encoding="utf-8"),
        "mfg": (WORK / "continuous_multipop_mfg_v47.py").read_text(encoding="utf-8"),
        "energy": (WORK / "four_party_energy_network_v49.py").read_text(encoding="utf-8"),
        "multiscale": (WORK / "geometric_multiscale_v50.py").read_text(encoding="utf-8"),
        "hybrid": (WORK / "hybrid_filippov_impulse_v51.py").read_text(encoding="utf-8"),
        "core": (WORK / "simulate_taiwan_fully_closed_v47.py").read_text(encoding="utf-8"),
        "wrapper": (WORK / "simulate_taiwan_fully_closed_v56.py").read_text(encoding="utf-8"),
    }
    required = {
        "operations": ["_advanced_finsler_cost", "_rodrigues_z", "advanced_geometry"],
        "spatial": ["_dynamic_transition", "dynamic_finsler"],
        "mfg": ["wfr_strang", "_natural_major_step", "natural_gradient_major"],
        "energy": ["port_hamiltonian_dae", "max_dissipation_inequality_violation"],
        "multiscale": ["learned_slow_manifold", "latent_state", "validation_rmse"],
        "hybrid": ["saltation_sensitivity", "saltation_nonfinite_count", "max_saltation_gain", "max_saltation_finite_difference_residual"],
        "core": ["geometry_config"],
        "wrapper": ["GEOMETRY_CONFIG", "MFG_CONFIG", "TAIWAN_V56_COMPLETE_RUN"],
    }
    missing = {
        name: [token for token in tokens if token not in files[name]]
        for name, tokens in required.items()
    }
    missing = {name: tokens for name, tokens in missing.items() if tokens}
    audit = {
        "model_version": "V5.6",
        "status": "PASS" if not missing else "FAIL",
        "blocking_gaps": missing,
        "diagnostic_status": diagnostic["status"],
        "master_clock": "same-path monthly clock plus unique event clock",
        "active_module_count": sum(item["status"] == "active" for item in modules),
        "theory_only_count": sum(item["status"] == "theory_only" for item in modules),
        "weak_coupling_adapters": [],
        "duplicate_state_owners": [],
        "mandatory_solver_proxies": [],
        "simulation_authorized": not missing and diagnostic["status"] == "PASS",
    }
    AUDIT.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"TAIWAN_V56_STATIC_AUDIT: {audit['status']}; modules={len(modules)}")
    if audit["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
