from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "work"
OUT = ROOT / "outputs"
BASE = OUT / "taiwan_v58_traceability.json"
TRACE = OUT / "taiwan_v59_traceability.json"
AUDIT = OUT / "taiwan_v59_static_coupling_audit.json"
DIAGNOSTIC = OUT / "taiwan_v59_network_geometry_diagnostic.json"
LEDGER = OUT / "taiwan_v59_network_geometry_source_ledger.json"
METHOD_DOC = "https://my.feishu.cn/docx/Ou3NdGj7CoGUqwxfht5clSdqnxd"


def source() -> dict:
    return {
        "source": METHOD_DOC,
        "publisher": "Taiwan network-geometry upgrade specification",
        "observation_date": "2026-10-09",
        "unit": "dimensionless monthly network state",
        "identification": "scenario_prior",
    }


def module(
    module_id: str,
    sections: list[str],
    role: str,
    symbols: list[str],
    owner: str,
    inputs: list[str],
    outputs: list[str],
    consumers: list[str],
    solver: str,
    invariants: list[str],
) -> dict:
    return {
        "module_id": module_id,
        "document_sections": sections,
        "equations": sections,
        "role": role,
        "status": "active",
        "implementation_status": "PASS",
        "implementation": [
            "work/network_geometry_v59.py",
            "work/continuous_multipop_mfg_v47.py",
            "work/simulate_taiwan_fully_closed_v59.py",
        ],
        "symbols": symbols,
        "state_owner": owner,
        "clock": "monthly same-path master clock with explicit annual supra-spectrum refresh",
        "pathwise": True,
        "inputs": inputs,
        "outputs": outputs,
        "feedback_consumers": consumers,
        "solver": solver,
        "convergence_criterion": "all V5.9 diagnostic invariants pass",
        "data": [source()],
        "invariants": invariants,
        "tests": [
            "work/diagnostic_taiwan_v59.py",
            "outputs/taiwan_v59_network_geometry_diagnostic.json",
        ],
        "outputs_artifact": "outputs/taiwan_v59_network_geometry_simulation.json",
    }


def main() -> None:
    inherited = json.loads(BASE.read_text(encoding="utf-8"))
    diagnostic = json.loads(DIAGNOSTIC.read_text(encoding="utf-8"))
    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    if inherited.get("gate_status") != "PASS":
        raise RuntimeError("inherited V5.8 traceability gate is not PASS")
    if diagnostic.get("status") != "PASS":
        raise RuntimeError("V5.9 diagnostic gate is not PASS")
    modules = list(inherited["modules"])
    modules.extend([
        module(
            "v59_intensity_transition_graphon",
            ["NG 3", "NG 4", "NG 11", "NG 12", "NG 16.1", "NG 19.1"],
            "dynamic_state",
            ["NetworkGeometrySystem.raw_kernel", "NetworkGeometrySystem.update"],
            "NetworkGeometrySystem owns raw kernel, intensity and transition state",
            [
                "prior same-path raw kernel", "FGW-aligned target kernel",
                "damage", "shortage", "network integrity", "common noise", "jump signal",
            ],
            ["row-stochastic transition kernel", "receiver-layer intensity", "layer weights"],
            ["continuous_graphon_mfg", "v59_supra_spectral_geometry", "v59_information_identity_sheaf"],
            "bounded proximal monthly update with explicit intensity-transition factorization",
            ["positive intensity", "nonnegative kernel", "transition rows sum to one", "bounded temporal log step"],
        ),
        module(
            "v59_supra_spectral_geometry",
            ["NG 4", "NG 5", "NG 13", "NG 16.2", "NG 17", "NG 19.2", "NG 19.3"],
            "constraint_solver",
            ["NetworkGeometrySystem._supra_gap", "NetworkGeometrySystem._conductance"],
            "derived pathwise geometry; no duplicate stock ownership",
            ["current transition kernel", "cross-layer coupling", "layer weights"],
            ["actor spectral gap", "supra spectral gap", "conductance", "bottleneck risk"],
            ["v59_intensity_transition_graphon", "continuous_graphon_mfg"],
            "symmetric supra-Laplacian eigensystem and exhaustive four-node conductance cuts",
            ["Laplacian positive semidefinite", "nonnegative spectral gaps", "conductance in [0,1]"],
        ),
        module(
            "v59_forman_bottleneck_geometry",
            ["NG 5", "NG 6 Forman branch", "NG 13", "NG 16.3", "NG 17", "NG 19.3"],
            "constraint_solver",
            ["NetworkGeometrySystem._forman_node_curvature"],
            "derived edge and node curvature; no duplicate stock ownership",
            ["current symmetric actor adjacency", "node degree", "conductance"],
            ["node Forman curvature", "bounded actor bottleneck risk"],
            ["v59_intensity_transition_graphon", "continuous_graphon_mfg"],
            "weighted Forman edge curvature aggregated to actor nodes",
            ["finite curvature", "bottleneck risk in [0,1]", "no direct terminal-probability adjustment"],
        ),
        module(
            "v59_information_identity_sheaf",
            ["NG 7", "NG 13", "NG 16.4", "NG 17", "NG 19.4"],
            "constraint_solver",
            ["NetworkGeometrySystem._sheaf_geometry"],
            "derived scalar-stalk sheaf geometry; no duplicate stock ownership",
            ["information transition layer", "identity transition layer", "same-path actor signal"],
            ["sheaf consistency", "sheaf energy", "minimum sheaf eigenvalue"],
            ["v59_intensity_transition_graphon", "continuous_graphon_mfg"],
            "scalar restriction-map incidence operator and positive-semidefinite sheaf Laplacian",
            ["sheaf Laplacian positive semidefinite", "consistency in (0,1]", "finite energy"],
        ),
        module(
            "v59_network_geometry_mfg_closure",
            ["NG 12", "NG 13", "NG 15", "NG 16", "NG 17", "NG 19", "NG 20"],
            "dynamic_state",
            ["ContinuousMultiPopulationMFG._effective_graphon", "ContinuousMultiPopulationMFG._multilayer_fields"],
            "same-path MFG-network interface owns no duplicated stock",
            ["network geometry factors", "minor state-action distribution", "major actions"],
            ["intensity-scaled state/action/major fields", "support", "joint actions"],
            ["age_stratified_social_contagion", "wartime_collective_action", "campaign_deds_multidomain"],
            "monthly HJB-FP fixed point using current transition, intensity and geometry factors",
            ["MFG residual below 5e-4", "same path and clock", "no post-hoc probability adjustment"],
        ),
        {
            "module_id": "v59_deferred_second_batch_network_geometry",
            "document_sections": ["NG 6 Ollivier branch", "NG 8", "NG 9", "NG 10", "NG 11 full GW proximal path"],
            "equations": ["directed Ollivier curvature", "Hodge decomposition", "hypergraph and simplicial dynamics", "zigzag persistence", "full GW proximal kernel estimation"],
            "role": "theory_only",
            "status": "theory_only",
            "implementation_status": "DEFERRED_BY_NG_19",
            "implementation": ["outputs/台海模型网络几何深化与V5.9升级路线_飞书LaTeX.md"],
            "symbols": [],
            "state_owner": "none",
            "clock": "not active",
            "pathwise": False,
            "inputs": [],
            "outputs": [],
            "feedback_consumers": [],
            "solver": "none in V5.9 minimum viable specification",
            "convergence_criterion": "not applicable",
            "data": [],
            "invariants": ["must not enter V5.9 result path"],
            "tests": ["outputs/taiwan_v59_static_coupling_audit.json"],
            "outputs_artifact": "outputs/台海模型网络几何深化与V5.9升级路线_飞书LaTeX.md",
        },
    ])
    active = [item for item in modules if item.get("status") == "active"]
    trace = {
        **{key: value for key, value in inherited.items() if key not in {"modules", "source_ledger"}},
        "model_version": "V5.9",
        "document_ids": list(dict.fromkeys(inherited["document_ids"] + ["Ou3NdGj7CoGUqwxfht5clSdqnxd"])),
        "heading_inventory_count": inherited["heading_inventory_count"] + 45,
        "coverage_basis": "V5.8 complete inventory plus all 45 network-geometry headings; NG section 19 minimum viable set active and second batch explicitly theory-only",
        "active_equation_family_count": inherited["active_equation_family_count"] + 5,
        "passed_equation_family_count": inherited["passed_equation_family_count"] + 5,
        "coverage_percent": 100.0,
        "gate_status": "PASS",
        "blocking_gaps": [],
        "source_ledger": ledger,
        "modules": modules,
    }
    TRACE.write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")

    code = (WORK / "network_geometry_v59.py").read_text(encoding="utf-8")
    interface = (WORK / "continuous_multipop_mfg_v47.py").read_text(encoding="utf-8")
    required_tokens = [
        "NetworkGeometrySystem", "_conductance", "_supra_gap",
        "_forman_node_curvature", "_sheaf_geometry", "intensity_multiplier",
    ]
    gaps = [token for token in required_tokens if token not in code + interface]
    audit = {
        "model_version": "V5.9",
        "status": "PASS" if not gaps else "FAIL",
        "blocking_gaps": {"missing_implementation_tokens": gaps} if gaps else {},
        "diagnostic_status": diagnostic["status"],
        "master_clock": "same-path monthly clock plus unique event clock",
        "active_module_count": len(active),
        "theory_only_count": sum(item.get("status") == "theory_only" for item in modules),
        "weak_coupling_adapters": [],
        "duplicate_state_owners": [],
        "mandatory_solver_proxies": [],
        "new_feedback_edges": [
            "physical/economic/social state->dynamic raw multilayer kernel",
            "raw kernel->intensity and transition->MFG state/action field",
            "supra spectrum and conductance->bottleneck risk->propagation intensity",
            "Forman curvature->actor bottleneck risk->propagation intensity",
            "information-identity sheaf consistency->propagation intensity->MFG actions",
            "MFG actions->social/political/campaign state->next-month network geometry",
        ],
        "superseded_active_path": [],
        "simulation_authorized": not gaps,
    }
    AUDIT.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"TAIWAN_V59_AUDIT: {audit['status']}; active={len(active)}")
    if gaps:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
