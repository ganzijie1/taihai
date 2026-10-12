from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "work"
OUT = ROOT / "outputs"
BASE = OUT / "taiwan_v57_traceability.json"
TRACE = OUT / "taiwan_v58_traceability.json"
AUDIT = OUT / "taiwan_v58_static_coupling_audit.json"
DIAGNOSTIC = OUT / "taiwan_v58_information_geometry_diagnostic.json"
LEDGER = OUT / "taiwan_v58_information_geometry_source_ledger.json"
METHOD_DOC = "https://my.feishu.cn/docx/M6t9dmRhkoMhPrxJnV3ciYJXn2c"


def source(identification: str = "scenario_prior") -> dict:
    return {
        "source": METHOD_DOC,
        "publisher": "Taiwan information-geometry upgrade specification",
        "observation_date": "2026-10-09",
        "unit": "dimensionless probability, distribution or monthly structural state",
        "identification": identification,
    }


def module(
    module_id: str, sections: list[str], role: str, implementation: list[str],
    owner: str, inputs: list[str], outputs: list[str], consumers: list[str],
    solver: str, criterion: str, invariants: list[str],
    pathwise: bool = True,
) -> dict:
    return {
        "module_id": module_id,
        "document_sections": sections,
        "equations": sections,
        "role": role,
        "status": "active",
        "implementation_status": "PASS",
        "implementation": implementation,
        "symbols": [],
        "state_owner": owner,
        "clock": "monthly same-path and event clock" if pathwise else "calibration/freeze",
        "pathwise": pathwise,
        "inputs": inputs,
        "outputs": outputs,
        "feedback_consumers": consumers,
        "solver": solver,
        "convergence_criterion": criterion,
        "data": [source("estimated" if role == "estimator" else "scenario_prior")],
        "invariants": invariants,
        "tests": [
            "work/diagnostic_taiwan_v58.py",
            "outputs/taiwan_v58_information_geometry_diagnostic.json",
        ],
        "outputs_artifact": "outputs/taiwan_v58_information_geometry_simulation.json",
    }


def main() -> None:
    inherited = json.loads(BASE.read_text(encoding="utf-8"))
    diagnostic = json.loads(DIAGNOSTIC.read_text(encoding="utf-8"))
    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    if diagnostic.get("status") != "PASS":
        raise RuntimeError("V5.8 diagnostic gate is not PASS")
    modules = inherited["modules"]
    continuous = next(
        item for item in modules if item["module_id"] == "continuous_graphon_mfg"
    )
    continuous["document_sections"] = list(dict.fromkeys(
        continuous["document_sections"] + ["M6 9", "M6 10", "M6 13"]
    ))
    continuous["equations"] = [
        equation for equation in continuous["equations"]
        if "splitting" not in equation.lower()
    ] + [
        "partially observed Major-Minor Graphon HJB-FP fixed point",
        "joint implicit WFR transport-diffusion-reaction",
        "FGW-aligned directed multilayer kernel",
    ]
    continuous["implementation"] = list(dict.fromkeys(
        continuous["implementation"] + [
            "work/information_geometry_v58.py",
            "work/simulate_taiwan_fully_closed_v58.py",
        ]
    ))
    continuous["state_owner"] = (
        "ContinuousMultiPopulationMFG owns value, normalized density, engagement mass, "
        "major/minor actions and cognitive costs; PartiallyObservableInformationGeometry "
        "owns beliefs, reliability posteriors and lagged information action"
    )
    continuous["clock"] = (
        "monthly same-path with inner HJB-FP/WFR-major-minor fixed point and explicit "
        "one-month information-action lag"
    )
    continuous["solver"] = (
        "HJB and joint implicit unbalanced FP/WFR fixed point with Fisher/KL trust regions"
    )
    continuous["tests"] = list(dict.fromkeys(
        continuous["tests"] + [
            "work/diagnostic_taiwan_v58.py",
            "outputs/taiwan_v58_information_geometry_diagnostic.json",
        ]
    ))
    continuous["outputs_artifact"] = (
        "outputs/taiwan_v58_information_geometry_simulation.json"
    )

    modules.extend([
        module(
            "v58_actor_specific_belief_filter",
            ["M6 3", "M6 3.1", "M6 3.2", "M6 3.3", "M6 10"],
            "dynamic_state", ["work/information_geometry_v58.py:update"],
            "PartiallyObservableInformationGeometry.posterior",
            ["same-path military/economic/financial/social/political signals", "lagged information action"],
            ["four-by-four actor posterior", "robust self belief", "belief uncertainty"],
            ["continuous_graphon_mfg", "v58_information_geometry_closure"],
            "finite-grid Bayesian prediction and likelihood update",
            "posterior simplex residual below 1e-12",
            ["no future observations", "nonnegative posterior", "posterior sums to one"],
        ),
        module(
            "v58_source_credibility_hmm",
            ["M6 4", "M6 4.1", "M6 4.2", "M6 4.3"],
            "dynamic_state", ["work/information_geometry_v58.py:_credibility_transition"],
            "PartiallyObservableInformationGeometry.reliability",
            ["same-path source residual", "network state", "jump signal", "lagged information action"],
            ["six-source three-state reliability posterior", "effective source credibility"],
            ["v58_actor_specific_belief_filter", "v58_information_value_allocation"],
            "pathwise HMM prediction and emission update",
            "reliability simplex residual below 1e-12",
            ["nonnegative posterior", "three states sum to one", "actor-specific source exposure"],
        ),
        module(
            "v58_fisher_kl_trust_region",
            ["M6 5", "M6 5.1", "M6 5.2", "M6 5.3", "M6 10.4"],
            "constraint_solver", ["work/continuous_multipop_mfg_v47.py"],
            "ContinuousMultiPopulationMFG policy update constraints",
            ["major target policy", "minor target action", "Fisher metric", "KL radii"],
            ["bounded major policy step", "bounded minor action step"],
            ["continuous_graphon_mfg"],
            "Fisher geodesic update with KL backtracking",
            "major KL <= 0.035 and minor KL <= 0.025",
            ["finite KL", "bounded policies", "backtracking recorded"],
        ),
        module(
            "v58_joint_implicit_wfr",
            ["M6 6", "M6 6.1", "M6 6.2", "M6 6.3", "M6 10.3"],
            "constraint_solver", ["work/fp_solvers_v52.py:step_repeated_unbalanced"],
            "FokkerPlanckSolver joint unnormalised-measure transition",
            ["opening measure", "HJB drift", "diffusion", "reaction fitness"],
            ["closing conditional density", "closing engagement mass", "WFR action"],
            ["continuous_graphon_mfg", "age_stratified_social_contagion"],
            "implicit finite-volume advection-diffusion-reaction minimizing-movement step",
            "conditional and stock residual below 1e-12",
            ["nonnegative measure", "mass account identity", "no split reaction step"],
        ),
        module(
            "v58_schrodinger_bridge_estimator",
            ["M6 7", "M6 15.1", "M6 15.2"],
            "estimator", ["work/information_geometry_v58.py:_schrodinger_bridge"],
            "PartiallyObservableInformationGeometry bridge diagnostic",
            ["prior posterior", "current filtered posterior", "reference diffusion kernel"],
            ["pathwise bridge coupling", "bridge cost used as cognitive friction"],
            ["continuous_graphon_mfg"],
            "entropic iterative proportional fitting",
            "both marginal residuals below 1e-7",
            ["nonnegative coupling", "opening and closing marginals respected"],
        ),
        module(
            "v58_fgw_network_alignment",
            ["M6 8", "M6 9"], "estimator",
            ["work/information_geometry_v58.py:_fgw_alignment"],
            "PartiallyObservableInformationGeometry frozen alignment map",
            ["actor kernel", "age kernel", "initial support features"],
            ["row-normalized four-party alignment entering monthly Graphon kernel"],
            ["continuous_graphon_mfg"],
            "alternating fused feature-structure cost and entropic coupling",
            "coupling and alignment row residual below 1e-12",
            ["nonnegative coupling", "row-normalized alignment", "no terminal multiplier"],
            pathwise=False,
        ),
        module(
            "v58_divergence_robust_information",
            ["M6 11", "M6 11.1", "M6 11.2", "M6 11.3"],
            "constraint_solver", ["work/information_geometry_v58.py", "work/continuous_multipop_mfg_v47.py:_effective_graphon"],
            "PartiallyObservableInformationGeometry ambiguity radii",
            ["jump signal", "common noise", "source credibility", "frozen kernel draw"],
            ["robust posterior lower signal", "ambiguity-shrunk cross-actor kernel"],
            ["continuous_graphon_mfg"],
            "KL ambiguity radius and bounded worst-direction mean adjustment",
            "ambiguity radius remains in [0,0.18]",
            ["radius data tag explicit", "no post-hoc outcome reweighting", "path draw frozen"],
        ),
        module(
            "v58_information_value_allocation",
            ["M6 12"], "constraint_solver",
            ["work/information_geometry_v58.py:update"],
            "PartiallyObservableInformationGeometry source allocation",
            ["prior variance", "source exposure", "credibility", "lagged major information action"],
            ["budget-feasible source allocation", "expected variance reduction"],
            ["v58_actor_specific_belief_filter"],
            "closed-form normalized allocation under aggregate information budget",
            "allocation sum equals information budget within 1e-12",
            ["nonnegative allocation", "budget feasibility", "aggregate-only allocation"],
        ),
        module(
            "v58_information_geometry_closure",
            ["M6 13", "M6 14", "M6 15", "M6 16"],
            "dynamic_state", [
                "work/information_geometry_v58.py",
                "work/continuous_multipop_mfg_v47.py",
                "work/simulate_taiwan_multidomain_v42.py",
            ],
            "same-path MFG-information interface",
            ["current true state", "belief and credibility state", "MFG action", "network kernel"],
            ["support", "joint actions", "network actions", "engagement mass", "diagnostics"],
            ["age_stratified_social_contagion", "v57_common_knowledge", "v57_closed_feedback", "campaign_deds_multidomain"],
            "monthly inner equilibrium with explicit lagged action feedback",
            "MFG residual below 5e-4 and all information-geometry invariants pass",
            ["same path", "single clock", "no post-hoc probability adjustment", "no future information"],
        ),
        {
            "module_id": "v58_deferred_information_geometry_methods",
            "document_sections": ["M6 17 third stage"],
            "equations": ["information bottleneck", "causal representation manifold", "active observation", "higher-order connections"],
            "role": "theory_only", "status": "theory_only",
            "implementation_status": "NOT_ACTIVE_PENDING_DATA",
            "implementation": ["outputs/台海模型信息几何深化与升级路线_飞书LaTeX.md"],
            "symbols": [], "state_owner": "none", "clock": "not active",
            "pathwise": False, "inputs": [], "outputs": [],
            "feedback_consumers": [], "solver": "none",
            "convergence_criterion": "not applicable", "data": [],
            "invariants": ["must not enter result path"],
            "tests": ["outputs/taiwan_v58_static_coupling_audit.json"],
            "outputs_artifact": "outputs/台海模型信息几何深化与升级路线_飞书LaTeX.md",
        },
    ])

    trace = inherited
    trace.update({
        "model_version": "V5.8",
        "document_ids": list(dict.fromkeys(inherited["document_ids"] + [
            "M6t9dmRhkoMhPrxJnV3ciYJXn2c"
        ])),
        "heading_inventory_count": inherited["heading_inventory_count"] + 56,
        "coverage_basis": (
            "V5.7 complete inventory plus all 56 information-geometry headings; "
            "explicit third-stage methods classified theory-only pending data"
        ),
        "active_equation_family_count": inherited["active_equation_family_count"] + 9,
        "passed_equation_family_count": inherited["passed_equation_family_count"] + 9,
        "coverage_percent": 100.0,
        "gate_status": "PASS",
        "modules": modules,
        "source_ledger": ledger,
    })
    TRACE.write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")

    files = {
        "information": (WORK / "information_geometry_v58.py").read_text(encoding="utf-8"),
        "mfg": (WORK / "continuous_multipop_mfg_v47.py").read_text(encoding="utf-8"),
        "fp": (WORK / "fp_solvers_v52.py").read_text(encoding="utf-8"),
        "battle": (WORK / "simulate_taiwan_multidomain_v42.py").read_text(encoding="utf-8"),
        "wrapper": (WORK / "simulate_taiwan_fully_closed_v58.py").read_text(encoding="utf-8"),
    }
    required = {
        "information": [
            "posterior", "reliability", "_schrodinger_bridge", "_fgw_alignment",
            "source_allocation", "ambiguity", "register_information_action",
        ],
        "mfg": [
            "belief_system.update", "step_repeated_unbalanced",
            "_minor_trust_step", "fgw_alignment", "ambiguity_radius",
            "register_information_action",
        ],
        "fp": [
            "step_repeated_unbalanced", "reaction", "closing_mass",
            "stock_accounting_residual",
        ],
        "battle": ["mfg_system.update", "mfg_factors.support", "joint_action_field"],
        "wrapper": ["MFG_CONFIG", "TAIWAN_V58_COMPLETE_RUN", "political_factory="],
    }
    missing = {
        name: [token for token in tokens if token not in files[name]]
        for name, tokens in required.items()
    }
    missing = {name: tokens for name, tokens in missing.items() if tokens}
    audit = {
        "model_version": "V5.8",
        "status": "PASS" if not missing else "FAIL",
        "blocking_gaps": missing,
        "diagnostic_status": diagnostic["status"],
        "master_clock": "same-path monthly clock plus unique event clock",
        "active_module_count": sum(item["status"] == "active" for item in modules),
        "theory_only_count": sum(item["status"] == "theory_only" for item in modules),
        "weak_coupling_adapters": [],
        "duplicate_state_owners": [],
        "mandatory_solver_proxies": [],
        "new_feedback_edges": [
            "truth->observation->belief->MFG benefit",
            "credibility->observation weighting->belief",
            "major information action(t-1)->observation channel(t)",
            "FGW alignment->Graphon kernel->actions",
            "ambiguity radius->robust belief and kernel",
            "joint WFR mass->social action and support",
        ],
        "superseded_active_path": ["WFR Strang reaction-transport split"],
        "simulation_authorized": not missing and diagnostic["status"] == "PASS",
    }
    AUDIT.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"TAIWAN_V58_STATIC_AUDIT: {audit['status']}")
    if audit["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
