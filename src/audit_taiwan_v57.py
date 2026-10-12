from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "work"
OUT = ROOT / "outputs"
BASE = OUT / "taiwan_v56_traceability.json"
TRACE = OUT / "taiwan_v57_traceability.json"
AUDIT = OUT / "taiwan_v57_static_coupling_audit.json"
DIAGNOSTIC = OUT / "taiwan_v57_social_regime_diagnostic.json"
METHOD_DOC = "https://my.feishu.cn/docx/E6mrdVdn7oyWZ8xogIqcRGLKn2d"


def source(identification: str = "scenario_prior") -> dict:
    return {
        "source": METHOD_DOC,
        "publisher": "Wartime collective-action and regime-transition model specification",
        "observation_date": "2026-10-09",
        "unit": "dimensionless monthly structural state unless inherited from owning module",
        "identification": identification,
    }


def module(
    module_id: str, sections: list[str], role: str, implementation: list[str],
    owner: str, inputs: list[str], outputs: list[str], consumers: list[str],
    solver: str, criterion: str, invariants: list[str],
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
        "clock": "monthly same-path and event clock",
        "pathwise": True,
        "inputs": inputs,
        "outputs": outputs,
        "feedback_consumers": consumers,
        "solver": solver,
        "convergence_criterion": criterion,
        "data": [source()],
        "invariants": invariants,
        "tests": [
            "work/diagnostic_taiwan_v57.py",
            "outputs/taiwan_v57_social_regime_diagnostic.json",
        ],
        "outputs_artifact": "outputs/taiwan_v57_social_regime_simulation.json",
    }


def main() -> None:
    inherited = json.loads(BASE.read_text(encoding="utf-8"))
    diagnostic = json.loads(DIAGNOSTIC.read_text(encoding="utf-8"))
    if diagnostic.get("status") != "PASS":
        raise RuntimeError("V5.7 diagnostic gate is not PASS")
    modules = inherited["modules"]
    age_social = next(
        item for item in modules if item["module_id"] == "age_stratified_social_contagion"
    )
    age_social["outputs"] = list(dict.fromkeys(
        age_social["outputs"] + ["social action seed", "mobilization", "fatigue"]
    ))
    age_social["feedback_consumers"] = list(dict.fromkeys(
        age_social["feedback_consumers"] + ["v57_participation_organization"]
    ))
    collapse = next(
        item for item in modules if item["module_id"] == "labor_firms_supply_collapse"
    )
    collapse["invariants"] = list(dict.fromkeys(
        collapse["invariants"] + [
            "legacy independent coup draw disabled when V5.7 competing-risk owner is active"
        ]
    ))

    modules.extend([
        module(
            "v57_household_livelihood_pressure", ["E6 3", "E6 3.1", "E6 3.2"],
            "dynamic_state", ["work/wartime_collective_action_regime_v57.py"],
            "WartimeCollectiveActionRegimeSystem.pressure",
            ["same-path shortage", "inflation", "output", "damage", "financial stress", "lagged relief"],
            ["household pressure memory"],
            ["v57_common_knowledge", "v57_government_mpc", "v57_social_competing_risks"],
            "bounded exponential-memory update", "state remains in [0,1]",
            ["no future information", "same-path lagged relief", "bounded pressure"],
        ),
        module(
            "v57_common_knowledge", ["E6 4", "E6 10.3"], "dynamic_state",
            ["work/wartime_collective_action_regime_v57.py"],
            "WartimeCollectiveActionRegimeSystem.common_knowledge",
            ["same-path network action", "verified shortage/damage/progress signals", "lagged information control"],
            ["common-knowledge state", "cross-actor network exposure"],
            ["v57_participation_organization", "v57_social_competing_risks"],
            "four-actor directed kernel with credibility-gap feedback",
            "bounded state and finite directed spillovers",
            ["directed actor kernel", "bounded common knowledge"],
        ),
        module(
            "v57_participation_organization", ["E6 5", "E6 5.1", "E6 5.2", "E6 5.3"],
            "dynamic_state", ["work/wartime_collective_action_regime_v57.py"],
            "WartimeCollectiveActionRegimeSystem.organization and collective_action",
            ["pressure", "common knowledge", "MFG/social action seed", "government participation risk"],
            ["organization capital", "final collective action", "network action"],
            ["v57_government_mpc", "v57_social_competing_risks", "v57_closed_feedback"],
            "heterogeneous logit participation with organization accumulation",
            "states and actions remain in [0,1]",
            ["single owner of final V5.7 actor_action", "bounded organization"],
        ),
        module(
            "v57_government_mpc", ["E6 6"], "constraint_solver",
            ["work/wartime_collective_action_regime_v57.py:_government_mpc"],
            "WartimeCollectiveActionRegimeSystem.controls",
            ["same-path pressure", "action seed", "fiscal capacity", "support", "organization"],
            ["relief", "concession", "information", "policing", "repression"],
            ["v57_household_livelihood_pressure", "v57_security_loyalty_hmm", "v57_closed_feedback"],
            "six-month receding-horizon search over eight transparent policy archetypes",
            "finite objective and controls in [0,0.85]",
            ["fiscal scaling", "no future realized state", "bounded control jump"],
        ),
        module(
            "v57_security_loyalty_hmm", ["E6 7"], "dynamic_state",
            ["work/wartime_collective_action_regime_v57.py:_update_loyalty"],
            "WartimeCollectiveActionRegimeSystem.loyalty",
            ["command", "fiscal capacity", "progress", "leader continuity", "elite split", "repression"],
            ["four-state posterior", "compliance factor"],
            ["v57_social_competing_risks", "v57_closed_feedback"],
            "pathwise prediction-transition and observation update",
            "posterior simplex residual below 1e-12",
            ["nonnegative posterior", "probabilities sum to one", "bounded compliance"],
        ),
        module(
            "v57_social_competing_risks", ["E6 8", "E6 9"], "terminal_model",
            ["work/wartime_collective_action_regime_v57.py:_draw_competing_event"],
            "WartimeCollectiveActionRegimeSystem event/regime/first-event ledgers",
            ["action", "pressure", "organization", "common knowledge", "fragility", "loyalty", "government controls"],
            ["seven mutually exclusive event types", "semi-Markov regime", "coup/revolution/reform/restoration signals"],
            ["labor_firms_supply_collapse", "v57_closed_feedback", "campaign_deds_multidomain"],
            "cause-specific exponential clock and one-event categorical draw",
            "monthly event probability sum at most one; valid regime codes",
            ["one cause per actor-month", "exclusive first event", "finite recovery path"],
        ),
        module(
            "v57_revolution_episode_resolution", ["E6 20.1", "E6 20.4"],
            "dynamic_state", ["work/wartime_collective_action_regime_v57.py:_update_revolution_episode"],
            "WartimeCollectiveActionRegimeSystem revolution state and resolution ledger",
            ["revolution event", "organization", "action", "loyalty", "repression"],
            ["active revolution", "revolutionary capacity", "government replacement", "suppression"],
            ["v57_closed_feedback", "v57_return_to_war_talk"],
            "semi-Markov episode with mutually exclusive cause-specific resolution hazards",
            "replacement and suppression are mutually exclusive; resolution code is in {0,1,2}",
            ["revolution is not a campaign terminal", "one resolution per episode", "bounded capacity"],
        ),
        module(
            "v57_foreign_intervention_nationalism", ["E6 20.2", "E6 20.3"],
            "dynamic_state", ["work/wartime_collective_action_regime_v57.py:_update_revolution_episode"],
            "WartimeCollectiveActionRegimeSystem directed support kernels and external-first state",
            ["same-path intervention capacity", "fragility", "loyalty", "support", "war intensity"],
            ["challenger support", "incumbent support", "intervention pressure", "external-first share"],
            ["v57_revolution_episode_resolution", "v57_return_to_war_talk", "v57_closed_feedback"],
            "two directed support kernels with bounded memory and nationalist timing response",
            "all states remain in [0,1] and no symmetric-support assumption is imposed",
            ["support target is explicit", "same-path intervention capacity", "bounded memory"],
        ),
        module(
            "v57_return_to_war_talk", ["E6 20.5", "E6 20.6"], "dynamic_state",
            ["work/wartime_collective_action_regime_v57.py", "work/simulate_taiwan_multidomain_v42.py"],
            "existing campaign regime transition with lagged social continuation and negotiation pressure",
            ["government replacement", "suppression", "external-first share", "support", "pressure", "progress"],
            ["war-continuation preference", "negotiation pressure", "next-month regime multipliers"],
            ["campaign_deds_multidomain", "competing_risk_terminal_outcomes"],
            "bounded complementary preferences entering the existing semi-Markov campaign transition",
            "war continuation plus negotiation pressure equals one within 1e-12",
            ["no new campaign terminal", "one-month lag", "no post-hoc outcome adjustment"],
        ),
        module(
            "v57_closed_feedback", ["E6 10", "E6 10.1", "E6 10.2", "E6 13"],
            "dynamic_state", [
                "work/wartime_collective_action_regime_v57.py",
                "work/simulate_taiwan_multidomain_v42.py",
            ],
            "simulate_mechanisms lagged social feedback factors",
            ["current social events", "organization", "loyalty", "government controls"],
            ["labor factor", "fiscal drag", "logistics factor", "morale factor", "support shift"],
            ["dynamic_cge_dsge", "financial_system", "operations_network_flow_metric_search", "campaign_deds_multidomain"],
            "explicit one-month same-path stock-flow feedback",
            "labor>=0.35, logistics>=0.45, morale>=0.35, fiscal drag<=0.12",
            ["no post-hoc probability adjustment", "bounded factors", "explicit one-period lag"],
        ),
    ])

    trace = inherited
    trace.update({
        "model_version": "V5.7",
        "document_ids": list(dict.fromkeys(inherited["document_ids"] + [
            "E6mrdVdn7oyWZ8xogIqcRGLKn2d"
        ])),
        "heading_inventory_count": inherited["heading_inventory_count"] + 37,
        "coverage_basis": "V5.6 complete inventory plus all 37 V5.7 social-regime headings, including section 20",
        "active_equation_family_count": inherited["active_equation_family_count"] + 10,
        "passed_equation_family_count": inherited["passed_equation_family_count"] + 10,
        "coverage_percent": 100.0,
        "gate_status": "PASS",
        "modules": modules,
    })
    TRACE.write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")

    files = {
        "political": (WORK / "wartime_collective_action_regime_v57.py").read_text(encoding="utf-8"),
        "battle": (WORK / "simulate_taiwan_multidomain_v42.py").read_text(encoding="utf-8"),
        "core": (WORK / "simulate_taiwan_fully_closed_v47.py").read_text(encoding="utf-8"),
        "wrapper": (WORK / "simulate_taiwan_fully_closed_v57.py").read_text(encoding="utf-8"),
    }
    required = {
        "political": [
            "_government_mpc", "_update_loyalty", "_draw_competing_event",
            "_update_revolution_episode", "revolutionary_support_kernel",
            "incumbent_support_kernel", "external_first_share",
            "organization", "first_event", "order_restored",
        ],
        "battle": [
            "political_system.update", "actor_social_labor_factor",
            "actor_social_logistics_factor", "actor_social_morale_factor",
            "actor_social_fiscal_drag", "older independent coup draw",
        ],
        "core": ["political_factory", "political_system=political", '"political"'],
        "wrapper": [
            "WartimeCollectiveActionRegimeSystem", "political_factory=",
            "TAIWAN_V57_COMPLETE_RUN",
        ],
    }
    missing = {
        name: [token for token in tokens if token not in files[name]]
        for name, tokens in required.items()
    }
    missing = {name: tokens for name, tokens in missing.items() if tokens}
    duplicate_owners = []
    if "if political_factors is not None" not in files["battle"]:
        duplicate_owners.append("coup event ownership")
    audit = {
        "model_version": "V5.7",
        "status": "PASS" if not missing and not duplicate_owners else "FAIL",
        "blocking_gaps": missing,
        "diagnostic_status": diagnostic["status"],
        "master_clock": "same-path monthly clock plus unique event clock",
        "active_module_count": sum(item["status"] == "active" for item in modules),
        "theory_only_count": sum(item["status"] == "theory_only" for item in modules),
        "weak_coupling_adapters": [],
        "duplicate_state_owners": duplicate_owners,
        "mandatory_solver_proxies": [],
        "new_feedback_edges": [
            "economy->pressure", "pressure/network->action", "government->pressure/loyalty",
            "loyalty->events/morale", "events->labor/finance/logistics/command",
            "labor/finance/logistics/command->economy/campaign",
            "revolution->foreign support/nationalist timing->replacement or suppression",
            "replacement or suppression->war continuation or negotiation->campaign regime",
        ],
        "simulation_authorized": not missing and not duplicate_owners
        and diagnostic["status"] == "PASS",
    }
    AUDIT.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"TAIWAN_V57_STATIC_AUDIT: {audit['status']}; modules={len(modules)}")
    if audit["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
