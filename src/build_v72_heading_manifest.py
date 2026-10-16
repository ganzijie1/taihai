from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
DETAILS = OUT / "taiwan_v72_feishu_heading_details.json"
EXTRA_DETAILS = OUT / "taiwan_v72_extra_heading_details.json"
TRACE = OUT / "taiwan_v72_traceability.json"
TARGET = OUT / "taiwan_v72_heading_module_map.json"

MAIN_DOC = "UEnMdV2AGoaAuKxAsSwcg563nSb"

EXPLANATORY_WORDS = (
    "摘要", "参考文献", "参考资料", "资料入口", "一句话结论", "最终建议",
    "最终判断", "结论先行", "研究目标", "为什么", "术语澄清", "版本说明",
    "版本与边界", "解释边界", "方法边界", "文档边界", "不应采用", "不建议",
    "推荐的报告格式", "当前模型已经具备什么", "当前实现与主要缺口",
    "当前模型已经实现了什么", "现有基础和真正缺口", "本章解决的问题",
    "本次修正回答什么问题", "使用原则", "最终选型", "结论",
)
RESULT_WORDS = (
    "仿真结果", "正式仿真", "中央情景结果", "中心情景结果", "正式运行",
    "条件广泛控制率", "无条件联合概率", "加权条件广泛控制率",
    "三十年条件", "截至 2100 年", "截至2100年", "正式重跑", "正式复算",
)


DOC_ACTIVE: dict[str, list[str]] = {
    "R7qkdxtxCoIgqQxs0lgcrSofnVM": [
        "robust_dynamic_mechanism", "alliance_burden_vcg",
        "aid_moral_hazard_contract", "procurement_delivery_mechanism",
        "information_design_bce", "ceasefire_verification_contract",
        "multitask_governance_contract", "trade_relief_index_insurance",
    ],
    "SsfPdBT62o3abXxDAfSc37YMnfg": [
        "v70_tinbergen_algebraic_strategy", "v70_asymmetric_real_adjustment",
    ],
    "VqRNdUgRXob6ntxHBjscg4bPndf": ["v70_tinbergen_algebraic_strategy"],
    "E3SDdy2reoTeVjxiiZhcszw8nve": [
        "geometric_multiscale_coupling", "subriemannian_lie_group_reachability",
        "information_geometry_natural_gradient", "unbalanced_optimal_transport",
        "continuous_graphon_mfg", "v66_symplectic_lie_group_core",
        "v66_port_hamiltonian_transport_shell", "geospatial_pnt",
        "four_party_energy_maxflow_mincut",
    ],
    "QIQZd7QOCo7LQIxmJv0cHFb6nCg": [
        "hybrid_filippov_impulse", "symbolic_topological_dynamics",
    ],
    "U0Zydx0uSoxaxZxutUKc6lbDnBe": [
        "hybrid_filippov_impulse", "symbolic_topological_dynamics",
    ],
    "VqnJdDTrIoMrPyxOX7fcBvyande": [
        "mr_rd_cge_dq_v53", "msg_capital_labor_v53", "nonfinancial_sfc_v53",
        "four_party_social_accounting_matrix_v53", "age_activity_complementarity_v53",
        "hssw_welfare_v53", "dynamic_mr_cge_mrio_gravity_chips",
    ],
    "Uu2Mdtxw7oPqbLx9oumcHHdWn02": ["native_windows_cge_dsge_v54"],
    "AOM5dDZ2gooFEQxOiHacyTh5nke": ["native_windows_mfg_fp_v54"],
    "ZkYhdRSmYoLe06xyJ9Xc0cJnnEG": ["vectorized_differential_game_v54"],
    "HwsBdgnDnopW2fxAMW4cCYavn9f": ["batched_equation_kernels_v54"],
    "BZqxdzIzrolfo4x5Svsc1zHanLc": [
        "econometric_registry_v55", "ppml_trade_v55", "macro_lp_var_v55",
        "prewar_hmm_v55", "causal_aipw_v55", "competing_risk_prior_interface_v55",
        "alliance_multistate_prior_interface_v55", "posterior_path_interface_v55",
    ],
    "DkMsdAwMCoECTgxykCSc7D3Qneg": [
        "geometric_multiscale_coupling", "subriemannian_lie_group_reachability",
        "information_geometry_natural_gradient", "hybrid_filippov_impulse",
        "v66_symplectic_lie_group_core", "v66_port_hamiltonian_transport_shell",
    ],
    "E6mrdVdn7oyWZ8xogIqcRGLKn2d": [f"v57_{name}" for name in (
        "household_livelihood_pressure", "common_knowledge", "participation_organization",
        "government_mpc", "security_loyalty_hmm", "social_competing_risks",
        "revolution_episode_resolution", "foreign_intervention_nationalism",
        "return_to_war_talk", "closed_feedback",
    )],
    "M6t9dmRhkoMhPrxJnV3ciYJXn2c": [f"v58_{name}" for name in (
        "actor_specific_belief_filter", "source_credibility_hmm",
        "fisher_kl_trust_region", "joint_implicit_wfr", "schrodinger_bridge_estimator",
        "fgw_network_alignment", "divergence_robust_information",
        "information_value_allocation", "information_geometry_closure",
    )],
    "Ou3NdGj7CoGUqwxfht5clSdqnxd": [f"v59_{name}" for name in (
        "intensity_transition_graphon", "supra_spectral_geometry",
        "forman_bottleneck_geometry", "information_identity_sheaf",
        "network_geometry_mfg_closure",
    )],
    "UzrzdYIE7o43byxTBgbc4wRbnZf": [f"v60_{name}" for name in (
        "regularized_riemannian_anderson", "fisher_rao_policy_trust_region",
        "adaptive_bures_pnt_barycenter", "regime_riemannian_pod",
        "positive_product_manifold_metric",
    )],
    "IPmBdZpRVoQPv0xpoqucDan2nTb": [f"v61_{name}" for name in (
        "io_sam_three_sector_bridge", "agriculture_biophysical_land_inventory",
        "industry_network_firms_learning", "service_queue_digital_public",
        "labor_structural_transformation", "existing_full_economy_coupling",
    )],
    "U87BdiGmNo73F1xu9EvcZHg2nV7": [
        "v62_structural_transformation_and_labor_ot", "v62_dynamic_causal_fgw_bridge_dro",
        "v62_joint_state_action_wfr", "v66_path_conditioned_multilayer_ot",
        "v66_capacity_multicommodity_dynamic_ot",
    ],
    "Ac8Jd2DEgolpupxB855cq3e7nec": [
        "msg_capital_labor_v53", "nonfinancial_sfc_v53", "hssw_welfare_v53",
        "v61_labor_structural_transformation", "v62_structural_transformation_and_labor_ot",
    ],
    "UkOJdPutqoPGYPxcnZPcFQx4nrb": ["v63_online_system_stability"],
    "UQ5Cd1DwkoEpklx79eec8acwnEf": [
        "v64_public_gis_observation_layer", "v64_marked_cox_random_closed_state",
        "v64_coverage_service_solver", "v64_hualien_historical_replay",
    ],
    "Ob5WdBtBeoPftJxtHsCc6gzWn0e": [
        "v66_path_conditioned_multilayer_ot", "v66_capacity_multicommodity_dynamic_ot",
        "v69_financial_martingale_ot", "v69_block_conformal_dro",
        "v62_dynamic_causal_fgw_bridge_dro",
        "v58_schrodinger_bridge_estimator",
    ],
    "UsuFdiXrJomjGDxioUpcNFcMnzb": [
        "v66_symplectic_lie_group_core", "v66_port_hamiltonian_transport_shell",
    ],
    "TzW4dwMsdo14CIxUaW7c1Cb1nDg": [
        "v67_hybrid_large_deviation_importance_sampling",
    ],
    "G9KGdlUYko10qexv2k5cLL9inAd": ["v68_functional_operator_validation"],
    "V9f9dUVgroX9Zuxq3XxccH0tnlh": [
        "v71_time_expanded_multicommodity_network", "v71_cut_connectivity_and_cascade",
        "v71_matching_protection_and_recovery", "v71_hypergraph_complementarity",
        "v72_resistance_tree_spectrum", "v72_signed_coarsened_graph",
        "v72_temporal_survivable_network", "v72_matroid_conflict_scheduler",
        "v72_treewidth_solver_selection",
    ],
}

DOC_THEORY = {
    "E3SDdy2reoTeVjxiiZhcszw8nve": [
        "lie_group_and_symplectic_fast_mechanics", "contact_and_metriplectic_irreversibility",
        "tda_persistent_homology_candidate", "kahler_geometry_not_adopted",
        "category_theory_interface_language",
    ],
    "BZqxdzIzrolfo4x5Svsc1zHanLc": [
        "recursive_svar_diagnostic_v55", "unselected_causal_estimators_v55",
        "unselected_measurement_extensions_v55",
    ],
    "DkMsdAwMCoECTgxykCSc7D3Qneg": [
        "kahler_geometry_v56_not_adopted", "contact_geometry_v56_not_adopted",
        "category_theory_v56_interface_only",
    ],
    "M6t9dmRhkoMhPrxJnV3ciYJXn2c": ["v58_deferred_information_geometry_methods"],
    "Ou3NdGj7CoGUqwxfht5clSdqnxd": ["v59_deferred_second_batch_network_geometry"],
    "UzrzdYIE7o43byxTBgbc4wRbnZf": ["v60_explicitly_deferred_riemannian_methods"],
    "IPmBdZpRVoQPv0xpoqucDan2nTb": ["v61_explicitly_deferred_alternative_estimators"],
    "Ac8Jd2DEgolpupxB855cq3e7nec": ["v62_explicitly_deferred_petty_natural_price_common_scale"],
}

MAIN_CHAPTER_MODULES: dict[str, list[str]] = {
    "2": ["historical_identification_causal_hmm"],
    "3": ["prewar_game_evolution_feedback"],
    "4": ["continuous_graphon_mfg", "age_stratified_social_contagion"],
    "5": ["unbalanced_optimal_transport"],
    "6": ["leader_gated_jump_process"],
    "7": ["stratified_competing_survival"],
    "9": ["endogenous_alliance_stackelberg", "campaign_deds_multidomain"],
    "10": ["principal_agent_governance"],
    "11": ["historical_identification_causal_hmm"],
    "14": ["historical_identification_causal_hmm", "econometric_registry_v55"],
    "15": ["growth_defense_capital", "prewar_game_evolution_feedback"],
    "18": ["campaign_deds_multidomain", "endogenous_alliance_stackelberg", "rare_tail_hawkes_evt_importance"],
    "19": ["campaign_deds_multidomain", "operations_network_flow_metric_search"],
    "20": ["operations_network_flow_metric_search"],
    "21": ["operations_network_flow_metric_search", "geospatial_pnt", "continuous_graphon_mfg"],
    "22": [
        "historical_identification_causal_hmm", "prewar_game_evolution_feedback",
        "stratified_competing_survival", "campaign_deds_multidomain",
        "endogenous_alliance_stackelberg",
    ],
    "23": ["finance_fiscal_money_tax_debt_assets", "labor_firms_supply_collapse", "open_dsge_hank_mundell_fleming"],
    "24": ["opinion_election_arms_pipeline", "continuous_graphon_mfg", "age_stratified_social_contagion"],
    "25": [
        "rare_tail_hawkes_evt_importance", "endogenous_alliance_stackelberg",
        "campaign_deds_multidomain", "leader_gated_jump_process", "principal_agent_governance",
    ],
    "26": [
        "continuous_graphon_mfg", "campaign_deds_multidomain", "labor_firms_supply_collapse",
        "finance_fiscal_money_tax_debt_assets", "age_stratified_social_contagion",
        "spatial_population_control", "pathwise_terminal_aggregation",
    ],
    "27": [
        "dynamic_mr_cge_mrio_gravity_chips", "open_dsge_hank_mundell_fleming",
        "labor_firms_supply_collapse", "finance_fiscal_money_tax_debt_assets",
        "actual_bilateral_aid_clearing",
    ],
    "28": ["age_stratified_social_contagion", "pathwise_terminal_aggregation"],
    "29": [
        "continuous_graphon_mfg", "dynamic_mr_cge_mrio_gravity_chips",
        "open_dsge_hank_mundell_fleming", "rolling_pomdp_robust_game",
        "campaign_deds_multidomain", "geospatial_pnt", "spatial_population_control",
        "pathwise_terminal_aggregation",
    ],
    "30": [
        "geometric_multiscale_coupling", "v66_symplectic_lie_group_core",
        "v66_port_hamiltonian_transport_shell", "four_party_energy_maxflow_mincut",
    ],
    "31": ["hybrid_filippov_impulse", "symbolic_topological_dynamics", "continuous_graphon_mfg"],
    "32": [
        "continuous_graphon_mfg", "native_windows_mfg_fp_v54",
        "v58_information_geometry_closure", "v59_network_geometry_mfg_closure",
    ],
    "33": [
        "jorgenson_parameter_interface_v53", "mr_rd_cge_dq_v53", "msg_capital_labor_v53",
        "nonfinancial_sfc_v53", "four_party_social_accounting_matrix_v53",
        "age_activity_complementarity_v53", "hssw_welfare_v53",
    ],
    "34": [
        "econometric_registry_v55", "ppml_trade_v55", "macro_lp_var_v55",
        "prewar_hmm_v55", "causal_aipw_v55", "competing_risk_prior_interface_v55",
        "alliance_multistate_prior_interface_v55", "posterior_path_interface_v55",
    ],
    "35": [
        "geometric_multiscale_coupling", "subriemannian_lie_group_reachability",
        "information_geometry_natural_gradient", "v66_symplectic_lie_group_core",
    ],
    "36": [f"v57_{name}" for name in (
        "household_livelihood_pressure", "common_knowledge", "participation_organization",
        "government_mpc", "security_loyalty_hmm", "social_competing_risks",
        "revolution_episode_resolution", "foreign_intervention_nationalism",
        "return_to_war_talk", "closed_feedback",
    )],
    "37": [
        "v71_time_expanded_multicommodity_network", "v71_cut_connectivity_and_cascade",
        "v71_matching_protection_and_recovery", "v71_hypergraph_complementarity",
        "v72_resistance_tree_spectrum", "v72_signed_coarsened_graph",
        "v72_temporal_survivable_network", "v72_matroid_conflict_scheduler",
        "v72_treewidth_solver_selection",
    ],
    "v5.8": [
        "v58_actor_specific_belief_filter", "v58_source_credibility_hmm",
        "v58_fisher_kl_trust_region", "v58_joint_implicit_wfr",
        "v58_schrodinger_bridge_estimator", "v58_fgw_network_alignment",
        "v58_divergence_robust_information", "v58_information_value_allocation",
        "v58_information_geometry_closure",
    ],
    "v5.9": [
        "v59_intensity_transition_graphon", "v59_supra_spectral_geometry",
        "v59_forman_bottleneck_geometry", "v59_information_identity_sheaf",
        "v59_network_geometry_mfg_closure",
    ],
    "v6.1": [
        "v61_io_sam_three_sector_bridge", "v61_agriculture_biophysical_land_inventory",
        "v61_industry_network_firms_learning", "v61_service_queue_digital_public",
        "v61_labor_structural_transformation", "v61_existing_full_economy_coupling",
    ],
    "v6.2": [
        "v62_structural_transformation_and_labor_ot", "v62_dynamic_causal_fgw_bridge_dro",
        "v62_joint_state_action_wfr", "v63_online_system_stability",
    ],
}


def numbered_chapter(text: str) -> str | None:
    match = re.match(r"\s*(\d+)(?:\.|\s)", text)
    return match.group(1) if match else None


def is_explanatory(text: str) -> bool:
    return any(word in text for word in EXPLANATORY_WORDS)


def classify_main(text: str, chapter: str | None, module_ids: set[str]) -> tuple[str, list[str], str]:
    if any(word in text for word in RESULT_WORDS):
        return "superseded", [], "historical numerical result; excluded from the V7.2 active result path"
    if "旧版" in text or "旧结果" in text:
        return "superseded", [], "document explicitly marks this version as obsolete"
    if chapter in {"16", "17"}:
        return "superseded", [], "child heading of a chapter explicitly marked old and replaced by chapter 18"
    if chapter in {"8", "12", "13"}:
        return "explanatory", [], "output/reproducibility/limitation heading, not an equation family"
    modules = [m for m in MAIN_CHAPTER_MODULES.get(chapter or "", []) if m in module_ids]
    if modules:
        return "active", modules, f"main-document chapter {chapter} maps to current active implementation"
    if is_explanatory(text) or text.startswith("台海演化模型"):
        return "explanatory", [], "narrative scope, reference, limitation or conclusion heading"
    return "unresolved", [], "no explicit main-document chapter rule"


def classify_standalone(
    doc_id: str, text: str, module_ids: set[str], module_status: dict[str, str]
) -> tuple[str, list[str], str]:
    lower = text.lower()
    if doc_id == "R7qkdxtxCoIgqQxs0lgcrSofnVM" and (
        "研究诚信声明" in text or "摘要" in text
    ):
        return "superseded", [], (
            "the document's pre-V4.8 theory-only status statement was superseded by the "
            "audited V4.8 same-path implementation; a Feishu status addendum is required"
        )
    if doc_id == "SsfPdBT62o3abXxDAfSc37YMnfg":
        chapter = numbered_chapter(text)
        if chapter in {"19", "20"} or text.startswith(("19.", "20.")):
            return "active", ["v70_asymmetric_real_adjustment"], (
                "Tinbergen asymmetric adjustment-cost family implemented in V7.0"
            )
        if chapter == "22" or text.startswith("22."):
            return "active", ["v70_tinbergen_algebraic_strategy"], (
                "Tinbergen policy target-tool and macro model family implemented in V7.0"
            )
        return "theory_only", [], (
            "historical mathematical/economic literature review; no standalone quantitative-result claim"
        )
    theory = [m for m in DOC_THEORY.get(doc_id, []) if m in module_ids]
    theory_trigger = any(token in lower for token in (
        "暂不建议", "不采用", "候选", "deferred", "kähler", "kahler", "范畴论",
        "tda", "拓扑数据分析", "接触几何", "measurement extensions",
    ))
    if theory and theory_trigger:
        return "theory_only", theory, "document explicitly presents this method as deferred, diagnostic or not adopted"
    if any(word in text for word in RESULT_WORDS) or "旧版" in text or "旧结果" in text:
        return "superseded", [], "historical result heading; not used as a V7.2 quantitative conclusion"
    active = [m for m in DOC_ACTIVE.get(doc_id, []) if m in module_ids and module_status[m] == "active"]
    if active and not is_explanatory(text):
        return "active", active, "method-family document maps to its implemented active solver family"
    if is_explanatory(text) or text in {"摘要", "版本说明", "版本与边界"}:
        return "explanatory", [], "abstract, scope, references, recommendation or limitation heading"
    if active:
        return "active", active, "document-level active method family"
    return "unresolved", [], "document has no declared active or theory-only profile"


def main() -> None:
    details = json.loads(DETAILS.read_text(encoding="utf-8"))
    extras = json.loads(EXTRA_DETAILS.read_text(encoding="utf-8"))
    details["documents"].extend(extras["documents"])
    trace = json.loads(TRACE.read_text(encoding="utf-8"))
    modules = {item["module_id"]: item for item in trace["modules"]}
    module_ids = set(modules)
    module_status = {key: value["status"] for key, value in modules.items()}

    rows: list[dict] = []
    by_document: dict[str, Counter] = defaultdict(Counter)
    for document in details["documents"]:
        doc_id = document["document_id"]
        current_main_chapter: str | None = None
        for ordinal, heading in enumerate(document["headings"], start=1):
            text = heading["text"].strip()
            parsed = numbered_chapter(text)
            if doc_id == MAIN_DOC:
                version_context = next(
                    (version for version in ("v5.8", "v5.9", "v6.1", "v6.2")
                     if version.upper() in text.upper()),
                    None,
                )
                if version_context is not None:
                    current_main_chapter = version_context
                elif parsed is not None and not (
                    current_main_chapter in {"v5.8", "v5.9", "v6.1", "v6.2"}
                    and heading["level"] >= 2
                ):
                    current_main_chapter = parsed
            if doc_id == MAIN_DOC:
                effective_chapter = current_main_chapter
                classification, mapped, reason = classify_main(
                    text, effective_chapter, module_ids
                )
            else:
                classification, mapped, reason = classify_standalone(
                    doc_id, text, module_ids, module_status
                )
            rows.append({
                "document_id": doc_id,
                "ordinal": ordinal,
                "block_id": heading["block_id"],
                "level": heading["level"],
                "text": text,
                "classification": classification,
                "module_ids": mapped,
                "reason": reason,
            })
            by_document[doc_id][classification] += 1

    counts = Counter(row["classification"] for row in rows)
    unresolved = [row for row in rows if row["classification"] == "unresolved"]
    active_without_modules = [
        row for row in rows if row["classification"] == "active" and not row["module_ids"]
    ]
    missing_modules = sorted({
        module_id for row in rows for module_id in row["module_ids"] if module_id not in module_ids
    })
    mapped_active_modules = {
        module_id for row in rows if row["classification"] == "active"
        for module_id in row["module_ids"]
    }
    active_modules_without_headings = sorted(
        module_id for module_id, item in modules.items()
        if item["status"] == "active" and module_id not in mapped_active_modules
    )
    classified = len(rows) - len(unresolved)
    payload = {
        "model_version": "V7.2-ALGEBRAIC-DISCRETE-GRAPH",
        "classification": "HEADING_LEVEL_DOCUMENT_TO_CODE_TRACEABILITY; no outcome probabilities",
        "source_snapshot": str(DETAILS.relative_to(ROOT)).replace("\\", "/"),
        "total_headings": len(rows),
        "classification_counts": dict(sorted(counts.items())),
        "classified_heading_count": classified,
        "coverage_percent": round(100.0 * classified / max(len(rows), 1), 6),
        "status": "PASS" if not unresolved and not active_without_modules and not missing_modules and not active_modules_without_headings else "FAIL",
        "unresolved_count": len(unresolved),
        "active_without_modules_count": len(active_without_modules),
        "missing_module_ids": missing_modules,
        "active_modules_without_headings": active_modules_without_headings,
        "document_counts": {key: dict(sorted(value.items())) for key, value in by_document.items()},
        "unresolved": unresolved,
        "rows": rows,
    }
    TARGET.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: payload[key] for key in (
        "status", "total_headings", "classification_counts", "coverage_percent",
        "unresolved_count", "active_without_modules_count", "missing_module_ids",
        "active_modules_without_headings",
    )}, ensure_ascii=False, indent=2))
    if payload["status"] != "PASS":
        print("UNRESOLVED SAMPLE")
        for row in unresolved[:30]:
            print(row["document_id"], row["ordinal"], row["text"])
        raise SystemExit(1)


if __name__ == "__main__":
    main()
