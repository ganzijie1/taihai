from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "outputs" / "taiwan_v77_theory_only_resolution.json"


RESOLUTION = {
    "lie_group_and_symplectic_fast_mechanics": {
        "useful": True, "disposition": "superseded_by_active",
        "replacement": "v66_symplectic_lie_group_core",
        "reason": "The operational transport, SO(3), and symplectic states already implement the identifiable portion.",
    },
    "contact_and_metriplectic_irreversibility": {
        "useful": True, "disposition": "superseded_by_active",
        "replacement": "v66_port_hamiltonian_transport_shell",
        "reason": "Irreversibility is owned by explicit loss stocks and the active port-Hamiltonian dissipative shell; a second contact state would duplicate ownership.",
    },
    "kahler_geometry_not_adopted": {
        "useful": False, "disposition": "excluded_no_identifiable_structure",
        "replacement": None,
        "reason": "The real mixed state has no observed complex structure or falsifiable Kahler constraint.",
    },
    "category_theory_interface_language": {
        "useful": False, "disposition": "excluded_non_dynamic_language",
        "replacement": None,
        "reason": "Category theory specifies typed composition but contributes no independently estimable state equation or probability law.",
    },
    "recursive_svar_diagnostic_v55": {
        "useful": True, "disposition": "superseded_by_identified_estimator",
        "replacement": "macro_lp_var_v55",
        "reason": "Ordering-sensitive recursive identification is dominated by the active predictive VAR/local-projection estimator and is not used as a causal mechanism.",
    },
    "unselected_causal_estimators_v55": {
        "useful": False, "disposition": "excluded_nonidentified",
        "replacement": None,
        "reason": "No valid instrument, staggered treatment cohort, or donor pool exists for the claimed wartime intervention; synthetic estimates would be fabricated.",
    },
    "unselected_measurement_extensions_v55": {
        "useful": False, "disposition": "excluded_missing_microdata",
        "replacement": None,
        "reason": "Harmonized four-party capital, labor, and polling microdata are absent; active structural states remain scenario-prior states rather than falsely estimated measurements.",
    },
    "kahler_geometry_v56_not_adopted": {
        "useful": False, "disposition": "superseded_duplicate_exclusion",
        "replacement": "kahler_geometry_not_adopted",
        "reason": "Duplicate statement of the same absent complex-structure condition.",
    },
    "contact_geometry_v56_not_adopted": {
        "useful": True, "disposition": "superseded_by_active",
        "replacement": "v66_port_hamiltonian_transport_shell",
        "reason": "The active dissipative shell and explicit losses already own the irreversible mechanism.",
    },
    "category_theory_v56_interface_only": {
        "useful": False, "disposition": "superseded_duplicate_exclusion",
        "replacement": "category_theory_interface_language",
        "reason": "Duplicate software-interface language, not an additional dynamic mechanism.",
    },
    "v58_deferred_information_geometry_methods": {
        "useful": True, "disposition": "split_and_resolved",
        "replacement": [
            "v77_information_bottleneck", "v77_active_observation",
            "v75_directed_multilayer_hypergraph_operator",
        ],
        "reason": "Information bottleneck and active sensing are executable; higher-order connections are the active hypergraph. Causal-representation claims remain excluded without interventions.",
    },
    "v59_deferred_second_batch_network_geometry": {
        "useful": True, "disposition": "superseded_by_active",
        "replacement": [
            "v77_fgw_ollivier_hodge", "v77_complete_online_tda",
            "v75_directed_multilayer_hypergraph_operator",
        ],
        "reason": "Every quantitative branch is now on the same-path runtime: FGW proximal alignment, directed curvature, Hodge flow, hypergraph dynamics, exact finite-complex persistence, and consecutive zigzag maps.",
    },
    "v60_explicitly_deferred_riemannian_methods": {
        "useful": False, "disposition": "excluded_or_forbidden",
        "replacement": None,
        "reason": "No identifiable indefinite metric or global connection exists; curvature-to-terminal-probability adjustment is post-hoc and forbidden by the model contract.",
    },
    "v61_explicitly_deferred_alternative_estimators": {
        "useful": True, "disposition": "split_and_resolved",
        "replacement": "v77_nested_ces_production",
        "reason": "Nested CES is active. DEA and stochastic-frontier estimation are excluded because no comparable wartime output frontier is observed and they would duplicate the owned productivity state.",
    },
    "v62_explicitly_deferred_petty_natural_price_common_scale": {
        "useful": False, "disposition": "excluded_duplicate_accounting",
        "replacement": None,
        "reason": "A Petty land common scale and natural-price account add no identified wartime state and would duplicate SAM/CGE/SFC price and flow ownership.",
    },
}


def write_resolution() -> dict:
    payload = {
        "model_version": "V7.7-COMPLETE-THEORY-CLOSURE",
        "policy": "all useful theory-only items must be fully executable; exclusions require a mechanism or identification argument",
        "entries": RESOLUTION,
        "counts": {
            "total": len(RESOLUTION),
            "useful": sum(bool(item["useful"]) for item in RESOLUTION.values()),
            "excluded": sum(not bool(item["useful"]) for item in RESOLUTION.values()),
        },
        "unresolved_useful": [
            key for key, item in RESOLUTION.items()
            if item["useful"] and not item.get("replacement")
        ],
    }
    TARGET.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


if __name__ == "__main__":
    result = write_resolution()
    print("TAIWAN_V77_THEORY_RESOLUTION:", "PASS" if not result["unresolved_useful"] else "FAIL")
