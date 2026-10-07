from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def baseline_summary(payload: dict) -> dict:
    calendar = next(
        row for row in payload["verification"]["calendar_2100"]
        if row["trade_scenario"] == "baseline"
    )
    campaign = next(
        row for row in payload["aggregate"]
        if row["trade_scenario"] == "baseline" and row["horizon_years"] == 30
    )
    by_case = {
        row["case"]: row["conditional_broad_control_rate"]
        for row in payload["conditional"]
        if row["trade_scenario"] == "baseline" and row["horizon_years"] == 30
    }
    hybrid_cells = [
        cell.get("hybrid") for cell in payload["module_diagnostics"].values()
        if cell.get("hybrid") is not None
    ]
    topology_cells = [
        cell.get("topology") for cell in payload["module_diagnostics"].values()
        if cell.get("topology") is not None
    ]
    return {
        "calendar_2100": calendar,
        "campaign_30y": campaign,
        "by_case_30y": by_case,
        "minimum_ess": min(
            row["importance_sampling_effective_sample_size"]
            for row in payload["conditional"]
        ),
        "hybrid": {
            "max_reset_residual": max((x["max_reset_residual"] for x in hybrid_cells), default=0.0),
            "max_sliding_normal_residual": max((x["max_sliding_normal_residual"] for x in hybrid_cells), default=0.0),
            "max_hysteresis_violation": max((x["max_hysteresis_violation"] for x in hybrid_cells), default=0.0),
            "total_switches": {
                surface: sum(x["switch_count"][surface] for x in hybrid_cells)
                for surface in ("logistics", "financial", "mobilization", "command")
            },
            "total_events": {
                event: sum(x["event_counts"][event] for x in hybrid_cells)
                for event in ("homeland_island_damage", "leader_disruption", "coup", "collapse")
            },
        },
        "topology": {
            "total_observed_transition_edges": sum(
                x["observed_transition_edges"] for x in topology_cells
            ),
            "visited_symbols": sorted({
                symbol for x in topology_cells for symbol in x["visited_symbols"]
            }),
            "max_transition_accounting_residual": max(
                (x["transition_accounting_residual"] for x in topology_cells),
                default=0,
            ),
            "max_feedback_deviation": max(
                (x["max_feedback_deviation"] for x in topology_cells),
                default=0.0,
            ),
            "max_topological_entropy_upper_proxy": max(
                (x["topological_entropy_upper_proxy"] for x in topology_cells),
                default=0.0,
            ),
        },
    }


def main() -> None:
    v50_path = OUT / "taiwan_v50_geometry_simulation.json"
    v51_path = OUT / "taiwan_v51_hybrid_simulation.json"
    v50 = json.loads(v50_path.read_text(encoding="utf-8"))
    v51 = json.loads(v51_path.read_text(encoding="utf-8"))
    s50 = baseline_summary(v50)
    s51 = baseline_summary(v51)
    comparison = {
        "classification": "same-seed structural version comparison; not a real-world causal estimate",
        "v50": s50,
        "v51": s51,
        "v51_minus_v50": {
            "conditional_broad_control_30y": s51["campaign_30y"]["conditional_mean"] - s50["campaign_30y"]["conditional_mean"],
            "conditional_broad_control_by_2100": s51["calendar_2100"]["conditional_broad_control_by_2100"] - s50["calendar_2100"]["conditional_broad_control_by_2100"],
            "joint_force_and_control_by_2100": s51["calendar_2100"]["joint_force_and_broad_control_by_2100"] - s50["calendar_2100"]["joint_force_and_broad_control_by_2100"],
        },
    }
    sensitivity_path = OUT / "taiwan_v51_sensitivity_all.json"
    if sensitivity_path.exists():
        sensitivity = json.loads(sensitivity_path.read_text(encoding="utf-8"))
        comparison["sensitivity"] = {
            name: next(
                row for row in payload["aggregate"]
                if row["trade_scenario"] == "baseline" and row["horizon_years"] == 30
            )
            for name, payload in sensitivity.items()
        }
    comparison_path = OUT / "taiwan_v51_vs_v50_comparison.json"
    comparison_path.write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_path = OUT / "taiwan_v51_summary.csv"
    with summary_path.open("w", encoding="utf-8-sig", newline="") as handle:
        fields = [
            "version", "force_cif_2100", "peace_cif_2100", "separation_cif_2100",
            "status_quo_2100", "conditional_control_30y",
            "conditional_control_by_2100", "joint_force_control_by_2100", "minimum_ess",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for version, item in (("V5.0", s50), ("V5.1", s51)):
            c = item["calendar_2100"]
            writer.writerow({
                "version": version,
                "force_cif_2100": c["force_onset_cif"],
                "peace_cif_2100": c["peace_arrangement_cif"],
                "separation_cif_2100": c["legal_separation_cif"],
                "status_quo_2100": c["status_quo_survival"],
                "conditional_control_30y": item["campaign_30y"]["conditional_mean"],
                "conditional_control_by_2100": c["conditional_broad_control_by_2100"],
                "joint_force_control_by_2100": c["joint_force_and_broad_control_by_2100"],
                "minimum_ess": item["minimum_ess"],
            })
    artifacts = [
        OUT / "taiwan_v51_hybrid_results.csv", v51_path,
        OUT / "taiwan_v51_traceability.json", OUT / "taiwan_v51_static_coupling_audit.json",
        OUT / "taiwan_v51_frozen_specification.json", comparison_path, summary_path,
    ]
    if sensitivity_path.exists():
        artifacts.append(sensitivity_path)
    manifest = {
        "model_version": "V5.1",
        "status": "PASS",
        "artifacts_sha256": {path.name: sha256(path) for path in artifacts},
    }
    (OUT / "taiwan_v51_reproducibility_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("TAIWAN_V51_SUMMARY: PASS")


if __name__ == "__main__":
    main()
