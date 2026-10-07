from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
V47 = OUT / "taiwan_v47_fully_closed_simulation.json"
V49 = OUT / "taiwan_v49_energy_mechanism_simulation.json"
COMPARISON = OUT / "taiwan_v49_vs_v47_comparison.json"
SUMMARY_CSV = OUT / "taiwan_v49_summary.csv"
MANIFEST = OUT / "taiwan_v49_reproducibility_manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def central(data: dict) -> dict:
    calendar = next(
        row for row in data["verification"]["calendar_2100"]
        if row["trade_scenario"] == "baseline"
    )
    aggregate = next(
        row for row in data["aggregate"]
        if row["trade_scenario"] == "baseline" and row["horizon_years"] == 30
    )
    conditional = [
        row for row in data["conditional"]
        if row["trade_scenario"] == "baseline" and row["horizon_years"] == 30
    ]
    return {
        "calendar_2100": calendar,
        "campaign_30y": aggregate,
        "by_case_30y": {row["case"]: row["conditional_broad_control_rate"] for row in conditional},
        "minimum_ess": min(row["importance_sampling_effective_sample_size"] for row in conditional),
    }


def main() -> None:
    old_data = json.loads(V47.read_text(encoding="utf-8"))
    new_data = json.loads(V49.read_text(encoding="utf-8"))
    old = central(old_data)
    new = central(new_data)
    delta = {
        "conditional_broad_control_30y": new["campaign_30y"]["conditional_mean"] - old["campaign_30y"]["conditional_mean"],
        "conditional_broad_control_by_2100": new["calendar_2100"]["conditional_broad_control_by_2100"] - old["calendar_2100"]["conditional_broad_control_by_2100"],
        "joint_force_and_control_by_2100": new["calendar_2100"]["joint_force_and_broad_control_by_2100"] - old["calendar_2100"]["joint_force_and_broad_control_by_2100"],
    }
    COMPARISON.write_text(json.dumps({
        "classification": "same-seed structural version comparison; combined mechanism plus energy effect, not a real-world causal estimate",
        "v47": old,
        "v49": new,
        "v49_minus_v47": delta,
        "mechanism_ablation": json.loads((OUT / "taiwan_v48_mechanism_ablation.json").read_text(encoding="utf-8")),
        "energy_ablation": json.loads((OUT / "taiwan_v49_energy_ablation.json").read_text(encoding="utf-8")),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    rows = []
    for version, block in (("V4.7", old), ("V4.9", new)):
        rows.append({
            "version": version,
            "force_cif_2100": block["calendar_2100"]["force_onset_cif"],
            "peace_cif_2100": block["calendar_2100"]["peace_arrangement_cif"],
            "separation_cif_2100": block["calendar_2100"]["legal_separation_cif"],
            "status_quo_2100": block["calendar_2100"]["status_quo_survival"],
            "conditional_control_30y": block["campaign_30y"]["conditional_mean"],
            "conditional_control_by_2100": block["calendar_2100"]["conditional_broad_control_by_2100"],
            "joint_force_control_by_2100": block["calendar_2100"]["joint_force_and_broad_control_by_2100"],
            "minimum_ess": block["minimum_ess"],
        })
    with SUMMARY_CSV.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    artifacts = [
        "work/dynamic_mechanism_design_v48.py",
        "work/dynamic_macro_equilibrium_v47.py",
        "work/four_party_energy_network_v49.py",
        "work/numerical_acceleration.py",
        "work/accelerated_solver_infrastructure.py",
        "work/simulate_taiwan_multidomain_v42.py",
        "work/simulate_taiwan_fully_closed_v47.py",
        "work/simulate_taiwan_fully_closed_v49.py",
        "work/summarize_taiwan_v49.py",
        "work/audit_taiwan_v49.py",
        "work/diagnostic_taiwan_v49.py",
        "work/ablate_taiwan_mechanisms_v48.py",
        "work/ablate_taiwan_energy_v49.py",
        "outputs/taiwan_v49_traceability.json",
        "outputs/taiwan_v49_static_coupling_audit.json",
        "outputs/taiwan_v48_mechanism_ablation.json",
        "outputs/taiwan_v49_energy_ablation.json",
        "outputs/taiwan_v49_energy_mechanism_results.csv",
        "outputs/taiwan_v49_energy_mechanism_simulation.json",
        "outputs/taiwan_v49_vs_v47_comparison.json",
        "outputs/taiwan_v49_summary.csv",
        "outputs/taiwan_v49_native_windows_run_config.json",
        "outputs/数值求解基础设施实施说明.md",
        "outputs/台海V4.9正式闭合仿真结果_飞书增补.md",
    ]
    MANIFEST.write_text(json.dumps({
        "model_version": "V4.9",
        "runtime_validation": new_data["verification"]["runtime_validation"],
        "artifacts": {
            name: {"sha256": sha256(ROOT / name), "bytes": (ROOT / name).stat().st_size}
            for name in artifacts
        },
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print("TAIWAN_V49_SUMMARY_AND_MANIFEST: PASS")


if __name__ == "__main__":
    main()
