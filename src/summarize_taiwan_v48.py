from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
V47 = OUT / "taiwan_v47_fully_closed_simulation.json"
V48 = OUT / "taiwan_v48_mechanism_simulation.json"
COMPARISON = OUT / "taiwan_v48_mechanism_comparison.json"
SUMMARY_CSV = OUT / "taiwan_v48_mechanism_summary.csv"
MANIFEST = OUT / "taiwan_v48_reproducibility_manifest.json"


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
    by_case = {
        row["case"]: row["conditional_broad_control_rate"]
        for row in data["conditional"]
        if row["trade_scenario"] == "baseline" and row["horizon_years"] == 30
    }
    ess = min(
        row["importance_sampling_effective_sample_size"]
        for row in data["conditional"]
        if row["trade_scenario"] == "baseline" and row["horizon_years"] == 30
    )
    return {"calendar_2100": calendar, "campaign_30y": aggregate, "by_case_30y": by_case, "minimum_ess": ess}


def main() -> None:
    v47 = json.loads(V47.read_text(encoding="utf-8"))
    v48 = json.loads(V48.read_text(encoding="utf-8"))
    old = central(v47)
    new = central(v48)
    delta = {
        "conditional_broad_control_30y": new["campaign_30y"]["conditional_mean"] - old["campaign_30y"]["conditional_mean"],
        "conditional_broad_control_by_2100": new["calendar_2100"]["conditional_broad_control_by_2100"] - old["calendar_2100"]["conditional_broad_control_by_2100"],
        "joint_force_and_control_by_2100": new["calendar_2100"]["joint_force_and_broad_control_by_2100"] - old["calendar_2100"]["joint_force_and_broad_control_by_2100"],
    }
    comparison = {
        "classification": "same-seed structural version comparison; not a causal estimate of adopting a real institution",
        "v47": old,
        "v48": new,
        "v48_minus_v47": delta,
        "ablation": json.loads((OUT / "taiwan_v48_mechanism_ablation.json").read_text(encoding="utf-8")),
    }
    COMPARISON.write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")
    rows = []
    for version, block in (("V4.7", old), ("V4.8", new)):
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
        "work/simulate_taiwan_multidomain_v42.py",
        "work/simulate_taiwan_fully_closed_v47.py",
        "work/simulate_taiwan_fully_closed_v48.py",
        "work/audit_taiwan_v48.py",
        "work/diagnostic_taiwan_v48.py",
        "work/ablate_taiwan_mechanisms_v48.py",
        "outputs/taiwan_v48_traceability.json",
        "outputs/taiwan_v48_static_coupling_audit.json",
        "outputs/taiwan_v48_mechanism_ablation.json",
        "outputs/taiwan_v48_mechanism_results.csv",
        "outputs/taiwan_v48_mechanism_simulation.json",
        "outputs/taiwan_v48_mechanism_comparison.json",
        "outputs/taiwan_v48_mechanism_summary.csv",
    ]
    manifest = {
        "model_version": "V4.8",
        "runtime_validation": v48["verification"]["runtime_validation"],
        "artifacts": {
            name: {"sha256": sha256(ROOT / name), "bytes": (ROOT / name).stat().st_size}
            for name in artifacts
        },
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print("TAIWAN_V48_SUMMARY_AND_MANIFEST: PASS")


if __name__ == "__main__":
    main()
