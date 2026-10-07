from __future__ import annotations

import csv
import hashlib
import json
import platform
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
V49 = OUT / "taiwan_v49_energy_mechanism_simulation.json"
V50 = OUT / "taiwan_v50_geometry_simulation.json"
COMPARISON = OUT / "taiwan_v50_vs_v49_comparison.json"
SUMMARY = OUT / "taiwan_v50_summary.csv"
MANIFEST = OUT / "taiwan_v50_reproducibility_manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def record(path: Path) -> dict:
    return {
        "path": str(path.relative_to(ROOT)),
        "exists": path.exists(),
        "bytes": path.stat().st_size if path.exists() else None,
        "sha256": sha256(path) if path.exists() else None,
    }


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
        "by_case_30y": {
            row["case"]: row["conditional_broad_control_rate"] for row in conditional
        },
        "minimum_ess": min(
            row["importance_sampling_effective_sample_size"] for row in conditional
        ),
    }


def main() -> None:
    old_data = json.loads(V49.read_text(encoding="utf-8"))
    new_data = json.loads(V50.read_text(encoding="utf-8"))
    old = central(old_data)
    new = central(new_data)
    delta = {
        "conditional_broad_control_30y": (
            new["campaign_30y"]["conditional_mean"]
            - old["campaign_30y"]["conditional_mean"]
        ),
        "conditional_broad_control_by_2100": (
            new["calendar_2100"]["conditional_broad_control_by_2100"]
            - old["calendar_2100"]["conditional_broad_control_by_2100"]
        ),
        "joint_force_and_control_by_2100": (
            new["calendar_2100"]["joint_force_and_broad_control_by_2100"]
            - old["calendar_2100"]["joint_force_and_broad_control_by_2100"]
        ),
    }
    COMPARISON.write_text(json.dumps({
        "classification": "same-seed structural version comparison; geometry effect is not a real-world causal estimate",
        "v49": old,
        "v50": new,
        "v50_minus_v49": delta,
        "geometry_ablation": json.loads(
            (OUT / "taiwan_v50_geometry_ablation.json").read_text(encoding="utf-8")
        ),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    rows = []
    for version, block in (("V4.9", old), ("V5.0", new)):
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
    with SUMMARY.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    code = [
        "work/continuous_multipop_mfg_v47.py",
        "work/operational_constraints_v47.py",
        "work/spatial_control_network_v47.py",
        "work/four_party_energy_network_v49.py",
        "work/geometric_multiscale_v50.py",
        "work/simulate_taiwan_multidomain_v42.py",
        "work/simulate_taiwan_fully_closed_v47.py",
        "work/simulate_taiwan_fully_closed_v50.py",
        "work/audit_taiwan_v50.py",
        "work/diagnostic_taiwan_v50.py",
        "work/ablate_taiwan_geometry_v50.py",
        "work/summarize_taiwan_v50.py"
    ]
    inputs = [
        "work/SIPRI-Milex-data-1949-2025_v1.2.xlsx",
        "outputs/taiwan_v50_traceability.json",
        "outputs/taiwan_v50_static_coupling_audit.json",
        "outputs/taiwan_v50_frozen_specification.json",
        "outputs/taiwan_v50_geometry_ablation.json",
    ]
    results = [
        "outputs/taiwan_v50_geometry_results.csv",
        "outputs/taiwan_v50_geometry_simulation.json",
        "outputs/taiwan_v50_vs_v49_comparison.json",
        "outputs/taiwan_v50_summary.csv",
        "outputs/taiwan_v50_formal_run.stdout.log",
        "outputs/taiwan_v50_formal_run.stderr.log",
        "outputs/台海V5.0几何动力学闭合仿真_飞书增补.md",
    ]
    manifest = {
        "model_version": "V5.0",
        "runtime_validation": new_data["verification"]["runtime_validation"],
        "seed_contract": {
            "campaign_seed_symbol": "simulate_taiwan_multidomain_v42.SEED",
            "common_random_numbers": True,
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
        "code": [record(ROOT / path) for path in code],
        "inputs": [record(ROOT / path) for path in inputs],
        "results": [record(ROOT / path) for path in results],
    }
    missing = [
        row["path"] for group in ("code", "inputs", "results")
        for row in manifest[group] if not row["exists"]
    ]
    manifest["status"] = "PASS" if not missing else "FAIL"
    manifest["missing_required"] = missing
    MANIFEST.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"TAIWAN_V50_SUMMARY_AND_MANIFEST: {manifest['status']}")
    if missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
