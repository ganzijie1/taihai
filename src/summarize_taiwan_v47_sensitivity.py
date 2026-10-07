from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"


def main():
    rows = []
    for path in sorted(OUT.glob("taiwan_v47_sensitivity_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "PASS":
            raise RuntimeError(f"non-PASS sensitivity file: {path.name}")
        thirty = next(row for row in payload["aggregate"] if row["horizon_years"] == 30)
        terminal = payload["verification"]["calendar_2100"][0]
        total = sum(terminal[key] for key in (
            "force_onset_cif", "peace_arrangement_cif", "legal_separation_cif",
            "status_quo_survival",
        ))
        if abs(total - 1.0) > 1e-9:
            raise RuntimeError(f"competing-risk simplex failed: {path.name}: {total}")
        rows.append({
            "scenario_code": payload["scenario_code"],
            "prewar_scenario": payload["prewar_scenario"],
            "paths": payload["paths"],
            "conditional_broad_control_30y": thirty["conditional_mean"],
            "joint_force_broad_control_30y": thirty["unconditional_mean"],
            "force_by_2100": terminal["force_onset_cif"],
            "peace_by_2100": terminal["peace_arrangement_cif"],
            "separation_by_2100": terminal["legal_separation_cif"],
            "status_quo_by_2100": terminal["status_quo_survival"],
            "runtime_validation": payload["verification"]["runtime_validation"],
            "source_file": path.name,
        })
    if len(rows) != 11:
        raise RuntimeError(f"expected 11 sensitivity cells, found {len(rows)}")
    destination = OUT / "taiwan_v47_sensitivity_summary.csv"
    with destination.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("TAIWAN_V47_SENSITIVITY_SUMMARY: PASS; cells=11")


if __name__ == "__main__":
    main()
