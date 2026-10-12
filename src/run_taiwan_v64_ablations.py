from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path

import simulate_taiwan_fully_closed_v64 as v64


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
TARGET_JSON = OUT / "taiwan_v64_stochastic_geometry_ablations.json"
TARGET_CSV = OUT / "taiwan_v64_stochastic_geometry_ablations.csv"
VARIANTS = {
    "without_random_field": {
        "random_field_enabled": False, "cascade_enabled": True,
        "feedback_enabled": True,
    },
    "without_cascade": {
        "random_field_enabled": True, "cascade_enabled": False,
        "feedback_enabled": True,
    },
    "read_only_geometry": {
        "random_field_enabled": True, "cascade_enabled": True,
        "feedback_enabled": False,
    },
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main() -> None:
    if os.environ.get("TAIWAN_ALLOW_COMPLETE_RUN") != "1":
        raise RuntimeError("formal ablations require the complete-run authorization flag")
    freeze = json.loads((OUT / "taiwan_v64_frozen_specification.json").read_text(encoding="utf-8"))
    if freeze.get("status") != "FROZEN":
        raise RuntimeError("V6.4 specification is not frozen")
    payload = {
        "classification": "pre-registered structural outcome ablations",
        "paths": 128, "months": 360, "trade_scenarios": ["central"],
        "cases": ["timely_full", "limited", "delayed", "japan_only", "taiwan_alone"],
        "variants": {},
    }
    csv_rows = []
    for name, flags in VARIANTS.items():
        rows, aggregate, diagnostics, verification = v64.run(
            paths=128, months=360, scenario_names=["central"], **flags
        )
        payload["variants"][name] = {
            "conditional": rows, "aggregate": aggregate,
            "verification": verification,
            "geometry_diagnostics": {
                cell: item["stochastic_geometry"] for cell, item in diagnostics.items()
            },
        }
        for row in rows:
            csv_rows.append({"variant": name, **row})
    with TARGET_CSV.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader(); writer.writerows(csv_rows)
    payload["csv_sha256"] = sha256(TARGET_CSV)
    payload["status"] = "PASS"
    TARGET_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("TAIWAN_V64_STOCHASTIC_GEOMETRY_ABLATIONS: PASS")


if __name__ == "__main__":
    main()
