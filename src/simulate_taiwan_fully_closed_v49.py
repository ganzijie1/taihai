from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path

import simulate_taiwan_fully_closed_v47 as core


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
TRACE = OUT / "taiwan_v49_traceability.json"
AUDIT = OUT / "taiwan_v49_static_coupling_audit.json"
OUT_CSV = OUT / "taiwan_v49_energy_mechanism_results.csv"
OUT_JSON = OUT / "taiwan_v49_energy_mechanism_simulation.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def run(paths: int = 128, months: int = 360,
        scenario_names: list[str] | None = None, case_names: list[str] | None = None):
    return core.run(
        paths=paths,
        months=months,
        scenario_names=scenario_names,
        case_names=case_names,
        mechanism_enabled=True,
        energy_enabled=True,
        traceability_path=TRACE,
        coupling_audit_path=AUDIT,
    )


def write(rows, aggregate, diagnostics, verification) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    verification["result_sha256"] = sha256(OUT_CSV)
    verification["model_version"] = "V4.9"
    verification["mechanism_design"] = "active_same_path"
    verification["energy_network"] = "active_same_path_exact_cut_enumeration"
    OUT_JSON.write_text(json.dumps({
        "verification": verification,
        "aggregate": aggregate,
        "conditional": rows,
        "module_diagnostics": diagnostics,
    }, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    if os.environ.get("TAIWAN_ALLOW_COMPLETE_RUN") != "1":
        raise RuntimeError("V4.9 outcome simulation blocked until the complete-model gate passes")
    paths = int(os.environ.get("TAIWAN_V49_PATHS", "128"))
    months = int(os.environ.get("TAIWAN_V49_MONTHS", "360"))
    rows, aggregate, diagnostics, verification = run(paths=paths, months=months)
    write(rows, aggregate, diagnostics, verification)
    print("TAIWAN_V49_COMPLETE_RUN: PASS")


if __name__ == "__main__":
    main()
