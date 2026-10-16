from __future__ import annotations

import json
import os
from pathlib import Path

import simulate_taiwan_fully_closed_v69 as model
from simulate_dual_circulation_v44 import v43


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
CHECKPOINT_DIR = OUT / "taiwan_v69_scenario_checkpoints"
SCENARIOS = tuple(v43.SCENARIOS)


def checkpoint_path(index: int, scenario: str) -> Path:
    return CHECKPOINT_DIR / f"{index:02d}_{scenario}.json"


def valid(payload: dict, paths: int, months: int, scenario: str) -> bool:
    meta = payload.get("checkpoint", {})
    return (
        meta.get("model_version") == "V6.9-OT-SYM-ACTIVE"
        and meta.get("paths") == paths and meta.get("months") == months
        and meta.get("scenario") == scenario
        and meta.get("freeze_sha256") == model.sha256(model.FREEZE)
    )


def run_or_load(index: int, scenario: str, paths: int, months: int) -> dict:
    target = checkpoint_path(index, scenario)
    if target.exists():
        payload = json.loads(target.read_text(encoding="utf-8"))
        if valid(payload, paths, months, scenario):
            print(f"TAIWAN_V69_CHECKPOINT_REUSED: {index}", flush=True)
            return payload
    rows, aggregate, diagnostics, verification = model.run(
        paths=paths, months=months, scenario_names=[scenario]
    )
    payload = {
        "checkpoint": {
            "model_version": "V6.9-OT-SYM-ACTIVE", "paths": paths,
            "months": months, "scenario": scenario,
            "freeze_sha256": model.sha256(model.FREEZE),
        },
        "rows": rows, "aggregate": aggregate,
        "diagnostics": diagnostics, "verification": verification,
    }
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    temporary.replace(target)
    print(f"TAIWAN_V69_CHECKPOINT_WRITTEN: {index}", flush=True)
    return payload


def merge(payloads: list[dict]):
    rows, aggregate, calendar = [], [], []
    diagnostics = {}
    verification = dict(payloads[0]["verification"])
    for payload in payloads:
        rows.extend(payload["rows"]); aggregate.extend(payload["aggregate"])
        diagnostics.update(payload["diagnostics"])
        calendar.extend(payload["verification"]["calendar_2100"])
    verification["calendar_2100"] = calendar
    verification["functional_analysis"] = {
        "scope": "read_only_validation", "cell_count": len(diagnostics),
        "all_cells_passed": all(c["functional_analysis"]["passed"] for c in diagnostics.values()),
        "probability_adjustment": False,
        "same_outcomes_as_v67_by_construction": False,
    }
    return rows, aggregate, diagnostics, verification


def main() -> None:
    if os.environ.get("TAIWAN_ALLOW_COMPLETE_RUN") != "1":
        raise RuntimeError("V6.9 checkpointed run requires the complete-model gate")
    frozen = json.loads(model.FREEZE.read_text(encoding="utf-8"))
    if frozen.get("status") != "FROZEN":
        raise RuntimeError("V6.9 frozen specification is missing or invalid")
    paths = int(os.environ.get("TAIWAN_V69_PATHS", "128"))
    months = int(os.environ.get("TAIWAN_V69_MONTHS", "360"))
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    payloads = [run_or_load(i, s, paths, months) for i, s in enumerate(SCENARIOS)]
    model.write(*merge(payloads))
    print("TAIWAN_V69_CHECKPOINTED_COMPLETE_RUN: PASS")


if __name__ == "__main__":
    main()
