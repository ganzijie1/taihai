from __future__ import annotations

import json
import os
from pathlib import Path

import simulate_taiwan_fully_closed_v70 as model
from simulate_dual_circulation_v44 import v43
from simulate_taiwan_multidomain_v42 import CASES


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
CHECKPOINT_DIR = OUT / "taiwan_v70_scenario_checkpoints"
SCENARIOS = tuple(v43.SCENARIOS)


def checkpoint_path(scenario_index: int, case_index: int) -> Path:
    return CHECKPOINT_DIR / f"{scenario_index:02d}_{case_index:02d}.json"


def valid(payload: dict, paths: int, months: int, scenario: str,
          case: str) -> bool:
    meta = payload.get("checkpoint", {})
    return (
        meta.get("model_version") == "V7.0-TINBERGEN-ALGEBRAIC"
        and meta.get("paths") == paths and meta.get("months") == months
        and meta.get("scenario") == scenario
        and meta.get("case") == case
        and meta.get("freeze_sha256") == model.sha256(model.FREEZE)
    )


def run_or_load(scenario_index: int, case_index: int, scenario: str,
                case: str, paths: int, months: int) -> dict:
    target = checkpoint_path(scenario_index, case_index)
    if target.exists():
        payload = json.loads(target.read_text(encoding="utf-8"))
        if valid(payload, paths, months, scenario, case):
            print(
                f"TAIWAN_V70_CHECKPOINT_REUSED: {scenario_index}:{case_index}",
                flush=True,
            )
            return payload
    rows, aggregate, diagnostics, verification = model.run(
        paths=paths, months=months, scenario_names=[scenario],
        case_names=[case],
    )
    payload = {
        "checkpoint": {
            "model_version": "V7.0-TINBERGEN-ALGEBRAIC",
            "paths": paths, "months": months, "scenario": scenario,
            "case": case,
            "freeze_sha256": model.sha256(model.FREEZE),
        },
        "rows": rows, "aggregate": aggregate,
        "diagnostics": diagnostics, "verification": verification,
    }
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    temporary.replace(target)
    print(
        f"TAIWAN_V70_CHECKPOINT_WRITTEN: {scenario_index}:{case_index}",
        flush=True,
    )
    return payload


def merge(payloads: list[dict]):
    rows, aggregate, calendar = [], [], []
    diagnostics = {}
    verification = dict(payloads[0]["verification"])
    for payload in payloads:
        rows.extend(payload["rows"])
        aggregate.extend(payload["aggregate"])
        diagnostics.update(payload["diagnostics"])
        calendar.extend(payload["verification"]["calendar_2100"])
    verification["calendar_2100"] = calendar
    verification["functional_analysis"] = {
        "scope": "read_only_validation", "cell_count": len(diagnostics),
        "all_cells_passed": all(
            cell["functional_analysis"]["passed"]
            for cell in diagnostics.values()
        ),
        "probability_adjustment": False,
        "same_outcomes_as_v67_by_construction": False,
    }
    return rows, aggregate, diagnostics, verification


def main() -> None:
    if os.environ.get("TAIWAN_ALLOW_COMPLETE_RUN") != "1":
        raise RuntimeError("V7.0 checkpointed run requires the complete-model gate")
    frozen = json.loads(model.FREEZE.read_text(encoding="utf-8"))
    if frozen.get("status") != "FROZEN":
        raise RuntimeError("V7.0 frozen specification is missing or invalid")
    paths = int(os.environ.get("TAIWAN_V70_PATHS", "128"))
    months = int(os.environ.get("TAIWAN_V70_MONTHS", "360"))
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    payloads = [
        run_or_load(scenario_index, case_index, scenario, case, paths, months)
        for scenario_index, scenario in enumerate(SCENARIOS)
        for case_index, case in enumerate(CASES)
    ]
    model.write(*merge(payloads))
    print("TAIWAN_V70_CHECKPOINTED_COMPLETE_RUN: PASS")


if __name__ == "__main__":
    main()
