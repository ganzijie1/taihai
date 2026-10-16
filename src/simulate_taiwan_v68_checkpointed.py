from __future__ import annotations

import json
import os
from pathlib import Path

import simulate_taiwan_fully_closed_v68 as model
from simulate_taiwan_multidomain_v42 import SCENARIOS


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
CHECKPOINT_DIR = OUT / "taiwan_v68_scenario_checkpoints"


def checkpoint_path(index: int, scenario: str) -> Path:
    return CHECKPOINT_DIR / f"{index:02d}_{scenario}.json"


def valid(payload: dict, paths: int, months: int, scenario: str) -> bool:
    meta = payload.get("checkpoint", {})
    return (
        meta.get("model_version") == "V6.8-FA-GPU"
        and meta.get("paths") == paths
        and meta.get("months") == months
        and meta.get("scenario") == scenario
        and meta.get("freeze_sha256") == model.sha256(model.FREEZE)
    )


def run_or_load(index: int, scenario: str, paths: int, months: int) -> dict:
    target = checkpoint_path(index, scenario)
    if target.exists():
        payload = json.loads(target.read_text(encoding="utf-8"))
        if valid(payload, paths, months, scenario):
            print(f"TAIWAN_V68_CHECKPOINT_REUSED: {index}", flush=True)
            return payload
    rows, aggregate, diagnostics, verification = model.run(
        paths=paths, months=months, scenario_names=[scenario]
    )
    payload = {
        "checkpoint": {
            "model_version": "V6.8-FA-GPU",
            "paths": paths,
            "months": months,
            "scenario": scenario,
            "freeze_sha256": model.sha256(model.FREEZE),
        },
        "rows": rows,
        "aggregate": aggregate,
        "diagnostics": diagnostics,
        "verification": verification,
    }
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    temporary.replace(target)
    print(f"TAIWAN_V68_CHECKPOINT_WRITTEN: {index}", flush=True)
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
        for key in ("intervention_probability_means", "force_onset_mean", "long_term_checkpoints"):
            if payload["verification"].get(key) != verification.get(key):
                raise RuntimeError(f"cross-scenario common state mismatch: {key}")
    verification["calendar_2100"] = calendar
    verification["functional_analysis"] = {
        "scope": "read_only_validation",
        "cell_count": len(diagnostics),
        "all_cells_passed": all(
            cell["functional_analysis"]["passed"] for cell in diagnostics.values()
        ),
        "probability_adjustment": False,
        "same_outcomes_as_v67_by_construction": True,
    }
    return rows, aggregate, diagnostics, verification


def main() -> None:
    if os.environ.get("TAIWAN_ALLOW_COMPLETE_RUN") != "1":
        raise RuntimeError("V6.8 checkpointed run requires the complete-model gate")
    frozen = json.loads(model.FREEZE.read_text(encoding="utf-8"))
    if frozen.get("status") != "FROZEN":
        raise RuntimeError("V6.8 frozen specification is missing or invalid")
    paths = int(os.environ.get("TAIWAN_V68_PATHS", "128"))
    months = int(os.environ.get("TAIWAN_V68_MONTHS", "360"))
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    payloads = [
        run_or_load(index, scenario, paths, months)
        for index, scenario in enumerate(SCENARIOS)
    ]
    model.write(*merge(payloads))
    print("TAIWAN_V68_CHECKPOINTED_COMPLETE_RUN: PASS")


if __name__ == "__main__":
    main()

