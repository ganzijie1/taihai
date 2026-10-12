from __future__ import annotations

import json
import os
from pathlib import Path

import simulate_taiwan_fully_closed_v59 as model


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
CHECKPOINT_DIR = OUT / "taiwan_v59_scenario_checkpoints"


def checkpoint_path(index: int, scenario: str) -> Path:
    return CHECKPOINT_DIR / f"{index:02d}_{scenario}.json"


def safe_label(value: str) -> str:
    return value.encode("unicode_escape").decode("ascii")


def valid_checkpoint(payload: dict, paths: int, months: int, scenario: str) -> bool:
    metadata = payload.get("checkpoint", {})
    return (
        metadata.get("model_version") == "V5.9"
        and metadata.get("paths") == paths
        and metadata.get("months") == months
        and metadata.get("scenario") == scenario
        and metadata.get("frozen_specification_sha256") == model.sha256(model.FREEZE)
    )


def run_or_load(index: int, scenario: str, paths: int, months: int) -> dict:
    target = checkpoint_path(index, scenario)
    if target.exists():
        payload = json.loads(target.read_text(encoding="utf-8"))
        if valid_checkpoint(payload, paths, months, scenario):
            print(f"TAIWAN_V59_CHECKPOINT_REUSED: {safe_label(scenario)}", flush=True)
            return payload
    rows, aggregate, diagnostics, verification = model.run(
        paths=paths, months=months, scenario_names=[scenario]
    )
    payload = {
        "checkpoint": {
            "model_version": "V5.9",
            "paths": paths,
            "months": months,
            "scenario": scenario,
            "frozen_specification_sha256": model.sha256(model.FREEZE),
        },
        "rows": rows,
        "aggregate": aggregate,
        "diagnostics": diagnostics,
        "verification": verification,
    }
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    temporary.replace(target)
    print(f"TAIWAN_V59_CHECKPOINT_WRITTEN: {safe_label(scenario)}", flush=True)
    return payload


def merge(payloads: list[dict]) -> tuple[list, list, dict, dict]:
    rows: list = []
    aggregate: list = []
    diagnostics: dict = {}
    verification = dict(payloads[0]["verification"])
    calendar = []
    for payload in payloads:
        rows.extend(payload["rows"])
        aggregate.extend(payload["aggregate"])
        diagnostics.update(payload["diagnostics"])
        calendar.extend(payload["verification"]["calendar_2100"])
        for key in ("intervention_probability_means", "force_onset_mean", "long_term_checkpoints"):
            if payload["verification"].get(key) != verification.get(key):
                raise RuntimeError(f"cross-scenario common state mismatch: {key}")
    verification["calendar_2100"] = calendar
    return rows, aggregate, diagnostics, verification


def main() -> None:
    if os.environ.get("TAIWAN_ALLOW_COMPLETE_RUN") != "1":
        raise RuntimeError("V5.9 checkpointed run requires the complete-model gate")
    frozen = json.loads(model.FREEZE.read_text(encoding="utf-8"))
    if frozen.get("status") != "FROZEN":
        raise RuntimeError("V5.9 frozen specification is missing or invalid")
    paths = int(os.environ.get("TAIWAN_V59_PATHS", "128"))
    months = int(os.environ.get("TAIWAN_V59_MONTHS", "360"))
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    scenarios = list(model.v58.core.trade.v43.SCENARIOS)
    payloads = [
        run_or_load(index, scenario, paths, months)
        for index, scenario in enumerate(scenarios)
    ]
    rows, aggregate, diagnostics, verification = merge(payloads)
    model.write(rows, aggregate, diagnostics, verification)
    print("TAIWAN_V59_CHECKPOINTED_COMPLETE_RUN: PASS")


if __name__ == "__main__":
    main()
