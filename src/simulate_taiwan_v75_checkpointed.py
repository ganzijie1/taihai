from __future__ import annotations

import json
import os
from pathlib import Path

import simulate_taiwan_fully_closed_v75 as model
from simulate_dual_circulation_v44 import v43
from simulate_taiwan_multidomain_v42 import CASES


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
CHECKPOINT_DIR = OUT / "taiwan_v75_scenario_checkpoints"
SCENARIOS = tuple(v43.SCENARIOS)
MODEL_VERSION = "V7.5-HYPERGRAPH-MINSKY-FINANCE"


def checkpoint_path(scenario_index: int, case_index: int) -> Path:
    return CHECKPOINT_DIR / f"{scenario_index:02d}_{case_index:02d}.json"


def valid(payload: dict, paths: int, months: int, scenario: str, case: str) -> bool:
    meta = payload.get("checkpoint", {})
    return (
        meta.get("model_version") == MODEL_VERSION
        and meta.get("paths") == paths
        and meta.get("months") == months
        and meta.get("scenario") == scenario
        and meta.get("case") == case
        and meta.get("freeze_sha256") == model.sha256(model.FREEZE)
    )


def run_or_load(i: int, j: int, scenario: str, case: str, paths: int, months: int) -> dict:
    target = checkpoint_path(i, j)
    if target.exists():
        payload = json.loads(target.read_text(encoding="utf-8"))
        if valid(payload, paths, months, scenario, case):
            print(f"TAIWAN_V75_CHECKPOINT_REUSED: {i}:{j}", flush=True)
            return payload
    rows, aggregate, diagnostics, verification = model.run(
        paths=paths,
        months=months,
        scenario_names=[scenario],
        case_names=[case],
    )
    payload = {
        "checkpoint": {
            "model_version": MODEL_VERSION,
            "paths": paths,
            "months": months,
            "scenario": scenario,
            "case": case,
            "freeze_sha256": model.sha256(model.FREEZE),
        },
        "rows": rows,
        "aggregate": aggregate,
        "diagnostics": diagnostics,
        "verification": verification,
    }
    temporary = target.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    temporary.replace(target)
    print(f"TAIWAN_V75_CHECKPOINT_WRITTEN: {i}:{j}", flush=True)
    return payload


def load_all(paths: int, months: int) -> list[dict]:
    payloads = []
    missing = []
    for i, scenario in enumerate(SCENARIOS):
        for j, case in enumerate(CASES):
            target = checkpoint_path(i, j)
            if not target.exists():
                missing.append(f"{i}:{j}")
                continue
            payload = json.loads(target.read_text(encoding="utf-8"))
            if not valid(payload, paths, months, scenario, case):
                missing.append(f"{i}:{j}:invalid")
                continue
            payloads.append(payload)
    if missing:
        raise RuntimeError("V7.5 checkpoints incomplete: " + ", ".join(missing))
    return payloads


def merge(payloads: list[dict]):
    rows, aggregate, calendar, diagnostics = [], [], [], {}
    verification = dict(payloads[0]["verification"])
    for payload in payloads:
        rows.extend(payload["rows"])
        aggregate.extend(payload["aggregate"])
        diagnostics.update(payload["diagnostics"])
        calendar.extend(payload["verification"].get("calendar_2100", []))
    verification["calendar_2100"] = calendar
    verification["checkpoint_contract"] = {
        "atomic_per_scenario_case": True,
        "cell_count": len(payloads),
        "freeze_hash_checked": True,
        "posthoc_probability_adjustment": False,
    }
    model.validate_v75(diagnostics, int(payloads[0]["checkpoint"]["months"]))
    return rows, aggregate, diagnostics, verification


def main() -> None:
    if os.environ.get("TAIWAN_ALLOW_COMPLETE_RUN") != "1":
        raise RuntimeError("V7.5 checkpointed run requires explicit authorization")
    passed, gaps = model.complete_model_gate()
    if not passed:
        raise RuntimeError("V7.5 complete-model gate failed: " + "; ".join(gaps))
    paths = int(os.environ.get("TAIWAN_V75_PATHS", "128"))
    months = int(os.environ.get("TAIWAN_V75_MONTHS", "360"))
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    payloads = [
        run_or_load(i, j, scenario, case, paths, months)
        for i, scenario in enumerate(SCENARIOS)
        for j, case in enumerate(CASES)
    ]
    model.write(*merge(payloads))
    print("TAIWAN_V75_CHECKPOINTED_COMPLETE_RUN: PASS")


if __name__ == "__main__":
    main()
