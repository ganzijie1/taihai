from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import simulate_taiwan_fully_closed_v81 as model
from simulate_dual_circulation_v44 import v43
from simulate_taiwan_multidomain_v42 import CASES


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
PLAN = OUT / "taiwan_v81_inherited_ablation_plan.json"
ABLATION_ROOT = OUT / "taiwan_v81_inherited_ablations"
SCENARIOS = tuple(v43.SCENARIOS)
MODEL_VERSION = "V8.1-INHERITED-ONE-AT-A-TIME-ABLATION"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def load_plan(ablation: str) -> dict:
    payload = json.loads(PLAN.read_text(encoding="utf-8"))
    if payload.get("status") != "FROZEN" or ablation not in payload["ablations"]:
        raise RuntimeError("inherited ablation plan is not frozen or is incomplete")
    for relative, expected in payload["source_hashes"].items():
        if sha256(ROOT / relative) != expected:
            raise RuntimeError(f"source changed after freeze: {relative}")
    return payload


def checkpoint_path(ablation: str, i: int, j: int) -> Path:
    return ABLATION_ROOT / ablation / "checkpoints" / f"{i:02d}_{j:02d}.json"


def valid(
    payload: dict, *, ablation: str, paths: int, months: int,
    scenario: str, case: str, plan_sha256: str,
) -> bool:
    meta = payload.get("checkpoint", {})
    return (
        meta.get("model_version") == MODEL_VERSION
        and meta.get("ablation") == ablation
        and meta.get("paths") == paths
        and meta.get("months") == months
        and meta.get("scenario") == scenario
        and meta.get("case") == case
        and meta.get("plan_sha256") == plan_sha256
        and meta.get("freeze_sha256") == sha256(model.FREEZE)
    )


def configure_environment(plan: dict, ablation: str) -> None:
    for key, value in plan["ablations"][ablation]["environment"].items():
        os.environ[key] = value


def run_or_load(
    ablation: str, i: int, j: int, scenario: str, case: str,
    paths: int, months: int, plan_sha256: str,
) -> dict:
    target = checkpoint_path(ablation, i, j)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        payload = json.loads(target.read_text(encoding="utf-8"))
        if valid(
            payload, ablation=ablation, paths=paths, months=months,
            scenario=scenario, case=case, plan_sha256=plan_sha256,
        ):
            print(f"TAIWAN_V81_ABLATION_REUSED: {ablation}:{i}:{j}", flush=True)
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
            "ablation": ablation,
            "paths": paths,
            "months": months,
            "scenario": scenario,
            "case": case,
            "plan_sha256": plan_sha256,
            "freeze_sha256": sha256(model.FREEZE),
        },
        "rows": rows,
        "aggregate": aggregate,
        "diagnostics": diagnostics,
        "verification": verification,
    }
    temporary = target.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    temporary.replace(target)
    print(f"TAIWAN_V81_ABLATION_WRITTEN: {ablation}:{i}:{j}", flush=True)
    return payload


def load_all(
    ablation: str, paths: int, months: int, plan_sha256: str
) -> list[dict]:
    payloads = []
    missing = []
    for i, scenario in enumerate(SCENARIOS):
        for j, case in enumerate(CASES):
            target = checkpoint_path(ablation, i, j)
            if not target.exists():
                missing.append(f"{i}:{j}:missing")
                continue
            payload = json.loads(target.read_text(encoding="utf-8"))
            if valid(
                payload, ablation=ablation, paths=paths, months=months,
                scenario=scenario, case=case, plan_sha256=plan_sha256,
            ):
                payloads.append(payload)
            else:
                missing.append(f"{i}:{j}:invalid")
    if missing:
        raise RuntimeError("ablation checkpoints incomplete: " + ", ".join(missing))
    return payloads


def merge_and_write(ablation: str, payloads: list[dict], plan: dict) -> None:
    rows, aggregate, diagnostics, calendar = [], [], {}, []
    verification = dict(payloads[0]["verification"])
    for payload in payloads:
        rows.extend(payload["rows"])
        meta = payload["checkpoint"]
        for source in payload["aggregate"]:
            item = dict(source)
            item["trade_scenario"] = meta["scenario"]
            item["case"] = meta["case"]
            aggregate.append(item)
        diagnostics.update(payload["diagnostics"])
        calendar.extend(payload["verification"].get("calendar_2100", []))
    model.validate_v81(diagnostics, int(payloads[0]["checkpoint"]["months"]))
    target_dir = ABLATION_ROOT / ablation
    csv_path = target_dir / "results.csv"
    json_path = target_dir / "simulation.json"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    verification["calendar_2100"] = calendar
    verification["model_version"] = MODEL_VERSION
    verification["runtime_validation"] = "TAIWAN_V81_ABLATION_RUNTIME_VALIDATION: PASS"
    verification["result_sha256"] = sha256(csv_path)
    verification["ablation_contract"] = {
        "ablation": ablation,
        "disabled_only": plan["ablations"][ablation]["disabled_only"],
        "cell_count": len(payloads),
        "plan_sha256": sha256(PLAN),
        "same_paths_seeds_and_clock": True,
        "posthoc_probability_adjustment": False,
    }
    json_path.write_text(json.dumps({
        "verification": verification,
        "aggregate": aggregate,
        "conditional": rows,
        "module_diagnostics": diagnostics,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"TAIWAN_V81_ABLATION_COMPLETE: {ablation}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "ablation", choices=("no_statistical_field", "no_analytical_mechanics")
    )
    args = parser.parse_args()
    if os.environ.get("TAIWAN_ALLOW_COMPLETE_RUN") != "1":
        raise RuntimeError("inherited outcome ablations require complete-run authorization")
    plan = load_plan(args.ablation)
    configure_environment(plan, args.ablation)
    paths = int(plan["baseline"]["paths"])
    months = int(plan["baseline"]["months"])
    plan_hash = sha256(PLAN)
    for i, scenario in enumerate(SCENARIOS):
        for j, case in enumerate(CASES):
            run_or_load(
                args.ablation, i, j, scenario, case,
                paths, months, plan_hash,
            )
    merge_and_write(
        args.ablation, load_all(args.ablation, paths, months, plan_hash), plan
    )


if __name__ == "__main__":
    main()
