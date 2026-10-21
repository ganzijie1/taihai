from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import simulate_taiwan_fully_closed_v82 as model
from simulate_dual_circulation_v44 import v43
from simulate_taiwan_multidomain_v42 import CASES


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
PLAN = OUT / "taiwan_v82_ablation_plan.json"
RUN_ROOT = OUT / "taiwan_v82_runs"
SCENARIOS = tuple(v43.SCENARIOS)
MODEL_VERSION = "V8.2-ACTOR-SPECIFIC-SECTOR-DEVELOPMENT-LABOR"
VARIANTS = (
    "baseline",
    "no_agriculture_deepening",
    "no_industry_deepening",
    "no_development_deepening",
    "no_labor_deepening",
    "no_statistical_field",
    "no_analytical_mechanics",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def load_plan(variant: str) -> dict:
    payload = json.loads(PLAN.read_text(encoding="utf-8"))
    if payload.get("status") != "FROZEN" or variant not in payload["variants"]:
        raise RuntimeError("V8.2 ablation plan is not frozen or is incomplete")
    for relative, expected in payload["source_hashes"].items():
        if sha256(ROOT / relative) != expected:
            raise RuntimeError(f"source changed after freeze: {relative}")
    return payload


def checkpoint_path(variant: str, i: int, j: int) -> Path:
    return RUN_ROOT / variant / "checkpoints" / f"{i:02d}_{j:02d}.json"


def valid(payload: dict, variant: str, paths: int, months: int,
          scenario: str, case: str, plan_hash: str) -> bool:
    meta = payload.get("checkpoint", {})
    return (
        meta.get("model_version") == MODEL_VERSION
        and meta.get("variant") == variant
        and meta.get("paths") == paths
        and meta.get("months") == months
        and meta.get("scenario") == scenario
        and meta.get("case") == case
        and meta.get("plan_sha256") == plan_hash
        and meta.get("freeze_sha256") == sha256(model.FREEZE)
    )


def variant_kwargs(variant: str) -> dict:
    flags = {
        "agriculture_deepening_enabled": True,
        "industry_deepening_enabled": True,
        "development_deepening_enabled": True,
        "labor_deepening_enabled": True,
    }
    os.environ.pop("TAIWAN_V81_DISABLE_STATISTICAL_FIELD", None)
    os.environ.pop("TAIWAN_V81_DISABLE_ANALYTICAL_MECHANICS", None)
    if variant == "no_agriculture_deepening":
        flags["agriculture_deepening_enabled"] = False
    elif variant == "no_industry_deepening":
        flags["industry_deepening_enabled"] = False
    elif variant == "no_development_deepening":
        flags["development_deepening_enabled"] = False
    elif variant == "no_labor_deepening":
        flags["labor_deepening_enabled"] = False
    elif variant == "no_statistical_field":
        os.environ["TAIWAN_V81_DISABLE_STATISTICAL_FIELD"] = "1"
    elif variant == "no_analytical_mechanics":
        os.environ["TAIWAN_V81_DISABLE_ANALYTICAL_MECHANICS"] = "1"
    return flags


def run_or_load(variant: str, i: int, j: int, scenario: str, case: str,
                paths: int, months: int, plan_hash: str) -> dict:
    target = checkpoint_path(variant, i, j)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        payload = json.loads(target.read_text(encoding="utf-8"))
        if valid(payload, variant, paths, months, scenario, case, plan_hash):
            print(f"TAIWAN_V82_CHECKPOINT_REUSED: {variant}:{i}:{j}", flush=True)
            return payload
    rows, aggregate, diagnostics, verification = model.run(
        paths=paths, months=months,
        scenario_names=[scenario], case_names=[case],
        **variant_kwargs(variant),
    )
    payload = {
        "checkpoint": {
            "model_version": MODEL_VERSION,
            "variant": variant,
            "paths": paths,
            "months": months,
            "scenario": scenario,
            "case": case,
            "plan_sha256": plan_hash,
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
    print(f"TAIWAN_V82_CHECKPOINT_WRITTEN: {variant}:{i}:{j}", flush=True)
    return payload


def merge_and_write(variant: str, payloads: list[dict], plan: dict) -> None:
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
    model.validate_v82(diagnostics, int(payloads[0]["checkpoint"]["months"]))
    target_dir = RUN_ROOT / variant
    csv_path = target_dir / "results.csv"
    json_path = target_dir / "simulation.json"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    verification.update({
        "calendar_2100": calendar,
        "model_version": MODEL_VERSION,
        "runtime_validation": "TAIWAN_V82_RUNTIME_VALIDATION: PASS",
        "result_sha256": sha256(csv_path),
        "checkpoint_contract": {
            "variant": variant,
            "disabled_only": plan["variants"][variant]["disabled_only"],
            "cell_count": len(payloads),
            "same_paths_seeds_and_clock": True,
            "posthoc_probability_adjustment": False,
        },
    })
    payload = {
        "verification": verification,
        "aggregate": aggregate,
        "conditional": rows,
        "module_diagnostics": diagnostics,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if variant == "baseline":
        model.write(rows, aggregate, diagnostics, verification)
    print(f"TAIWAN_V82_VARIANT_COMPLETE: {variant}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("variant", choices=VARIANTS)
    parser.add_argument("--cell", help="optional i:j single-cell diagnostic/checkpoint")
    args = parser.parse_args()
    if os.environ.get("TAIWAN_ALLOW_COMPLETE_RUN") != "1":
        raise RuntimeError("V8.2 checkpointed run requires authorization")
    plan = load_plan(args.variant)
    paths, months = int(plan["paths"]), int(plan["months"])
    plan_hash = sha256(PLAN)
    if args.cell:
        i, j = map(int, args.cell.split(":"))
        run_or_load(args.variant, i, j, SCENARIOS[i], CASES[j],
                    paths, months, plan_hash)
        return
    payloads = [
        run_or_load(args.variant, i, j, scenario, case,
                    paths, months, plan_hash)
        for i, scenario in enumerate(SCENARIOS)
        for j, case in enumerate(CASES)
    ]
    merge_and_write(args.variant, payloads, plan)


if __name__ == "__main__":
    main()
