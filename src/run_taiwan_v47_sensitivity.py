from __future__ import annotations

import json
import os
import time
from pathlib import Path

import simulate_taiwan_fully_closed_v47 as model


def main() -> None:
    os.environ["TAIWAN_ALLOW_COMPLETE_RUN"] = "1"
    scenario_code = os.environ["TAIWAN_V47_SCENARIO"]
    prewar_scenario = os.environ.get("TAIWAN_V47_PREWAR_SCENARIO", "central")
    paths = int(os.environ.get("TAIWAN_V47_PATHS", "24"))
    months = int(os.environ.get("TAIWAN_V47_MONTHS", "360"))
    matches = [name for name, code in model.trade.v43.SCENARIOS.items() if code == scenario_code]
    if len(matches) != 1:
        raise ValueError("select exactly one trade scenario code")
    if scenario_code == "baseline" and prewar_scenario == "central":
        raise ValueError("sensitivity must change trade or prewar scenario")
    started = time.time()
    rows, aggregate, diagnostics, verification = model.run(
        paths=paths, months=months, scenario_names=matches,
        prewar_scenario=prewar_scenario,
    )
    payload = {
        "status": "PASS",
        "scenario_code": scenario_code,
        "prewar_scenario": prewar_scenario,
        "paths": paths,
        "months": months,
        "elapsed_seconds": time.time() - started,
        "aggregate": aggregate,
        "conditional": rows,
        "verification": verification,
        "module_diagnostics": diagnostics,
    }
    destination = Path(model.OUT) / (
        f"taiwan_v47_sensitivity_{scenario_code}_{prewar_scenario}.json"
    )
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "status": "PASS", "scenario_code": scenario_code,
        "prewar_scenario": prewar_scenario,
        "elapsed_seconds": payload["elapsed_seconds"], "aggregate": aggregate,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
