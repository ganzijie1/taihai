from __future__ import annotations

import json
import os
import time

import simulate_taiwan_fully_closed_v47 as model


def main() -> None:
    os.environ["TAIWAN_ALLOW_COMPLETE_RUN"] = "1"
    paths = int(os.environ.get("TAIWAN_V47_PATHS", "128"))
    baseline = next(
        name for name, code in model.trade.v43.SCENARIOS.items() if code == "baseline"
    )
    started = time.time()
    rows, aggregate, diagnostics, verification = model.run(
        paths=paths, months=360, scenario_names=[baseline]
    )
    verification["formal_run"] = {
        "paths": paths,
        "months": 360,
        "long_term_calendar": [2026, 2100],
        "prewar_scenario": "central",
        "trade_scenarios": ["baseline"],
        "elapsed_seconds": time.time() - started,
    }
    model.write(rows, aggregate, diagnostics, verification)
    print(json.dumps({
        "status": "PASS",
        "elapsed_seconds": verification["formal_run"]["elapsed_seconds"],
        "aggregate": aggregate,
        "verification": verification,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
