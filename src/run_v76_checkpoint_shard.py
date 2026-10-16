from __future__ import annotations

import os

from simulate_dual_circulation_v44 import v43
from simulate_taiwan_multidomain_v42 import CASES
from simulate_taiwan_v76_checkpointed import CHECKPOINT_DIR, run_or_load


def main() -> None:
    if os.environ.get("TAIWAN_ALLOW_COMPLETE_RUN") != "1":
        raise RuntimeError("V7.6 checkpoint shard requires authorization")
    paths = int(os.environ.get("TAIWAN_V76_PATHS", "128"))
    months = int(os.environ.get("TAIWAN_V76_MONTHS", "360"))
    start = int(os.environ["TAIWAN_V76_CELL_START"])
    end = int(os.environ["TAIWAN_V76_CELL_END"])
    cells = [
        (i, j, scenario, case)
        for i, scenario in enumerate(tuple(v43.SCENARIOS))
        for j, case in enumerate(tuple(CASES))
    ]
    if not (0 <= start < end <= len(cells)):
        raise ValueError(f"invalid cell interval [{start}, {end}) for {len(cells)} cells")
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    for i, j, scenario, case in cells[start:end]:
        run_or_load(i, j, scenario, case, paths, months)
    print(f"TAIWAN_V76_SHARD_COMPLETE: {start}:{end}", flush=True)


if __name__ == "__main__":
    main()
