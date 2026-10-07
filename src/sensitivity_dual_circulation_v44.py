from __future__ import annotations

import json
from pathlib import Path

import simulate_dual_circulation_v44 as model


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "台海V4.4双循环_政治经济社会反馈敏感性.json"


def main() -> None:
    data = model.build_nested_accounts()
    network = model.build_network(data)
    payload = {
        "paths_per_variant": 300,
        "months": 360,
        "seed": 20261004,
        "variants": {},
    }
    for scale in (0.0, 0.5, 1.5):
        _, summary = model.simulate(
            network,
            paths=300,
            months=360,
            seed=20261004,
            feedback_scale=scale,
        )
        payload["variants"][str(scale)] = {
            scenario: {
                horizon: values
                for horizon, values in horizons.items()
                if horizon in {"10年", "30年"}
            }
            for scenario, horizons in summary.items()
        }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("DUAL_CIRCULATION_V44_SENSITIVITY: PASS")
    print(OUT)


if __name__ == "__main__":
    main()
