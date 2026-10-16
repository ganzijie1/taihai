"""DIAGNOSTIC_ONLY: long-horizon V7.4 invariant stress test."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from diagnostic_taiwan_v72 import inputs
from finsler_network_v74 import FinslerOperationalNetworkV74


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "outputs" / "taiwan_v74_finsler_stress.json"


def main() -> None:
    paths, months = 4, 120
    terrain = {
        "land_share_above_500m": 0.42,
        "sky_view_proxy_p10_p50_p90": [0.31, 0.58, 0.83],
    }
    model = FinslerOperationalNetworkV74(
        paths, 74_301, terrain, geometry_enabled=True, advanced_geometry=True
    )
    for month in range(months):
        args = inputs(paths, month)
        if month in {18, 37, 61, 88}:
            args["damage"] = np.clip(args["damage"] + 0.30, 0.0, 1.0)
            args["interdiction"] = np.clip(args["interdiction"] + 0.25, 0.0, 1.0)
        model.update(**args)
    diag = model.diagnostics()
    finsler = diag["finsler_network_v74"]
    inherited = diag["combinatorial_network_v71"]
    continuous = diag["continuous_graph_network_v73"]
    checks = {
        "monthly_clock": finsler["months"] == months,
        "randers_strong_convexity": finsler["maximum_randers_one_form_norm"] < 1.0,
        "strictly_positive_cost": finsler["minimum_finsler_cost"] > 0.0,
        "finite_arrival": np.isfinite(finsler["mean_sink_arrival"]),
        "positive_homogeneity": finsler["maximum_positive_homogeneity_residual"] <= 1.0e-12,
        "eikonal_inequality": finsler["maximum_eikonal_residual"] <= 1.0e-10,
        "capacity": inherited["maximum_capacity_residual"] <= 1.0e-10,
        "stock": inherited["maximum_stock_residual"] <= 1.0e-9,
        "finite_volume": continuous["maximum_metric_balance_residual"] <= 1.0e-10,
        "pipeline_nonnegative": inherited["maximum_pipeline_negative_mass"] <= 1.0e-12,
        "no_posthoc": not finsler["terminal_probability_adjustment"],
    }
    checks = {name: bool(value) for name, value in checks.items()}
    payload = {
        "model_version": "V7.4-FINSLER-CLOSED-LOOP",
        "classification": "DIAGNOSTIC_ONLY; no terminal probabilities",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "diagnostics": finsler,
        "inherited_network": inherited,
        "continuous_graph": continuous,
    }
    TARGET.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"TAIWAN_V74_STRESS: {payload['status']}")
    if payload["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
