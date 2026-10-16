"""DIAGNOSTIC_ONLY: V7.3 long-horizon stress; no outcomes."""

from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np

from continuous_graph_network_v73 import ContinuousGraphOperationalNetworkV73


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "outputs" / "taiwan_v73_continuous_graph_stress.json"


def main() -> None:
    paths, months = 32, 360
    terrain = {
        "land_share_above_500m": 0.42,
        "sky_view_proxy_p10_p50_p90": [0.31, 0.58, 0.83],
    }
    active = ContinuousGraphOperationalNetworkV73(
        paths, 73_201, terrain, geometry_enabled=True, advanced_geometry=True
    )
    rng = np.random.default_rng(73_202)
    delivery, scarcity, continuum, enterprise = [], [], [], []
    start = time.perf_counter()
    for month in range(months):
        seasonal = 0.5 + 0.5 * np.sin(2.0 * np.pi * month / 18.0)
        common = rng.normal(0.0, 0.025, (paths, 1))
        idiosyncratic = rng.normal(0.0, 0.018, (paths, 4))
        damage = np.clip(0.10 + 0.22 * seasonal + common + idiosyncratic, 0.01, 0.72)
        interdiction = np.clip(0.16 + 0.30 * seasonal + 0.8 * common + idiosyncratic, 0.02, 0.82)
        active.update(**{
            "production": np.clip(0.115 - 0.050 * damage, 0.015, 0.13),
            "consumption": np.clip(0.105 + 0.045 * seasonal + 0.025 * damage, 0.02, 0.19),
            "damage": damage,
            "interdiction": interdiction,
            "command": np.clip(0.96 - 0.42 * damage, 0.18, 0.99),
            "pnt": np.clip(0.97 - 0.45 * damage, 0.16, 0.99),
            "financial_capacity": np.clip(0.96 - 0.34 * damage, 0.18, 0.99),
        })
        delivery.append(float(np.mean(active.last_delivery_ratio)))
        scarcity.append(float(np.mean(active.last_shadow_scarcity)))
        continuum.append(float(np.mean(active.last_continuous_capacity_gate)))
        enterprise.append(float(np.mean(active.last_enterprise_gate)))
    elapsed = time.perf_counter() - start
    diag = active.diagnostics()
    graph = diag["continuous_graph_network_v73"]
    inherited = diag["combinatorial_network_v71"]
    checks = {
        "monthly_clock": graph["months"] == months,
        "finite_trajectories": bool(np.all(np.isfinite(delivery + scarcity + continuum + enterprise))),
        "metric_balance": graph["maximum_metric_balance_residual"] <= 1.0e-10,
        "cfl_stable": graph["maximum_cfl"] < 1.0,
        "bounded_continuous_gates": min(continuum) > 0.0 and max(continuum) <= 1.0,
        "bounded_enterprise_gates": min(enterprise) > 0.0 and max(enterprise) <= 1.0,
        "capacity_feasible": inherited["maximum_capacity_residual"] <= 1.0e-10,
        "stock_conserved": inherited["maximum_stock_residual"] <= 1.0e-9,
        "cascade_converged": inherited["maximum_cascade_residual"] <= 1.0e-7,
        "no_probability_postprocessing": not graph["terminal_probability_adjustment"],
    }
    payload = {
        "model_version": "V7.3-CONTINUOUS-GRAPH",
        "classification": "DIAGNOSTIC_ONLY; no terminal probabilities",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "paths": paths,
        "months": months,
        "elapsed_seconds": elapsed,
        "checks": checks,
        "summary": {
            "mean_delivery_ratio": float(np.mean(delivery)),
            "mean_scarcity": float(np.mean(scarcity)),
            "mean_continuous_capacity_gate": float(np.mean(continuum)),
            "minimum_continuous_capacity_gate": float(np.min(continuum)),
            "mean_enterprise_gate": float(np.mean(enterprise)),
        },
        "continuous_graph_diagnostics": graph,
        "inherited_network_diagnostics": inherited,
    }
    TARGET.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"TAIWAN_V73_STRESS: {payload['status']} ({elapsed:.3f}s)")
    if payload["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
