"""DIAGNOSTIC_ONLY: long-horizon operational stress test, no outcome probabilities."""

from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np

from algebraic_discrete_network_v72 import AlgebraicDiscreteOperationalNetworkV72
from combinatorial_network_v71 import CombinatorialOperationalNetworkV71


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "outputs" / "taiwan_v72_operational_stress.json"


def main() -> None:
    paths, months = 64, 120
    terrain = {
        "land_share_above_500m": 0.42,
        "sky_view_proxy_p10_p50_p90": [0.31, 0.58, 0.83],
    }
    active = AlgebraicDiscreteOperationalNetworkV72(
        paths, 72_201, terrain, geometry_enabled=True, advanced_geometry=True
    )
    ablated = CombinatorialOperationalNetworkV71(
        paths, 72_201, terrain, geometry_enabled=True, advanced_geometry=True
    )
    rng = np.random.default_rng(72_202)
    delivery, shortage, graph_gate = [], [], []
    start = time.perf_counter()
    for month in range(months):
        seasonal = 0.5 + 0.5 * np.sin(2.0 * np.pi * month / 18.0)
        common = rng.normal(0.0, 0.025, (paths, 1))
        idiosyncratic = rng.normal(0.0, 0.018, (paths, 4))
        damage = np.clip(0.10 + 0.22 * seasonal + common + idiosyncratic, 0.01, 0.72)
        interdiction = np.clip(0.16 + 0.30 * seasonal + 0.8 * common + idiosyncratic, 0.02, 0.82)
        args = {
            "production": np.clip(0.115 - 0.050 * damage, 0.015, 0.13),
            "consumption": np.clip(0.105 + 0.045 * seasonal + 0.025 * damage, 0.02, 0.19),
            "damage": damage,
            "interdiction": interdiction,
            "command": np.clip(0.96 - 0.42 * damage, 0.18, 0.99),
            "pnt": np.clip(0.97 - 0.45 * damage, 0.16, 0.99),
            "financial_capacity": np.clip(0.96 - 0.34 * damage, 0.18, 0.99),
        }
        active.update(**args)
        ablated.update(**args)
        delivery.append(float(np.mean(active.last_delivery_ratio)))
        shortage.append(float(np.mean(active.last_shadow_scarcity)))
        graph_gate.append(float(np.mean(active.last_graph_gate)))
    elapsed = time.perf_counter() - start
    diag = active.diagnostics()
    graph = diag["algebraic_discrete_network_v72"]
    network = diag["combinatorial_network_v71"]
    checks = {
        "monthly_clock": graph["months"] == months,
        "finite_delivery": bool(np.all(np.isfinite(delivery))),
        "finite_shortage": bool(np.all(np.isfinite(shortage))),
        "capacity_feasible": network["maximum_capacity_residual"] <= 1.0e-10,
        "stock_conserved": network["maximum_stock_residual"] <= 1.0e-9,
        "cascade_converged": (
            network["maximum_cascade_residual"] <= 1.0e-7
            and network["cascade_nonconvergence_months"] == 0
        ),
        "spectral_interlacing": graph["maximum_interlacing_violation"] <= 1.0e-9,
        "protection_feasible": (
            graph["maximum_protection_partition_violation"] == 0
            and graph["protection_conflict_violations"] == 0
        ),
        "bounded_graph_gate": min(graph_gate) >= 0.70 and max(graph_gate) <= 1.0,
        "no_probability_postprocessing": not graph["terminal_probability_adjustment"],
    }
    payload = {
        "model_version": "V7.2-ALGEBRAIC-DISCRETE-GRAPH",
        "classification": "DIAGNOSTIC_ONLY; no terminal outcome probabilities",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "paths": paths,
        "months": months,
        "elapsed_seconds": elapsed,
        "checks": checks,
        "operational_summary": {
            "mean_delivery_ratio": float(np.mean(delivery)),
            "minimum_monthly_delivery_ratio": float(np.min(delivery)),
            "mean_shadow_scarcity": float(np.mean(shortage)),
            "maximum_monthly_shadow_scarcity": float(np.max(shortage)),
            "mean_graph_gate": float(np.mean(graph_gate)),
            "minimum_graph_gate": float(np.min(graph_gate)),
        },
        "graph_diagnostics": graph,
        "network_diagnostics": network,
    }
    TARGET.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"TAIWAN_V72_OPERATIONAL_STRESS: {payload['status']} ({elapsed:.3f}s)")
    if payload["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
