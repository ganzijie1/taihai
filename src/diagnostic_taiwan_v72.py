"""DIAGNOSTIC_ONLY: V7.2 graph invariants; no outcome probabilities."""

from __future__ import annotations

from itertools import combinations
import json
from pathlib import Path

import numpy as np

from algebraic_discrete_network_v72 import (
    AlgebraicDiscreteOperationalNetworkV72,
    EDGE_HEAD,
    EDGE_TAIL,
    NODE_COUNT,
    SINK,
    TREEWIDTH,
)
from combinatorial_network_v71 import CombinatorialOperationalNetworkV71


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "outputs" / "taiwan_v72_algebraic_discrete_diagnostic.json"


def inputs(paths: int, month: int) -> dict:
    phase = (month + 1) / 24.0
    damage = np.repeat(
        np.clip(np.array([[0.08, 0.16, 0.06, 0.09]]) + 0.28 * phase, 0.0, 0.70),
        paths, axis=0,
    )
    interdiction = np.repeat(
        np.clip(np.array([[0.22, 0.36, 0.14, 0.18]]) + 0.25 * phase, 0.0, 0.78),
        paths, axis=0,
    )
    return {
        "production": np.clip(0.10 - 0.035 * damage, 0.015, 0.12),
        "consumption": np.clip(0.11 + 0.025 * phase + 0.02 * damage, 0.02, 0.18),
        "damage": damage,
        "interdiction": interdiction,
        "command": np.clip(0.94 - 0.38 * damage, 0.20, 0.98),
        "pnt": np.clip(0.96 - 0.42 * damage, 0.18, 0.99),
        "financial_capacity": np.clip(0.95 - 0.28 * damage, 0.20, 0.99),
    }


def state_delta(left, right) -> float:
    names = ("inventory", "spares", "repair_queue_state", "integrity", "search_prior")
    return max(
        float(np.max(np.abs(getattr(left, name) - getattr(right, name))))
        for name in names
    )


def undirected_tree_sum(weights: np.ndarray) -> float:
    total = 0.0
    for chosen in combinations(range(len(weights)), NODE_COUNT - 1):
        parent = list(range(NODE_COUNT))

        def find(node: int) -> int:
            while parent[node] != node:
                parent[node] = parent[parent[node]]
                node = parent[node]
            return node

        valid = True
        for edge in chosen:
            left, right = find(int(EDGE_TAIL[edge])), find(int(EDGE_HEAD[edge]))
            if left == right:
                valid = False
                break
            parent[left] = right
        if valid and len({find(node) for node in range(NODE_COUNT)}) == 1:
            # The symmetric Laplacian assigns half of the directed capacity.
            total += float(np.prod(0.5 * weights[list(chosen)]))
    return total


def directed_arborescence_sum(weights: np.ndarray) -> float:
    total = 0.0
    for chosen in combinations(range(len(weights)), NODE_COUNT - 1):
        outgoing = {node: [] for node in range(NODE_COUNT)}
        for edge in chosen:
            outgoing[int(EDGE_TAIL[edge])].append(int(EDGE_HEAD[edge]))
        if outgoing[SINK] or any(len(outgoing[node]) != 1 for node in range(SINK)):
            continue
        valid = True
        for start in range(SINK):
            node, visited = start, set()
            while node != SINK and node not in visited:
                visited.add(node)
                node = outgoing[node][0]
            valid &= node == SINK
        if valid:
            total += float(np.prod(weights[list(chosen)]))
    return total


def main() -> None:
    paths = 8
    terrain = {
        "land_share_above_500m": 0.42,
        "sky_view_proxy_p10_p50_p90": [0.31, 0.58, 0.83],
    }
    active = AlgebraicDiscreteOperationalNetworkV72(
        paths, 72_101, terrain, geometry_enabled=True, advanced_geometry=True
    )
    ablated = CombinatorialOperationalNetworkV71(
        paths, 72_101, terrain, geometry_enabled=True, advanced_geometry=True
    )
    maximum_state_delta = 0.0
    for month in range(24):
        args = inputs(paths, month)
        active.update(**args)
        ablated.update(**args)
        maximum_state_delta = max(maximum_state_delta, state_delta(active, ablated))

    diag = active.diagnostics()
    graph = diag["algebraic_discrete_network_v72"]
    inherited = diag["combinatorial_network_v71"]
    capacity = active.edge_base_capacity[0, 0] * active.edge_integrity[0, 0]
    batch = capacity[None, None, :]
    metrics = active._algebraic_metrics(batch)
    larger = active._algebraic_metrics(1.08 * batch)
    tree_enum = undirected_tree_sum(capacity)
    tree_det = float(np.exp(metrics["tree_logdet"][0, 0]))
    directed_enum = directed_arborescence_sum(capacity)
    directed_det = float(np.exp(metrics["directed_tree_logdet"][0, 0]))
    resistance_monotonicity = float(np.max(
        larger["resistance"] - metrics["resistance"]
    ))
    tree_relative_error = abs(tree_enum - tree_det) / max(tree_enum, 1.0e-12)
    directed_relative_error = abs(directed_enum - directed_det) / max(
        directed_enum, 1.0e-12
    )

    checks = {
        "multilayer_graph_replaces_parallel_toy": (
            graph["nodes"] == 7 and graph["directed_edges"] == 12
            and graph["simple_paths"] > 3
        ),
        "same_path_ablation_changes_state": maximum_state_delta > 1.0e-6,
        "monthly_clock": graph["months"] == 24,
        "capacity_feasible": inherited["maximum_capacity_residual"] < 1.0e-10,
        "stock_flow_accounting": inherited["maximum_stock_residual"] < 1.0e-9,
        "cascade_converged": (
            inherited["maximum_cascade_residual"] < 1.0e-7
            and inherited["cascade_nonconvergence_months"] == 0
        ),
        "effective_resistance_nonnegative": float(np.min(metrics["resistance"])) >= -1.0e-10,
        "rayleigh_monotonicity": resistance_monotonicity <= 1.0e-9,
        "matrix_tree_enumeration": tree_relative_error <= 1.0e-9,
        "directed_matrix_tree_enumeration": directed_relative_error <= 1.0e-9,
        "nonbacktracking_finite_positive": (
            np.isfinite(graph["mean_nonbacktracking_radius"])
            and graph["mean_nonbacktracking_radius"] > 0.0
        ),
        "signed_balance_bounded": 0.0 < graph["mean_signed_cohesion"] <= 1.0,
        "spectral_interlacing": graph["maximum_interlacing_violation"] <= 1.0e-9,
        "temporal_state_well_defined": (
            graph["minimum_temporal_connectivity"] >= 0
            and np.isfinite(graph["mean_temporal_arrival"])
            and graph["mean_temporal_arrival"] >= 0.0
        ),
        "matroid_partition_feasible": graph["maximum_protection_partition_violation"] == 0,
        "conflict_graph_feasible": graph["protection_conflict_violations"] == 0,
        "treewidth_selects_exact_solver": TREEWIDTH <= 4 and graph["solver_branch"] == "exact-enumeration",
        "no_probability_postprocessing": not graph["terminal_probability_adjustment"],
    }
    payload = {
        "model_version": "V7.2-ALGEBRAIC-DISCRETE-GRAPH",
        "classification": "DIAGNOSTIC_ONLY; no outcome probabilities",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "active_ablation_max_state_delta": maximum_state_delta,
        "matrix_tree_relative_error": tree_relative_error,
        "directed_matrix_tree_relative_error": directed_relative_error,
        "rayleigh_monotonicity_violation": max(resistance_monotonicity, 0.0),
        "diagnostics": graph,
        "inherited_network_diagnostics": inherited,
    }
    TARGET.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"TAIWAN_V72_DIAGNOSTIC: {payload['status']}")
    if payload["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
