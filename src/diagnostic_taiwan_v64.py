"""DIAGNOSTIC_ONLY: stochastic-geometry invariants and historical replay."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from stochastic_geometry_v64 import DynamicStochasticGeometryV64, hualien_2024_replay


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "outputs" / "taiwan_v64_stochastic_geometry_diagnostic.json"


def exercise(flags: dict) -> dict:
    paths = 8
    model = DynamicStochasticGeometryV64(
        paths, 64_101, {"source": "public DEM"}, **flags
    )
    last = None
    for month in range(18):
        phase = (month + 1) / 18.0
        intensity = np.linspace(0.08, 0.92, paths) * phase
        actor_axis = np.array([[0.72, 1.0, 0.52, 0.64]])
        damage = np.clip(intensity[:, None] * actor_axis, 0.0, 0.86)
        last = model.update(
            month=month,
            conflict_intensity=intensity,
            common_mark=np.where(month in (4, 11), 0.42, 0.0) * np.ones(paths),
            actor_damage=damage,
            infrastructure=np.clip(0.94 - 0.62 * damage, 0.12, 1.0),
            repair_capacity=np.clip(0.78 - 0.30 * damage, 0.10, 1.0),
            actor_network=np.clip(0.91 - 0.38 * damage, 0.12, 1.0),
            financial_capacity=np.clip(0.88 - 0.34 * damage, 0.12, 1.0),
        )
    diag = model.diagnostics()
    return {
        "diagnostics": diag,
        "terminal": {
            key: getattr(last, key).tolist()
            for key in (
                "coverage", "coverage_p10", "double_coverage", "service",
                "logistics", "pnt", "network", "outage", "social_exposure",
            )
        },
    }


def main() -> None:
    complete = exercise({
        "random_field_enabled": True,
        "cascade_enabled": True,
        "feedback_enabled": True,
    })
    no_field = exercise({
        "random_field_enabled": False,
        "cascade_enabled": True,
        "feedback_enabled": True,
    })
    no_cascade = exercise({
        "random_field_enabled": True,
        "cascade_enabled": False,
        "feedback_enabled": True,
    })
    replay = hualien_2024_replay()
    diag = complete["diagnostics"]
    complete_terminal = np.asarray(complete["terminal"]["logistics"])
    field_terminal = np.asarray(no_field["terminal"]["logistics"])
    cascade_terminal = np.asarray(no_cascade["terminal"]["network"])
    complete_network_terminal = np.asarray(complete["terminal"]["network"])
    checks = {
        "probability_bounds": diag["max_probability_residual"] <= 1e-12,
        "service_capacity": diag["max_service_capacity_violation"] <= 1e-12,
        "causal_clock": diag["max_causal_clock_violation"] <= 1e-12,
        "finite_nontrivial_coverage": (
            0.0 <= diag["minimum_coverage"] < diag["maximum_coverage"] <= 1.0
        ),
        "dynamic_birth_death_recovery": (
            diag["dynamic_births"] > 0
            and diag["dynamic_failures"] > 0
            and diag["dynamic_recoveries"] > 0
        ),
        "real_gis_loaded": (
            diag["real_gis_conditioning"] and diag["terrain_tile_count"] > 0
            and bool(diag["road_cache_sha256"])
        ),
        "random_field_ablation_changes_state": float(
            np.max(np.abs(complete_terminal - field_terminal))
        ) > 1e-6,
        "cascade_ablation_changes_state": float(
            np.max(np.abs(complete_network_terminal - cascade_terminal))
        ) > 1e-6,
        "hualien_replay": all(replay["checks"].values()),
        "no_posthoc_probability_adjustment": not diag["posthoc_probability_adjustment"],
        "no_future_information": not diag["future_information_used"],
    }
    payload = {
        "classification": "DIAGNOSTIC_ONLY; no outcome probabilities",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": {k: bool(v) for k, v in checks.items()},
        "complete": complete,
        "ablations": {"without_random_field": no_field, "without_cascade": no_cascade},
        "historical_replay": replay,
    }
    TARGET.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"TAIWAN_V64_STOCHASTIC_GEOMETRY_DIAGNOSTIC: {payload['status']}")
    if payload["status"] != "PASS":
        print(json.dumps({k: v for k, v in checks.items() if not v}, indent=2))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
