from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

import simulate_taiwan_multidomain_v42 as v42
import simulate_dual_circulation_v44 as trade


ROOT = Path(__file__).resolve().parents[1]
OUT_CSV = ROOT / "outputs" / "台海V4.4双循环耦合_条件广泛控制率.csv"
OUT_JSON = ROOT / "outputs" / "台海V4.4双循环耦合_条件广泛控制率摘要.json"
YEAR = 2030


def run() -> tuple[list[dict], dict]:
    base, v40_rows, terrain, sipri = v42.load_inputs()
    trade_scenarios = list(trade.v43.SCENARIOS)
    profiles = {
        name: trade.load_coupling_profile(
            name,
            v42.PATHS,
            v42.MONTHS,
            v42.SEED + 700_000 + index * 10_007,
        )
        for index, name in enumerate(trade_scenarios)
    }

    results: list[dict] = []
    baseline_score_differences: list[float] = []
    for case in v42.CASES:
        cell_seed = v42.SEED + YEAR * 101 + v42.CASES.index(case) * 1009
        prior_rng = np.random.default_rng(cell_seed)
        z_standard = prior_rng.normal(0.0, 1.0, v42.PATHS)
        snapshots_by_trade = {}
        reference_rng = np.random.default_rng(cell_seed)
        reference_snapshots, _ = v42.simulate_mechanisms(
            reference_rng, YEAR, case, "central", terrain, sipri
        )
        for name in trade_scenarios:
            # The central military path and seed are fixed across trade scenarios.
            rng = np.random.default_rng(cell_seed)
            snapshots, _ = v42.simulate_mechanisms(
                rng,
                YEAR,
                case,
                "central",
                terrain,
                sipri,
                economy_profile=profiles[name],
            )
            snapshots_by_trade[name] = snapshots

        previous = {name: np.zeros(v42.PATHS) for name in trade_scenarios}
        for horizon_months in v42.HORIZON_MONTHS:
            prior_log_odds = v42.make_prior_paths(
                z_standard,
                base[(YEAR, case)],
                v40_rows[(YEAR, case)],
                horizon_months,
            )
            prior_probability = 1.0 - math.exp(
                math.log1p(-base[(YEAR, case)]) * horizon_months / 180.0
            )
            for name in trade_scenarios:
                snap = snapshots_by_trade[name][horizon_months]
                probability_paths = v42.sigmoid(prior_log_odds + snap["score"])
                probability_paths = np.maximum(probability_paths, previous[name])
                previous[name] = probability_paths
                summary = v42.summarize_probability(probability_paths)
                results.append(
                    {
                        "conflict_year": YEAR,
                        "horizon_years": horizon_months // 12,
                        "case": case,
                        "case_label": v42.CASE_LABELS[case],
                        "trade_scenario": name,
                        "trade_scenario_code": trade.v43.SCENARIOS[name],
                        "prior_probability": prior_probability,
                        "probability": summary["probability"],
                        "p10": summary["path_probability_p10_p50_p90"][0],
                        "p50": summary["path_probability_p10_p50_p90"][1],
                        "p90": summary["path_probability_p10_p50_p90"][2],
                        "cvar99": summary["cvar99_path_probability"],
                        "bridgehead_share": snap["diagnostics"]["mechanical_bridgehead_share"],
                        "majority_control_share": snap["diagnostics"]["mechanical_majority_control_share"],
                    }
                )

        baseline_name = next(name for name, code in trade.v43.SCENARIOS.items() if code == "baseline")
        for horizon_months in v42.HORIZON_MONTHS:
            score_difference = np.max(np.abs(
                snapshots_by_trade[baseline_name][horizon_months]["score"]
                - reference_snapshots[horizon_months]["score"]
            ))
            baseline_score_differences.append(float(score_difference))

    errors = []
    if max(baseline_score_differences, default=0.0) > 1e-12:
        errors.append(f"baseline regression mismatch {max(baseline_score_differences)}")
    for row in results:
        if not 0.0 <= row["probability"] <= 1.0:
            errors.append("probability outside [0,1]")
    for case in v42.CASES:
        for name in trade_scenarios:
            series = sorted(
                (r for r in results if r["case"] == case and r["trade_scenario"] == name),
                key=lambda r: r["horizon_years"],
            )
            if any(b["probability"] + 1e-12 < a["probability"] for a, b in zip(series, series[1:])):
                errors.append(f"non-monotone {case} {name}")
    if errors:
        raise RuntimeError("; ".join(sorted(set(errors))))

    verification = {
        "status": "PASS",
        "baseline_trade_scenario": baseline_name,
        "baseline_regression_max_abs_difference": max(baseline_score_differences, default=0.0),
        "paths": v42.PATHS,
        "months": v42.MONTHS,
        "horizons_years": list(v42.HORIZON_YEARS),
        "warning": (
            "These are conditional broad-control probabilities after conflict onset and a specified "
            "intervention/trade scenario, not unconditional probabilities of war or unification. "
            "The economic and trade state is simulated monthly through year 30."
        ),
    }
    return results, verification


def write(results: list[dict], verification: dict) -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    OUT_JSON.write_text(
        json.dumps({"verification": verification, "results": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    results, verification = run()
    write(results, verification)
    print("TAIWAN_DUAL_CIRCULATION_COUPLED_V44_VERIFICATION: PASS")
    print(json.dumps(verification, ensure_ascii=False))
    for row in results:
        if row["horizon_years"] in (15, 30):
            print(
                row["case"], row["trade_scenario_code"], row["horizon_years"],
                f"{100 * row['probability']:.3f}%",
            )


if __name__ == "__main__":
    main()
