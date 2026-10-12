from __future__ import annotations

import json
import os
from pathlib import Path

import simulate_taiwan_fully_closed_v57 as model
from wartime_collective_action_regime_v57 import (
    WartimeCollectiveActionRegimeSystem,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
TARGET = OUT / "taiwan_v57_one_at_a_time_ablation.json"


def factory(features: dict[str, bool]):
    def build(paths: int, seed: int, support):
        return WartimeCollectiveActionRegimeSystem(
            paths, seed, support, features=features
        )
    return build


def summarize(rows: list[dict]) -> dict:
    return {
        f'{row["case"]}|{row["horizon_years"]}y': row[
            "conditional_broad_control_rate"
        ]
        for row in rows if row["horizon_years"] in (5, 15, 30)
    }


def run_variant(paths: int, months: int, features: dict[str, bool]):
    frozen_draws = model.load_draws(128)
    draws = {key: value[:paths].copy() for key, value in frozen_draws.items()}
    baseline_scenario = list(model.core.trade.v43.SCENARIOS)[0]
    return model.core.run(
        paths=paths,
        months=months,
        scenario_names=[baseline_scenario],
        mechanism_enabled=True,
        energy_enabled=True,
        geometry_enabled=True,
        hybrid_enabled=True,
        topology_enabled=True,
        mfg_config=model.MFG_CONFIG,
        geometry_config=model.GEOMETRY_CONFIG,
        macro_factory=model.CoupledMRRDCGEDQSFCEconomicSystem,
        political_factory=factory(features),
        econometric_draws=draws,
        traceability_path=model.TRACE,
        coupling_audit_path=model.AUDIT,
    )


def main() -> None:
    if os.environ.get("TAIWAN_ALLOW_COMPLETE_RUN") != "1":
        raise RuntimeError("V5.7 ablation blocked before the complete-model gate")
    frozen = json.loads(model.FREEZE.read_text(encoding="utf-8"))
    if frozen.get("status") != "FROZEN":
        raise RuntimeError("V5.7 frozen specification is missing or invalid")
    paths = int(os.environ.get("TAIWAN_V57_ABLATION_PATHS", "16"))
    months = int(os.environ.get("TAIWAN_V57_ABLATION_MONTHS", "360"))
    names = (
        "full", "common_knowledge_network", "organization_capital",
        "government_mpc", "loyalty_hmm", "competing_risks", "closed_feedback",
        "revolution_episode", "external_intervention_nationalism",
    )
    runs: dict[str, dict] = {}
    for name in names:
        features = {
            "common_knowledge_network": True,
            "organization_capital": True,
            "government_mpc": True,
            "loyalty_hmm": True,
            "competing_risks": True,
            "revolution_episode": True,
            "external_intervention_nationalism": True,
            "closed_feedback": True,
        }
        if name != "full":
            features[name] = False
        rows, _, diagnostics, _ = run_variant(paths, months, features)
        political = [
            cell["political"] for cell in diagnostics.values()
            if cell.get("political") is not None
        ]
        runs[name] = {
            "conditional_control": summarize(rows),
            "diagnostic_cells": len(diagnostics),
            "max_loyalty_simplex_residual": max(
                item["max_loyalty_simplex_residual"] for item in political
            ),
            "max_event_probability_sum": max(
                item["max_event_probability_sum"] for item in political
            ),
        }
        print(f"TAIWAN_V57_ABLATION_CELL_COMPLETE {name}", flush=True)
    baseline = runs["full"]["conditional_control"]
    for result in runs.values():
        result["change_pp_from_full"] = {
            key: 100.0 * (value - baseline[key])
            for key, value in result["conditional_control"].items()
        }
    payload = {
        "model_version": "V5.7",
        "classification": "one-at-a-time common-random-number module ablation",
        "paths": paths,
        "months": months,
        "trade_scenarios": ["baseline"],
        "intervention_cases": 5,
        "status": "PASS",
        "runs": runs,
        "draw_subset_contract": "deterministic first paths of the 128 frozen V5.5 draws",
        "statistical_warning": "small-path ablations diagnose mechanism direction; formal levels come from the 128-path run",
    }
    TARGET.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("TAIWAN_V57_ABLATION: PASS")


if __name__ == "__main__":
    main()
