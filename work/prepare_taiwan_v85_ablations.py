from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "taiwan_v85_ablation_plan.json"


def main() -> None:
    switches = [
        "electoral_competition", "selectorate_coalition",
        "bureaucratic_execution", "interest_group_rent",
        "fiscal_illusion_common_pool", "veto_points",
        "commitment_reputation", "alliance_two_level",
    ]
    runs = [{"name": "baseline", "disabled": []}]
    runs.extend({"name": f"without_{name}", "disabled": [name]} for name in switches)
    runs.append({"name": "public_choice_neutral", "disabled": ["feedback_enabled"]})
    payload = {
        "classification": "PRE_REGISTERED_ABLATION_PLAN",
        "model_version": "V8.5-FOUR-PARTY-PUBLIC-CHOICE",
        "status": "READY_NOT_RUN",
        "design": "one factor at a time with matched path count, horizon, seeds and exogenous shocks",
        "required_outputs": [
            "conditional broad-control probability by intervention case",
            "peaceful-settlement probability",
            "budget share and execution distributions",
            "alliance participation and aid authorization distributions",
            "maximum conservation and solver residuals"
        ],
        "runs": runs,
        "outcome_simulation_performed": False
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("TAIWAN_V85_ABLATION_PLAN: READY")


if __name__ == "__main__":
    main()
