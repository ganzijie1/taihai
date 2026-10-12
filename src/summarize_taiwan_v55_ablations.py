from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def index(rows: list[dict], key_fields: tuple[str, ...]) -> dict[tuple, dict]:
    return {tuple(row[field] for field in key_fields): row for row in rows}


def main() -> None:
    full_path = OUT / "taiwan_v55_econometric_simulation.json"
    ablation_path = OUT / "taiwan_v55_econometric_ablations.json"
    spec_path = OUT / "taiwan_v55_ablation_specification.json"

    full = json.loads(full_path.read_text(encoding="utf-8"))
    ablations = json.loads(ablation_path.read_text(encoding="utf-8"))

    full_aggregate = index(full["aggregate"], ("trade_scenario", "horizon_years"))
    full_calendar = index(full["verification"]["calendar_2100"], ("trade_scenario",))

    residual_limits = {
        "macro_outer_residual": 0.003,
        "cge_residual": 0.002,
        "dsge_residual": 0.0008,
        "mfg_residual": 0.0005,
    }
    comparisons = []
    all_pass = True

    for name, payload in ablations["ablations"].items():
        aggregate = index(payload["aggregate"], ("trade_scenario", "horizon_years"))
        calendar = index(payload["calendar_2100"], ("trade_scenario",))
        row = {"ablation": name, "effects": [], "residual_checks": {}}

        for scenario in ("baseline", "combined"):
            complete_30 = full_aggregate[(scenario, 30)]["conditional_mean"]
            ablated_30 = aggregate[(scenario, 30)]["conditional_mean"]
            complete_2100 = full_calendar[(scenario,)]["force_onset_cif"]
            ablated_2100 = calendar[(scenario,)]["force_onset_cif"]
            row["effects"].append(
                {
                    "trade_scenario": scenario,
                    "conditional_broad_control_30y_complete": complete_30,
                    "conditional_broad_control_30y_ablated": ablated_30,
                    "conditional_broad_control_30y_delta_pp": 100.0
                    * (complete_30 - ablated_30),
                    "force_onset_2100_complete": complete_2100,
                    "force_onset_2100_ablated": ablated_2100,
                    "force_onset_2100_delta_pp": 100.0 * (complete_2100 - ablated_2100),
                }
            )

        residuals = payload["maximum_residuals"]
        for metric, limit in residual_limits.items():
            passed = residuals[metric] <= limit + 1e-12
            row["residual_checks"][metric] = {
                "value": residuals[metric],
                "limit": limit,
                "pass": passed,
            }
            all_pass = all_pass and passed
        for metric in ("macro_nonconverged_months", "mfg_nonconverged_months"):
            passed = residuals[metric] == 0
            row["residual_checks"][metric] = {
                "value": residuals[metric],
                "limit": 0,
                "pass": passed,
            }
            all_pass = all_pass and passed
        comparisons.append(row)

    expected = {
        "remove_ppml_trade_uncertainty",
        "remove_hmm_transition_posterior",
        "remove_growth_calibration",
        "remove_macro_lp_var_posterior",
        "remove_terminal_and_alliance_prior_uncertainty",
    }
    observed = set(ablations["ablations"])
    coverage_pass = observed == expected
    all_pass = all_pass and coverage_pass

    summary = {
        "model_version": "V5.5",
        "validation_status": "PASS" if all_pass else "FAIL",
        "coverage": {
            "expected": sorted(expected),
            "observed": sorted(observed),
            "pass": coverage_pass,
        },
        "comparisons": comparisons,
        "artifact_hashes": {
            "full_simulation_sha256": sha256(full_path),
            "ablation_results_sha256": sha256(ablation_path),
            "ablation_specification_sha256": sha256(spec_path),
        },
        "interpretation_constraints": [
            "PPML estimates bilateral trade baselines and common shocks, not Armington elasticities.",
            "Macro VAR and local projections are predictive diagnostics, not identified wartime causal effects.",
            "Terminal-risk and alliance coefficients remain explicit scenario priors where public data do not identify them.",
            "Conditional broad control is conditioned on force onset and averaged over the intervention distribution.",
        ],
    }
    target = OUT / "taiwan_v55_econometric_ablation_summary.json"
    target.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"TAIWAN_V55_ABLATION_VALIDATION: {summary['validation_status']}")
    print(target)
    if not all_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
