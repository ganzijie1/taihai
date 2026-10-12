from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
TARGET = OUT / "taiwan_v58_validation_and_v57_comparison.json"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def indexed(rows: list[dict]) -> dict[tuple, dict]:
    return {(row["trade_scenario"], row["horizon_years"]): row for row in rows}


def main() -> None:
    old = json.loads((OUT / "taiwan_v57_social_regime_simulation.json").read_text(encoding="utf-8"))
    current_path = OUT / "taiwan_v58_information_geometry_simulation.json"
    current = json.loads(current_path.read_text(encoding="utf-8"))
    ablation = json.loads((OUT / "taiwan_v58_information_geometry_ablations.json").read_text(encoding="utf-8"))
    old_aggregate = indexed(old["aggregate"])
    differences = []
    for row in current["aggregate"]:
        key = (row["trade_scenario"], row["horizon_years"])
        baseline = old_aggregate[key]
        differences.append({
            "trade_scenario": key[0],
            "horizon_years": key[1],
            "conditional_v57": baseline["conditional_mean"],
            "conditional_v58": row["conditional_mean"],
            "conditional_change_pp": 100.0 * (
                row["conditional_mean"] - baseline["conditional_mean"]
            ),
            "unconditional_v57": baseline["unconditional_mean"],
            "unconditional_v58": row["unconditional_mean"],
            "unconditional_change_pp": 100.0 * (
                row["unconditional_mean"] - baseline["unconditional_mean"]
            ),
        })

    diagnostics = list(current["module_diagnostics"].values())
    beliefs = [item["mfg"]["partial_observation"] for item in diagnostics]
    residuals = {
        "macro_outer": max(item["macro"]["max_outer_residual"] for item in diagnostics),
        "cge": max(item["macro"]["max_cge_solver_residual"] for item in diagnostics),
        "dsge": max(item["macro"]["max_dsge_solver_residual"] for item in diagnostics),
        "mfg": max(item["mfg"]["max_fixed_point_residual"] for item in diagnostics),
        "macro_nonconverged_months": max(item["macro"]["nonconverged_months"] for item in diagnostics),
        "mfg_nonconverged_months": max(item["mfg"]["nonconverged_months"] for item in diagnostics),
        "posterior_simplex": max(item["max_posterior_simplex_residual"] for item in beliefs),
        "credibility_simplex": max(item["max_reliability_simplex_residual"] for item in beliefs),
        "bridge_marginal": max(item["max_bridge_marginal_residual"] for item in beliefs),
        "information_budget": max(item["max_information_budget_residual"] for item in beliefs),
        "wfr_conditional": max(item["mfg"]["max_full_wfr_conditional_residual"] for item in diagnostics),
        "wfr_stock": max(item["mfg"]["max_full_wfr_stock_residual"] for item in diagnostics),
        "minimum_wfr_measure": min(item["mfg"]["fp_solver"]["minimum_wfr_measure"] for item in diagnostics),
        "major_kl": max(item["mfg"]["max_major_trust_region_kl"] for item in diagnostics),
        "minor_kl": max(item["mfg"]["max_minor_trust_region_kl"] for item in diagnostics),
    }
    decomposition = max(abs(
        row["peace_arrangement_cif"] + row["legal_separation_cif"]
        + row["status_quo_survival"] + row["joint_force_and_broad_control_by_2100"]
        + row["force_without_broad_control_by_2100"] - 1.0
    ) for row in current["verification"]["calendar_2100"])
    ablation_variants = ablation.get("variants", {})
    ablation_converged = all(
        value["diagnostics"]["macro_nonconverged_months"] == 0
        and value["diagnostics"]["mfg_nonconverged_months"] == 0
        and value["diagnostics"]["max_macro_residual"] <= 3.0e-3
        and value["diagnostics"]["max_cge_residual"] <= 2.0e-3
        and value["diagnostics"]["max_dsge_residual"] <= 8.0e-4
        and value["diagnostics"]["max_mfg_residual"] <= 5.0e-4
        for value in ablation_variants.values()
    )
    csv_path = OUT / "taiwan_v58_information_geometry_results.csv"
    checks = {
        "csv_hash_matches": digest(csv_path) == current["verification"]["result_sha256"],
        "runtime_validation_pass": current["verification"]["runtime_validation"].endswith("PASS"),
        "calendar_decomposition": decomposition <= 1.0e-9,
        "probabilities_bounded": all(
            0.0 <= row["conditional_broad_control_rate"] <= 1.0
            for row in current["conditional"]
        ),
        "macro_converged": residuals["macro_nonconverged_months"] == 0,
        "mfg_converged": residuals["mfg_nonconverged_months"] == 0,
        "macro_outer": residuals["macro_outer"] <= 3.0e-3,
        "cge": residuals["cge"] <= 2.0e-3,
        "dsge": residuals["dsge"] <= 8.0e-4,
        "mfg": residuals["mfg"] <= 5.0e-4,
        "posterior_simplex": residuals["posterior_simplex"] <= 1.0e-12,
        "credibility_simplex": residuals["credibility_simplex"] <= 1.0e-12,
        "bridge_marginal": residuals["bridge_marginal"] <= 1.0e-7,
        "information_budget": residuals["information_budget"] <= 1.0e-12,
        "wfr_conditional": residuals["wfr_conditional"] <= 1.0e-12,
        "wfr_stock": residuals["wfr_stock"] <= 1.0e-12,
        "wfr_nonnegative": residuals["minimum_wfr_measure"] >= 0.0,
        "major_trust_region": residuals["major_kl"] <= 0.035 + 1.0e-12,
        "minor_trust_region": residuals["minor_kl"] <= 0.025 + 1.0e-12,
        "ablation_complete": set(ablation_variants) == {
            "complete_v58", "without_belief_filter", "without_credibility_hmm",
            "without_fisher_kl_trust_region", "without_joint_wfr",
            "without_schrodinger_bridge", "without_fgw_alignment",
            "without_divergence_robustness", "without_information_value",
            "without_all_v58_information_geometry",
        },
        "ablation_converged": ablation_converged,
    }
    checks = {key: bool(value) for key, value in checks.items()}
    status = "PASS" if all(checks.values()) else "FAIL"
    payload = {
        "model_version": "V5.8",
        "status": status,
        "checks": checks,
        "calendar_decomposition_residual": decomposition,
        "maximum_residuals": residuals,
        "v57_v58_differences": differences,
        "calendar_2100": current["verification"]["calendar_2100"],
        "intervention_probability_means": current["verification"]["intervention_probability_means"],
        "force_onset_mean": current["verification"]["force_onset_mean"],
        "ablation_path_count": ablation["paths"],
        "ablation_variant_count": len(ablation_variants),
    }
    TARGET.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"TAIWAN_V58_VALIDATION: {status}")
    if status != "PASS":
        print(json.dumps({key: value for key, value in checks.items() if not value}, indent=2))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
