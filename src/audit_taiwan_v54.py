from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "work"
OUT = ROOT / "outputs"


def solver_module(module_id, document_id, implementation, symbols, solver):
    return {
        "module_id": module_id,
        "document_sections": [f"acceleration:{document_id}"],
        "equations": [solver],
        "role": "constraint_solver",
        "status": "active",
        "implementation_status": "PASS",
        "implementation": implementation,
        "symbols": symbols,
        "state_owner": "none; numerical backend preserves inherited state ownership",
        "clock": "same inherited path and monthly event clock",
        "pathwise": True,
        "inputs": ["inherited V5.3 solver state and coefficients"],
        "outputs": ["numerically equivalent accepted state"],
        "feedback_consumers": ["all inherited same-path downstream modules"],
        "solver": solver,
        "convergence_criterion": "V5.4 native-Windows equivalence and residual gate",
        "data": [{
            "source": f"https://my.feishu.cn/docx/{document_id}",
            "publisher": "numerical acceleration research specification",
            "observation_date": "2026-10-08",
            "unit": "algorithm specification",
            "identification": "derived",
        }],
        "invariants": [
            "same equations and tolerances",
            "same path index and event clock",
            "no post-hoc outcome adjustment",
        ],
        "tests": [
            "work/diagnostic_windows_acceleration_v54.py",
            "outputs/taiwan_v54_windows_acceleration_diagnostic.json",
        ],
        "outputs_artifact": "outputs/taiwan_v54_windows_acceleration_diagnostic.json",
    }


def main() -> None:
    inherited_trace = json.loads(
        (OUT / "taiwan_v53_traceability.json").read_text(encoding="utf-8")
    )
    inherited_audit = json.loads(
        (OUT / "taiwan_v53_static_coupling_audit.json").read_text(encoding="utf-8")
    )
    diagnostic = json.loads(
        (OUT / "taiwan_v54_windows_acceleration_diagnostic.json").read_text(
            encoding="utf-8"
        )
    )
    checks = diagnostic["checks"]
    source = "\n".join(
        (WORK / name).read_text(encoding="utf-8")
        for name in (
            "dynamic_macro_equilibrium_v47.py",
            "continuous_multipop_mfg_v47.py",
            "fp_solvers_v52.py",
            "rolling_differential_game_v47.py",
            "numerical_acceleration.py",
        )
    )
    static_checks = {
        "experimental_cge_compiled_present": "_solve_cge_picard_compiled" in source,
        "compiled_dsge_present": "_solve_dsge_picard_compiled" in source,
        "mfg_numba_hjb_present": "batched_hjb_policy" in source,
        "fused_fp_present": "step_repeated" in source,
        "candidate_axis_present": "_evaluate_candidates" in source,
        "wsl_not_required": not diagnostic["platform_policy"]["wsl_required"],
        "inherited_trace_pass": inherited_trace.get("gate_status") == "PASS",
        "inherited_audit_pass": inherited_audit.get("status") == "PASS",
        "all_acceleration_checks_pass": all(checks.values()),
    }
    blocking = [name for name, passed in static_checks.items() if not passed]

    additions = [
        solver_module(
            "native_windows_cge_dsge_v54", "Uu2Mdtxw7oPqbLx9oumcHHdWn02",
            ["work/dynamic_macro_equilibrium_v47.py"],
            ["DynamicMultiRegionalCGE.solve", "_solve_dsge_picard_compiled"],
            "equivalence-preserving vectorized CGE and Numba-compiled DSGE",
        ),
        solver_module(
            "native_windows_mfg_fp_v54", "AOM5dDZ2gooFEQxOiHacyTh5nke",
            ["work/continuous_multipop_mfg_v47.py", "work/fp_solvers_v52.py"],
            ["batched_hjb_policy", "FokkerPlanckSolver.step_repeated"],
            "Numba HJB plus fused-transfer PyTorch CUDA finite volume",
        ),
        solver_module(
            "vectorized_differential_game_v54", "ZkYhdRSmYoLe06xyJ9Xc0cJnnEG",
            ["work/rolling_differential_game_v47.py"],
            ["RollingHorizonDifferentialGame._evaluate_candidates"],
            "candidate-axis robust rolling differential game",
        ),
        solver_module(
            "batched_equation_kernels_v54", "HwsBdgnDnopW2fxAMW4cCYavn9f",
            ["work/numerical_acceleration.py", "work/wartime_social_contagion.py"],
            ["rk4_step", "euler_maruyama_step", "WartimeSocialContagionSystem.update"],
            "path-batched ODE/SDE kernels and monthly random batches",
        ),
    ]
    trace = deepcopy(inherited_trace)
    trace["model_version"] = "V5.4"
    trace["document_ids"] = list(dict.fromkeys(trace.get("document_ids", []) + [
        "Uu2Mdtxw7oPqbLx9oumcHHdWn02",
        "AOM5dDZ2gooFEQxOiHacyTh5nke",
        "ZkYhdRSmYoLe06xyJ9Xc0cJnnEG",
        "HwsBdgnDnopW2fxAMW4cCYavn9f",
    ]))
    trace["coverage_basis"] = (
        "V5.3 complete model plus four native-Windows numerical solver mappings"
    )
    trace["active_equation_family_count"] += len(additions)
    trace["passed_equation_family_count"] += len(additions)
    trace["coverage_percent"] = 100.0 if not blocking else (
        100.0 * trace["passed_equation_family_count"]
        / trace["active_equation_family_count"]
    )
    trace["gate_status"] = "PASS" if not blocking else "FAIL"
    trace["modules"].extend(additions)
    trace["blocking_gaps"] = blocking

    audit = {
        "model_version": "V5.4",
        "status": "PASS" if not blocking else "FAIL",
        "master_clock": inherited_audit["master_clock"],
        "required_edges": inherited_audit["required_edges"] + [
            {"edge": name, "status": "PASS" if passed else "FAIL"}
            for name, passed in static_checks.items()
        ],
        "duplicate_state_owners": inherited_audit["duplicate_state_owners"],
        "mandatory_solver_inheritance": "V5.3 PASS; equations and ownership unchanged",
        "new_diagnostic_status": diagnostic["status"],
        "blocking_gaps": blocking,
        "simulation_authorized": not blocking,
    }
    (OUT / "taiwan_v54_traceability.json").write_text(
        json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUT / "taiwan_v54_static_coupling_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if blocking:
        raise RuntimeError(f"V5.4 acceleration audit failed: {blocking}")
    print("TAIWAN_V54_ACCELERATION_AUDIT: PASS")


if __name__ == "__main__":
    main()
