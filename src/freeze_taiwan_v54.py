from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "work"
OUT = ROOT / "outputs"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main() -> None:
    trace_path = OUT / "taiwan_v54_traceability.json"
    audit_path = OUT / "taiwan_v54_static_coupling_audit.json"
    diagnostic_path = OUT / "taiwan_v54_windows_acceleration_diagnostic.json"
    benchmark_path = OUT / "numerical_acceleration_benchmark.json"
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    diagnostic = json.loads(diagnostic_path.read_text(encoding="utf-8"))
    if not (
        trace.get("gate_status") == "PASS"
        and audit.get("status") == "PASS"
        and audit.get("simulation_authorized") is True
        and diagnostic.get("status") == "PASS"
    ):
        raise RuntimeError("V5.4 cannot be frozen before all gates pass")

    files = [
        WORK / "dynamic_macro_equilibrium_v47.py",
        WORK / "dynamic_macro_dqsfc_v53.py",
        WORK / "social_accounting_matrix_v53.py",
        WORK / "continuous_multipop_mfg_v47.py",
        WORK / "fp_solvers_v52.py",
        WORK / "rolling_differential_game_v47.py",
        WORK / "numerical_acceleration.py",
        WORK / "wartime_social_contagion.py",
        WORK / "simulate_taiwan_fully_closed_v47.py",
        WORK / "simulate_taiwan_fully_closed_v54.py",
        WORK / "diagnostic_windows_acceleration_v54.py",
        WORK / "benchmark_numerical_acceleration.py",
        OUT / "taiwan_v54_windows_acceleration_specification.md",
        trace_path,
        audit_path,
        diagnostic_path,
        benchmark_path,
        ROOT / "data" / "oecd_icio_2025" / "icio_2022_8x10.npz",
        ROOT / "data" / "china_mrio" / "dual_circulation_2022_140nodes.npz",
    ]
    payload = {
        "model_version": "V5.4",
        "status": "FROZEN",
        "frozen_at": "2026-10-08 Asia/Shanghai",
        "simulation_contract": {
            "paths": 128,
            "months": 360,
            "intervention_cases": 5,
            "trade_scenarios": 6,
            "master_clock": "monthly same-path",
            "equations_and_parameters": "identical to V5.3",
            "common_random_numbers": "identical seed ownership to V5.3",
        },
        "platform_contract": {
            "native_windows": True,
            "wsl_required": False,
            "jax_cuda_required": False,
            "multi_gpu_required": False,
            "pytorch_cuda_fp": True,
            "numba_cge": False,
            "numba_dsge_hjb": True,
        },
        "acceleration_documents": [
            "Uu2Mdtxw7oPqbLx9oumcHHdWn02",
            "AOM5dDZ2gooFEQxOiHacyTh5nke",
            "ZkYhdRSmYoLe06xyJ9Xc0cJnnEG",
            "HwsBdgnDnopW2fxAMW4cCYavn9f",
        ],
        "files": {str(path.relative_to(ROOT)): digest(path) for path in files},
    }
    target = OUT / "taiwan_v54_frozen_specification.json"
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"TAIWAN_V54_FREEZE: PASS; sha256={digest(target)}")


if __name__ == "__main__":
    main()
