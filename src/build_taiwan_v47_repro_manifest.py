from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"

ACTIVE_CODE = [
    "audit_taiwan_v47.py",
    "closed_prewar_system_v47.py",
    "competing_risk_survival_v47.py",
    "continuous_multipop_mfg_v47.py",
    "dynamic_macro_equilibrium_v47.py",
    "endogenous_intervention_v47.py",
    "operational_constraints_v47.py",
    "rolling_differential_game_v47.py",
    "simulate_four_party_financial_v45.py",
    "simulate_taiwan_causal_evolution_v37.py",
    "simulate_taiwan_fully_closed_v47.py",
    "simulate_taiwan_multidomain_v42.py",
    "spatial_control_network_v47.py",
    "summarize_taiwan_v47_sensitivity.py",
    "validate_taiwan_v47_ablations.py",
    "wartime_social_contagion.py",
]

INPUTS = [
    ROOT / "work" / "SIPRI-Milex-data-1949-2025_v1.2.xlsx",
    ROOT / "data" / "oecd_icio_2025" / "2016-2022" / "2022_SML.csv",
    ROOT / "data" / "china_mrio" / "MRIO2017_42+CEADS.xlsx",
    OUT / "台海V3.6_GIS机动走廊摘要.json",
    OUT / "taiwan_v47_traceability.json",
    OUT / "taiwan_v47_static_coupling_audit.json",
]

RESULTS = [
    OUT / "taiwan_v47_fully_closed_results.csv",
    OUT / "taiwan_v47_fully_closed_simulation.json",
    OUT / "taiwan_v47_dynamic_ablation_validation.json",
    OUT / "taiwan_v47_sensitivity_summary.csv",
    OUT / "台海V4.7全模块闭合仿真_飞书增补.md",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def record(path: Path) -> dict:
    return {
        "path": str(path.relative_to(ROOT)),
        "exists": path.exists(),
        "bytes": path.stat().st_size if path.exists() else None,
        "sha256": sha256(path) if path.exists() else None,
    }


def main() -> None:
    manifest = {
        "model_version": "V4.7",
        "seed_contract": {
            "campaign_seed_symbol": "simulate_taiwan_multidomain_v42.SEED",
            "conflict_year": "pathwise cause-specific onset from 2026 through 2100",
            "common_random_numbers": True,
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
        "code": [record(ROOT / "work" / name) for name in ACTIVE_CODE],
        "inputs": [record(path) for path in INPUTS],
        "results": [record(path) for path in RESULTS]
        + [record(path) for path in sorted(OUT.glob("taiwan_v47_sensitivity_*.json"))],
    }
    missing_required = [
        row["path"] for group in ("code", "inputs", "results")
        for row in manifest[group] if not row["exists"]
    ]
    manifest["status"] = "PASS" if not missing_required else "FAIL"
    manifest["missing_required"] = missing_required
    destination = OUT / "taiwan_v47_reproducibility_manifest.json"
    destination.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"TAIWAN_V47_REPRO_MANIFEST: {manifest['status']}")
    if missing_required:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
