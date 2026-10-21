from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
FREEZE = OUT / "taiwan_v84_frozen_specification.json"
PLAN = OUT / "taiwan_v84_ablation_plan.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main() -> None:
    if not FREEZE.exists() or json.loads(FREEZE.read_text(encoding="utf-8")).get("status") != "FROZEN":
        raise RuntimeError("V8.4 specification must be frozen before the ablation plan")
    variants = {
        "baseline": {},
        "no_nonparametric": {"nonparametric_enabled": False},
        "no_multivariate": {"multivariate_enabled": False},
        "no_time_series": {"time_series_enabled": False},
        "no_wavelet": {"wavelet_enabled": False},
        "no_contact": {"contact_enabled": False},
        "no_wavelet_temporal": {"wavelet_temporal_enabled": False},
        "no_wavelet_coherence": {"wavelet_coherence_enabled": False},
        "no_wavelet_graph": {"wavelet_graph_enabled": False},
        "no_wavelet_scattering": {"wavelet_scattering_enabled": False},
        "no_contact_military": {"contact_military_enabled": False},
        "no_contact_economic": {"contact_economic_enabled": False},
        "no_contact_financial": {"contact_financial_enabled": False},
        "no_contact_social": {"contact_social_enabled": False},
    }
    PLAN.write_text(json.dumps({
        "model_version": "V8.4-STATISTICAL-MULTISCALE-CONTACT",
        "status": "FROZEN",
        "freeze_sha256": sha256(FREEZE),
        "matched_seeds_paths_clock": True,
        "one_component_disabled_per_variant": True,
        "variants": {name: {"kwargs": kwargs, "disabled_only": sorted(kwargs)}
                     for name, kwargs in variants.items()},
        "simulation_executed": False,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print("TAIWAN_V84_ABLATION_PLAN: FROZEN")


if __name__ == "__main__":
    main()
