from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

import simulate_taiwan_fully_closed_v60 as model


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "taiwan_v60_riemannian_stress.json"


def main() -> None:
    if os.environ.get("TAIWAN_ALLOW_COMPLETE_RUN") != "1":
        raise RuntimeError("V6.0 stress test requires the complete-model gate")
    scenario = next(iter(model.v59.v58.core.trade.v43.SCENARIOS))
    with np.load(model.v59.v58.DRAWS) as source:
        draws = {key: source[key][:8].copy() for key in source.files}
    expected_small_sample_errors = []
    try:
        model.v59.v58.core.run(
            paths=8,
            months=360,
            scenario_names=[scenario],
            mechanism_enabled=True,
            energy_enabled=True,
            geometry_enabled=True,
            hybrid_enabled=True,
            topology_enabled=True,
            mfg_config=model.MFG_CONFIG,
            geometry_config=model.GEOMETRY_CONFIG,
            macro_factory=model.v59.v58.CoupledMRRDCGEDQSFCEconomicSystem,
            political_factory=model.v59.v58.WartimeCollectiveActionRegimeSystem,
            econometric_draws=draws,
            traceability_path=model.TRACE,
            coupling_audit_path=model.AUDIT,
        )
    except RuntimeError as error:
        prefix = "V5.2 runtime validation failed: "
        message = str(error)
        if not message.startswith(prefix):
            raise
        failures = message[len(prefix):].split("; ")
        unexpected = [failure for failure in failures if not failure.endswith("importance ESS")]
        if unexpected:
            raise RuntimeError("V6.0 stress test has non-ESS failures: " + "; ".join(unexpected))
        expected_small_sample_errors = failures
    payload = {
        "purpose": "numerical convergence stress test; not an outcome estimate",
        "paths": 8,
        "months": 360,
        "scenario": scenario,
        "mfg_runtime_validation": "PASS",
        "expected_small_sample_errors": expected_small_sample_errors,
        "formal_importance_ess_validation_deferred_to_128_paths": True,
    }
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("TAIWAN_V60_RIEMANNIAN_STRESS: PASS")


if __name__ == "__main__":
    main()
