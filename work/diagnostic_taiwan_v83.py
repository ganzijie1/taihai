from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from particle_international_economics_v83 import (
    AdaptiveMultilevelSplittingV83,
    LatentIntentParticleSystemV83,
    ParticleValidatedMFGV83,
    StructuralInternationalCGEV83,
)


ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "data" / "particle_trade_v83" / "structural_priors.json"
OUTPUT = ROOT / "outputs" / "taiwan_v83_diagnostic_only.json"


def main() -> None:
    config = json.loads(LEDGER.read_text(encoding="utf-8"))
    paths = 4
    initial = np.full((paths, 4), 0.58)
    particle = LatentIntentParticleSystemV83(
        paths, 83_003, initial, config["particle"]
    )
    zeros = np.zeros((paths, 4))
    ones = np.ones((paths, 4))
    for _ in range(6):
        particle.update(
            progress=np.full((paths, 4), 0.08), damage=zeros,
            shortage=np.full((paths, 4), 0.05), command=0.82 * ones,
            network=0.76 * ones, common_noise=zeros, jump_signal=zeros,
        )
    particle_diag = particle.diagnostics()

    ams = AdaptiveMultilevelSplittingV83(
        replicas=config["particle"]["ams_replicas"],
        kill_fraction=config["particle"]["ams_kill_fraction"],
    )

    def transition(rng, parents, n):
        opening = np.zeros(n) if parents is None else np.asarray(parents)
        return opening + rng.normal(0.18, 0.55, n)

    ams_diag = ams.estimate(transition, lambda x: x, threshold=1.8)

    cge = StructuralInternationalCGEV83(2, 83_101, "baseline")
    actor = np.zeros((2, 4))
    levels = np.ones((2, 4))
    regional_levels = np.ones((2, cge.n_actors))
    result = cge.solve(
        month=0, final_demand_factor=regional_levels,
        investment_factor=regional_levels,
        actor_damage=0.03 * levels, actor_logistics=0.92 * levels,
        actor_labor=0.96 * levels, actor_financial_stress=0.04 * levels,
        actor_payment=0.94 * levels, actor_fiscal=0.90 * levels,
        actor_social_action=actor, actor_panic=0.03 * levels,
        blockade_feedback=np.full(2, 0.08), max_iterations=80,
        tolerance=8.0e-3,
    )
    trade_diag = cge.structural_trade_diagnostics()
    mfg = ParticleValidatedMFGV83(
        1, 83_201, np.full((1, 4), 0.58), grid_points=15,
        finite_particle_validation_enabled=True,
    )
    mfg.check_interval = 1
    mfg_factors = mfg.update(
        progress=np.full((1, 4), 0.08), damage=np.full((1, 4), 0.03),
        shortage=np.full((1, 4), 0.05), inflation=np.full((1, 4), 0.02),
        financial_stress=np.full((1, 4), 0.04), command=np.full((1, 4), 0.82),
        network=np.full((1, 4), 0.76),
        contagion_mobilization=np.full((1, 4), 0.12),
        contagion_fatigue=np.full((1, 4), 0.08),
        common_noise=np.zeros((1, 4)), jump_signal=np.zeros((1, 4)),
        max_fixed_point_iterations=48, tolerance=2.0e-3,
    )
    mfg_diag = mfg.diagnostics()["v83_particle_methods"]
    checks = {
        "particle_weight_simplex": particle_diag["maximum_weight_normalization_residual"] <= 1.0e-12,
        "particle_ess_positive": particle_diag["minimum_effective_sample_size"] > 0.0,
        "ams_probability_bounded": 0.0 <= ams_diag["probability"] <= 1.0,
        "ams_estimator_only": ams_diag["estimator_only"],
        "trade_source_simplex": trade_diag["maximum_source_share_residual"] <= 1.0e-12,
        "trade_external_account_finite": np.isfinite(result.external_account_residual),
        "trade_effect_active": trade_diag["maximum_structural_shock_shift"] > 0.0,
        "nested_ces_preserved": "v77_nested_ces" in cge.extension_diagnostics(),
        "mfg_particle_clock": mfg_diag["months"] == 1,
        "mfg_particle_feedback_active": mfg_diag["maximum_intent_feedback"] > 0.0,
        "mfg_finite_validation_active": mfg_diag["finite_particle_validation_checks"] == 1,
        "mfg_support_finite": np.all(np.isfinite(mfg_factors.support)),
        "no_posthoc_probability_adjustment": not trade_diag["terminal_probability_adjustment"],
    }
    checks = {name: bool(value) for name, value in checks.items()}
    payload = {
        "model_version": "V8.3-PARTICLE-INTERNATIONAL-ECONOMICS",
        "classification": "DIAGNOSTIC_ONLY_NO_OUTCOME_PROBABILITIES",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "particle": particle_diag,
        "adaptive_multilevel_splitting": ams_diag,
        "international_economics": trade_diag,
        "particle_mfg": mfg_diag,
    }
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if not all(checks.values()):
        raise RuntimeError("V8.3 diagnostic checks failed")
    print("TAIWAN_V83_DIAGNOSTIC: PASS")


if __name__ == "__main__":
    main()
