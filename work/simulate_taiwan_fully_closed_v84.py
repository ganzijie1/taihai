from __future__ import annotations

import csv
import hashlib
import json
import os
from functools import partial
from pathlib import Path

import numpy as np

import dynamic_macro_equilibrium_v47 as macro_core
import simulate_taiwan_fully_closed_v82 as v82
import simulate_taiwan_fully_closed_v83 as v83
from particle_international_economics_v83 import StructuralInternationalCGEV83
from statistical_contact_mfg_v84 import StatisticalContactMFGV84
from variant_batching import VariantBatchLayout, shared_variant_random_context


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
TRACE = OUT / "taiwan_v84_traceability.json"
AUDIT = OUT / "taiwan_v84_static_coupling_audit.json"
FREEZE = OUT / "taiwan_v84_frozen_specification.json"
OUT_CSV = OUT / "taiwan_v84_results.csv"
OUT_JSON = OUT / "taiwan_v84_simulation.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def complete_model_gate() -> tuple[bool, list[str]]:
    if not TRACE.exists() or not AUDIT.exists() or not FREEZE.exists():
        return False, ["V8.4 traceability, audit or frozen specification missing"]
    trace = json.loads(TRACE.read_text(encoding="utf-8"))
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    gaps = list(trace.get("blocking_gaps", []))
    passed = (
        trace.get("gate_status") == "PASS"
        and audit.get("simulation_authorized") is True
        and freeze.get("status") == "FROZEN"
        and not gaps
    )
    return passed, gaps


def validate_v84(diagnostics: dict, months: int) -> None:
    v83.validate_v83(diagnostics, months)
    errors: list[str] = []
    required_components = {
        "nonparametric", "multivariate", "time_series", "wavelet", "contact",
        "wavelet_temporal", "wavelet_coherence", "wavelet_graph",
        "wavelet_scattering", "contact_military", "contact_economic",
        "contact_financial", "contact_social",
    }
    for cell, item in diagnostics.items():
        layer = item.get("mfg", {}).get("v84_statistical_multiscale_contact")
        if layer is None:
            errors.append(f"{cell}: missing V8.4 statistical/contact diagnostics")
            continue
        checks = {
            "clock": layer["months"] >= months,
            "causal": layer["no_future_information"],
            "covariance finite": np_isfinite(layer["maximum_covariance_condition_number"]),
            "precision symmetric": layer["maximum_graphical_precision_asymmetry"] <= 1.0e-10,
            "wavelet reconstruction": layer["maximum_wavelet_reconstruction_residual"] <= 1.0e-10,
            "contact monotone": layer["maximum_contact_monotonicity_violation"] <= 1.0e-12,
            "feedback active": layer["maximum_feedback_shift"] > 0.0,
            "no duplicate state": not layer["structural_state_duplication"],
            "no posthoc": not layer["terminal_probability_adjustment"],
            "component registry": required_components <= set(layer["enabled"]),
            "boolean switches": all(
                isinstance(value, bool) for value in layer["enabled"].values()
            ),
        }
        errors.extend(f"{cell}: {name}" for name, ok in checks.items() if not ok)
    if errors:
        raise RuntimeError("V8.4 runtime validation failed: " + "; ".join(errors))


def np_isfinite(value) -> bool:
    try:
        return bool(float("-inf") < float(value) < float("inf"))
    except (TypeError, ValueError):
        return False


def run(paths: int = 128, months: int = 360, *,
        nonparametric_enabled: bool = True,
        multivariate_enabled: bool = True,
        time_series_enabled: bool = True,
        wavelet_enabled: bool = True,
        contact_enabled: bool = True,
        wavelet_temporal_enabled: bool = True,
        wavelet_coherence_enabled: bool = True,
        wavelet_graph_enabled: bool = True,
        wavelet_scattering_enabled: bool = True,
        contact_military_enabled: bool = True,
        contact_economic_enabled: bool = True,
        contact_financial_enabled: bool = True,
        contact_social_enabled: bool = True,
        particle_intent_enabled: bool = True,
        intent_particles: int | None = None,
        structural_gravity_enabled: bool = True,
        melitz_enabled: bool = True,
        external_liquidity_enabled: bool = True, **kwargs):
    gate, gaps = complete_model_gate()
    if not gate:
        raise RuntimeError("V8.4 complete-model gate failed: " + "; ".join(gaps or ["gate not passed"]))
    v77 = v82.v81.v80.v78.v77
    old_cge = macro_core.DynamicMultiRegionalCGE
    old_nested_cge = v77.NestedCESCGEV77
    old_mfg = v82.v81.DifferentialManifoldMFGV81
    old_paths = (v82.TRACE, v82.AUDIT, v82.FREEZE)
    cge_factory = partial(
        StructuralInternationalCGEV83,
        structural_gravity_enabled=structural_gravity_enabled,
        melitz_enabled=melitz_enabled,
        external_liquidity_enabled=external_liquidity_enabled,
    )
    mfg_factory = partial(
        StatisticalContactMFGV84,
        particle_intent_enabled=particle_intent_enabled,
        intent_particles=intent_particles,
        v84_nonparametric_enabled=nonparametric_enabled,
        v84_multivariate_enabled=multivariate_enabled,
        v84_time_series_enabled=time_series_enabled,
        v84_wavelet_enabled=wavelet_enabled,
        v84_contact_enabled=contact_enabled,
        v84_wavelet_temporal_enabled=wavelet_temporal_enabled,
        v84_wavelet_coherence_enabled=wavelet_coherence_enabled,
        v84_wavelet_graph_enabled=wavelet_graph_enabled,
        v84_wavelet_scattering_enabled=wavelet_scattering_enabled,
        v84_contact_military_enabled=contact_military_enabled,
        v84_contact_economic_enabled=contact_economic_enabled,
        v84_contact_financial_enabled=contact_financial_enabled,
        v84_contact_social_enabled=contact_social_enabled,
    )
    macro_core.DynamicMultiRegionalCGE = cge_factory
    v77.NestedCESCGEV77 = cge_factory
    v82.v81.DifferentialManifoldMFGV81 = mfg_factory
    v82.TRACE, v82.AUDIT, v82.FREEZE = TRACE, AUDIT, FREEZE
    try:
        result = v82.run(paths=paths, months=months, **kwargs)
    finally:
        macro_core.DynamicMultiRegionalCGE = old_cge
        v77.NestedCESCGEV77 = old_nested_cge
        v82.v81.DifferentialManifoldMFGV81 = old_mfg
        v82.TRACE, v82.AUDIT, v82.FREEZE = old_paths
    validate_v84(result[2], months)
    return result


def run_variant_batch(
    paths: int, months: int, variant_specs: dict[str, dict], **kwargs,
):
    """Execute ablations as one explicit variant-path state batch with CRN."""
    if not variant_specs:
        raise ValueError("variant_specs must not be empty")
    layout = VariantBatchLayout(tuple(variant_specs), int(paths))
    flag_names = (
        "nonparametric", "multivariate", "time_series", "wavelet", "contact",
        "wavelet_temporal", "wavelet_coherence", "wavelet_graph",
        "wavelet_scattering", "contact_military", "contact_economic",
        "contact_financial", "contact_social",
    )
    flag_values = {}
    for flag in flag_names:
        by_variant = np.array([
            bool(variant_specs[name].get(f"{flag}_enabled", True))
            for name in layout.names
        ])
        flag_values[f"{flag}_enabled"] = np.repeat(by_variant, layout.paths_per_variant)
    with shared_variant_random_context(layout):
        result = run(
            paths=layout.total_paths, months=months,
            path_group_names=layout.names, paths_per_group=layout.paths_per_variant,
            **flag_values, **kwargs,
        )
    result[3]["variant_batch"].update({
        "state_shape_prefix": [layout.variants, layout.paths_per_variant],
        "variant_count": layout.variants,
        "shared_exogenous_draws": True,
        "independent_endogenous_states": True,
    })
    return result


def write(rows, aggregate, diagnostics, verification) -> None:
    with OUT_CSV.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    verification["model_version"] = "V8.4-STATISTICAL-MULTISCALE-CONTACT"
    verification["runtime_validation"] = "TAIWAN_V84_RUNTIME_VALIDATION: PASS"
    verification["result_sha256"] = sha256(OUT_CSV)
    verification["v84_upgrade"] = {
        "five_documents_registered": True,
        "causal_online_statistics": True,
        "causal_multiscale_wavelets": True,
        "contact_dissipation_state": True,
        "posthoc_probability_adjustment": False,
        "frozen_specification_sha256": sha256(FREEZE),
    }
    OUT_JSON.write_text(json.dumps({
        "verification": verification,
        "aggregate": aggregate,
        "conditional": rows,
        "module_diagnostics": diagnostics,
    }, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    if os.environ.get("TAIWAN_ALLOW_COMPLETE_RUN") != "1":
        raise RuntimeError("V8.4 outcome simulation requires complete-run authorization")
    result = run(
        paths=int(os.environ.get("TAIWAN_V84_PATHS", "128")),
        months=int(os.environ.get("TAIWAN_V84_MONTHS", "360")),
    )
    write(*result)
    print("TAIWAN_V84_COMPLETE_RUN: PASS")


if __name__ == "__main__":
    main()
