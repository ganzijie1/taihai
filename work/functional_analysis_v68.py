"""Read-only functional-analytic validation for the complete Taiwan model.

The diagnostics consume solver ledgers from one completed cell.  They never
modify path states, event clocks, likelihood weights, or terminal outcomes.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
import math


@dataclass(frozen=True)
class FunctionalAnalysisReport:
    weak_mass_residual: float
    minimum_density: float
    graphon_row_residual: float
    graphon_markov_spectral_radius: float
    mfg_fixed_point_residual: float
    mfg_empirical_residual_contraction: float
    macro_outer_residual: float
    macro_cge_residual: float
    macro_dsge_residual: float
    rare_event_likelihood_residual: float
    rare_event_empirical_mass_residual: float
    all_values_finite: bool
    passed: bool

    def as_dict(self) -> dict:
        payload = asdict(self)
        payload.update({
            "scope": "read_only_validation",
            "probability_adjustment": False,
            "semigroup_note": (
                "The contraction quotient is an empirical fixed-point residual "
                "ratio, not a claimed generator spectral bound."
            ),
            "graphon_note": (
                "The radius equals one only for the implemented nonnegative "
                "row-stochastic Markov kernel; the row residual is audited."
            ),
        })
        return payload


def _finite_max(*values: float) -> float:
    finite = [abs(float(value)) for value in values if math.isfinite(float(value))]
    return max(finite, default=math.inf)


def analyse_cell(cell: dict) -> FunctionalAnalysisReport:
    mfg = cell.get("mfg") or {}
    fp = mfg.get("fp_solver") or {}
    macro = cell.get("macro") or {}
    rare = cell.get("large_deviation") or cell.get("hybrid_large_deviation") or {}

    weak_mass = _finite_max(
        mfg.get("max_mass_residual", math.inf),
        fp.get("max_mass_residual_before_normalisation", math.inf),
        fp.get("max_wfr_conditional_mass_residual", 0.0),
        fp.get("max_wfr_stock_accounting_residual", 0.0),
    )
    fp_calls = int(fp.get("calls", 0))
    wfr_calls = int(fp.get("wfr_calls", 0))
    conservative_minimum = float(
        fp.get("minimum_density_before_projection", math.inf)
    )
    wfr_minimum = float(fp.get("minimum_wfr_measure", math.inf))
    if fp_calls > wfr_calls and math.isfinite(conservative_minimum):
        minimum_density = conservative_minimum
    elif wfr_calls > 0 and math.isfinite(wfr_minimum):
        # V6.2+ can evolve the joint state-action measure entirely through the
        # unbalanced WFR solver. In that case the conservative FP accumulator
        # is intentionally untouched, so its +inf initializer is not evidence.
        minimum_density = wfr_minimum
    else:
        minimum_density = math.inf
    graphon_row = abs(float(mfg.get("max_kernel_row_residual", math.inf)))
    graphon_radius = 1.0 if graphon_row <= 1.0e-10 else math.nan

    fixed = abs(float(mfg.get("last_fixed_point_residual", math.inf)))
    fixed_max = abs(float(mfg.get("max_fixed_point_residual", math.inf)))
    contraction = fixed / max(fixed_max, 1.0e-30)

    outer = abs(float(macro.get("max_outer_residual", math.inf)))
    cge = abs(float(macro.get("max_cge_solver_residual", math.inf)))
    dsge = abs(float(macro.get("max_dsge_solver_residual", math.inf)))
    likelihood = abs(float(rare.get("max_likelihood_identity_residual", 0.0)))
    empirical_mass = abs(float(rare.get("max_empirical_measure_mass_residual", 0.0)))

    values = (
        weak_mass, minimum_density, graphon_row, graphon_radius, fixed,
        contraction, outer, cge, dsge, likelihood, empirical_mass,
    )
    finite = all(math.isfinite(value) for value in values)
    passed = bool(
        finite
        and weak_mass <= 1.0e-10
        and minimum_density >= -1.0e-12
        and graphon_row <= 1.0e-10
        and 0.0 <= contraction <= 1.05
        and fixed <= 5.0e-4
        and outer <= 3.1e-3
        and cge <= 2.1e-3
        and dsge <= 8.1e-4
        and likelihood <= 1.0e-12
        and empirical_mass <= 1.0e-12
    )
    return FunctionalAnalysisReport(
        weak_mass_residual=weak_mass,
        minimum_density=minimum_density,
        graphon_row_residual=graphon_row,
        graphon_markov_spectral_radius=graphon_radius,
        mfg_fixed_point_residual=fixed,
        mfg_empirical_residual_contraction=contraction,
        macro_outer_residual=outer,
        macro_cge_residual=cge,
        macro_dsge_residual=dsge,
        rare_event_likelihood_residual=likelihood,
        rare_event_empirical_mass_residual=empirical_mass,
        all_values_finite=finite,
        passed=passed,
    )


def attach_functional_diagnostics(diagnostics: dict) -> dict:
    failures: list[str] = []
    for name, cell in diagnostics.items():
        report = analyse_cell(cell)
        cell["functional_analysis"] = report.as_dict()
        if not report.passed:
            failures.append(
                f"{name} (weak={report.weak_mass_residual:.3g}, "
                f"density={report.minimum_density:.3g}, "
                f"graphon={report.graphon_row_residual:.3g}, "
                f"mfg={report.mfg_fixed_point_residual:.3g}, "
                f"contraction={report.mfg_empirical_residual_contraction:.3g}, "
                f"outer={report.macro_outer_residual:.3g}, "
                f"cge={report.macro_cge_residual:.3g}, "
                f"dsge={report.macro_dsge_residual:.3g}, "
                f"likelihood={report.rare_event_likelihood_residual:.3g}, "
                f"mass={report.rare_event_empirical_mass_residual:.3g})"
            )
    if failures:
        raise RuntimeError(
            "V6.8 functional-analysis validation failed for: "
            + ", ".join(failures)
        )
    return diagnostics
