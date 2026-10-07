from __future__ import annotations

from dataclasses import dataclass

import numpy as np


CAUSES = ("force", "peace", "separation")


@dataclass
class SurvivalFactors:
    cumulative_incidence: np.ndarray
    survival: np.ndarray
    cause_specific_hazard: np.ndarray
    fine_gray_hazard: np.ndarray
    aft_delay_months: np.ndarray
    simplex_residual: float
    calibration_residual: float
    cumulative_incidence_history: np.ndarray
    survival_history: np.ndarray
    cause_increment: np.ndarray


class StratifiedCompetingRiskSurvival:
    """Monthly stratified Cox risks, Fine-Gray CIFs, and an AFT delay model."""

    def __init__(self, paths: int, seed: int):
        self.paths = paths
        self.rng = np.random.default_rng(seed)
        self.last_result: SurvivalFactors | None = None

    @staticmethod
    def _integrate(hazards: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        cif, survival, fine_gray, _, _, _ = StratifiedCompetingRiskSurvival._integrate_history(hazards)
        return cif, survival, fine_gray

    @staticmethod
    def _integrate_history(
        hazards: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        paths, months, causes = hazards.shape
        survival = np.ones(paths)
        cif = np.zeros((paths, causes))
        previous_cif = np.zeros_like(cif)
        fine_gray = np.zeros_like(hazards)
        cif_history = np.zeros_like(hazards)
        survival_history = np.zeros((paths, months))
        increments = np.zeros_like(hazards)
        for month in range(months):
            total = np.sum(hazards[:, month, :], axis=1)
            exit_probability = 1.0 - np.exp(-total)
            shares = hazards[:, month, :] / np.maximum(total[:, None], 1e-12)
            increment = survival[:, None] * exit_probability[:, None] * shares
            increments[:, month, :] = increment
            cif += increment
            denominator = np.maximum(1.0 - previous_cif, 1e-12)
            fine_gray[:, month, :] = -np.log(np.maximum(1.0 - increment / denominator, 1e-12))
            previous_cif = cif.copy()
            survival *= np.exp(-total)
            cif_history[:, month, :] = cif
            survival_history[:, month] = survival
        return cif, survival, fine_gray, cif_history, survival_history, increments

    def evaluate(
        self, *, actor_support: np.ndarray, taiwan_readiness: np.ndarray,
        leader_continuity: np.ndarray, fiscal_capacity: np.ndarray,
        financial_stress: np.ndarray, arms_support_us: np.ndarray,
        arms_support_japan: np.ndarray, target_force_probability: float,
        months: int = 60, pressure_belief: np.ndarray | None = None,
    ) -> SurvivalFactors:
        support = np.clip(actor_support, 0.01, 0.99)
        readiness = np.clip(taiwan_readiness, 0.35, 1.65)
        continuity = np.clip(leader_continuity, 0.10, 1.0)
        fiscal = np.clip(fiscal_capacity, 0.05, 1.0)
        stress = np.clip(financial_stress, 0.0, 1.0)
        target = float(np.clip(target_force_probability, 1e-6, 0.96))
        if pressure_belief is None:
            pressure = np.zeros(self.paths)
        else:
            pressure_state = np.asarray(pressure_belief, dtype=float)
            pressure = np.clip(pressure_state[:, 1] + 2.0 * pressure_state[:, 2], 0.0, 2.0)

        # Structural strata alter baseline risks without assuming proportional
        # baselines across institutionally distinct path classes.
        weak_taiwan = readiness < np.median(readiness)
        disrupted_leader = continuity[:, 0] < 0.80
        strata = weak_taiwan.astype(int) + 2 * disrupted_leader.astype(int)
        force_stratum = np.array([0.86, 1.12, 1.28, 1.58])[strata]
        peace_stratum = np.array([1.10, 0.92, 0.88, 0.72])[strata]
        separation_stratum = np.array([0.92, 1.16, 0.84, 1.04])[strata]

        x_force = (
            1.10 * (support[:, 0] - 0.50)
            + 0.58 * (0.58 - support[:, 1])
            - 0.66 * (support[:, 2] - 0.42)
            + 0.42 * (1.0 - continuity[:, 0])
            - 0.44 * np.log(readiness)
            + 0.30 * stress[:, 0] - 0.24 * fiscal[:, 0] + 0.38 * pressure
        )
        x_peace = (
            0.74 * (support[:, 0] - 0.50) + 0.82 * (support[:, 1] - 0.50)
            + 0.40 * (support[:, 2] - 0.42) + 0.28 * readiness
            - 0.34 * stress[:, 0] - 0.16 * pressure
        )
        x_separation = (
            0.92 * (0.58 - support[:, 1]) + 0.48 * (support[:, 2] - 0.42)
            + 0.35 * readiness + 0.22 * fiscal[:, 1] + 0.12 * pressure
        )
        t = np.arange(months, dtype=float)[None, :]
        time_force = 0.78 + 0.44 * (t / max(months - 1, 1)) ** 1.35
        time_peace = 0.90 + 0.20 * np.sqrt(t / max(months - 1, 1))
        time_separation = 0.72 + 0.50 * (t / max(months - 1, 1)) ** 1.60

        peace = (
            0.00034 * peace_stratum[:, None] * np.exp(x_peace[:, None]) * time_peace
        )
        separation = (
            0.00025 * separation_stratum[:, None]
            * np.exp(x_separation[:, None]) * time_separation
        )

        def hazards_for(log_scale: float) -> np.ndarray:
            force = (
                np.exp(log_scale) * force_stratum[:, None]
                * np.exp(x_force[:, None]) * time_force
            )
            return np.stack((force, peace, separation), axis=2)

        low, high = -16.0, -1.0
        for _ in range(70):
            midpoint = 0.5 * (low + high)
            cif_mid, _, _ = self._integrate(hazards_for(midpoint))
            if float(np.mean(cif_mid[:, 0])) < target:
                low = midpoint
            else:
                high = midpoint
        hazards = hazards_for(0.5 * (low + high))
        cif, survival, fine_gray, cif_history, survival_history, increments = self._integrate_history(hazards)

        aft_noise = self.rng.normal(0.0, 0.34, self.paths)
        log_delay = (
            np.log(2.2) - 0.55 * arms_support_us - 0.25 * arms_support_japan
            - 0.32 * support[:, 2] - 0.18 * support[:, 3]
            + 0.52 * stress[:, 2] + 0.30 * (1.0 - fiscal[:, 2])
            + 0.35 * (1.0 - continuity[:, 2]) - 0.28 * np.log(readiness)
            + aft_noise
        )
        delay = np.clip(np.exp(log_delay), 0.10, 18.0)
        simplex_residual = float(np.max(np.abs(np.sum(cif, axis=1) + survival - 1.0)))
        calibration_residual = abs(float(np.mean(cif[:, 0])) - target)
        self.last_result = SurvivalFactors(
            cumulative_incidence=cif,
            survival=survival,
            cause_specific_hazard=hazards,
            fine_gray_hazard=fine_gray,
            aft_delay_months=delay,
            simplex_residual=simplex_residual,
            calibration_residual=calibration_residual,
            cumulative_incidence_history=cif_history,
            survival_history=survival_history,
            cause_increment=increments,
        )
        return self.last_result

    def aft_delay_at_onset(
        self, *, actor_support: np.ndarray, taiwan_readiness: np.ndarray,
        leader_continuity: np.ndarray, arms_support_us: np.ndarray,
        arms_support_japan: np.ndarray, fiscal_capacity: np.ndarray,
        financial_stress: np.ndarray,
    ) -> np.ndarray:
        noise = self.rng.normal(0.0, 0.34, self.paths)
        log_delay = (
            np.log(2.2) - 0.55 * arms_support_us - 0.25 * arms_support_japan
            - 0.32 * actor_support[:, 2] - 0.18 * actor_support[:, 3]
            + 0.52 * financial_stress[:, 2] + 0.30 * (1.0 - fiscal_capacity[:, 2])
            + 0.35 * (1.0 - leader_continuity[:, 2])
            - 0.28 * np.log(np.clip(taiwan_readiness, 0.35, 1.65)) + noise
        )
        return np.clip(np.exp(log_delay), 0.10, 18.0)

    def evaluate_trajectory(
        self, *, actor_support: np.ndarray, taiwan_readiness: np.ndarray,
        leader_continuity: np.ndarray, pressure_belief: np.ndarray,
        economic_capital: np.ndarray, interdependence: np.ndarray,
        target_force_probability: float,
        calibration_months: int = 60,
    ) -> SurvivalFactors:
        """Time-varying 2026-2100 competing risks on the same prewar paths."""
        months, paths, _ = actor_support.shape
        if paths != self.paths:
            raise ValueError("trajectory path count does not match survival system")
        support = np.clip(actor_support, 0.01, 0.99)
        readiness = np.clip(taiwan_readiness, 0.35, 1.65)
        continuity = np.clip(leader_continuity, 0.10, 1.0)
        pressure = np.clip(pressure_belief[:, :, 1] + 2.0 * pressure_belief[:, :, 2], 0.0, 2.0)
        economic = np.clip(economic_capital, 0.10, 12.0)
        linkage = np.clip(interdependence, 0.10, 1.50)
        fiscal = np.clip(0.58 + 0.16 * np.log(economic) + 0.12 * continuity, 0.05, 1.0)
        stress = np.clip(0.28 - 0.10 * np.log(economic) + 0.18 * (1.0 - continuity), 0.0, 1.0)

        weak_taiwan = readiness < np.median(readiness, axis=1, keepdims=True)
        disrupted_leader = continuity[:, :, 0] < 0.80
        strata = weak_taiwan.astype(int) + 2 * disrupted_leader.astype(int)
        force_stratum = np.take(np.array([0.86, 1.12, 1.28, 1.58]), strata)
        peace_stratum = np.take(np.array([1.10, 0.92, 0.88, 0.72]), strata)
        separation_stratum = np.take(np.array([0.92, 1.16, 0.84, 1.04]), strata)
        x_force = (
            1.10 * (support[:, :, 0] - 0.50)
            + 0.58 * (0.58 - support[:, :, 1])
            - 0.66 * (support[:, :, 2] - 0.42)
            + 0.42 * (1.0 - continuity[:, :, 0])
            - 0.44 * np.log(readiness)
            + 0.30 * stress[:, :, 0] - 0.24 * fiscal[:, :, 0] + 0.38 * pressure
            - 0.24 * (linkage - 1.0)
        )
        x_peace = (
            0.74 * (support[:, :, 0] - 0.50) + 0.82 * (support[:, :, 1] - 0.50)
            + 0.40 * (support[:, :, 2] - 0.42) + 0.28 * readiness
            - 0.34 * stress[:, :, 0] - 0.16 * pressure + 0.30 * (linkage - 1.0)
        )
        x_separation = (
            0.92 * (0.58 - support[:, :, 1]) + 0.48 * (support[:, :, 2] - 0.42)
            + 0.35 * readiness + 0.22 * fiscal[:, :, 1] + 0.12 * pressure
            - 0.10 * (linkage - 1.0)
        )
        elapsed = np.arange(months, dtype=float)[:, None] / max(months - 1, 1)
        time_force = 0.78 + 0.44 * elapsed ** 1.35
        time_peace = 0.90 + 0.20 * np.sqrt(elapsed)
        time_separation = 0.72 + 0.50 * elapsed ** 1.60
        # These are 75-year monthly structural priors. Reusing the old
        # five-year baselines here would multiply cumulative incidence by the
        # horizon ratio and create a spurious terminal result.
        peace = 0.000035 * peace_stratum * np.exp(x_peace) * time_peace
        separation = 0.000075 * separation_stratum * np.exp(x_separation) * time_separation

        def hazards_for(log_scale: float) -> np.ndarray:
            force = np.exp(log_scale) * force_stratum * np.exp(x_force) * time_force
            return np.stack((force, peace, separation), axis=2).transpose(1, 0, 2)

        calibration_index = min(max(calibration_months, 1), months) - 1
        target = float(np.clip(target_force_probability, 1e-6, 0.96))
        low, high = -18.0, -1.0
        for _ in range(70):
            midpoint = 0.5 * (low + high)
            _, _, _, history, _, _ = self._integrate_history(hazards_for(midpoint))
            if float(np.mean(history[:, calibration_index, 0])) < target:
                low = midpoint
            else:
                high = midpoint
        hazards = hazards_for(0.5 * (low + high))
        cif, survival, fine_gray, history, survival_history, increments = self._integrate_history(hazards)
        calibration_residual = abs(float(np.mean(history[:, calibration_index, 0])) - target)
        # AFT delay is evaluated at the conditionally sampled onset state by
        # the alliance model; this placeholder is replaced in the driver.
        delay = np.ones(paths)
        simplex_residual = float(np.max(np.abs(np.sum(cif, axis=1) + survival - 1.0)))
        self.last_result = SurvivalFactors(
            cumulative_incidence=cif, survival=survival,
            cause_specific_hazard=hazards, fine_gray_hazard=fine_gray,
            aft_delay_months=delay, simplex_residual=simplex_residual,
            calibration_residual=calibration_residual,
            cumulative_incidence_history=history,
            survival_history=survival_history,
            cause_increment=increments,
        )
        return self.last_result

    def diagnostics(self) -> dict:
        if self.last_result is None:
            return {"evaluated": False}
        result = self.last_result
        return {
            "evaluated": True,
            "simplex_residual": result.simplex_residual,
            "force_calibration_residual": result.calibration_residual,
            "minimum_cause_specific_hazard": float(np.min(result.cause_specific_hazard)),
            "minimum_fine_gray_hazard": float(np.min(result.fine_gray_hazard)),
            "aft_delay_p10_p50_p90": [
                float(x) for x in np.quantile(result.aft_delay_months, (0.1, 0.5, 0.9))
            ],
        }
