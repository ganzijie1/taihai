from __future__ import annotations

from dataclasses import replace

import numpy as np

from information_geometry_v58 import (
    ACTORS,
    InformationGeometryFactors,
    PartiallyObservableInformationGeometry,
    _normalise,
)


class DynamicCausalOptimalTransportGeometry(PartiallyObservableInformationGeometry):
    """Causal V6.2 transport geometry for beliefs and multilayer networks.

    The bridge estimated in month t becomes the Markov kernel used at t+1.
    FGW alignment is refreshed from current/past observations only. The DRO
    radius is a rolling upper quantile of past forecast errors, so no future
    outcome enters a current decision.
    """

    def __init__(self, *args, fgw_refresh_interval: int = 3,
                 dro_window: int = 24, **kwargs):
        super().__init__(*args, **kwargs)
        self.fgw_refresh_interval = max(int(fgw_refresh_interval), 1)
        self.dro_window = max(int(dro_window), 6)
        self.dynamic_fgw_alignment = np.broadcast_to(
            self.fgw_alignment, (self.paths, len(ACTORS), len(ACTORS))
        ).copy()
        self._error_history: list[np.ndarray] = []
        self._pending_transition = self.transition.copy()
        self.fgw_refresh_count = 0
        self.max_dynamic_fgw_change = 0.0
        self.max_bridge_transition_residual = 0.0
        self.min_online_radius = np.inf
        self.max_online_radius = 0.0
        self.radius_exceedances = 0
        self.radius_observations = 0

    def _schrodinger_bridge(self, opening: np.ndarray,
                            closing: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        cost, coupling = super()._schrodinger_bridge(opening, closing)
        conditional = coupling / np.maximum(opening[..., :, None], 1.0e-14)
        conditional = _normalise(conditional, axis=-1)
        weights = opening / np.maximum(np.sum(opening, axis=(0, 1, 2)), 1.0e-14)
        aggregate = np.sum(weights[..., :, None] * conditional, axis=(0, 1, 2))
        aggregate = _normalise(aggregate, axis=-1)
        self._pending_transition = aggregate
        residual = float(np.max(np.abs(np.sum(aggregate, axis=1) - 1.0)))
        self.max_bridge_transition_residual = max(
            self.max_bridge_transition_residual, residual
        )
        return cost, coupling

    def _refresh_dynamic_fgw(self, factors: InformationGeometryFactors,
                             network: np.ndarray, jump_signal: np.ndarray) -> None:
        means = factors.posterior_mean
        credibility = np.mean(factors.credibility, axis=2)
        feature_gap = np.abs(means[:, :, :, None] - means[:, :, None, :]).mean(axis=1)
        credibility_gap = np.abs(
            credibility[:, :, None] - credibility[:, None, :]
        )
        structural = np.broadcast_to(self.fgw_cost, feature_gap.shape).copy()
        structural /= max(float(np.median(self.fgw_cost[self.fgw_cost > 0])), 1.0e-6)
        fused_cost = 0.45 * feature_gap + 0.35 * credibility_gap + 0.20 * structural
        event_scale = 1.0 + 0.25 * np.mean(jump_signal, axis=1)[:, None, None]
        candidate = np.exp(-event_scale * fused_cost)
        candidate *= np.maximum(self.dynamic_fgw_alignment, 1.0e-6) ** 0.35
        candidate *= (0.82 + 0.18 * network[:, :, None])
        candidate = _normalise(candidate, axis=2)
        change = float(np.max(np.abs(candidate - self.dynamic_fgw_alignment)))
        self.max_dynamic_fgw_change = max(self.max_dynamic_fgw_change, change)
        self.dynamic_fgw_alignment = 0.65 * self.dynamic_fgw_alignment + 0.35 * candidate
        self.dynamic_fgw_alignment = _normalise(self.dynamic_fgw_alignment, axis=2)
        self.fgw_refresh_count += 1

    def update(self, **kwargs) -> InformationGeometryFactors:
        if self._error_history:
            history = np.stack(self._error_history[-self.dro_window:], axis=0)
            if hasattr(self, "_causal_conformal_radius"):
                self.ambiguity_radius_base = self._causal_conformal_radius(history)
            else:
                online_radius = np.quantile(history, 0.90, axis=0)
                self.ambiguity_radius_base = float(np.clip(
                    np.median(online_radius), 0.008, 0.12
                ))
        factors = super().update(**kwargs)
        # Activate the entropic bridge only after the current posterior update.
        self.transition = self._pending_transition.copy()
        error = np.abs(factors.self_belief - factors.self_truth)
        if self._error_history:
            radius = max(self.ambiguity_radius_base, 1.0e-8)
            self.radius_exceedances += int(np.sum(error > radius))
            self.radius_observations += error.size
        self._error_history.append(error.copy())
        if len(self._error_history) > self.dro_window:
            self._error_history.pop(0)
        event_refresh = float(np.max(kwargs["jump_signal"])) > 0.35
        if self.month % self.fgw_refresh_interval == 0 or event_refresh:
            self._refresh_dynamic_fgw(
                factors, kwargs["network"], kwargs["jump_signal"]
            )
        self.min_online_radius = min(
            self.min_online_radius, self.ambiguity_radius_base
        )
        self.max_online_radius = max(
            self.max_online_radius, self.ambiguity_radius_base
        )
        return replace(
            factors,
            fgw_alignment=self.dynamic_fgw_alignment.copy(),
        )

    def diagnostics(self) -> dict:
        result = super().diagnostics()
        result["v62_dynamic_transport"] = {
            "causal_clock": "current-and-past only; bridge kernel enters at t+1",
            "dynamic_fgw_refresh_interval_months": self.fgw_refresh_interval,
            "dynamic_fgw_refresh_count": self.fgw_refresh_count,
            "max_dynamic_fgw_change": self.max_dynamic_fgw_change,
            "max_bridge_transition_residual": self.max_bridge_transition_residual,
            "online_dro_window_months": self.dro_window,
            "minimum_online_radius": (
                0.0 if not np.isfinite(self.min_online_radius)
                else self.min_online_radius
            ),
            "maximum_online_radius": self.max_online_radius,
            "historical_radius_exceedance_rate": (
                self.radius_exceedances / max(self.radius_observations, 1)
            ),
            "fgw_pathwise": True,
            "future_information_used": False,
        }
        return result
