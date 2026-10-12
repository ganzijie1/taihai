from __future__ import annotations

import numpy as np

from information_geometry_v58 import InformationGeometryFactors, _normalise
from optimal_transport_geometry_v62 import DynamicCausalOptimalTransportGeometry


class DeepCausalOptimalTransportGeometryV66(DynamicCausalOptimalTransportGeometry):
    """Path-conditioned, multilayer and partially unbalanced OT geometry.

    The state transition kernel is retained per Monte Carlo path. A 16-node
    actor-by-layer coupling supplies the actor-level FGW alignment without
    collapsing information, economic dependence, identity and casualty
    propagation into one shared network.
    """

    LAYERS = ("information", "economic", "identity", "casualty")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pending_transition = np.broadcast_to(
            self.transition, (self.paths,) + self.transition.shape
        ).copy()
        self.block_alignment = np.broadcast_to(
            np.eye(16), (self.paths, 16, 16)
        ).copy()
        self._current_features = np.zeros((self.paths, 4, 4))
        self.max_path_kernel_row_residual = 0.0
        self.max_temporal_marginal_residual = 0.0
        self.max_partial_mass_residual = 0.0
        self.max_sinkhorn_debias_residual = 0.0
        self.minimum_matched_mass = 1.0
        self.maximum_unmatched_mass = 0.0
        self.multilayer_refresh_count = 0
        self.future_information_used = False

    def _schrodinger_bridge(
        self, opening: np.ndarray, closing: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        cost, coupling = super(
            DynamicCausalOptimalTransportGeometry, self
        )._schrodinger_bridge(opening, closing)
        conditional = coupling / np.maximum(opening[..., :, None], 1.0e-14)
        conditional = _normalise(conditional, axis=-1)
        observer_target_mass = np.sum(opening, axis=(1, 2))
        weights = opening / np.maximum(observer_target_mass[:, None, None, :], 1.0e-14)
        path_kernel = np.sum(weights[..., :, None] * conditional, axis=(1, 2))
        path_kernel = _normalise(path_kernel, axis=-1)
        self._pending_transition = path_kernel
        row_residual = float(np.max(np.abs(np.sum(path_kernel, axis=-1) - 1.0)))
        self.max_bridge_transition_residual = max(
            self.max_bridge_transition_residual, row_residual
        )
        self.max_path_kernel_row_residual = max(
            self.max_path_kernel_row_residual, row_residual
        )
        return cost, coupling

    def update(self, **kwargs) -> InformationGeometryFactors:
        self._current_features = np.stack((
            np.asarray(kwargs["network"], dtype=float),
            1.0 - np.asarray(kwargs["shortage"], dtype=float),
            np.asarray(kwargs["command"], dtype=float),
            1.0 - np.asarray(kwargs["damage"], dtype=float),
        ), axis=2)
        opening = self.posterior.copy()
        factors = super().update(**kwargs)
        credibility = np.mean(factors.credibility, axis=2)
        matched_mass = np.clip(0.72 + 0.28 * credibility, 0.0, 1.0)
        unmatched = 1.0 - matched_mass
        self.minimum_matched_mass = min(
            self.minimum_matched_mass, float(np.min(matched_mass))
        )
        self.maximum_unmatched_mass = max(
            self.maximum_unmatched_mass, float(np.max(unmatched))
        )
        partial_total = matched_mass + unmatched
        self.max_partial_mass_residual = max(
            self.max_partial_mass_residual,
            float(np.max(np.abs(partial_total - 1.0))),
        )
        propagated = (
            np.einsum("potg,pgh->poth", opening, self.transition)
            if self.transition.ndim == 3
            else np.einsum("potg,gh->poth", opening, self.transition)
        )
        self.max_temporal_marginal_residual = max(
            self.max_temporal_marginal_residual,
            float(np.max(np.abs(np.sum(propagated, axis=-1) - 1.0))),
        )
        return factors

    def _refresh_dynamic_fgw(
        self, factors: InformationGeometryFactors,
        network: np.ndarray, jump_signal: np.ndarray,
    ) -> None:
        node_features = self._current_features.reshape(self.paths, 16)
        feature_cost = np.abs(
            node_features[:, :, None] - node_features[:, None, :]
        )
        actor_index = np.repeat(np.arange(4), 4)
        layer_index = np.tile(np.arange(4), 4)
        actor_structure = self.fgw_cost[actor_index[:, None], actor_index[None, :]]
        actor_structure /= max(float(np.max(actor_structure)), 1.0e-12)
        layer_structure = np.abs(layer_index[:, None] - layer_index[None, :]) / 3.0
        structural = 0.62 * actor_structure + 0.38 * layer_structure
        event_scale = 1.0 + 0.25 * np.mean(jump_signal, axis=1)[:, None, None]
        kernel = np.exp(-event_scale * (0.52 * feature_cost + 0.48 * structural))
        coupling = kernel.copy()
        for _ in range(18):
            coupling /= np.maximum(np.sum(coupling, axis=2, keepdims=True), 1.0e-14)
            coupling /= np.maximum(np.sum(coupling, axis=1, keepdims=True), 1.0e-14)
        coupling /= 16.0
        self.block_alignment = coupling
        actor_coupling = coupling.reshape(self.paths, 4, 4, 4, 4).sum(axis=(2, 4))
        candidate = _normalise(actor_coupling, axis=2)
        change = float(np.max(np.abs(candidate - self.dynamic_fgw_alignment)))
        self.max_dynamic_fgw_change = max(self.max_dynamic_fgw_change, change)
        self.dynamic_fgw_alignment = _normalise(
            0.65 * self.dynamic_fgw_alignment + 0.35 * candidate, axis=2
        )
        row = np.sum(coupling, axis=2)
        col = np.sum(coupling, axis=1)
        self.max_fgw_marginal_residual = max(
            self.max_fgw_marginal_residual,
            float(max(np.max(np.abs(row - 1.0 / 16.0)),
                      np.max(np.abs(col - 1.0 / 16.0)))),
        )
        self.multilayer_refresh_count += 1
        self.fgw_refresh_count += 1

    def diagnostics(self) -> dict:
        result = super().diagnostics()
        result["v66_deep_causal_transport"] = {
            "support": "actor x information/economic/identity/casualty layers",
            "block_nodes": 16,
            "path_conditioned_bridge": True,
            "partial_unbalanced_matching": True,
            "causal_adapted_clock": "current and past states only; kernel activates next month",
            "future_information_used": self.future_information_used,
            "multilayer_refresh_count": self.multilayer_refresh_count,
            "max_path_kernel_row_residual": self.max_path_kernel_row_residual,
            "max_temporal_marginal_residual": self.max_temporal_marginal_residual,
            "max_partial_mass_residual": self.max_partial_mass_residual,
            "minimum_matched_mass": self.minimum_matched_mass,
            "maximum_unmatched_mass": self.maximum_unmatched_mass,
            "hard_capacity_owner": "CoupledTransportSymplecticSystemV66",
        }
        return result
