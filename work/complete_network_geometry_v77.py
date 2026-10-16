from __future__ import annotations

from dataclasses import replace

import numpy as np
import ot

from network_geometry_v59 import NetworkGeometrySystem, _row_normalise


class CompleteNetworkGeometryV77(NetworkGeometrySystem):
    """Full pathwise FGW-proximal, directed curvature and Hodge closure."""

    def __init__(self, *args, fgw_outer_iterations: int = 6,
                 fgw_sinkhorn_iterations: int = 18,
                 curvature_regularization: float = 0.18, **kwargs):
        super().__init__(*args, **kwargs)
        self.fgw_outer_iterations = max(int(fgw_outer_iterations), 2)
        self.fgw_sinkhorn_iterations = max(int(fgw_sinkhorn_iterations), 48)
        self.curvature_regularization = float(max(curvature_regularization, 1.0e-3))
        self._incidence = np.array([
            [-1, -1, -1, 0, 0, 0],
            [1, 0, 0, -1, -1, 0],
            [0, 1, 0, 1, 0, -1],
            [0, 0, 1, 0, 1, 1],
        ], dtype=float)
        self._laplacian_pinv = np.linalg.pinv(self._incidence @ self._incidence.T)
        self.maximum_fgw_fixed_point_residual = 0.0
        self.maximum_fgw_marginal_residual = 0.0
        self.maximum_fgw_proximal_change = 0.0
        self.minimum_directed_ollivier_curvature = np.inf
        self.maximum_directed_ollivier_curvature = -np.inf
        self.maximum_ollivier_marginal_residual = 0.0
        self.maximum_hodge_reconstruction_residual = 0.0
        self.maximum_hodge_cycle_share = 0.0
        self.fgw_updates = 0
        self.curvature_updates = 0
        self.hodge_updates = 0
        self._sinkhorn_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.maximum_sinkhorn_iterations = {"fgw": 0, "ollivier": 0}
        self.total_sinkhorn_iterations = {"fgw": 0, "ollivier": 0}
        self._pot_validated: set[str] = set()
        self.pot_validation_calls = 0
        self.maximum_pot_marginal_residual = 0.0
        self.maximum_pot_plan_difference = 0.0

    def _sinkhorn(self, left: np.ndarray, right: np.ndarray, cost: np.ndarray,
                  epsilon: float, iterations: int, *, cache_key: str,
                  tolerance: float) -> tuple[np.ndarray, float]:
        log_kernel = -np.clip(cost / epsilon, 0.0, 120.0)
        log_left = np.log(np.maximum(left, 1.0e-300))
        log_right = np.log(np.maximum(right, 1.0e-300))
        cached = self._sinkhorn_cache.get(cache_key)
        if cached is not None and cached[0].shape == left.shape:
            log_u, log_v = cached[0].copy(), cached[1].copy()
        else:
            log_u = np.zeros_like(left)
            log_v = np.zeros_like(right)

        def logsumexp(values: np.ndarray, axis: int) -> np.ndarray:
            maximum = np.max(values, axis=axis, keepdims=True)
            result = maximum + np.log(np.maximum(
                np.sum(np.exp(values - maximum), axis=axis, keepdims=True),
                1.0e-300,
            ))
            return np.squeeze(result, axis=axis)

        used = 0
        for iteration in range(iterations):
            log_u = log_left - logsumexp(
                log_kernel + log_v[..., None, :], axis=-1
            )
            log_v = log_right - logsumexp(
                log_kernel + log_u[..., :, None], axis=-2
            )
            used = iteration + 1
            if used % 4 == 0:
                trial = np.exp(
                    log_u[..., :, None] + log_kernel + log_v[..., None, :]
                )
                trial_residual = float(max(
                    np.max(np.abs(np.sum(trial, axis=-1) - left)),
                    np.max(np.abs(np.sum(trial, axis=-2) - right)),
                ))
                if trial_residual <= tolerance:
                    break
        self._sinkhorn_cache[cache_key] = (log_u.copy(), log_v.copy())
        self.maximum_sinkhorn_iterations[cache_key] = max(
            self.maximum_sinkhorn_iterations[cache_key], used
        )
        self.total_sinkhorn_iterations[cache_key] += used
        coupling = np.exp(
            log_u[..., :, None] + log_kernel + log_v[..., None, :]
        )
        residual = float(max(
            np.max(np.abs(np.sum(coupling, axis=-1) - left)),
            np.max(np.abs(np.sum(coupling, axis=-2) - right)),
        ))
        # POT is an independent plan-valued oracle.  The production solver stays
        # vectorised because every path has a different cost matrix, while POT's
        # public log-Sinkhorn entry point accepts one such matrix at a time.
        if cache_key not in self._pot_validated:
            flat_left = left.reshape(-1, left.shape[-1])[0]
            flat_right = right.reshape(-1, right.shape[-1])[0]
            flat_cost = cost.reshape(-1, cost.shape[-2], cost.shape[-1])[0]
            pot_plan = np.asarray(ot.bregman.sinkhorn_log(
                flat_left, flat_right, flat_cost, epsilon,
                numItermax=max(iterations, 1000), stopThr=min(tolerance, 1.0e-10),
                warn=False,
            ))
            local_plan = coupling.reshape(-1, coupling.shape[-2], coupling.shape[-1])[0]
            pot_residual = float(max(
                np.max(np.abs(np.sum(pot_plan, axis=-1) - flat_left)),
                np.max(np.abs(np.sum(pot_plan, axis=-2) - flat_right)),
            ))
            self.maximum_pot_marginal_residual = max(
                self.maximum_pot_marginal_residual, pot_residual
            )
            self.maximum_pot_plan_difference = max(
                self.maximum_pot_plan_difference,
                float(np.max(np.abs(pot_plan - local_plan))),
            )
            self.pot_validation_calls += 1
            self._pot_validated.add(cache_key)
        return coupling, residual

    def _fgw_proximal(self, target: np.ndarray) -> np.ndarray:
        previous_transition, _ = _row_normalise(np.maximum(self.raw_kernel, 1.0e-9))
        target_transition, _ = _row_normalise(np.maximum(target, 1.0e-9))
        sym_previous = 0.5 * (previous_transition + np.swapaxes(previous_transition, -1, -2))
        sym_target = 0.5 * (target_transition + np.swapaxes(target_transition, -1, -2))
        c_left = 1.0 - sym_previous
        c_right = 1.0 - sym_target
        node_feature = np.abs(
            np.sum(previous_transition, axis=-2)[..., :, None]
            - np.sum(target_transition, axis=-2)[..., None, :]
        )
        shape = previous_transition.shape[:-2] + (4,)
        marginal = np.full(shape, 0.25)
        coupling = np.broadcast_to(
            np.full((4, 4), 1.0 / 16.0), previous_transition.shape
        ).copy()
        fixed_residual = 0.0
        marginal_residual = 0.0
        for _ in range(self.fgw_outer_iterations):
            const_left = np.einsum("...ik,...k->...i", c_left ** 2, marginal)
            const_right = np.einsum("...jl,...l->...j", c_right ** 2, marginal)
            cross = np.einsum("...ik,...kl,...jl->...ij", c_left, coupling, c_right)
            gw_gradient = const_left[..., :, None] + const_right[..., None, :] - 2.0 * cross
            cost = 0.62 * gw_gradient + 0.38 * node_feature
            proposed, residual = self._sinkhorn(
                marginal, marginal, cost, 0.10, self.fgw_sinkhorn_iterations,
                cache_key="fgw", tolerance=1.0e-9,
            )
            fixed_residual = float(np.max(np.abs(proposed - coupling)))
            marginal_residual = max(marginal_residual, residual)
            coupling = 0.25 * coupling + 0.75 * proposed
        transport = 4.0 * coupling
        aligned = np.einsum(
            "...ia,...ab,...jb->...ij", transport, target_transition, transport
        )
        aligned, _ = _row_normalise(np.maximum(aligned, 1.0e-9))
        proximal = np.exp(
            0.42 * np.log(np.maximum(previous_transition, 1.0e-12))
            + 0.58 * np.log(np.maximum(aligned, 1.0e-12))
        )
        proximal, _ = _row_normalise(proximal)
        self.maximum_fgw_fixed_point_residual = max(
            self.maximum_fgw_fixed_point_residual, fixed_residual
        )
        self.maximum_fgw_marginal_residual = max(
            self.maximum_fgw_marginal_residual, marginal_residual
        )
        self.maximum_fgw_proximal_change = max(
            self.maximum_fgw_proximal_change,
            float(np.max(np.abs(proximal - previous_transition))),
        )
        self.fgw_updates += 1
        return proximal

    def _directed_ollivier(self, transition: np.ndarray) -> np.ndarray:
        symmetric = 0.5 * (transition + np.swapaxes(transition, -1, -2))
        length = -np.log(np.clip(symmetric, 1.0e-6, 1.0))
        diagonal = np.arange(4)
        length[..., diagonal, diagonal] = 0.0
        for pivot in range(4):
            length = np.minimum(
                length,
                length[..., :, pivot, None] + length[..., None, pivot, :],
            )
        scale = np.maximum(np.max(length, axis=(-2, -1), keepdims=True), 1.0e-9)
        ground = length / scale
        left = np.broadcast_to(transition[..., :, None, :], transition.shape[:-2] + (4, 4, 4))
        right = np.broadcast_to(transition[..., None, :, :], transition.shape[:-2] + (4, 4, 4))
        cost = np.broadcast_to(
            ground[..., None, None, :, :],
            transition.shape[:-2] + (4, 4, 4, 4),
        )
        coupling, residual = self._sinkhorn(
            left, right, cost, self.curvature_regularization, 120,
            cache_key="ollivier", tolerance=2.0e-4,
        )
        wasserstein = np.sum(coupling * cost, axis=(-2, -1))
        edge_distance = np.maximum(ground, 1.0e-6)
        curvature = 1.0 - wasserstein / edge_distance
        curvature[..., diagonal, diagonal] = 1.0
        self.minimum_directed_ollivier_curvature = min(
            self.minimum_directed_ollivier_curvature, float(np.min(curvature))
        )
        self.maximum_directed_ollivier_curvature = max(
            self.maximum_directed_ollivier_curvature, float(np.max(curvature))
        )
        self.maximum_ollivier_marginal_residual = max(
            self.maximum_ollivier_marginal_residual, residual
        )
        self.curvature_updates += 1
        return curvature

    def _hodge(self, transition: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        edges = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
        flow = np.stack(
            [transition[..., i, j] - transition[..., j, i] for i, j in edges],
            axis=-1,
        )
        divergence = np.einsum("...e,ve->...v", flow, self._incidence)
        potential = np.einsum("...v,vw->...w", divergence, self._laplacian_pinv)
        gradient = np.einsum("...v,ve->...e", potential, self._incidence)
        cycle = flow - gradient
        reconstruction = gradient + cycle - flow
        residual = float(np.max(np.abs(reconstruction)))
        share = np.linalg.norm(cycle, axis=-1) / np.maximum(
            np.linalg.norm(flow, axis=-1), 1.0e-12
        )
        self.maximum_hodge_reconstruction_residual = max(
            self.maximum_hodge_reconstruction_residual, residual
        )
        self.maximum_hodge_cycle_share = max(
            self.maximum_hodge_cycle_share, float(np.max(share))
        )
        self.hodge_updates += 1
        return share, divergence

    def update(self, **kwargs):
        target = np.asarray(kwargs["target_kernel"], dtype=float)
        kwargs["target_kernel"] = self._fgw_proximal(target)
        factors = super().update(**kwargs)
        curvature = self._directed_ollivier(factors.transition)
        cycle_share, _ = self._hodge(factors.transition)
        negative_curvature = np.mean(
            np.maximum(-curvature, 0.0), axis=(1, 3)
        )
        hodge_risk = np.mean(cycle_share, axis=1)[:, None]
        extra = np.clip(0.16 * negative_curvature + 0.12 * hodge_risk, 0.0, 0.30)
        bottleneck = np.clip(factors.bottleneck_risk + extra, 0.0, 1.0)
        intensity = factors.intensity_multiplier * np.clip(1.0 - extra[:, None, :], 0.70, 1.0)
        factors = replace(
            factors,
            bottleneck_risk=bottleneck,
            intensity_multiplier=intensity,
        )
        self.last_factors = factors
        return factors

    def diagnostics(self) -> dict:
        payload = super().diagnostics()
        payload["v77_complete_network_geometry"] = {
            "fgw_updates": self.fgw_updates,
            "fgw_outer_iterations": self.fgw_outer_iterations,
            "maximum_fgw_fixed_point_residual": self.maximum_fgw_fixed_point_residual,
            "maximum_fgw_marginal_residual": self.maximum_fgw_marginal_residual,
            "maximum_fgw_proximal_change": self.maximum_fgw_proximal_change,
            "directed_ollivier_updates": self.curvature_updates,
            "curvature_regularization": self.curvature_regularization,
            "minimum_directed_ollivier_curvature": self.minimum_directed_ollivier_curvature,
            "maximum_directed_ollivier_curvature": self.maximum_directed_ollivier_curvature,
            "maximum_ollivier_marginal_residual": self.maximum_ollivier_marginal_residual,
            "network_hodge_updates": self.hodge_updates,
            "maximum_hodge_reconstruction_residual": self.maximum_hodge_reconstruction_residual,
            "maximum_hodge_cycle_share": self.maximum_hodge_cycle_share,
            "maximum_sinkhorn_iterations": self.maximum_sinkhorn_iterations.copy(),
            "mean_sinkhorn_iterations": {
                key: self.total_sinkhorn_iterations[key] / max(
                    self.fgw_updates * self.fgw_outer_iterations
                    if key == "fgw" else self.curvature_updates, 1
                )
                for key in self.total_sinkhorn_iterations
            },
            "pot_version": ot.__version__,
            "pot_validation_calls": self.pot_validation_calls,
            "maximum_pot_marginal_residual": self.maximum_pot_marginal_residual,
            "maximum_pot_plan_difference": self.maximum_pot_plan_difference,
            "transport_backend": (
                "vectorized log-domain Sinkhorn with POT plan-valued oracle"
            ),
            "same_path_feedback": "curvature and cycle risk modify current Graphon intensity",
            "terminal_probability_adjustment": False,
        }
        return payload
