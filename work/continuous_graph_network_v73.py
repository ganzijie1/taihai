from __future__ import annotations

import numpy as np

from algebraic_discrete_network_v72 import (
    AlgebraicDiscreteOperationalNetworkV72,
    DISJOINT_PATH_SETS,
    EDGE_DELAY,
    EDGE_HEAD,
    EDGE_TAIL,
    NODE_COUNT,
    PATHS,
    PATH_INCIDENCE,
    SINK,
    SOURCE,
)


EDGE_CELLS = 6
FIRM_TYPES = 8
FINITE_FIRMS = 16
PDE_SUBSTEPS = 8


class ContinuousGraphOperationalNetworkV73(AlgebraicDiscreteOperationalNetworkV72):
    """Continuous graph dynamics coupled to the V7.2 hard network solver.

    The continuous states own normalized edge occupancy, signal propagation,
    corridor control and firm-network availability.  Physical inventories and
    hard capacities remain uniquely owned by the inherited discrete solver.
    Continuous outputs modify the next monthly transition on the same Monte
    Carlo path; they never post-process terminal probabilities.
    """

    def __init__(self, paths: int, seed: int, terrain: dict, **kwargs) -> None:
        super().__init__(paths, seed, terrain, **kwargs)
        edges = len(EDGE_TAIL)
        self.edge_dx = 1.0 / EDGE_CELLS
        self.edge_occupancy = np.zeros((paths, 4, edges, EDGE_CELLS, 3))
        self.edge_signal = np.full((paths, 4, edges, EDGE_CELLS), 0.88)
        self.edge_signal_velocity = np.zeros_like(self.edge_signal)
        self.edge_control = np.full((paths, 4, edges, EDGE_CELLS), 0.84)
        self.last_continuous_capacity_gate = np.ones((paths, 4, edges))
        self.last_continuous_delay_factor = np.ones((paths, 4, edges))
        self.last_edge_outflow = np.zeros((paths, 4, edges, 3))
        self.last_effective_edge_capacity = self.edge_base_capacity.copy()

        index = np.arange(FIRM_TYPES)
        distance = np.abs(index[:, None] - index[None, :])
        sector = index // 3
        raw_kernel = (
            0.10 + 0.62 * np.exp(-distance / 1.8)
            + 0.24 * (sector[:, None] == sector[None, :])
        )
        np.fill_diagonal(raw_kernel, 0.32)
        self.firm_kernel_raw = raw_kernel
        uncertainty = self.rng.lognormal(
            0.0, 0.055, (paths, 4, FIRM_TYPES, FIRM_TYPES)
        )
        kernel = raw_kernel[None, None, :, :] * uncertainty
        self.firm_kernel = kernel / np.maximum(
            np.sum(kernel, axis=-1, keepdims=True), 1.0e-12
        )
        self.firm_state = np.clip(
            self.rng.normal(0.90, 0.025, (paths, 4, FIRM_TYPES)), 0.75, 0.98
        )
        self.finite_type = np.arange(FINITE_FIRMS) % FIRM_TYPES
        probability = np.clip(
            0.34 * raw_kernel[self.finite_type[:, None], self.finite_type[None, :]],
            0.0,
            0.72,
        )
        finite = self.rng.random((FINITE_FIRMS, FINITE_FIRMS)) < probability
        np.fill_diagonal(finite, False)
        for node in range(FINITE_FIRMS):
            if not np.any(finite[node]):
                finite[node, (node + 1) % FINITE_FIRMS] = True
        self.finite_adjacency = finite.astype(float)
        self.finite_kernel = self.finite_adjacency / np.maximum(
            np.sum(self.finite_adjacency, axis=1, keepdims=True), 1.0
        )
        self.finite_firm_state = self.firm_state[..., self.finite_type].copy()
        self.last_enterprise_gate = np.ones((paths, 4))

        self.maximum_metric_balance_residual = 0.0
        self.maximum_negative_density = 0.0
        self.maximum_cfl = 0.0
        self.maximum_signal_bound_violation = 0.0
        self.maximum_control_bound_violation = 0.0
        self.maximum_signal_limiter_correction = 0.0
        self.maximum_control_limiter_correction = 0.0
        self.maximum_finite_graphon_rollback_error = 0.0
        self.maximum_continuous_state_delta = 0.0
        self.minimum_continuous_capacity_gate = 1.0
        self.minimum_enterprise_gate = 1.0
        self.months_v73 = 0

    def _cascade_capacities(self, corridor, supply, need):
        base = self.edge_base_capacity
        self.edge_base_capacity = base * self.last_continuous_capacity_gate
        try:
            edge_capacity, path_capacity = super()._cascade_capacities(
                corridor, supply, need
            )
        finally:
            self.edge_base_capacity = base
        self.last_effective_edge_capacity = edge_capacity.copy()
        return edge_capacity, path_capacity

    def _temporal_metrics(self, edge_capacity: np.ndarray):
        integrity_penalty = (self.edge_integrity < 0.58).astype(int) + (
            self.edge_integrity < 0.30
        ).astype(int)
        continuum_penalty = (self.last_continuous_delay_factor > 1.30).astype(int) + (
            self.last_continuous_delay_factor > 1.75
        ).astype(int)
        delays = EDGE_DELAY[None, None, :] + integrity_penalty + continuum_penalty
        arrival = np.full((self.paths, 4, NODE_COUNT), 10_000, dtype=int)
        arrival[..., SOURCE] = 0
        for node in range(NODE_COUNT):
            for edge in np.flatnonzero(EDGE_TAIL == node):
                head = EDGE_HEAD[edge]
                candidate = arrival[..., node] + delays[..., edge]
                candidate = np.where(
                    edge_capacity[..., edge] > 1.0e-6, candidate, 10_000
                )
                arrival[..., head] = np.minimum(arrival[..., head], candidate)
        feasible_path = np.ones((self.paths, 4, len(PATHS)), dtype=bool)
        for path_index, path in enumerate(PATHS):
            path_list = list(path)
            feasible_path[..., path_index] = (
                np.sum(delays[..., path_list], axis=-1) <= 8
            ) & np.all(edge_capacity[..., path_list] > 1.0e-6, axis=-1)
        connectivity = np.zeros((self.paths, 4), dtype=int)
        for chosen in DISJOINT_PATH_SETS:
            feasible = np.all(feasible_path[..., list(chosen)], axis=-1)
            connectivity = np.maximum(
                connectivity, feasible.astype(int) * len(chosen)
            )
        return arrival[..., SINK].astype(float), connectivity

    def _finite_volume_step(
        self,
        density: np.ndarray,
        inflow: np.ndarray,
        speed: np.ndarray,
        diffusion: np.ndarray,
        loss: np.ndarray,
        dt: float,
    ) -> tuple[np.ndarray, np.ndarray, float, float]:
        faces = np.zeros(density.shape[:-2] + (EDGE_CELLS + 1, 3))
        faces[..., 0, :] = inflow
        gradient = (density[..., 1:, :] - density[..., :-1, :]) / self.edge_dx
        faces[..., 1:-1, :] = (
            speed[..., None, None] * density[..., :-1, :]
            - diffusion[..., None, None] * gradient
        )
        faces[..., -1, :] = speed[..., None] * density[..., -1, :]
        opening_mass = np.sum(density, axis=(-2, -1)) * self.edge_dx
        updated = density + dt / self.edge_dx * (
            faces[..., :-1, :] - faces[..., 1:, :]
        ) - dt * loss[..., None, None] * density
        negative = float(np.max(np.maximum(-updated, 0.0)))
        updated = np.maximum(updated, 0.0)
        closing_mass = np.sum(updated, axis=(-2, -1)) * self.edge_dx
        expected = opening_mass + dt * (
            np.sum(inflow, axis=-1)
            - np.sum(faces[..., -1, :], axis=-1)
            - loss * opening_mass
        )
        residual = float(np.max(np.abs(closing_mass - expected)))
        return updated, faces[..., -1, :], residual, negative

    def _update_metric_graph(
        self,
        command: np.ndarray,
        pnt: np.ndarray,
        damage: np.ndarray,
        interdiction: np.ndarray,
    ) -> None:
        edge_load = np.einsum(
            "...pc,pe->...ec", self.last_route_flow, PATH_INCIDENCE
        )
        capacity = np.maximum(self.last_effective_edge_capacity, 1.0e-8)
        inflow = np.clip(edge_load / capacity[..., None], 0.0, 1.25)
        dt = 1.0 / PDE_SUBSTEPS
        for _ in range(PDE_SUBSTEPS):
            mean_control = np.mean(self.edge_control, axis=-1)
            speed = np.clip(0.28 + 0.18 * mean_control, 0.24, 0.48)
            diffusion = np.clip(
                0.004 + 0.004 * interdiction[..., None], 0.004, 0.010
            )
            loss = np.clip(
                0.012 + 0.035 * damage[..., None]
                + 0.022 * interdiction[..., None],
                0.01,
                0.09,
            )
            cfl = (
                speed * dt / self.edge_dx
                + 2.0 * diffusion * dt / (self.edge_dx ** 2)
            )
            self.maximum_cfl = max(self.maximum_cfl, float(np.max(cfl)))
            self.edge_occupancy, outflow, residual, negative = self._finite_volume_step(
                self.edge_occupancy, inflow, speed, diffusion, loss, dt
            )
            self.last_edge_outflow = outflow
            self.maximum_metric_balance_residual = max(
                self.maximum_metric_balance_residual, residual
            )
            self.maximum_negative_density = max(
                self.maximum_negative_density, negative
            )

            signal_target = np.clip(
                command[..., None, None] * pnt[..., None, None]
                * (0.72 + 0.28 * self.edge_integrity[..., None]),
                0.0,
                1.0,
            )
            padded_signal = np.pad(
                self.edge_signal, ((0, 0), (0, 0), (0, 0), (1, 1)), mode="edge"
            )
            signal_laplacian = (
                padded_signal[..., 2:] - 2.0 * self.edge_signal
                + padded_signal[..., :-2]
            ) / (self.edge_dx ** 2)
            raw_signal_velocity = self.edge_signal_velocity + dt * (
                0.016 * signal_laplacian
                - 0.34 * self.edge_signal_velocity
                - 0.26 * (self.edge_signal - signal_target)
            )
            self.edge_signal_velocity = np.clip(
                raw_signal_velocity,
                -self.edge_signal / dt,
                (1.0 - self.edge_signal) / dt,
            )
            self.maximum_signal_limiter_correction = max(
                self.maximum_signal_limiter_correction,
                float(np.max(np.abs(self.edge_signal_velocity - raw_signal_velocity))),
            )
            candidate_signal = self.edge_signal + dt * self.edge_signal_velocity
            violation = np.maximum(
                np.maximum(-candidate_signal, 0.0),
                np.maximum(candidate_signal - 1.0, 0.0),
            )
            self.maximum_signal_bound_violation = max(
                self.maximum_signal_bound_violation, float(np.max(violation))
            )
            self.edge_signal = np.clip(candidate_signal, 0.0, 1.0)

            control_target = np.clip(
                command[..., None, None]
                * (1.0 - 0.62 * damage[..., None, None])
                * (1.0 - 0.44 * interdiction[..., None, None])
                * (0.65 + 0.35 * self.edge_signal),
                0.0,
                1.0,
            )
            padded_control = np.pad(
                self.edge_control, ((0, 0), (0, 0), (0, 0), (1, 1)), mode="edge"
            )
            control_laplacian = (
                padded_control[..., 2:] - 2.0 * self.edge_control
                + padded_control[..., :-2]
            ) / (self.edge_dx ** 2)
            raw_control_rate = (
                0.006 * control_laplacian
                + 0.24 * (control_target - self.edge_control)
            )
            control_rate = np.clip(
                raw_control_rate,
                -self.edge_control / dt,
                (1.0 - self.edge_control) / dt,
            )
            self.maximum_control_limiter_correction = max(
                self.maximum_control_limiter_correction,
                float(np.max(np.abs(control_rate - raw_control_rate))),
            )
            candidate_control = self.edge_control + dt * control_rate
            control_violation = np.maximum(
                np.maximum(-candidate_control, 0.0),
                np.maximum(candidate_control - 1.0, 0.0),
            )
            self.maximum_control_bound_violation = max(
                self.maximum_control_bound_violation,
                float(np.max(control_violation)),
            )
            self.edge_control = np.clip(candidate_control, 0.0, 1.0)

        density = np.mean(np.sum(self.edge_occupancy, axis=-1), axis=-1)
        signal = np.mean(self.edge_signal, axis=-1)
        control = np.mean(self.edge_control, axis=-1)
        congestion = 1.0 / (1.0 + 0.38 * density)
        self.last_continuous_capacity_gate = np.clip(
            congestion * (0.72 + 0.14 * signal + 0.14 * control), 0.24, 1.0
        )
        self.last_continuous_delay_factor = np.clip(
            1.0 + 0.42 * density + 0.48 * (1.0 - signal), 1.0, 3.0
        )
        self.minimum_continuous_capacity_gate = min(
            self.minimum_continuous_capacity_gate,
            float(np.min(self.last_continuous_capacity_gate)),
        )

    def _update_enterprise_graphon(
        self,
        production: np.ndarray,
        financial: np.ndarray,
        damage: np.ndarray,
        interdiction: np.ndarray,
    ) -> None:
        graphon_field = np.einsum("paij,paj->pai", self.firm_kernel, self.firm_state)
        target = np.clip(
            0.12 + 0.34 * financial[..., None]
            + 0.20 * np.clip(production[..., None] / 0.18, 0.0, 1.0)
            + 0.28 * graphon_field
            - 0.28 * damage[..., None]
            - 0.12 * interdiction[..., None],
            0.02,
            1.0,
        )
        graphon_next = 0.82 * self.firm_state + 0.18 * target

        finite_field = np.einsum(
            "ij,paj->pai", self.finite_kernel, self.finite_firm_state
        )
        finite_target = np.clip(
            0.12 + 0.34 * financial[..., None]
            + 0.20 * np.clip(production[..., None] / 0.18, 0.0, 1.0)
            + 0.28 * finite_field
            - 0.28 * damage[..., None]
            - 0.12 * interdiction[..., None],
            0.02,
            1.0,
        )
        finite_next = 0.82 * self.finite_firm_state + 0.18 * finite_target
        finite_by_type = np.stack(
            [
                np.mean(finite_next[..., self.finite_type == kind], axis=-1)
                for kind in range(FIRM_TYPES)
            ],
            axis=-1,
        )
        rollback_error = float(np.max(np.abs(finite_by_type - graphon_next)))
        self.maximum_finite_graphon_rollback_error = max(
            self.maximum_finite_graphon_rollback_error, rollback_error
        )
        self.firm_state = graphon_next
        self.finite_firm_state = finite_next
        self.last_enterprise_gate = np.clip(
            0.54 + 0.46 * np.mean(self.firm_state, axis=-1), 0.54, 1.0
        )
        self.minimum_enterprise_gate = min(
            self.minimum_enterprise_gate, float(np.min(self.last_enterprise_gate))
        )

    def update(self, **kwargs):
        original = {name: np.asarray(value) for name, value in kwargs.items()}
        communication = np.mean(self.edge_signal, axis=(2, 3))
        spatial_control = np.mean(self.edge_control, axis=(2, 3))
        transport = np.mean(self.last_continuous_capacity_gate, axis=2)
        args = dict(original)
        args["production"] = original["production"] * self.last_enterprise_gate * (
            0.82 + 0.18 * transport
        )
        args["command"] = original["command"] * (
            0.72 + 0.14 * communication + 0.14 * spatial_control
        )
        args["pnt"] = original["pnt"] * (0.76 + 0.24 * communication)
        args["financial_capacity"] = original["financial_capacity"] * (
            0.86 + 0.14 * self.last_enterprise_gate
        )
        result = super().update(**args)
        previous_gate = self.last_continuous_capacity_gate.copy()
        self._update_metric_graph(
            args["command"], args["pnt"], original["damage"], original["interdiction"]
        )
        self._update_enterprise_graphon(
            original["production"], original["financial_capacity"],
            original["damage"], original["interdiction"]
        )
        self.maximum_continuous_state_delta = max(
            self.maximum_continuous_state_delta,
            float(np.max(np.abs(self.last_continuous_capacity_gate - previous_gate))),
        )
        self.months_v73 += 1
        return result

    def diagnostics(self) -> dict:
        base = super().diagnostics()
        raw_radius = float(np.max(np.abs(np.linalg.eigvals(self.firm_kernel_raw))))
        sparse_density = float(np.mean(self.finite_adjacency))
        base["continuous_graph_network_v73"] = {
            "months": self.months_v73,
            "edge_cells": EDGE_CELLS,
            "pde_substeps": PDE_SUBSTEPS,
            "metric_graph_finite_volume_active": True,
            "telegraph_communication_active": True,
            "continuous_spatial_control_active": True,
            "enterprise_financial_graphon_active": True,
            "sparse_finite_network_rollback_active": True,
            "major_minor_social_graphon_inherited": True,
            "hard_capacity_solver_inherited": True,
            "maximum_metric_balance_residual": self.maximum_metric_balance_residual,
            "maximum_negative_density": self.maximum_negative_density,
            "maximum_cfl": self.maximum_cfl,
            "maximum_signal_bound_violation": self.maximum_signal_bound_violation,
            "maximum_control_bound_violation": self.maximum_control_bound_violation,
            "maximum_signal_limiter_correction": self.maximum_signal_limiter_correction,
            "maximum_control_limiter_correction": self.maximum_control_limiter_correction,
            "maximum_finite_graphon_rollback_error": self.maximum_finite_graphon_rollback_error,
            "maximum_continuous_state_delta": self.maximum_continuous_state_delta,
            "minimum_continuous_capacity_gate": self.minimum_continuous_capacity_gate,
            "minimum_enterprise_gate": self.minimum_enterprise_gate,
            "mean_edge_occupancy": float(np.mean(self.edge_occupancy)),
            "mean_signal_integrity": float(np.mean(self.edge_signal)),
            "mean_continuous_control": float(np.mean(self.edge_control)),
            "mean_enterprise_availability": float(np.mean(self.firm_state)),
            "firm_graphon_operator_radius": raw_radius,
            "finite_network_density": sparse_density,
            "explicit_one_period_feedback": True,
            "continuous_state_owns_physical_inventory": False,
            "terminal_probability_adjustment": False,
        }
        return base
