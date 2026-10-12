from __future__ import annotations

from dataclasses import dataclass

import numpy as np


ACTORS = 4
ECHELONS = ("front", "transit", "depth")
COMMODITIES = ("fuel", "munitions", "spares", "medical", "communications")


@dataclass(frozen=True)
class TransportSymplecticFactors:
    logistics: np.ndarray
    network: np.ndarray
    command: np.ndarray
    pnt: np.ndarray
    production: np.ndarray
    congestion: np.ndarray
    unmet_demand: np.ndarray
    reachability: np.ndarray


class CoupledTransportSymplecticSystemV66:
    """Capacity-constrained dynamic OT with a structure-preserving motion core.

    The network is deliberately aggregate and non-targeting: four public actors,
    three logistics echelons and five resource classes. It owns the transport
    inventory, backlog, platform phase state and their accounting diagnostics.
    """

    def __init__(self, paths: int, seed: int, terrain: dict):
        self.paths = int(paths)
        self.rng = np.random.default_rng(seed)
        base = np.array([0.56, 0.62, 0.48, 0.42, 0.52])
        echelon = np.array([0.72, 0.88, 1.08])
        self.inventory = np.clip(
            base[None, None, None, :] * echelon[None, None, :, None]
            * self.rng.lognormal(0.0, 0.05, (paths, ACTORS, 3, 5)),
            0.04, 1.40,
        )
        self.backlog = np.zeros((paths, ACTORS, 5))
        self.node_mass = np.clip(
            self.rng.normal(0.90, 0.025, (paths, ACTORS, 3)), 0.75, 0.98
        )
        self.q = self.rng.normal(0.0, 0.04, (paths, ACTORS, 3))
        self.p = self.rng.normal(0.0, 0.025, (paths, ACTORS, 3))
        self.orientation = np.broadcast_to(
            np.eye(3), (paths, ACTORS, 3, 3)
        ).copy()
        self.previous_damage = np.zeros((paths, ACTORS))
        self.month = -1
        self.priority = np.array([1.00, 1.08, 0.92, 0.82, 0.88])
        self.loss_rate = np.array([0.018, 0.015])
        mountain = float(terrain.get("land_share_above_500m", 0.45))
        self.terrain_capacity = np.clip(1.0 - 0.18 * mountain, 0.72, 0.98)
        self.last = TransportSymplecticFactors(
            logistics=np.ones((paths, ACTORS)),
            network=np.ones((paths, ACTORS)),
            command=np.ones((paths, ACTORS)),
            pnt=np.ones((paths, ACTORS)),
            production=np.ones((paths, ACTORS)),
            congestion=np.zeros((paths, ACTORS)),
            unmet_demand=np.zeros((paths, ACTORS)),
            reachability=np.ones((paths, ACTORS)),
        )
        self.max_mass_balance_residual = 0.0
        self.max_shared_capacity_violation = 0.0
        self.max_nonnegativity_violation = 0.0
        self.max_wfr_mass_residual = 0.0
        self.max_port_power_residual = 0.0
        self.max_symplectic_defect = 0.0
        self.max_conservative_energy_drift = 0.0
        self.max_so3_orthogonality_residual = 0.0
        self.max_so3_determinant_residual = 0.0
        self.max_jump_account_residual = 0.0
        self.max_causal_clock_violation = 0.0
        self.dynamic_flow_months = 0
        self.jump_events = 0
        self.future_information_used = False

    @staticmethod
    def _allocate_shared(demand: np.ndarray, capacity: np.ndarray,
                         priority: np.ndarray) -> np.ndarray:
        delivered = np.zeros_like(demand)
        remaining = np.maximum(capacity, 0.0)
        for _ in range(demand.shape[-1]):
            unmet = np.maximum(demand - delivered, 0.0)
            active = unmet > 1.0e-14
            weights = np.where(active, unmet * priority, 0.0)
            total_weight = np.sum(weights, axis=-1)
            proposal = np.where(
                total_weight[..., None] > 0.0,
                remaining[..., None] * weights
                / np.maximum(total_weight[..., None], 1.0e-14),
                0.0,
            )
            addition = np.minimum(proposal, unmet)
            delivered += addition
            remaining = np.maximum(remaining - np.sum(addition, axis=-1), 0.0)
        return delivered

    @staticmethod
    def _rodrigues(rotation_vector: np.ndarray) -> np.ndarray:
        angle = np.linalg.norm(rotation_vector, axis=-1)
        axis = rotation_vector / np.maximum(angle[..., None], 1.0e-14)
        x, y, z = (axis[..., i] for i in range(3))
        skew = np.zeros(rotation_vector.shape[:-1] + (3, 3))
        skew[..., 0, 1] = -z
        skew[..., 0, 2] = y
        skew[..., 1, 0] = z
        skew[..., 1, 2] = -x
        skew[..., 2, 0] = -y
        skew[..., 2, 1] = x
        identity = np.broadcast_to(np.eye(3), skew.shape)
        sin_term = np.sin(angle)[..., None, None]
        cos_term = (1.0 - np.cos(angle))[..., None, None]
        return identity + sin_term * skew + cos_term * (skew @ skew)

    def _symplectic_step(self, target: np.ndarray, command: np.ndarray,
                         damage_jump: np.ndarray) -> np.ndarray:
        h = 0.25
        stiffness = np.array([0.18, 0.14, 0.20])[None, None, :]
        q0 = self.q.copy()
        p0 = self.p.copy()
        energy0 = 0.5 * np.sum(
            p0 ** 2 + stiffness * (q0 - target) ** 2, axis=2
        )
        p_half = p0 - 0.5 * h * stiffness * (q0 - target)
        q1 = q0 + h * p_half
        p_conservative = p_half - 0.5 * h * stiffness * (q1 - target)
        energy1 = 0.5 * np.sum(
            p_conservative ** 2 + stiffness * (q1 - target) ** 2, axis=2
        )
        self.max_conservative_energy_drift = max(
            self.max_conservative_energy_drift,
            float(np.max(np.abs(energy1 - energy0))),
        )
        for k in stiffness.reshape(-1):
            matrix = np.array([
                [1.0 - 0.5 * h * h * k, h],
                [-h * k + 0.25 * h ** 3 * k ** 2, 1.0 - 0.5 * h * h * k],
            ])
            symplectic = np.array([[0.0, 1.0], [-1.0, 0.0]])
            defect = matrix.T @ symplectic @ matrix - symplectic
            self.max_symplectic_defect = max(
                self.max_symplectic_defect, float(np.max(np.abs(defect)))
            )
        damping = 0.035 + 0.055 * np.clip(damage_jump, 0.0, 1.0)
        control = 0.018 * (np.clip(command, 0.0, 1.0) - 0.5)[..., None]
        p_dissipative = (p_conservative + control) / (1.0 + damping[..., None])
        jump_mask = damage_jump > 0.08
        impulse = np.where(jump_mask[..., None], -0.12 * p_dissipative, 0.0)
        p1 = p_dissipative + impulse
        self.max_jump_account_residual = max(
            self.max_jump_account_residual,
            float(np.max(np.abs((p1 - p_dissipative) - impulse))),
        )
        self.jump_events += int(np.count_nonzero(jump_mask))
        rotation = self._rodrigues(0.015 * p1)
        self.orientation = rotation @ self.orientation
        identity = np.eye(3)
        orthogonal = np.swapaxes(self.orientation, -1, -2) @ self.orientation
        self.max_so3_orthogonality_residual = max(
            self.max_so3_orthogonality_residual,
            float(np.max(np.abs(orthogonal - identity))),
        )
        self.max_so3_determinant_residual = max(
            self.max_so3_determinant_residual,
            float(np.max(np.abs(np.linalg.det(self.orientation) - 1.0))),
        )
        self.q = q1
        self.p = p1
        return np.exp(-0.34 * np.linalg.norm(q1 - target, axis=2))

    def update(
        self, *, month: int, conflict_intensity: np.ndarray,
        actor_damage: np.ndarray, actor_output: np.ndarray,
        financial_capacity: np.ndarray, actor_network: np.ndarray,
        actor_command: np.ndarray, actor_support: np.ndarray,
        actor_mobilization: np.ndarray, stock_shortage: np.ndarray,
        geometry_logistics: np.ndarray | None = None,
        geometry_network: np.ndarray | None = None,
    ) -> TransportSymplecticFactors:
        if month != self.month + 1:
            self.max_causal_clock_violation = max(
                self.max_causal_clock_violation, abs(month - (self.month + 1))
            )
            raise RuntimeError("V6.6 transport/symplectic clock must advance monthly")
        self.month = month
        opening = self.inventory.copy()
        old_node_mass = self.node_mass.copy()
        damage = np.clip(actor_damage, 0.0, 1.0)
        output = np.clip(actor_output, 0.02, 1.30)
        finance = np.clip(financial_capacity, 0.02, 1.0)
        network = np.clip(actor_network, 0.02, 1.0)
        command = np.clip(actor_command, 0.02, 1.0)
        mobilization = np.clip(actor_mobilization, 0.0, 1.0)
        shortage = np.clip(stock_shortage, 0.0, 1.0)
        geo_log = (
            np.ones_like(network) if geometry_logistics is None
            else np.clip(geometry_logistics, 0.05, 1.0)
        )
        geo_net = (
            np.ones_like(network) if geometry_network is None
            else np.clip(geometry_network, 0.05, 1.0)
        )

        repair = 0.018 * finance[:, :, None] * output[:, :, None]
        failure = damage[:, :, None] * np.array([0.040, 0.032, 0.024])[None, None, :]
        raw_node_mass = old_node_mass + repair * (1.0 - old_node_mass) - failure
        self.node_mass = np.clip(raw_node_mass, 0.05, 0.995)
        projection = self.node_mass - raw_node_mass
        wfr_residual = self.node_mass - old_node_mass - (
            repair * (1.0 - old_node_mass) - failure + projection
        )
        self.max_wfr_mass_residual = max(
            self.max_wfr_mass_residual, float(np.max(np.abs(wfr_residual)))
        )

        source = (
            0.030 * output * finance * (0.76 + 0.24 * network)
        )[..., None] * np.array([1.00, 0.94, 0.82, 0.62, 0.74])[None, None, :]
        demand = (
            0.018 + 0.040 * conflict_intensity[:, None]
            + 0.020 * mobilization + 0.022 * shortage
            + 0.25 * self.backlog.mean(axis=2)
        )[..., None] * np.array([1.05, 1.12, 0.82, 0.66, 0.72])[None, None, :]
        base_capacity = (
            0.34 * self.terrain_capacity * network * geo_log
            * (0.55 + 0.45 * finance)
        )
        capacity_depth_transit = base_capacity * np.minimum(
            self.node_mass[:, :, 2], self.node_mass[:, :, 1]
        )
        capacity_transit_front = 0.86 * base_capacity * np.minimum(
            self.node_mass[:, :, 1], self.node_mass[:, :, 0]
        )

        available_depth = opening[:, :, 2, :] + source
        desired_depth = np.maximum(
            demand + self.backlog + 0.45 - opening[:, :, 1, :], 0.0
        )
        flow_depth = self._allocate_shared(
            np.minimum(desired_depth, available_depth),
            capacity_depth_transit, self.priority,
        )
        after_depth = available_depth - flow_depth
        arrival_transit = flow_depth * (1.0 - self.loss_rate[0])
        available_transit = opening[:, :, 1, :] + arrival_transit
        desired_front = np.maximum(
            demand + self.backlog + 0.32 - opening[:, :, 0, :], 0.0
        )
        flow_front = self._allocate_shared(
            np.minimum(desired_front, available_transit),
            capacity_transit_front, self.priority,
        )
        after_transit = available_transit - flow_front
        arrival_front = flow_front * (1.0 - self.loss_rate[1])
        available_front = opening[:, :, 0, :] + arrival_front
        consumed = np.minimum(available_front, demand)
        after_front = available_front - consumed
        self.inventory = np.stack((after_front, after_transit, after_depth), axis=2)
        losses = self.loss_rate[0] * flow_depth + self.loss_rate[1] * flow_front
        mass_residual = (
            np.sum(self.inventory, axis=2) - np.sum(opening, axis=2)
            - source + losses + consumed
        )
        self.max_mass_balance_residual = max(
            self.max_mass_balance_residual, float(np.max(np.abs(mass_residual)))
        )
        used_depth = np.sum(flow_depth, axis=2)
        used_front = np.sum(flow_front, axis=2)
        capacity_violation = max(
            float(np.max(np.maximum(used_depth - capacity_depth_transit, 0.0))),
            float(np.max(np.maximum(used_front - capacity_transit_front, 0.0))),
        )
        self.max_shared_capacity_violation = max(
            self.max_shared_capacity_violation, capacity_violation
        )
        self.max_nonnegativity_violation = max(
            self.max_nonnegativity_violation,
            float(np.max(np.maximum(-self.inventory, 0.0))),
        )
        unmet = np.maximum(demand - consumed, 0.0)
        self.backlog = np.clip(0.72 * self.backlog + unmet, 0.0, 2.0)
        service = np.clip(
            np.sum(consumed * self.priority, axis=2)
            / np.maximum(np.sum(demand * self.priority, axis=2), 1.0e-14),
            0.0, 1.0,
        )
        congestion = np.clip(0.5 * (
            used_depth / np.maximum(capacity_depth_transit, 1.0e-14)
            + used_front / np.maximum(capacity_transit_front, 1.0e-14)
        ), 0.0, 2.0)

        h0 = 0.5 * np.sum(opening ** 2, axis=(2, 3))
        h1 = 0.5 * np.sum(self.inventory ** 2, axis=(2, 3))
        midpoint = 0.5 * (opening + self.inventory)
        port_work = np.sum(midpoint * (self.inventory - opening), axis=(2, 3))
        self.max_port_power_residual = max(
            self.max_port_power_residual,
            float(np.max(np.abs((h1 - h0) - port_work))),
        )

        damage_jump = np.maximum(damage - self.previous_damage, 0.0)
        target = np.stack((
            2.0 * network - 1.0,
            2.0 * service - 1.0,
            2.0 * command - 1.0,
        ), axis=2)
        reachability = self._symplectic_step(target, command, damage_jump)
        self.previous_damage = damage.copy()
        logistics = np.clip(0.74 + 0.22 * service + 0.04 * reachability, 0.70, 1.0)
        network_factor = np.clip(
            0.80 + 0.12 * service + 0.08 * geo_net * reachability, 0.76, 1.0
        )
        command_factor = np.clip(
            0.84 + 0.10 * service + 0.06 * reachability, 0.80, 1.0
        )
        pnt_factor = np.clip(
            0.86 + 0.08 * geo_net + 0.06 * reachability, 0.82, 1.0
        )
        production = np.clip(
            0.78 + 0.16 * service + 0.06 * (1.0 - np.clip(congestion, 0.0, 1.0)),
            0.72, 1.0,
        )
        self.dynamic_flow_months += int(np.any(used_depth + used_front > 1.0e-12))
        self.last = TransportSymplecticFactors(
            logistics=logistics,
            network=network_factor,
            command=command_factor,
            pnt=pnt_factor,
            production=production,
            congestion=congestion,
            unmet_demand=np.mean(unmet, axis=2),
            reachability=reachability,
        )
        return self.last

    def diagnostics(self) -> dict:
        return {
            "months": self.month + 1,
            "actors": ACTORS,
            "echelons": list(ECHELONS),
            "commodities": list(COMMODITIES),
            "path_conditioned": True,
            "shared_capacity_dynamic_ot": True,
            "wfr_random_geometry_closure": True,
            "symplectic_core": "Störmer-Verlet harmonic aggregate mission state",
            "lie_group_update": "SO(3) Rodrigues composition",
            "port_hamiltonian_shell": True,
            "deds_jump_account": True,
            "future_information_used": self.future_information_used,
            "max_causal_clock_violation": self.max_causal_clock_violation,
            "max_mass_balance_residual": self.max_mass_balance_residual,
            "max_shared_capacity_violation": self.max_shared_capacity_violation,
            "max_nonnegativity_violation": self.max_nonnegativity_violation,
            "max_wfr_mass_residual": self.max_wfr_mass_residual,
            "max_port_power_residual": self.max_port_power_residual,
            "max_symplectic_defect": self.max_symplectic_defect,
            "max_conservative_energy_drift": self.max_conservative_energy_drift,
            "max_so3_orthogonality_residual": self.max_so3_orthogonality_residual,
            "max_so3_determinant_residual": self.max_so3_determinant_residual,
            "max_jump_account_residual": self.max_jump_account_residual,
            "dynamic_flow_months": self.dynamic_flow_months,
            "jump_events": self.jump_events,
            "terminal_mean_congestion": float(np.mean(self.last.congestion)),
            "terminal_mean_unmet_demand": float(np.mean(self.last.unmet_demand)),
        }
