from __future__ import annotations

from dataclasses import dataclass

import numpy as np


ACTORS = ("china", "taiwan", "united_states", "japan")
CARRIERS = ("electricity", "liquid_fuel", "gas")

# Dimensionless capacity anchors. Domestic and import shares are separated so
# blockade and trade shocks cannot incorrectly destroy domestic production.
PUBLIC_CALIBRATION = {
    "china": {"domestic": [0.94, 0.46, 0.63], "imports": [0.06, 0.54, 0.37]},
    "taiwan": {"domestic": [0.22, 0.05, 0.04], "imports": [0.78, 0.95, 0.96]},
    "united_states": {"domestic": [1.08, 1.18, 1.10], "imports": [0.05, 0.08, 0.06]},
    "japan": {"domestic": [0.19, 0.06, 0.08], "imports": [0.81, 0.94, 0.92]},
}

SOURCE_URLS = {
    "china": "https://www.eia.gov/international/analysis/country/CHN",
    "japan": "https://www.eia.gov/international/analysis/country/JPN",
    "united_states": "https://www.eia.gov/international/analysis/country/USA",
    "taiwan": "https://www.moeaea.gov.tw/ECW/english/content/Content.aspx?menu_id=1540",
}


@dataclass
class EnergyFactors:
    availability: np.ndarray
    production: np.ndarray
    logistics: np.ndarray
    civilian: np.ndarray
    inflation_impulse: np.ndarray
    shortage: np.ndarray
    min_cut_capacity: np.ndarray
    bottleneck_mask: np.ndarray
    graph_resilience: np.ndarray
    hodge_cycle_share: np.ndarray


class FourPartyEnergyNetwork:
    """Three-carrier four-party energy flow, inventory, damage and repair model.

    For each carrier and path, all 2^4 cuts of the four actor transshipment
    nodes are enumerated. The minimum cut is therefore the exact maximum-flow
    value for the current source, cross-border and demand capacities.
    """

    def __init__(self, paths: int, seed: int, geometry_enabled: bool = False):
        self.paths = paths
        self.geometry_enabled = bool(geometry_enabled)
        self.rng = np.random.default_rng(seed)
        domestic = np.array([PUBLIC_CALIBRATION[a]["domestic"] for a in ACTORS])
        imports = np.array([PUBLIC_CALIBRATION[a]["imports"] for a in ACTORS])
        heterogeneity = self.rng.lognormal(0.0, 0.035, (paths, 4, 3))
        self.domestic_capacity = domestic[None, :, :] * heterogeneity
        self.import_capacity = imports[None, :, :] * heterogeneity
        self.infrastructure = np.clip(
            self.rng.normal(0.91, 0.025, (paths, 4, 3)), 0.78, 0.98
        )
        self.inventory = np.clip(
            self.rng.normal(
                np.array([[[0.10, 0.42, 0.20], [0.08, 0.36, 0.18],
                           [0.12, 0.48, 0.26], [0.09, 0.40, 0.21]]]),
                0.025, (paths, 4, 3),
            ), 0.03, 0.62,
        )
        # Directed cross-party transfer capacity; electricity, fuel, gas.
        self.cross_capacity = np.zeros((4, 4, 3))
        self.cross_capacity[2, 1] = [0.05, 0.20, 0.10]
        self.cross_capacity[2, 3] = [0.04, 0.16, 0.14]
        self.cross_capacity[3, 1] = [0.07, 0.14, 0.08]
        self.cross_capacity[1, 3] = [0.02, 0.03, 0.02]
        self.cross_capacity[0, 1] = [0.03, 0.04, 0.03]
        self.cross_capacity[0, 3] = [0.03, 0.05, 0.04]
        self.demand_anchor = np.ones((1, 4, 3))
        self.cut_masks = ((np.arange(16)[:, None] >> np.arange(4)) & 1).astype(float)
        self.last_factors = EnergyFactors(
            availability=np.ones((paths, 4, 3)),
            production=np.ones((paths, 4)),
            logistics=np.ones((paths, 4)),
            civilian=np.ones((paths, 4)),
            inflation_impulse=np.zeros((paths, 4)),
            shortage=np.zeros((paths, 4)),
            min_cut_capacity=np.full((paths, 3), 4.0),
            bottleneck_mask=np.zeros((paths, 3), dtype=np.int8),
            graph_resilience=np.ones((paths, 3)),
            hodge_cycle_share=np.zeros((paths, 3)),
        )
        self.months = 0
        self.max_flow_cut_residual = 0.0
        self.max_delivery_residual = 0.0
        self.min_infrastructure = 1.0
        self.max_shortage = 0.0
        self.cut_frequency = np.zeros((3, 16), dtype=np.int64)
        self.hodge_edges = tuple((i, j) for i in range(4) for j in range(i + 1, 4))
        incidence = np.zeros((4, len(self.hodge_edges)))
        for edge, (source, target) in enumerate(self.hodge_edges):
            incidence[source, edge] = -1.0
            incidence[target, edge] = 1.0
        self.hodge_incidence = incidence
        self.hodge_laplacian_pinv = np.linalg.pinv(incidence @ incidence.T)
        self.max_hodge_divergence_residual = 0.0
        self.max_dirac_interconnection_residual = 0.0
        self.max_port_hamiltonian_residual = 0.0

    def current(self) -> EnergyFactors:
        return self.last_factors

    def _minimum_cut(self, source: np.ndarray, demand: np.ndarray,
                     cross: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # source,demand: paths x actors; cross: paths x actor x actor
        masks = self.cut_masks
        in_s = masks[None, :, :]
        source_cut = np.sum(source[:, None, :] * (1.0 - in_s), axis=2)
        demand_cut = np.sum(demand[:, None, :] * in_s, axis=2)
        cross_cut = np.einsum(
            "pij,ki,kj->pk", cross, masks, 1.0 - masks, optimize=True
        )
        cut_capacity = source_cut + demand_cut + cross_cut
        bottleneck = np.argmin(cut_capacity, axis=1)
        minimum = cut_capacity[np.arange(self.paths), bottleneck]
        return minimum, bottleneck

    def _hodge_metrics(self, cross: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Project directed edge flows into gradient and cycle components."""
        cycle_share = np.zeros((self.paths, 3))
        divergence_residual = 0.0
        dirac_residual = 0.0
        for carrier in range(3):
            edge_flow = np.column_stack([
                cross[:, i, j, carrier] - cross[:, j, i, carrier]
                for i, j in self.hodge_edges
            ])
            divergence = edge_flow @ self.hodge_incidence.T
            potential = divergence @ self.hodge_laplacian_pinv.T
            gradient = potential @ self.hodge_incidence
            cycle = edge_flow - gradient
            cycle_share[:, carrier] = np.linalg.norm(cycle, axis=1) / np.maximum(
                np.linalg.norm(edge_flow, axis=1), 1.0e-12
            )
            divergence_residual = max(
                divergence_residual,
                float(np.max(np.abs(cycle @ self.hodge_incidence.T))),
            )
            antisymmetric = np.zeros((self.paths, 4, 4))
            for edge, (i, j) in enumerate(self.hodge_edges):
                antisymmetric[:, i, j] = edge_flow[:, edge]
                antisymmetric[:, j, i] = -edge_flow[:, edge]
            dirac_residual = max(
                dirac_residual,
                float(np.max(np.abs(antisymmetric + np.swapaxes(antisymmetric, 1, 2)))),
            )
        self.max_hodge_divergence_residual = max(
            self.max_hodge_divergence_residual, divergence_residual
        )
        self.max_dirac_interconnection_residual = max(
            self.max_dirac_interconnection_residual, dirac_residual
        )
        resilience = np.clip(0.82 + 0.18 * cycle_share, 0.82, 1.0)
        return resilience, cycle_share

    @staticmethod
    def _allocate_flow(total: np.ndarray, demand: np.ndarray,
                       access: np.ndarray) -> np.ndarray:
        delivered = np.zeros_like(demand)
        remaining = total.copy()
        for _ in range(4):
            unmet = np.maximum(demand - delivered, 0.0)
            active = unmet > 1e-12
            weights = np.where(active, unmet * np.maximum(access, 0.05), 0.0)
            weight_sum = weights.sum(axis=1)
            proposed = np.where(
                weight_sum[:, None] > 0.0,
                remaining[:, None] * weights / np.maximum(weight_sum[:, None], 1e-12),
                0.0,
            )
            addition = np.minimum(proposed, unmet)
            delivered += addition
            remaining -= addition.sum(axis=1)
            remaining = np.maximum(remaining, 0.0)
        return delivered

    def update(
        self, *, month: int, actor_damage: np.ndarray, actor_output: np.ndarray,
        actor_mobilization: np.ndarray, financial_capacity: np.ndarray,
        trade_loss: np.ndarray, blockade: np.ndarray, sanction_relief: np.ndarray,
        us_level: np.ndarray, japan_level: np.ndarray,
    ) -> EnergyFactors:
        opening_inventory = self.inventory.copy()
        damage = np.clip(actor_damage, 0.0, 1.0)
        output = np.clip(actor_output, 0.01, 1.20)
        mobilization = np.clip(actor_mobilization, 0.0, 1.0)
        fiscal = np.clip(financial_capacity, 0.02, 1.0)
        relief = np.clip(sanction_relief, 0.0, 0.25)

        repair = (
            0.010 + 0.020 * fiscal[:, :, None] * output[:, :, None]
        ) * (1.0 - self.infrastructure)
        carrier_damage = damage[:, :, None] * np.array([0.070, 0.050, 0.058])[None, None, :]
        self.infrastructure = np.clip(
            self.infrastructure + repair - carrier_damage, 0.08, 0.995
        )

        trade_access = np.clip(
            1.0 - trade_loss[:, None] - 0.55 * blockade[:, None] + relief[:, None],
            0.02, 1.0,
        )
        actor_exposure = np.array([[0.78, 1.00, 0.30, 0.84]])
        import_access = np.clip(
            1.0 - actor_exposure * (1.0 - trade_access), 0.02, 1.0
        )
        source = (
            self.domestic_capacity
            + self.import_capacity * import_access[:, :, None]
            + 0.12 * self.inventory
        ) * self.infrastructure

        demand = self.demand_anchor * (
            0.78 + 0.20 * output[:, :, None]
            + 0.16 * mobilization[:, :, None] * np.array([0.45, 1.00, 0.55])[None, None, :]
        )
        alliance = np.column_stack((
            np.ones(self.paths),
            np.clip(0.35 + 0.35 * us_level + 0.30 * japan_level, 0.20, 1.0),
            np.clip(us_level, 0.05, 1.0),
            np.clip(japan_level, 0.05, 1.0),
        ))
        cross = np.broadcast_to(
            self.cross_capacity[None, :, :, :], (self.paths, 4, 4, 3)
        ).copy()
        cross *= np.sqrt(alliance[:, :, None, None] * alliance[:, None, :, None])
        cross *= np.clip(1.0 - 0.50 * blockade[:, None, None, None], 0.15, 1.0)
        cross *= np.sqrt(
            self.infrastructure[:, :, None, :] * self.infrastructure[:, None, :, :]
        )
        if self.geometry_enabled:
            graph_resilience, hodge_cycle_share = self._hodge_metrics(cross)
            cross *= graph_resilience[:, None, None, :]
        else:
            graph_resilience = np.ones((self.paths, 3))
            hodge_cycle_share = np.zeros((self.paths, 3))

        delivered = np.zeros((self.paths, 4, 3))
        min_cut = np.zeros((self.paths, 3))
        bottleneck = np.zeros((self.paths, 3), dtype=np.int8)
        for carrier in range(3):
            total, cut = self._minimum_cut(
                source[:, :, carrier], demand[:, :, carrier], cross[:, :, :, carrier]
            )
            access = source[:, :, carrier] + cross[:, :, :, carrier].sum(axis=1)
            delivered[:, :, carrier] = self._allocate_flow(
                total, demand[:, :, carrier], access
            )
            min_cut[:, carrier] = total
            bottleneck[:, carrier] = cut
            self.max_flow_cut_residual = max(
                self.max_flow_cut_residual,
                float(np.max(np.abs(delivered[:, :, carrier].sum(axis=1) - total))),
            )
            counts = np.bincount(cut, minlength=16)
            self.cut_frequency[carrier] += counts

        availability = np.clip(delivered / np.maximum(demand, 1e-12), 0.0, 1.0)
        unused = np.maximum(source - delivered, 0.0)
        inventory_target = np.minimum(0.62, 0.55 * self.inventory + 0.10 * unused)
        self.inventory = np.clip(
            0.97 * self.inventory + 0.03 * inventory_target
            - 0.025 * np.maximum(1.0 - availability, 0.0), 0.01, 0.62
        )
        if self.geometry_enabled:
            inventory_delta = self.inventory - opening_inventory
            hamiltonian_delta = 0.5 * np.sum(
                self.inventory ** 2 - opening_inventory ** 2, axis=(1, 2)
            )
            midpoint_power = np.sum(
                0.5 * (self.inventory + opening_inventory) * inventory_delta,
                axis=(1, 2),
            )
            self.max_port_hamiltonian_residual = max(
                self.max_port_hamiltonian_residual,
                float(np.max(np.abs(hamiltonian_delta - midpoint_power))),
            )
        production = np.clip(
            0.46 * availability[:, :, 0]
            + 0.34 * availability[:, :, 1]
            + 0.20 * availability[:, :, 2], 0.12, 1.0
        )
        logistics = np.clip(
            0.18 * availability[:, :, 0] + 0.72 * availability[:, :, 1]
            + 0.10 * availability[:, :, 2], 0.10, 1.0
        )
        civilian = np.clip(
            0.60 * availability[:, :, 0] + 0.12 * availability[:, :, 1]
            + 0.28 * availability[:, :, 2], 0.10, 1.0
        )
        shortage = np.clip(1.0 - availability.mean(axis=2), 0.0, 1.0)
        inflation_impulse = np.clip(
            0.018 * shortage + 0.010 * (1.0 - availability[:, :, 1]), 0.0, 0.035
        )
        delivery_residual = np.max(
            np.maximum(delivered - demand, 0.0)
        )
        self.max_delivery_residual = max(self.max_delivery_residual, float(delivery_residual))
        self.min_infrastructure = min(self.min_infrastructure, float(self.infrastructure.min()))
        self.max_shortage = max(self.max_shortage, float(shortage.max()))
        self.months += 1
        self.last_factors = EnergyFactors(
            availability=availability,
            production=production,
            logistics=logistics,
            civilian=civilian,
            inflation_impulse=inflation_impulse,
            shortage=shortage,
            min_cut_capacity=min_cut,
            bottleneck_mask=bottleneck,
            graph_resilience=graph_resilience,
            hodge_cycle_share=hodge_cycle_share,
        )
        return self.last_factors

    def diagnostics(self) -> dict:
        return {
            "months": self.months,
            "max_flow_min_cut_residual": self.max_flow_cut_residual,
            "max_delivery_above_demand_residual": self.max_delivery_residual,
            "minimum_infrastructure": self.min_infrastructure,
            "maximum_actor_shortage": self.max_shortage,
            "bottleneck_cut_frequency": self.cut_frequency.tolist(),
            "geometry_enabled": self.geometry_enabled,
            "max_hodge_divergence_residual": self.max_hodge_divergence_residual,
            "max_dirac_interconnection_residual": self.max_dirac_interconnection_residual,
            "max_port_hamiltonian_residual": self.max_port_hamiltonian_residual,
            "mean_graph_resilience": float(np.mean(self.last_factors.graph_resilience)),
            "mean_hodge_cycle_share": float(np.mean(self.last_factors.hodge_cycle_share)),
            "source_urls": SOURCE_URLS,
            "calibration_classification": (
                "public import/domestic structure anchors plus explicit scenario priors "
                "for wartime transfer capacity, repair and demand priority"
            ),
        }
