"""DIAGNOSTIC_ONLY: V4.9 energy-network semantics and full-path coupling."""

from __future__ import annotations

import numpy as np

import diagnostic_taiwan_v48
from four_party_energy_network_v49 import FourPartyEnergyNetwork


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def run(paths: int, blockade: float, trade_loss: float):
    system = FourPartyEnergyNetwork(paths, 901)
    zero = np.zeros((paths, 4))
    one = np.ones((paths, 4))
    for month in range(18):
        factors = system.update(
            month=month,
            actor_damage=zero,
            actor_output=one,
            actor_mobilization=zero,
            financial_capacity=one,
            trade_loss=np.full(paths, trade_loss),
            blockade=np.full(paths, blockade),
            sanction_relief=np.zeros(paths),
            us_level=np.full(paths, 0.65),
            japan_level=np.full(paths, 0.55),
        )
    return system, factors


def main() -> None:
    diagnostic_taiwan_v48.main()
    low_system, low = run(24, 0.05, 0.05)
    high_system, high = run(24, 0.80, 0.65)
    require(
        np.mean(high.availability[:, 1, :]) < np.mean(low.availability[:, 1, :]),
        "Taiwan energy availability must respond to blockade and trade loss",
    )
    require(
        np.mean(high.inflation_impulse[:, 1]) > np.mean(low.inflation_impulse[:, 1]),
        "energy shortage must enter inflation",
    )
    for system in (low_system, high_system):
        diagnostic = system.diagnostics()
        require(diagnostic["max_flow_min_cut_residual"] < 1e-9, "max-flow/min-cut")
        require(diagnostic["max_delivery_above_demand_residual"] < 1e-10, "delivery feasibility")
        require(sum(map(sum, diagnostic["bottleneck_cut_frequency"])) > 0, "cut activation")
    print("DIAGNOSTIC_ONLY_TAIWAN_V49: PASS")


if __name__ == "__main__":
    main()
