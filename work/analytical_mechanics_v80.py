from __future__ import annotations

from dataclasses import dataclass

import numpy as np


CHANNELS = ("logistics", "financial", "command", "social")


@dataclass
class MechanicsFactorsV80:
    logistics: np.ndarray
    financial: np.ndarray
    command: np.ndarray
    social: np.ndarray


class AnalyticalMechanicsV80:
    """Constrained adjustment layer for cross-domain residual dynamics.

    This layer owns only generalized residual adjustment coordinates and their
    momenta.  It exposes bounded factors to the state-owning modules and uses a
    discrete Lagrange-d'Alembert step, a skew Dirac interconnection and a
    symmetric dissipative operator.  It is not a replacement for accounting,
    CGE, DSGE, logistics, command or social states.
    """

    def __init__(self, paths: int, actors: int = 4) -> None:
        self.paths = paths
        self.actors = actors
        self.q = np.zeros((paths, actors, len(CHANNELS)))
        self.velocity = np.zeros_like(self.q)
        self.mass = np.array([1.30, 1.05, 0.90, 0.85])
        self.stiffness = np.array([0.34, 0.30, 0.40, 0.36])
        self.damping = np.array([0.18, 0.20, 0.24, 0.22])
        self.dirac = np.array([
            [0.00, 0.10, -0.08, -0.04],
            [-0.10, 0.00, -0.05, 0.08],
            [0.08, 0.05, 0.00, -0.10],
            [0.04, -0.08, 0.10, 0.00],
        ])
        self.dt = 0.20
        self.updates = 0
        self.maximum_constraint_residual = 0.0
        self.maximum_dirac_power_residual = 0.0
        self.maximum_balance_residual = 0.0
        self.maximum_pmp_disagreement = 0.0
        self.maximum_output = 0.0

    @staticmethod
    def _pressure(**kwargs) -> np.ndarray:
        damage = kwargs["damage"]
        shortage = kwargs["shortage"]
        inflation = kwargs["inflation"]
        financial_stress = kwargs["financial_stress"]
        command = kwargs["command"]
        network = kwargs["network"]
        fatigue = kwargs["contagion_fatigue"]
        return np.stack((
            0.62 * shortage + 0.22 * damage + 0.16 * (1.0 - network),
            0.56 * financial_stress + 0.24 * np.maximum(inflation, 0.0)
            + 0.20 * shortage,
            0.58 * (1.0 - command) + 0.28 * damage
            + 0.14 * (1.0 - network),
            0.44 * fatigue + 0.28 * damage + 0.16 * shortage
            + 0.12 * (1.0 - network),
        ), axis=-1)

    def update(self, **kwargs) -> MechanicsFactorsV80:
        target = np.clip(self._pressure(**kwargs), 0.0, 1.5)
        opening_q = self.q.copy()
        opening_v = self.velocity.copy()
        gradient = self.stiffness * (opening_q - target)
        momentum_gradient = self.mass * opening_v

        # Skew symmetry gives an exact zero-power interconnection before
        # floating-point error: grad(H)^T J grad(H) = 0.
        reversible = np.einsum("de,pae->pad", self.dirac, gradient)
        dirac_power = np.sum(gradient * reversible, axis=-1)
        self.maximum_dirac_power_residual = max(
            self.maximum_dirac_power_residual,
            float(np.max(np.abs(dirac_power))),
        )

        force = -gradient + reversible - self.damping * opening_v
        impulse = np.asarray(
            kwargs.get("jump_signal", np.zeros((self.paths, self.actors))),
            dtype=float,
        )
        force += 0.06 * impulse[..., None]

        # Project the internal exchange force onto the zero-sum tangent space.
        # External pressure remains in the potential target and is not erased.
        internal = reversible - np.mean(reversible, axis=-1, keepdims=True)
        force += internal - reversible
        constraint_residual = np.sum(internal, axis=-1)
        self.maximum_constraint_residual = max(
            self.maximum_constraint_residual,
            float(np.max(np.abs(constraint_residual))),
        )

        new_velocity = opening_v + self.dt * force / self.mass
        new_q = np.clip(opening_q + self.dt * new_velocity, -0.25, 1.5)
        self.velocity = np.clip(new_velocity, -0.50, 0.50)
        self.q = new_q

        old_energy = 0.5 * np.sum(
            self.mass * np.square(opening_v)
            + self.stiffness * np.square(opening_q - target), axis=-1
        )
        new_energy = 0.5 * np.sum(
            self.mass * np.square(self.velocity)
            + self.stiffness * np.square(self.q - target), axis=-1
        )
        external_work = self.dt * np.sum(
            (force + self.damping * opening_v) * self.velocity, axis=-1
        )
        dissipation = self.dt * np.sum(
            self.damping * np.square(self.velocity), axis=-1
        )
        balance_residual = new_energy - old_energy - external_work + dissipation
        self.maximum_balance_residual = max(
            self.maximum_balance_residual,
            float(np.max(np.abs(balance_residual))),
        )

        # PMP is used as a cross-check, not as a second controller.  The
        # costate-implied bounded action should agree in sign with the realized
        # generalized velocity when the local approximation is informative.
        costate = self.stiffness * (self.q - target) + self.mass * self.velocity
        pmp_action = np.clip(-costate / np.maximum(self.mass, 1.0e-12), -0.5, 0.5)
        disagreement = np.abs(pmp_action - self.velocity)
        self.maximum_pmp_disagreement = max(
            self.maximum_pmp_disagreement, float(np.max(disagreement))
        )

        residual = np.clip(self.q, 0.0, 1.0)
        self.maximum_output = max(self.maximum_output, float(np.max(residual)))
        self.updates += 1
        return MechanicsFactorsV80(
            logistics=residual[..., 0],
            financial=residual[..., 1],
            command=residual[..., 2],
            social=residual[..., 3],
        )

    def diagnostics(self) -> dict:
        return {
            "updates": self.updates,
            "channels": list(CHANNELS),
            "maximum_constraint_residual": self.maximum_constraint_residual,
            "maximum_dirac_power_residual": self.maximum_dirac_power_residual,
            "maximum_discrete_energy_balance_residual": self.maximum_balance_residual,
            "maximum_pmp_action_disagreement": self.maximum_pmp_disagreement,
            "maximum_residual_adjustment": self.maximum_output,
            "state_owner": "generalized residual adjustment coordinates only",
            "integration": "discrete Lagrange-d'Alembert symplectic-Euler step",
            "thermodynamic_split": (
                "skew reversible Dirac exchange plus symmetric damping"
            ),
            "noether_scope": (
                "constraint and zero-power exchange audit; primary stock-flow accounting remains with its owning modules"
            ),
            "pmp_scope": "costate cross-check only; no duplicate policy controller",
            "identification": (
                "mass, stiffness, damping and interconnection coefficients are structural scenario priors pending calibration"
            ),
            "terminal_probability_adjustment": False,
        }
