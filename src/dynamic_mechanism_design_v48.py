from __future__ import annotations

from dataclasses import dataclass

import numpy as np


ACTORS = ("china", "taiwan", "united_states", "japan")


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -35.0, 35.0)))


@dataclass
class MechanismFactors:
    alliance_multiplier: np.ndarray
    responsibility: np.ndarray
    transfer: np.ndarray
    aid_multiplier: np.ndarray
    procurement_factor: np.ndarray
    information_support_shift: np.ndarray
    panic_reduction: np.ndarray
    ceasefire_entry_multiplier: np.ndarray
    ceasefire_break_multiplier: np.ndarray
    settlement_multiplier: np.ndarray
    governance_recruitment_multiplier: np.ndarray
    governance_trust_bonus: np.ndarray
    governance_leakage_reduction: np.ndarray
    sanction_relief: np.ndarray
    insurance_transfer: np.ndarray
    verification_precision: np.ndarray
    converged: bool


class RobustDynamicMechanismDesigner:
    """Same-path robust dynamic mechanism layer for the V4.8 campaign model.

    The solver uses aggregate public-state proxies and documented scenario
    priors. It is deliberately institution-level: no targeting or deployment
    decisions are represented here.
    """

    MODULES = (
        "robust_design", "alliance_vcg", "aid_contract", "procurement",
        "information_design", "ceasefire_contract", "governance_contract",
        "trade_insurance",
    )

    def __init__(self, paths: int, seed: int, module_flags: dict | None = None):
        self.paths = paths
        self.rng = np.random.default_rng(seed)
        n = paths
        self.private_cost = self.rng.lognormal(
            np.log(np.array([[1.00, 1.12, 0.92, 0.98]])), 0.10, (n, 4)
        )
        self.private_quality = np.clip(
            self.rng.normal(np.array([[0.92, 0.84, 1.05, 1.00]]), 0.05, (n, 4)),
            0.62, 1.20,
        )
        self.private_commitment = np.clip(
            self.rng.normal(np.array([[0.72, 0.68, 0.78, 0.74]]), 0.07, (n, 4)),
            0.35, 0.96,
        )
        self.promised_utility = np.zeros((n, 4))
        self.reputation = np.full((n, 4), 0.70)
        self.audit_memory = np.zeros((n, 4))
        self.contract_effort = np.full((n, 4), 0.45)
        self.last_factors: MechanismFactors | None = None
        self.months = 0
        self.nonconverged_months = 0
        self.max_ic_residual = 0.0
        self.max_ir_residual = 0.0
        self.max_budget_residual = 0.0
        self.max_resource_residual = 0.0
        self.max_limited_liability_residual = 0.0
        self.max_commitment_residual = 0.0
        self.max_transfer_balance_residual = 0.0
        self.minimum_truthful_surplus = np.inf
        self.minimum_verification_precision = 1.0
        self.flags = {name: True for name in self.MODULES}
        if module_flags:
            unknown = set(module_flags) - set(self.flags)
            if unknown:
                raise ValueError(f"unknown mechanism flags: {sorted(unknown)}")
            self.flags.update(module_flags)

    @staticmethod
    def _efficient_allocation(target, reported_cost, quality):
        k = np.maximum(reported_cost / np.maximum(quality, 1e-6), 1e-6)
        inv = 1.0 / k
        allocation = target[:, None] * inv / np.maximum(inv.sum(axis=1, keepdims=True), 1e-12)
        return allocation, k

    def _vcg_alliance(self, target, fiscal, support):
        # Taiwan, United States, and Japan share the modeled alliance burden.
        # The robust effective cost is the current private type. Reporting that
        # type truthfully is therefore the benchmark used by the VCG audit.
        ambiguity = 1.0 + 0.12 * (1.0 - self.reputation[:, 1:])
        actual_cost = self.private_cost[:, 1:] * ambiguity
        quality = self.private_quality[:, 1:] * (0.78 + 0.22 * support[:, 1:])
        reported_cost = actual_cost.copy()
        budget = 0.055 + 0.095 * np.mean(fiscal[:, 1:], axis=1)

        allocation, k_report = self._efficient_allocation(target, reported_cost, quality)
        payments = np.zeros_like(allocation)
        for i in range(3):
            others = np.arange(3) != i
            without_cost = 0.5 * target ** 2 / np.maximum(
                np.sum(1.0 / k_report[:, others], axis=1), 1e-12
            )
            others_with = 0.5 * np.sum(
                k_report[:, others] * allocation[:, others] ** 2, axis=1
            )
            payments[:, i] = np.maximum(without_cost - others_with, 0.0)
        total_payment = payments.sum(axis=1)
        target_scale = np.minimum(1.0, np.sqrt(budget / np.maximum(total_payment, 1e-12)))
        target = target * target_scale
        allocation, k_report = self._efficient_allocation(target, reported_cost, quality)
        for i in range(3):
            others = np.arange(3) != i
            without_cost = 0.5 * target ** 2 / np.maximum(
                np.sum(1.0 / k_report[:, others], axis=1), 1e-12
            )
            others_with = 0.5 * np.sum(
                k_report[:, others] * allocation[:, others] ** 2, axis=1
            )
            payments[:, i] = np.maximum(without_cost - others_with, 0.0)

        actual_k = actual_cost / np.maximum(quality, 1e-6)
        truthful_utility = payments - 0.5 * actual_k * allocation ** 2
        max_deviation_gain = np.zeros_like(allocation)
        for shade in (0.75, 1.25):
            for i in range(3):
                deviating_report = reported_cost.copy()
                deviating_report[:, i] *= shade
                dev_alloc, dev_k = self._efficient_allocation(target, deviating_report, quality)
                others = np.arange(3) != i
                without_cost = 0.5 * target ** 2 / np.maximum(
                    np.sum(1.0 / dev_k[:, others], axis=1), 1e-12
                )
                others_with = 0.5 * np.sum(
                    dev_k[:, others] * dev_alloc[:, others] ** 2, axis=1
                )
                dev_payment = np.maximum(without_cost - others_with, 0.0)
                dev_utility = dev_payment - 0.5 * actual_k[:, i] * dev_alloc[:, i] ** 2
                max_deviation_gain[:, i] = np.maximum(
                    max_deviation_gain[:, i], dev_utility - truthful_utility[:, i]
                )
        ic_residual = float(np.max(np.maximum(max_deviation_gain, 0.0)))
        ir_residual = float(np.max(np.maximum(-truthful_utility, 0.0)))
        budget_residual = float(np.max(np.maximum(payments.sum(axis=1) - budget, 0.0)))
        resource_residual = float(np.max(np.abs(allocation.sum(axis=1) - target)))
        return allocation, payments, truthful_utility, ic_residual, ir_residual, budget_residual, resource_residual

    def update(
        self, *, month: int, actor_support: np.ndarray, actor_damage: np.ndarray,
        actor_shortage: np.ndarray, actor_output: np.ndarray,
        financial_capacity: np.ndarray, debt_stress: np.ndarray,
        leader_continuity: np.ndarray, political_resilience: np.ndarray,
        conflict_regime: np.ndarray, regime_duration: np.ndarray,
        land_control: np.ndarray, proxy_capacity: np.ndarray,
        proxy_leakage: np.ndarray, trade_loss: np.ndarray,
        prior_blockade: np.ndarray, us_ceiling: float, japan_ceiling: float,
    ) -> MechanismFactors:
        n = self.paths
        support = np.clip(actor_support, 0.01, 0.99)
        damage = np.clip(actor_damage, 0.0, 1.0)
        shortage = np.clip(actor_shortage, 0.0, 1.0)
        output = np.clip(actor_output, 0.01, 1.20)
        fiscal = np.clip(financial_capacity, 0.02, 1.0)
        political = np.clip(political_resilience, 0.02, 1.0)

        base_target = np.clip(
            0.20 + 0.34 * shortage[:, 1] + 0.18 * damage[:, 1]
            + 0.16 * (1.0 - political[:, 1]), 0.12, 0.78
        )
        # Four ambiguity scenarios create an empirical upper-tail social loss.
        risk_scenarios = np.stack((
            0.35 * damage.mean(axis=1) + 0.25 * shortage.mean(axis=1),
            0.48 * damage[:, 1] + 0.32 * shortage[:, 1],
            0.30 * debt_stress.mean(axis=1) + 0.30 * (1.0 - output.mean(axis=1)),
            0.25 * (1.0 - political.mean(axis=1)) + 0.30 * prior_blockade,
        ), axis=1)
        tail_loss = np.max(risk_scenarios, axis=1)
        robust_premium = tail_loss - np.mean(risk_scenarios, axis=1)
        target = np.clip(
            base_target + (0.16 * robust_premium if self.flags["robust_design"] else 0.0),
            0.12, 0.82,
        )
        if self.flags["alliance_vcg"]:
            allocation, payments, truthful_utility, ic, ir, budget, resource = self._vcg_alliance(
                target, fiscal, support
            )
        else:
            allocation = np.zeros((n, 3))
            payments = np.zeros((n, 3))
            truthful_utility = np.zeros((n, 3))
            ic = ir = budget = resource = 0.0
        responsibility = np.zeros((n, 4))
        responsibility[:, 1:] = allocation
        transfer = np.zeros((n, 4))
        transfer[:, 1:] = payments
        # Mechanism transfers are accounting credits financed by a common
        # alliance pool. The pool debit is assigned pro rata to fiscal capacity.
        payer_weight = fiscal[:, 1:] / np.maximum(fiscal[:, 1:].sum(axis=1, keepdims=True), 1e-12)
        transfer[:, 1:] -= payments.sum(axis=1, keepdims=True) * payer_weight
        transfer_balance = float(np.max(np.abs(transfer.sum(axis=1))))

        us_share = allocation[:, 1] / np.maximum(target, 1e-9)
        jp_share = allocation[:, 2] / np.maximum(target, 1e-9)
        alliance_multiplier = np.column_stack((
            np.clip(0.78 + 0.52 * us_share, 0.65, 1.30),
            np.clip(0.78 + 0.52 * jp_share, 0.65, 1.30),
        ))
        if self.flags["robust_design"]:
            # With four equiprobable ambiguity states, their maximum is the
            # empirical CVaR at alpha=0.75. Its premium changes execution,
            # rather than remaining only an objective reported after the run.
            alliance_multiplier *= 1.0 + 0.15 * robust_premium[:, None]
            alliance_multiplier = np.clip(alliance_multiplier, 0.65, 1.34)
        if not self.flags["alliance_vcg"]:
            alliance_multiplier[:] = 1.0

        # Milestone aid contract. Effort is hidden; payment slope induces effort.
        effort_cost = self.private_cost / np.maximum(self.private_quality, 1e-6)
        incentive_slope = np.clip(
            0.35 + 0.35 * shortage + 0.20 * damage + 0.10 * (1.0 - output),
            0.20, 0.90,
        )
        effort_target = np.clip(incentive_slope / np.maximum(effort_cost, 0.25), 0.08, 0.92)
        self.contract_effort += 0.16 * (effort_target - self.contract_effort)
        milestone = np.clip(
            self.private_quality * self.contract_effort * (0.70 + 0.30 * output)
            - 0.22 * damage + self.rng.normal(0.0, 0.025, (n, 4)), 0.0, 1.0
        )
        audit_intensity = np.clip(0.20 + 0.40 * (1.0 - self.reputation) + 0.25 * damage, 0.10, 0.85)
        detected_deviation = np.clip(
            audit_intensity * np.maximum(effort_target - self.contract_effort, 0.0), 0.0, 1.0
        )
        aid_multiplier = np.clip(
            0.70 + 0.55 * milestone - 0.45 * detected_deviation, 0.45, 1.25
        )
        if not self.flags["aid_contract"]:
            aid_multiplier[:] = 1.0

        # Multi-attribute procurement uses quality, cost, delay, and resilience.
        delay_risk = np.clip(0.35 * shortage + 0.25 * damage + 0.20 * debt_stress, 0.0, 1.0)
        resilience = np.clip(0.45 * political + 0.30 * fiscal + 0.25 * output, 0.0, 1.0)
        score = (
            0.34 * self.private_quality - 0.24 * self.private_cost / 1.5
            - 0.20 * delay_risk + 0.22 * resilience
        )
        procurement_factor = np.clip(0.86 + 0.28 * _sigmoid(4.0 * (score - 0.15)), 0.82, 1.14)
        if not self.flags["procurement"]:
            procurement_factor[:] = 1.0

        # Information design selects a disclosure precision from a finite menu.
        precisions = np.array([0.25, 0.45, 0.65, 0.82])
        misperception = np.mean(np.abs(support - 0.5), axis=1)
        exposure = np.mean(damage + 0.5 * shortage, axis=1)
        info_loss = np.stack([
            (1.0 - p) * (0.34 + misperception) + p * p * (0.12 + 0.45 * exposure)
            for p in precisions
        ], axis=1)
        precision = precisions[np.argmin(info_loss, axis=1)]
        truth_signal = np.clip(
            0.45 * (political - 0.5) - 0.35 * damage - 0.30 * shortage,
            -0.65, 0.65,
        )
        information_support_shift = precision[:, None] * 0.012 * truth_signal
        panic_reduction = np.clip(
            precision[:, None] * (0.010 + 0.020 * self.reputation), 0.0, 0.028
        )
        if not self.flags["information_design"]:
            precision[:] = 0.0
            information_support_shift[:] = 0.0
            panic_reduction[:] = 0.0

        # Ceasefire mechanism jointly chooses verification and reversible benefits.
        candidate_audit = np.array([0.20, 0.40, 0.60, 0.80])
        fragility = np.clip(
            0.65 * (1.0 - np.mean(political, axis=1))
            + 0.35 * (1.0 - np.mean(leader_continuity, axis=1)),
            0.0, 1.0,
        )
        audit_loss = np.stack([
            np.exp(-3.0 * a) * (0.30 + fragility)
            + (0.006 + 0.008 * a) * (0.35 + exposure) + 0.08 * a * a
            for a in candidate_audit
        ], axis=1)
        audit = candidate_audit[np.argmin(audit_loss, axis=1)]
        false_positive = 0.006 + 0.008 * audit
        false_negative = np.exp(-3.0 * audit)
        compliance = np.clip(
            0.42 * np.mean(self.private_commitment, axis=1)
            + 0.28 * np.mean(political, axis=1)
            + 0.18 * audit - 0.20 * fragility, 0.05, 0.95
        )
        cooling = np.clip(1.0 - false_positive - false_negative * (1.0 - compliance), 0.20, 0.98)
        ceasefire_entry_multiplier = np.clip(0.78 + 0.85 * cooling, 0.70, 1.55)
        ceasefire_break_multiplier = np.clip(1.18 - 0.82 * cooling, 0.32, 1.08)
        settlement_multiplier = np.clip(
            0.75 + 0.72 * cooling + 0.18 * np.minimum(regime_duration / 24.0, 1.0),
            0.70, 1.60,
        )
        if not self.flags["ceasefire_contract"]:
            ceasefire_entry_multiplier[:] = 1.0
            ceasefire_break_multiplier[:] = 1.0
            settlement_multiplier[:] = 1.0

        # Multi-task governance contract values service and trust, and penalizes abuse/leakage.
        service = np.clip(0.42 * land_control + 0.32 * proxy_capacity + 0.26 * output[:, 1], 0.0, 1.0)
        trust = np.clip(0.45 * support[:, 1] + 0.35 * political[:, 1] + 0.20 * self.reputation[:, 1], 0.0, 1.0)
        abuse = np.clip(0.55 * damage[:, 1] + 0.45 * (1.0 - political[:, 1]), 0.0, 1.0)
        governance_score = 0.38 * service + 0.32 * trust - 0.18 * abuse - 0.12 * proxy_leakage
        governance_recruitment_multiplier = np.clip(0.72 + 0.70 * _sigmoid(5.0 * (governance_score - 0.18)), 0.65, 1.35)
        governance_trust_bonus = np.clip(0.012 * governance_score, -0.008, 0.012)
        governance_leakage_reduction = np.clip(0.02 + 0.16 * audit * trust, 0.0, 0.16)
        if not self.flags["governance_contract"]:
            governance_recruitment_multiplier[:] = 1.0
            governance_trust_bonus[:] = 0.0
            governance_leakage_reduction[:] = 0.0

        # Reversible trade relief and zero-sum index insurance.
        public_compliance = np.clip(0.55 * compliance + 0.25 * np.mean(self.reputation, axis=1) + 0.20 * cooling, 0.0, 1.0)
        cooling_state = np.select(
            (conflict_regime == 1, conflict_regime == 2, conflict_regime == 3,
             conflict_regime == 4, conflict_regime == 5),
            (0.25, 0.65, 0.90, 0.45, 1.00),
            default=0.0,
        )
        sanction_relief = np.clip(
            0.005 + 0.175 * public_compliance * cooling_state, 0.0, 0.18
        )
        insured_loss = np.clip(trade_loss + 0.35 * prior_blockade, 0.0, 1.0)
        gross_claim = np.maximum(insured_loss - 0.28, 0.0) * 0.010
        insurance_transfer = np.zeros((n, 4))
        recipient_weights = np.column_stack((
            0.42 + 0.28 * damage[:, 0], 0.58 + 0.42 * damage[:, 1],
            0.08 + 0.12 * damage[:, 2], 0.12 + 0.18 * damage[:, 3],
        ))
        recipient_weights /= np.maximum(recipient_weights.sum(axis=1, keepdims=True), 1e-12)
        payer_weights = fiscal / np.maximum(fiscal.sum(axis=1, keepdims=True), 1e-12)
        insurance_transfer = gross_claim[:, None] * (recipient_weights - payer_weights)
        if not self.flags["trade_insurance"]:
            sanction_relief[:] = 0.0
            insurance_transfer[:] = 0.0

        # Update persistent reputation and promised utility after realized milestones.
        self.reputation = np.clip(
            self.reputation + 0.07 * (milestone - self.reputation)
            - 0.10 * detected_deviation, 0.05, 0.99
        )
        self.audit_memory = 0.80 * self.audit_memory + 0.20 * detected_deviation
        self.promised_utility = 0.94 * self.promised_utility + transfer + 0.02 * milestone
        commitment_gain = 0.02 * public_compliance - 0.03 * fragility
        required_bond = np.maximum(-commitment_gain, 0.0)
        posted_bond = np.minimum(required_bond, 0.04 + 0.03 * fiscal.mean(axis=1))
        commitment_residual = float(np.max(np.maximum(required_bond - posted_bond, 0.0)))
        limited_liability_residual = float(np.max(np.maximum(-0.08 - transfer, 0.0)))

        self.months += 1
        self.max_ic_residual = max(self.max_ic_residual, ic)
        self.max_ir_residual = max(self.max_ir_residual, ir)
        self.max_budget_residual = max(self.max_budget_residual, budget)
        self.max_resource_residual = max(self.max_resource_residual, resource)
        self.max_limited_liability_residual = max(
            self.max_limited_liability_residual, limited_liability_residual
        )
        self.max_commitment_residual = max(self.max_commitment_residual, commitment_residual)
        self.max_transfer_balance_residual = max(self.max_transfer_balance_residual, transfer_balance)
        self.minimum_truthful_surplus = min(
            self.minimum_truthful_surplus, float(np.min(truthful_utility))
        )
        self.minimum_verification_precision = min(
            self.minimum_verification_precision, float(np.min(precision))
        )
        converged = (
            ic < 5e-10 and ir < 5e-10 and budget < 1e-10
            and resource < 1e-10 and limited_liability_residual < 1e-10
            and commitment_residual < 1e-10 and transfer_balance < 1e-10
        )
        if not converged:
            self.nonconverged_months += 1

        factors = MechanismFactors(
            alliance_multiplier=alliance_multiplier,
            responsibility=responsibility,
            transfer=transfer,
            aid_multiplier=aid_multiplier,
            procurement_factor=procurement_factor,
            information_support_shift=information_support_shift,
            panic_reduction=panic_reduction,
            ceasefire_entry_multiplier=ceasefire_entry_multiplier,
            ceasefire_break_multiplier=ceasefire_break_multiplier,
            settlement_multiplier=settlement_multiplier,
            governance_recruitment_multiplier=governance_recruitment_multiplier,
            governance_trust_bonus=governance_trust_bonus,
            governance_leakage_reduction=governance_leakage_reduction,
            sanction_relief=sanction_relief,
            insurance_transfer=insurance_transfer,
            verification_precision=precision,
            converged=converged,
        )
        self.last_factors = factors
        return factors

    def diagnostics(self):
        return {
            "months": self.months,
            "nonconverged_months": self.nonconverged_months,
            "max_incentive_compatibility_residual": self.max_ic_residual,
            "max_participation_residual": self.max_ir_residual,
            "max_budget_residual": self.max_budget_residual,
            "max_resource_residual": self.max_resource_residual,
            "max_limited_liability_residual": self.max_limited_liability_residual,
            "max_dynamic_commitment_residual": self.max_commitment_residual,
            "max_transfer_balance_residual": self.max_transfer_balance_residual,
            "minimum_truthful_surplus": float(self.minimum_truthful_surplus),
            "minimum_verification_precision": self.minimum_verification_precision,
        }
