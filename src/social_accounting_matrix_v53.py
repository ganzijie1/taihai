from __future__ import annotations

from dataclasses import dataclass

import numpy as np


SECTORS = tuple(f"commodity_{index}" for index in range(10))
SAM_ACCOUNTS = SECTORS + (
    "labor",
    "capital_factor",
    "household_low",
    "household_middle",
    "household_high",
    "enterprises",
    "government",
    "capital_account",
    "rest_of_world",
)


@dataclass(frozen=True)
class SAMCalibration:
    benchmark: np.ndarray
    ras_iterations: np.ndarray
    ras_residual: np.ndarray
    observed_share: np.ndarray


class FourPartySocialAccountingMatrix:
    """Balanced benchmark and monthly transaction SAM for four actors.

    Rows receive and columns pay. Production, intermediate demand, value added
    and final demand come from the public IO base. Household splitting and the
    institutional allocation of value added are transparent structural priors;
    they only allocate observed control totals. A biproportional projection
    balances the benchmark without creating negative cells.
    """

    def __init__(self, cge, paths: int):
        self.paths = paths
        self.n_actors = 4
        self.n_accounts = len(SAM_ACCOUNTS)
        self.actor_of_node = cge.actor_of_node.copy()
        self.actor_of_region = cge.actor_of_region.copy()
        self.sector_of_node = cge.sector.copy()
        self.calibration = self._build_benchmark(cge)
        self.benchmark = self.calibration.benchmark
        self.transactions = np.zeros(
            (paths, self.n_actors, self.n_accounts, self.n_accounts)
        )
        self.cumulative_transactions = np.zeros_like(self.transactions)
        self.household_disposable_income = np.zeros((paths, 4, 3))
        self.institutional_saving = np.zeros((paths, 4, 5))
        self.max_dynamic_balance_residual = 0.0
        self.max_saving_investment_residual = 0.0
        self.minimum_transaction = 0.0
        self.months = 0

    @staticmethod
    def _post(matrix: np.ndarray, recipient: int, payer: int, amount: np.ndarray) -> None:
        matrix[:, :, recipient, payer] += np.maximum(amount, 0.0)

    @staticmethod
    def _ras(prior: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, int, float]:
        matrix = np.maximum(prior, 1.0e-12)
        target = np.maximum(target, 1.0e-9)
        matrix *= target.sum() / max(matrix.sum(), 1.0e-12)
        residual = np.inf
        for iteration in range(4000):
            matrix *= (target / np.maximum(matrix.sum(axis=1), 1.0e-15))[:, None]
            matrix *= (target / np.maximum(matrix.sum(axis=0), 1.0e-15))[None, :]
            residual = float(max(
                np.max(np.abs(matrix.sum(axis=1) - target)),
                np.max(np.abs(matrix.sum(axis=0) - target)),
            ))
            if residual < 1.0e-9:
                break
        return matrix, iteration + 1, residual

    def _build_benchmark(self, cge) -> SAMCalibration:
        result = np.zeros((4, self.n_accounts, self.n_accounts))
        iterations = np.zeros(4, dtype=int)
        residuals = np.zeros(4)
        observed_shares = np.zeros(4)
        household_income = np.array([0.20, 0.50, 0.30])
        household_consumption = np.array([0.26, 0.51, 0.23])

        for actor in range(4):
            prior = np.zeros((self.n_accounts, self.n_accounts))
            node_mask = self.actor_of_node == actor
            region_mask = self.actor_of_region == actor
            nodes = np.flatnonzero(node_mask)
            regions = np.flatnonzero(region_mask)

            # Observed domestic intermediate transactions, aggregated to ten sectors.
            for source in nodes:
                s = self.sector_of_node[source]
                for destination in nodes:
                    d = self.sector_of_node[destination]
                    prior[s, d] += cge.z[source, destination]

            sector_va = np.bincount(
                self.sector_of_node[nodes], weights=cge.va[nodes], minlength=10
            )
            prior[10, :10] += 0.58 * sector_va
            prior[11, :10] += 0.42 * sector_va
            prior[12:15, 10] += household_income * prior[10, :10].sum()
            prior[14, 11] += 0.38 * prior[11, :10].sum()
            prior[15, 11] += 0.62 * prior[11, :10].sum()

            final_by_sector = np.bincount(
                self.sector_of_node[nodes],
                weights=cge.fd[np.ix_(nodes, regions)].sum(axis=1),
                minlength=10,
            )
            final_weights = final_by_sector / max(final_by_sector.sum(), 1.0e-12)
            household_total = 0.58 * final_by_sector.sum()
            government_total = 0.18 * final_by_sector.sum()
            investment_total = 0.24 * final_by_sector.sum()
            for group in range(3):
                prior[:10, 12 + group] += (
                    household_total * household_consumption[group] * final_weights
                )
            prior[:10, 16] += government_total * final_weights
            prior[:10, 17] += investment_total * final_weights

            # Cross-border intermediate and final flows supply observed ROW controls.
            outside_nodes = np.flatnonzero(~node_mask)
            imports = cge.z[np.ix_(outside_nodes, nodes)].sum(axis=0)
            exports = cge.z[np.ix_(nodes, outside_nodes)].sum(axis=1)
            import_sector = np.bincount(
                self.sector_of_node[nodes], weights=imports, minlength=10
            )
            export_sector = np.bincount(
                self.sector_of_node[nodes], weights=exports, minlength=10
            )
            prior[18, :10] += import_sector
            prior[:10, 18] += export_sector

            # Institutional priors allocate observed totals; balancing determines
            # the consistent residual savings and transfers.
            labor_income = prior[10, :10].sum()
            capital_income = prior[11, :10].sum()
            prior[16, 12:15] += np.array([0.04, 0.10, 0.17]) * labor_income
            prior[16, 15] += 0.16 * capital_income
            prior[12:15, 16] += np.array([0.055, 0.035, 0.015]) * labor_income
            prior[17, 12:15] += np.array([0.02, 0.08, 0.18]) * labor_income
            prior[17, 15] += 0.28 * capital_income
            prior[17, 16] += 0.04 * final_by_sector.sum()

            row = prior.sum(axis=1)
            column = prior.sum(axis=0)
            target = 0.5 * (row + column)
            observed_mass = (
                prior[:12, :10].sum() + prior[:10, 12:].sum()
                + prior[18, :10].sum() + prior[:10, 18].sum()
            )
            observed_shares[actor] = observed_mass / max(prior.sum(), 1.0e-12)
            # Store the benchmark as transaction shares. This avoids mixing
            # national-currency levels and makes the balancing residual
            # comparable across actors while preserving every cell ratio.
            prior /= max(prior.sum(), 1.0e-12)
            row = prior.sum(axis=1)
            column = prior.sum(axis=0)
            target = 0.5 * (row + column)
            balanced, count, error = self._ras(prior, target)
            result[actor] = balanced
            iterations[actor] = count
            residuals[actor] = error
        return SAMCalibration(result, iterations, residuals, observed_shares)

    def update(
        self,
        *,
        actual_nodes: np.ndarray,
        node_prices: np.ndarray,
        employment: np.ndarray,
        consumption: np.ndarray,
        taxes: np.ndarray,
        procurement: np.ndarray,
        transfers: np.ndarray,
        investment: np.ndarray,
        trade_balance: np.ndarray,
    ) -> None:
        matrix = np.zeros_like(self.transactions)
        sector_sales = np.zeros((self.paths, 4, 10))
        for actor in range(4):
            for sector in range(10):
                mask = (self.actor_of_node == actor) & (self.sector_of_node == sector)
                sector_sales[:, actor, sector] = np.sum(
                    actual_nodes[:, mask] * node_prices[:, mask], axis=1
                )
        sector_weights = sector_sales / np.maximum(
            sector_sales.sum(axis=2, keepdims=True), 1.0e-12
        )
        household_income_shares = np.array([0.20, 0.50, 0.30])
        household_spending_shares = np.array([0.28, 0.51, 0.21])

        intermediate_total = 0.34 * sector_sales.sum(axis=2)
        for supplier in range(10):
            for buyer in range(10):
                amount = (
                    intermediate_total * sector_weights[:, :, supplier]
                    * sector_weights[:, :, buyer]
                )
                self._post(matrix, supplier, buyer, amount)

        wage_bill = 0.50 * sector_sales.sum(axis=2) * employment
        capital_income = 0.22 * sector_sales.sum(axis=2)
        for sector in range(10):
            self._post(matrix, 10, sector, wage_bill * sector_weights[:, :, sector])
            self._post(matrix, 11, sector, capital_income * sector_weights[:, :, sector])
        for group in range(3):
            self._post(matrix, 12 + group, 10, wage_bill * household_income_shares[group])
        self._post(matrix, 14, 11, 0.38 * capital_income)
        self._post(matrix, 15, 11, 0.62 * capital_income)

        for group in range(3):
            group_consumption = consumption * household_spending_shares[group]
            for sector in range(10):
                self._post(
                    matrix, sector, 12 + group,
                    group_consumption * sector_weights[:, :, sector],
                )
        household_tax_shares = np.array([0.08, 0.34, 0.38])
        for group in range(3):
            self._post(matrix, 16, 12 + group, taxes * household_tax_shares[group])
            self._post(matrix, 12 + group, 16, transfers * household_income_shares[group])
        self._post(matrix, 16, 15, 0.20 * taxes)
        for sector in range(10):
            self._post(matrix, sector, 16, procurement * sector_weights[:, :, sector])
            self._post(matrix, sector, 17, investment * sector_weights[:, :, sector])

        exports = np.maximum(trade_balance, 0.0)
        imports = np.maximum(-trade_balance, 0.0)
        for sector in range(10):
            self._post(matrix, sector, 18, exports * sector_weights[:, :, sector])
            self._post(matrix, 18, sector, imports * sector_weights[:, :, sector])

        # Capital account clears every institutional budget. Since every entry
        # is double-entry, balancing all other accounts also balances capital.
        for account in range(self.n_accounts):
            if account == 17:
                continue
            net_receipts = matrix[:, :, account, :].sum(axis=2) - matrix[:, :, :, account].sum(axis=2)
            self._post(matrix, 17, account, np.maximum(net_receipts, 0.0))
            self._post(matrix, account, 17, np.maximum(-net_receipts, 0.0))

        balance = matrix.sum(axis=3) - matrix.sum(axis=2)
        self.max_dynamic_balance_residual = max(
            self.max_dynamic_balance_residual, float(np.max(np.abs(balance)))
        )
        capital_receipts = matrix[:, :, 17, :].sum(axis=2)
        capital_spending = matrix[:, :, :, 17].sum(axis=2)
        self.max_saving_investment_residual = max(
            self.max_saving_investment_residual,
            float(np.max(np.abs(capital_receipts - capital_spending))),
        )
        self.minimum_transaction = min(self.minimum_transaction, float(matrix.min()))
        self.transactions = matrix
        self.cumulative_transactions += matrix
        self.household_disposable_income = (
            matrix[:, :, 12:15, :].sum(axis=3)
            - matrix[:, :, :, 12:15].sum(axis=2)
            + matrix[:, :, 17, 12:15]
        )
        self.institutional_saving = matrix[:, :, 17, 12:17]
        self.months += 1

    def diagnostics(self) -> dict:
        benchmark_balance = self.benchmark.sum(axis=2) - self.benchmark.sum(axis=1)
        return {
            "account_names": list(SAM_ACCOUNTS),
            "account_count": self.n_accounts,
            "months": self.months,
            "benchmark_balance_residual": float(np.max(np.abs(benchmark_balance))),
            "benchmark_ras_residual": float(np.max(self.calibration.ras_residual)),
            "benchmark_ras_iterations": self.calibration.ras_iterations.tolist(),
            "benchmark_minimum_cell": float(np.min(self.benchmark)),
            "observed_control_total_share": self.calibration.observed_share.tolist(),
            "max_dynamic_balance_residual": self.max_dynamic_balance_residual,
            "max_saving_investment_residual": self.max_saving_investment_residual,
            "minimum_transaction": self.minimum_transaction,
            "data_classification": {
                "production_intermediate_value_added_final_demand_trade": "observed_or_derived_public_IO",
                "household_groups_and_institutional_allocation": "structural_prior_balanced_to_observed_control_totals",
            },
            "state_owner": {
                "sam_transactions_and_institutional_distribution": "FourPartySocialAccountingMatrix",
                "inventory_backlog_contract_prices": "MRRDCGEDQSFCState",
                "bank_capital_public_debt_reserves": "FourPartyFinancialSystem",
            },
        }
