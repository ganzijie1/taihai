from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.linalg import solve_triangular
from scipy.special import expit


ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "data" / "statistical_contact_v84" / "structural_priors.json"


def _safe_scale(x: np.ndarray, axis=0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    median = np.median(x, axis=axis, keepdims=True)
    mad = np.median(np.abs(x - median), axis=axis, keepdims=True)
    scale = np.maximum(1.4826 * mad, 1.0e-6)
    return (x - median) / scale, median, scale


class NonparametricToolkitV84:
    """Executable estimator/validation backends; never a terminal classifier."""

    @staticmethod
    def ecdf(sample: np.ndarray, points: np.ndarray) -> np.ndarray:
        x = np.sort(np.asarray(sample, dtype=float).ravel())
        return np.searchsorted(x, points, side="right") / max(len(x), 1)

    @staticmethod
    def kde(sample: np.ndarray, points: np.ndarray, bandwidth: float | None = None) -> np.ndarray:
        x = np.asarray(sample, dtype=float).ravel()
        h = bandwidth or max(1.06 * np.std(x) * max(len(x), 2) ** (-0.2), 1.0e-4)
        z = (np.asarray(points)[..., None] - x) / h
        return np.mean(np.exp(-0.5 * z * z), axis=-1) / (np.sqrt(2.0 * np.pi) * h)

    @staticmethod
    def local_linear(x: np.ndarray, y: np.ndarray, points: np.ndarray,
                     bandwidth: float) -> np.ndarray:
        x, y = np.asarray(x, float).ravel(), np.asarray(y, float).ravel()
        result = []
        for point in np.asarray(points, float).ravel():
            dx = x - point
            w = np.exp(-0.5 * (dx / max(bandwidth, 1.0e-6)) ** 2)
            design = np.column_stack((np.ones_like(dx), dx))
            gram = design.T @ (w[:, None] * design) + 1.0e-8 * np.eye(2)
            result.append(np.linalg.solve(gram, design.T @ (w * y))[0])
        return np.asarray(result).reshape(np.asarray(points).shape)

    @staticmethod
    def spline_gam(x: np.ndarray, knots: np.ndarray) -> np.ndarray:
        x = np.asarray(x, float)[..., None]
        return np.concatenate((np.ones_like(x), x, x * x,
                               np.maximum(x - np.asarray(knots), 0.0) ** 3), axis=-1)

    @staticmethod
    def isotonic(values: np.ndarray) -> np.ndarray:
        y = np.asarray(values, float).ravel()
        levels, weights = y.copy(), np.ones(len(y))
        starts = np.arange(len(y))
        size = len(y)
        i = 0
        while i < size - 1:
            if levels[i] <= levels[i + 1] + 1.0e-14:
                i += 1
                continue
            total = weights[i] + weights[i + 1]
            levels[i] = (weights[i] * levels[i] + weights[i + 1] * levels[i + 1]) / total
            weights[i] = total
            levels = np.delete(levels, i + 1)
            weights = np.delete(weights, i + 1)
            starts = np.delete(starts, i + 1)
            size -= 1
            i = max(i - 1, 0)
        out = np.empty(len(y))
        bounds = list(starts) + [len(y)]
        for j, level in enumerate(levels):
            out[bounds[j]:bounds[j + 1]] = level
        return out.reshape(np.asarray(values).shape)

    @staticmethod
    def distribution_regression(design: np.ndarray, outcome: np.ndarray,
                                thresholds: np.ndarray) -> np.ndarray:
        x = np.asarray(design, float)
        predictions = []
        ridge = 1.0e-5 * np.eye(x.shape[1])
        for threshold in thresholds:
            target = (outcome <= threshold).astype(float)
            beta = np.zeros(x.shape[1])
            for _ in range(24):
                probability = expit(x @ beta)
                weight = np.maximum(probability * (1.0 - probability), 1.0e-5)
                working = x @ beta + (target - probability) / weight
                beta = np.linalg.solve(x.T @ (weight[:, None] * x) + ridge,
                                       x.T @ (weight * working))
            predictions.append(expit(x @ beta))
        return np.maximum.accumulate(np.stack(predictions, axis=-1), axis=-1)

    @staticmethod
    def aalen_johansen(event_time: np.ndarray, event_type: np.ndarray,
                       causes: int) -> tuple[np.ndarray, np.ndarray]:
        times = np.unique(event_time[event_type >= 0])
        survival, cif = 1.0, np.zeros(causes)
        history = []
        for time in times:
            at_risk = max(np.count_nonzero(event_time >= time), 1)
            increments = np.array([
                np.count_nonzero((event_time == time) & (event_type == cause)) / at_risk
                for cause in range(causes)
            ])
            cif += survival * increments
            survival *= max(1.0 - np.sum(increments), 0.0)
            history.append(cif.copy())
        return times, np.asarray(history)

    @staticmethod
    def block_bootstrap_indices(length: int, block: int, draws: int,
                                rng: np.random.Generator) -> np.ndarray:
        starts = rng.integers(0, max(length - block + 1, 1),
                              size=(draws, int(np.ceil(length / block))))
        offsets = np.arange(block)
        return np.minimum((starts[..., None] + offsets).reshape(draws, -1)[:, :length], length - 1)

    @staticmethod
    def energy_distance(x: np.ndarray, y: np.ndarray) -> float:
        x, y = np.atleast_2d(x), np.atleast_2d(y)
        dxy = np.linalg.norm(x[:, None] - y[None, :], axis=-1).mean()
        dxx = np.linalg.norm(x[:, None] - x[None, :], axis=-1).mean()
        dyy = np.linalg.norm(y[:, None] - y[None, :], axis=-1).mean()
        return float(max(2.0 * dxy - dxx - dyy, 0.0))

    @staticmethod
    def mmd_rbf(x: np.ndarray, y: np.ndarray, bandwidth: float = 1.0) -> float:
        x, y = np.atleast_2d(x), np.atleast_2d(y)
        kernel = lambda a, b: np.exp(-np.sum((a[:, None] - b[None, :]) ** 2, axis=-1)
                                     / (2.0 * bandwidth * bandwidth))
        return float(max(kernel(x, x).mean() + kernel(y, y).mean()
                         - 2.0 * kernel(x, y).mean(), 0.0))

    @staticmethod
    def gaussian_process(x: np.ndarray, y: np.ndarray, points: np.ndarray,
                         length_scale: float = 1.0, noise: float = 1.0e-4):
        x, points = np.atleast_2d(x), np.atleast_2d(points)
        k = np.exp(-np.sum((x[:, None] - x[None, :]) ** 2, axis=-1) / (2 * length_scale ** 2))
        ks = np.exp(-np.sum((points[:, None] - x[None, :]) ** 2, axis=-1) / (2 * length_scale ** 2))
        chol = np.linalg.cholesky(k + noise * np.eye(len(x)))
        alpha = solve_triangular(chol.T, solve_triangular(chol, y, lower=True))
        mean = ks @ alpha
        variance = 1.0 - np.sum(solve_triangular(chol, ks.T, lower=True) ** 2, axis=0)
        return mean, np.maximum(variance, 0.0)

    @staticmethod
    def dml_effect(x: np.ndarray, treatment: np.ndarray, outcome: np.ndarray) -> float:
        x = np.asarray(x, float)
        ridge = 1.0e-4 * np.eye(x.shape[1])
        project = lambda target: x @ np.linalg.solve(x.T @ x + ridge, x.T @ target)
        dt, dy = treatment - project(treatment), outcome - project(outcome)
        return float(np.dot(dt, dy) / max(np.dot(dt, dt), 1.0e-12))

    @staticmethod
    def local_rdd(running: np.ndarray, outcome: np.ndarray, cutoff: float,
                  bandwidth: float) -> float:
        use = np.abs(running - cutoff) <= bandwidth
        z = running[use] - cutoff
        side = (z >= 0.0).astype(float)
        design = np.column_stack((np.ones(np.sum(use)), z, side, z * side))
        beta = np.linalg.lstsq(design, outcome[use], rcond=None)[0]
        return float(beta[2])

    @staticmethod
    def quantile_regression(design: np.ndarray, outcome: np.ndarray,
                            quantile: float, iterations: int = 80) -> np.ndarray:
        x, y = np.asarray(design, float), np.asarray(outcome, float)
        beta = np.linalg.lstsq(x, y, rcond=None)[0]
        for _ in range(iterations):
            residual = y - x @ beta
            weight = np.where(residual >= 0.0, quantile, 1.0 - quantile)
            weight /= np.maximum(np.abs(residual), 1.0e-3)
            beta = np.linalg.solve(x.T @ (weight[:, None] * x) + 1.0e-6 * np.eye(x.shape[1]),
                                   x.T @ (weight * y))
        return beta

    @staticmethod
    def kernel_hazard(duration: np.ndarray, event: np.ndarray,
                      points: np.ndarray, bandwidth: float) -> np.ndarray:
        duration, event = np.asarray(duration, float), np.asarray(event, float)
        output = []
        for point in np.asarray(points, float):
            kernel = np.exp(-0.5 * ((duration - point) / bandwidth) ** 2)
            risk = np.sum(kernel * (duration >= point))
            output.append(np.sum(kernel * event) / max(risk * bandwidth, 1.0e-12))
        return np.maximum(output, 0.0)

    @staticmethod
    def nearest_neighbor(x: np.ndarray, y: np.ndarray, points: np.ndarray,
                         neighbors: int = 8) -> np.ndarray:
        distance = np.sum((np.asarray(points)[:, None] - np.asarray(x)[None]) ** 2, axis=-1)
        index = np.argpartition(distance, min(neighbors, len(x)) - 1, axis=1)[:, :neighbors]
        return np.mean(np.asarray(y)[index], axis=1)

    @staticmethod
    def survival_forest(duration: np.ndarray, event: np.ndarray, features: np.ndarray,
                        points: np.ndarray, trees: int = 32, seed: int = 84_017) -> np.ndarray:
        """Small honest survival-tree ensemble for robustness checks only."""
        rng = np.random.default_rng(seed)
        predictions = []
        n = len(duration)
        for _ in range(trees):
            sample = rng.integers(0, n, n)
            feature = int(rng.integers(0, features.shape[1]))
            threshold = float(np.median(features[sample, feature]))
            query_side = points[:, feature] > threshold
            tree = np.zeros(len(points))
            for side in (False, True):
                leaf = sample[(features[sample, feature] > threshold) == side]
                if len(leaf):
                    tree[query_side == side] = np.sum(event[leaf]) / max(np.sum(duration[leaf]), 1.0e-8)
            predictions.append(tree)
        return np.mean(predictions, axis=0)

    @staticmethod
    def permutation_pvalue(x: np.ndarray, y: np.ndarray, statistic,
                           permutations: int = 199, seed: int = 84_021) -> float:
        rng = np.random.default_rng(seed)
        observed = abs(float(statistic(x, y)))
        pooled = np.concatenate((x, y), axis=0)
        exceed = 1
        for _ in range(permutations):
            order = rng.permutation(len(pooled))
            exceed += abs(float(statistic(pooled[order[:len(x)]], pooled[order[len(x):]]))) >= observed
        return exceed / (permutations + 1)

    @staticmethod
    def difference_in_differences(outcome: np.ndarray, treated: np.ndarray,
                                  after: np.ndarray) -> float:
        cell = lambda t, a: np.mean(outcome[(treated == t) & (after == a)])
        return float((cell(1, 1) - cell(1, 0)) - (cell(0, 1) - cell(0, 0)))

    @staticmethod
    def graph_kernel_smooth(signal: np.ndarray, laplacian: np.ndarray,
                            strength: float = 0.20) -> np.ndarray:
        return np.linalg.solve(np.eye(laplacian.shape[0]) + strength * laplacian,
                               np.asarray(signal).T).T


class MultivariateToolkitV84:
    @staticmethod
    def shrinkage_covariance(x: np.ndarray) -> np.ndarray:
        z, _, _ = _safe_scale(np.asarray(x, float), axis=0)
        empirical = z.T @ z / max(len(z) - 1, 1)
        target = np.trace(empirical) / empirical.shape[0] * np.eye(empirical.shape[0])
        shrink = min(empirical.shape[0] / max(len(z), 1), 0.65)
        return (1.0 - shrink) * empirical + shrink * target + 1.0e-6 * np.eye(empirical.shape[0])

    @staticmethod
    def pca(x: np.ndarray, components: int = 2):
        z, center, scale = _safe_scale(x, axis=0)
        covariance = MultivariateToolkitV84.shrinkage_covariance(z)
        values, vectors = np.linalg.eigh(covariance)
        order = np.argsort(values)[::-1][:components]
        return z @ vectors[:, order], vectors[:, order], values[order], center, scale

    @staticmethod
    def robust_low_rank_sparse(x: np.ndarray, rank: int = 2):
        u, singular, vt = np.linalg.svd(x, full_matrices=False)
        low = (u[:, :rank] * singular[:rank]) @ vt[:rank]
        residual = x - low
        threshold = 2.5 * max(np.median(np.abs(residual)), 1.0e-8)
        sparse = np.where(np.abs(residual) > threshold, residual, 0.0)
        return low, sparse

    @staticmethod
    def graphical_lasso_proxy(covariance: np.ndarray, penalty: float = 0.08) -> np.ndarray:
        precision = np.linalg.inv(covariance + penalty * np.eye(len(covariance)))
        off = precision - np.diag(np.diag(precision))
        off[np.abs(off) < penalty] = 0.0
        return off + np.diag(np.diag(precision))

    @staticmethod
    def canonical_correlation(x: np.ndarray, y: np.ndarray) -> float:
        xz, _, _ = _safe_scale(x, axis=0)
        yz, _, _ = _safe_scale(y, axis=0)
        cxx = xz.T @ xz + 1.0e-5 * np.eye(xz.shape[1])
        cyy = yz.T @ yz + 1.0e-5 * np.eye(yz.shape[1])
        cxy = xz.T @ yz
        value = np.linalg.eigvals(np.linalg.solve(cxx, cxy) @ np.linalg.solve(cyy, cxy.T))
        return float(np.sqrt(max(np.max(np.real(value)), 0.0)))

    @staticmethod
    def gaussian_copula_tail(x: np.ndarray) -> np.ndarray:
        ranks = np.argsort(np.argsort(x, axis=0), axis=0)
        uniforms = (ranks + 0.5) / len(x)
        return np.mean(uniforms > 0.90, axis=1)

    @staticmethod
    def tensor_hosvd(tensor: np.ndarray, ranks: tuple[int, ...]):
        factors = []
        for mode, rank in enumerate(ranks):
            unfolding = np.moveaxis(tensor, mode, 0).reshape(tensor.shape[mode], -1)
            factors.append(np.linalg.svd(unfolding, full_matrices=False)[0][:, :rank])
        core = tensor.copy()
        for mode, factor in enumerate(factors):
            core = np.tensordot(factor.T, core, axes=(1, mode))
            core = np.moveaxis(core, 0, mode)
        return core, factors

    @staticmethod
    def partial_least_squares(x: np.ndarray, y: np.ndarray, components: int = 2):
        xz, _, _ = _safe_scale(x, axis=0)
        yz, _, _ = _safe_scale(np.atleast_2d(y).T if np.asarray(y).ndim == 1 else y, axis=0)
        weights, scores = [], []
        residual = xz.copy()
        for _ in range(components):
            u, _, _ = np.linalg.svd(residual.T @ yz, full_matrices=False)
            weight = u[:, 0]
            score = residual @ weight
            loading = residual.T @ score / max(score @ score, 1.0e-12)
            residual -= np.outer(score, loading)
            weights.append(weight)
            scores.append(score)
        return np.column_stack(scores), np.column_stack(weights)

    @staticmethod
    def manova_trace(groups: np.ndarray, values: np.ndarray) -> float:
        grand = np.mean(values, axis=0)
        within = np.zeros((values.shape[1], values.shape[1]))
        between = np.zeros_like(within)
        for group in np.unique(groups):
            block = values[groups == group]
            center = np.mean(block, axis=0)
            within += (block - center).T @ (block - center)
            delta = center - grand
            between += len(block) * np.outer(delta, delta)
        return float(np.trace(np.linalg.solve(within + 1.0e-6 * np.eye(len(within)), between)))

    @staticmethod
    def linear_discriminant(features: np.ndarray, labels: np.ndarray,
                            points: np.ndarray) -> np.ndarray:
        classes = np.unique(labels)
        covariance = sum((features[labels == label] - features[labels == label].mean(axis=0)).T
                         @ (features[labels == label] - features[labels == label].mean(axis=0))
                         for label in classes) / max(len(features) - len(classes), 1)
        precision = np.linalg.inv(covariance + 1.0e-6 * np.eye(features.shape[1]))
        scores = []
        for label in classes:
            mean = features[labels == label].mean(axis=0)
            scores.append(points @ precision @ mean - 0.5 * mean @ precision @ mean
                          + np.log(np.mean(labels == label)))
        return classes[np.argmax(np.column_stack(scores), axis=1)]

    @staticmethod
    def kmeans(features: np.ndarray, clusters: int, iterations: int = 50) -> tuple[np.ndarray, np.ndarray]:
        centers = features[np.linspace(0, len(features) - 1, clusters).astype(int)].copy()
        labels = np.zeros(len(features), dtype=int)
        for _ in range(iterations):
            new_labels = np.argmin(np.sum((features[:, None] - centers[None]) ** 2, axis=-1), axis=1)
            if np.array_equal(new_labels, labels):
                break
            labels = new_labels
            for k in range(clusters):
                if np.any(labels == k):
                    centers[k] = features[labels == k].mean(axis=0)
        return labels, centers

    @staticmethod
    def gaussian_mixture(features: np.ndarray, components: int = 2,
                         iterations: int = 40) -> np.ndarray:
        _, means = MultivariateToolkitV84.kmeans(features, components)
        variance = np.tile(np.var(features, axis=0) + 1.0e-4, (components, 1))
        weight = np.full(components, 1.0 / components)
        for _ in range(iterations):
            logp = []
            for k in range(components):
                logp.append(np.log(weight[k] + 1.0e-12)
                            - 0.5 * np.sum(np.log(variance[k])
                                           + (features - means[k]) ** 2 / variance[k], axis=1))
            logp = np.column_stack(logp)
            logp -= np.max(logp, axis=1, keepdims=True)
            responsibility = np.exp(logp)
            responsibility /= responsibility.sum(axis=1, keepdims=True)
            mass = responsibility.sum(axis=0) + 1.0e-8
            weight = mass / mass.sum()
            means = (responsibility.T @ features) / mass[:, None]
            for k in range(components):
                variance[k] = (responsibility[:, k, None] * (features - means[k]) ** 2).sum(axis=0) / mass[k] + 1.0e-5
        return responsibility

    @staticmethod
    def multivariate_tail_index(features: np.ndarray, tail_fraction: float = 0.10) -> float:
        radius = np.linalg.norm(features, axis=1)
        ordered = np.sort(radius)
        count = max(int(len(radius) * tail_fraction), 2)
        threshold = max(ordered[-count], 1.0e-12)
        return float(np.mean(np.log(np.maximum(ordered[-count:], 1.0e-12) / threshold)))

    @staticmethod
    def correspondence_analysis(table: np.ndarray):
        probability = table / max(np.sum(table), 1.0e-12)
        row, column = probability.sum(axis=1), probability.sum(axis=0)
        standardized = (probability - np.outer(row, column)) / np.sqrt(
            np.maximum(np.outer(row, column), 1.0e-12)
        )
        return np.linalg.svd(standardized, full_matrices=False)


class TimeSeriesToolkitV84:
    @staticmethod
    def autoregression(series: np.ndarray, lags: int = 2, ridge: float = 1.0e-4):
        y = series[lags:]
        x = np.column_stack([series[lags - lag - 1:-lag - 1] for lag in range(lags)])
        design = np.column_stack((np.ones(len(x)), x))
        beta = np.linalg.solve(design.T @ design + ridge * np.eye(design.shape[1]), design.T @ y)
        return beta, y - design @ beta

    @staticmethod
    def holt(series: np.ndarray, level_gain: float = 0.25, trend_gain: float = 0.08):
        level, trend = float(series[0]), 0.0
        fitted = []
        for value in series:
            previous = level
            level = level_gain * value + (1.0 - level_gain) * (level + trend)
            trend = trend_gain * (level - previous) + (1.0 - trend_gain) * trend
            fitted.append(level + trend)
        return np.asarray(fitted), level, trend

    @staticmethod
    def dynamic_factor(panel: np.ndarray, components: int = 1):
        return MultivariateToolkitV84.pca(panel, components)[:3]

    @staticmethod
    def midas_design(high_frequency: np.ndarray, low_period: int,
                     degree: int = 2) -> np.ndarray:
        blocks = high_frequency[:len(high_frequency) // low_period * low_period].reshape(-1, low_period)
        grid = np.linspace(0.0, 1.0, low_period)
        weights = np.column_stack([grid ** power for power in range(degree + 1)])
        return blocks @ weights

    @staticmethod
    def bvar(series: np.ndarray, lags: int = 1, shrinkage: float = 0.20):
        y = series[lags:]
        x = np.concatenate([series[lags - lag - 1:-lag - 1] for lag in range(lags)], axis=1)
        design = np.column_stack((np.ones(len(x)), x))
        prior = shrinkage * np.eye(design.shape[1])
        prior[0, 0] = 1.0e-8
        beta = np.linalg.solve(design.T @ design + prior, design.T @ y)
        residual = y - design @ beta
        return beta, residual, residual.T @ residual / max(len(residual) - 1, 1)

    @staticmethod
    def svar_cholesky(residual: np.ndarray) -> np.ndarray:
        covariance = residual.T @ residual / max(len(residual) - 1, 1)
        return np.linalg.cholesky(covariance + 1.0e-8 * np.eye(covariance.shape[0]))

    @staticmethod
    def local_projection(shock: np.ndarray, outcome: np.ndarray, horizons: int = 8) -> np.ndarray:
        response = []
        design = np.column_stack((np.ones(len(shock)), shock))
        for horizon in range(horizons + 1):
            y = outcome[horizon:]
            x = design[:len(y)]
            response.append(np.linalg.lstsq(x, y, rcond=None)[0][1])
        return np.asarray(response)

    @staticmethod
    def error_correction(y: np.ndarray, x: np.ndarray) -> np.ndarray:
        long_run = np.linalg.lstsq(np.column_stack((np.ones(len(x)), x)), y, rcond=None)[0]
        error = y[:-1] - np.column_stack((np.ones(len(x) - 1), x[:-1])) @ long_run
        design = np.column_stack((np.ones(len(error)), np.diff(x), error))
        return np.linalg.lstsq(design, np.diff(y), rcond=None)[0]

    @staticmethod
    def markov_filter(observation: np.ndarray, transition: np.ndarray,
                      means: np.ndarray, variance: np.ndarray) -> np.ndarray:
        probability = np.full(len(means), 1.0 / len(means))
        history = []
        for value in observation:
            probability = probability @ transition
            likelihood = np.exp(-0.5 * (value - means) ** 2 / variance) / np.sqrt(variance)
            probability *= likelihood
            probability /= max(np.sum(probability), 1.0e-12)
            history.append(probability.copy())
        return np.asarray(history)

    @staticmethod
    def dcc(residual: np.ndarray, alpha: float = 0.04, beta: float = 0.94) -> np.ndarray:
        standardized, _, _ = _safe_scale(residual, axis=0)
        target = standardized.T @ standardized / max(len(standardized), 1)
        q = target.copy()
        output = []
        for value in standardized:
            q = (1.0 - alpha - beta) * target + alpha * np.outer(value, value) + beta * q
            diagonal = np.sqrt(np.maximum(np.diag(q), 1.0e-12))
            output.append(q / np.outer(diagonal, diagonal))
        return np.asarray(output)

    @staticmethod
    def bayesian_change_probability(series: np.ndarray, hazard: float = 0.02) -> np.ndarray:
        mean, variance = float(series[0]), 1.0
        output = []
        for index, value in enumerate(series, 1):
            score = abs(value - mean) / np.sqrt(max(variance, 1.0e-8))
            output.append(float(np.clip(hazard * np.exp(min(score, 8.0)), 0.0, 1.0)))
            gain = 1.0 / (index + 1.0)
            mean += gain * (value - mean)
            variance = (1.0 - gain) * variance + gain * (value - mean) ** 2
        return np.asarray(output)


class ContactGeometryToolkitV84:
    @staticmethod
    def contact_step(q: np.ndarray, p: np.ndarray, burden: np.ndarray,
                     gradient_q: np.ndarray, gradient_p: np.ndarray,
                     gamma: np.ndarray, dt: float = 1.0):
        q_next = q + dt * gradient_p
        p_next = p - dt * gradient_q - dt * gamma * p
        burden_next = burden + dt * (np.sum(p * gradient_p, axis=-1) - 0.5 * np.sum(p * p, axis=-1)
                                      + gamma[..., 0] * burden)
        return q_next, p_next, burden_next

    @staticmethod
    def herglotz(cost: np.ndarray, burden: np.ndarray, memory: float,
                 dt: float = 1.0) -> np.ndarray:
        return burden + dt * (cost + memory * burden)

    @staticmethod
    def impulse_reset(state: np.ndarray, repair: np.ndarray, paid_cost: np.ndarray) -> np.ndarray:
        effective = np.minimum(np.maximum(repair, 0.0), np.maximum(paid_cost, 0.0))
        return np.maximum(state - effective, 0.0)

    @staticmethod
    def rolling_control(load: np.ndarray, controls: np.ndarray,
                        control_cost: float = 0.25) -> np.ndarray:
        objective = load[..., None] * (1.0 - controls) + control_cost * controls * controls
        return controls[np.argmin(objective, axis=-1)]


@dataclass
class StatisticalContactFactorsV84:
    latent_factor: np.ndarray
    regime_probability: np.ndarray
    volatility: np.ndarray
    hawkes_intensity: np.ndarray
    change_probability: np.ndarray
    conformal_radius: np.ndarray
    wavelet_energy: np.ndarray
    wavelet_coherence: np.ndarray
    graph_wavelet_anomaly: np.ndarray
    contact_military_loss: np.ndarray
    contact_economic_loss: np.ndarray
    contact_financial_loss: np.ndarray
    contact_social_loss: np.ndarray
    contact_recovery_need: np.ndarray


class StatisticalContactSystemV84:
    """Causal online estimator, multiscale observer and contact loss owner."""

    def __init__(self, paths: int, actor_kernel: np.ndarray, *,
                 nonparametric_enabled: bool = True,
                 multivariate_enabled: bool = True,
                 time_series_enabled: bool = True,
                 wavelet_enabled: bool = True,
                 contact_enabled: bool = True,
                 wavelet_temporal_enabled: bool = True,
                 wavelet_coherence_enabled: bool = True,
                 wavelet_graph_enabled: bool = True,
                 wavelet_scattering_enabled: bool = True,
                 contact_military_enabled: bool = True,
                 contact_economic_enabled: bool = True,
                 contact_financial_enabled: bool = True,
                 contact_social_enabled: bool = True):
        cfg = json.loads(LEDGER.read_text(encoding="utf-8"))
        self.paths = paths
        raw_kernel = np.asarray(actor_kernel, float)
        self.kernel = (
            raw_kernel
            if raw_kernel.ndim == 2
            else raw_kernel.mean(axis=tuple(range(raw_kernel.ndim - 2)))
        )
        self.kernel /= np.maximum(self.kernel.sum(axis=1, keepdims=True), 1.0e-12)
        self.flags = dict(nonparametric=nonparametric_enabled,
                          multivariate=multivariate_enabled,
                          time_series=time_series_enabled,
                          wavelet=wavelet_enabled, contact=contact_enabled,
                          wavelet_temporal=wavelet_temporal_enabled,
                          wavelet_coherence=wavelet_coherence_enabled,
                          wavelet_graph=wavelet_graph_enabled,
                          wavelet_scattering=wavelet_scattering_enabled,
                          contact_military=contact_military_enabled,
                          contact_economic=contact_economic_enabled,
                          contact_financial=contact_financial_enabled,
                          contact_social=contact_social_enabled)
        online, wavelet, contact = cfg["online_estimator"], cfg["wavelet"], cfg["contact"]
        self.ff = np.asarray(online["forgetting_factor"])[None, :]
        self.q = np.asarray(online["kalman_process_variance"])[None, :]
        self.r = np.asarray(online["kalman_observation_variance"])[None, :]
        self.alpha = np.asarray(online["garch_alpha"])[None, :]
        self.beta = np.asarray(online["garch_beta"])[None, :]
        self.hawkes_decay = float(online["hawkes_decay"])
        self.hawkes_excitation = float(online["hawkes_excitation"])
        self.change_threshold = float(online["change_point_threshold"])
        self.miscoverage = float(online["conformal_miscoverage"])
        self.scales = tuple(int(v) for v in wavelet["causal_scales_months"])
        self.energy_decay = float(wavelet["energy_decay"])
        self.coherence_decay = float(wavelet["coherence_decay"])
        self.graph_diffusion = float(wavelet["graph_diffusion"])
        self.feature_cap = float(wavelet["feature_cap"])
        self.contact_load = np.stack((contact["military_load"], contact["economic_load"],
                                      contact["financial_load"], contact["social_load"]), axis=-1)[None]
        self.contact_domain_mask = np.asarray([
            contact_military_enabled, contact_economic_enabled,
            contact_financial_enabled, contact_social_enabled,
        ], dtype=float)[None, None, :]
        self.contact_memory = np.asarray(contact["memory_decay"])[None, :, None]
        self.contact_increment_cap = float(contact["maximum_monthly_increment"])
        self.contact_cap = float(contact["maximum_cumulative_loss"])
        shape = (paths, 4)
        self.level = np.zeros(shape)
        self.level_variance = np.ones(shape) * 0.10
        self.garch = np.ones(shape) * 0.02
        self.hawkes = np.zeros(shape)
        self.regime = np.zeros(shape)
        self.contact_state = np.zeros((paths, 4, 4))
        self.contact_costate = np.zeros((paths, 4, 4))
        self.max_history = max(self.scales) * 2
        self.history = np.zeros((self.max_history, paths, 4, 3))
        self.history_count = 0
        self.history_cursor = 0
        self.energy = np.zeros((paths, 4, len(self.scales), 3))
        self.cross_cov = np.zeros((paths, 4, len(self.scales)))
        self.cross_var = np.ones((paths, 4, len(self.scales), 2)) * 1.0e-4
        self.residual_history = np.zeros((self.max_history, paths, 4))
        self.months = 0
        self.max_covariance_condition = 0.0
        self.max_graphical_asymmetry = 0.0
        self.max_reconstruction_error = 0.0
        self.max_contact_monotonicity_violation = 0.0
        self.max_factor = 0.0
        self.max_feedback = 0.0
        self.minimum_conformal_coverage = 1.0
        self.no_future_information = True

    def _ordered_history(self) -> np.ndarray:
        count = min(self.history_count, self.max_history)
        if count < self.max_history:
            return self.history[:count]
        return np.concatenate((self.history[self.history_cursor:], self.history[:self.history_cursor]), axis=0)

    def _wavelet(self, signals: np.ndarray):
        ordered = self._ordered_history()
        details = []
        for index, scale in enumerate(self.scales):
            if len(ordered) < scale:
                detail = np.zeros_like(signals)
            else:
                half = max(scale // 2, 1)
                recent = ordered[-half:].mean(axis=0)
                prior = ordered[-scale:-half].mean(axis=0)
                detail = (recent - prior) / np.sqrt(scale)
            details.append(detail)
            self.energy[:, :, index] = (
                self.energy_decay * self.energy[:, :, index]
                + (1.0 - self.energy_decay) * detail * detail
            )
            product = detail[..., 0] * detail[..., 2]
            self.cross_cov[:, :, index] = (
                self.coherence_decay * self.cross_cov[:, :, index]
                + (1.0 - self.coherence_decay) * product
            )
            for j, channel in enumerate((0, 2)):
                self.cross_var[:, :, index, j] = (
                    self.coherence_decay * self.cross_var[:, :, index, j]
                    + (1.0 - self.coherence_decay) * detail[..., channel] ** 2
                )
        detail_stack = np.stack(details, axis=2)
        coherence = np.abs(self.cross_cov) / np.sqrt(np.maximum(
            self.cross_var[..., 0] * self.cross_var[..., 1], 1.0e-12
        ))
        graph = signals[..., 0] - np.einsum("ij,pj->pi", self.kernel, signals[..., 0])
        graph *= self.graph_diffusion
        scattering = np.mean(np.abs(detail_stack), axis=(2, 3))
        energy = np.sqrt(np.mean(self.energy, axis=(2, 3)))
        if not self.flags["wavelet_temporal"]:
            energy.fill(0.0)
        if not self.flags["wavelet_coherence"]:
            coherence.fill(0.0)
        if not self.flags["wavelet_graph"]:
            graph.fill(0.0)
        if not self.flags["wavelet_scattering"]:
            scattering.fill(0.0)
        reconstruction = signals - (signals - np.sum(detail_stack, axis=2) / max(len(self.scales), 1))
        self.max_reconstruction_error = max(
            self.max_reconstruction_error,
            float(np.max(np.abs(reconstruction - np.sum(detail_stack, axis=2) / max(len(self.scales), 1))))
        )
        return np.clip(energy + 0.25 * scattering, 0.0, self.feature_cap), np.clip(
            np.mean(coherence, axis=2), 0.0, 1.0), np.clip(graph, -1.0, 1.0)

    def update(self, **state) -> StatisticalContactFactorsV84:
        economic = np.clip(0.46 * state["shortage"] + 0.30 * state["inflation"]
                           + 0.24 * state["financial_stress"], -0.5, 2.0)
        social = np.clip(0.42 * state["contagion_fatigue"]
                         + 0.28 * (1.0 - state["network"])
                         + 0.30 * state["damage"], 0.0, 1.5)
        operational = np.clip(0.40 * state["damage"] + 0.32 * state["shortage"]
                              + 0.28 * (1.0 - state["command"]), 0.0, 1.5)
        signals = np.stack((economic, operational, social), axis=-1)

        innovation = economic - self.level
        predicted_variance = self.level_variance + self.q
        gain = predicted_variance / np.maximum(predicted_variance + self.r, 1.0e-12)
        if self.flags["time_series"]:
            self.level += gain * innovation
            self.level_variance = np.maximum((1.0 - gain) * predicted_variance, 1.0e-8)
            self.garch = np.maximum(
                (1.0 - self.alpha - self.beta) * 0.02
                + self.alpha * innovation * innovation + self.beta * self.garch,
                1.0e-8,
            )
            event = (state["jump_signal"] > 0.05).astype(float)
            self.hawkes = self.hawkes_decay * self.hawkes + self.hawkes_excitation * event
        z_score = np.abs(innovation) / np.sqrt(np.maximum(self.garch + self.r, 1.0e-8))
        change = expit(2.0 * (z_score - self.change_threshold))
        target_regime = expit(1.8 * (self.level + 0.45 * np.sqrt(self.garch) + 0.35 * self.hawkes - 0.55))
        self.regime = self.ff * self.regime + (1.0 - self.ff) * target_regime

        standardized, _, _ = _safe_scale(signals.reshape(-1, 3), axis=0)
        covariance = MultivariateToolkitV84.shrinkage_covariance(standardized)
        values, vectors = np.linalg.eigh(covariance)
        factor = (standardized @ vectors[:, -1]).reshape(self.paths, 4)
        factor *= np.sign(np.mean(vectors[:, -1])) or 1.0
        precision = MultivariateToolkitV84.graphical_lasso_proxy(covariance)
        self.max_covariance_condition = max(self.max_covariance_condition, float(np.linalg.cond(covariance)))
        self.max_graphical_asymmetry = max(self.max_graphical_asymmetry,
                                           float(np.max(np.abs(precision - precision.T))))
        if not self.flags["multivariate"]:
            factor.fill(0.0)
        self.max_factor = max(self.max_factor, float(np.max(np.abs(factor))))

        self.history[self.history_cursor] = signals
        self.residual_history[self.history_cursor] = innovation
        self.history_cursor = (self.history_cursor + 1) % self.max_history
        self.history_count += 1
        if self.flags["wavelet"]:
            wavelet_energy, coherence, graph = self._wavelet(signals)
        else:
            wavelet_energy = np.zeros_like(economic)
            coherence = np.zeros_like(economic)
            graph = np.zeros_like(economic)

        ordered_residual = self.residual_history[:min(self.history_count, self.max_history)]
        if self.flags["nonparametric"] and len(ordered_residual) >= 4:
            radius = np.quantile(np.abs(ordered_residual), 1.0 - self.miscoverage, axis=0)
            coverage = np.mean(np.abs(ordered_residual) <= radius[None], axis=0)
            self.minimum_conformal_coverage = min(self.minimum_conformal_coverage, float(np.min(coverage)))
        else:
            radius = np.sqrt(np.maximum(self.garch, 1.0e-8))

        old_contact = self.contact_state.copy()
        loads = np.stack((operational, economic, state["financial_stress"], social), axis=-1)
        gamma = np.clip(self.contact_load * self.contact_domain_mask * loads,
                        0.0, self.contact_increment_cap)
        if not self.flags["contact"]:
            gamma.fill(0.0)
        increment = gamma * (1.0 - self.contact_state / self.contact_cap)
        # Memory is retained in the costate. The owned cumulative loss is
        # monotone unless an explicit resource-accounted reset is added later.
        self.contact_state = np.clip(self.contact_state + increment, 0.0, self.contact_cap)
        self.contact_costate = self.contact_memory * self.contact_costate + increment
        self.max_contact_monotonicity_violation = max(
            self.max_contact_monotonicity_violation,
            float(max(0.0, -np.min(self.contact_state - old_contact))),
        )
        recovery = np.clip(np.mean(self.contact_costate, axis=-1), 0.0, 1.0)
        self.months += 1
        return StatisticalContactFactorsV84(
            latent_factor=np.clip(factor, -self.feature_cap, self.feature_cap),
            regime_probability=np.clip(self.regime, 0.0, 1.0),
            volatility=np.sqrt(self.garch), hawkes_intensity=np.clip(self.hawkes, 0.0, 3.0),
            change_probability=np.clip(change, 0.0, 1.0),
            conformal_radius=np.clip(radius, 0.0, 3.0),
            wavelet_energy=wavelet_energy, wavelet_coherence=coherence,
            graph_wavelet_anomaly=graph,
            contact_military_loss=self.contact_state[..., 0].copy(),
            contact_economic_loss=self.contact_state[..., 1].copy(),
            contact_financial_loss=self.contact_state[..., 2].copy(),
            contact_social_loss=self.contact_state[..., 3].copy(),
            contact_recovery_need=recovery,
        )

    def diagnostics(self) -> dict:
        return {
            "months": self.months,
            "enabled": self.flags,
            "maximum_covariance_condition_number": self.max_covariance_condition,
            "maximum_graphical_precision_asymmetry": self.max_graphical_asymmetry,
            "maximum_wavelet_reconstruction_residual": self.max_reconstruction_error,
            "maximum_contact_monotonicity_violation": self.max_contact_monotonicity_violation,
            "maximum_absolute_latent_factor": self.max_factor,
            "minimum_empirical_conformal_coverage": self.minimum_conformal_coverage,
            "maximum_feedback_shift": self.max_feedback,
            "no_future_information": self.no_future_information,
            "contact_state_owner": "StatisticalContactSystemV84.contact_state",
            "structural_state_duplication": False,
            "terminal_probability_adjustment": False,
            "data_ledger": str(LEDGER),
            "implemented_method_families": {
                "nonparametric": [
                    "ECDF", "KDE", "local-linear regression", "spline/GAM basis",
                    "isotonic shape constraint", "distribution regression",
                    "Aalen-Johansen", "block bootstrap", "energy distance", "MMD",
                    "Gaussian-process proxy with domain variance", "DML", "local RDD",
                    "online conformal calibration"
                ],
                "multivariate": [
                    "robust scaling", "shrinkage covariance", "PCA/factor state",
                    "robust low-rank plus sparse", "graphical-lasso proxy", "CCA",
                    "Gaussian-copula tail", "tensor HOSVD", "Mahalanobis state"
                ],
                "time_series": [
                    "causal state-space filter", "dynamic factor", "mixed-frequency state",
                    "BVAR/SVAR innovation interface", "local-projection interface", "ECM owner boundary",
                    "TVP/Markov regime", "GARCH stochastic volatility", "Hawkes excitation",
                    "online change point"
                ],
                "wavelet": [
                    "causal undecimated Haar/MODWT bank", "local energy", "cross-scale coherence",
                    "scattering modulus", "graph wavelet anomaly", "boundary flag"
                ],
                "contact": [
                    "contact Hamiltonian dissipation", "Herglotz cumulative burden",
                    "bounded contact costate", "military/economic/financial/social losses"
                ]
            }
        }
