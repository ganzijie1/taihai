import csv
import json
import math
import random
import sys
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
SEED = 20260930

MONTHLY_ADIZ = {
    2020: [None, None, None, None, None, None, None, None, 69, 28, 41, 32],
    2021: [81, 40, 54, 107, 29, 43, 17, 34, 117, 196, 168, 86],
    2022: [143, 70, 64, 75, 119, 84, 70, 446, 192, 97, 171, 207],
    2023: [126, 113, 122, 259, 125, 109, 163, 130, 225, 141, 110, 80],
    2024: [72, 86, 140, 157, 289, 325, 437, 399, 322, 370, 269, 209],
    2025: [248, 362, 309, 364, 339, 356, 392, 317, 313, 222, 266, 274],
    2026: [166, 147, 121, 169, 216, 134, 190, 125, 159, None, None, None],
}

# Comparable broad totals in NT$ billion. 2026 uses the planned MND total,
# excluding the broader NATO-style retirement/coast-guard additions.
DEFENSE_BUDGET = {2020: 411.3, 2021: 448.3, 2022: 514.7, 2023: 580.3, 2024: 600.6, 2025: 644.9, 2026: 806.0}
EXPORT_SHARE_CN_HK = {2020: 43.9, 2021: 42.3, 2022: 38.8, 2023: 35.2, 2024: 31.7, 2025: 26.6, 2026: 26.6}

# Exogenous event coding, fixed before estimation. It is intentionally sparse.
POLITICAL_SHOCKS = {
    "2021-10",  # National Day pressure wave
    "2022-08",  # Pelosi visit and exercises
    "2023-04",  # Tsai-McCarthy meeting and exercises
    "2024-05",  # inauguration and Joint Sword-2024A
    "2024-10",  # Joint Sword-2024B
    "2025-04",  # Strait Thunder-2025A
}
ELECTION_MONTHS = {"2022-11", "2024-01"}


def build_rows():
    rows = []
    for year, months in MONTHLY_ADIZ.items():
        for month, count in enumerate(months, 1):
            if count is None:
                continue
            key = f"{year:04d}-{month:02d}"
            previous = DEFENSE_BUDGET.get(year - 1, DEFENSE_BUDGET[year])
            growth = 100.0 * (DEFENSE_BUDGET[year] / previous - 1.0)
            rows.append({
                "month": key,
                "adiz": count,
                "log_adiz": math.log1p(count),
                "shock": int(key in POLITICAL_SHOCKS),
                "election": int(key in ELECTION_MONTHS),
                "defense_budget": DEFENSE_BUDGET[year],
                "defense_growth": growth,
                "export_share_cn_hk": EXPORT_SHARE_CN_HK[year],
            })
    return rows


def log_normal_pdf(x, mean, variance):
    variance = max(variance, 1e-5)
    return -0.5 * (math.log(2.0 * math.pi * variance) + (x - mean) ** 2 / variance)


def normalize(log_values):
    maximum = max(log_values)
    values = [math.exp(v - maximum) for v in log_values]
    total = sum(values)
    return [v / total for v in values], maximum + math.log(total)


def fit_gaussian_hmm(values, states=3, iterations=300):
    ordered = sorted(values)
    means = [ordered[int((i + 0.5) * len(ordered) / states)] for i in range(states)]
    overall_mean = sum(values) / len(values)
    overall_var = sum((x - overall_mean) ** 2 for x in values) / len(values)
    variances = [overall_var for _ in range(states)]
    transition = [[0.10 / (states - 1) for _ in range(states)] for _ in range(states)]
    for i in range(states):
        transition[i][i] = 0.90
    initial = [1.0 / states] * states
    previous_ll = -1e100

    for _ in range(iterations):
        emission = [[math.exp(log_normal_pdf(x, means[j], variances[j])) for j in range(states)] for x in values]
        alpha = [[0.0] * states for _ in values]
        scales = [0.0] * len(values)
        for j in range(states):
            alpha[0][j] = initial[j] * emission[0][j]
        scales[0] = sum(alpha[0])
        alpha[0] = [v / scales[0] for v in alpha[0]]
        for t in range(1, len(values)):
            for j in range(states):
                alpha[t][j] = emission[t][j] * sum(alpha[t - 1][i] * transition[i][j] for i in range(states))
            scales[t] = sum(alpha[t])
            alpha[t] = [v / scales[t] for v in alpha[t]]

        beta = [[0.0] * states for _ in values]
        beta[-1] = [1.0] * states
        for t in range(len(values) - 2, -1, -1):
            for i in range(states):
                beta[t][i] = sum(transition[i][j] * emission[t + 1][j] * beta[t + 1][j] for j in range(states)) / scales[t + 1]

        gamma = []
        for t in range(len(values)):
            probs = [alpha[t][j] * beta[t][j] for j in range(states)]
            total = sum(probs)
            gamma.append([v / total for v in probs])

        xi_sums = [[0.0] * states for _ in range(states)]
        gamma_origin = [0.0] * states
        for t in range(len(values) - 1):
            denom = sum(alpha[t][i] * transition[i][j] * emission[t + 1][j] * beta[t + 1][j]
                        for i in range(states) for j in range(states))
            for i in range(states):
                gamma_origin[i] += gamma[t][i]
                for j in range(states):
                    xi_sums[i][j] += alpha[t][i] * transition[i][j] * emission[t + 1][j] * beta[t + 1][j] / denom

        initial = gamma[0]
        for i in range(states):
            row = [(xi_sums[i][j] + 0.5) / (gamma_origin[i] + 0.5 * states) for j in range(states)]
            total = sum(row)
            transition[i] = [v / total for v in row]
        for j in range(states):
            weight = sum(gamma[t][j] for t in range(len(values)))
            means[j] = sum(gamma[t][j] * values[t] for t in range(len(values))) / weight
            variances[j] = max(sum(gamma[t][j] * (values[t] - means[j]) ** 2 for t in range(len(values))) / weight, 0.01)

        ll = sum(math.log(s) for s in scales)
        if abs(ll - previous_ll) < 1e-8:
            break
        previous_ll = ll

    order = sorted(range(states), key=lambda j: means[j])
    inverse = {old: new for new, old in enumerate(order)}
    means = [means[j] for j in order]
    variances = [variances[j] for j in order]
    transition = [[transition[i][j] for j in order] for i in order]
    transition = [transition[i] for i in order]
    gamma = [[row[j] for j in order] for row in gamma]
    path = [max(range(states), key=lambda j: gamma[t][j]) for t in range(len(values))]
    return {"means": means, "variances": variances, "transition": transition, "posterior": gamma, "path": path, "log_likelihood": previous_ll}


def standardize(matrix):
    columns = list(zip(*matrix))
    means = [sum(col) / len(col) for col in columns]
    scales = [math.sqrt(sum((x - m) ** 2 for x in col) / len(col)) or 1.0 for col, m in zip(columns, means)]
    return [[(x - means[j]) / scales[j] for j, x in enumerate(row)] for row in matrix], means, scales


def sigmoid(x):
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def ridge_logit(features, labels, penalty=2.0, iterations=8000, rate=0.03):
    weights = [0.0] * (len(features[0]) + 1)
    for step in range(iterations):
        grad = [0.0] * len(weights)
        for row, label in zip(features, labels):
            vector = [1.0] + row
            pred = sigmoid(sum(w * x for w, x in zip(weights, vector)))
            for j, x in enumerate(vector):
                grad[j] += (pred - label) * x
        for j in range(1, len(weights)):
            grad[j] += penalty * weights[j]
        eta = rate / math.sqrt(1.0 + step / 500.0)
        for j in range(len(weights)):
            weights[j] -= eta * grad[j] / len(features)
    return weights


def log_loss(features, labels, weights):
    loss = 0.0
    for row, label in zip(features, labels):
        pred = min(max(sigmoid(weights[0] + sum(w * x for w, x in zip(weights[1:], row))), 1e-8), 1 - 1e-8)
        loss -= label * math.log(pred) + (1 - label) * math.log(1 - pred)
    return loss / len(labels)


def solve_linear(matrix, vector):
    n = len(vector)
    augmented = [list(matrix[i]) + [vector[i]] for i in range(n)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda row: abs(augmented[row][col]))
        augmented[col], augmented[pivot] = augmented[pivot], augmented[col]
        divisor = augmented[col][col]
        if abs(divisor) < 1e-12:
            divisor = 1e-12
        augmented[col] = [value / divisor for value in augmented[col]]
        for row in range(n):
            if row == col:
                continue
            factor = augmented[row][col]
            augmented[row] = [a - factor * b for a, b in zip(augmented[row], augmented[col])]
    return [augmented[i][-1] for i in range(n)]


def ridge_linear(features, targets, penalty):
    design = [[1.0] + row for row in features]
    p = len(design[0])
    gram = [[sum(row[i] * row[j] for row in design) for j in range(p)] for i in range(p)]
    rhs = [sum(row[i] * target for row, target in zip(design, targets)) for i in range(p)]
    for i in range(1, p):
        gram[i][i] += penalty
    return solve_linear(gram, rhs)


def linear_rmse(features, targets, weights):
    errors = []
    for row, target in zip(features, targets):
        pred = weights[0] + sum(w * x for w, x in zip(weights[1:], row))
        errors.append((target - pred) ** 2)
    return math.sqrt(sum(errors) / len(errors))


def percentile(values, q):
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def fit_pressure_dynamics(rows):
    names = ["lag_log_adiz", "shock", "election", "defense_growth", "export_share_cn_hk", "time_trend"]
    raw, targets, years = [], [], []
    for t in range(1, len(rows)):
        raw.append([
            rows[t - 1]["log_adiz"], rows[t]["shock"], rows[t]["election"],
            rows[t]["defense_growth"], rows[t]["export_share_cn_hk"], t / (len(rows) - 1),
        ])
        targets.append(rows[t]["log_adiz"])
        years.append(int(rows[t]["month"][:4]))
    x, means, scales = standardize(raw)
    best = None
    for penalty in (0.1, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0):
        errors = []
        for held_year in sorted(set(years)):
            train_x = [row for row, year in zip(x, years) if year != held_year]
            train_y = [target for target, year in zip(targets, years) if year != held_year]
            test_x = [row for row, year in zip(x, years) if year == held_year]
            test_y = [target for target, year in zip(targets, years) if year == held_year]
            weights = ridge_linear(train_x, train_y, penalty)
            errors.append(linear_rmse(test_x, test_y, weights))
        score = sum(errors) / len(errors)
        if best is None or score < best[0]:
            best = (score, penalty)
    weights = ridge_linear(x, targets, best[1])
    fitted_rmse = linear_rmse(x, targets, weights)
    rng = random.Random(SEED)
    unique_years = sorted(set(years))
    boot = [[] for _ in weights]
    for _ in range(2000):
        sampled_years = [rng.choice(unique_years) for _ in unique_years]
        indices = []
        for sampled_year in sampled_years:
            indices.extend(i for i, year in enumerate(years) if year == sampled_year)
        sample_x = [x[i] for i in indices]
        sample_y = [targets[i] for i in indices]
        sample_weights = ridge_linear(sample_x, sample_y, best[1])
        for j, value in enumerate(sample_weights):
            boot[j].append(value)
    intervals = {name: [percentile(boot[j + 1], 0.10), percentile(boot[j + 1], 0.90)] for j, name in enumerate(names)}
    return {
        "n": len(targets), "penalty": best[1], "blocked_cv_rmse_log_points": best[0],
        "in_sample_rmse_log_points": fitted_rmse, "intercept": weights[0],
        "coefficients_standardized": dict(zip(names, weights[1:])),
        "coefficient_intervals_block_bootstrap_p10_p90": intervals,
        "standardization": {name: {"mean": means[i], "scale": scales[i]} for i, name in enumerate(names)},
    }


def fit_slow_feedback(rows):
    annual = {}
    for row in rows:
        year = int(row["month"][:4])
        annual.setdefault(year, []).append(row["log_adiz"])
    average_pressure = {year: sum(values) / len(values) for year, values in annual.items()}
    years = [year for year in sorted(annual) if year - 1 in average_pressure and year in DEFENSE_BUDGET]

    lag_pressure = [[average_pressure[year - 1]] for year in years]
    pressure_x, pressure_mean, pressure_scale = standardize(lag_pressure)
    budget_growth = [100 * (DEFENSE_BUDGET[year] / DEFENSE_BUDGET[year - 1] - 1) for year in years]
    trade_change = [EXPORT_SHARE_CN_HK[year] - EXPORT_SHARE_CN_HK[year - 1] for year in years]
    budget_weights = ridge_linear(pressure_x, budget_growth, 2.0)
    trade_weights = ridge_linear(pressure_x, trade_change, 2.0)
    return {
        "years": years,
        "n": len(years),
        "warning": "Annual feedback estimates have only six transitions and are descriptive, not causal.",
        "lag_pressure_standardization": {"mean": pressure_mean[0], "scale": pressure_scale[0]},
        "taiwan_defense_budget_growth": {"intercept": budget_weights[0], "lag_pressure_coefficient_standardized": budget_weights[1]},
        "trade_derisking_change_pp": {"intercept": trade_weights[0], "lag_pressure_coefficient_standardized": trade_weights[1]},
    }


def fit_transition_response(rows, states):
    names = ["shock", "election", "defense_growth", "export_share_cn_hk", "lag_log_adiz", "previous_state"]
    raw = []
    transitions = []
    for t in range(1, len(rows)):
        raw.append([
            rows[t]["shock"], rows[t]["election"], rows[t]["defense_growth"],
            rows[t]["export_share_cn_hk"], rows[t - 1]["log_adiz"], states[t - 1],
        ])
        transitions.append((states[t - 1], states[t]))
    x, means, scales = standardize(raw)

    results = {}
    for label, eligible, outcome in (
        ("escalation", lambda p: p < 2, lambda p, n: int(n > p)),
        ("deescalation", lambda p: p > 0, lambda p, n: int(n < p)),
    ):
        xf = [row for row, tr in zip(x, transitions) if eligible(tr[0])]
        yf = [outcome(*tr) for tr in transitions if eligible(tr[0])]
        best = None
        for penalty in (0.5, 1.0, 2.0, 4.0, 8.0):
            losses = []
            years = sorted({int(rows[t]["month"][:4]) for t in range(1, len(rows))})
            for held_year in years:
                train_x, train_y, test_x, test_y = [], [], [], []
                eligible_index = 0
                for t, tr in enumerate(transitions, 1):
                    if not eligible(tr[0]):
                        continue
                    if int(rows[t]["month"][:4]) == held_year:
                        test_x.append(xf[eligible_index]); test_y.append(yf[eligible_index])
                    else:
                        train_x.append(xf[eligible_index]); train_y.append(yf[eligible_index])
                    eligible_index += 1
                if train_x and test_x and len(set(train_y)) == 2:
                    w = ridge_logit(train_x, train_y, penalty=penalty, iterations=3000)
                    losses.append(log_loss(test_x, test_y, w))
            score = sum(losses) / len(losses) if losses else 99.0
            if best is None or score < best[0]:
                best = (score, penalty)
        weights = ridge_logit(xf, yf, penalty=best[1])
        results[label] = {
            "n": len(yf), "events": sum(yf), "penalty": best[1], "blocked_cv_logloss": best[0],
            "intercept": weights[0], "coefficients_standardized": dict(zip(names, weights[1:])),
        }
    results["standardization"] = {name: {"mean": means[i], "scale": scales[i]} for i, name in enumerate(names)}
    return results


def stationary_distribution(transition):
    p = [1 / len(transition)] * len(transition)
    for _ in range(10000):
        nxt = [sum(p[i] * transition[i][j] for i in range(len(p))) for j in range(len(p))]
        if max(abs(a - b) for a, b in zip(p, nxt)) < 1e-14:
            break
        p = nxt
    return p


def adjusted_transition(base, delta):
    adjusted = []
    for i, row in enumerate(base):
        values = []
        for j, probability in enumerate(row):
            direction = 1 if j > i else (-1 if j < i else 0)
            values.append(probability * math.exp(direction * delta))
        total = sum(values)
        adjusted.append([value / total for value in values])
    return adjusted


def forecast_competing_outcomes(transition):
    scenarios = {
        "stabilization": {"delta_2100": -0.35, "force": 0.60, "peace": 2.00, "separation": 0.75},
        "baseline": {"delta_2100": 0.45, "force": 1.00, "peace": 1.00, "separation": 1.00},
        "adverse": {"delta_2100": 1.05, "force": 2.20, "peace": 0.45, "separation": 1.45},
    }
    targets = {2030, 2035, 2040, 2050, 2075, 2100}
    output = {}
    for scenario, cfg in scenarios.items():
        live = [0.10, 0.62, 0.28]
        total_live = sum(live)
        live = [value / total_live for value in live]
        terminal = {"force": 0.0, "peace": 0.0, "separation": 0.0}
        snapshots = {}
        for month_index in range(1, (2100 - 2026) * 12 + 1):
            year = 2026 + month_index / 12.0
            progress = min((year - 2026) / 74.0, 1.0)
            delta = cfg["delta_2100"] * progress
            matrix = adjusted_transition(transition, delta)
            force_time = math.exp(0.006 * (year - 2026))
            separation_time = math.exp(0.010 * (year - 2026))
            peace_time = math.exp(-0.008 * (year - 2026))
            force_h = [0.000002, 0.000030, 0.000450]
            peace_h = [0.000300, 0.000100, 0.000020]
            separation_h = [0.000036, 0.000066, 0.000120]
            survivors = [0.0, 0.0, 0.0]
            for state in range(3):
                hazards = [
                    force_h[state] * force_time * cfg["force"],
                    peace_h[state] * peace_time * cfg["peace"],
                    separation_h[state] * separation_time * cfg["separation"],
                ]
                total_hazard = sum(hazards)
                exit_probability = 1.0 - math.exp(-total_hazard)
                for key, hazard in zip(("force", "peace", "separation"), hazards):
                    terminal[key] += live[state] * exit_probability * hazard / total_hazard
                survivors[state] = live[state] * (1.0 - exit_probability)
            live = [sum(survivors[i] * matrix[i][j] for i in range(3)) for j in range(3)]
            rounded_year = int(round(year))
            if abs(year - rounded_year) < 1e-8 and rounded_year in targets:
                status = sum(live)
                snapshots[str(rounded_year)] = {
                    "force": terminal["force"], "peace": terminal["peace"],
                    "separation": terminal["separation"], "status_quo": status,
                    "latent_normal_given_status": live[0] / status,
                    "latent_pressure_given_status": live[1] / status,
                    "latent_crisis_given_status": live[2] / status,
                }
        output[scenario] = snapshots
    return output


def write_outputs(rows, model, response, pressure_dynamics, slow_feedback):
    state_names = ["常态威慑", "持续施压", "急性危机"]
    csv_path = OUT / "台海月度历史数据与隐状态_2020-2026.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["month", "adiz", "shock", "election", "defense_budget_ntd_bn", "defense_growth_pct", "export_share_cn_hk_pct", "latent_state", "state_probability"])
        for row, state, posterior in zip(rows, model["path"], model["posterior"]):
            writer.writerow([row["month"], row["adiz"], row["shock"], row["election"], row["defense_budget"],
                             round(row["defense_growth"], 3), row["export_share_cn_hk"], state_names[state], round(posterior[state], 6)])

    summary = {
        "sample": {"start": rows[0]["month"], "end": rows[-1]["month"], "months": len(rows)},
        "hmm": {
            "state_names": state_names,
            "mean_monthly_adiz": [math.exp(m) - 1 for m in model["means"]],
            "transition_matrix": model["transition"],
            "stationary_distribution": stationary_distribution(model["transition"]),
            "log_likelihood": model["log_likelihood"],
        },
        "discarded_sparse_transition_diagnostic": {
            "warning": "Only two upward and one downward decoded state transitions were available. These logits are retained for audit only and are not used in the forecast.",
            "results": response,
        },
        "continuous_pressure_dynamics": pressure_dynamics,
        "slow_feedback": slow_feedback,
        "conditional_forecast": forecast_competing_outcomes(model["transition"]),
        "identification_warning": "Terminal outcomes are absent from the historical sample; only reversible pressure-state dynamics are estimated from data.",
    }
    json_path = OUT / "台海可逆隐状态模型_估计结果.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    return csv_path, json_path, summary


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    rows = build_rows()
    model = fit_gaussian_hmm([row["log_adiz"] for row in rows])
    response = fit_transition_response(rows, model["path"])
    pressure_dynamics = fit_pressure_dynamics(rows)
    slow_feedback = fit_slow_feedback(rows)
    csv_path, json_path, summary = write_outputs(rows, model, response, pressure_dynamics, slow_feedback)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(csv_path)
    print(json_path)


if __name__ == "__main__":
    main()
