import math
import random
from statistics import median


SEED = 20260930
N_DRAWS = 30000
TARGET_YEARS = (2030, 2035, 2040, 2045, 2050, 2060, 2070, 2080, 2090, 2100)


def percentile(values, q):
    values = sorted(values)
    pos = (len(values) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return values[lo]
    return values[lo] * (hi - pos) + values[hi] * (pos - lo)


def hazards(year, m_force, m_peace, m_separation):
    # These are transparent reference-scenario priors, not empirical frequencies.
    t = year - 2027
    election_cycle = 0.0025 if year in (2028, 2032, 2036, 2040, 2044, 2048) else 0.0
    force_total = (0.0075 + 0.00065 * min(t, 18) + election_cycle) * m_force
    invasion_share = min(0.28 + 0.008 * t, 0.46)
    force_full = force_total * invasion_share
    force_limited = force_total - force_full

    # Identity divergence lowers the baseline for a negotiated political settlement,
    # while allowing a small late-horizon reopening under leadership/policy change.
    peace = (0.0014 + 0.00005 * max(t - 8, 0)) * m_peace

    # This endpoint means a formal/legal break or broadly consolidated permanent
    # separation, not the already-existing de facto self-government.
    separation = (0.0018 + 0.00022 * min(t, 18)) * m_separation
    return force_limited, force_full, peace, separation


def simulate_draw(rng):
    # Lognormal multipliers encode deep model uncertainty and preserve positivity.
    multipliers = (
        rng.lognormvariate(0.0, 0.42),
        rng.lognormvariate(0.0, 0.60),
        rng.lognormvariate(0.0, 0.52),
    )
    state = {"limited": 0.0, "full": 0.0, "peace": 0.0, "separation": 0.0, "status_quo": 1.0}
    snapshots = {}
    for year in range(2027, 2101):
        hs = hazards(year, *multipliers)
        total_h = sum(hs)
        exit_probability = 1.0 - math.exp(-total_h)
        old_status = state["status_quo"]
        if total_h > 0:
            allocations = [old_status * exit_probability * h / total_h for h in hs]
            for key, amount in zip(("limited", "full", "peace", "separation"), allocations):
                state[key] += amount
        state["status_quo"] = old_status * (1.0 - exit_probability)
        if year in TARGET_YEARS:
            snapshots[year] = dict(state)
    return snapshots


def reference_snapshots():
    state = {"limited": 0.0, "full": 0.0, "peace": 0.0, "separation": 0.0, "status_quo": 1.0}
    snapshots = {}
    for year in range(2027, 2101):
        hs = hazards(year, 1.0, 1.0, 1.0)
        total_h = sum(hs)
        exit_probability = 1.0 - math.exp(-total_h)
        old_status = state["status_quo"]
        allocations = [old_status * exit_probability * h / total_h for h in hs]
        for key, amount in zip(("limited", "full", "peace", "separation"), allocations):
            state[key] += amount
        state["status_quo"] = old_status * (1.0 - exit_probability)
        if year in TARGET_YEARS:
            snapshots[year] = dict(state)
            snapshots[year]["force"] = state["limited"] + state["full"]
    return snapshots


def main():
    rng = random.Random(SEED)
    values = {year: {key: [] for key in ("force", "limited", "full", "peace", "separation", "status_quo")} for year in TARGET_YEARS}
    for _ in range(N_DRAWS):
        snapshots = simulate_draw(rng)
        for year, state in snapshots.items():
            state["force"] = state["limited"] + state["full"]
            for key in values[year]:
                values[year][key].append(state[key])

    reference = reference_snapshots()
    print("reference_year,state,percent")
    for year in TARGET_YEARS:
        for key in values[year]:
            print(f"{year},{key},{100*reference[year][key]:.4f}")

    print("uncertainty_year,state,median,p10,p90")
    for year in TARGET_YEARS:
        for key in values[year]:
            series = values[year][key]
            print(f"{year},{key},{100*median(series):.1f},{100*percentile(series,0.10):.1f},{100*percentile(series,0.90):.1f}")


if __name__ == "__main__":
    main()
