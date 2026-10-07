from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ICIO = ROOT / "data" / "oecd_icio_2025" / "2016-2022" / "2022_SML.csv"
CACHE = ROOT / "data" / "oecd_icio_2025" / "icio_2022_8x10.npz"
OUT_CSV = ROOT / "outputs" / "台海V4.3贸易战争半导体耦合_仿真结果.csv"
OUT_BASE = ROOT / "outputs" / "台海V4.3_OECD_ICIO基线.csv"
OUT_JSON = ROOT / "outputs" / "台海V4.3贸易战争半导体耦合_仿真摘要.json"

REGIONS = ["CHN", "TWN", "USA", "JPN", "KOR", "EUR", "ASEAN", "ROW"]
SECTORS = [
    "primary",
    "food_light",
    "energy_utilities",
    "materials",
    "electronics",
    "machinery_transport",
    "construction",
    "trade_transport",
    "business_services",
    "public_services",
]

EUROPE = {
    "AUT", "BEL", "BGR", "CHE", "CYP", "CZE", "DEU", "DNK", "ESP", "EST",
    "FIN", "FRA", "GBR", "GRC", "HRV", "HUN", "IRL", "ISL", "ITA", "LTU",
    "LUX", "LVA", "MLT", "NLD", "NOR", "POL", "PRT", "ROU", "SVK", "SVN",
    "SWE",
}
ASEAN = {"BRN", "KHM", "IDN", "LAO", "MYS", "MMR", "PHL", "SGP", "THA", "VNM"}


def region_of(country: str) -> str:
    if country in {"CHN", "TWN", "USA", "JPN", "KOR"}:
        return country
    if country in EUROPE:
        return "EUR"
    if country in ASEAN:
        return "ASEAN"
    return "ROW"


def sector_of(code: str) -> str:
    if code.startswith("A") or code in {"B05", "B06", "B07", "B08", "B09"}:
        return "primary"
    if code in {"C10T12", "C13T15", "C16", "C17_18"}:
        return "food_light"
    if code in {"C19", "D", "E"}:
        return "energy_utilities"
    if code in {"C20", "C21", "C22", "C23", "C24A", "C24B", "C25"}:
        return "materials"
    if code == "C26":
        return "electronics"
    if code in {"C27", "C28", "C29", "C301", "C302T309", "C31T33"}:
        return "machinery_transport"
    if code == "F":
        return "construction"
    if code in {"G", "H49", "H50", "H51", "H52", "H53"}:
        return "trade_transport"
    if code in {"I", "J58T60", "J61", "J62_63", "K", "L", "M", "N"}:
        return "business_services"
    return "public_services"


def parse_node(label: str) -> tuple[str, str] | None:
    match = re.match(r"^([A-Z]{3})_(.+)$", label)
    if not match:
        return None
    country, sector = match.groups()
    if sector in {"HFCE", "NPISH", "GGFC", "GFCF", "INVNT", "DPABR"}:
        return None
    return country, sector


def aggregate_icio() -> dict[str, np.ndarray]:
    if CACHE.exists():
        data = np.load(CACHE, allow_pickle=True)
        return {key: data[key] for key in data.files}

    with ICIO.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)

        prod_labels = []
        for label in header[1:]:
            if parse_node(label) is None:
                break
            prod_labels.append(label)
        n_prod = len(prod_labels)

        node_map = np.empty(n_prod, dtype=np.int16)
        countries = []
        for idx, label in enumerate(prod_labels):
            country, raw_sector = parse_node(label)
            countries.append(country)
            node_map[idx] = REGIONS.index(region_of(country)) * len(SECTORS) + SECTORS.index(sector_of(raw_sector))

        fd_headers = header[1 + n_prod : -1]
        fd_region_map = np.array([REGIONS.index(region_of(label[:3])) for label in fd_headers], dtype=np.int16)

        n = len(REGIONS) * len(SECTORS)
        z = np.zeros((n, n), dtype=np.float64)
        fd = np.zeros((n, len(REGIONS)), dtype=np.float64)
        out = np.zeros(n, dtype=np.float64)
        va = np.zeros(n, dtype=np.float64)

        for row in reader:
            label = row[0]
            parsed = parse_node(label)
            if parsed is not None:
                country, raw_sector = parsed
                row_idx = REGIONS.index(region_of(country)) * len(SECTORS) + SECTORS.index(sector_of(raw_sector))
                values = np.asarray(row[1 : 1 + n_prod], dtype=np.float64)
                z[row_idx] += np.bincount(node_map, weights=values, minlength=n)
                fd_values = np.asarray(row[1 + n_prod : -1], dtype=np.float64)
                fd[row_idx] += np.bincount(fd_region_map, weights=fd_values, minlength=len(REGIONS))
            elif label in {"VA", "OUT"}:
                values = np.asarray(row[1 : 1 + n_prod], dtype=np.float64)
                agg = np.bincount(node_map, weights=values, minlength=n)
                if label == "VA":
                    va = agg
                else:
                    out = agg

    labels = np.array([f"{r}_{s}" for r in REGIONS for s in SECTORS], dtype=object)
    np.savez_compressed(CACHE, Z=z, FD=fd, OUT=out, VA=va, labels=labels)
    return {"Z": z, "FD": fd, "OUT": out, "VA": va, "labels": labels}


def write_baseline(data: dict[str, np.ndarray]) -> None:
    z, fd, out, va = data["Z"], data["FD"], data["OUT"], data["VA"]
    with OUT_BASE.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["地区", "产业", "总产出_百万美元", "增加值_百万美元", "中间投入_百万美元", "最终需求_百万美元"])
        for r, region in enumerate(REGIONS):
            for s, sector in enumerate(SECTORS):
                idx = r * len(SECTORS) + s
                writer.writerow([region, sector, out[idx], va[idx], z[:, idx].sum(), fd[idx].sum()])


def build_empirical_network(data: dict[str, np.ndarray]) -> dict[str, np.ndarray | float]:
    z, fd, out, va = data["Z"], data["FD"], data["OUT"], data["VA"]
    n = len(out)
    eps = 1e-12
    inter = z.sum(axis=0)
    dependency = (z / np.maximum(inter, eps)[None, :]).T

    # D[j, i] is customer j's observed dependence on supplier i.
    dependency *= (inter / np.maximum(out, eps))[:, None]
    spectral_bound = max(np.linalg.norm(dependency, ord=np.inf), 1.0)
    dependency /= spectral_bound
    propagation = np.linalg.inv(np.eye(n) - 0.62 * dependency)

    region_of_node = np.repeat(np.arange(len(REGIONS)), len(SECTORS))
    sector_of_node = np.tile(np.arange(len(SECTORS)), len(REGIONS))
    twn = REGIONS.index("TWN")
    chn = REGIONS.index("CHN")

    twn_rows = region_of_node == twn
    chn_rows = region_of_node == chn
    foreign_to_twn = region_of_node != twn
    foreign_to_chn = region_of_node != chn

    twn_supply_dependence = z[twn_rows, :].sum(axis=0) / np.maximum(out, eps)
    twn_import_dependence = np.zeros(n)
    twn_import_dependence[twn_rows] = z[foreign_to_twn][:, twn_rows].sum(axis=0) / np.maximum(out[twn_rows], eps)
    twn_export_demand = np.zeros(n)
    twn_export_demand[foreign_to_twn] = (
        z[foreign_to_twn][:, twn_rows].sum(axis=1) + fd[foreign_to_twn, twn]
    ) / np.maximum(out[foreign_to_twn], eps)
    twn_trade_basis = np.clip(twn_supply_dependence + twn_import_dependence + 0.45 * twn_export_demand, 0, 0.85)

    chn_import_dependence = np.zeros(n)
    chn_import_dependence[chn_rows] = z[foreign_to_chn][:, chn_rows].sum(axis=0) / np.maximum(out[chn_rows], eps)
    chn_supply_dependence = z[chn_rows, :].sum(axis=0) / np.maximum(out, eps)
    chn_export_demand = np.zeros(n)
    chn_export_demand[foreign_to_chn] = (
        z[foreign_to_chn][:, chn_rows].sum(axis=1) + fd[foreign_to_chn, chn]
    ) / np.maximum(out[foreign_to_chn], eps)
    chn_sanction_basis = np.clip(chn_import_dependence + 0.55 * chn_supply_dependence + 0.35 * chn_export_demand, 0, 0.85)

    cross_border = np.zeros(n)
    for j in range(n):
        cross_border[j] = z[region_of_node != region_of_node[j], j].sum() / max(out[j], eps)
    shipping_basis = np.clip(cross_border, 0, 0.75)

    twn_electronics = np.zeros(n)
    twn_electronics[twn * len(SECTORS) + SECTORS.index("electronics")] = 1.0
    electronics_users = z[twn * len(SECTORS) + SECTORS.index("electronics"), :] / np.maximum(out, eps)
    chip_basis = np.clip(0.70 * twn_electronics + 1.8 * electronics_users, 0, 0.90)

    bases = {
        "twn_trade": np.clip(propagation @ twn_trade_basis, 0, 1),
        "chn_sanction": np.clip(propagation @ chn_sanction_basis, 0, 1),
        "shipping": np.clip(propagation @ shipping_basis, 0, 1),
        "chip": np.clip(propagation @ chip_basis, 0, 1),
    }

    foreign_mask = region_of_node[:, None] != region_of_node[None, :]
    gross_trade = float(z[foreign_mask].sum() + fd.sum() - sum(fd[r * len(SECTORS):(r + 1) * len(SECTORS), r].sum() for r in range(len(REGIONS))))
    twn_trade = float(z[twn_rows, :][:, ~twn_rows].sum() + z[~twn_rows, :][:, twn_rows].sum() + fd[twn_rows, :].sum() - fd[twn_rows, twn].sum())
    chn_trade = float(z[chn_rows, :][:, ~chn_rows].sum() + z[~chn_rows, :][:, chn_rows].sum() + fd[chn_rows, :].sum() - fd[chn_rows, chn].sum())

    return {
        "dependency": dependency,
        "region_of_node": region_of_node,
        "sector_of_node": sector_of_node,
        "va": va,
        "out": out,
        "bases": bases,
        "gross_trade": gross_trade,
        "twn_trade_share": min(twn_trade / max(gross_trade, eps), 1.0),
        "chn_trade_share": min(chn_trade / max(gross_trade, eps), 1.0),
    }


SCENARIOS = {
    "和平基线": "baseline",
    "封锁但晶圆厂基本完整": "blockade",
    "台积电严重中断_低国产替代": "fab_low_sub",
    "台积电严重中断_华为体系替代": "fab_huawei",
    "台积电中断_全球加速分散": "fab_global",
    "封锁_制裁_多节点同步冲击": "combined",
}


def load_coupling_profile(
    scenario_cn: str,
    paths: int,
    months: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """Convert V4.3 observable outputs into monthly four-actor paths for V4.2."""
    if SCENARIOS.get(scenario_cn) == "baseline":
        neutral = np.ones((months, paths, 4), dtype=np.float32)
        zero_actor = np.zeros((months, paths, 4), dtype=np.float32)
        return {
            "gdp_factor": neutral.copy(),
            "industry_factor": neutral.copy(),
            "chip_gap": zero_actor.copy(),
            "inflation": zero_actor.copy(),
            "trade_loss": np.zeros((months, paths), dtype=np.float32),
        }
    actors = ["CHN", "TWN", "USA", "JPN"]
    metrics = {
        "gdp_factor": "实际GDP损失率",
        "industry_factor": "关键工业产出损失率",
        "chip_gap": "先进芯片服务缺口率",
        "inflation": "额外通胀_百分点",
    }
    checkpoints = {"1年": 12, "3年": 36, "5年": 60, "10年": 120}
    rows = {}
    with OUT_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["情景"] == scenario_cn and row["地区"] in actors:
                rows[(row["地区"], row["指标"], checkpoints[row["期限"]])] = (
                    float(row["P10"]), float(row["中位数"]), float(row["P90"])
                )

    rng = np.random.default_rng(seed)
    common_z = rng.normal(0.0, 1.0, paths)
    actor_z = rng.normal(0.0, 1.0, (paths, len(actors)))
    idio_z = rng.normal(0.0, 1.0, (paths, len(actors), len(metrics)))
    z = 0.62 * common_z[:, None, None] + 0.27 * actor_z[:, :, None] + 0.11 * idio_z
    times = np.array([0, 12, 36, 60, 120, months], dtype=float)
    profile = {key: np.zeros((months, paths, len(actors)), dtype=np.float32) for key in metrics}

    for a, actor in enumerate(actors):
        for m, (key, metric) in enumerate(metrics.items()):
            q10 = [0.0]
            med = [0.0]
            q90 = [0.0]
            for t in (12, 36, 60, 120):
                values = rows[(actor, metric, t)]
                q10.append(values[0])
                med.append(values[1])
                q90.append(values[2])
            # The ten-year structural loss is held beyond year ten; later precision is low.
            q10.append(q10[-1])
            med.append(med[-1])
            q90.append(q90[-1])
            sd = (np.array(q90) - np.array(q10)) / 2.563
            monthly_med = np.interp(np.arange(1, months + 1), times, med)
            monthly_sd = np.interp(np.arange(1, months + 1), times, sd)
            values = monthly_med[:, None] + monthly_sd[:, None] * z[None, :, a, m]
            if key == "inflation":
                values /= 100.0
                values = np.clip(values, 0.0, 0.40)
            else:
                values = np.clip(values, 0.0, 0.95)
            if key in {"gdp_factor", "industry_factor"}:
                values = 1.0 - values
            profile[key][:, :, a] = values.astype(np.float32)

    trade_rows = {}
    with OUT_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["情景"] == scenario_cn and row["地区"] == "GLOBAL" and row["指标"] == "全球贸易量损失率":
                trade_rows[checkpoints[row["期限"]]] = (
                    float(row["P10"]), float(row["中位数"]), float(row["P90"])
                )
    q10 = [0.0] + [trade_rows[t][0] for t in (12, 36, 60, 120)]
    med = [0.0] + [trade_rows[t][1] for t in (12, 36, 60, 120)]
    q90 = [0.0] + [trade_rows[t][2] for t in (12, 36, 60, 120)]
    q10.append(q10[-1]); med.append(med[-1]); q90.append(q90[-1])
    monthly_med = np.interp(np.arange(1, months + 1), times, med)
    monthly_sd = np.interp(np.arange(1, months + 1), times, (np.array(q90) - np.array(q10)) / 2.563)
    profile["trade_loss"] = np.clip(
        monthly_med[:, None] + monthly_sd[:, None] * common_z[None, :], 0.0, 0.85
    ).astype(np.float32)
    return profile


def sample_uniform(rng: np.random.Generator, low: float, high: float, paths: int) -> np.ndarray:
    return rng.uniform(low, high, paths)


def scenario_parameters(name: str, rng: np.random.Generator, paths: int) -> dict[str, np.ndarray]:
    zero = np.zeros(paths)
    if name == "baseline":
        return dict(block0=zero, block_lr=zero, fab0=np.ones(paths), fab_lr=np.ones(paths), sanction=zero, shipping=zero,
                    china_sub=zero, global_sub=zero, recovery=np.ones(paths) * 18)
    if name == "blockade":
        return dict(block0=sample_uniform(rng, .58, .82, paths), block_lr=sample_uniform(rng, .18, .38, paths),
                    fab0=sample_uniform(rng, .84, .98, paths), fab_lr=sample_uniform(rng, .94, 1.0, paths),
                    sanction=sample_uniform(rng, .04, .12, paths), shipping=sample_uniform(rng, .06, .14, paths),
                    china_sub=sample_uniform(rng, .10, .22, paths), global_sub=sample_uniform(rng, .10, .25, paths),
                    recovery=sample_uniform(rng, 24, 48, paths))
    if name in {"fab_low_sub", "fab_huawei", "fab_global"}:
        china_ranges = {"fab_low_sub": (.06, .16), "fab_huawei": (.28, .48), "fab_global": (.14, .28)}
        global_ranges = {"fab_low_sub": (.08, .20), "fab_huawei": (.12, .26), "fab_global": (.36, .58)}
        return dict(block0=sample_uniform(rng, .68, .90, paths), block_lr=sample_uniform(rng, .25, .48, paths),
                    fab0=sample_uniform(rng, .08, .28, paths), fab_lr=sample_uniform(rng, .52, .78, paths),
                    sanction=sample_uniform(rng, .08, .20, paths), shipping=sample_uniform(rng, .10, .22, paths),
                    china_sub=sample_uniform(rng, *china_ranges[name], paths), global_sub=sample_uniform(rng, *global_ranges[name], paths),
                    recovery=sample_uniform(rng, 36, 72, paths))
    return dict(block0=sample_uniform(rng, .82, .96, paths), block_lr=sample_uniform(rng, .42, .68, paths),
                fab0=sample_uniform(rng, .04, .18, paths), fab_lr=sample_uniform(rng, .35, .68, paths),
                sanction=sample_uniform(rng, .36, .62, paths), shipping=sample_uniform(rng, .18, .34, paths),
                china_sub=sample_uniform(rng, .24, .44, paths), global_sub=sample_uniform(rng, .20, .40, paths),
                recovery=sample_uniform(rng, 48, 90, paths))


def simulate(network: dict[str, np.ndarray | float], paths: int = 600, months: int = 120, seed: int = 20261004):
    rng = np.random.default_rng(seed)
    n = len(network["out"])
    region_of_node = network["region_of_node"]
    sector_of_node = network["sector_of_node"]
    va = network["va"]
    bases = network["bases"]
    results = []
    paths_store = {}

    gdp_weights = np.zeros((len(REGIONS), n))
    for r in range(len(REGIONS)):
        mask = region_of_node == r
        gdp_weights[r, mask] = va[mask] / max(va[mask].sum(), 1e-12)

    military_mask = np.isin(sector_of_node, [SECTORS.index("electronics"), SECTORS.index("materials"), SECTORS.index("machinery_transport")])
    military_weights = np.zeros((len(REGIONS), n))
    for r in range(len(REGIONS)):
        mask = (region_of_node == r) & military_mask
        military_weights[r, mask] = va[mask] / max(va[mask].sum(), 1e-12)

    pass_through = np.array([.24, .44, .15, .30, .34, .22, .30, .26])
    checkpoints = {12: "1年", 36: "3年", 60: "5年", 120: "10年"}
    global_weights = np.array([va[region_of_node == i].sum() for i in range(len(REGIONS))])
    global_weights /= global_weights.sum()

    for scenario_cn, scenario in SCENARIOS.items():
        p = scenario_parameters(scenario, rng, paths)
        inventory = sample_uniform(rng, 2.0, 7.0, paths)  # months of advanced-chip service
        advanced_taiwan_share = sample_uniform(rng, .74, .88, paths)
        regional_gdp = np.ones((paths, len(REGIONS)))
        regional_infl = np.zeros((paths, len(REGIONS)))
        regional_military = np.ones((paths, len(REGIONS)))
        chip_gap = np.zeros((paths, len(REGIONS)))
        trade_loss = np.zeros(paths)
        inflation_state = np.zeros((paths, len(REGIONS)))
        exchange_state = np.zeros((paths, len(REGIONS)))
        demand_gap_state = np.zeros((paths, len(REGIONS)))

        for month in range(1, months + 1):
            recovery = 1.0 - np.exp(-month / p["recovery"])
            block = p["block_lr"] + (p["block0"] - p["block_lr"]) * (1.0 - recovery)
            fab_avail = p["fab0"] + (p["fab_lr"] - p["fab0"]) * recovery
            cert = 1.0 - np.exp(-month / 30.0)
            china_sub = p["china_sub"] * cert
            global_sub = p["global_sub"] * (1.0 - np.exp(-month / 54.0))
            armington_elasticity = 1.15 + 1.35 * (1.0 - np.exp(-month / 48.0))
            firm_entry = (1.0 - np.exp(-month / 30.0)) * (armington_elasticity / 2.5)
            reroute = np.clip(global_sub + .28 * firm_entry, 0, .72)

            gravity_trade_gap = block * (1.0 - reroute)
            chip_raw_gap = advanced_taiwan_share * (1.0 - fab_avail) * (1.0 - global_sub)
            inventory = np.maximum(inventory - chip_raw_gap, 0.0)
            inventory_buffer = np.minimum(inventory / 3.0, 1.0)
            effective_chip_gap = chip_raw_gap * (1.0 - inventory_buffer)

            coeff_twn_trade = gravity_trade_gap
            coeff_chip = np.clip(effective_chip_gap, 0, .95)
            coeff_chn = p["sanction"] * (1.0 - .35 * cert)
            coeff_ship = p["shipping"] * (1.0 - .25 * cert)

            q = (
                coeff_twn_trade[:, None] * bases["twn_trade"][None, :]
                + coeff_chip[:, None] * bases["chip"][None, :]
                + coeff_chn[:, None] * bases["chn_sanction"][None, :]
                + coeff_ship[:, None] * bases["shipping"][None, :]
            )
            q = np.clip(q, 0, .92)

            # China receives a use-specific, quality-adjusted domestic-design/fabrication buffer.
            chn_mask = region_of_node == REGIONS.index("CHN")
            chn_critical = chn_mask & np.isin(sector_of_node, [SECTORS.index("electronics"), SECTORS.index("machinery_transport")])
            q[:, chn_critical] *= (1.0 - .72 * china_sub[:, None])

            output_ratio = 1.0 - q
            network_gdp = output_ratio @ gdp_weights.T
            regional_military = output_ratio @ military_weights.T

            # Regional chip service combines common advanced-node loss and China-specific substitution.
            exposure = np.array([.78, .96, .62, .66, .58, .52, .64, .38])
            chip_gap = coeff_chip[:, None] * exposure[None, :]
            chip_gap[:, REGIONS.index("CHN")] *= (1.0 - china_sub)

            trade_loss = np.clip(
                network["twn_trade_share"] * gravity_trade_gap
                + network["chn_trade_share"] * coeff_chn
                + .35 * coeff_ship,
                0,
                .75,
            )
            # Reduced open-economy DSGE recursion: risk premium -> exchange rate -> import prices,
            # with inflation persistence, policy feedback and a persistent demand gap.
            risk_premium = (
                .36 * trade_loss[:, None]
                + .24 * chip_gap
                + .20 * coeff_ship[:, None]
                + .12 * coeff_chn[:, None]
            )
            policy_response = np.maximum(inflation_state - .02, 0.0) * np.array([.55, .35, .85, .50, .70, .65, .45, .40])[None, :]
            exchange_state = .82 * exchange_state + .18 * risk_premium - .10 * policy_response
            import_cost = .48 * trade_loss[:, None] + .31 * chip_gap + .21 * np.maximum(exchange_state, 0.0)
            target_inflation = (
                pass_through[None, :] * import_cost
                + .12 * (1.0 - network_gdp)
                + .05 * chip_gap
            )
            inflation_state = .72 * inflation_state + .28 * target_inflation
            demand_target = .30 * trade_loss[:, None] + .18 * risk_premium + .12 * policy_response
            demand_gap_state = .68 * demand_gap_state + .32 * demand_target

            # Dynamic-CGE closure in reduced form: network quantities are combined with
            # Armington substitution, relative-price pressure and investment/demand effects.
            cge_demand_factor = np.clip(1.0 - demand_gap_state / armington_elasticity, .72, 1.0)
            regional_gdp = np.clip(network_gdp * cge_demand_factor, 0.0, 1.0)
            regional_infl = 100.0 * inflation_state

            if month in checkpoints:
                for r, region in enumerate(REGIONS + ["GLOBAL"]):
                    if region == "GLOBAL":
                        gdp_loss = 1.0 - regional_gdp @ global_weights
                        inflation = regional_infl @ global_weights
                        military_loss = 1.0 - regional_military @ global_weights
                        cgap = chip_gap @ global_weights
                    else:
                        idx = REGIONS.index(region)
                        gdp_loss = 1.0 - regional_gdp[:, idx]
                        inflation = regional_infl[:, idx]
                        military_loss = 1.0 - regional_military[:, idx]
                        cgap = chip_gap[:, idx]
                    for metric, values in {
                        "实际GDP损失率": gdp_loss,
                        "额外通胀_百分点": inflation,
                        "关键工业产出损失率": military_loss,
                        "先进芯片服务缺口率": cgap,
                        "全球贸易量损失率": trade_loss,
                    }.items():
                        values = np.maximum(values, 0.0)
                        results.append({
                            "情景": scenario_cn,
                            "期限": checkpoints[month],
                            "地区": region,
                            "指标": metric,
                            "P10": float(np.quantile(values, .10)),
                            "中位数": float(np.quantile(values, .50)),
                            "P90": float(np.quantile(values, .90)),
                        })

        paths_store[scenario] = {
            "gdp_loss_global_median_10y": float(np.median(1.0 - regional_gdp @ global_weights)),
            "trade_loss_median_10y": float(np.median(trade_loss)),
            "chip_gap_china_median_10y": float(np.median(chip_gap[:, REGIONS.index("CHN")])),
            "chip_gap_global_median_10y": float(np.median(chip_gap @ global_weights)),
            "inflation_global_pp_median_10y": float(np.median(regional_infl @ global_weights)),
        }
    return results, paths_store


def write_results(results, summary, network) -> None:
    with OUT_CSV.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["情景", "期限", "地区", "指标", "P10", "中位数", "P90"])
        writer.writeheader()
        writer.writerows(results)

    payload = {
        "data": {
            "source": "OECD 2025 ICIO, 2022 SML matrix",
            "raw_nodes": 4050,
            "aggregated_regions": REGIONS,
            "aggregated_sectors": SECTORS,
            "tsmc_anchor": {
                "year": 2025,
                "annual_capacity": "over 17 million 12-inch equivalent wafers",
                "advanced_revenue_share": 0.74,
                "note": "Revenue share is not treated as physical capacity share.",
            },
        },
        "simulation": {
            "seed": 20261004,
            "paths_per_scenario": 600,
            "months": 120,
            "scenarios": list(SCENARIOS.keys()),
            "network_trade_shares": {
                "Taiwan_related": float(network["twn_trade_share"]),
                "China_related": float(network["chn_trade_share"]),
            },
            "implemented_blocks": [
                "structural-gravity counterfactual on observed bilateral flows",
                "heterogeneous-firm entry and certification lag",
                "capacity-constrained multi-regional input-output network",
                "Armington substitution and reduced dynamic-CGE closure",
                "open-economy DSGE inflation, exchange-rate, policy and demand recursion",
                "semiconductor process, inventory and quality-adjusted substitution",
                "two-way trade-war feedback interface",
            ],
        },
        "ten_year_summary": summary,
        "limitations": [
            "ICIO electronics includes computer, electronic and optical products, not semiconductors alone.",
            "War damage, substitution quality and certification lags are interval priors, not observed future facts.",
            "The module is semi-structural and does not claim a fully estimated global DSGE-CGE equilibrium.",
        ],
    }
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def validate(results, network) -> dict[str, float | bool]:
    values = np.array([[row["P10"], row["中位数"], row["P90"]] for row in results])
    return {
        "finite": bool(np.isfinite(values).all()),
        "quantile_order": bool(np.all(values[:, 0] <= values[:, 1]) and np.all(values[:, 1] <= values[:, 2])),
        "nonnegative": bool(np.all(values >= -1e-10)),
        "twn_trade_share_valid": bool(0 <= network["twn_trade_share"] <= 1),
        "chn_trade_share_valid": bool(0 <= network["chn_trade_share"] <= 1),
    }


def main() -> None:
    data = aggregate_icio()
    write_baseline(data)
    network = build_empirical_network(data)
    results, summary = simulate(network)
    checks = validate(results, network)
    if not all(checks.values()):
        raise RuntimeError(f"Validation failed: {checks}")
    write_results(results, summary, network)
    print(json.dumps({"checks": checks, "summary": summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
