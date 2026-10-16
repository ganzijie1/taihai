from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import openpyxl

import simulate_trade_war_v43 as v43


ROOT = Path(__file__).resolve().parents[1]
MRIO_XLSX = ROOT / "data" / "china_mrio" / "MRIO2017_42+CEADS.xlsx"
DOMESTIC_CACHE = ROOT / "data" / "china_mrio" / "mrio2017_7x10.npz"
NESTED_CACHE = ROOT / "data" / "china_mrio" / "dual_circulation_2022_140nodes.npz"
OUT_CSV = ROOT / "outputs" / "台海V4.4双循环贸易战争_仿真结果.csv"
OUT_BASE = ROOT / "outputs" / "台海V4.4双循环140节点基线.csv"
OUT_JSON = ROOT / "outputs" / "台海V4.4双循环贸易战争_仿真摘要.json"

CHINA_ZONES = [
    "CHN_NORTH",
    "CHN_NORTHEAST",
    "CHN_EAST_COAST",
    "CHN_SOUTHEAST_COAST",
    "CHN_CENTRAL",
    "CHN_SOUTHWEST",
    "CHN_NORTHWEST",
]
EXTERNAL_REGIONS = ["TWN", "USA", "JPN", "KOR", "EUR", "ASEAN", "ROW"]
REGIONS = CHINA_ZONES + EXTERNAL_REGIONS
ACTORS = v43.REGIONS
SECTORS = v43.SECTORS

PROVINCES = [
    "Beijing", "Tianjin", "Hebei", "Shanxi", "Inner Mongolia", "Liaoning", "Jilin", "Heilongjiang",
    "Shanghai", "Jiangsu", "Zhejiang", "Anhui", "Fujian", "Jiangxi", "Shandong", "Henan", "Hubei",
    "Hunan", "Guangdong", "Guangxi", "Hainan", "Chongqing", "Sichuan", "Guizhou", "Yunnan", "Tibet",
    "Shannxi", "Gansu", "Qinghai", "Ningxia", "Xinjiang",
]

ZONE_BY_PROVINCE = {
    "Beijing": "CHN_NORTH", "Tianjin": "CHN_NORTH", "Hebei": "CHN_NORTH", "Shandong": "CHN_NORTH",
    "Liaoning": "CHN_NORTHEAST", "Jilin": "CHN_NORTHEAST", "Heilongjiang": "CHN_NORTHEAST",
    "Shanghai": "CHN_EAST_COAST", "Jiangsu": "CHN_EAST_COAST", "Zhejiang": "CHN_EAST_COAST",
    "Fujian": "CHN_SOUTHEAST_COAST", "Guangdong": "CHN_SOUTHEAST_COAST", "Hainan": "CHN_SOUTHEAST_COAST",
    "Shanxi": "CHN_CENTRAL", "Anhui": "CHN_CENTRAL", "Jiangxi": "CHN_CENTRAL", "Henan": "CHN_CENTRAL",
    "Hubei": "CHN_CENTRAL", "Hunan": "CHN_CENTRAL",
    "Guangxi": "CHN_SOUTHWEST", "Chongqing": "CHN_SOUTHWEST", "Sichuan": "CHN_SOUTHWEST",
    "Guizhou": "CHN_SOUTHWEST", "Yunnan": "CHN_SOUTHWEST", "Tibet": "CHN_SOUTHWEST",
    "Inner Mongolia": "CHN_NORTHWEST", "Shannxi": "CHN_NORTHWEST", "Gansu": "CHN_NORTHWEST",
    "Qinghai": "CHN_NORTHWEST", "Ningxia": "CHN_NORTHWEST", "Xinjiang": "CHN_NORTHWEST",
}


def domestic_sector(raw_index: int) -> int:
    code = raw_index + 1
    if code <= 5:
        name = "primary"
    elif code <= 10 or code == 22:
        name = "food_light"
    elif code in {11, 24, 25, 26}:
        name = "energy_utilities"
    elif code in {12, 13, 14, 15}:
        name = "materials"
    elif code in {20, 21}:
        name = "electronics"
    elif code in {16, 17, 18, 19, 23}:
        name = "machinery_transport"
    elif code == 27:
        name = "construction"
    elif code in {28, 29, 30}:
        name = "trade_transport"
    elif code in {31, 32, 33, 34, 35, 36}:
        name = "business_services"
    else:
        name = "public_services"
    return SECTORS.index(name)


def aggregate_domestic_mrio() -> dict[str, np.ndarray]:
    if DOMESTIC_CACHE.exists():
        cached = np.load(DOMESTIC_CACHE, allow_pickle=True)
        return {key: cached[key] for key in cached.files}

    workbook = openpyxl.load_workbook(MRIO_XLSX, read_only=True, data_only=True)
    sheet = workbook["English_Version"]
    n_raw = len(PROVINCES) * 42
    n_domestic = len(CHINA_ZONES) * len(SECTORS)
    raw_to_agg = np.empty(n_raw, dtype=np.int16)
    province_to_zone = np.empty(len(PROVINCES), dtype=np.int16)
    for p, province in enumerate(PROVINCES):
        zone = CHINA_ZONES.index(ZONE_BY_PROVINCE[province])
        province_to_zone[p] = zone
        for s in range(42):
            raw_to_agg[p * 42 + s] = zone * len(SECTORS) + domestic_sector(s)

    z = np.zeros((n_domestic, n_domestic), dtype=np.float64)
    fd = np.zeros((n_domestic, len(CHINA_ZONES)), dtype=np.float64)
    exports = np.zeros(n_domestic, dtype=np.float64)
    out = np.zeros(n_domestic, dtype=np.float64)
    imports = np.zeros(n_domestic, dtype=np.float64)
    va = np.zeros(n_domestic, dtype=np.float64)
    final_col_map = np.repeat(province_to_zone, 5)

    rows = sheet.iter_rows(min_row=5, max_row=1313, min_col=1, max_col=1464, values_only=True)
    for excel_row, row in enumerate(rows, start=5):
        if excel_row <= 1306:
            raw_row = excel_row - 5
            agg_row = raw_to_agg[raw_row]
            intermediate = np.nan_to_num(np.asarray(row[3:1305], dtype=np.float64))
            z[agg_row] += np.bincount(raw_to_agg, weights=intermediate, minlength=n_domestic)
            final_use = np.nan_to_num(np.asarray(row[1305:1460], dtype=np.float64))
            fd[agg_row] += np.bincount(final_col_map, weights=final_use, minlength=len(CHINA_ZONES))
            exports[agg_row] += float(row[1460] or 0.0)
            out[agg_row] += float(row[1463] or 0.0)
        elif excel_row == 1307:
            values = np.nan_to_num(np.asarray(row[3:1305], dtype=np.float64))
            imports += np.bincount(raw_to_agg, weights=values, minlength=n_domestic)
        elif excel_row == 1313:
            values = np.nan_to_num(np.asarray(row[3:1305], dtype=np.float64))
            va += np.bincount(raw_to_agg, weights=values, minlength=n_domestic)
    workbook.close()

    labels = np.array([f"{r}_{s}" for r in CHINA_ZONES for s in SECTORS], dtype=object)
    np.savez_compressed(
        DOMESTIC_CACHE,
        Z=z,
        FD=fd,
        EXPORT=exports,
        IMPORT=imports,
        OUT=out,
        VA=va,
        labels=labels,
    )
    return {"Z": z, "FD": fd, "EXPORT": exports, "IMPORT": imports, "OUT": out, "VA": va, "labels": labels}


def normalized(values: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    values = np.maximum(np.asarray(values, dtype=np.float64), 0.0)
    if values.sum() > 1e-12:
        return values / values.sum()
    if fallback is not None and np.maximum(fallback, 0.0).sum() > 1e-12:
        fallback = np.maximum(fallback, 0.0)
        return fallback / fallback.sum()
    return np.full(len(values), 1.0 / len(values))


def build_nested_accounts() -> dict[str, np.ndarray]:
    if NESTED_CACHE.exists():
        cached = np.load(NESTED_CACHE, allow_pickle=True)
        return {key: cached[key] for key in cached.files}

    global_data = v43.aggregate_icio()
    domestic = aggregate_domestic_mrio()
    old_z, old_fd, old_out, old_va = global_data["Z"], global_data["FD"], global_data["OUT"], global_data["VA"]
    nr, ns = len(REGIONS), len(SECTORS)
    n = nr * ns
    z = np.zeros((n, n), dtype=np.float64)
    fd = np.zeros((n, nr), dtype=np.float64)
    out = np.zeros(n, dtype=np.float64)
    va = np.zeros(n, dtype=np.float64)
    chn_old = v43.REGIONS.index("CHN")

    domestic_nodes = np.arange(len(CHINA_ZONES) * ns)
    domestic_zone = domestic_nodes // ns
    domestic_sector_idx = domestic_nodes % ns
    prod_share = np.zeros((len(CHINA_ZONES), ns))
    export_share = np.zeros_like(prod_share)
    absorption_share = np.zeros_like(prod_share)
    final_region_share = normalized(domestic["FD"].sum(axis=0))
    for s in range(ns):
        idx = np.arange(s, len(domestic_nodes), ns)
        prod_share[:, s] = normalized(domestic["OUT"][idx])
        export_share[:, s] = normalized(domestic["EXPORT"][idx], domestic["OUT"][idx])
        absorption = domestic["Z"][:, idx].sum(axis=0) + domestic["IMPORT"][idx]
        absorption_share[:, s] = normalized(absorption, domestic["OUT"][idx])
        target_out = old_out[chn_old * ns + s]
        target_va = old_va[chn_old * ns + s]
        out[idx] = target_out * prod_share[:, s]
        va[idx] = target_va * normalized(domestic["VA"][idx], domestic["OUT"][idx])

    # Preserve every China sector-to-sector ICIO cell while retaining the MRIO regional pattern.
    for supplier_s in range(ns):
        supply_idx = np.arange(supplier_s, len(domestic_nodes), ns)
        for customer_s in range(ns):
            use_idx = np.arange(customer_s, len(domestic_nodes), ns)
            target = old_z[chn_old * ns + supplier_s, chn_old * ns + customer_s]
            block = domestic["Z"][np.ix_(supply_idx, use_idx)]
            if block.sum() > 1e-12:
                allocation = block / block.sum()
            else:
                allocation = np.outer(prod_share[:, supplier_s], absorption_share[:, customer_s])
            z[np.ix_(supply_idx, use_idx)] = target * allocation

    # Domestic final demand keeps province-zone patterns, rescaled to the ICIO China total.
    for supplier_s in range(ns):
        supply_idx = np.arange(supplier_s, len(domestic_nodes), ns)
        target = old_fd[chn_old * ns + supplier_s, chn_old]
        block = domestic["FD"][supply_idx, :]
        allocation = block / block.sum() if block.sum() > 1e-12 else np.outer(prod_share[:, supplier_s], final_region_share)
        fd[np.ix_(supply_idx, np.arange(len(CHINA_ZONES)))] = target * allocation

    # Copy external accounts and split every China interface using empirical domestic shares.
    for old_r in range(1, len(v43.REGIONS)):
        new_r = len(CHINA_ZONES) + old_r - 1
        for s in range(ns):
            old_i = old_r * ns + s
            new_i = new_r * ns + s
            out[new_i] = old_out[old_i]
            va[new_i] = old_va[old_i]
            for old_q in range(1, len(v43.REGIONS)):
                new_q = len(CHINA_ZONES) + old_q - 1
                for k in range(ns):
                    z[new_i, new_q * ns + k] = old_z[old_i, old_q * ns + k]
            for old_q in range(1, len(v43.REGIONS)):
                new_q = len(CHINA_ZONES) + old_q - 1
                fd[new_i, new_q] = old_fd[old_i, old_q]

            # Foreign supply to Chinese intermediate and final demand.
            for customer_s in range(ns):
                use_idx = np.arange(customer_s, len(domestic_nodes), ns)
                target = old_z[old_i, chn_old * ns + customer_s]
                z[new_i, use_idx] = target * absorption_share[:, customer_s]
            fd[new_i, :len(CHINA_ZONES)] = old_fd[old_i, chn_old] * final_region_share

    for supplier_s in range(ns):
        supply_idx = np.arange(supplier_s, len(domestic_nodes), ns)
        for old_q in range(1, len(v43.REGIONS)):
            new_q = len(CHINA_ZONES) + old_q - 1
            for customer_s in range(ns):
                target = old_z[chn_old * ns + supplier_s, old_q * ns + customer_s]
                z[supply_idx, new_q * ns + customer_s] = target * export_share[:, supplier_s]
            fd[supply_idx, new_q] = old_fd[chn_old * ns + supplier_s, old_q] * export_share[:, supplier_s]

    region_of_node = np.repeat(np.arange(nr), ns)
    sector_of_node = np.tile(np.arange(ns), nr)
    labels = np.array([f"{r}_{s}" for r in REGIONS for s in SECTORS], dtype=object)
    np.savez_compressed(
        NESTED_CACHE,
        Z=z,
        FD=fd,
        OUT=out,
        VA=va,
        labels=labels,
        region_of_node=region_of_node,
        sector_of_node=sector_of_node,
    )
    return {
        "Z": z, "FD": fd, "OUT": out, "VA": va, "labels": labels,
        "region_of_node": region_of_node, "sector_of_node": sector_of_node,
    }


def aggregate_nested_to_global(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    ns = len(SECTORS)
    z = np.zeros((len(ACTORS) * ns, len(ACTORS) * ns))
    fd = np.zeros((len(ACTORS) * ns, len(ACTORS)))
    out = np.zeros(len(ACTORS) * ns)
    va = np.zeros(len(ACTORS) * ns)
    actor_by_region = np.array([0] * len(CHINA_ZONES) + list(range(1, len(ACTORS))), dtype=int)
    for i in range(len(data["OUT"])):
        ai = actor_by_region[data["region_of_node"][i]] * ns + data["sector_of_node"][i]
        out[ai] += data["OUT"][i]
        va[ai] += data["VA"][i]
        for j in range(len(data["OUT"])):
            aj = actor_by_region[data["region_of_node"][j]] * ns + data["sector_of_node"][j]
            z[ai, aj] += data["Z"][i, j]
        for r in range(len(REGIONS)):
            fd[ai, actor_by_region[r]] += data["FD"][i, r]
    return {"Z": z, "FD": fd, "OUT": out, "VA": va}


def build_network(data: dict[str, np.ndarray]) -> dict[str, np.ndarray | float]:
    z, fd, out, va = data["Z"], data["FD"], data["OUT"], data["VA"]
    region_of_node = data["region_of_node"]
    sector_of_node = data["sector_of_node"]
    eps = 1e-12
    inter = z.sum(axis=0)
    dependency = (z / np.maximum(inter, eps)[None, :]).T
    dependency *= (inter / np.maximum(out, eps))[:, None]
    dependency /= max(np.linalg.norm(dependency, ord=np.inf), 1.0)
    propagation = np.linalg.inv(np.eye(len(out)) - 0.62 * dependency)
    twn_region = REGIONS.index("TWN")
    twn_rows = region_of_node == twn_region
    chn_rows = region_of_node < len(CHINA_ZONES)
    foreign_twn = ~twn_rows
    foreign_chn = ~chn_rows

    twn_supply = z[twn_rows, :].sum(axis=0) / np.maximum(out, eps)
    twn_import = np.zeros(len(out))
    twn_import[twn_rows] = z[foreign_twn][:, twn_rows].sum(axis=0) / np.maximum(out[twn_rows], eps)
    twn_export = np.zeros(len(out))
    twn_export[foreign_twn] = (z[foreign_twn][:, twn_rows].sum(axis=1) + fd[foreign_twn, twn_region]) / np.maximum(out[foreign_twn], eps)
    twn_trade_basis = np.clip(twn_supply + twn_import + 0.45 * twn_export, 0, 0.85)

    chn_import = np.zeros(len(out))
    chn_import[chn_rows] = z[foreign_chn][:, chn_rows].sum(axis=0) / np.maximum(out[chn_rows], eps)
    chn_supply = z[chn_rows, :].sum(axis=0) / np.maximum(out, eps)
    chn_export = np.zeros(len(out))
    chn_export[foreign_chn] = (z[foreign_chn][:, chn_rows].sum(axis=1) + fd[foreign_chn][:, :len(CHINA_ZONES)].sum(axis=1)) / np.maximum(out[foreign_chn], eps)
    chn_sanction_basis = np.clip(chn_import + 0.55 * chn_supply + 0.35 * chn_export, 0, 0.85)

    cross_region = np.array([z[region_of_node != region_of_node[j], j].sum() / max(out[j], eps) for j in range(len(out))])
    domestic_cross = np.zeros(len(out))
    for j in np.where(chn_rows)[0]:
        other_zone = chn_rows & (region_of_node != region_of_node[j])
        domestic_cross[j] = z[other_zone, j].sum() / max(out[j], eps)
    coastal_regions = np.array([REGIONS.index(x) for x in ("CHN_NORTH", "CHN_EAST_COAST", "CHN_SOUTHEAST_COAST")])
    coastal_rows = np.isin(region_of_node, coastal_regions)
    port_direct = np.zeros(len(out))
    port_direct[coastal_rows] = (
        z[foreign_chn][:, coastal_rows].sum(axis=0) + z[coastal_rows][:, foreign_chn].sum(axis=1)
    ) / np.maximum(out[coastal_rows], eps)

    chip_basis = np.zeros(len(out))
    twn_elec = twn_region * len(SECTORS) + SECTORS.index("electronics")
    chip_basis[twn_elec] = 0.70
    chip_basis += np.clip(1.8 * z[twn_elec, :] / np.maximum(out, eps), 0, 0.90)
    bases = {
        "twn_trade": np.clip(propagation @ twn_trade_basis, 0, 1),
        "chn_sanction": np.clip(propagation @ chn_sanction_basis, 0, 1),
        "shipping": np.clip(propagation @ np.clip(cross_region, 0, .75), 0, 1),
        "domestic": np.clip(propagation @ np.clip(domestic_cross, 0, .75), 0, 1),
        "ports": np.clip(propagation @ np.clip(port_direct, 0, .85), 0, 1),
        "chip": np.clip(propagation @ chip_basis, 0, 1),
    }
    zone_spill = np.zeros((len(CHINA_ZONES), len(out)))
    for zone in range(len(CHINA_ZONES)):
        direct = np.zeros(len(out))
        zone_nodes = region_of_node == zone
        direct[zone_nodes] = 0.35 + 0.65 * np.clip(domestic_cross[zone_nodes], 0, 1)
        zone_spill[zone] = np.clip(propagation @ direct, 0, 1)
    bases["zone_spill"] = zone_spill
    external_mask = (region_of_node[:, None] < len(CHINA_ZONES)) != (region_of_node[None, :] < len(CHINA_ZONES))
    chn_trade = float(z[external_mask].sum() + fd[chn_rows][:, len(CHINA_ZONES):].sum() + fd[foreign_chn][:, :len(CHINA_ZONES)].sum())
    twn_mask = region_of_node[:, None] != twn_region
    twn_trade = float(z[twn_rows, :][:, foreign_twn].sum() + z[foreign_twn, :][:, twn_rows].sum() + fd[twn_rows].sum() - fd[twn_rows, twn_region].sum())
    gross_trade = float(z[region_of_node[:, None] != region_of_node[None, :]].sum() + fd.sum())
    return {
        "dependency": dependency, "region_of_node": region_of_node, "sector_of_node": sector_of_node,
        "va": va, "out": out, "bases": bases,
        "chn_trade_share": min(chn_trade / max(gross_trade, eps), 1.0),
        "twn_trade_share": min(twn_trade / max(gross_trade, eps), 1.0),
    }


def actor_of_region(region_index: int) -> int:
    if region_index < len(CHINA_ZONES):
        return ACTORS.index("CHN")
    return ACTORS.index(REGIONS[region_index])


def simulate(
    network: dict[str, np.ndarray | float],
    paths: int = 600,
    months: int = 360,
    seed: int = 20261004,
    feedback_scale: float = 1.0,
):
    rng = np.random.default_rng(seed)
    n = len(network["out"])
    region_of_node = network["region_of_node"]
    sector_of_node = network["sector_of_node"]
    va = network["va"]
    bases = network["bases"]
    actor_of_node = np.array([actor_of_region(r) for r in region_of_node])
    gdp_weights = np.zeros((len(ACTORS), n))
    military_weights = np.zeros((len(ACTORS), n))
    military_mask = np.isin(sector_of_node, [SECTORS.index("electronics"), SECTORS.index("materials"), SECTORS.index("machinery_transport")])
    for a in range(len(ACTORS)):
        mask = actor_of_node == a
        gdp_weights[a, mask] = va[mask] / max(va[mask].sum(), 1e-12)
        mmask = mask & military_mask
        military_weights[a, mmask] = va[mmask] / max(va[mmask].sum(), 1e-12)
    global_weights = np.array([va[actor_of_node == a].sum() for a in range(len(ACTORS))])
    global_weights /= global_weights.sum()
    zone_weights = np.zeros((len(CHINA_ZONES), n))
    for zone in range(len(CHINA_ZONES)):
        mask = region_of_node == zone
        zone_weights[zone, mask] = va[mask] / max(va[mask].sum(), 1e-12)
    pass_through = np.array([.24, .44, .15, .30, .34, .22, .30, .26])
    exposure = np.array([.78, .96, .62, .66, .58, .52, .64, .38])
    results, summaries = [], {}
    checkpoints = {12: "1年", 36: "3年", 60: "5年", 120: "10年", 180: "15年", 240: "20年", 360: "30年"}

    for scenario_cn, scenario in v43.SCENARIOS.items():
        p = v43.scenario_parameters(scenario, rng, paths)
        if scenario == "baseline":
            domestic0 = domestic_lr = port0 = port_lr = np.zeros(paths)
        elif scenario == "blockade":
            domestic0 = v43.sample_uniform(rng, .04, .10, paths); domestic_lr = v43.sample_uniform(rng, .01, .04, paths)
            port0 = v43.sample_uniform(rng, .10, .22, paths); port_lr = v43.sample_uniform(rng, .03, .09, paths)
        elif scenario in {"fab_low_sub", "fab_huawei", "fab_global"}:
            domestic0 = v43.sample_uniform(rng, .06, .14, paths); domestic_lr = v43.sample_uniform(rng, .02, .06, paths)
            port0 = v43.sample_uniform(rng, .12, .25, paths); port_lr = v43.sample_uniform(rng, .04, .11, paths)
        else:
            domestic0 = v43.sample_uniform(rng, .15, .30, paths); domestic_lr = v43.sample_uniform(rng, .05, .14, paths)
            port0 = v43.sample_uniform(rng, .30, .55, paths); port_lr = v43.sample_uniform(rng, .10, .25, paths)

        inventory = v43.sample_uniform(rng, 2.0, 7.0, paths)
        advanced_taiwan_share = v43.sample_uniform(rng, .74, .88, paths)
        regional_gdp = np.ones((paths, len(ACTORS)))
        regional_infl = np.zeros_like(regional_gdp)
        regional_military = np.ones_like(regional_gdp)
        chip_gap = np.zeros_like(regional_gdp)
        inflation_state = np.zeros_like(regional_gdp)
        exchange_state = np.zeros_like(regional_gdp)
        demand_gap_state = np.zeros_like(regional_gdp)
        trade_loss = np.zeros(paths)
        zone_fiscal_stress = np.zeros((paths, len(CHINA_ZONES)))
        zone_social_stress = np.zeros_like(zone_fiscal_stress)
        zone_political_fragility = np.zeros_like(zone_fiscal_stress)
        zone_market_segmentation = np.zeros_like(zone_fiscal_stress)
        scenario_summary = {}

        for month in range(1, months + 1):
            recovery = 1.0 - np.exp(-month / p["recovery"])
            block = p["block_lr"] + (p["block0"] - p["block_lr"]) * (1.0 - recovery)
            fab_avail = p["fab0"] + (p["fab_lr"] - p["fab0"]) * recovery
            cert = 1.0 - np.exp(-month / 30.0)
            china_sub = p["china_sub"] * cert
            global_sub = p["global_sub"] * (1.0 - np.exp(-month / 54.0))
            armington = 1.15 + 1.35 * (1.0 - np.exp(-month / 48.0))
            firm_entry = (1.0 - np.exp(-month / 30.0)) * (armington / 2.5)
            reroute = np.clip(global_sub + .28 * firm_entry, 0, .72)
            gravity_gap = block * (1.0 - reroute)
            chip_raw_gap = advanced_taiwan_share * (1.0 - fab_avail) * (1.0 - global_sub)
            inventory = np.maximum(inventory - chip_raw_gap, 0.0)
            chip_coeff = np.clip(chip_raw_gap * (1.0 - np.minimum(inventory / 3.0, 1.0)), 0, .95)
            chn_coeff = p["sanction"] * (1.0 - .35 * cert)
            ship_coeff = p["shipping"] * (1.0 - .25 * cert)
            domestic_coeff = (domestic_lr + (domestic0 - domestic_lr) * (1.0 - recovery)) * (1.0 - .35 * cert)
            port_coeff = (port_lr + (port0 - port_lr) * (1.0 - recovery)) * (1.0 - .30 * cert)
            q = (
                gravity_gap[:, None] * bases["twn_trade"][None, :]
                + chip_coeff[:, None] * bases["chip"][None, :]
                + chn_coeff[:, None] * bases["chn_sanction"][None, :]
                + ship_coeff[:, None] * bases["shipping"][None, :]
                + domestic_coeff[:, None] * bases["domestic"][None, :]
                + port_coeff[:, None] * bases["ports"][None, :]
            )
            # Political coordination, local fiscal capacity and social stress change
            # interregional trade costs. The resulting shock is propagated through
            # both domestic and international customer-supplier links.
            endogenous_zone_friction = np.clip(
                .30 * zone_market_segmentation
                + .25 * zone_fiscal_stress
                + .25 * zone_social_stress
                + .20 * zone_political_fragility,
                0,
                .75,
            )
            q += .24 * feedback_scale * (endogenous_zone_friction @ bases["zone_spill"])
            q = np.clip(q, 0, .94)
            chn_critical = (actor_of_node == ACTORS.index("CHN")) & np.isin(sector_of_node, [SECTORS.index("electronics"), SECTORS.index("machinery_transport")])
            q[:, chn_critical] *= 1.0 - .72 * china_sub[:, None]
            output_ratio = 1.0 - q
            zone_output = output_ratio @ zone_weights.T
            zone_shortage = np.zeros_like(zone_output)
            for zone in range(len(CHINA_ZONES)):
                mask = region_of_node == zone
                weights = zone_weights[zone, mask]
                zone_shortage[:, zone] = q[:, mask] @ weights
            china_inflation = inflation_state[:, ACTORS.index("CHN")][:, None]
            zone_fiscal_target = np.clip(
                .46 * (1.0 - zone_output) + .22 * trade_loss[:, None]
                + .18 * domestic_coeff[:, None] + .14 * port_coeff[:, None], 0, 1
            )
            zone_social_target = np.clip(
                .34 * (1.0 - zone_output) + .26 * china_inflation
                + .28 * zone_shortage + .12 * domestic_coeff[:, None], 0, 1
            )
            zone_political_target = np.clip(
                .38 * zone_social_stress + .32 * zone_fiscal_stress
                + .18 * zone_shortage + .12 * port_coeff[:, None], 0, 1
            )
            zone_segmentation_target = np.clip(
                .42 * zone_political_fragility + .30 * zone_fiscal_stress
                + .18 * zone_social_stress + .10 * domestic_coeff[:, None], 0, 1
            )
            zone_fiscal_stress = .76 * zone_fiscal_stress + .24 * zone_fiscal_target
            zone_social_stress = .80 * zone_social_stress + .20 * zone_social_target
            zone_political_fragility = .86 * zone_political_fragility + .14 * zone_political_target
            zone_market_segmentation = .82 * zone_market_segmentation + .18 * zone_segmentation_target
            network_gdp = output_ratio @ gdp_weights.T
            regional_military = output_ratio @ military_weights.T
            chip_gap = chip_coeff[:, None] * exposure[None, :]
            chip_gap[:, ACTORS.index("CHN")] *= 1.0 - china_sub
            china_zone_weights = np.array([va[region_of_node == z].sum() for z in range(len(CHINA_ZONES))])
            china_zone_weights /= china_zone_weights.sum()
            internal_feedback = zone_market_segmentation @ china_zone_weights
            trade_loss = np.clip(
                network["twn_trade_share"] * gravity_gap
                + network["chn_trade_share"] * chn_coeff
                + .35 * ship_coeff + .08 * port_coeff + .05 * internal_feedback,
                0,
                .80,
            )
            risk = .36 * trade_loss[:, None] + .24 * chip_gap + .20 * ship_coeff[:, None] + .12 * chn_coeff[:, None]
            policy = np.maximum(inflation_state - .02, 0.0) * np.array([.55, .35, .85, .50, .70, .65, .45, .40])[None, :]
            exchange_state = .82 * exchange_state + .18 * risk - .10 * policy
            import_cost = .48 * trade_loss[:, None] + .31 * chip_gap + .21 * np.maximum(exchange_state, 0.0)
            target_inflation = pass_through[None, :] * import_cost + .12 * (1.0 - network_gdp) + .05 * chip_gap
            inflation_state = .72 * inflation_state + .28 * target_inflation
            demand_target = .30 * trade_loss[:, None] + .18 * risk + .12 * policy
            demand_gap_state = .68 * demand_gap_state + .32 * demand_target
            regional_gdp = np.clip(network_gdp * np.clip(1.0 - demand_gap_state / armington, .68, 1.0), 0, 1)
            regional_infl = 100.0 * inflation_state

            if month in checkpoints:
                for a, actor in enumerate(ACTORS + ["GLOBAL"]):
                    if actor == "GLOBAL":
                        gdp_loss = 1.0 - regional_gdp @ global_weights
                        inflation = regional_infl @ global_weights
                        military_loss = 1.0 - regional_military @ global_weights
                        cgap = chip_gap @ global_weights
                    else:
                        gdp_loss = 1.0 - regional_gdp[:, a]
                        inflation = regional_infl[:, a]
                        military_loss = 1.0 - regional_military[:, a]
                        cgap = chip_gap[:, a]
                    for metric, values in {
                        "实际GDP损失率": gdp_loss, "额外通胀_百分点": inflation,
                        "关键工业产出损失率": military_loss, "先进芯片服务缺口率": cgap,
                        "全球贸易量损失率": trade_loss,
                    }.items():
                        values = np.maximum(values, 0.0)
                        results.append({
                            "情景": scenario_cn, "期限": checkpoints[month], "地区": actor, "指标": metric,
                            "P10": float(np.quantile(values, .10)), "中位数": float(np.quantile(values, .50)),
                            "P90": float(np.quantile(values, .90)),
                        })
                china_zone_weights = np.array([va[region_of_node == z].sum() for z in range(len(CHINA_ZONES))])
                china_zone_weights /= china_zone_weights.sum()
                scenario_summary[checkpoints[month]] = {
                    "gdp_loss_global_median": float(np.median(1.0 - regional_gdp @ global_weights)),
                    "trade_loss_median": float(np.median(trade_loss)),
                    "gdp_loss_china_median": float(np.median(1.0 - regional_gdp[:, ACTORS.index("CHN")])),
                    "military_loss_china_median": float(np.median(1.0 - regional_military[:, ACTORS.index("CHN")])),
                    "chip_gap_china_median": float(np.median(chip_gap[:, ACTORS.index("CHN")])),
                    "china_fiscal_stress_median": float(np.median(zone_fiscal_stress @ china_zone_weights)),
                    "china_social_stress_median": float(np.median(zone_social_stress @ china_zone_weights)),
                    "china_political_fragility_median": float(np.median(zone_political_fragility @ china_zone_weights)),
                    "china_market_segmentation_median": float(np.median(zone_market_segmentation @ china_zone_weights)),
                }
        summaries[scenario] = scenario_summary
    return results, summaries


def load_coupling_profile(scenario_cn: str, paths: int, months: int, seed: int) -> dict[str, np.ndarray]:
    if v43.SCENARIOS.get(scenario_cn) == "baseline":
        one = np.ones((months, paths, 4), dtype=np.float32)
        zero = np.zeros((months, paths, 4), dtype=np.float32)
        return {"gdp_factor": one.copy(), "industry_factor": one.copy(), "chip_gap": zero.copy(), "inflation": zero.copy(), "trade_loss": np.zeros((months, paths), dtype=np.float32)}
    actors = ["CHN", "TWN", "USA", "JPN"]
    metrics = {"gdp_factor": "实际GDP损失率", "industry_factor": "关键工业产出损失率", "chip_gap": "先进芯片服务缺口率", "inflation": "额外通胀_百分点"}
    checkpoints = {"1年": 12, "3年": 36, "5年": 60, "10年": 120, "15年": 180, "20年": 240, "30年": 360}
    checkpoint_months = (12, 36, 60, 120, 180, 240, 360)
    rows = {}
    with OUT_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["情景"] == scenario_cn and row["地区"] in actors:
                rows[(row["地区"], row["指标"], checkpoints[row["期限"]])] = (float(row["P10"]), float(row["中位数"]), float(row["P90"]))
    rng = np.random.default_rng(seed)
    common_z = rng.normal(0, 1, paths)
    actor_z = rng.normal(0, 1, (paths, 4))
    idio_z = rng.normal(0, 1, (paths, 4, 4))
    zscore = .62 * common_z[:, None, None] + .27 * actor_z[:, :, None] + .11 * idio_z
    usable_months = tuple(t for t in checkpoint_months if t <= months)
    times = np.array((0,) + usable_months, dtype=float)
    profile = {key: np.zeros((months, paths, 4), dtype=np.float32) for key in metrics}
    for a, actor in enumerate(actors):
        for m, (key, metric) in enumerate(metrics.items()):
            vals = [rows[(actor, metric, t)] for t in usable_months]
            lo = [0] + [x[0] for x in vals]
            md = [0] + [x[1] for x in vals]
            hi = [0] + [x[2] for x in vals]
            med = np.interp(np.arange(1, months + 1), times, md)
            sd = np.interp(np.arange(1, months + 1), times, (np.asarray(hi) - np.asarray(lo)) / 2.563)
            values = med[:, None] + sd[:, None] * zscore[None, :, a, m]
            if key == "inflation":
                values = np.clip(values / 100.0, 0, .40)
            else:
                values = np.clip(values, 0, .95)
            if key in {"gdp_factor", "industry_factor"}:
                values = 1.0 - values
            profile[key][:, :, a] = values.astype(np.float32)
    trade_rows = {}
    with OUT_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["情景"] == scenario_cn and row["地区"] == "GLOBAL" and row["指标"] == "全球贸易量损失率":
                trade_rows[checkpoints[row["期限"]]] = (float(row["P10"]), float(row["中位数"]), float(row["P90"]))
    vals = [trade_rows[t] for t in usable_months]
    lo = [0] + [x[0] for x in vals]
    md = [0] + [x[1] for x in vals]
    hi = [0] + [x[2] for x in vals]
    med = np.interp(np.arange(1, months + 1), times, md)
    sd = np.interp(np.arange(1, months + 1), times, (np.asarray(hi) - np.asarray(lo)) / 2.563)
    profile["trade_loss"] = np.clip(med[:, None] + sd[:, None] * common_z[None, :], 0, .85).astype(np.float32)
    return profile


def validate_nested(data: dict[str, np.ndarray]) -> dict[str, float | bool]:
    old = v43.aggregate_icio()
    aggregated = aggregate_nested_to_global(data)
    checks = {}
    for key in ("Z", "FD", "OUT", "VA"):
        denominator = max(float(np.max(np.abs(old[key]))), 1e-12)
        checks[f"aggregate_{key}_max_relative_error"] = float(np.max(np.abs(aggregated[key] - old[key])) / denominator)
    checks["nonnegative_Z"] = bool(np.min(data["Z"]) >= -1e-10)
    checks["finite_FD"] = bool(np.isfinite(data["FD"]).all())
    checks["minimum_FD_including_inventory_change"] = float(np.min(data["FD"]))
    return checks


def write_outputs(data, network, results, summaries, checks):
    with OUT_BASE.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["地区", "产业", "总产出_百万美元", "增加值_百万美元", "中间投入_百万美元", "最终需求_百万美元"])
        for i, label in enumerate(data["labels"]):
            region = REGIONS[data["region_of_node"][i]]
            sector = SECTORS[data["sector_of_node"][i]]
            writer.writerow([region, sector, data["OUT"][i], data["VA"][i], data["Z"][:, i].sum(), data["FD"][i].sum()])
    with OUT_CSV.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["情景", "期限", "地区", "指标", "P10", "中位数", "P90"])
        writer.writeheader(); writer.writerows(results)
    payload = {
        "model": "V4.4 nested Chinese provincial and international production-network equilibrium",
        "data": {"domestic": "CEADs China MRIO 2017, 31 provinces x 42 sectors", "international": "OECD ICIO 2022", "mrio_md5": hashlib.md5(MRIO_XLSX.read_bytes()).hexdigest()},
        "dimensions": {"domestic_aggregated_nodes": 70, "nested_nodes": 140, "china_zones": CHINA_ZONES, "sectors": SECTORS},
        "account_validation": checks,
        "simulation": {"paths": 600, "months": 360, "seed": 20261004, "scenarios": list(v43.SCENARIOS)},
        "horizon_summary": summaries,
        "limitations": ["2017 domestic topology is rescaled to the 2022 OECD China account.", "Post-2022 provincial restructuring is not directly observed.", "Port, repair, certification and substitution paths are interval structural priors."],
    }
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    data = build_nested_accounts()
    checks = validate_nested(data)
    if any(v > 1e-10 for k, v in checks.items() if k.endswith("error")) or not checks["nonnegative_Z"] or not checks["finite_FD"]:
        raise RuntimeError(f"Nested account validation failed: {checks}")
    network = build_network(data)
    results, summaries = simulate(network)
    values = np.array([[r["P10"], r["中位数"], r["P90"]] for r in results])
    if not np.isfinite(values).all() or not np.all(values[:, 0] <= values[:, 1]) or not np.all(values[:, 1] <= values[:, 2]):
        raise RuntimeError("Simulation quantile validation failed")
    write_outputs(data, network, results, summaries, checks)
    print("DUAL_CIRCULATION_V44_VERIFICATION: PASS")
    print(json.dumps({"account_checks": checks, "summary": summaries}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
