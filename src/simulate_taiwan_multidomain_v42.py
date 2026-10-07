import csv
import json
import math
import os
import re
import sys
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

LOCAL_DEPS = Path(__file__).resolve().parent / ".v42deps"
if LOCAL_DEPS.exists():
    sys.path.insert(0, str(LOCAL_DEPS))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

CHINESE_FONT = Path(r"C:\Windows\Fonts\msyh.ttc")
if CHINESE_FONT.exists():
    font_manager.fontManager.addfont(str(CHINESE_FONT))
    plt.rcParams["font.family"] = "Microsoft YaHei"
plt.rcParams["axes.unicode_minus"] = False


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
WORK = ROOT / "work"
V40_PATH = OUT / "台海V4.0双方对称相关失效与稀有事件_仿真摘要.json"
V41_PATH = OUT / "台海V4.1三维地形与多源PNT_仿真摘要.json"
SUMMARY_PATH = OUT / "台海V4.2海空陆认知四场联合仿真摘要.json"
RESULTS_PATH = OUT / "台海V4.2多域联合条件结果.csv"
ABLATION_PATH = OUT / "台海V4.2消融实验.csv"
STATE_PATH = OUT / "台海V4.2代表路径状态.csv"
FIGURE_PATH = OUT / "台海V4.2条件广泛控制率.png"

SEED = 20261004
PATHS = int(os.environ.get("TAIWAN_V42_PATHS", "600"))
MONTHS = int(os.environ.get("TAIWAN_V42_MONTHS", "360"))
HORIZON_YEARS = (5, 10, 15, 20, 30)
HORIZON_MONTHS = tuple(years * 12 for years in HORIZON_YEARS if years * 12 <= MONTHS)
YEARS = (2030, 2050, 2075, 2100)
CASES = ("timely_full", "limited", "delayed", "japan_only", "taiwan_alone")
CASE_LABELS = {
    "timely_full": "美日及时全面介入",
    "limited": "有限介入",
    "delayed": "延迟介入",
    "japan_only": "日本单独介入",
    "taiwan_alone": "台湾单独作战",
}

CASE_CONFIG = {
    "timely_full": {"us": 1.00, "jp": 0.82, "minor": 0.22, "delay": 0, "def_log": 1.00, "csg": 1.00},
    "limited": {"us": 0.48, "jp": 0.42, "minor": 0.10, "delay": 0, "def_log": 0.72, "csg": 0.60},
    "delayed": {"us": 0.82, "jp": 0.62, "minor": 0.16, "delay": 12, "def_log": 0.84, "csg": 0.78},
    "japan_only": {"us": 0.03, "jp": 0.58, "minor": 0.02, "delay": 0, "def_log": 0.55, "csg": 0.24},
    "taiwan_alone": {"us": 0.02, "jp": 0.05, "minor": 0.00, "delay": 0, "def_log": 0.34, "csg": 0.08},
}

SCENARIOS = {
    "central": {
        "label": "中心数据区间",
        "china_shock": 1.0, "def_shock": 1.0, "common_shock": 1.0,
        "china_supply": 1.0, "def_supply": 1.0, "cognitive": 1.0,
    },
    "external_system_stress": {
        "label": "美日台体系同步受损",
        "china_shock": 0.90, "def_shock": 1.85, "common_shock": 1.25,
        "china_supply": 1.05, "def_supply": 0.76, "cognitive": 1.18,
    },
    "china_system_stress": {
        "label": "中国体系同步受损",
        "china_shock": 1.85, "def_shock": 0.92, "common_shock": 1.25,
        "china_supply": 0.76, "def_supply": 1.04, "cognitive": 1.18,
    },
    "symmetric_extreme": {
        "label": "双方共同极端失效",
        "china_shock": 1.65, "def_shock": 1.65, "common_shock": 1.75,
        "china_supply": 0.84, "def_supply": 0.84, "cognitive": 1.32,
    },
    "protracted_blockade": {
        "label": "长期封锁与补给收缩",
        "china_shock": 1.20, "def_shock": 1.12, "common_shock": 1.25,
        "china_supply": 0.70, "def_supply": 0.88, "cognitive": 1.28,
    },
}

COMPONENT_NAMES = ("fields", "killweb_ooda", "logistics", "terrain_pnt", "cognition", "resilience", "resolution", "cross_domain")
TAIL_TARGET_PROB = np.array([0.78, 0.05, 0.04, 0.035, 0.055, 0.04])
TAIL_PROPOSAL_PROB = np.array([0.28, 0.15, 0.14, 0.13, 0.15, 0.15])


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -35.0, 35.0)))


def logit(p):
    p = np.clip(p, 1e-9, 1.0 - 1e-9)
    return np.log(p / (1.0 - p))


def q(values):
    return [float(x) for x in np.quantile(values, (0.10, 0.50, 0.90))]


def entropy_transport_fulfillment(
    supply, demand, epsilon=0.22, marginal_penalty=0.70, iterations=20
):
    """KL-relaxed unbalanced Sinkhorn allocation with capacity projection."""
    original_shape = supply.shape[:-1]
    a = np.maximum(supply.reshape(-1, 4), 1e-9)
    b = np.maximum(demand.reshape(-1, 2), 1e-9)
    cost = np.array([
        [0.08, 0.72], [0.35, 0.28], [0.46, 0.20], [0.62, 0.12],
    ])
    kernel = np.exp(-cost / epsilon)[None, :, :]
    u = np.ones_like(a)
    v = np.ones_like(b)
    relaxation = marginal_penalty / (marginal_penalty + epsilon)
    for _ in range(iterations):
        u = np.power(
            a / np.maximum(np.sum(kernel * v[:, None, :], axis=2), 1e-12),
            relaxation,
        )
        v = np.power(
            b / np.maximum(np.sum(kernel * u[:, :, None], axis=1), 1e-12),
            relaxation,
        )
    plan = kernel * u[:, :, None] * v[:, None, :]
    source_scale = np.minimum(
        1.0, a / np.maximum(np.sum(plan, axis=2), 1e-12)
    )
    plan *= source_scale[:, :, None]
    demand_scale = np.minimum(
        1.0, b / np.maximum(np.sum(plan, axis=1), 1e-12)
    )
    plan *= demand_scale[:, None, :]
    # Conversion is lossy, so relaxed source and demand marginals need not
    # match. Idle capacity and unmet orders remain explicit.
    conversion_yield = np.exp(-0.65 * cost)[None, :, :]
    delivered = np.sum(plan * conversion_yield, axis=1)
    fulfillment = np.clip(delivered / np.maximum(b, 1e-9), 0.0, 1.0)
    return fulfillment.reshape(*original_shape, 2)


def xlsx_rows(path, sheet_number):
    ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    with zipfile.ZipFile(path) as archive:
        shared_root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
        shared = ["".join(t.text or "" for t in item.iter(ns + "t"))
                  for item in shared_root.findall(ns + "si")]
        root = ET.fromstring(archive.read(f"xl/worksheets/sheet{sheet_number}.xml"))
    rows = []
    for row in root.iter(ns + "row"):
        values = {}
        for cell in row.findall(ns + "c"):
            column = re.match(r"[A-Z]+", cell.attrib["r"]).group()
            node = cell.find(ns + "v")
            value = "" if node is None else node.text or ""
            if cell.attrib.get("t") == "s" and value:
                value = shared[int(value)]
            values[column] = value
        rows.append(values)
    return rows


def load_sipri_2025():
    rows = xlsx_rows(WORK / "SIPRI-Milex-data-1949-2025_v1.2.xlsx", 5)
    header = next(row for row in rows if row.get("A") == "Country")
    year_column = next(column for column, value in header.items() if value == "2025")
    names = {
        "China": "china",
        "United States of America": "united_states",
        "Japan": "japan",
        "Taiwan": "taiwan",
    }
    values = {}
    for row in rows:
        actor = names.get(row.get("A", ""))
        if actor:
            values[actor] = float(row[year_column])
    return values


def load_inputs():
    v40 = json.loads(V40_PATH.read_text(encoding="utf-8"))
    v41 = json.loads(V41_PATH.read_text(encoding="utf-8"))
    base = {}
    for row in v41["results"]:
        if row["scenario"] == "baseline_contested":
            base[(int(row["conflict_year"]), row["case"])] = float(row["v41_probability"])
    v40_rows = {(int(r["conflict_year"]), r["case"]): r for r in v40["results"]}
    return base, v40_rows, v41["terrain"], load_sipri_2025()


def initialize(rng, year, case, terrain, sipri):
    n = PATHS
    cfg = CASE_CONFIG[case]
    year_scale = np.clip((year - 2030) / 70.0, 0.0, 1.0)
    budget_share = sipri["china"] / (
        sipri["china"] + sipri["united_states"] + sipri["japan"] + sipri["taiwan"]
    )
    china_growth = 0.92 + 0.24 * year_scale
    external_depth = 0.42 + 0.38 * cfg["us"] + 0.14 * cfg["jp"]
    military_anchor = np.clip(0.88 + 0.55 * budget_share, 0.90, 1.10)

    air = np.clip(rng.normal(0.46 + 0.10 * china_growth - 0.11 * external_depth, 0.055, n), 0.08, 0.92)
    sea = np.clip(rng.normal(0.48 + 0.11 * china_growth - 0.13 * external_depth, 0.055, n), 0.08, 0.92)
    land = np.clip(rng.beta(1.5, 42.0, n), 0.0, 0.12)
    cognition_cn = np.clip(rng.normal(0.69, 0.045, n), 0.35, 0.90)
    cognition_def = np.clip(rng.normal(0.72, 0.045, n), 0.35, 0.92)

    nodes_cn = np.clip(rng.normal([0.92, 0.93, 0.91, 0.90, 0.88, 0.90], 0.035, (n, 6)), 0.45, 0.995)
    nodes_def = np.clip(rng.normal([0.91, 0.91, 0.93, 0.92, 0.91, 0.90], 0.035, (n, 6)), 0.45, 0.995)
    stocks_cn = np.clip(rng.normal([0.91, 0.96, 0.90], 0.055, (n, 3)), 0.35, 1.20)
    stocks_def = np.clip(rng.normal([0.92, 0.93, 0.91], 0.055, (n, 3)), 0.35, 1.20)
    repair_cn = np.zeros(n)
    repair_def = np.zeros(n)
    pnt_cn = np.clip(rng.normal(0.80, 0.045, n), 0.35, 0.98)
    pnt_def = np.clip(rng.normal(0.81 + 0.02 * cfg["us"], 0.045, n), 0.35, 0.99)
    web_cn = np.clip(0.48 * nodes_cn[:, 2] + 0.30 * pnt_cn + 0.22 * nodes_cn[:, 3], 0.20, 0.98)
    web_def = np.clip(0.48 * nodes_def[:, 2] + 0.30 * pnt_def + 0.22 * nodes_def[:, 3], 0.20, 0.98)
    ooda_cn = np.clip(rng.normal(0.75, 0.045, n), 0.30, 0.95)
    ooda_def = np.clip(rng.normal(0.77 + 0.03 * cfg["us"], 0.045, n), 0.30, 0.97)
    search_cn = np.clip(rng.normal(0.42, 0.07, n), 0.08, 0.82)
    search_def = np.clip(rng.normal(0.47 + 0.08 * cfg["us"], 0.07, n), 0.08, 0.88)
    csg = np.clip(rng.normal(0.35 + 0.50 * cfg["csg"], 0.06, n), 0.05, 0.96)
    resilience_cn = np.clip(rng.normal([0.86, 0.84, 0.82], 0.035, (n, 3)), 0.55, 0.98)
    resilience_def = np.clip(rng.normal([0.84, 0.83, 0.84], 0.035, (n, 3)), 0.55, 0.98)
    actor_resilience = np.clip(
        rng.normal(
            np.array([
                [0.86, 0.84, 0.82, 0.87],
                [0.82, 0.80, 0.83, 0.79],
                [0.91, 0.82, 0.80, 0.92],
                [0.88, 0.84, 0.82, 0.88],
            ])[None, :, :],
            0.035,
            (n, 4, 4),
        ),
        0.50, 0.99,
    )
    manpower_base = np.array([
        [[0.92, 0.88, 0.86, 0.80, 0.88], [0.62, 0.52, 0.46, 0.32, 0.54], [1.00, 0.95, 0.92, 0.82, 0.96]],
        [[0.84, 0.78, 0.76, 0.70, 0.78], [0.70, 0.55, 0.48, 0.30, 0.58], [0.94, 0.90, 0.88, 0.80, 0.92]],
        [[0.96, 0.95, 0.95, 0.94, 0.95], [0.82, 0.78, 0.76, 0.70, 0.80], [1.00, 1.00, 1.00, 0.98, 1.00]],
        [[0.82, 0.88, 0.90, 0.86, 0.86], [0.48, 0.50, 0.48, 0.42, 0.52], [0.86, 0.88, 0.90, 0.88, 0.88]],
    ])

    return {
        "air": air, "sea": sea, "land": land,
        "cog_cn": cognition_cn, "cog_def": cognition_def,
        "nodes_cn": nodes_cn, "nodes_def": nodes_def,
        "stocks_cn": stocks_cn, "stocks_def": stocks_def,
        "repair_cn": repair_cn, "repair_def": repair_def,
        "pnt_cn": pnt_cn, "pnt_def": pnt_def,
        "web_cn": web_cn, "web_def": web_def,
        "ooda_cn": ooda_cn, "ooda_def": ooda_def,
        "search_cn": search_cn, "search_def": search_def,
        "csg": csg,
        "res_cn": resilience_cn, "res_def": resilience_def,
        "mobil_cn": np.zeros(n), "mobil_def": np.zeros(n),
        "actor_res": actor_resilience,
        "actor_collapse": np.zeros((n, 4, 4), dtype=bool),
        "actor_collapse_pressure_duration": np.zeros((n, 4, 4), dtype=np.int16),
        "actor_coup": np.zeros((n, 4), dtype=bool),
        "actor_panic": np.clip(rng.normal([0.08, 0.10, 0.06, 0.07], 0.02, (n, 4)), 0.01, 0.30),
        "actor_norm": np.clip(rng.normal([0.78, 0.75, 0.72, 0.76], 0.035, (n, 4)), 0.40, 0.95),
        "actor_network": np.clip(rng.normal([0.90, 0.87, 0.92, 0.90], 0.025, (n, 4)), 0.60, 0.99),
        "actor_action": np.zeros((n, 4)),
        "actor_social_mobilization": np.zeros((n, 4)),
        "actor_social_fatigue": np.zeros((n, 4)),
        "actor_firm_distress": np.clip(
            rng.normal(np.array([0.04, 0.05, 0.06, 0.08])[None, None, :], 0.012, (n, 4, 4)),
            0.01, 0.18,
        ),
        "actor_debt_stress": np.clip(rng.normal([0.42, 0.38, 0.52, 0.56], 0.035, (n, 4)), 0.20, 0.75),
        "actor_sector_output": np.clip(rng.normal(0.94, 0.025, (n, 4, 4)), 0.70, 1.0),
        "actor_firm_factors": np.clip(rng.normal(0.95, 0.025, (n, 4, 4, 3)), 0.72, 1.05),
        "actor_firm_tfp": np.clip(rng.normal(0.98, 0.018, (n, 4, 4)), 0.82, 1.06),
        "actor_command": np.clip(rng.normal(0.92, 0.025, (n, 4, 3)), 0.70, 0.99),
        "actor_manpower": np.clip(rng.normal(manpower_base[None, :, :, :], 0.025, (n, 4, 3, 5)), 0.05, 1.10),
        "actor_training": np.zeros((n, 4, 5)),
        "actor_manpower_eff": np.clip(rng.normal([0.87, 0.78, 0.95, 0.86], 0.025, (n, 4)), 0.60, 1.0),
        "actor_support": np.clip(rng.normal([0.72, 0.68, 0.58, 0.62], 0.04, (n, 4)), 0.35, 0.90),
        "actor_labor_fatigue": np.clip(rng.normal(0.08, 0.02, (n, 4)), 0.01, 0.18),
        "actor_labor_mismatch": np.clip(rng.normal([0.20, 0.18, 0.14, 0.16], 0.025, (n, 4)), 0.06, 0.32),
        "actor_labor_skill": np.clip(
            rng.normal(
                np.array([
                    [0.95, 0.82, 0.62, 0.38],
                    [0.82, 0.78, 0.68, 0.46],
                    [0.86, 0.88, 0.82, 0.68],
                    [0.72, 0.84, 0.80, 0.62],
                ])[None, :, :],
                0.025,
                (n, 4, 4),
            ),
            0.20, 1.05,
        ),
        "actor_skill_training": np.zeros((n, 4, 3)),
        "actor_inflation": np.clip(rng.normal([0.025, 0.022, 0.024, 0.020], 0.006, (n, 4)), 0.005, 0.06),
        "actor_order_backlog": np.zeros((n, 4, 2)),
        "actor_cumulative_economic_aid": np.zeros((n, 4)),
        "actor_financial_capacity": np.ones((n, 4)),
        "actor_strategy_controls": np.repeat(
            np.array([0.26, 0.25, 0.21, 0.19, 0.09])[None, None, :],
            n * 4, axis=0,
        ).reshape(n, 4, 5),
        "conflict_regime": np.zeros(n, dtype=np.int8),
        "regime_duration": np.zeros(n, dtype=np.int16),
        "actor_homeland_damage": np.zeros((n, 4)),
        "actor_island_chain_damage": np.zeros((n, 4)),
        "actor_front_rear_integrity": np.clip(rng.normal(0.92, 0.025, (n, 4, 3)), 0.70, 0.99),
        "actor_backlash": np.zeros((n, 4)),
        "leader_continuity": np.ones((n, 4)),
        "leader_disruption_count": np.zeros((n, 4), dtype=np.int16),
        "taiwan_proxy_capacity": np.zeros(n),
        "taiwan_proxy_leakage": np.zeros(n),
        "taiwan_agent_type_share": np.zeros((n, 4)),
        "settlement_outcome": np.zeros(n, dtype=np.int8),
        "broad_control_streak": np.zeros(n, dtype=np.int16),
        "broad_control_achieved": np.zeros(n, dtype=bool),
        "broad_control_month": np.zeros(n, dtype=np.int16),
        "china_growth": china_growth,
        "external_depth": external_depth,
        "military_anchor": military_anchor,
        "mountain_share": float(terrain["land_share_above_500m"]),
        "sky": float(terrain["sky_view_proxy_p10_p50_p90"][1]),
    }


def intervention_level(case, month):
    cfg = CASE_CONFIG[case]
    if cfg["delay"] <= 0:
        return cfg["us"], cfg["jp"]
    ramp = sigmoid((month - cfg["delay"]) / 2.5)
    return cfg["us"] * ramp, cfg["jp"] * ramp


def simulate_mechanisms(
    rng, year, case, scenario_key, terrain, sipri, keep_trace=False,
    economy_profile=None, financial_system=None, prewar_state=None,
    module_flags=None, social_system=None, macro_system=None, mfg_system=None,
    operations_system=None, pnt_system=None, strategy_system=None,
    spatial_system=None, mechanism_system=None, energy_system=None,
    geometry_system=None, hybrid_system=None, topology_system=None,
):
    scenario = SCENARIOS[scenario_key]
    cfg = CASE_CONFIG[case]
    s = initialize(rng, year, case, terrain, sipri)
    n = PATHS
    tail_regime = rng.choice(len(TAIL_TARGET_PROB), n, p=TAIL_PROPOSAL_PROB)
    importance_weight = TAIL_TARGET_PROB[tail_regime] / TAIL_PROPOSAL_PROB[tail_regime]
    tail_actor_multiplier = np.ones((n, 4))
    tail_actor_multiplier[tail_regime == 1, 0] = 3.2
    tail_actor_multiplier[tail_regime == 2, 1] = 3.2
    tail_actor_multiplier[tail_regime == 3, 2:] = 3.0
    tail_actor_multiplier[tail_regime == 5, :] = 2.7
    tail_common_multiplier = np.where(tail_regime == 4, 3.5, 1.0)
    tail_common_multiplier = np.where(tail_regime == 5, 2.6, tail_common_multiplier)
    common_hawkes_memory = np.zeros(n)
    actor_hawkes_memory = np.zeros((n, 4))
    flags = {
        "stackelberg_alliance": True,
        "optimal_transport": True,
        "graphon_social": True,
        "stratified_social_contagion": social_system is not None,
        "continuous_multipop_mfg": mfg_system is not None,
        "dynamic_cge_dsge": macro_system is not None,
        "operational_constraints": operations_system is not None,
        "dynamic_geospatial_pnt": pnt_system is not None,
        "rolling_differential_game": strategy_system is not None,
        "spatial_control_network": spatial_system is not None,
        "dynamic_mechanism_design": mechanism_system is not None,
        "four_party_energy_network": energy_system is not None,
        "geometric_multiscale_coupling": geometry_system is not None,
        "hybrid_filippov_impulse": hybrid_system is not None,
        "topological_dynamics": topology_system is not None,
        "principal_agent": True,
    }
    if module_flags is not None:
        flags.update(module_flags)
    if prewar_state is not None:
        s["actor_support"] = np.clip(prewar_state["actor_support"], 0.01, 0.99)
        readiness = np.clip(prewar_state["taiwan_readiness"], 0.70, 1.45)
        s["stocks_def"] = np.clip(s["stocks_def"] * readiness[:, None], 0.01, 1.35)
        s["nodes_def"][:, :5] = np.clip(
            s["nodes_def"][:, :5] * np.sqrt(readiness[:, None]), 0.03, 0.995
        )
        continuity = np.clip(prewar_state["leader_continuity"], 0.20, 1.0)
        s["leader_continuity"] = continuity.copy()
        s["actor_command"] *= continuity[:, :, None]
        s["actor_norm"] *= 0.85 + 0.15 * continuity
        s["actor_panic"] += 0.10 * (1.0 - continuity)
        s["actor_command"] = np.clip(s["actor_command"], 0.03, 0.995)
        s["actor_norm"] = np.clip(s["actor_norm"], 0.0, 1.0)
        s["actor_panic"] = np.clip(s["actor_panic"], 0.0, 1.0)
        economic_capital = np.clip(prewar_state["economic_capital"], 0.20, 8.0)
        military_capital = np.clip(prewar_state["military_capital"], 0.35, 5.0)
        economic_factor = np.clip(0.78 + 0.22 * np.sqrt(economic_capital), 0.75, 1.45)
        military_factor = np.clip(0.76 + 0.24 * military_capital, 0.72, 1.55)
        s["actor_sector_output"] = np.clip(
            s["actor_sector_output"] * economic_factor[:, :, None], 0.20, 1.30
        )
        s["actor_firm_tfp"] = np.clip(
            s["actor_firm_tfp"] * np.sqrt(economic_factor)[:, :, None], 0.35, 1.25
        )
        s["actor_res"] = np.clip(
            s["actor_res"] * (0.88 + 0.12 * military_factor[:, :, None]), 0.20, 1.10
        )
        s["actor_manpower_eff"] = np.clip(
            s["actor_manpower_eff"] * np.sqrt(military_factor), 0.30, 1.20
        )
        s["stocks_cn"] = np.clip(
            s["stocks_cn"] * military_factor[:, 0, None], 0.01, 1.60
        )
        s["stocks_def"] = np.clip(
            s["stocks_def"] * military_factor[:, 1, None], 0.01, 1.60
        )
        s["nodes_cn"] = np.clip(
            s["nodes_cn"] * np.sqrt(military_factor[:, 0, None]), 0.03, 0.995
        )
        s["nodes_def"] = np.clip(
            s["nodes_def"] * np.sqrt(military_factor[:, 1, None]), 0.03, 0.995
        )
    sums = {name: np.zeros(n) for name in (
        "air", "sea", "csi", "web_diff", "stock_diff", "port_diff",
        "pnt_diff", "cog_diff", "cross", "blockade", "throughput",
        "resilience_diff", "mobilization_diff",
        "command_diff", "output_diff", "conflict_intensity", "homeland_damage_diff", "terminal_resolution",
    )}
    min_stock_cn = np.ones(n) * 2.0
    min_stock_def = np.ones(n) * 2.0
    max_repair_cn = np.zeros(n)
    max_repair_def = np.zeros(n)
    trace = []
    snapshots = {}
    regime_intensity = np.array([1.00, 0.42, 0.06, 0.025, 0.12, 0.00])

    for month in range(MONTHS):
        if hybrid_system is not None:
            hybrid_system.begin_month(month)
        us_ceiling, jp_ceiling = intervention_level(case, month)
        # Partial Stackelberg response: US signaling moves first, while Japan
        # retains an independent cost/support response inside the case ceiling.
        if flags["stackelberg_alliance"]:
            prior_damage = s["actor_homeland_damage"]
            us_response = sigmoid(
                1.10 + 1.15 * (s["actor_support"][:, 2] - 0.50)
                + 0.55 * s["actor_support"][:, 1] - 0.65 * prior_damage[:, 2]
                - 0.30 * s["actor_debt_stress"][:, 2]
            )
            us = us_ceiling * np.clip(0.55 + 0.45 * us_response, 0.0, 1.0)
            japan_response = sigmoid(
                0.55 + 0.70 * us + 1.05 * (s["actor_support"][:, 3] - 0.50)
                + 0.45 * s["actor_support"][:, 1] - 0.75 * prior_damage[:, 3]
                - 0.28 * s["actor_debt_stress"][:, 3]
            )
            jp = jp_ceiling * np.clip(0.48 + 0.52 * japan_response, 0.0, 1.0)
            jp = np.minimum(jp_ceiling, np.maximum(jp, np.minimum(0.55 * us, jp_ceiling)))
        else:
            us = np.full(n, us_ceiling)
            jp = np.full(n, jp_ceiling)
        if macro_system is not None:
            macro_factors = macro_system.current()
            trade_gdp_factor = macro_factors.gdp_factor
            trade_industry_factor = macro_factors.industry_factor
            trade_chip_gap = macro_factors.chip_gap
            trade_inflation = macro_factors.inflation
            trade_volume_loss = macro_factors.trade_loss
            macro_trade_balance = macro_factors.trade_balance
            macro_government_balance = macro_factors.government_balance
        elif economy_profile is None:
            trade_gdp_factor = np.ones((n, 4))
            trade_industry_factor = np.ones((n, 4))
            trade_chip_gap = np.zeros((n, 4))
            trade_inflation = np.zeros((n, 4))
            trade_volume_loss = np.zeros(n)
            macro_trade_balance = np.zeros((n, 4))
            macro_government_balance = np.zeros((n, 4))
        else:
            trade_gdp_factor = economy_profile["gdp_factor"][month]
            trade_industry_factor = economy_profile["industry_factor"][month]
            trade_chip_gap = economy_profile["chip_gap"][month]
            trade_inflation = economy_profile["inflation"][month]
            trade_volume_loss = economy_profile["trade_loss"][month]
            macro_trade_balance = np.zeros((n, 4))
            macro_government_balance = np.zeros((n, 4))
        energy_opening = energy_system.current() if energy_system is not None else None
        hybrid_factors = None
        topology_factors = None
        if flags["hybrid_filippov_impulse"] and hybrid_system is not None:
            opening_shortage_cn = np.clip(1.0 - np.min(s["stocks_cn"], axis=1), 0.0, 1.0)
            opening_shortage_def = np.clip(1.0 - np.min(s["stocks_def"], axis=1), 0.0, 1.0)
            opening_shortage = np.column_stack((
                opening_shortage_cn, opening_shortage_def,
                0.35 * opening_shortage_def, 0.40 * opening_shortage_def,
            ))
            opening_logistics = np.mean(s["actor_front_rear_integrity"], axis=2)
            if energy_opening is not None:
                opening_logistics *= energy_opening.logistics
            opening_command = (
                np.prod(np.clip(s["actor_command"], 0.03, 1.0), axis=2) ** (1.0 / 3.0)
            ) * s["leader_continuity"]
            hybrid_factors = hybrid_system.update_filippov(
                logistics_service=opening_logistics,
                financial_capacity=s["actor_financial_capacity"],
                debt_stress=np.clip(s["actor_debt_stress"], 0.0, 1.0),
                stock_shortage=opening_shortage,
                support=s["actor_support"],
                conflict_intensity=regime_intensity[s["conflict_regime"]],
                command_service=opening_command,
            )
            if flags["topological_dynamics"] and topology_system is not None:
                topology_factors = topology_system.update(
                    regime=hybrid_factors.regime,
                    conflict_regime=s["conflict_regime"],
                    collapse=s["actor_collapse"],
                )
            s["actor_financial_capacity"] = np.clip(
                s["actor_financial_capacity"] * hybrid_factors.financial
                * (topology_factors.financial if topology_factors is not None else 1.0),
                0.03, 1.0,
            )
            if topology_factors is not None:
                s["actor_panic"] = np.clip(
                    s["actor_panic"] + topology_factors.panic_impulse, 0.0, 1.0
                )
        mechanism_factors = None
        if flags["dynamic_mechanism_design"]:
            lagged_shortage_cn = np.clip(1.0 - np.min(s["stocks_cn"], axis=1), 0.0, 1.0)
            lagged_shortage_def = np.clip(1.0 - np.min(s["stocks_def"], axis=1), 0.0, 1.0)
            lagged_shortage = np.column_stack((
                lagged_shortage_cn, lagged_shortage_def,
                0.35 * lagged_shortage_def, 0.40 * lagged_shortage_def,
            ))
            lagged_damage = np.clip(
                s["actor_homeland_damage"]
                + 0.12 * np.mean(s["actor_collapse"], axis=2), 0.0, 1.0
            )
            mechanism_factors = mechanism_system.update(
                month=month,
                actor_support=s["actor_support"],
                actor_damage=lagged_damage,
                actor_shortage=lagged_shortage,
                actor_output=np.mean(s["actor_sector_output"], axis=2),
                financial_capacity=s["actor_financial_capacity"],
                debt_stress=s["actor_debt_stress"],
                leader_continuity=s["leader_continuity"],
                political_resilience=s["actor_res"][:, :, 1],
                conflict_regime=s["conflict_regime"],
                regime_duration=s["regime_duration"],
                land_control=s["land"],
                proxy_capacity=s["taiwan_proxy_capacity"],
                proxy_leakage=s["taiwan_proxy_leakage"],
                trade_loss=trade_volume_loss,
                prior_blockade=np.clip(1.0 - s["sea"], 0.0, 1.0),
                us_ceiling=us_ceiling,
                japan_ceiling=jp_ceiling,
            )
            us = np.minimum(us_ceiling, us * mechanism_factors.alliance_multiplier[:, 0])
            jp = np.minimum(jp_ceiling, jp * mechanism_factors.alliance_multiplier[:, 1])
            jp = np.minimum(jp_ceiling, np.maximum(jp, np.minimum(0.55 * us, jp_ceiling)))
            s["actor_support"] = np.clip(
                s["actor_support"] + mechanism_factors.information_support_shift,
                0.01, 0.99,
            )
            s["actor_panic"] = np.clip(
                s["actor_panic"] - mechanism_factors.panic_reduction, 0.0, 1.0
            )
            mechanism_transfer = (
                mechanism_factors.transfer / 12.0
                + mechanism_factors.insurance_transfer
            )
            s["actor_cumulative_economic_aid"] += mechanism_transfer
            s["actor_debt_stress"] = np.clip(
                s["actor_debt_stress"] - 0.80 * mechanism_transfer,
                0.0, 1.5,
            )
            relief = mechanism_factors.sanction_relief
            trade_volume_loss = trade_volume_loss * (1.0 - relief)
            trade_gdp_factor = trade_gdp_factor + relief[:, None] * (1.0 - trade_gdp_factor)
            trade_industry_factor = trade_industry_factor + relief[:, None] * (1.0 - trade_industry_factor)
        minor = cfg["minor"] * (us if cfg["delay"] <= 0 else sigmoid((month - cfg["delay"]) / 2.5))
        allied = np.clip(0.58 * us + 0.27 * jp + 0.10 + 0.05 * minor, 0.02, 1.0)
        coalition_weights = np.column_stack((np.ones(n), us, jp))
        coalition_weights /= coalition_weights.sum(axis=1, keepdims=True)
        if operations_system is not None:
            operational_opening = operations_system.current()
            s["stocks_cn"] = operational_opening.front_inventory[:, 0, :]
            s["stocks_def"] = np.sum(
                operational_opening.front_inventory[:, 1:, :]
                * coalition_weights[:, :, None], axis=1
            )
            s["actor_front_rear_integrity"] = operational_opening.echelon_integrity
            s["repair_cn"] = np.mean(operational_opening.repair_queue[:, 0, :], axis=1)
            s["repair_def"] = np.sum(
                np.mean(operational_opening.repair_queue[:, 1:, :], axis=2)
                * coalition_weights, axis=1
            )
            s["search_cn"] = operational_opening.search_effectiveness[:, 0]
            s["search_def"] = np.sum(
                operational_opening.search_effectiveness[:, 1:] * coalition_weights,
                axis=1,
            )
        if hybrid_factors is not None:
            s["actor_front_rear_integrity"] = np.clip(
                s["actor_front_rear_integrity"] * hybrid_factors.logistics[:, :, None]
                * (
                    topology_factors.logistics[:, :, None]
                    if topology_factors is not None else 1.0
                ),
                0.02, 1.0,
            )
        if pnt_system is not None:
            s["pnt_cn"] = pnt_system.integrity[:, 0]
            s["pnt_def"] = np.sum(
                pnt_system.integrity[:, 1:] * coalition_weights, axis=1
            )
        exhaustion = np.clip(
            1.0 - 0.50 * (np.mean(s["stocks_cn"], axis=1) + np.mean(s["stocks_def"], axis=1)),
            0.0, 1.0,
        )
        stalemate = 1.0 - np.clip(np.abs(s["air"] - 0.5) + np.abs(s["sea"] - 0.5), 0.0, 1.0)
        regime = s["conflict_regime"]
        duration = s["regime_duration"]
        u_regime = rng.random(n)
        next_regime = regime.copy()
        high = regime == 0
        negotiation_lag = np.mean(s["actor_strategy_controls"][:, :, 4], axis=1)
        p_exit_high = np.where(
            duration >= 6,
            0.002 + 0.012 * exhaustion * stalemate + 0.008 * negotiation_lag,
            0.0,
        )
        entry_multiplier = (
            mechanism_factors.ceasefire_entry_multiplier
            if mechanism_factors is not None else np.ones(n)
        )
        break_multiplier = (
            mechanism_factors.ceasefire_break_multiplier
            if mechanism_factors is not None else np.ones(n)
        )
        settlement_multiplier = (
            mechanism_factors.settlement_multiplier
            if mechanism_factors is not None else np.ones(n)
        )
        p_exit_high = np.clip(p_exit_high * entry_multiplier, 0.0, 0.20)
        next_regime[high & (u_regime < 0.70 * p_exit_high)] = 1
        next_regime[high & (u_regime >= 0.70 * p_exit_high) & (u_regime < p_exit_high)] = 2
        low = regime == 1
        p_low_break = np.clip(0.004 * break_multiplier, 0.0, 0.08)
        p_low_ceasefire = np.clip((0.006 + 0.006 * exhaustion) * entry_multiplier, 0.0, 0.12)
        next_regime[low & (u_regime < p_low_break)] = 0
        next_regime[low & (u_regime >= p_low_break) & (u_regime < p_low_break + p_low_ceasefire)] = 2
        ceasefire = regime == 2
        p_cf_break = np.clip(0.002 * break_multiplier, 0.0, 0.08)
        p_cf_freeze = np.clip(0.010 * entry_multiplier, 0.0, 0.12)
        next_regime[ceasefire & (u_regime < p_cf_break)] = 0
        next_regime[ceasefire & (u_regime >= p_cf_break) & (u_regime < p_cf_break + p_cf_freeze)] = 3
        frozen = regime == 3
        p_frozen_break = np.clip(0.0015 * break_multiplier, 0.0, 0.06)
        p_frozen_rearm = np.full(n, 0.008)
        p_frozen_settle = np.clip(0.001 * settlement_multiplier, 0.0, 0.04)
        next_regime[frozen & (u_regime < p_frozen_break)] = 0
        next_regime[frozen & (u_regime >= p_frozen_break) & (u_regime < p_frozen_break + p_frozen_rearm)] = 4
        next_regime[frozen & (u_regime >= p_frozen_break + p_frozen_rearm) & (u_regime < p_frozen_break + p_frozen_rearm + p_frozen_settle)] = 5
        rearm = regime == 4
        p_rearm_break = np.clip(0.004 * break_multiplier, 0.0, 0.08)
        p_rearm_freeze = np.clip(0.006 * entry_multiplier, 0.0, 0.10)
        p_rearm_settle = np.clip(0.0008 * settlement_multiplier, 0.0, 0.04)
        next_regime[rearm & (u_regime < p_rearm_break)] = 0
        next_regime[rearm & (u_regime >= p_rearm_break) & (u_regime < p_rearm_break + p_rearm_freeze)] = 3
        next_regime[rearm & (u_regime >= p_rearm_break + p_rearm_freeze) & (u_regime < p_rearm_break + p_rearm_freeze + p_rearm_settle)] = 5
        new_settlement = (next_regime == 5) & (regime != 5)
        pre_balance = 0.36 * (s["air"] - 0.5) + 0.36 * (s["sea"] - 0.5) + 0.28 * (s["land"] - 0.2)
        p_unification_settlement = sigmoid(
            -3.6 + 5.0 * s["land"] + 2.2 * pre_balance
            + 0.8 * (s["cog_cn"] - s["cog_def"])
        )
        settlement_draw = rng.random(n)
        s["settlement_outcome"][new_settlement] = np.where(
            settlement_draw[new_settlement] < p_unification_settlement[new_settlement], 2, 1
        )
        s["regime_duration"] = np.where(next_regime == regime, duration + 1, 0)
        s["conflict_regime"] = next_regime
        intensity = regime_intensity[next_regime]
        common_probability = np.clip(
            0.0065 * scenario["common_shock"] * (0.15 + 0.85 * intensity)
            * tail_common_multiplier * (1.0 + 0.22 * common_hawkes_memory),
            0.0, 0.45,
        )
        common = rng.random(n) < common_probability
        cn_probability = np.clip(
            0.0045 * scenario["china_shock"] * (0.10 + 0.90 * intensity)
            * tail_actor_multiplier[:, 0] * (1.0 + 0.20 * actor_hawkes_memory[:, 0]),
            0.0, 0.40,
        )
        def_multiplier = np.mean(tail_actor_multiplier[:, 1:], axis=1)
        def_probability = np.clip(
            0.0045 * scenario["def_shock"] * (0.10 + 0.90 * intensity)
            * def_multiplier * (1.0 + 0.20 * np.mean(actor_hawkes_memory[:, 1:], axis=1)),
            0.0, 0.40,
        )
        cn_event = rng.random(n) < cn_probability
        def_event = rng.random(n) < def_probability
        u_common = np.clip(rng.random(n), 1e-10, 1.0 - 1e-10)
        u_cn = np.clip(rng.random(n), 1e-10, 1.0 - 1e-10)
        u_def = np.clip(rng.random(n), 1e-10, 1.0 - 1e-10)
        common_mark = np.where(
            common, 0.08 + 0.10 / 0.24 * ((1.0 - u_common) ** (-0.24) - 1.0), 0.0
        )
        cn_mark = np.where(
            cn_event, 0.08 + 0.13 / 0.22 * ((1.0 - u_cn) ** (-0.22) - 1.0), 0.0
        )
        def_mark = np.where(
            def_event, 0.08 + 0.13 / 0.22 * ((1.0 - u_def) ** (-0.22) - 1.0), 0.0
        )
        common_hawkes_memory = 0.72 * common_hawkes_memory + common
        actor_hawkes_memory *= 0.72
        actor_hawkes_memory[:, 0] += cn_event
        actor_hawkes_memory[:, 1:] += def_event[:, None]
        china_spill_signal = s["actor_homeland_damage"][:, 0]
        homeland_base = np.column_stack([
            np.full(n, 0.0004),
            np.full(n, 0.0020),
            0.00003 + 0.00010 * china_spill_signal,
            0.00050 + 0.00150 * china_spill_signal,
        ])
        homeland_probability = np.clip(
            intensity[:, None] * homeland_base * tail_actor_multiplier
            * (1.0 + 0.16 * actor_hawkes_memory), 0.0, 0.35
        )
        homeland_event = rng.random((n, 4)) < homeland_probability
        u_homeland = np.clip(rng.random((n, 4)), 1e-10, 1.0 - 1e-10)
        homeland_mark = np.where(
            homeland_event,
            0.018 + 0.035 / 0.20 * ((1.0 - u_homeland) ** (-0.20) - 1.0),
            0.0,
        )
        island_base = np.array([0.0015, 0.0040, 0.0010, 0.0030])[None, :]
        island_probability = np.clip(
            intensity[:, None] * island_base * tail_actor_multiplier
            * (1.0 + 0.14 * actor_hawkes_memory), 0.0, 0.40
        )
        island_event = rng.random((n, 4)) < island_probability
        u_island = np.clip(rng.random((n, 4)), 1e-10, 1.0 - 1e-10)
        island_mark = np.where(
            island_event,
            0.022 + 0.045 / 0.18 * ((1.0 - u_island) ** (-0.18) - 1.0),
            0.0,
        )
        actor_hawkes_memory += homeland_event + island_event
        if hybrid_system is not None:
            hybrid_system.apply_damage_reset(
                s, homeland_event, homeland_mark, island_event, island_mark
            )
        else:
            s["actor_homeland_damage"] = np.clip(
                0.95 * s["actor_homeland_damage"] + homeland_mark, 0.0, 0.35
            )
            s["actor_island_chain_damage"] = np.clip(
                0.93 * s["actor_island_chain_damage"] + island_mark, 0.0, 0.55
            )
            s["actor_backlash"] = np.clip(
                0.975 * s["actor_backlash"]
                + 1.6 * homeland_mark * (1.0 - s["actor_panic"]),
                0.0, 1.0,
            )
        leader_attempt_hazard = np.clip(
            intensity[:, None] * (
                0.00015 + 0.0018 * (1.0 - s["actor_command"][:, :, 0])
                + 0.0012 * s["actor_homeland_damage"]
                + 0.0008 * s["actor_island_chain_damage"]
            ),
            0.0, 0.015,
        )
        leader_attempt = rng.random((n, 4)) < leader_attempt_hazard
        intercept_probability = np.clip(
            0.38 + 0.42 * s["actor_command"][:, :, 0]
            + 0.16 * s["actor_network"] - 0.22 * s["actor_panic"],
            0.08, 0.96,
        )
        leader_disruption = leader_attempt & (
            rng.random((n, 4)) > intercept_probability
        )
        leader_severity = rng.uniform(0.18, 0.62, (n, 4))
        if hybrid_system is not None:
            hybrid_system.apply_leader_reset(
                s, leader_attempt, leader_disruption, leader_severity
            )
        else:
            s["leader_continuity"] = np.where(
                leader_disruption,
                np.maximum(0.12, s["leader_continuity"] * (1.0 - leader_severity)),
                s["leader_continuity"],
            )
            replacement_rate = np.array([0.045, 0.070, 0.055, 0.060])[None, :]
            s["leader_continuity"] = np.clip(
                s["leader_continuity"]
                + replacement_rate * (1.0 - s["leader_continuity"])
                * s["actor_res"][:, :, 1],
                0.10, 1.0,
            )
            s["leader_disruption_count"] += leader_disruption
            s["actor_command"] *= (
                0.985 + 0.015 * s["leader_continuity"]
            )[:, :, None]
            s["actor_panic"] = np.clip(
                s["actor_panic"] + 0.12 * leader_disruption
                + 0.035 * leader_attempt * (~leader_disruption), 0.0, 1.0
            )
            s["actor_support"] = np.clip(
                s["actor_support"] - 0.08 * leader_disruption
                + 0.025 * leader_attempt * (~leader_disruption), 0.01, 0.99
            )
        s["actor_support"] = np.clip(s["actor_support"] + 0.035 * s["actor_backlash"], 0.01, 0.99)
        s["actor_norm"] = np.clip(s["actor_norm"] + 0.020 * s["actor_backlash"], 0.0, 1.0)
        s["cog_cn"] = np.clip(s["cog_cn"] + 0.020 * s["actor_backlash"][:, 0], 0.04, 0.96)
        s["cog_def"] = np.clip(
            s["cog_def"] + 0.020 * s["actor_backlash"][:, 1]
            + 0.008 * s["actor_backlash"][:, 2] + 0.010 * s["actor_backlash"][:, 3],
            0.04, 0.96,
        )

        if strategy_system is not None:
            preliminary_balance = (
                0.38 * (s["air"] - 0.5) + 0.38 * (s["sea"] - 0.5)
                + 0.24 * (s["land"] - 0.2)
            )
            game_progress = np.column_stack((
                preliminary_balance, -preliminary_balance,
                -us * preliminary_balance, -jp * preliminary_balance,
            ))
            defender_stock = np.mean(s["stocks_def"], axis=1)
            game_stocks = np.column_stack((
                np.mean(s["stocks_cn"], axis=1), defender_stock,
                defender_stock, defender_stock,
            ))
            strategy_factors = strategy_system.update(
                progress=game_progress,
                stocks=game_stocks,
                damage=np.clip(
                    s["actor_homeland_damage"] + s["actor_island_chain_damage"],
                    0.0, 1.0,
                ),
                economic_capacity=np.mean(s["actor_sector_output"], axis=2),
                support=s["actor_support"],
                command=np.prod(
                    np.clip(s["actor_command"], 0.03, 1.0), axis=2
                ) ** (1.0 / 3.0),
                intervention=np.column_stack((np.ones(n), np.ones(n), us, jp)),
            )
            s["actor_strategy_controls"] = strategy_factors.controls

        command_eff = np.prod(np.clip(s["actor_command"], 0.03, 1.0), axis=2) ** (1.0 / 3.0)
        if hybrid_factors is not None:
            command_eff = np.clip(command_eff * hybrid_factors.command, 0.02, 1.0)
        if topology_factors is not None:
            command_eff = np.clip(command_eff * topology_factors.command, 0.02, 1.0)
        geometry_factors = None
        if flags["geometric_multiscale_coupling"] and geometry_system is not None:
            geometry_factors = geometry_system.update(
                front_rear_integrity=s["actor_front_rear_integrity"],
                sector_output=s["actor_sector_output"],
                financial_capacity=s["actor_financial_capacity"],
                command=command_eff,
                energy_logistics=(
                    energy_opening.logistics
                    if energy_opening is not None else np.ones((n, 4))
                ),
            )
            command_eff = np.clip(
                command_eff * geometry_factors.combat_factor, 0.02, 1.0
            )
        front_integrity = s["actor_front_rear_integrity"][:, :, 0]
        transit_integrity = s["actor_front_rear_integrity"][:, :, 1]
        depth_integrity = s["actor_front_rear_integrity"][:, :, 2]
        operational_control = np.clip(
            0.70 + 0.80 * np.sum(s["actor_strategy_controls"][:, :, :2], axis=2),
            0.65, 1.25,
        )
        tempo_cn = np.clip((0.34 + 0.42 * s["web_cn"] * s["ooda_cn"] + 0.18 * s["cog_cn"]) * command_eff[:, 0] * s["actor_manpower_eff"][:, 0] * front_integrity[:, 0] * operational_control[:, 0], 0.10, 0.96)
        tempo_tw = np.clip((0.25 + 0.30 * s["web_def"] * s["ooda_def"] + 0.18 * s["cog_def"]) * command_eff[:, 1] * s["actor_manpower_eff"][:, 1] * front_integrity[:, 1] * operational_control[:, 1], 0.06, 0.84)
        tempo_us = us * np.clip((0.18 + 0.46 * s["web_def"] * s["ooda_def"] + 0.20 * s["csg"]) * command_eff[:, 2] * s["actor_manpower_eff"][:, 2] * front_integrity[:, 2] * operational_control[:, 2], 0.05, 0.94)
        tempo_jp = jp * np.clip((0.16 + 0.38 * s["web_def"] * s["ooda_def"] + 0.18 * s["cog_def"]) * command_eff[:, 3] * s["actor_manpower_eff"][:, 3] * front_integrity[:, 3] * operational_control[:, 3], 0.05, 0.90)
        tempo_minor = minor * np.clip(0.12 + 0.30 * s["web_def"] * s["ooda_def"], 0.03, 0.72)
        tempo_def = np.clip(0.52 * tempo_tw + 0.34 * tempo_us + 0.24 * tempo_jp + 0.08 * tempo_minor, 0.08, 0.96)
        strike_cn = 0.0045 * intensity * tempo_cn * np.mean(s["stocks_cn"][:, :2], axis=1)
        strike_def = 0.0042 * intensity * tempo_def * np.mean(s["stocks_def"][:, :2], axis=1)

        damage_cn = strike_def[:, None] * np.array([0.75, 0.82, 0.92, 0.62, 0.52, 0.66])[None, :]
        damage_def = strike_cn[:, None] * np.array([0.82, 0.75, 0.92, 0.60, 0.50, 0.68])[None, :]
        damage_cn += (cn_mark + 0.55 * common_mark)[:, None] * np.array([0.22, 0.22, 0.30, 0.20, 0.16, 0.18])[None, :]
        damage_def += (def_mark + 0.55 * common_mark)[:, None] * np.array([0.22, 0.22, 0.30, 0.20, 0.16, 0.18])[None, :]

        coalition_repair_control = np.sum(
            s["actor_strategy_controls"][:, 1:, 3] * coalition_weights, axis=1
        )
        repair_cap_cn = 0.020 * (0.65 + 1.40 * s["actor_strategy_controls"][:, 0, 3]) * s["nodes_cn"][:, 4] * s["stocks_cn"][:, 2] / (1.0 + s["repair_cn"])
        repair_cap_def = 0.020 * (0.65 + 1.40 * coalition_repair_control) * s["nodes_def"][:, 4] * s["stocks_def"][:, 2] * cfg["def_log"] / (1.0 + s["repair_def"])
        s["repair_cn"] = np.clip(s["repair_cn"] + damage_cn.mean(axis=1) - repair_cap_cn, 0.0, 2.5)
        s["repair_def"] = np.clip(s["repair_def"] + damage_def.mean(axis=1) - repair_cap_def, 0.0, 2.5)
        s["nodes_cn"] = np.clip(s["nodes_cn"] + repair_cap_cn[:, None] * 0.55 - damage_cn, 0.03, 0.995)
        s["nodes_def"] = np.clip(s["nodes_def"] + repair_cap_def[:, None] * 0.55 - damage_def, 0.03, 0.995)

        pnt_hit_cn = 0.018 * cn_event + 0.012 * common
        pnt_hit_def = 0.018 * def_event + 0.012 * common
        if pnt_system is not None:
            pnt_damage = np.column_stack((
                pnt_hit_cn,
                pnt_hit_def,
                us * (0.45 * pnt_hit_def + 0.08 * homeland_mark[:, 2]),
                jp * (0.55 * pnt_hit_def + 0.10 * homeland_mark[:, 3]),
            ))
            pnt_infrastructure = np.column_stack((
                s["nodes_cn"][:, 3], s["nodes_def"][:, 3],
                np.clip(0.92 - 0.50 * homeland_mark[:, 2], 0.05, 1.0),
                np.clip(0.90 - 0.55 * homeland_mark[:, 3], 0.05, 1.0),
            ))
            pnt_factors = pnt_system.update(
                infrastructure=pnt_infrastructure,
                cyber_damage=pnt_damage,
                command=command_eff,
                terrain_mask=s["sky"],
            )
            s["pnt_cn"] = pnt_factors.integrity[:, 0]
            s["pnt_def"] = np.sum(
                pnt_factors.integrity[:, 1:] * coalition_weights, axis=1
            )
        else:
            s["pnt_cn"] = np.clip(s["pnt_cn"] + 0.018 * (0.80 - s["pnt_cn"]) - pnt_hit_cn, 0.12, 0.98)
            s["pnt_def"] = np.clip(s["pnt_def"] + 0.018 * (0.82 - s["pnt_def"]) - pnt_hit_def, 0.12, 0.99)
        target_web_cn = 0.45 * s["nodes_cn"][:, 2] + 0.28 * s["pnt_cn"] + 0.17 * s["nodes_cn"][:, 3] + 0.10 * s["nodes_cn"][:, 5]
        target_web_def = 0.45 * s["nodes_def"][:, 2] + 0.28 * s["pnt_def"] + 0.17 * s["nodes_def"][:, 3] + 0.10 * s["nodes_def"][:, 5]
        s["web_cn"] = np.clip(s["web_cn"] + 0.12 * (target_web_cn - s["web_cn"]), 0.05, 0.99)
        s["web_def"] = np.clip(s["web_def"] + 0.12 * (target_web_def - s["web_def"]), 0.05, 0.99)
        queue_cn = np.clip(tempo_cn - s["nodes_cn"][:, 2], 0.0, 1.0)
        queue_def = np.clip(tempo_def - s["nodes_def"][:, 2], 0.0, 1.0)
        s["ooda_cn"] = np.clip(0.96 * s["ooda_cn"] + 0.04 * s["web_cn"] - 0.05 * queue_cn, 0.08, 0.98)
        s["ooda_def"] = np.clip(0.96 * s["ooda_def"] + 0.04 * s["web_def"] - 0.05 * queue_def, 0.08, 0.98)

        shortage_signal_cn = np.clip(0.72 - np.mean(s["stocks_cn"], axis=1), 0.0, 0.72)
        shortage_signal_def = np.clip(0.72 - np.mean(s["stocks_def"], axis=1), 0.0, 0.72)
        s["mobil_cn"] = np.clip(0.94 * s["mobil_cn"] + 0.10 * shortage_signal_cn, 0.0, 1.0)
        s["mobil_def"] = np.clip(0.94 * s["mobil_def"] + 0.10 * shortage_signal_def, 0.0, 1.0)
        coalition_collapse = np.sum(
            s["actor_collapse"][:, 1:, :] * coalition_weights[:, :, None], axis=1
        )
        actor_output_prior = 0.55 * s["actor_sector_output"][:, :, 0] + 0.45 * np.mean(
            s["actor_sector_output"][:, :, 1:], axis=2
        )
        actor_output_prior *= depth_integrity
        coalition_output_prior = np.sum(actor_output_prior[:, 1:] * coalition_weights, axis=1)
        actor_supply_route = np.minimum(np.minimum(front_integrity, transit_integrity), depth_integrity)
        coalition_front_prior = np.sum(actor_supply_route[:, 1:] * coalition_weights, axis=1)
        collapse_penalty_cn = (
            1.0 - 0.24 * s["actor_collapse"][:, 0, 0]
            - 0.12 * s["actor_collapse"][:, 0, 1]
            - 0.22 * s["actor_collapse"][:, 0, 3]
        )
        collapse_penalty_def = (
            1.0 - 0.24 * coalition_collapse[:, 0]
            - 0.12 * coalition_collapse[:, 1]
            - 0.22 * coalition_collapse[:, 3]
        )
        production_cn = (
            0.0155 + 0.012 * s["mobil_cn"]
        ) * (0.70 + 1.40 * s["actor_strategy_controls"][:, 0, 2]) * s["china_growth"] * s["military_anchor"] * s["nodes_cn"][:, 4] * scenario["china_supply"] * collapse_penalty_cn * actor_output_prior[:, 0] * actor_supply_route[:, 0] * trade_industry_factor[:, 0]
        trade_industry_def = np.sum(
            trade_industry_factor[:, 1:] * coalition_weights, axis=1
        )
        production_def = (
            0.0145 + 0.012 * s["mobil_def"]
        ) * (0.70 + 1.40 * np.sum(s["actor_strategy_controls"][:, 1:, 2] * coalition_weights, axis=1)) * (0.45 + 0.55 * allied) * s["nodes_def"][:, 4] * cfg["def_log"] * scenario["def_supply"] * collapse_penalty_def * coalition_output_prior * coalition_front_prior * trade_industry_def
        if mechanism_factors is not None:
            production_cn *= mechanism_factors.procurement_factor[:, 0]
            production_def *= np.sum(
                mechanism_factors.procurement_factor[:, 1:] * coalition_weights,
                axis=1,
            )
        if energy_opening is not None:
            production_cn *= energy_opening.production[:, 0] * energy_opening.logistics[:, 0]
            coalition_energy_production = np.sum(
                energy_opening.production[:, 1:] * coalition_weights, axis=1
            )
            coalition_energy_logistics = np.sum(
                energy_opening.logistics[:, 1:] * coalition_weights, axis=1
            )
            production_def *= coalition_energy_production * coalition_energy_logistics
        consumption_cn = (0.004 + intensity * (0.008 + 0.014 * tempo_cn))[:, None] * np.array([1.15, 0.90, 0.55])[None, :]
        consumption_def = (0.004 + intensity * (0.007 + 0.014 * tempo_def))[:, None] * np.array([1.12, 0.92, 0.58])[None, :]
        if operations_system is not None:
            actor_production = np.column_stack((
                production_cn,
                production_def * coalition_weights[:, 0],
                production_def * coalition_weights[:, 1],
                production_def * coalition_weights[:, 2],
            ))
            base_def_consumption = np.mean(consumption_def, axis=1)
            actor_consumption = np.column_stack((
                np.mean(consumption_cn, axis=1),
                base_def_consumption * coalition_weights[:, 0],
                base_def_consumption * coalition_weights[:, 1],
                base_def_consumption * coalition_weights[:, 2],
            ))
            actor_operational_damage = np.column_stack((
                np.mean(damage_cn, axis=1), np.mean(damage_def, axis=1),
                us * (0.35 * np.mean(damage_def, axis=1) + homeland_mark[:, 2]),
                jp * (0.45 * np.mean(damage_def, axis=1) + homeland_mark[:, 3]),
            ))
            actor_interdiction = np.column_stack((
                np.clip(1.0 - s["sea"], 0.0, 1.0),
                np.clip(s["sea"], 0.0, 1.0),
                np.clip(0.28 * s["sea"] + homeland_mark[:, 2], 0.0, 1.0),
                np.clip(0.36 * s["sea"] + homeland_mark[:, 3], 0.0, 1.0),
            ))
            pnt_by_actor = (
                pnt_system.integrity if pnt_system is not None
                else np.column_stack((s["pnt_cn"], s["pnt_def"], s["pnt_def"], s["pnt_def"]))
            )
            operational_factors = operations_system.update(
                production=actor_production,
                consumption=actor_consumption,
                damage=actor_operational_damage,
                interdiction=actor_interdiction,
                command=command_eff,
                pnt=pnt_by_actor,
                financial_capacity=s["actor_financial_capacity"],
            )
            s["stocks_cn"] = operational_factors.front_inventory[:, 0, :]
            s["stocks_def"] = np.sum(
                operational_factors.front_inventory[:, 1:, :]
                * coalition_weights[:, :, None], axis=1
            )
            s["search_cn"] = operational_factors.search_effectiveness[:, 0]
            s["search_def"] = np.sum(
                operational_factors.search_effectiveness[:, 1:] * coalition_weights,
                axis=1,
            )
        else:
            s["stocks_cn"] = np.clip(s["stocks_cn"] + production_cn[:, None] - consumption_cn, 0.01, 1.35)
            s["stocks_def"] = np.clip(s["stocks_def"] + production_def[:, None] - consumption_def, 0.01, 1.35)
            search_target_cn = np.clip(0.22 + 0.42 * s["web_cn"] * s["pnt_cn"] * s["nodes_cn"][:, 5], 0.05, 0.90)
            search_target_tw = 0.18 + 0.24 * s["web_def"] * s["pnt_def"] * s["nodes_def"][:, 5]
            search_target_us = us * (0.16 + 0.38 * s["web_def"] * s["pnt_def"] * s["nodes_def"][:, 5])
            search_target_jp = jp * (0.15 + 0.34 * s["web_def"] * s["pnt_def"] * s["nodes_def"][:, 5])
            search_target_def = np.clip(search_target_tw + 0.55 * search_target_us + 0.50 * search_target_jp, 0.05, 0.92)
            s["search_cn"] = np.clip(0.90 * s["search_cn"] + 0.10 * search_target_cn + rng.normal(0, 0.012, n), 0.03, 0.95)
            s["search_def"] = np.clip(0.90 * s["search_def"] + 0.10 * search_target_def + rng.normal(0, 0.012, n), 0.03, 0.95)
        underwater = np.clip(s["search_cn"] - s["search_def"], -0.8, 0.8)

        air_cn = 0.58 * s["web_cn"] * s["ooda_cn"] * s["stocks_cn"][:, 0]
        air_tw = 0.31 * s["web_def"] * s["ooda_def"] * s["stocks_def"][:, 0]
        air_us = 0.31 * us * s["web_def"] * s["ooda_def"] * s["stocks_def"][:, 0]
        air_jp = 0.20 * jp * s["web_def"] * s["ooda_def"] * s["stocks_def"][:, 0]
        air_minor = 0.07 * minor * s["web_def"] * s["ooda_def"] * s["stocks_def"][:, 0]
        air_pressure = air_cn - air_tw - air_us - air_jp - air_minor
        sea_cn = 0.50 * s["web_cn"] * s["stocks_cn"][:, 0] + 0.20 * underwater
        sea_tw = 0.18 * s["web_def"] * s["stocks_def"][:, 0]
        sea_us = 0.30 * us * s["web_def"] * s["stocks_def"][:, 0] + 0.16 * us * s["csg"]
        sea_jp = 0.21 * jp * s["web_def"] * s["stocks_def"][:, 0] + 0.08 * jp * s["csg"]
        sea_minor = 0.05 * minor * s["web_def"] * s["stocks_def"][:, 0]
        sea_pressure = sea_cn - sea_tw - sea_us - sea_jp - sea_minor
        s["air"] = np.clip(s["air"] + intensity * 0.020 * air_pressure + rng.normal(0, 0.004 + 0.003 * intensity, n), 0.02, 0.98)
        s["sea"] = np.clip(s["sea"] + intensity * 0.018 * sea_pressure + rng.normal(0, 0.004 + 0.003 * intensity, n), 0.02, 0.98)
        s["csg"] = np.clip(
            s["csg"] + 0.012 * (cfg["csg"] * s["nodes_def"][:, 0] - s["csg"])
            - 0.008 * tempo_cn * s["web_cn"] * (1.0 - s["sea"]), 0.02, 0.98
        )

        csi = sigmoid(
            2.2 * (s["sea"] - 0.5) + 0.55 * (s["nodes_cn"][:, 0] - s["nodes_def"][:, 0])
            + 0.45 * underwater + 0.42 * (s["stocks_cn"][:, 1] - s["stocks_def"][:, 1])
        )
        blockade = np.clip(1.0 - csi + 0.16 * allied * s["csg"], 0.0, 1.0)
        if mechanism_factors is not None:
            blockade *= 1.0 - mechanism_factors.sanction_relief
        exit_capacity = np.clip(0.64 - 0.48 * s["mountain_share"] + 0.28 * s["nodes_cn"][:, 5], 0.10, 0.76)
        beach_service = np.clip(
            0.10 * s["air"] * s["sea"] * s["web_cn"] * np.min(s["stocks_cn"][:, :2], axis=1)
            * exit_capacity * (1.0 - 0.62 * allied * s["web_def"]), 0.0, 0.10
        )
        if flags["principal_agent"]:
            agent = s["taiwan_agent_type_share"]
            available_pool = np.clip(0.75 * s["land"] - np.sum(agent, axis=1), 0.0, 0.75)
            collaboration_incentive = np.clip(
                0.45 * s["land"] + 0.28 * s["cog_cn"]
                + 0.18 * (1.0 - s["actor_backlash"][:, 1]), 0.0, 1.0
            )
            recruitment = 0.010 * available_pool * collaboration_incentive
            if mechanism_factors is not None:
                recruitment *= mechanism_factors.governance_recruitment_multiplier
            recruit_mix = np.stack((
                0.34 + 0.22 * s["cog_cn"],
                0.33 + 0.18 * (1.0 - s["actor_support"][:, 1]),
                0.20 + 0.20 * s["land"],
                0.13 + 0.10 * s["actor_backlash"][:, 1],
            ), axis=1)
            recruit_mix /= np.maximum(np.sum(recruit_mix, axis=1, keepdims=True), 1e-12)
            agent += recruitment[:, None] * recruit_mix
            # Geography and communications isolation make sustained double
            # contact harder than ordinary opportunistic compliance.
            isolation = np.clip(
                blockade * (1.0 - s["actor_network"][:, 1]), 0.0, 1.0
            )
            loyal_to_opportunist = 0.0025 * agent[:, 0] * s["actor_backlash"][:, 1]
            opportunist_to_loyal = 0.0035 * agent[:, 1] * collaboration_incentive
            coerced_exit = 0.0050 * agent[:, 2] * (1.0 - s["land"])
            dual_exit = 0.0045 * agent[:, 3] * isolation
            dual_recruit = 0.0020 * agent[:, 1] * s["actor_backlash"][:, 1] * (1.0 - isolation)
            agent[:, 0] += opportunist_to_loyal - loyal_to_opportunist
            agent[:, 1] += loyal_to_opportunist - opportunist_to_loyal - dual_recruit
            agent[:, 2] -= coerced_exit
            agent[:, 3] += dual_recruit - dual_exit
            agent = np.clip(agent, 0.0, 0.75)
            total_agent = np.sum(agent, axis=1)
            ceiling_scale = np.minimum(
                1.0, 0.75 * np.maximum(s["land"], 1e-6) / np.maximum(total_agent, 1e-12)
            )
            agent *= ceiling_scale[:, None]
            s["taiwan_agent_type_share"] = agent
            capacity_weights = np.array([0.95, 0.58, 0.32, 0.18])
            leakage_weights = np.array([0.03, 0.25, 0.45, 0.72])
            s["taiwan_proxy_capacity"] = np.clip(agent @ capacity_weights, 0.0, 0.75)
            s["taiwan_proxy_leakage"] = np.clip(
                np.sum(agent * leakage_weights[None, :], axis=1)
                / np.maximum(np.sum(agent, axis=1), 1e-9),
                0.0, 0.90,
            )
            if mechanism_factors is not None:
                s["taiwan_proxy_leakage"] = np.clip(
                    s["taiwan_proxy_leakage"]
                    - mechanism_factors.governance_leakage_reduction,
                    0.0, 0.90,
                )
                s["actor_support"][:, 1] = np.clip(
                    s["actor_support"][:, 1]
                    + mechanism_factors.governance_trust_bonus,
                    0.01, 0.99,
                )
            effective_proxy = np.clip(
                s["taiwan_proxy_capacity"] * (1.0 - s["taiwan_proxy_leakage"]),
                0.0, 0.75,
            )
        else:
            effective_proxy = np.zeros(n)
        resistance_penalty = np.clip(
            1.0 - 0.45 * s["actor_backlash"][:, 1] - 0.18 * effective_proxy, 0.42, 1.0
        )
        land_growth = intensity * beach_service * (0.74 + 0.26 * s["cog_cn"]) * (1.0 - s["land"]) * resistance_penalty
        land_loss = intensity * 0.010 * allied * s["web_def"] * s["ooda_def"] * (0.35 + 0.65 * s["air"]) * s["land"]
        s["land"] = np.clip(s["land"] + land_growth - land_loss, 0.0, 0.995)

        observed_balance = 0.34 * (s["air"] - 0.5) + 0.34 * (s["sea"] - 0.5) + 0.32 * (s["land"] - 0.2)
        shortage_cn = 1.0 - np.min(s["stocks_cn"], axis=1)
        shortage_def = 1.0 - np.min(s["stocks_def"], axis=1)
        s["cog_cn"] = np.clip(
            s["cog_cn"] + scenario["cognitive"] * (0.010 * observed_balance - 0.006 * shortage_cn)
            + 0.004 * (0.58 - s["cog_cn"])
            + rng.normal(0, 0.0035, n), 0.04, 0.96
        )
        s["cog_def"] = np.clip(
            s["cog_def"] + scenario["cognitive"] * (-0.009 * observed_balance - 0.006 * shortage_def)
            + 0.004 * (0.58 - s["cog_def"])
            + rng.normal(0, 0.0035, n), 0.04, 0.96
        )

        # Four-party economic, political, social and military resilience drives DEDS collapse events.
        mean_node_cn = np.mean(s["nodes_cn"][:, :5], axis=1)
        mean_node_def = np.mean(s["nodes_def"][:, :5], axis=1)
        actor_shortage = np.stack([
            shortage_cn,
            shortage_def,
            shortage_def * (0.20 + 0.55 * us),
            shortage_def * (0.24 + 0.52 * jp),
        ], axis=1)
        trade_shortage = np.clip(
            0.48 * trade_chip_gap
            + 0.30 * (1.0 - trade_gdp_factor)
            + 0.22 * trade_volume_loss[:, None],
            0.0, 0.85,
        )
        actor_shortage = np.clip(
            1.0 - (1.0 - actor_shortage) * (1.0 - trade_shortage), 0.0, 1.0
        )
        if energy_opening is not None:
            actor_shortage = np.clip(
                1.0 - (1.0 - actor_shortage) * (1.0 - energy_opening.shortage),
                0.0, 1.0,
            )
        actor_damage = np.stack([
            np.mean(damage_cn, axis=1),
            np.mean(damage_def, axis=1),
            us * (0.38 * np.mean(damage_def, axis=1) + 0.15 * common_mark),
            jp * (0.52 * np.mean(damage_def, axis=1) + 0.18 * common_mark),
        ], axis=1) + 0.20 * homeland_mark + 0.30 * island_mark
        front_repair = 0.018 * np.mean(s["actor_sector_output"], axis=2) * s["actor_res"][:, :, 0]
        transit_repair = 0.016 * np.mean(s["actor_sector_output"], axis=2) * s["actor_res"][:, :, 0]
        depth_repair = 0.014 * np.mean(s["actor_sector_output"], axis=2) * s["actor_res"][:, :, 0]
        if operations_system is None:
            s["actor_front_rear_integrity"][:, :, 0] = np.clip(
                s["actor_front_rear_integrity"][:, :, 0] + front_repair
                - 0.16 * actor_damage - 0.22 * island_mark,
                0.03, 0.995,
            )
            s["actor_front_rear_integrity"][:, :, 1] = np.clip(
                s["actor_front_rear_integrity"][:, :, 1] + transit_repair
                - 0.13 * actor_damage - 0.12 * island_mark - 0.10 * homeland_mark,
                0.03, 0.995,
            )
            s["actor_front_rear_integrity"][:, :, 2] = np.clip(
                s["actor_front_rear_integrity"][:, :, 2] + depth_repair
                - 0.08 * actor_damage - 0.25 * homeland_mark,
                0.03, 0.995,
            )
        actor_mobil = np.stack([
            s["mobil_cn"], s["mobil_def"], us * s["mobil_def"], jp * s["mobil_def"]
        ], axis=1)
        if hybrid_factors is not None:
            actor_mobil = np.clip(actor_mobil * hybrid_factors.mobilization, 0.0, 1.0)
        if topology_factors is not None:
            actor_mobil = np.clip(actor_mobil * topology_factors.mobilization, 0.0, 1.0)
        if energy_system is not None:
            energy_system.update(
                month=month,
                actor_damage=actor_damage,
                actor_output=np.mean(s["actor_sector_output"], axis=2),
                actor_mobilization=actor_mobil,
                financial_capacity=s["actor_financial_capacity"],
                trade_loss=trade_volume_loss,
                blockade=blockade,
                sanction_relief=(
                    mechanism_factors.sanction_relief
                    if mechanism_factors is not None else np.zeros(n)
                ),
                us_level=us,
                japan_level=jp,
            )
        effective_trade_inflation = trade_inflation + (
            energy_opening.inflation_impulse
            if energy_opening is not None else 0.0
        )
        if financial_system is None:
            financial_production = np.ones((n, 4))
            financial_credit = np.ones((n, 4))
            financial_payment = np.ones((n, 4))
            financial_fiscal = np.ones((n, 4))
            financial_aid = np.ones((n, 4))
            financial_stress = np.zeros((n, 4))
            financial_inflation = effective_trade_inflation
        else:
            financial_factors = financial_system.update(
                month=month,
                actor_damage=actor_damage,
                actor_shortage=actor_shortage,
                actor_mobilization=actor_mobil,
                firm_distress=s["actor_firm_distress"],
                sector_output=s["actor_sector_output"],
                trade_gdp_factor=trade_gdp_factor,
                trade_industry_factor=trade_industry_factor,
                trade_inflation=effective_trade_inflation,
                trade_loss=trade_volume_loss,
                macro_trade_balance=macro_trade_balance,
                macro_government_balance=macro_government_balance,
                us_level=us,
                japan_level=jp,
            )
            financial_production = financial_factors.production
            financial_credit = financial_factors.credit
            financial_payment = financial_factors.payment
            financial_fiscal = financial_factors.fiscal
            financial_aid = financial_factors.aid
            financial_stress = financial_factors.stress
            financial_inflation = financial_factors.inflation
            s["actor_financial_capacity"] = np.clip(
                0.40 * financial_fiscal + 0.35 * financial_credit
                + 0.25 * financial_payment, 0.05, 1.0
            )
        progress_signal = np.stack([
            observed_balance, -observed_balance, -us * observed_balance, -jp * observed_balance
        ], axis=1)
        cognition_signal = np.stack([
            s["cog_cn"], s["cog_def"],
            np.clip(0.72 - 0.20 * us * shortage_def, 0.05, 0.95),
            np.clip(0.74 - 0.22 * jp * shortage_def, 0.05, 0.95),
        ], axis=1)
        if mfg_system is None:
            support_target = sigmoid(
                logit(np.clip(cognition_signal, 0.02, 0.98))
                + 0.90 * progress_signal + 0.55 * (s["actor_norm"] - 0.5)
                - 0.65 * s["actor_panic"]
            )
            s["actor_support"] = np.clip(
                0.90 * s["actor_support"] + 0.10 * support_target, 0.01, 0.99
            )
        service_exposure = np.array([1.15, 0.92, 0.88, 0.62, 0.78])[None, None, :]
        active_attrition = np.clip(0.060 * actor_damage[:, :, None] * service_exposure, 0.0, 0.08)
        s["actor_manpower"][:, :, 0, :] = np.clip(
            s["actor_manpower"][:, :, 0, :] * (1.0 - active_attrition), 0.01, 1.20
        )
        reserve_flow = np.minimum(
            s["actor_manpower"][:, :, 1, :],
            (0.010 + 0.035 * actor_mobil[:, :, None])
            * (0.35 + 0.65 * s["actor_support"][:, :, None])
            * s["actor_manpower"][:, :, 1, :],
        )
        skill_trainability = np.array([0.90, 0.52, 0.48, 0.28, 0.72])[None, None, :]
        civilian_intake = np.minimum(
            s["actor_manpower"][:, :, 2, :],
            0.006 * actor_mobil[:, :, None]
            * (0.25 + 0.75 * s["actor_support"][:, :, None])
            * skill_trainability * s["actor_manpower"][:, :, 2, :],
        )
        s["actor_manpower"][:, :, 1, :] -= reserve_flow
        s["actor_manpower"][:, :, 2, :] -= civilian_intake
        skill_draw_matrix = np.array([
            [0.62, 0.25, 0.10, 0.03],
            [0.15, 0.30, 0.38, 0.17],
            [0.10, 0.24, 0.42, 0.24],
            [0.03, 0.12, 0.38, 0.47],
            [0.25, 0.38, 0.27, 0.10],
        ])
        skill_draw = np.einsum("nav,vk->nak", civilian_intake, skill_draw_matrix)
        s["actor_labor_skill"] = np.clip(s["actor_labor_skill"] - 0.12 * skill_draw, 0.02, 1.10)
        training_entry = 0.004 * (0.35 + 0.65 * s["actor_support"][:, :, None]) * s["actor_labor_skill"][:, :, :3]
        s["actor_skill_training"] = np.clip(s["actor_skill_training"] + training_entry, 0.0, 0.50)
        skill_graduation_rate = np.array([0.10, 0.045, 0.018])[None, None, :]
        skill_graduates = skill_graduation_rate * s["actor_skill_training"]
        s["actor_skill_training"] -= skill_graduates
        s["actor_labor_skill"][:, :, 1:] = np.clip(
            s["actor_labor_skill"][:, :, 1:] + skill_graduates, 0.02, 1.10
        )
        s["actor_training"] = np.clip(s["actor_training"] + civilian_intake, 0.0, 1.5)
        graduation_rate = np.array([0.11, 0.055, 0.050, 0.030, 0.080])[None, None, :]
        graduates = graduation_rate * s["actor_training"] * (1.0 - 0.35 * s["actor_labor_fatigue"][:, :, None])
        s["actor_training"] -= graduates
        s["actor_manpower"][:, :, 0, :] = np.clip(
            s["actor_manpower"][:, :, 0, :] + 0.78 * reserve_flow + graduates, 0.01, 1.25
        )
        manpower_effectiveness = np.average(
            s["actor_manpower"][:, :, 0, :], axis=2,
            weights=np.array([0.24, 0.18, 0.20, 0.20, 0.18]),
        )
        s["actor_manpower_eff"] = np.clip(manpower_effectiveness, 0.03, 1.25)
        labor_crowding = np.clip(np.sum(civilian_intake, axis=2), 0.0, 0.15)
        s["actor_labor_fatigue"] = np.clip(
            s["actor_labor_fatigue"] + 0.025 * actor_mobil + 0.080 * labor_crowding
            + 0.020 * actor_damage - 0.040 * (1.0 - actor_mobil) * s["actor_labor_fatigue"],
            0.0, 0.90,
        )
        mobility_capacity = np.clip(
            s["actor_network"] * (1.0 - 0.55 * s["actor_panic"]), 0.02, 1.0
        )
        mismatch_target = np.clip(
            0.10 + 0.34 * actor_damage + 0.22 * labor_crowding
            + 0.18 * (1.0 - mobility_capacity) + 0.12 * actor_shortage,
            0.02, 0.85,
        )
        s["actor_labor_mismatch"] = np.clip(
            s["actor_labor_mismatch"]
            + 0.080 * (mismatch_target - s["actor_labor_mismatch"]) * mobility_capacity
            + 0.025 * actor_damage,
            0.0, 0.90,
        )
        firm_skill_requirements = np.array([
            [0.04, 0.16, 0.36, 0.44],
            [0.08, 0.28, 0.42, 0.22],
            [0.15, 0.42, 0.34, 0.09],
            [0.38, 0.42, 0.17, 0.03],
        ])
        substitution_discount = np.array([1.00, 0.82, 0.58, 0.22])
        skill_capacity = s["actor_labor_skill"] * substitution_discount[None, None, :]
        firm_skill_fill = np.clip(
            np.einsum("nak,fk->naf", skill_capacity, firm_skill_requirements), 0.05, 1.05
        )
        labor_efficiency = (
            1.0 - labor_crowding[:, :, None]
        ) * (1.0 - 0.45 * s["actor_labor_fatigue"][:, :, None]) * (
            1.0 - 0.35 * s["actor_labor_mismatch"][:, :, None]
        ) * firm_skill_fill
        military_orders = 0.006 + 0.025 * actor_mobil + 0.008 * actor_shortage
        civilian_orders = np.clip(0.015 * (1.0 - 0.60 * s["actor_inflation"]), 0.003, 0.020)
        order_demand = np.stack([military_orders, civilian_orders], axis=2) + 0.05 * s["actor_order_backlog"]
        transport_supply = 0.030 * s["actor_sector_output"]
        if flags["optimal_transport"]:
            transport_fill = entropy_transport_fulfillment(transport_supply, order_demand)
        else:
            aggregate_fill = np.clip(
                np.sum(transport_supply, axis=2) / np.maximum(np.sum(order_demand, axis=2), 1e-9),
                0.0,
                1.0,
            )
            transport_fill = np.repeat(aggregate_fill[:, :, None], 2, axis=2)
        deliveries = order_demand * transport_fill
        s["actor_order_backlog"] = np.clip(
            s["actor_order_backlog"]
            + np.stack([military_orders, civilian_orders], axis=2) - deliveries,
            0.0, 2.5,
        )
        order_reallocation = np.clip(
            s["actor_order_backlog"][:, :, 0] - s["actor_order_backlog"][:, :, 1],
            -1.0, 1.0,
        )
        war_labor_transfer = np.clip(
            0.020 * actor_mobil * (0.35 + 0.65 * s["actor_support"])
            + 0.006 * order_reallocation,
            0.0, 0.040,
        )
        actor_cog = np.stack([
            s["cog_cn"], s["cog_def"],
            np.clip(0.80 - 0.35 * us * shortage_def, 0.05, 0.95),
            np.clip(0.82 - 0.38 * jp * shortage_def, 0.05, 0.95),
        ], axis=1)
        supply_exposure = np.array([0.52, 0.82, 0.34, 0.58])[None, :, None]
        class_infection = np.array([0.010, 0.013, 0.016, 0.020])[None, None, :]
        class_recovery = np.array([0.052, 0.046, 0.040, 0.032])[None, None, :]
        infection_force = (
            intensity[:, None, None] * class_infection
            * s["actor_firm_distress"] * (1.0 - s["actor_firm_distress"])
            + intensity[:, None, None] * 0.008 * actor_shortage[:, :, None] * supply_exposure
            + 0.020 * actor_damage[:, :, None]
        )
        recovery_force = (class_recovery + 0.070 * (1.0 - intensity[:, None, None])) \
            * s["actor_firm_distress"] * np.clip(s["actor_sector_output"] + 0.20, 0.0, 1.20)
        s["actor_firm_distress"] = np.clip(
            s["actor_firm_distress"] + infection_force - recovery_force, 0.0, 1.0
        )
        factor_damage = np.stack([
            0.020 * actor_damage,
            0.014 * s["actor_panic"] + 0.010 * actor_damage,
            0.020 * actor_shortage + 0.012 * actor_damage,
        ], axis=2)[:, :, None, :]
        factor_recovery = np.array([0.016, 0.020, 0.018])[None, None, None, :]
        s["actor_firm_factors"] = np.clip(
            s["actor_firm_factors"]
            + factor_recovery * (1.0 - s["actor_firm_factors"])
            - factor_damage,
            0.03, 1.05,
        )
        s["actor_firm_tfp"] = np.clip(
            s["actor_firm_tfp"] + 0.008 * (1.0 - s["actor_firm_tfp"])
            - 0.012 * s["actor_firm_distress"] - 0.008 * actor_damage[:, :, None],
            0.20, 1.08,
        )
        exponents = np.array([0.40, 0.35, 0.25])[None, None, None, :]
        effective_factors = s["actor_firm_factors"].copy()
        effective_factors[:, :, :, 1] *= labor_efficiency
        effective_factors[:, :, 0, 1] = np.clip(
            effective_factors[:, :, 0, 1] + war_labor_transfer, 0.03, 1.08
        )
        effective_factors[:, :, 1:, 1] *= 1.0 - war_labor_transfer[:, :, None]
        production_function = s["actor_firm_tfp"] * np.prod(
            np.power(effective_factors, exponents), axis=3
        ) * (1.0 - s["actor_firm_distress"])
        production_function[:, :, 0] *= trade_industry_factor
        if mechanism_factors is not None:
            production_function[:, :, 0] *= mechanism_factors.procurement_factor
        production_function[:, :, 1:] *= trade_gdp_factor[:, :, None]
        if energy_opening is not None:
            production_function[:, :, 0] *= energy_opening.production
            production_function[:, :, 1:] *= energy_opening.civilian[:, :, None]
        production_function *= financial_production[:, :, None]
        s["actor_sector_output"] = np.clip(production_function, 0.01, 1.10)
        mean_output = np.mean(s["actor_sector_output"], axis=2)
        civilian_output = np.mean(s["actor_sector_output"][:, :, 1:], axis=2)
        mean_distress = np.mean(s["actor_firm_distress"], axis=2)
        taiwan_need = np.clip(
            0.45 * actor_shortage[:, 1] + 0.30 * mean_distress[:, 1]
            + 0.25 * (1.0 - civilian_output[:, 1]), 0.0, 1.0
        )
        us_aid = 0.012 * us * taiwan_need * mean_output[:, 2] * np.clip(1.20 - s["actor_debt_stress"][:, 2], 0.10, 1.0)
        japan_aid = 0.009 * jp * taiwan_need * mean_output[:, 3] * np.clip(1.15 - s["actor_debt_stress"][:, 3], 0.10, 1.0)
        us_aid *= financial_aid[:, 2] * financial_payment[:, 2] * financial_payment[:, 1]
        japan_aid *= financial_aid[:, 3] * financial_payment[:, 3] * financial_payment[:, 1]
        if mechanism_factors is not None:
            us_aid *= mechanism_factors.aid_multiplier[:, 2]
            japan_aid *= mechanism_factors.aid_multiplier[:, 3]
        minor_aid = 0.003 * minor * taiwan_need
        total_aid = np.clip(us_aid + japan_aid + minor_aid, 0.0, 0.025)
        aid_flow = np.zeros((n, 4))
        aid_flow[:, 1] = total_aid
        aid_flow[:, 2] = -us_aid
        aid_flow[:, 3] = -japan_aid
        s["actor_cumulative_economic_aid"] += aid_flow
        if financial_system is not None:
            financial_system.register_realized_aid(us_aid, japan_aid, minor_aid)
        s["stocks_def"] = np.clip(
            s["stocks_def"] + total_aid[:, None] * np.array([0.28, 0.32, 0.40])[None, :],
            0.01, 1.35,
        )
        inflation_jump = 0.005 * actor_damage + 0.003 * actor_shortage
        s["actor_inflation"] = np.clip(
            s["actor_inflation"]
            + 0.004 * (1.0 - civilian_output) + 0.002 * actor_mobil
            + 0.002 * s["actor_order_backlog"][:, :, 1]
            + inflation_jump - 0.120 * (s["actor_inflation"] - 0.025)
            + 0.120 * financial_inflation
            - 0.035 * np.clip(aid_flow, 0.0, None),
            -0.02, 0.80,
        )
        s["actor_debt_stress"] = np.clip(
            s["actor_debt_stress"]
            + 0.002 * actor_mobil + 0.002 * actor_shortage
            + 0.002 * mean_distress + 0.003 * s["actor_inflation"]
            + 0.003 * (1.0 - trade_gdp_factor)
            + 0.006 * financial_stress + 0.003 * (1.0 - financial_fiscal)
            - 0.006 * (mean_output - 0.55) - 0.005 * (s["actor_debt_stress"] - 0.50)
            - 0.050 * aid_flow,
            0.0, 1.5,
        )
        s["actor_panic"] = np.clip(
            s["actor_panic"]
            + intensity[:, None] * 0.032 * s["actor_panic"] * (1.0 - s["actor_panic"])
            + intensity[:, None] * 0.018 * actor_shortage + 0.040 * actor_damage
            + 0.012 * financial_stress + 0.008 * (1.0 - financial_payment)
            - (0.070 * s["actor_norm"] + 0.025 * (1.0 - intensity[:, None])) * s["actor_panic"],
            0.0, 1.0,
        )
        norm_payoff = (
            0.50 * actor_cog + 0.25 * s["actor_res"][:, :, 1]
            - 0.60 * actor_shortage - 0.40 * actor_damage - 0.35 * s["actor_panic"]
        )
        s["actor_norm"] = np.clip(
            s["actor_norm"]
            + 0.040 * s["actor_norm"] * (1.0 - s["actor_norm"]) * norm_payoff,
            0.0, 1.0,
        )
        s["actor_network"] = np.clip(
            s["actor_network"]
            + (0.035 + 0.035 * (1.0 - intensity[:, None]))
            * (1.0 - s["actor_network"]) * s["actor_res"][:, :, 1]
            - 0.060 * actor_damage - intensity[:, None] * 0.015 * s["actor_panic"],
            0.0, 1.0,
        )
        grievance = np.clip(0.55 * actor_shortage + 0.45 * actor_damage + 0.35 * s["actor_panic"], 0.0, 1.5)
        graphon_kernel = np.array([
            [0.58, 0.10, 0.16, 0.16],
            [0.12, 0.54, 0.18, 0.16],
            [0.12, 0.16, 0.58, 0.14],
            [0.14, 0.18, 0.18, 0.50],
        ])
        mfg_factors = None
        if flags["continuous_multipop_mfg"] and mfg_system is not None:
            mfg_common_noise = np.tanh(
                0.70 * progress_signal + 0.40 * s["actor_backlash"]
                - 0.40 * s["actor_panic"]
            )
            mfg_jump_signal = np.clip(
                homeland_mark + island_mark
                + leader_disruption * leader_severity, 0.0, 1.0
            )
            mfg_factors = mfg_system.update(
                progress=progress_signal,
                damage=actor_damage,
                shortage=actor_shortage,
                inflation=s["actor_inflation"],
                financial_stress=financial_stress,
                command=command_eff,
                network=s["actor_network"],
                contagion_mobilization=s["actor_social_mobilization"],
                contagion_fatigue=s["actor_social_fatigue"],
                common_noise=mfg_common_noise,
                jump_signal=mfg_jump_signal,
            )
            s["actor_support"] = np.clip(mfg_factors.support, 0.01, 0.99)
            if mfg_factors.joint_action_field is not None:
                joint_action = mfg_factors.joint_action_field
                collective, service, migration, hoarding, forwarding = np.moveaxis(
                    joint_action, -1, 0
                )
                s["actor_social_mobilization"] = np.clip(
                    s["actor_social_mobilization"] + 0.010 * collective, 0.0, 1.0
                )
                s["actor_labor_fatigue"] = np.clip(
                    s["actor_labor_fatigue"] - 0.006 * service, 0.0, 1.0
                )
                s["actor_labor_mismatch"] = np.clip(
                    s["actor_labor_mismatch"] + 0.012 * migration, 0.0, 1.0
                )
                s["actor_inflation"] = np.clip(
                    s["actor_inflation"] + 0.0025 * hoarding, -0.05, 0.80
                )
                s["actor_network"] = np.clip(
                    s["actor_network"] + 0.005 * forwarding, 0.0, 1.0
                )
        if flags["stratified_social_contagion"] and social_system is not None:
            social = social_system.update(
                prior_support=s["actor_support"],
                actor_damage=actor_damage,
                actor_shortage=actor_shortage,
                actor_inflation=s["actor_inflation"],
                actor_backlash=s["actor_backlash"],
                actor_network=s["actor_network"],
                actor_norm=s["actor_norm"],
                actor_resilience=s["actor_res"][:, :, 1],
                progress_signal=progress_signal,
                command_integrity=command_eff,
                conflict_intensity=intensity,
                mfg_threshold_shift=(
                    mfg_factors.threshold_shift if mfg_factors is not None else None
                ),
                mfg_mobilization=(
                    mfg_factors.mobilization_preference if mfg_factors is not None else None
                ),
                mfg_fatigue=(
                    mfg_factors.fatigue_preference if mfg_factors is not None else None
                ),
            )
            s["actor_support"] = np.clip(social["support"], 0.01, 0.99)
            s["actor_social_mobilization"] = social["mobilization"]
            s["actor_social_fatigue"] = social["fatigue"]
            s["actor_panic"] = np.clip(0.72 * s["actor_panic"] + 0.28 * social["panic"], 0.0, 1.0)
            s["actor_norm"] = np.clip(0.82 * s["actor_norm"] + 0.18 * social["norm"], 0.0, 1.0)
            s["actor_action"] = np.clip(social["collective_action"], 0.0, 1.0)
            if (
                mfg_factors is not None
                and mfg_factors.network_action_field is not None
                and getattr(mfg_system, "advanced_graphon", False)
            ):
                mean_action = mfg_factors.network_action_field.copy()
            else:
                mean_action = s["actor_action"] @ graphon_kernel.T
        else:
            if flags["graphon_social"]:
                mean_action = s["actor_action"] @ graphon_kernel.T
            else:
                mean_action = np.repeat(np.mean(s["actor_action"], axis=1, keepdims=True), 4, axis=1)
            s["actor_action"] = sigmoid(
                3.0 * (
                    grievance + 0.40 * mean_action - s["actor_norm"]
                    - 0.50 * s["actor_res"][:, :, 1] - 0.35 * s["actor_network"]
                )
            )
        mean_action[:, 1] = np.clip(
            mean_action[:, 1] - 0.20 * effective_proxy
            + 0.16 * s["taiwan_proxy_leakage"], 0.0, 1.0
        )
        if flags["stratified_social_contagion"] and social_system is not None:
            s["actor_action"][:, 1] = np.clip(
                s["actor_action"][:, 1] - 0.20 * effective_proxy
                + 0.16 * s["taiwan_proxy_leakage"], 0.0, 1.0
            )
        social_target = np.clip(
            0.34 * actor_cog + 0.28 * s["actor_norm"] + 0.25 * s["actor_network"]
            + 0.13 * (1.0 - s["actor_panic"]) - 0.20 * s["actor_action"],
            0.0, 1.0,
        )
        political_fragility = np.clip(
            0.22 * np.clip(s["actor_debt_stress"], 0.0, 1.0)
            + 0.16 * mean_distress + 0.18 * s["actor_panic"]
            + 0.18 * s["actor_action"] + 0.14 * (1.0 - s["actor_network"])
            + 0.12 * (1.0 - s["actor_res"][:, :, 3]),
            0.0, 1.0,
        )
        political_fragility = np.clip(
            political_fragility + 0.12 * actor_mobil * (1.0 - s["actor_support"]), 0.0, 1.0
        )
        political_fragility = np.clip(
            political_fragility + 0.10 * financial_stress
            + 0.06 * (1.0 - financial_payment) + 0.05 * (1.0 - financial_fiscal),
            0.0, 1.0,
        )
        actor_capability = np.stack([
            0.35 * mean_node_cn + 0.25 * np.mean(s["stocks_cn"], axis=1) + 0.20 * s["web_cn"] + 0.20 * s["cog_cn"],
            0.28 * mean_node_def + 0.22 * np.mean(s["stocks_def"], axis=1) + 0.18 * s["web_def"] + 0.18 * s["cog_def"] + 0.14 * (1.0 - s["land"]),
            0.34 * mean_node_def + 0.24 * np.mean(s["stocks_def"], axis=1) + 0.22 * s["web_def"] + 0.20 * s["csg"],
            0.34 * mean_node_def + 0.24 * np.mean(s["stocks_def"], axis=1) + 0.22 * s["web_def"] + 0.20 * s["cog_def"],
        ], axis=1) * manpower_effectiveness * s["actor_front_rear_integrity"][:, :, 0]
        economic_target = np.stack([
            mean_node_cn, mean_node_def,
            np.clip(0.92 - 0.20 * us * actor_shortage[:, 2], 0.30, 0.98),
            np.clip(0.88 - 0.25 * jp * actor_shortage[:, 3], 0.30, 0.98),
        ], axis=1)
        economic_target = np.clip(economic_target - 0.18 * s["actor_inflation"], 0.0, 1.0)
        economic_target = np.clip(
            economic_target - 0.35 * (1.0 - trade_gdp_factor), 0.0, 1.0
        )
        economic_target = np.clip(
            economic_target - 0.14 * financial_stress
            - 0.10 * (1.0 - financial_credit)
            - 0.08 * (1.0 - financial_payment),
            0.0, 1.0,
        )
        economic_target[:, 1] = np.clip(economic_target[:, 1] + 0.40 * total_aid, 0.0, 1.0)
        s["actor_res"][:, :, 0] = np.clip(
            s["actor_res"][:, :, 0]
            + 0.006 * (economic_target - s["actor_res"][:, :, 0])
            - 0.007 * actor_shortage - 0.003 * actor_mobil ** 2 - 0.018 * actor_damage,
            0.0, 1.0,
        )
        s["actor_res"][:, :, 2] = np.clip(
            s["actor_res"][:, :, 2]
            + 0.010 * (social_target - s["actor_res"][:, :, 2])
            - 0.004 * actor_shortage - 0.002 * actor_mobil - 0.010 * actor_damage,
            0.0, 1.0,
        )
        s["actor_res"][:, :, 1] = np.clip(
            s["actor_res"][:, :, 1]
            + 0.005 * (0.5 * (s["actor_res"][:, :, 0] + s["actor_res"][:, :, 2]) - s["actor_res"][:, :, 1])
            - 0.003 * actor_damage - 0.006 * political_fragility, 0.0, 1.0,
        )
        s["actor_res"][:, :, 3] = np.clip(
            s["actor_res"][:, :, 3]
            + 0.008 * (actor_capability - s["actor_res"][:, :, 3])
            - 0.006 * actor_shortage - 0.014 * actor_damage,
            0.0, 1.0,
        )
        actor_hazard = np.clip(0.00005 + 0.10 * (0.12 - s["actor_res"]), 0.0, 0.03)
        collapse_coupling = np.array([
            [0.0000, 0.0015, 0.0020, 0.0018],
            [0.0018, 0.0000, 0.0022, 0.0030],
            [0.0020, 0.0022, 0.0000, 0.0025],
            [0.0018, 0.0030, 0.0025, 0.0000],
        ])
        actor_hazard += np.einsum("nai,ij->naj", s["actor_collapse"].astype(float), collapse_coupling)
        actor_hazard[:, :, 0] += np.clip(0.020 * (s["actor_debt_stress"] - 0.80), 0.0, 0.025)
        actor_hazard[:, :, 1] += np.clip(0.025 * (political_fragility - 0.62), 0.0, 0.025)
        actor_hazard[:, :, 2] += np.clip(
            0.010 * s["actor_panic"] + 0.010 * s["actor_action"]
            + 0.010 * (1.0 - s["actor_network"]) - 0.008,
            0.0, 0.025,
        )
        defeat_pressure = np.stack([
            np.clip(-observed_balance, 0.0, 1.0),
            np.clip(observed_balance, 0.0, 1.0),
            us * np.clip(observed_balance, 0.0, 1.0),
            jp * np.clip(observed_balance, 0.0, 1.0),
        ], axis=1)
        military_pressure = (
            0.25 * actor_shortage + 0.20 * defeat_pressure
            + 0.20 * (1.0 - actor_cog) + 0.20 * (1.0 - command_eff)
            + 0.15 * np.clip(4.0 * actor_damage, 0.0, 1.0)
            + 0.16 * np.clip(0.72 - manpower_effectiveness, 0.0, 1.0)
            + 0.14 * (1.0 - s["actor_front_rear_integrity"][:, :, 0])
        )
        actor_hazard[:, :, 3] += np.clip(0.035 * (military_pressure - 0.72), 0.0, 0.03)
        actor_hazard = np.clip(actor_hazard, 0.0, 0.05)
        coup_hazard = np.clip(
            0.00002 + 0.030 * np.clip(political_fragility - 0.75, 0.0, 1.0)
            + 0.020 * np.clip(0.20 - s["actor_res"][:, :, 3], 0.0, 1.0)
            + 0.010 * np.clip(s["actor_action"] - 0.75, 0.0, 1.0)
            + 0.010 * s["actor_collapse"][:, :, 2]
            + 0.012 * s["actor_collapse"][:, :, 3],
            0.0, 0.06,
        )
        new_coup = (~s["actor_coup"]) & (rng.random((n, 4)) < coup_hazard)
        if hybrid_system is not None:
            hybrid_system.apply_coup_reset(s, new_coup)
        else:
            s["actor_coup"] |= new_coup
            coup_loss = np.array([0.12, 0.35, 0.18, 0.28])[None, None, :]
            s["actor_res"] = np.clip(s["actor_res"] - new_coup[:, :, None] * coup_loss, 0.0, 1.0)
        collapse_pressure = s["actor_res"] < 0.12
        s["actor_collapse_pressure_duration"] = np.where(
            collapse_pressure,
            np.minimum(
                s["actor_collapse_pressure_duration"] + 1,
                np.iinfo(np.int16).max,
            ),
            0,
        )
        persistent_collapse = s["actor_collapse_pressure_duration"] >= 3
        jump_collapse = rng.random((n, 4, 4)) < actor_hazard
        if hybrid_system is not None:
            hybrid_system.apply_collapse_reset(
                s, persistent_collapse, jump_collapse, new_coup
            )
        else:
            s["actor_collapse"] |= persistent_collapse | jump_collapse
            s["actor_collapse"][:, :, 1] |= new_coup
            # Coups propagate alliance doubt and adversary morale shifts without forcing identical outcomes.
            coup_signal = new_coup.astype(float)
            s["cog_cn"] = np.clip(s["cog_cn"] + 0.018 * (coup_signal[:, 1] + coup_signal[:, 2] + coup_signal[:, 3]) - 0.030 * coup_signal[:, 0], 0.04, 0.96)
            s["cog_def"] = np.clip(s["cog_def"] + 0.018 * coup_signal[:, 0] - 0.020 * coup_signal[:, 1] - 0.012 * coup_signal[:, 2] - 0.012 * coup_signal[:, 3], 0.04, 0.96)
        command_damage = actor_damage[:, :, None] * np.array([0.85, 1.00, 1.12])[None, None, :]
        command_repair = 0.018 * mean_output[:, :, None] * s["actor_res"][:, :, 3, None]
        hierarchy_cascade = np.stack([
            np.zeros((n, 4)),
            0.020 * (1.0 - s["actor_command"][:, :, 0]),
            0.016 * (1.0 - s["actor_command"][:, :, 1]),
        ], axis=2)
        s["actor_command"] = np.clip(
            s["actor_command"] + command_repair - 0.070 * command_damage - hierarchy_cascade,
            0.03, 0.995,
        )
        s["res_cn"] = s["actor_res"][:, 0, :3]
        s["res_def"] = np.sum(s["actor_res"][:, 1:, :3] * coalition_weights[:, :, None], axis=1)
        s["nodes_cn"][:, 2] *= 1.0 - 0.005 * s["actor_collapse"][:, 0, 1] - 0.006 * s["actor_collapse"][:, 0, 3]
        coalition_pol_collapse = np.sum(s["actor_collapse"][:, 1:, 1] * coalition_weights, axis=1)
        coalition_mil_collapse = np.sum(s["actor_collapse"][:, 1:, 3] * coalition_weights, axis=1)
        s["nodes_def"][:, 2] *= 1.0 - 0.005 * coalition_pol_collapse - 0.006 * coalition_mil_collapse
        s["cog_cn"] *= 1.0 - 0.004 * s["actor_collapse"][:, 0, 2]
        coalition_soc_collapse = np.sum(s["actor_collapse"][:, 1:, 2] * coalition_weights, axis=1)
        s["cog_def"] *= 1.0 - 0.004 * coalition_soc_collapse

        if macro_system is not None:
            labor_capacity = np.clip(
                manpower_effectiveness
                * (1.0 - 0.38 * s["actor_labor_fatigue"])
                * (1.0 - 0.28 * s["actor_labor_mismatch"]),
                0.05, 1.15,
            )
            macro_system.update(
                month=month,
                actor_damage=actor_damage,
                actor_logistics=(
                    np.mean(s["actor_front_rear_integrity"], axis=2)
                    * (energy_opening.logistics if energy_opening is not None else 1.0)
                    * (
                        geometry_factors.macro_logistics_factor
                        if geometry_factors is not None else 1.0
                    )
                ),
                actor_labor=labor_capacity,
                actor_financial_stress=financial_stress,
                actor_payment=financial_payment,
                actor_fiscal=financial_fiscal,
                actor_social_action=s["actor_action"],
                actor_panic=s["actor_panic"],
                actor_mobilization=actor_mobil,
                blockade=blockade,
            )

        # Broad control is an endogenous absorbing path event, not a post-hoc
        # logit correction. Physical control must be sustained together with
        # a viable command and front-logistics state; a unification settlement
        # also closes the event immediately.
        if spatial_system is not None:
            spatial_factors = spatial_system.update(
                aggregate_land=s["land"],
                attacker_logistics=s["actor_front_rear_integrity"][:, 0, 0],
                attacker_command=command_eff[:, 0],
                defender_command=command_eff[:, 1],
                defender_resistance=np.clip(
                    0.60 * s["actor_support"][:, 1]
                    + 0.40 * (1.0 - s["actor_social_fatigue"][:, 1]), 0.0, 1.0
                ),
                external_intervention=np.clip(0.58 * us + 0.42 * jp, 0.0, 1.0),
                local_agent_capacity=s["taiwan_proxy_capacity"],
                damage=actor_damage[:, 1],
                attacker_pnt=s["pnt_cn"],
                blockade=blockade,
            )
            broad_control_condition = (
                spatial_factors.stable_broad_control
                & (s["settlement_outcome"] != 1)
                & (command_eff[:, 0] >= 0.28)
                & (s["actor_front_rear_integrity"][:, 0, 0] >= 0.24)
                & (np.mean(s["stocks_cn"], axis=1) >= 0.10)
            )
            s["broad_control_streak"] = spatial_system.control_duration.copy()
        else:
            spatial_factors = None
            broad_control_condition = (
                (s["land"] >= 0.55)
                & (s["settlement_outcome"] != 1)
                & (command_eff[:, 0] >= 0.28)
                & (s["actor_front_rear_integrity"][:, 0, 0] >= 0.24)
                & (np.mean(s["stocks_cn"], axis=1) >= 0.10)
            )
            s["broad_control_streak"] = np.where(
                broad_control_condition,
                np.minimum(s["broad_control_streak"] + 1, np.iinfo(np.int16).max),
                0,
            )
        newly_broad = (~s["broad_control_achieved"]) & (
            (s["broad_control_streak"] >= 6) | (s["settlement_outcome"] == 2)
        )
        s["broad_control_achieved"] |= newly_broad
        s["broad_control_month"][newly_broad] = month + 1

        cross = (s["air"] - 0.5) * (s["sea"] - 0.5) + 0.55 * s["land"] * (csi - 0.5)
        values = {
            "air": s["air"], "sea": s["sea"], "csi": csi,
            "web_diff": s["web_cn"] * s["ooda_cn"] - s["web_def"] * s["ooda_def"],
            "stock_diff": np.mean(s["stocks_cn"], axis=1) - np.mean(s["stocks_def"], axis=1),
            "port_diff": s["nodes_cn"][:, 0] - s["nodes_def"][:, 0],
            "pnt_diff": s["pnt_cn"] - s["pnt_def"],
            "cog_diff": s["cog_cn"] - s["cog_def"],
            "resilience_diff": np.mean(s["res_cn"], axis=1) - np.mean(s["res_def"], axis=1),
            "mobilization_diff": s["mobil_cn"] - s["mobil_def"],
            "command_diff": command_eff[:, 0] - np.sum(command_eff[:, 1:] * coalition_weights, axis=1),
            "output_diff": mean_output[:, 0] - np.sum(mean_output[:, 1:] * coalition_weights, axis=1),
            "conflict_intensity": intensity,
            "homeland_damage_diff": s["actor_homeland_damage"][:, 0]
            - np.sum(s["actor_homeland_damage"][:, 1:] * coalition_weights, axis=1),
            "terminal_resolution": np.where(
                s["settlement_outcome"] == 2, 1.0,
                np.where(s["settlement_outcome"] == 1, -1.0, 0.0),
            ),
            "cross": cross, "blockade": blockade, "throughput": beach_service,
        }
        for name, value in values.items():
            sums[name] += value
        min_stock_cn = np.minimum(min_stock_cn, np.min(s["stocks_cn"], axis=1))
        min_stock_def = np.minimum(min_stock_def, np.min(s["stocks_def"], axis=1))
        max_repair_cn = np.maximum(max_repair_cn, s["repair_cn"])
        max_repair_def = np.maximum(max_repair_def, s["repair_def"])

        if keep_trace and month in (0, 5, 11, 23, 59, 119, 179, 239, 359):
            trace.append({
                "month": month + 1,
                "air_control_median": float(np.median(s["air"])),
                "sea_control_median": float(np.median(s["sea"])),
                "land_control_median": float(np.median(s["land"])),
                "csi_median": float(np.median(csi)),
                "china_killweb_median": float(np.median(s["web_cn"] * s["ooda_cn"])),
                "defender_killweb_median": float(np.median(s["web_def"] * s["ooda_def"])),
                "china_min_stock_median": float(np.median(np.min(s["stocks_cn"], axis=1))),
                "defender_min_stock_median": float(np.median(np.min(s["stocks_def"], axis=1))),
                "china_cognition_median": float(np.median(s["cog_cn"])),
                "defender_cognition_median": float(np.median(s["cog_def"])),
            })
        elapsed = month + 1
        if elapsed in HORIZON_MONTHS:
            avg = {name: value / elapsed for name, value in sums.items()}
            components = {
                "fields": 0.36 * (avg["air"] - 0.5) + 0.42 * (avg["csi"] - 0.5) + 0.55 * (s["land"] - 0.18),
                "killweb_ooda": 0.34 * avg["web_diff"] + 0.14 * avg["command_diff"],
                "logistics": 0.30 * avg["stock_diff"] + 0.16 * avg["port_diff"] - 0.12 * (avg["blockade"] - 0.5) + 0.10 * avg["output_diff"],
                "terrain_pnt": 0.18 * avg["pnt_diff"] - 0.10 * s["mountain_share"] * (1.0 - s["land"]),
                "cognition": 0.18 * avg["cog_diff"],
                "resilience": 0.18 * avg["resilience_diff"] + 0.06 * avg["mobilization_diff"]
                - 0.08 * avg["homeland_damage_diff"],
                "resolution": 0.80 * avg["terminal_resolution"],
                "cross_domain": 0.30 * avg["cross"],
            }
            snapshots[elapsed] = {
                "score": sum(components.values()),
                "components": components,
                "broad_control_paths": s["broad_control_achieved"].copy(),
                "broad_control_month_paths": s["broad_control_month"].copy(),
                "mfg_kernel_score_paths": (
                    mfg_system.kernel_draw_score()
                    if mfg_system is not None else np.zeros(n)
                ),
                "settlement_outcome_paths": s["settlement_outcome"].copy(),
                "terminal_land_control_paths": s["land"].copy(),
                "population_control_paths": (
                    spatial_factors.population_control.copy()
                    if spatial_factors is not None else s["land"].copy()
                ),
                "administrative_control_paths": (
                    spatial_factors.administrative_control.copy()
                    if spatial_factors is not None else s["land"].copy()
                ),
                "organized_defense_paths": (
                    spatial_factors.organized_defense.copy()
                    if spatial_factors is not None else 1.0 - s["land"].copy()
                ),
                "importance_weight_paths": importance_weight.copy(),
                "diagnostics": {
                    "air_control_p10_p50_p90": q(avg["air"]),
                    "sea_control_p10_p50_p90": q(avg["sea"]),
                    "csi_p10_p50_p90": q(avg["csi"]),
                    "terminal_land_control_p10_p50_p90": q(s["land"]),
                    "population_control_p10_p50_p90": q(
                        spatial_factors.population_control
                        if spatial_factors is not None else s["land"]
                    ),
                    "administrative_control_p10_p50_p90": q(
                        spatial_factors.administrative_control
                        if spatial_factors is not None else s["land"]
                    ),
                    "organized_defense_p10_p50_p90": q(
                        spatial_factors.organized_defense
                        if spatial_factors is not None else 1.0 - s["land"]
                    ),
                    "mean_beach_throughput_p10_p50_p90": q(avg["throughput"]),
                    "china_min_stock_p10_p50_p90": q(min_stock_cn),
                    "defender_min_stock_p10_p50_p90": q(min_stock_def),
                    "china_max_repair_queue_p10_p50_p90": q(max_repair_cn),
                    "defender_max_repair_queue_p10_p50_p90": q(max_repair_def),
                    "china_terminal_cognition_p10_p50_p90": q(s["cog_cn"]),
                    "defender_terminal_cognition_p10_p50_p90": q(s["cog_def"]),
                    "china_terminal_resilience_p10_p50_p90": q(np.mean(s["res_cn"], axis=1)),
                    "defender_terminal_resilience_p10_p50_p90": q(np.mean(s["res_def"], axis=1)),
                    "actor_any_collapse_share": {
                        actor: float(np.mean(np.any(s["actor_collapse"][:, i, :], axis=1)))
                        for i, actor in enumerate(("china", "taiwan", "united_states", "japan"))
                    },
                    "actor_collapse_by_type": {
                        actor: [float(np.mean(s["actor_collapse"][:, i, j])) for j in range(4)]
                        for i, actor in enumerate(("china", "taiwan", "united_states", "japan"))
                    },
                    "actor_coup_share": {
                        actor: float(np.mean(s["actor_coup"][:, i]))
                        for i, actor in enumerate(("china", "taiwan", "united_states", "japan"))
                    },
                    "actor_social_state_medians": {
                        actor: {
                            "panic": float(np.median(s["actor_panic"][:, i])),
                            "norm": float(np.median(s["actor_norm"][:, i])),
                            "network": float(np.median(s["actor_network"][:, i])),
                            "collective_action": float(np.median(s["actor_action"][:, i])),
                            "active_mobilization": float(np.median(s["actor_social_mobilization"][:, i])),
                            "active_fatigue": float(np.median(s["actor_social_fatigue"][:, i])),
                        }
                        for i, actor in enumerate(("china", "taiwan", "united_states", "japan"))
                    },
                    "actor_economic_state_medians": {
                        actor: {
                            "firm_distress_by_class": [
                                float(np.median(s["actor_firm_distress"][:, i, j])) for j in range(4)
                            ],
                            "debt_stress": float(np.median(s["actor_debt_stress"][:, i])),
                            "inflation": float(np.median(s["actor_inflation"][:, i])),
                            "military_order_backlog": float(np.median(s["actor_order_backlog"][:, i, 0])),
                            "civilian_order_backlog": float(np.median(s["actor_order_backlog"][:, i, 1])),
                            "output_by_class": [
                                float(np.median(s["actor_sector_output"][:, i, j])) for j in range(4)
                            ],
                        }
                        for i, actor in enumerate(("china", "taiwan", "united_states", "japan"))
                    },
                    "actor_cumulative_economic_aid_medians": {
                        actor: float(np.median(s["actor_cumulative_economic_aid"][:, i]))
                        for i, actor in enumerate(("china", "taiwan", "united_states", "japan"))
                    },
                    "actor_command_effectiveness_medians": {
                        actor: float(np.median(command_eff[:, i]))
                        for i, actor in enumerate(("china", "taiwan", "united_states", "japan"))
                    },
                    "conflict_regime_shares": {
                        label: float(np.mean(s["conflict_regime"] == i))
                        for i, label in enumerate(("high_intensity", "low_intensity", "ceasefire", "frozen", "rearmament", "settlement"))
                    },
                    "actor_homeland_damage_medians": {
                        actor: float(np.median(s["actor_homeland_damage"][:, i]))
                        for i, actor in enumerate(("china", "taiwan", "united_states", "japan"))
                    },
                    "actor_island_chain_damage_medians": {
                        actor: float(np.median(s["actor_island_chain_damage"][:, i]))
                        for i, actor in enumerate(("china", "taiwan", "united_states", "japan"))
                    },
                    "actor_front_transit_depth_integrity_medians": {
                        actor: {
                            "front": float(np.median(s["actor_front_rear_integrity"][:, i, 0])),
                            "transit": float(np.median(s["actor_front_rear_integrity"][:, i, 1])),
                            "strategic_depth": float(np.median(s["actor_front_rear_integrity"][:, i, 2])),
                        }
                        for i, actor in enumerate(("china", "taiwan", "united_states", "japan"))
                    },
                    "actor_backlash_medians": {
                        actor: float(np.median(s["actor_backlash"][:, i]))
                        for i, actor in enumerate(("china", "taiwan", "united_states", "japan"))
                    },
                    "taiwan_proxy_agent_medians": {
                        "capacity": float(np.median(s["taiwan_proxy_capacity"])),
                        "principal_agent_leakage": float(np.median(s["taiwan_proxy_leakage"])),
                    },
                    "settlement_outcome_shares": {
                        "unresolved_or_frozen": float(np.mean(s["settlement_outcome"] == 0)),
                        "status_quo_restored": float(np.mean(s["settlement_outcome"] == 1)),
                        "unification": float(np.mean(s["settlement_outcome"] == 2)),
                    },
                    "actor_manpower_effectiveness_medians": {
                        actor: float(np.median(s["actor_manpower_eff"][:, i]))
                        for i, actor in enumerate(("china", "taiwan", "united_states", "japan"))
                    },
                    "actor_support_and_fatigue_medians": {
                        actor: {
                            "mobilization_support": float(np.median(s["actor_support"][:, i])),
                            "labor_fatigue": float(np.median(s["actor_labor_fatigue"][:, i])),
                            "labor_spatial_mismatch": float(np.median(s["actor_labor_mismatch"][:, i])),
                            "labor_skill_stock": [
                                float(np.median(s["actor_labor_skill"][:, i, j])) for j in range(4)
                            ],
                        }
                        for i, actor in enumerate(("china", "taiwan", "united_states", "japan"))
                    },
                    "mechanical_bridgehead_share": float(np.mean(s["land"] >= 0.10)),
                    "aggregate_land_majority_share": float(np.mean(s["land"] >= 0.55)),
                    "spatial_majority_control_share": float(np.mean(
                        (
                            (spatial_factors.population_control >= 0.50)
                            & (spatial_factors.administrative_control >= 0.50)
                            & (spatial_factors.organized_defense <= 0.35)
                        ) if spatial_factors is not None else (s["land"] >= 0.55)
                    )),
                    "mechanical_majority_control_share": float(np.mean(s["land"] >= 0.55)),
                    "absorbing_broad_control_share": float(np.mean(s["broad_control_achieved"])),
                    "importance_sampling_effective_sample_size": float(
                        np.sum(importance_weight) ** 2 / np.sum(importance_weight ** 2)
                    ),
                    "continuous_mfg_fixed_point": (
                        mfg_system.diagnostics() if mfg_system is not None else None
                    ),
                    "cge_dsge_fixed_point": (
                        macro_system.diagnostics() if macro_system is not None else None
                    ),
                    "operational_constraints": (
                        operations_system.diagnostics() if operations_system is not None else None
                    ),
                    "dynamic_geospatial_pnt": (
                        pnt_system.diagnostics() if pnt_system is not None else None
                    ),
                    "rolling_differential_game": (
                        strategy_system.diagnostics() if strategy_system is not None else None
                    ),
                    "spatial_control_network": (
                        spatial_system.diagnostics() if spatial_system is not None else None
                    ),
                    "geometric_multiscale_coupling": (
                        geometry_system.diagnostics() if geometry_system is not None else None
                    ),
                },
            }
    return snapshots, trace


def make_prior_paths(z_standard, base_probability_15y, v40_row, horizon_months):
    p10, p50, p90 = v40_row["path_probability_p10_p50_p90"]
    spread_15y = min(max((float(logit(p90)) - float(logit(p10))) / 2.563, 0.22), 1.35)
    spread = min(spread_15y * math.sqrt(horizon_months / 180.0), 1.65)
    base_probability = 1.0 - math.exp(math.log1p(-base_probability_15y) * horizon_months / 180.0)
    raw = sigmoid(logit(base_probability) + z_standard * spread)
    correction = logit(base_probability) - logit(float(np.mean(raw)))
    return logit(raw) + correction


def summarize_probability(probabilities):
    mean = float(np.mean(probabilities))
    se = math.sqrt(max(mean * (1.0 - mean), 1e-12) / probabilities.size)
    threshold = float(np.quantile(probabilities, 0.99))
    return {
        "probability": mean,
        "path_probability_p10_p50_p90": q(probabilities),
        "approx_95pct_monte_carlo_interval": [max(0.0, mean - 1.96 * se), min(1.0, mean + 1.96 * se)],
        "cvar99_path_probability": float(np.mean(probabilities[probabilities >= threshold])),
    }


def run_model():
    base, v40_rows, terrain, sipri = load_inputs()
    results = []
    ablations = []
    representative_trace = []

    for year in YEARS:
        for case in CASES:
            cell_seed = SEED + year * 101 + CASES.index(case) * 1009
            prior_rng = np.random.default_rng(cell_seed)
            z_standard = prior_rng.normal(0.0, 1.0, PATHS)
            scenario_snapshots = {}
            for scenario_index, scenario_key in enumerate(SCENARIOS):
                rng = np.random.default_rng(cell_seed + scenario_index * 100003)
                keep_trace = year == 2030 and case == "timely_full" and scenario_key == "central"
                snapshots, trace = simulate_mechanisms(
                    rng, year, case, scenario_key, terrain, sipri, keep_trace
                )
                scenario_snapshots[scenario_key] = snapshots
                if keep_trace:
                    representative_trace = trace

            previous_by_scenario = {key: np.zeros(PATHS) for key in SCENARIOS}
            previous_by_ablation = {key: np.zeros(PATHS) for key in ("full_v42",) + tuple(f"without_{x}" for x in COMPONENT_NAMES)}
            for horizon_months in HORIZON_MONTHS:
                prior_log_odds = make_prior_paths(
                    z_standard, base[(year, case)], v40_rows[(year, case)], horizon_months
                )
                prior_horizon_probability = 1.0 - math.exp(
                    math.log1p(-base[(year, case)]) * horizon_months / 180.0
                )
                for scenario_key in SCENARIOS:
                    snap = scenario_snapshots[scenario_key][horizon_months]
                    probabilities = sigmoid(prior_log_odds + snap["score"])
                    probabilities = np.maximum(probabilities, previous_by_scenario[scenario_key])
                    previous_by_scenario[scenario_key] = probabilities
                    summary = summarize_probability(probabilities)
                    results.append({
                        "conflict_year": year,
                        "horizon_years": horizon_months // 12,
                        "case": case,
                        "case_label": CASE_LABELS[case],
                        "scenario": scenario_key,
                        "scenario_label": SCENARIOS[scenario_key]["label"],
                        "v41_15y_prior_probability": base[(year, case)],
                        "prior_horizon_probability": prior_horizon_probability,
                        **summary,
                        "diagnostics": snap["diagnostics"],
                        "component_contribution_p10_p50_p90": {
                            name: q(value) for name, value in snap["components"].items()
                        },
                    })

                central_components = scenario_snapshots["central"][horizon_months]["components"]
                full_score = sum(central_components.values())
                probability_sets = {"full_v42": sigmoid(prior_log_odds + full_score)}
                for removed in COMPONENT_NAMES:
                    probability_sets[f"without_{removed}"] = sigmoid(
                        prior_log_odds + full_score - central_components[removed]
                    )
                for ablation_name, probabilities in probability_sets.items():
                    probabilities = np.maximum(probabilities, previous_by_ablation[ablation_name])
                    previous_by_ablation[ablation_name] = probabilities
                    ablations.append({
                        "conflict_year": year, "horizon_years": horizon_months // 12,
                        "case": case, "ablation": ablation_name,
                        "probability": float(np.mean(probabilities)),
                    })

    return results, ablations, representative_trace, terrain, sipri


def validate(results, ablations, terrain, sipri):
    expected = len(YEARS) * len(CASES) * len(SCENARIOS) * len(HORIZON_MONTHS)
    errors = []
    if len(results) != expected:
        errors.append(f"result cells {len(results)} != {expected}")
    if len(ablations) != len(YEARS) * len(CASES) * (1 + len(COMPONENT_NAMES)) * len(HORIZON_MONTHS):
        errors.append("ablation cell count")
    for row in results:
        if not 0.0 <= row["probability"] <= 1.0:
            errors.append("probability range")
        if row["diagnostics"]["mechanical_majority_control_share"] > row["diagnostics"]["mechanical_bridgehead_share"] + 1e-12:
            errors.append("land-state nesting")
        for key in ("china_min_stock_p10_p50_p90", "defender_min_stock_p10_p50_p90"):
            if row["diagnostics"][key][0] < 0.0:
                errors.append("negative stock")
    if abs(terrain["land_share_above_500m"] - 0.46568105522266234) > 1e-8:
        errors.append("terrain snapshot mismatch")
    if set(sipri) != {"china", "united_states", "japan", "taiwan"}:
        errors.append("SIPRI actor coverage")
    for year in YEARS:
        central = {r["case"]: r["probability"] for r in results if r["conflict_year"] == year and r["scenario"] == "central" and r["horizon_years"] == 15}
        if not central["timely_full"] < central["taiwan_alone"]:
            errors.append(f"intervention ordering {year}")
    for year in YEARS:
        for case in CASES:
            for scenario in SCENARIOS:
                series = sorted(
                    (r for r in results if r["conflict_year"] == year and r["case"] == case and r["scenario"] == scenario),
                    key=lambda r: r["horizon_years"],
                )
                if any(b["probability"] + 1e-12 < a["probability"] for a, b in zip(series, series[1:])):
                    errors.append("non-monotone cumulative probability")
    if errors:
        raise RuntimeError("; ".join(sorted(set(errors))))


def write_csvs(results, ablations, trace):
    with RESULTS_PATH.open("w", encoding="utf-8-sig", newline="") as handle:
        fields = [
            "conflict_year", "horizon_years", "case", "case_label", "scenario", "scenario_label",
            "v41_15y_prior_probability", "prior_horizon_probability", "probability", "p10", "p50", "p90", "cvar99",
            "bridgehead_share", "majority_control_share",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in results:
            writer.writerow({
                "conflict_year": row["conflict_year"], "horizon_years": row["horizon_years"], "case": row["case"],
                "case_label": row["case_label"], "scenario": row["scenario"],
                "scenario_label": row["scenario_label"],
                "v41_15y_prior_probability": row["v41_15y_prior_probability"],
                "prior_horizon_probability": row["prior_horizon_probability"],
                "probability": row["probability"],
                "p10": row["path_probability_p10_p50_p90"][0],
                "p50": row["path_probability_p10_p50_p90"][1],
                "p90": row["path_probability_p10_p50_p90"][2],
                "cvar99": row["cvar99_path_probability"],
                "bridgehead_share": row["diagnostics"]["mechanical_bridgehead_share"],
                "majority_control_share": row["diagnostics"]["mechanical_majority_control_share"],
            })
    with ABLATION_PATH.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("conflict_year", "horizon_years", "case", "ablation", "probability"))
        writer.writeheader()
        writer.writerows(ablations)
    with STATE_PATH.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=trace[0].keys())
        writer.writeheader()
        writer.writerows(trace)


def render(results):
    central = [r for r in results if r["scenario"] == "central" and r["horizon_years"] == 15]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4), dpi=160)
    colors = ("#b42318", "#d97706", "#8a5a00", "#2563eb", "#475569")
    for case, color in zip(CASES, colors):
        rows = sorted((r for r in central if r["case"] == case), key=lambda r: r["conflict_year"])
        axes[0].plot([r["conflict_year"] for r in rows], [100 * r["probability"] for r in rows], marker="o", label=CASE_LABELS[case], color=color)
    axes[0].set_title("V4.2中心情景：条件广泛控制率")
    axes[0].set_xlabel("冲突年份")
    axes[0].set_ylabel("概率（%）")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8)

    rows = sorted(
        (r for r in results if r["conflict_year"] == 2030 and r["case"] == "timely_full" and r["scenario"] == "central"),
        key=lambda r: r["horizon_years"],
    )
    axes[1].plot([r["horizon_years"] for r in rows], [100 * r["probability"] for r in rows], marker="o", color="#287271")
    axes[1].set_title("2030及时全面介入：累计结果随期限变化")
    axes[1].set_xlabel("演化期限（年）")
    axes[1].set_ylabel("条件广泛控制率（%）")
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURE_PATH, bbox_inches="tight")
    plt.close(fig)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    results, ablations, trace, terrain, sipri = run_model()
    validate(results, ablations, terrain, sipri)
    write_csvs(results, ablations, trace)
    render(results)
    payload = {
        "model": "V4.2 four-major-actor Bayesian modular update with semi-Markov conflict regimes and homeland spillover, air-sea-land-cognition fields, major-minor MFG assistance, hierarchical command networks, kill-web/OODA, skill-stratified manpower, SIR supply-chain distress, production functions, orders, inflation, debt/default stress and coupled economic-political-social-military DEDS collapse",
        "seed": SEED,
        "paths_per_cell": PATHS,
        "months": MONTHS,
        "horizon_years": HORIZON_YEARS,
        "conflict_years": YEARS,
        "cases": CASES,
        "scenarios": SCENARIOS,
        "real_data_snapshot": {
            "terrain": terrain,
            "sipri_2025_constant_2024_usd_million": sipri,
            "prior": "V4.1 baseline-contested conditional probabilities, inheriting V4.0 alliance/economic/tail calibration",
        },
        "results": results,
        "ablations": ablations,
        "representative_trace_2030_timely_full": trace,
        "identification_warning": (
            "Terrain and SIPRI values are observed public inputs. Weapon effectiveness, readiness, "
            "undersea detection, wartime repair and network resilience remain interval structural priors. "
            "The V4.1 15-year cumulative prior is converted to a constant monthly baseline hazard for 5-30 year views. "
            "V4.2 is a modular Bayesian update to V4.1, not an independently identified war-frequency model. "
            "The 30-year view includes structural semi-Markov hazards for de-escalation, ceasefire, frozen conflict, rearmament, re-escalation and settlement. "
            "Named stress scenarios are conditional tests, not estimated occurrence probabilities."
        ),
        "verification": "PASS",
    }
    SUMMARY_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("TAIWAN_MULTIDOMAIN_V42_VERIFICATION: PASS")
    for row in results:
        if row["scenario"] == "central" and row["horizon_years"] in (15, 30):
            print(row["conflict_year"], row["case"], row["horizon_years"], f"{100 * row['probability']:.3f}%")
    print(SUMMARY_PATH)


if __name__ == "__main__":
    main()
