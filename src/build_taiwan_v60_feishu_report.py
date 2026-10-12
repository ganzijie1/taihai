from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
CURRENT = OUT / "taiwan_v60_riemannian_simulation.json"
PREVIOUS = OUT / "taiwan_v59_network_geometry_simulation.json"
ABLATION = OUT / "taiwan_v60_riemannian_ablations.json"
DIAGNOSTIC = OUT / "taiwan_v60_riemannian_diagnostic.json"
VALIDATION = OUT / "taiwan_v60_validation_and_v59_comparison.json"
REPORT = OUT / "台海模型V6.0黎曼几何深化实现与仿真结果_飞书LaTeX.md"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def pct(value: float) -> str:
    return f"{100.0 * float(value):.3f}%"


def by_key(rows: list[dict], *keys: str) -> dict[tuple, dict]:
    return {tuple(row[key] for key in keys): row for row in rows}


def main() -> None:
    required = [CURRENT, PREVIOUS, ABLATION, DIAGNOSTIC, VALIDATION]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"required validated artifacts missing: {missing}")
    current = load(CURRENT)
    previous = load(PREVIOUS)
    ablation = load(ABLATION)
    validation = load(VALIDATION)
    if validation.get("status") != "PASS":
        raise RuntimeError("V6.0 validation has not passed")
    cur_agg = by_key(current["aggregate"], "trade_scenario", "horizon_years")
    prev_agg = by_key(previous["aggregate"], "trade_scenario", "horizon_years")
    verification = current["verification"]
    residuals = validation["maximum_residuals_and_bounds"]
    lines = [
        "# V6.0 黎曼几何深化：实现、消融与完整仿真结果",
        "",
        "> 本章是黎曼几何深化方法文档的实现与验收附录。所有概率都是冻结模型和给定情景下的条件输出，不是客观频率或确定性预言。",
        "",
        "## 1. 活动实现范围",
        "",
        "V6.0 在 V5.9 完整闭环上实现五项升级：Fisher 平方根球面上的正则化黎曼 Anderson、主要玩家与普通群体的 Fisher--Rao 策略信赖域、PNT 四信源自适应 Bures--Wasserstein 重心、常态—危机—战争—恢复四套局部切空间 POD，以及物流—产出—金融—指挥—能源的正定乘积流形度量。",
        "",
        "Kähler 几何、半黎曼优化、曲率直接修正终局概率和没有下游消费者的全局联络仍明确不进入活动结果路径。",
        "",
        "## 2. 同路径闭环",
        "",
        "$$",
        "x_t \\longrightarrow (\\operatorname{Log}_{x_t},\\mathcal T_t,g_t) \\longrightarrow \\xi_t \\longrightarrow R_{x_t}(\\xi_t) \\longrightarrow x_{t+1}.",
        "$$",
        "",
        "其中 $x_t$ 仍由原模块拥有；黎曼模块只改变合法的求解方向、融合和降阶，不建立事后概率修正层。",
        "",
        "## 3. 数值验收",
        "",
        f"- 冻结规格哈希：`{verification['riemannian_upgrade']['frozen_specification_sha256']}`。",
        f"- 结果文件哈希：`{verification['result_sha256']}`。",
        f"- 验证状态：`{validation['status']}`。",
        f"- 最大宏观外层残差：{residuals['macro_outer']:.8f}。",
        f"- 最大 CGE 残差：{residuals['cge']:.8f}。",
        f"- 最大 DSGE 残差：{residuals['dsge']:.8f}。",
        f"- 最大 MFG 固定点残差：{residuals['mfg']:.8f}。",
        f"- RAM 平行运输切向残差：{residuals['ram_tangency']:.3e}。",
        f"- RAM 球面残差：{residuals['ram_sphere']:.3e}。",
        f"- PNT 重心残差：{residuals['pnt_barycenter']:.3e}。",
        f"- 乘积度量最小特征值：{residuals['product_metric_minimum_eigenvalue']:.6f}。",
        f"- 制度切换次数：{residuals['regime_transition_count']}。",
        "",
        "## 4. V6.0 与 V5.9 的可比结果",
        "",
        "| 贸易情景 | 年限 | V6.0 条件广泛控制率 | V5.9 条件广泛控制率 | 差值 | V6.0 无条件率 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for scenario in sorted({key[0] for key in cur_agg}):
        for horizon in (5, 10, 15, 20, 30):
            current_row = cur_agg[(scenario, horizon)]
            previous_row = prev_agg[(scenario, horizon)]
            delta = float(current_row["conditional_mean"]) - float(previous_row["conditional_mean"])
            lines.append(
                f"| {scenario} | {horizon} | {pct(current_row['conditional_mean'])} | "
                f"{pct(previous_row['conditional_mean'])} | {100.0 * delta:+.3f} 个百分点 | "
                f"{pct(current_row['unconditional_mean'])} |"
            )
    lines.extend([
        "",
        "## 5. 延伸到 2100 年的竞争风险",
        "",
        "| 贸易情景 | 武力事件 | 和平安排 | 法理分离 | 现状存续 | 条件广泛控制率 | 联合无条件率 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in verification["calendar_2100"]:
        lines.append(
            f"| {row['trade_scenario']} | {pct(row['force_onset_cif'])} | "
            f"{pct(row['peace_arrangement_cif'])} | {pct(row['legal_separation_cif'])} | "
            f"{pct(row['status_quo_survival'])} | {pct(row['conditional_broad_control_by_2100'])} | "
            f"{pct(row['joint_force_and_broad_control_by_2100'])} |"
        )
    lines.extend([
        "",
        "## 6. 同随机数消融",
        "",
        "消融使用 16 条冻结路径、基线贸易情景和全部五种介入状态，只识别模块边际影响，不替代 128 路径水平估计。",
        "",
        "| 版本 | 30 年条件率 | 30 年无条件率 | 最大 MFG 残差 | 非收敛月数 |",
        "|---|---:|---:|---:|---:|",
    ])
    for item in ablation["variants"]:
        lines.append(
            f"| {item['variant']} | {pct(item['conditional_30y'])} | "
            f"{pct(item['unconditional_30y'])} | {item['max_mfg_residual']:.8f} | "
            f"{item['mfg_nonconverged_months']} |"
        )
    lines.extend([
        "",
        "## 7. 解释边界",
        "",
        "黎曼升级首先是求解与状态表示改进。只有当共同随机数消融显示稳定差异，并且差异大于蒙特卡洛误差和求解容差时，才能讨论它对终局概率的结构影响。PNT 信源权重、制度阈值和乘积度量交叉项仍含结构先验，不得解释为已由战争数据因果识别。",
        "",
        "## 8. 复现文件",
        "",
        "- `outputs/taiwan_v60_riemannian_simulation.json`",
        "- `outputs/taiwan_v60_riemannian_results.csv`",
        "- `outputs/taiwan_v60_riemannian_ablations.json`",
        "- `outputs/taiwan_v60_riemannian_diagnostic.json`",
        "- `outputs/taiwan_v60_validation_and_v59_comparison.json`",
        "- `outputs/taiwan_v60_traceability.json`",
        "- `outputs/taiwan_v60_frozen_specification.json`",
        "- `outputs/taiwan_v60_riemannian_source_ledger.json`",
        "",
    ])
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print("TAIWAN_V60_FEISHU_REPORT: PASS")


if __name__ == "__main__":
    main()
