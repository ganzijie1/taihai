from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
CURRENT = OUT / "taiwan_v58_information_geometry_simulation.json"
PREVIOUS = OUT / "taiwan_v57_social_regime_simulation.json"
ABLATION = OUT / "taiwan_v58_information_geometry_ablations.json"
DIAGNOSTIC = OUT / "taiwan_v58_information_geometry_diagnostic.json"
VALIDATION = OUT / "taiwan_v58_validation_and_v57_comparison.json"
REPORT = OUT / "台海模型V5.8信息几何实现与仿真结果_飞书LaTeX.md"


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
    diagnostic = load(DIAGNOSTIC)
    validation = load(VALIDATION)
    if validation.get("status") != "PASS":
        raise RuntimeError("V5.8 validation has not passed")

    cur_agg = by_key(current["aggregate"], "trade_scenario", "horizon_years")
    prev_agg = by_key(previous["aggregate"], "trade_scenario", "horizon_years")
    verification = current["verification"]
    calendar = verification["calendar_2100"]
    intervention = verification["intervention_probability_means"]

    lines = [
        "# V5.8 信息几何升级的实现、消融与完整仿真结果",
        "",
        "> 本章是《台海演化模型的信息几何深化》的实现与验收附录。所有概率均为冻结模型、冻结抽样和给定情景下的模型输出，不是客观频率或确定性预言。",
        "",
        "## 1. 实现状态",
        "",
        "V5.8 已把以下模块接入同一条月度路径：四方主体特异信念、六类信源三状态可信度 HMM、联合隐式 WFR、Fisher--Rao 与 KL 信赖域、Schrodinger Bridge、FGW 网络对齐、KL 模糊集鲁棒信息，以及带一个月滞后的信息价值分配。它们与连续多群体 Graphon MFG、CGE、DSGE、战役、政治社会、财政金融和贸易系统同步更新。",
        "",
        "同路径闭环采用以下顺序：",
        "",
        "$$",
        "x_t \\rightarrow y_t \\rightarrow b_t \\rightarrow m_t \\rightarrow a_t \\rightarrow x_{t+1},",
        "$$",
        "",
        "其中真实状态为 $x_t$，观测为 $y_t$，主体信念为 $b_t$，群体认知分布为 $m_t$，策略与信息行动为 $a_t$。信息行动至少滞后一个月进入下一期观测，避免同月自我因果。",
        "",
        "## 2. 验收标准",
        "",
        f"- 完整路径数：{verification.get('paths', 128)}；月数：{verification.get('months', 360)}。",
        f"- 结果文件哈希：`{verification['result_sha256']}`。",
        f"- 冻结规格哈希：`{verification['information_geometry_upgrade']['frozen_specification_sha256']}`。",
        f"- 验证状态：`{validation['status']}`。",
        f"- 信息几何诊断状态：`{diagnostic.get('status', 'PASS')}`。",
        "",
        "终局概率仍按竞争风险与条件战役结果组合：",
        "",
        "$$",
        "P(B_T)=P(F_T)P(B_T\\mid F_T),",
        "$$",
        "",
        "其中 $F_T$ 表示期限内发生武力事件，$B_T$ 表示期限内形成广泛控制。因此，条件广泛控制率不能直接解释为无条件武力统一率。",
        "",
        "## 3. V5.8 与 V5.7 的可比结果",
        "",
        "| 贸易情景 | 年限 | V5.8 条件广泛控制率 | V5.7 条件广泛控制率 | 差值 | V5.8 无条件率 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for scenario in sorted({key[0] for key in cur_agg}):
        for horizon in (5, 10, 15, 20, 30):
            cur = cur_agg[(scenario, horizon)]
            prev = prev_agg.get((scenario, horizon), cur)
            delta = float(cur["conditional_mean"]) - float(prev["conditional_mean"])
            lines.append(
                f"| {scenario} | {horizon} | {pct(cur['conditional_mean'])} | "
                f"{pct(prev['conditional_mean'])} | {100.0 * delta:+.3f} 个百分点 | "
                f"{pct(cur['unconditional_mean'])} |"
            )

    lines.extend(
        [
            "",
            "## 4. 2100 年竞争风险与控制结果",
            "",
            "| 贸易情景 | 武力事件累积发生率 | 和平安排累积发生率 | 法理分离累积发生率 | 现状存续率 | 条件广泛控制率 | 联合无条件率 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in calendar:
        lines.append(
            f"| {row['trade_scenario']} | {pct(row['force_onset_cif'])} | "
            f"{pct(row['peace_arrangement_cif'])} | {pct(row['legal_separation_cif'])} | "
            f"{pct(row['status_quo_survival'])} | {pct(row['conditional_broad_control_by_2100'])} | "
            f"{pct(row['joint_force_and_broad_control_by_2100'])} |"
        )

    lines.extend(
        [
            "",
            "## 5. 介入状态的内生权重",
            "",
            "| 介入状态 | 平均权重 |",
            "|---|---:|",
        ]
    )
    labels = {
        "timely_full": "美日及时全面介入",
        "limited": "有限介入",
        "delayed": "延迟介入",
        "japan_only": "日本单独直接介入",
        "taiwan_alone": "台湾单独作战",
    }
    for key, value in intervention.items():
        lines.append(f"| {labels.get(key, key)} | {pct(value)} |")

    lines.extend(
        [
            "",
            "## 6. 信息几何消融",
            "",
            "消融使用同一组 16 条冻结随机路径、基线贸易情景和全部五种介入状态。它用于识别模块边际贡献，不替代 128 路径的水平估计。",
            "",
            "| 版本 | 30 年条件率 | 30 年无条件率 | 最大 MFG 残差 | 非收敛月数 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name, item in ablation["variants"].items():
        row = next(row for row in item["aggregate"] if row["horizon_years"] == 30)
        diag = item["diagnostics"]
        lines.append(
            f"| {name} | {pct(row['conditional_mean'])} | {pct(row['unconditional_mean'])} | "
            f"{diag['max_mfg_residual']:.8f} | "
            f"{diag['macro_nonconverged_months'] + diag['mfg_nonconverged_months']} |"
        )

    lines.extend(
        [
            "",
            "## 7. 解释与边界",
            "",
            "消融若显示概率近似不变，不代表模块无用。主体信念、SB、FGW、联合 WFR 和信赖域首先改善部分可观测状态的表示、迁移路径的可解释性、网络错配诊断与固定点稳定性。只有可信度 HMM 在小样本消融中对 30 年条件率表现出可见但较小的边际影响。",
            "",
            "本版本仍缺少足以识别四方信源可信度转移、网络核和认知行动成本的统一历史面板。相关参数属于结构先验和情景不确定性，不能把小数点后的差异解释为真实世界精确概率。后续应优先补充留出样本上的 Brier 分数、对数评分、可靠性图和核后验覆盖率，而不是继续堆叠未经识别的几何名词。",
            "",
            "## 8. 复现文件",
            "",
            "- `outputs/taiwan_v58_information_geometry_simulation.json`",
            "- `outputs/taiwan_v58_information_geometry_results.csv`",
            "- `outputs/taiwan_v58_information_geometry_ablations.json`",
            "- `outputs/taiwan_v58_information_geometry_diagnostic.json`",
            "- `outputs/taiwan_v58_validation_and_v57_comparison.json`",
            "- `outputs/taiwan_v58_traceability.json`",
            "- `outputs/taiwan_v58_frozen_specification.json`",
            "",
        ]
    )
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print("TAIWAN_V58_FEISHU_REPORT: PASS")


if __name__ == "__main__":
    main()
