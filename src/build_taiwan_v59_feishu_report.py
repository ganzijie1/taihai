from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
CURRENT = OUT / "taiwan_v59_network_geometry_simulation.json"
PREVIOUS = OUT / "taiwan_v58_information_geometry_simulation.json"
ABLATION = OUT / "taiwan_v59_network_geometry_ablations.json"
DIAGNOSTIC = OUT / "taiwan_v59_network_geometry_diagnostic.json"
VALIDATION = OUT / "taiwan_v59_validation_and_v58_comparison.json"
REPORT = OUT / "台海模型V5.9网络几何实现与仿真结果_飞书LaTeX.md"


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
        raise RuntimeError("V5.9 validation has not passed")
    cur_agg = by_key(current["aggregate"], "trade_scenario", "horizon_years")
    prev_agg = by_key(previous["aggregate"], "trade_scenario", "horizon_years")
    verification = current["verification"]
    residuals = validation["maximum_residuals_and_bounds"]
    lines = [
        "# V5.9 网络几何升级的实现、消融与完整仿真结果",
        "",
        "> 本章是网络几何方法文档的实现与验收附录。所有概率均为冻结模型与给定情景下的条件输出，不是客观频率或确定性预言。",
        "",
        "## 1. 活动实现范围",
        "",
        "V5.9 按方法文档第 19 节的最小可行范围实现四项：强度与转移核分离、四层超拉普拉斯、谱间隙/导通率/加权 Forman 曲率，以及信息—身份层的标量 stalk sheaf Laplacian。它们在同一条月度路径中更新，并通过传播强度和转移核进入连续多群体 Graphon MFG。",
        "",
        "有向 Ollivier 曲率、Hodge 分解、超图/单纯复形、zigzag persistence 和完整 GW 近端核估计仍属于第二批方法，不参与本章概率。",
        "",
        "## 2. 同路径闭环",
        "",
        "$$",
        "W_t \longrightarrow (s_t,P_t) \longrightarrow (L_t^{supra},\kappa_t,L_t^{sheaf}) \longrightarrow F_t \longrightarrow a_t \longrightarrow W_{t+1}.",
        "$$",
        "",
        "这里 $W_t$ 为原始多层网络，$s_t$ 为强度，$P_t$ 为行随机转移核，$F_t$ 为进入 MFG 的状态—行动场，$a_t$ 为群体与主要玩家行动。不存在事后概率修正。",
        "",
        "## 3. 数值验收",
        "",
        f"- 冻结规格哈希：`{verification['network_geometry_upgrade']['frozen_specification_sha256']}`。",
        f"- 结果文件哈希：`{verification['result_sha256']}`。",
        f"- 验证状态：`{validation['status']}`。",
        f"- 转移核最大行和残差：{residuals['transition_row']:.3e}。",
        f"- 最小网络强度：{residuals['minimum_intensity']:.6f}。",
        f"- 最小 actor 谱间隙：{residuals['minimum_actor_spectral_gap']:.6f}。",
        f"- 最小 supra 谱间隙：{residuals['minimum_supra_spectral_gap']:.6f}。",
        f"- sheaf Laplacian 最小特征值：{residuals['minimum_sheaf_eigenvalue']:.3e}。",
        f"- 最大 MFG 固定点残差：{residuals['mfg']:.6f}。",
        "",
        "## 4. V5.9 与 V5.8 的可比结果",
        "",
        "| 贸易情景 | 年限 | V5.9 条件广泛控制率 | V5.8 条件广泛控制率 | 差值 | V5.9 无条件率 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for scenario in sorted({key[0] for key in cur_agg}):
        for horizon in (5, 10, 15, 20, 30):
            cur = cur_agg[(scenario, horizon)]
            prev = prev_agg[(scenario, horizon)]
            delta = float(cur["conditional_mean"]) - float(prev["conditional_mean"])
            lines.append(
                f"| {scenario} | {horizon} | {pct(cur['conditional_mean'])} | "
                f"{pct(prev['conditional_mean'])} | {100.0 * delta:+.3f} 个百分点 | "
                f"{pct(cur['unconditional_mean'])} |"
            )
    lines.extend([
        "",
        "## 5. 2100 年竞争风险与广泛控制",
        "",
        "| 贸易情景 | 武力事件累积发生率 | 和平安排 | 法理分离 | 现状存续 | 条件广泛控制率 | 联合无条件率 |",
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
        "## 6. 同随机数网络几何消融",
        "",
        "消融使用 16 条冻结路径、基线贸易情景和全部五种介入状态，只用于识别模块边际影响，不替代 128 路径水平估计。",
        "",
        "| 版本 | 30 年条件率 | 30 年无条件率 | 最大 MFG 残差 | 非收敛月数 |",
        "|---|---:|---:|---:|---:|",
    ])
    for name, item in ablation["variants"].items():
        row = next(row for row in item["aggregate"] if row["horizon_years"] == 30)
        diag = item["diagnostics"]
        lines.append(
            f"| {name} | {pct(row['conditional_mean'])} | {pct(row['unconditional_mean'])} | "
            f"{diag['max_mfg_residual']:.8f} | "
            f"{diag['macro_nonconverged_months'] + diag['mfg_nonconverged_months']} |"
        )
    lines.extend([
        "",
        "## 7. 解释边界",
        "",
        "网络几何参数目前主要是结构先验。曲率、谱间隙与 sheaf 一致性通过网络传播状态进入均值场，而不是直接加到终局概率。即便概率变化较小，这些模块仍可能改善断裂识别、网络恢复和有限网络回代；反之，较大的概率变化也必须通过消融与外部验证后才可解释。",
        "",
        "## 8. 复现文件",
        "",
        "- `outputs/taiwan_v59_network_geometry_simulation.json`",
        "- `outputs/taiwan_v59_network_geometry_results.csv`",
        "- `outputs/taiwan_v59_network_geometry_ablations.json`",
        "- `outputs/taiwan_v59_network_geometry_diagnostic.json`",
        "- `outputs/taiwan_v59_validation_and_v58_comparison.json`",
        "- `outputs/taiwan_v59_traceability.json`",
        "- `outputs/taiwan_v59_frozen_specification.json`",
        "",
    ])
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print("TAIWAN_V59_FEISHU_REPORT: PASS")


if __name__ == "__main__":
    main()
