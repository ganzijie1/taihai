from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
CURRENT = OUT / "taiwan_v61_sectoral_simulation.json"
PREVIOUS = OUT / "taiwan_v60_riemannian_simulation.json"
ABLATION = OUT / "taiwan_v61_sectoral_ablations.json"
VALIDATION = OUT / "taiwan_v61_validation_and_v60_comparison.json"
REPORT = OUT / "台海模型V6.1三次产业闭环仿真结果_飞书LaTeX.md"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def pct(value: float) -> str:
    return f"{100.0 * float(value):.3f}%"


def by_key(rows: list[dict]) -> dict[tuple, dict]:
    return {(row["trade_scenario"], row["horizon_years"]): row for row in rows}


def main() -> None:
    required = [CURRENT, PREVIOUS, ABLATION, VALIDATION]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"required validated artifacts missing: {missing}")
    current = load(CURRENT)
    previous = load(PREVIOUS)
    ablation = load(ABLATION)
    validation = load(VALIDATION)
    if validation.get("status") != "PASS":
        raise RuntimeError("V6.1 validation has not passed")
    cur = by_key(current["aggregate"])
    old = by_key(previous["aggregate"])
    verification = current["verification"]
    residuals = validation["maximum_residuals_and_bounds"]
    lines = [
        "# V6.1 三次产业物理经济闭环：实现、验证与正式仿真结果",
        "",
        "> 本节报告冻结模型在给定情景和参数不确定性下的条件结果，不把模拟概率解释为客观预言，也不进行事后概率修正。",
        "",
        "## 1. 实现范围",
        "",
        "V6.1 保留 V6.0 的战役、联盟介入、政治、社会、金融、能源、贸易、CGE、DSGE、HANK、MFG、DEDS 与竞争风险模块，并新增以下同路径状态：",
        "",
        "- 农业：天气、水、土壤、土地用途、生物产量和粮食安全；粮食库存仍由 DQ-SFC 唯一持有。",
        "- 工业：中间投入瓶颈、设备可用率、维修队列、企业进入退出、质量和累积学习。",
        "- 服务业：不可储存服务队列、利用率、等待时间、数字采用和公共服务财政容量。",
        "- 结构转型：Engel 需求迁移、部门劳动再配置、Baumol 成本压力和 Kaldor-Verdoorn 学习。",
        "",
        "统一月度链路为：",
        "",
        "$$",
        "(Y_t^p,D_t,N_t) \to (A_t,I_t,S_t,L_t) \to Q_t^{real} \to (SAM_t,SFC_t) \to (Y_{t+1}^p,D_{t+1},N_{t+1}).",
        "$$",
        "",
        "其中计划产出先受三次产业物理状态约束，再进入实际交易、SAM 与 SFC；下一月 CGE、DSGE、金融、社会和战役模块读取这些已实现状态。",
        "",
        "## 2. 核心方程与状态所有权",
        "",
        "农业生物产量采用有界乘法结构：",
        "",
        "$$",
        "Y_{a,t}=s_t f(W_t) H_t^{\alpha_H} Soil_t^{\alpha_S} L_{a,t}^{\alpha_L} Log_t^{\alpha_G}(1-\delta_D D_t).",
        "$$",
        "",
        "工业产能由最短板决定，并乘企业数量和质量项：",
        "",
        "$$",
        "Cap_{i,t}=\min\{O_t,M_t,L_{i,t},Log_t Pay_t\}\,n_t^{\eta_n}q_t^{\eta_q}.",
        "$$",
        "",
        "服务业使用不可储存队列：",
        "",
        "$$",
        "B_{t+1}=\max\{0,\rho_B B_t+D_t-Cap_t\}, \qquad Inv_{s,t+1}=0.",
        "$$",
        "",
        "劳动份额在单纯形内缓慢迁移：",
        "",
        "$$",
        "\ell_{r,t+1}=(1-\mu_t)\ell_{r,t}+\mu_t\frac{\bar\ell_r\exp(\beta z_{r,t})}{\sum_j\bar\ell_j\exp(\beta z_{j,t})}.",
        "$$",
        "",
        "## 3. 冻结与验证",
        "",
        f"- 冻结规格哈希：`{verification['three_sector_upgrade']['frozen_specification_sha256']}`",
        f"- 结果 CSV 哈希：`{verification['result_sha256']}`",
        f"- 验证状态：`{validation['status']}`",
        f"- 最大宏观外层残差：{residuals['macro_outer']:.8f}",
        f"- 最大 CGE 残差：{residuals['cge']:.8f}",
        f"- 最大 DSGE 残差：{residuals['dsge']:.8f}",
        f"- 最大 MFG 残差：{residuals['mfg']:.8f}",
        f"- 部门劳动份额残差：{residuals['labor_share']:.3e}",
        f"- 土地份额残差：{residuals['land_share']:.3e}",
        f"- 服务库存残差：{residuals['service_stock']:.3e}",
        f"- 部门到主体核算残差：{residuals['sector_accounting']:.3e}",
        "",
        "## 4. V6.1 与 V6.0 的正式结果比较",
        "",
        "| 贸易情景 | 年限 | V6.1 条件广泛控制率 | V6.0 条件广泛控制率 | 条件差值 | V6.1 联合无条件率 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for scenario in sorted({key[0] for key in cur}):
        for horizon in (5, 10, 15, 20, 30):
            now = cur[(scenario, horizon)]
            before = old[(scenario, horizon)]
            delta = float(now["conditional_mean"]) - float(before["conditional_mean"])
            lines.append(
                f"| {scenario} | {horizon} | {pct(now['conditional_mean'])} | "
                f"{pct(before['conditional_mean'])} | {100.0 * delta:+.3f} 个百分点 | "
                f"{pct(now['unconditional_mean'])} |"
            )
    lines.extend([
        "",
        "## 5. 延伸至 2100 年的竞争风险分解",
        "",
        "| 贸易情景 | 武力事件 | 和平安排 | 法理分离 | 现状延续 | 条件广泛控制率 | 联合无条件率 |",
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
        "消融使用冻结后验的共同前 16 条路径、基线贸易情景、五种介入状态和完整 360 个月。它用于识别模块方向与敏感度，不替代 128 路径正式估计。",
        "",
        "| 变体 | 30 年条件率 | 30 年无条件率 | 最大宏观残差 | 非收敛月数 |",
        "|---|---:|---:|---:|---:|",
    ])
    for item in ablation["variants"]:
        lines.append(
            f"| {item['variant']} | {pct(item['conditional_30y'])} | "
            f"{pct(item['unconditional_30y'])} | {item['max_macro_residual']:.8f} | "
            f"{item['macro_nonconverged_months']} |"
        )
    lines.extend([
        "",
        "## 7. 解释边界",
        "",
        "本升级改变的是生产、短缺、价格、财政、社会和战役之间的传导机制。结果差异必须通过同随机数消融和残差检验解释，不能直接归因于单一产业参数。天气方差、进入退出率、数字采用速度和结构转型弹性仍属于有界结构先验，因此应阅读为情景条件结果。",
        "",
        "## 8. 可复现文件",
        "",
        "- `outputs/taiwan_v61_sectoral_simulation.json`",
        "- `outputs/taiwan_v61_sectoral_results.csv`",
        "- `outputs/taiwan_v61_sectoral_ablations.json`",
        "- `outputs/taiwan_v61_validation_and_v60_comparison.json`",
        "- `outputs/taiwan_v61_traceability.json`",
        "- `outputs/taiwan_v61_frozen_specification.json`",
        "- `outputs/taiwan_v61_sectoral_source_ledger.json`",
        "",
    ])
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print("TAIWAN_V61_FEISHU_REPORT: PASS")


if __name__ == "__main__":
    main()
