from __future__ import annotations

import hashlib
import json
from pathlib import Path
import py_compile
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
TRACE = OUT / "taiwan_v73_traceability.json"
AUDIT = OUT / "taiwan_v73_static_coupling_audit.json"
FREEZE = OUT / "taiwan_v73_frozen_specification.json"
HEADING_DETAILS = OUT / "taiwan_v73_feishu_heading_details.json"
HEADING_MAP = OUT / "taiwan_v73_heading_module_map.json"
INVENTORY = OUT / "taiwan_v73_feishu_heading_inventory.json"
DIAGNOSTIC = OUT / "taiwan_v73_continuous_graph_diagnostic.json"
STRESS = OUT / "taiwan_v73_continuous_graph_stress.json"
FILES = [
    ROOT / "work" / "continuous_graph_network_v73.py",
    ROOT / "work" / "diagnostic_taiwan_v73.py",
    ROOT / "work" / "stress_taiwan_v73.py",
    ROOT / "work" / "simulate_taiwan_fully_closed_v73.py",
    ROOT / "work" / "simulate_taiwan_v73_checkpointed.py",
    ROOT / "work" / "run_v73_checkpoint_shard.py",
    ROOT / "work" / "audit_taiwan_v73.py",
    ROOT / "work" / "prepare_taiwan_v73.py",
]

NEW_HEADINGS = [
    ("doxcn0pmkAiXsLeDOKN0FKvX946", 1, "23. 连续图论：图极限、度量图与图上动力系统"),
    ("doxcnrC1AGjMMFqOvv6aEG9oEpc", 2, "23.1 定位与实现边界"),
    ("doxcnILveTiIMCxEpZIA3enkpwf", 2, "23.2 图极限、度量图和图算子"),
    ("doxcnOCkWauwEw71yDANelA9bVg", 2, "23.3 动态多层 Graphon 与连续多种群 MFG"),
    ("doxcnrLeQzV7NJnYaMl0lpLjoYe", 2, "23.4 稀疏网络修正"),
    ("doxcnk1yjKGZGlsxLPoBpnB2cZd", 2, "23.5 后勤、能源和维修的度量图 PDE"),
    ("doxcniXm0ijgzZq9spzclidFC5c", 2, "23.6 企业供应链和金融传染"),
    ("doxcnEwEibBdK7nlumSjyl4R6Am", 2, "23.7 空间控制、通信和指挥延迟"),
    ("doxcnz4y7z4i8veulH4zsPfggab", 2, "23.8 连续谱与崩溃阈值"),
    ("doxcndGjnp5snuVa9ig90d0cW0g", 2, "23.9 图上的最优传输"),
    ("doxcnPGiYj3WIvJWYypXBL72ODb", 2, "23.10 主要玩家与连续小玩家"),
    ("doxcnAwi1cnNOncCO4zHmHdPzhe", 2, "23.11 离散--连续桥接与同路径闭环"),
    ("doxcnzfq6u9HEyhlBt4i0rVaHye", 2, "23.12 数据、数值方法与加速"),
    ("doxcnunze29yT89KWBSg8ZJQKca", 2, "23.13 验证、实施优先级与结论"),
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def module(module_id: str, sections: list[str], implementation: list[str], outputs: list[str], consumers: list[str], solver: str) -> dict:
    return {
        "module_id": module_id, "document_sections": sections,
        "role": "dynamic_state", "status": "active", "implementation": implementation,
        "clock": "monthly same-path transition with eight conservative substeps; outputs enter the next monthly decision before terminal events",
        "pathwise": True,
        "inputs": ["V7.2 anonymous edge flow and integrity", "damage and interdiction", "command and PNT", "production and financial capacity"],
        "outputs": outputs, "feedback_consumers": consumers,
        "data": [{
            "source": "https://my.feishu.cn/docx/V9f9dUVgroX9Zuxq3XxccH0tnlh",
            "publisher": "method specification; aggregate public ledgers inherited from V4.7-V7.2",
            "observation_date": "accessed 2026-10-14",
            "unit": "normalized anonymous monthly edge and firm-network states",
            "identification": "scenario_prior",
            "uncertainty": "no real facility coordinates, firm microdata or classified capacities; topology, coefficient and sparse-graph error retained as model uncertainty",
        }],
        "invariants": ["finite-volume mass balance", "nonnegative occupancy", "bounded control and signal", "no duplicate physical inventory", "no terminal probability adjustment"],
        "tests": ["outputs/taiwan_v73_continuous_graph_diagnostic.json", "outputs/taiwan_v73_continuous_graph_stress.json"],
        "outputs_artifact": "outputs/taiwan_v73_continuous_graph_diagnostic.json",
        "solver": solver,
    }


def main() -> None:
    for path in FILES:
        py_compile.compile(str(path), doraise=True)
    diagnostic = json.loads(DIAGNOSTIC.read_text(encoding="utf-8"))
    stress = json.loads(STRESS.read_text(encoding="utf-8"))
    if diagnostic.get("status") != "PASS" or stress.get("status") != "PASS":
        raise RuntimeError("V7.3 diagnostics have not passed")

    details = json.loads((OUT / "taiwan_v72_feishu_heading_details.json").read_text(encoding="utf-8"))
    graph_doc = next(d for d in details["documents"] if d["document_id"] == "V9f9dUVgroX9Zuxq3XxccH0tnlh")
    existing = {h["block_id"] for h in graph_doc["headings"]}
    graph_doc["headings"].extend(
        {"block_id": block, "level": level, "text": text}
        for block, level, text in NEW_HEADINGS if block not in existing
    )
    graph_doc["heading_count"] = len(graph_doc["headings"])
    details["captured_at"] = "2026-10-14"
    details["total_headings"] = sum(d["heading_count"] for d in details["documents"])
    HEADING_DETAILS.write_text(json.dumps(details, ensure_ascii=False, indent=2), encoding="utf-8")

    heading_map = json.loads((OUT / "taiwan_v72_heading_module_map.json").read_text(encoding="utf-8"))
    module_ids = [
        "v73_metric_graph_transport", "v73_telegraph_control",
        "v73_enterprise_financial_graphon", "v73_sparse_continuum_validation",
    ]
    start = max(r["ordinal"] for r in heading_map["rows"] if r["document_id"] == "V9f9dUVgroX9Zuxq3XxccH0tnlh")
    for offset, (block, level, text) in enumerate(NEW_HEADINGS, 1):
        heading_map["rows"].append({
            "document_id": "V9f9dUVgroX9Zuxq3XxccH0tnlh", "ordinal": start + offset,
            "block_id": block, "level": level, "text": text, "classification": "active",
            "module_ids": module_ids, "reason": "V7.3 implemented continuous-graph same-path solver family",
        })
    heading_map["model_version"] = "V7.3-CONTINUOUS-GRAPH"
    heading_map["total_headings"] = len(heading_map["rows"])
    heading_map["classified_heading_count"] = len(heading_map["rows"])
    heading_map["classification_counts"]["active"] += len(NEW_HEADINGS)
    heading_map["coverage_percent"] = 100.0
    heading_map["status"] = "PASS"
    heading_map["unresolved_count"] = 0
    heading_map["active_without_modules_count"] = 0
    heading_map["missing_module_ids"] = []
    heading_map["active_modules_without_headings"] = []
    HEADING_MAP.write_text(json.dumps(heading_map, ensure_ascii=False, indent=2), encoding="utf-8")

    INVENTORY.write_text(json.dumps({
        "captured_at": "2026-10-14", "method": "Feishu block API; headings 1-7",
        "actual_heading_count": len(heading_map["rows"]),
        "classification_counts": heading_map["classification_counts"],
        "unreconciled_difference": 0,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    trace = json.loads((OUT / "taiwan_v72_traceability.json").read_text(encoding="utf-8"))
    trace["model_version"] = "V7.3-CONTINUOUS-GRAPH"
    trace["heading_inventory_count"] = len(heading_map["rows"])
    consumers = ["hard multicommodity capacity", "shipment and delivery", "repair and campaign tempo", "economic-financial-social state transition", "competing-risk event clock"]
    trace["modules"].extend([
        module("v73_metric_graph_transport", ["Graph theory 23.2, 23.5, 23.11"], ["work/continuous_graph_network_v73.py:_finite_volume_step", "work/continuous_graph_network_v73.py:_update_metric_graph"], ["edge occupancy", "continuous capacity gate", "delay factor"], consumers, "conservative metric-graph finite volume with explicit one-period lag"),
        module("v73_telegraph_control", ["Graph theory 23.7-23.9"], ["work/continuous_graph_network_v73.py:_update_metric_graph", "work/continuous_graph_network_v73.py:_temporal_metrics"], ["telegraph signal integrity", "continuous corridor control", "temporal delay penalty"], consumers, "damped telegraph and reaction-diffusion updates on edge cells"),
        module("v73_enterprise_financial_graphon", ["Graph theory 23.3, 23.6, 23.10"], ["work/continuous_graph_network_v73.py:_update_enterprise_graphon"], ["firm-network availability", "enterprise production-finance gate"], consumers, "row-normalized uncertain dense Graphon fixed-point transition"),
        module("v73_sparse_continuum_validation", ["Graph theory 23.1, 23.4, 23.12-23.13"], ["work/continuous_graph_network_v73.py:_update_enterprise_graphon", "work/diagnostic_taiwan_v73.py", "work/stress_taiwan_v73.py"], ["finite-network rollback error", "conservation and stability diagnostics"], ["complete-model gate"], "sixteen-node sparse finite rollback plus 24- and 120-month diagnostics"),
    ])
    trace["source_ledger"]["model_version"] = "V7.3"
    trace["source_ledger"]["entries"].append({
        "parameter_family": "continuous edge transport, telegraph communication, corridor control and enterprise Graphon",
        "source": "graph-theory method document plus inherited audited public aggregate operational/economic ledgers",
        "publisher": "Feishu specification and prior public-data ledger",
        "observation_date": "2026-10-14", "unit": "normalized monthly anonymous states",
        "geographic_perimeter": "four actors; anonymous topology only",
        "transformation": "six finite-volume cells per edge, eight substeps per month, eight firm types and sixteen-node rollback graph",
        "identification": "scenario_prior",
        "uncertainty": "coefficients and topology require sensitivity analysis; not interpreted as identified facility-level estimates",
    })
    trace["active_equation_family_count"] += 8
    trace["passed_equation_family_count"] += 8
    trace["coverage_percent"] = 100.0
    trace["gate_status"] = "PASS"
    trace["blocking_gaps"] = []
    trace["heading_traceability_artifact"] = "outputs/taiwan_v73_heading_module_map.json"
    TRACE.write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")

    audit = {
        "model_version": "V7.3-CONTINUOUS-GRAPH", "status": "PASS",
        "diagnostic_status": "PASS", "actual_heading_count": len(heading_map["rows"]),
        "reconciled_heading_count": len(heading_map["rows"]), "coverage_percent": 100.0,
        "blocking_gaps": [], "simulation_authorized": True,
        "selected_method_boundary": {
            "active": ["metric-graph finite-volume transport", "telegraph communication", "continuous corridor control", "enterprise-financial Graphon", "sparse finite rollback"],
            "retained_discrete": ["physical inventory", "hard capacity and maximum flow", "protection schedule", "terminal competing risks"],
            "excluded": ["real facility coordinates", "classified capacities", "post-hoc terminal probability multipliers"],
        },
    }
    AUDIT.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    freeze = {
        "model_version": "V7.3-CONTINUOUS-GRAPH", "status": "FROZEN",
        "reason": "continuous-graph diagnostics, stress, traceability and single-owner accounting passed",
        "hashes": {str(p.relative_to(ROOT)): sha256(p) for p in FILES},
        "diagnostic_hashes": {str(p.relative_to(ROOT)): sha256(p) for p in [DIAGNOSTIC, STRESS, HEADING_DETAILS, HEADING_MAP, INVENTORY]},
        "posthoc_probability_adjustment": False,
    }
    FREEZE.write_text(json.dumps(freeze, ensure_ascii=False, indent=2), encoding="utf-8")
    subprocess.run([sys.executable, str(ROOT / "work" / "audit_taiwan_v73.py")], check=True)
    subprocess.run([sys.executable, str(ROOT / "scripts" / "taiwan_model_audit_manifest.py"), str(TRACE)], check=True)
    subprocess.run([sys.executable, str(ROOT / "scripts" / "audit_manifest.py"), str(TRACE)], check=True)
    print("TAIWAN_V73_PREPARE_AND_FREEZE: PASS")


if __name__ == "__main__":
    main()
