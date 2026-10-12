from __future__ import annotations

import csv
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
OLD_CSV = OUT / "taiwan_v53_mr_rd_cge_dq_sfc_results.csv"
NEW_CSV = OUT / "taiwan_v54_windows_accelerated_results.csv"
OLD_JSON = OUT / "taiwan_v53_mr_rd_cge_dq_sfc_simulation.json"
NEW_JSON = OUT / "taiwan_v54_windows_accelerated_simulation.json"
REPORT = OUT / "taiwan_v54_vs_v53_equivalence.json"


def numeric(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def compare_tree(left, right, path="root", differences=None):
    if differences is None:
        differences = []
    if numeric(left) and numeric(right):
        if math.isfinite(float(left)) and math.isfinite(float(right)):
            differences.append({
                "path": path,
                "left": float(left),
                "right": float(right),
                "absolute_gap": abs(float(left) - float(right)),
            })
        return differences
    if isinstance(left, dict) and isinstance(right, dict):
        for key in sorted(set(left) & set(right)):
            compare_tree(left[key], right[key], f"{path}.{key}", differences)
        return differences
    if isinstance(left, list) and isinstance(right, list):
        for index, (l_item, r_item) in enumerate(zip(left, right)):
            compare_tree(l_item, r_item, f"{path}[{index}]", differences)
    return differences


def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    old_rows = read_csv(OLD_CSV)
    new_rows = read_csv(NEW_CSV)
    if len(old_rows) != len(new_rows):
        raise AssertionError("CSV row count changed")
    identity = ("conflict_year", "horizon_years", "case", "trade_scenario")
    old_map = {tuple(row[key] for key in identity): row for row in old_rows}
    new_map = {tuple(row[key] for key in identity): row for row in new_rows}
    if old_map.keys() != new_map.keys():
        raise AssertionError("CSV row identities changed")
    csv_differences = []
    for key in old_map:
        for field in old_map[key]:
            if field in identity:
                continue
            try:
                left = float(old_map[key][field])
                right = float(new_map[key][field])
            except ValueError:
                continue
            csv_differences.append({
                "row": key,
                "field": field,
                "absolute_gap": abs(left - right),
            })

    old_json = json.loads(OLD_JSON.read_text(encoding="utf-8"))
    new_json = json.loads(NEW_JSON.read_text(encoding="utf-8"))
    aggregate_differences = compare_tree(
        old_json["aggregate"], new_json["aggregate"], "aggregate"
    )
    csv_max = max((item["absolute_gap"] for item in csv_differences), default=0.0)
    aggregate_max = max(
        (item["absolute_gap"] for item in aggregate_differences), default=0.0
    )
    report = {
        "comparison": "V5.3 baseline versus V5.4 native-Windows solver rewrite",
        "common_random_numbers": True,
        "csv_rows": len(old_rows),
        "csv_numeric_cells": len(csv_differences),
        "csv_max_absolute_gap": csv_max,
        "aggregate_numeric_cells": len(aggregate_differences),
        "aggregate_max_absolute_gap": aggregate_max,
        "csv_cells_above_1e_12": sum(
            item["absolute_gap"] > 1.0e-12 for item in csv_differences
        ),
        "aggregate_cells_above_1e_12": sum(
            item["absolute_gap"] > 1.0e-12 for item in aggregate_differences
        ),
        "largest_csv_differences": sorted(
            csv_differences, key=lambda item: item["absolute_gap"], reverse=True
        )[:20],
        "largest_aggregate_differences": sorted(
            aggregate_differences,
            key=lambda item: item["absolute_gap"], reverse=True,
        )[:20],
        "within_numerical_tolerance": (
            csv_max <= 1.0e-10 and aggregate_max <= 1.0e-10
        ),
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "csv_max_absolute_gap": csv_max,
        "aggregate_max_absolute_gap": aggregate_max,
        "within_numerical_tolerance": report["within_numerical_tolerance"],
    }))


if __name__ == "__main__":
    main()
