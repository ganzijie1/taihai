from __future__ import annotations

import json
from pathlib import Path

import ablate_taiwan_mechanisms_v48 as campaign


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "taiwan_v49_energy_ablation.json"


def main() -> None:
    audit = json.loads(
        (ROOT / "outputs" / "taiwan_v49_static_coupling_audit.json").read_text(encoding="utf-8")
    )
    if audit.get("status") != "PASS":
        raise RuntimeError("V4.9 energy ablation blocked by static audit")
    no_energy = campaign.run(None, energy_enabled=False)
    energy = campaign.run(None, energy_enabled=True)
    metrics = (
        "broad_control_weighted", "land_control_median",
        "population_control_median", "organized_defense_median",
        "china_min_stock_median", "defender_min_stock_median",
    )
    delta = {key: energy[key] - no_energy[key] for key in metrics}
    if not any(abs(value) > 1e-10 for value in delta.values()):
        raise RuntimeError("energy network did not alter campaign state")
    diagnostic = energy["energy_diagnostics"]
    if diagnostic["max_flow_min_cut_residual"] > 1e-9:
        raise RuntimeError("energy max-flow/min-cut residual")
    OUT.write_text(json.dumps({
        "model_version": "V4.9",
        "classification": "same-seed dynamic validation, not a formal probability estimate",
        "paths": campaign.PATHS,
        "months": campaign.MONTHS,
        "common_random_numbers": True,
        "without_energy": no_energy,
        "with_energy": energy,
        "with_minus_without": delta,
        "status": "PASS",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print("TAIWAN_V49_ENERGY_ABLATION: PASS")


if __name__ == "__main__":
    main()
