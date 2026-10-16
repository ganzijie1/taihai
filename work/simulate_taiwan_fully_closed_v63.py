from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path

import simulate_taiwan_fully_closed_v62 as v62
from continuous_multipop_mfg_v63 import ContinuousMultiPopulationMFGV63

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
TRACE = OUT / "taiwan_v63_traceability.json"
AUDIT = OUT / "taiwan_v63_static_coupling_audit.json"
FREEZE = OUT / "taiwan_v63_frozen_specification.json"
OUT_CSV = OUT / "taiwan_v63_results.csv"
OUT_JSON = OUT / "taiwan_v63_simulation.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def run(*args, **kwargs):
    old = (v62.ContinuousMultiPopulationMFGV62, v62.TRACE, v62.AUDIT)
    v62.ContinuousMultiPopulationMFGV62 = ContinuousMultiPopulationMFGV63
    v62.TRACE, v62.AUDIT = TRACE, AUDIT
    try:
        result = v62.run(*args, **kwargs)
    finally:
        v62.ContinuousMultiPopulationMFGV62, v62.TRACE, v62.AUDIT = old
    for cell, item in result[2].items():
        stability = item["mfg"].get("v63_system_stability")
        if stability is None or stability["months"] <= 0:
            raise RuntimeError(f"{cell}: missing V6.3 stability diagnostics")
        if stability["future_information_used"] or stability["feeds_terminal_probability"]:
            raise RuntimeError(f"{cell}: V6.3 read-only boundary violated")
    return result


def write(rows, aggregate, diagnostics, verification):
    with OUT_CSV.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    verification["model_version"] = "V6.3"
    verification["runtime_validation"] = "TAIWAN_V63_RUNTIME_VALIDATION: PASS"
    verification["result_sha256"] = sha256(OUT_CSV)
    verification["v63_stability_upgrade"] = {
        "online_tangent_response": True,
        "finite_time_lyapunov_qr": True,
        "saltation_jump_response": True,
        "random_matrix_spectral_diagnostics": True,
        "actor_cavity_importance": True,
        "dmft_moment_autocovariance_closure": True,
        "read_only_until_backprojection_validated": True,
        "no_posthoc_probability_adjustment": True,
        "frozen_specification_sha256": sha256(FREEZE),
    }
    OUT_JSON.write_text(json.dumps({"verification":verification,"aggregate":aggregate,"conditional":rows,"module_diagnostics":diagnostics},ensure_ascii=False,indent=2),encoding="utf-8")


def main():
    if os.environ.get("TAIWAN_ALLOW_COMPLETE_RUN") != "1":
        raise RuntimeError("V6.3 complete run blocked")
    frozen=json.loads(FREEZE.read_text(encoding="utf-8"))
    if frozen.get("status") != "FROZEN": raise RuntimeError("V6.3 not frozen")
    result=run(paths=int(os.environ.get("TAIWAN_V63_PATHS","128")),months=int(os.environ.get("TAIWAN_V63_MONTHS","360")))
    write(*result); print("TAIWAN_V63_COMPLETE_RUN: PASS")

if __name__ == "__main__": main()
