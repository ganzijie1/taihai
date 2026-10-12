from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "work"
OUT = ROOT / "outputs"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main() -> None:
    formal = OUT / "taiwan_v56_frozen_specification.json"
    validation = OUT / "taiwan_v56_validation_and_v55_comparison.json"
    formal_payload = json.loads(formal.read_text(encoding="utf-8"))
    validation_payload = json.loads(validation.read_text(encoding="utf-8"))
    if formal_payload.get("status") != "FROZEN" or validation_payload.get("status") != "PASS":
        raise RuntimeError("formal V5.6 run must remain frozen and validated")
    files = [
        WORK / "ablate_taiwan_v56.py", WORK / "freeze_taiwan_v56_ablation.py",
        formal, validation, OUT / "taiwan_v55_econometric_draws.npz",
    ]
    payload = {
        "model_version": "V5.6-ablation-amendment-2", "status": "FROZEN",
        "reason": "two pre-outcome entry failures corrected: 32-path frozen-draw subsetting and internal peace-baseline scenario key selection",
        "formal_run_unchanged": True,
        "formal_freeze_sha256": digest(formal),
        "contract": {
            "paths": 32, "months": 360, "trade_scenarios": ["baseline"],
            "intervention_cases": 5, "common_random_numbers": True,
            "draw_subset": "first 32 rows of every frozen 128-path draw array",
            "variants": ["full", "finsler_lie", "wfr_information", "spd_joseph", "port_hamiltonian", "slow_manifold", "saltation_sensitivity"],
        },
        "files": {str(path.relative_to(ROOT)): digest(path) for path in files},
    }
    target = OUT / "taiwan_v56_ablation_frozen_specification.json"
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"TAIWAN_V56_ABLATION_FREEZE: PASS; sha256={digest(target)}")


if __name__ == "__main__":
    main()
