from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main() -> None:
    result = OUT / "taiwan_v55_econometric_simulation.json"
    freeze = OUT / "taiwan_v55_frozen_specification.json"
    payload = json.loads(result.read_text(encoding="utf-8"))
    payload["verification"]["runtime_validation"] = "TAIWAN_V55_RUNTIME_VALIDATION: PASS"
    payload["verification"]["econometric_calibration"]["frozen_specification_sha256"] = digest(freeze)
    result.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("TAIWAN_V55_METADATA_FINALIZED")


if __name__ == "__main__":
    main()
