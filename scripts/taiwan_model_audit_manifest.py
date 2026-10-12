from __future__ import annotations

import json
import sys
from pathlib import Path


ROLES = {"estimator", "dynamic_state", "constraint_solver", "terminal_model", "superseded", "theory_only"}
STATUSES = {"active", "superseded", "theory_only"}


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: audit_manifest.py <manifest.json>")
        return 2
    path = Path(sys.argv[1])
    payload = json.loads(path.read_text(encoding="utf-8"))
    modules = payload.get("modules", payload if isinstance(payload, list) else [])
    errors: list[str] = []
    ids: set[str] = set()
    for index, module in enumerate(modules):
        label = module.get("module_id", f"item-{index}")
        if label in ids:
            errors.append(f"{label}: duplicate module_id")
        ids.add(label)
        if module.get("role") not in ROLES:
            errors.append(f"{label}: invalid role")
        if module.get("status") not in STATUSES:
            errors.append(f"{label}: invalid status")
        for key in ("document_sections", "implementation", "data", "invariants", "tests"):
            if not module.get(key):
                errors.append(f"{label}: missing {key}")
        if module.get("status") == "active":
            if not module.get("outputs_artifact"):
                errors.append(f"{label}: missing outputs_artifact")
            if module.get("role") in {"dynamic_state", "constraint_solver", "terminal_model"}:
                if not module.get("pathwise"):
                    errors.append(f"{label}: active simulation module is not pathwise")
                if not module.get("clock"):
                    errors.append(f"{label}: missing clock")
                if not module.get("inputs") or not module.get("outputs"):
                    errors.append(f"{label}: missing state inputs or outputs")
            if module.get("role") == "estimator" and not module.get("feedback_consumers"):
                errors.append(f"{label}: estimator has no active consumer")
        for source in module.get("data", []):
            for key in ("source", "publisher", "observation_date", "unit", "identification"):
                if not source.get(key):
                    errors.append(f"{label}: data entry missing {key}")
            if source.get("identification") not in {"observed", "derived", "estimated", "scenario_prior"}:
                errors.append(f"{label}: invalid identification tag")
    if not modules:
        errors.append("manifest has no modules")
    if errors:
        print("FAIL")
        for error in errors:
            print(f"- {error}")
        return 1
    print(f"PASS: {len(modules)} modules audited")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
