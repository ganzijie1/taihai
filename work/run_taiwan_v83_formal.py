from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
from pathlib import Path

import simulate_taiwan_v83_checkpointed as runs


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
LOG_ROOT = OUT / "taiwan_v83_runs" / "logs"


def run_command(args: list[str], log_path: Path) -> tuple[list[str], int]:
    env = os.environ.copy()
    env["TAIWAN_ALLOW_COMPLETE_RUN"] = "1"
    env.setdefault("OMP_NUM_THREADS", "4")
    env.setdefault("MKL_NUM_THREADS", "4")
    env.setdefault("OPENBLAS_NUM_THREADS", "4")
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(
            args, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
            check=False,
        )
    return args, result.returncode


def checkpoint_jobs() -> list[tuple[list[str], Path]]:
    jobs = []
    for variant in runs.VARIANTS:
        for i, _scenario in enumerate(runs.SCENARIOS):
            for j, _case in enumerate(runs.CASES):
                command = [
                    sys.executable, "work/simulate_taiwan_v83_checkpointed.py",
                    variant, "--cell", f"{i}:{j}",
                ]
                jobs.append((command, LOG_ROOT / f"{variant}_{i:02d}_{j:02d}.log"))
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    plan = json.loads(runs.PLAN.read_text(encoding="utf-8"))
    if plan.get("status") != "FROZEN" or plan.get("total_cells") != 270:
        raise RuntimeError("V8.3 formal plan is not frozen at 270 cells")
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    failures = []
    jobs = checkpoint_jobs()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        future_map = {
            pool.submit(run_command, command, log): (command, log)
            for command, log in jobs
        }
        completed = 0
        for future in concurrent.futures.as_completed(future_map):
            command, log = future_map[future]
            _args, code = future.result()
            completed += 1
            print(
                f"TAIWAN_V83_FORMAL_PROGRESS: {completed}/{len(jobs)} "
                f"exit={code} job={' '.join(command[-3:])}", flush=True,
            )
            if code != 0:
                failures.append({"command": command, "log": str(log), "exit": code})
    if failures:
        failure_path = OUT / "taiwan_v83_formal_failures.json"
        failure_path.write_text(
            json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        raise RuntimeError(f"{len(failures)} formal checkpoint jobs failed")
    for variant in runs.VARIANTS:
        command = [sys.executable, "work/simulate_taiwan_v83_checkpointed.py", variant]
        _, code = run_command(command, LOG_ROOT / f"merge_{variant}.log")
        if code != 0:
            raise RuntimeError(f"merge failed for {variant}")
    command = [sys.executable, "work/audit_taiwan_v83_results.py"]
    _, code = run_command(command, LOG_ROOT / "result_audit.log")
    if code != 0:
        raise RuntimeError("V8.3 result audit failed")
    print("TAIWAN_V83_FORMAL_RUN: PASS", flush=True)


if __name__ == "__main__":
    main()
