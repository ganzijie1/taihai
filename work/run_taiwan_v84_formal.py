from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import json
import os
from pathlib import Path

import simulate_taiwan_v84_checkpointed as runs


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
LOG_ROOT = OUT / "taiwan_v84_runs" / "logs"


def _initialise_worker() -> None:
    os.environ["TAIWAN_ALLOW_COMPLETE_RUN"] = "1"
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("MKL_NUM_THREADS", "4")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")


def _run_cell_variant_batch(job: tuple[int, int]) -> dict:
    """Run all variants in one persistent worker for a common scenario cell."""
    i, j = job
    scenario, case = runs.SCENARIOS[i], runs.CASES[j]
    log_path = LOG_ROOT / f"cell_{i:02d}_{j:02d}.log"
    completed: list[str] = []
    with log_path.open("a", encoding="utf-8") as log, \
            contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        plan = runs.load_plan(runs.VARIANTS[0])
        paths, months = int(plan["paths"]), int(plan["months"])
        payloads = runs.run_variant_batch_or_load(
            i, j, scenario, case, paths, months, runs.sha256(runs.PLAN), plan,
        )
        completed.extend(payloads)
    return {
        "cell": [i, j], "scenario": scenario, "case": case,
        "variants": completed, "worker_pid": os.getpid(), "log": str(log_path),
    }


def _checkpoint_jobs() -> list[tuple[int, int]]:
    return [
        (i, j)
        for i, _scenario in enumerate(runs.SCENARIOS)
        for j, _case in enumerate(runs.CASES)
    ]


def _load_cell_payloads(variant: str) -> list[dict]:
    return [
        json.loads(runs.checkpoint_path(variant, i, j).read_text(encoding="utf-8"))
        for i, _scenario in enumerate(runs.SCENARIOS)
        for j, _case in enumerate(runs.CASES)
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workers", type=int, default=1,
        help="persistent CUDA workers; keep at one per physical GPU",
    )
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    plan = json.loads(runs.PLAN.read_text(encoding="utf-8"))
    if plan.get("status") != "FROZEN" or plan.get("total_cells") != 420:
        raise RuntimeError("V8.4 formal plan is not frozen at 420 cells")
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    jobs = _checkpoint_jobs()
    failures: list[dict] = []
    worker_pids: set[int] = set()
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=args.workers, initializer=_initialise_worker,
    ) as pool:
        future_map = {pool.submit(_run_cell_variant_batch, job): job for job in jobs}
        completed = 0
        for future in concurrent.futures.as_completed(future_map):
            job = future_map[future]
            try:
                result = future.result()
                worker_pids.add(int(result["worker_pid"]))
                code = 0
            except Exception as exc:  # the complete failure is retained in the cell log
                result = {"cell": list(job), "error": repr(exc)}
                failures.append(result)
                code = 1
            completed += 1
            print(
                f"TAIWAN_V84_FORMAL_PROGRESS: {completed}/{len(jobs)} "
                f"exit={code} cell={job[0]}:{job[1]}", flush=True,
            )
    if failures:
        path = OUT / "taiwan_v84_formal_failures.json"
        path.write_text(json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8")
        raise RuntimeError(f"{len(failures)} formal checkpoint cell batches failed")
    if len(worker_pids) > args.workers:
        raise RuntimeError("persistent-worker contract violated")
    for variant in runs.VARIANTS:
        runs.merge_and_write(variant, _load_cell_payloads(variant), plan)
    import audit_taiwan_v84_results as audit
    audit.main()
    print(
        f"TAIWAN_V84_FORMAL_RUN: PASS persistent_workers={len(worker_pids)} ",
        flush=True,
    )


if __name__ == "__main__":
    main()
