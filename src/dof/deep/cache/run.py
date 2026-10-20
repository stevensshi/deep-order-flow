"""The phases of a cache build, in this order: probes and plan of the training days, the training groups, the
validation days, the test days last (refused until the protocol document exists, when the configuration asks for
one). Nothing but training-day folders is opened in the first phases.

Tasks run in spawned worker processes: a task starts when a worker is free and the memory estimates of the
running tasks plus its own stay under the budget. ``workers=0`` runs every task in this process, in list order.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

from dof.deep.cache import layout as L
from dof.deep.cache.build import day_log, eval_task, group_task, make_plan, probe_task
from dof.deep.cache.summary import summarise
from dof.deep.config import CachePaths, RunConfig, Split, protocol_sha256
from dof.deep.errors import CacheError, DiskFloor
from dof.deep.runtime.disk import check_free, free_gb
from dof.deep.runtime.io import now, write_json_atomic
from dof.deep.runtime.provenance import provenance
from dof.deep.universe import reused

# Memory estimates of a task, in bytes per row, from two measured stock-days (6.4 million state rows: probe 2.0 GB,
# full day 3.9 GB). They only decide how many tasks run at once.
PROBE_BYTES_PER_RAW_ROW = 350
DAY_BYTES_PER_RAW_ROW = 650
HELD_BYTES_PER_CLEAN_ROW = 280      # a group's days held for the scaler fit: order flow and labels in float64, indices
EVAL_BYTES_PER_RAW_ROW = 800
TASK_BASE_BYTES = 300 * 2**20


def run_tasks(tasks: list[tuple[int, str, tuple]], fn, workers: int, budget: int, every: int = 1, log=print) -> list:
    """``fn(*args)`` for every (memory estimate, label, args). With workers, the largest first; a task above the
    budget runs alone. The first failure stops the phase after the running tasks have ended (nothing is skipped
    silently)."""
    results, failed, done_n, t0 = [], [], 0, time.perf_counter()
    if workers == 0:
        for _est, label, args in tasks:
            try:
                results.append(fn(*args))
            except Exception as exc:
                log(f"  FAILED {label}: {exc!r}\n{''.join(traceback.format_exception(exc))}")
                raise CacheError(f"1 task failed: {[label]}") from exc
            done_n += 1
            if done_n % every == 0 or done_n == len(tasks):
                log(f"  {done_n}/{len(tasks)} done ({label}), {time.perf_counter() - t0:.0f} s")
        return results
    pending = sorted(tasks, key=lambda t: -t[0])
    # spawn: a worker imports its modules afresh
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"), max_tasks_per_child=40) as pool:
        running: dict = {}
        while (pending and not failed) or running:
            while pending and not failed and len(running) < workers:
                used = sum(est for est, _ in running.values())
                i = next((i for i, t in enumerate(pending) if not running or used + t[0] <= budget), None)
                if i is None:
                    break
                est, label, args = pending.pop(i)
                running[pool.submit(fn, *args)] = (est, label)
            finished, _ = wait(running, return_when=FIRST_COMPLETED)
            for fut in finished:
                _est, label = running.pop(fut)
                done_n += 1
                try:
                    results.append(fut.result())
                except BaseException as exc:  # noqa: BLE001 — reported below, then the phase stops
                    failed.append(label)
                    log(f"  FAILED {label}: {exc!r}\n{''.join(traceback.format_exception(exc))}")
                    continue
                if done_n % every == 0 or done_n == len(tasks):
                    log(f"  {done_n}/{len(tasks)} done ({label}), {time.perf_counter() - t0:.0f} s")
    if failed:
        raise CacheError(f"{len(failed)} task(s) failed: {failed[:10]}")
    return results


def phase_train(cfg: RunConfig, split: Split, cache: CachePaths, workers: int, budget: int, log=print) -> None:
    log(f"=== phase probe ({split.name}) {now()}")
    tasks = []
    for d in split.train_days:
        held, _bad = day_log(cfg, split, d)
        tasks += [(TASK_BASE_BYTES + PROBE_BYTES_PER_RAW_ROW * n, f"probe {s} {d}", (cfg, split, cache, s, d))
                  for s, n in held.items() if not reused(split, s, d) and not L.probe_path(cache, s, d).exists()]
    if tasks:
        run_tasks(tasks, probe_task, workers, budget, every=200, log=log)
    log(f"=== phase plan ({split.name}) {now()}")
    plan = make_plan(cfg, split, cache)
    path = L.plan_path(cache)
    if path.exists():
        if {k: v for k, v in json.loads(path.read_text(encoding="utf-8")).items() if k != "written"} != json.loads(json.dumps(plan)):
            raise CacheError(f"{path} was made from other probes or another rule; move the cache away before building")
    else:
        write_json_atomic(path, {**plan, "written": {"time": now(), "provenance": provenance()}}, indent=1)
    groups = [(s, g) for s, x in plan["stocks"].items() for g in x["groups"]]
    todo = [(s, g) for s, g in groups if g["status"] == "ok" and not L.group_path(cache, s, g["id"]).exists()]
    # A group record that exists must be the plan's group: the same days, the same Δt and the same mean count, to
    # the last digit. Otherwise it was built under another rule and the build stops.
    kept, stale = 0, []
    for s, g in groups:
        p = L.group_path(cache, s, g["id"])
        if g["status"] == "ok" and p.exists():
            doc = json.loads(p.read_text(encoding="utf-8"))
            same = (doc["days"], doc["delta_t_ms"], doc["mid_changes_per_day"]) == (g["days"], g["delta_t_ms"], g["mid_changes_per_day"])
            kept += same
            if not same:
                stale.append(f"{s} g{g['id']}")
    if stale:
        raise CacheError(f"{len(stale)} group records are not the plan's groups ({stale[:8]}): move them away before building")
    log(f"group records kept: {kept}; their days, Δt and mean count equal the plan's to the last digit")
    n_clean = sum(g["n_clean"] for _, g in groups if g["status"] == "ok")
    est_gb = n_clean * (L.REC.itemsize + 4) / 1e9
    left_gb = sum(g["n_clean"] for _, g in todo) * (L.REC.itemsize + 4) / 1e9
    line = (f"plan: {len(plan['stocks'])} stocks, {len(groups)} scaling groups ({sum(g['status'] != 'ok' for _, g in groups)} "
            f"without a horizon unit), {n_clean:,} cleaned rows on the training days; training cache {L.REC.itemsize} + 4 bytes "
            f"per row = {est_gb:.1f} GB, of which {left_gb:.1f} GB still to write")
    guard, floor = cfg.machine.disk_guard_path, cfg.machine.build_floor_gb
    if guard is None:
        log(line)
    else:
        avail = free_gb(guard)
        log(f"{line}; {guard} {avail} G available, {avail - left_gb:.0f} G after the build (floor {floor} G)")
        if avail - left_gb < floor:
            raise DiskFloor(f"the training cache would leave {guard} at {avail - left_gb:.0f} G, under {floor} G: stopped")
    log(f"=== phase train ({split.name}): {len(todo)} of {len(groups)} groups to build {now()}")
    tasks = [(TASK_BASE_BYTES + HELD_BYTES_PER_CLEAN_ROW * g["n_clean"] + DAY_BYTES_PER_RAW_ROW * g["max_raw"],
              f"train {s} g{g['id']} {g['days'][0]}..{g['days'][-1]}", (cfg, split, cache, s, g["id"])) for s, g in todo]
    if tasks:
        run_tasks(tasks, group_task, workers, budget, every=10, log=log)


def phase_eval(cfg: RunConfig, split: Split, cache: CachePaths, part: str, workers: int, budget: int, log=print) -> None:
    if part == "test" and cfg.protocol.test_days_need_document:
        log(f"=== phase test ({split.name}): protocol document sha256 {protocol_sha256(cfg)} {now()}")
    else:
        log(f"=== phase {part} ({split.name}) {now()}")
    check_free(cfg.machine.disk_guard_path, cfg.machine.disk_floor_gb, f"before the {part} days", log)
    tasks = []
    for d in split.days(part):
        held, _bad = day_log(cfg, split, d)
        tasks += [(TASK_BASE_BYTES + EVAL_BYTES_PER_RAW_ROW * n, f"{part} {s} {d}", (cfg, split, cache, part, s, d))
                  for s, n in held.items() if not L.eval_meta_path(cache, part, s, d).exists()]
    if tasks:
        run_tasks(tasks, eval_task, workers, budget, every=20, log=log)


def part_built(cfg: RunConfig, split: Split, cache: CachePaths, part: str) -> bool:
    """Every held stock-day of a validation or test part has its meta file. The folder is looked at first: the
    build logs of a part that was never built are not opened here."""
    return (cache.eval / part).is_dir() and all(L.eval_meta_path(cache, part, s, d).exists()
                                                for d in split.days(part) for s in day_log(cfg, split, d)[0])


def build(cfg: RunConfig, split: Split, cache: CachePaths, *, parts=("train", "valid", "test"), workers: int = 0,
          mem_gb: float = 12.0, log=print) -> dict:
    """Build what is missing of the cache, in the fixed order train, valid, test (whatever the order of ``parts``),
    then write and return the summary of the parts that are built. A finished unit is never rebuilt."""
    m = cfg.machine
    budget = int(mem_gb * 2**30)
    log(f"=== cache build start {now()}: split {split.name}, parts {list(parts)}, {workers} workers, memory budget "
        f"{mem_gb:g} GiB, book root {m.book_root}")
    check_free(m.disk_guard_path, max(m.disk_floor_gb, m.build_floor_gb), "before the build", log)
    for part in ("train", "valid", "test"):
        if part not in parts:
            continue
        if part == "train":
            phase_train(cfg, split, cache, workers, budget, log)
        else:
            phase_eval(cfg, split, cache, part, workers, budget, log)
    doc = summarise(cfg, split, cache, tuple(p for p in ("train", "valid", "test") if p == "train" or part_built(cfg, split, cache, p)))
    write_json_atomic(L.summary_path(cache), doc, indent=1)
    check_free(m.disk_guard_path, m.disk_floor_gb, "after the build", log)
    return doc
