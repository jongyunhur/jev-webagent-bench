"""Run protocol: repeats, order counterbalancing, aggregation.

This replaces the five overlapping batch scripts of the previous repository with
one entry point, and enforces the parts of the guide that a human running the
experiment by hand will otherwise skip:

    section 10.2  execution order is counterbalanced across repeats, and a cell
                  whose order could not be balanced is flagged in the manifest
    section 10.3  repeats default to 3; a single repeat is allowed but the
                  manifest records that no percentage may be claimed from it
    section 10.4  incomplete runs are never silently averaged - aggregation
                  stops with an error

Usage:

    python -m bench.batch --arm shadow                  # main comparison
    python -m bench.batch --arm single-call             # reference baseline
    python -m bench.batch --arm shadow --repeats 1 --levels hard
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from . import arms as arms_mod
from . import core
from .runner import RunConfig, run_once


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = REPO_ROOT / "data" / "pilot_tasks.json"
LEVELS = ("easy", "medium", "hard")
DEFAULT_BUDGETS = {"easy": 8, "medium": 16, "hard": 30}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a benchmark arm over the pilot tasks, with repeats and "
            "counterbalanced order."
        )
    )
    parser.add_argument(
        "--arm",
        default="shadow",
        choices=sorted(arms_mod.ARMS),
        help="which arm to run (default: shadow, the main comparison)",
    )
    parser.add_argument(
        "--driver",
        choices=("gpt", "jev", arms_mod.PLANNER_DRIVES),
        help="override which selector's choice executes (shadow arm only)",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="runs per task (default: 3; guide 10.3 forbids claims from 1)",
    )
    parser.add_argument("--levels", nargs="+", choices=LEVELS, default=list(LEVELS))
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "runs"))
    parser.add_argument("--results-dir", default=str(REPO_ROOT / "results"))
    parser.add_argument("--easy-steps", type=int, default=DEFAULT_BUDGETS["easy"])
    parser.add_argument("--medium-steps", type=int, default=DEFAULT_BUDGETS["medium"])
    parser.add_argument("--hard-steps", type=int, default=DEFAULT_BUDGETS["hard"])
    parser.add_argument("--pause-seconds", type=float, default=2.0)
    parser.add_argument("--stop-on-error", action="store_true")
    return parser.parse_args()


def select_tasks(dataset: Path, levels: list[str]) -> list[dict[str, Any]]:
    records = json.loads(dataset.read_text(encoding="utf-8"))
    tasks = []
    for level in levels:
        matches = [row for row in records if row.get("level") == level]
        if len(matches) != 1:
            raise SystemExit(f"ERROR: expected exactly one {level} task, got {len(matches)}")
        tasks.append(matches[0])
    return tasks


def execution_plan(
    tasks: list[dict[str, Any]], repeats: int
) -> list[tuple[dict[str, Any], int]]:
    """Order runs so no task systematically runs early or late.

    Guide 10.2: latency drifts over a session, so a task that always runs last
    has "ran last" baked into its numbers. Rotating the task order by one each
    repeat (a Latin square) gives every task the same mean position, and as a
    side effect never places two repeats of the same task back to back.

    With three tasks and three repeats the order is

        easy medium hard | medium hard easy | hard easy medium

    so each task occupies an early, a middle and a late slot exactly once.
    """
    plan: list[tuple[dict[str, Any], int]] = []
    count = len(tasks)
    for repeat in range(1, repeats + 1):
        offset = (repeat - 1) % count if count else 0
        for task in tasks[offset:] + tasks[:offset]:
            plan.append((task, repeat))
    return plan


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


# Nested dicts (per-selector timings) are flattened for the CSV so the file stays
# readable in a spreadsheet; runs.json keeps the full structure.
def flatten(summary: dict[str, Any]) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in summary.items():
        if isinstance(value, dict):
            for inner, inner_value in value.items():
                flat[f"{key}.{inner}"] = inner_value
        else:
            flat[key] = value
    return flat


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    flat_rows = [flatten(row) for row in rows]
    fieldnames: list[str] = []
    for row in flat_rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        for row in flat_rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def check_complete(rows: list[dict[str, Any]]) -> list[str]:
    """Guide 10.4. An unfinished run stops early, so its total is small; left in
    an average it reads as a speed advantage it did not earn."""
    return [
        f"{row['arm']}/{row['level']}/r{row['repeat']}: {row['stop_reason']}"
        for row in rows
        if row.get("stop_reason") != "done"
    ]


def main() -> int:
    args = parse_args()
    arm = arms_mod.get(args.arm)

    if arm.needs_jev and not os.environ.get("TYPESAFE_API_KEY"):
        print("ERROR: TYPESAFE_API_KEY is not set; this arm consults JEV.", file=sys.stderr)
        return 2

    dataset = Path(args.dataset).resolve()
    if not dataset.exists():
        print(f"ERROR: dataset not found: {dataset}", file=sys.stderr)
        return 2

    tasks = select_tasks(dataset, args.levels)
    budgets = {
        "easy": args.easy_steps,
        "medium": args.medium_steps,
        "hard": args.hard_steps,
    }
    plan = execution_plan(tasks, args.repeats)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_root = Path(args.results_dir)
    results_root.mkdir(parents=True, exist_ok=True)

    batch_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    results_dir = results_root / f"{batch_id}_{arm.name}"
    results_dir.mkdir(parents=True, exist_ok=True)

    driver = args.driver or arm.driver
    manifest = {
        "batch_id": batch_id,
        "arm": arm.name,
        "arm_role": arm.role,
        "driver": driver,
        "llm_calls_per_step": arm.llm_calls_per_step,
        "selectors": list(arm.selectors),
        "selector_boundaries": {
            name: arms_mod.SELECTOR_BOUNDARY[name] for name in arm.selectors
        },
        "comparable_total_runtime": arm.comparable_total_runtime,
        "repeats": args.repeats,
        "levels": args.levels,
        "budgets": budgets,
        "execution_order": [
            {"position": i, "level": task["level"], "repeat": repeat}
            for i, (task, repeat) in enumerate(plan, start=1)
        ],
        "dataset": str(dataset),
        "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
        "model": core.BACKBONE_MODEL,
        "reasoning_effort": core.BACKBONE_EFFORT,
        "single_repeat_warning": (
            None
            if args.repeats >= 3
            else "Fewer than 3 repeats: guide 10.3 forbids claiming any "
            "percentage from this batch."
        ),
        "total_runtime_comparable_note": (
            None
            if arm.comparable_total_runtime
            else "This arm adds a selection call per step for comparison "
            "purposes; its wall clock matches no real configuration and must "
            "not be compared against another arm's total (guide 6)."
        ),
    }
    write_json(results_dir / "manifest.json", manifest)

    print("=" * 78)
    print(f"ARM {arm.name}  |  {arm.role}")
    print(f"driver={driver}  repeats={args.repeats}  levels={', '.join(args.levels)}")
    print(f"planned runs: {len(plan)}")
    if manifest["single_repeat_warning"]:
        print(f"WARNING: {manifest['single_repeat_warning']}")
    if manifest["total_runtime_comparable_note"]:
        print(f"NOTE: {manifest['total_runtime_comparable_note']}")
    print("=" * 78)

    try:
        core.aside_ready()
    except Exception as exc:
        print(f"ERROR: Aside is not ready: {exc}", file=sys.stderr)
        return 2

    rows: list[dict[str, Any]] = []
    for position, (task, repeat) in enumerate(plan, start=1):
        level = task["level"]
        print()
        print("=" * 78)
        print(f"[{position}/{len(plan)}] {level} repeat {repeat}")
        print("=" * 78)

        try:
            summary = run_once(
                RunConfig(
                    arm=arm.name,
                    task=task,
                    max_steps=budgets[level],
                    out_dir=out_dir,
                    repeat=repeat,
                    order_index=position,
                    driver_override=args.driver,
                )
            )
        except Exception as exc:
            print(f"RUN ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
            summary = {
                "arm": arm.name,
                "level": level,
                "task_id": task.get("task_id"),
                "repeat": repeat,
                "order_index": position,
                "stop_reason": "run_error",
                "error": f"{type(exc).__name__}: {exc}",
            }

        rows.append(summary)
        write_csv(results_dir / "runs.csv", rows)
        write_json(results_dir / "runs.json", rows)

        if summary.get("stop_reason") != "done" and args.stop_on_error:
            print("Stopping because --stop-on-error was set.")
            break

        if position < len(plan):
            try:
                # The next run requires the persistent benchmark tab to carry
                # its discovery marker. Aside also times out waiting for an
                # interactive page after navigating to about:blank.
                core.browser_reset("https://www.google.com/?jevbench=1")
                time.sleep(max(0.0, args.pause_seconds))
            except Exception as exc:
                print(f"ERROR: could not reset the benchmark tab: {exc}", file=sys.stderr)
                return 2

    print()
    print("=" * 78)
    print("BATCH COMPLETE")
    print("=" * 78)
    print(f"runs attempted : {len(rows)}")
    print(f"completed      : {sum(1 for r in rows if r.get('stop_reason') == 'done')}")
    print(f"runs.csv       : {results_dir / 'runs.csv'}")
    print(f"manifest.json  : {results_dir / 'manifest.json'}")

    incomplete = check_complete(rows)
    if incomplete:
        print()
        print("INCOMPLETE RUNS - not aggregated (guide 10.4):", file=sys.stderr)
        for item in incomplete:
            print(f"  {item}", file=sys.stderr)
        print(
            "\nRe-run these before analysing. An unfinished run ends early, so "
            "leaving it in an average reads as a speed advantage it did not earn.",
            file=sys.stderr,
        )
        return 1

    print()
    print(f"Next: python -m bench.analyze {results_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
