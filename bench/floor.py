"""Measure the fixed per-call cost of the LLM call path (guide section 5.1).

The two selectors are not timed at the same boundary: the GPT selector's timer
wraps a CLI subprocess, the JEV selector's wraps an in-process SDK call. Before
any latency number is reported, the fixed part of the CLI path has to be known,
because it is charged to GPT and never to JEV.

This issues the same call the benchmark issues, with a prompt small enough that
model time is near its floor, and reports the distribution. The minimum is the
floor estimate: the sample that got the least queueing and the warmest caches.

Two things this deliberately does NOT do:

  - it does not subtract the floor from anything. Guide 5.2: report the raw
    numbers and the floor together, never a single corrected percentage.
  - it does not default to a small sample. The minimum of few draws
    overestimates a floor, so 20 is the default and fewer is warned about.

Usage:

    python -m bench.floor                 # 20 samples
    python -m bench.floor --calls 30
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from . import core


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO_ROOT / "results" / "call_floor.json"

FLOOR_SYSTEM_PROMPT = "Reply with exactly one word and nothing else."
FLOOR_USER_PROMPT = "Say: ok"

MIN_TRUSTWORTHY_CALLS = 20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure the fixed per-call cost of the Codex CLI path."
    )
    parser.add_argument("--calls", type=int, default=MIN_TRUSTWORTHY_CALLS)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    return parser.parse_args()


def measure(calls: int, warmup: int) -> list[float]:
    samples: list[float] = []
    total = warmup + calls
    for index in range(1, total + 1):
        counted = index > warmup
        label = "measure" if counted else "warmup "
        started = time.perf_counter()
        try:
            _, meta = core.codex_json(FLOOR_SYSTEM_PROMPT, FLOOR_USER_PROMPT)
        except Exception as exc:  # noqa: BLE001 - report and keep sampling
            print(f"  [{label} {index}/{total}] FAILED: {exc}", file=sys.stderr)
            continue
        elapsed = float(
            meta.get("e2e_latency_ms") or (time.perf_counter() - started) * 1000
        )
        if counted:
            samples.append(elapsed)
        print(f"  [{label} {index}/{total}] {elapsed / 1000:.2f}s")
    return samples


def describe(samples: list[float]) -> dict[str, float]:
    ordered = sorted(samples)
    p90_index = min(len(ordered) - 1, int(round(0.9 * (len(ordered) - 1))))
    return {
        "n": len(ordered),
        "min_ms": ordered[0],
        "median_ms": statistics.median(ordered),
        "mean_ms": statistics.fmean(ordered),
        "p90_ms": ordered[p90_index],
        "max_ms": ordered[-1],
        "stdev_ms": statistics.stdev(ordered) if len(ordered) > 1 else 0.0,
    }


def main() -> int:
    args = parse_args()
    print(f"Measuring the call floor over {args.calls} calls (+{args.warmup} warmup)...")
    samples = measure(args.calls, args.warmup)
    if not samples:
        print("ERROR: every measurement call failed", file=sys.stderr)
        return 2

    stats = describe(samples)
    spread = stats["max_ms"] - stats["min_ms"]

    print()
    print("=" * 72)
    print("CALL FLOOR")
    print("=" * 72)
    print(f"samples   : {stats['n']}")
    print(f"minimum   : {stats['min_ms'] / 1000:.2f}s   <- floor estimate")
    print(f"median    : {stats['median_ms'] / 1000:.2f}s")
    print(f"p90       : {stats['p90_ms'] / 1000:.2f}s")
    print(f"maximum   : {stats['max_ms'] / 1000:.2f}s")
    print(f"stdev     : {stats['stdev_ms'] / 1000:.2f}s")
    print(f"spread    : {spread / 1000:.2f}s on an identical prompt")
    print()
    print("Report this floor alongside raw per-call latencies. Do not subtract it")
    print("(guide 5.2). The spread is itself a result: if it exceeds the gap")
    print("between two arms, that gap is noise (guide 8).")

    if stats["n"] < MIN_TRUSTWORTHY_CALLS:
        print()
        print(
            f"WARNING: {stats['n']} samples. The minimum of few draws overestimates "
            f"the floor; use --calls {MIN_TRUSTWORTHY_CALLS} or more before quoting "
            "this anywhere.",
            file=sys.stderr,
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "measured_at": datetime.now().isoformat(),
        "model": core.BACKBONE_MODEL,
        "reasoning_effort": core.BACKBONE_EFFORT,
        "path": "codex CLI subprocess",
        "prompt": {"system": FLOOR_SYSTEM_PROMPT, "user": FLOOR_USER_PROMPT},
        "stats_ms": stats,
        "spread_ms": spread,
        "trustworthy": stats["n"] >= MIN_TRUSTWORTHY_CALLS,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWritten: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
