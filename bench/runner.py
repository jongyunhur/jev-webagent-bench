"""One run of one task under one arm.

The step loop is rewritten from the previous repository's loop for one reason:
it must be able to consult more than one selector per step and record all of
them, while executing only the driver's choice (guide section 6).

What each step records, beyond the previous loop's fields:

    selections      every consulted selector's choice, latency and confidence
    agreed          whether the consulted selectors picked the same candidate
    driver          which selector's choice was executed

Everything the run spends is attributed per phase, and selector time is kept
per selector rather than summed, because summing across different measurement
boundaries is exactly what guide section 4.4 forbids.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from . import arms as arms_mod
from . import core


@dataclass
class RunConfig:
    arm: str
    task: dict[str, Any]
    max_steps: int
    out_dir: Path
    repeat: int = 1
    order_index: int = 1
    driver_override: str | None = None


def _selector_choice(decision: dict[str, Any]) -> str | None:
    choice = decision.get("choice")
    return str(choice) if choice is not None else None


def run_once(config: RunConfig) -> dict[str, Any]:
    """Execute one task under one arm and return its summary."""
    arm = arms_mod.get(config.arm)
    driver = config.driver_override or arm.driver
    if driver != arms_mod.PLANNER_DRIVES and driver not in arm.selectors:
        raise SystemExit(
            f"ERROR: arm {arm.name!r} does not consult selector {driver!r}; "
            f"it consults {arm.selectors or '(none)'}."
        )

    task = config.task
    goal = task["confirmed_task"]
    task_id = task.get("task_id", "unknown")
    start_url = task.get("website") or "https://www.google.com"

    core.aside_ready()
    core.verify_clean_benchmark_start()
    core.initialize_benchmark_tab_identity()
    core.benchmark_tab_ready()

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_id = f"{stamp}_{task_id[:8]}_{arm.name}_r{config.repeat}"
    logger = core.RunLogger(config.out_dir, run_id)

    print(f"Run ID     : {run_id}")
    print(f"Arm        : {arm.name}  ({arm.llm_calls_per_step})")
    print(f"Selectors  : {', '.join(arm.selectors) or '(planner decides)'}")
    print(f"Driver     : {driver}")
    print(f"Task       : {task_id}  [{task.get('level')}]  max_steps={config.max_steps}")
    print()

    total_start = time.perf_counter()
    state = core.browser_reset(start_url)
    core.register_root_tab(state)

    memory = ""
    history: list[dict[str, Any]] = []
    final_answer: str | None = None
    stop_reason = "max_steps"

    totals: dict[str, Any] = {
        "browser_ms": state.latency_ms,
        "generation_ms": 0.0,
        "generation_calls": 0,
        "generation_attempts": 0,
        "steps_executed": 0,
        "action_errors": 0,
        # Per selector, never summed together.
        "selector_ms": {name: 0.0 for name in arm.selectors},
        "selector_calls": {name: 0 for name in arm.selectors},
        "agreement_steps": 0,
        "comparable_steps": 0,
    }

    for step in range(1, config.max_steps + 1):
        print(f"\n===== STEP {step} =====  {state.url}")

        planned, gen_meta = arm.planner(goal, state, memory, history)
        totals["generation_ms"] += float(gen_meta.get("e2e_latency_ms") or 0.0)
        totals["generation_calls"] += 1
        totals["generation_attempts"] += int(gen_meta.get("attempts") or 1)

        memory = core.merge_memory(memory, planned.get("memory_update", ""))
        candidates = planned["candidates"]
        for candidate in candidates:
            print(f"  {candidate['id']}: {candidate['description'][:90]}")

        # --- selection -----------------------------------------------------
        selections: dict[str, dict[str, Any]] = {}
        if driver == arms_mod.PLANNER_DRIVES:
            executable = [c for c in candidates if c["id"] != "abstain"]
            selected_id = executable[0]["id"] if len(executable) == 1 else "abstain"
            decision = {
                "selector": f"{arm.name}/no-post-selector",
                "choice": selected_id,
                "e2e_latency_ms": 0.0,
            }
        else:
            # Consult every selector this arm names, on the same candidate set.
            for name in arm.selectors:
                result = arms_mod.SELECTORS[name](goal, state, memory, candidates)
                selections[name] = result
                totals["selector_ms"][name] += float(result.get("e2e_latency_ms") or 0.0)
                totals["selector_calls"][name] += 1
                confidence = result.get("confidence")
                suffix = f"  conf={confidence}" if confidence is not None else ""
                print(f"  [{name}] -> {result.get('choice')}{suffix}")
            decision = selections[driver]
            selected_id = decision["choice"]

        # Agreement is only meaningful when more than one selector ran and both
        # returned a real choice (guide section 7.1).
        choices = [
            _selector_choice(result)
            for result in selections.values()
            if _selector_choice(result) not in (None, "abstain")
        ]
        agreed: bool | None = None
        if len(selections) > 1 and len(choices) == len(selections):
            agreed = len(set(choices)) == 1
            totals["comparable_steps"] += 1
            totals["agreement_steps"] += int(agreed)

        selected = next(c for c in candidates if c["id"] == selected_id)
        action = selected["action"]

        # --- execution -----------------------------------------------------
        before_url, before_title = state.url, state.title
        active_tab_before = core.active_tab_index()
        registry_before = core.browser_registry_snapshot()
        browser_ms = 0.0
        url_after, title_after = before_url, before_title
        action_error: str | None = None
        new_tabs: list[dict[str, Any]] = []

        if action["type"] == "done":
            final_answer = action["answer"].strip()
            stop_reason = "done"
        elif action["type"] == "abstain":
            stop_reason = "selector_abstained"
        else:
            try:
                next_state = core.execute_action(action)
                browser_ms = next_state.latency_ms
                totals["browser_ms"] += browser_ms
                totals["steps_executed"] += 1
                state = next_state
                url_after, title_after = state.url, state.title
                new_tabs = state.new_tabs
            except Exception as exc:
                # A live page can change under the agent. That is a trajectory
                # event, not a reason to lose the run.
                action_error = str(exc)
                totals["action_errors"] += 1
                print(f"  action failed: {action_error}")
                try:
                    recovery = core.observe_current(recover_missing_current=True)
                    browser_ms = recovery.latency_ms
                    totals["browser_ms"] += browser_ms
                    state = recovery
                    core.adopt_active_tab(state)
                    core.track_browser_state(state)
                    url_after, title_after = state.url, state.title
                except Exception as recovery_exc:
                    action_error += f" | recovery failed: {recovery_exc}"
                    stop_reason = "browser_recovery_failed"

        record = {
            "timestamp": core.now_iso(),
            "run_id": run_id,
            "task_id": task_id,
            "level": task.get("level"),
            "arm": arm.name,
            "driver": driver,
            "repeat": config.repeat,
            "step": step,
            "goal": goal,
            "url_before": before_url,
            "title_before": before_title,
            "memory_update": planned.get("memory_update", ""),
            "memory_after": memory,
            "candidates": candidates,
            "candidate_generation": gen_meta,
            "failed_action_candidates_filtered": planned.get(
                "failed_action_candidates_filtered", 0
            ),
            "completed_navigation_candidates_filtered": planned.get(
                "completed_navigation_candidates_filtered", 0
            ),
            "context_inferred_candidates": planned.get("context_inferred_candidates", 0),
            # The paired observation the main comparison is built from.
            "selections": selections,
            "agreed": agreed,
            "selected_id": selected_id,
            "selected_action": action,
            "browser_action_latency_ms": browser_ms,
            "new_tabs_opened": new_tabs,
            "active_tab_before": active_tab_before,
            "active_tab_after": core.active_tab_index(),
            "browser_registry_before": registry_before,
            "browser_registry_after": core.browser_registry_snapshot(),
            "action_error": action_error,
            "url_after": url_after,
            "title_after": title_after,
        }
        logger.log_step(record)
        history.append(record)

        if stop_reason != "max_steps":
            break

    total_ms = (time.perf_counter() - total_start) * 1000
    steps_logged = len(history)

    summary = {
        "run_id": run_id,
        "timestamp": core.now_iso(),
        "task_id": task_id,
        "level": task.get("level"),
        "arm": arm.name,
        "arm_role": arm.role,
        "driver": driver,
        "repeat": config.repeat,
        "order_index": config.order_index,
        "max_steps": config.max_steps,
        "llm_calls_per_step": arm.llm_calls_per_step,
        # Carried into every summary so a later reader cannot compare latencies
        # across boundaries without being told they differ (guide 4.4).
        "selector_boundaries": {
            name: arms_mod.SELECTOR_BOUNDARY[name] for name in arm.selectors
        },
        "comparable_total_runtime": arm.comparable_total_runtime,
        "stop_reason": stop_reason,
        "finished_done": stop_reason == "done",
        "steps_logged": steps_logged,
        "steps_executed": totals["steps_executed"],
        "total_wall_ms": round(total_ms, 2),
        "generation_ms": round(totals["generation_ms"], 2),
        "generation_calls": totals["generation_calls"],
        "generation_attempts": totals["generation_attempts"],
        "browser_ms": round(totals["browser_ms"], 2),
        "selector_ms": {k: round(v, 2) for k, v in totals["selector_ms"].items()},
        "selector_calls": totals["selector_calls"],
        "agreement_steps": totals["agreement_steps"],
        "comparable_steps": totals["comparable_steps"],
        "action_error_count": totals["action_errors"],
        "final_url": state.url,
        "final_answer": final_answer,
        "has_final_answer": bool(final_answer),
    }
    logger.write_summary(summary)

    print()
    print(f"stop_reason  : {stop_reason}")
    print(f"steps        : {steps_logged}")
    print(f"total wall   : {total_ms / 1000:.2f}s")
    print(f"generation   : {totals['generation_ms'] / 1000:.2f}s")
    for name in arm.selectors:
        calls = totals["selector_calls"][name]
        seconds = totals["selector_ms"][name] / 1000
        per_call = seconds / calls if calls else 0.0
        print(f"selector[{name}]: {seconds:.2f}s over {calls} calls ({per_call:.2f}s/call)")
    if totals["comparable_steps"]:
        rate = totals["agreement_steps"] / totals["comparable_steps"] * 100
        print(
            f"agreement    : {totals['agreement_steps']}/{totals['comparable_steps']}"
            f" ({rate:.1f}%)"
        )
    print(f"logs         : {logger.run_dir}")

    return summary
