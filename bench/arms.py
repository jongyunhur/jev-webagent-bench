"""Arm definitions.

An arm is a named configuration of three things:

    planner   what produces the candidate set for a step
    selectors which selectors are consulted on that candidate set
    driver    whose choice is actually executed

Naming follows guide section 2. "GPT-5.5 only" is deliberately not an arm name:
it reads as both `single-call` and `gen+gpt-select`, and that ambiguity is where
the wrong comparison starts.

The main comparison (guide section 3.1) is `gen+gpt-select` vs `gen+jev-select`.
Those two are never run separately - the `shadow` arm runs both selectors
against one shared generation call, which is what removes the per-step call-count
confound described in guide sections 4.3 and 5.3.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Protocol

from . import core
from . import gpt_confidence


# ---------------------------------------------------------------------------
# Selectors
# ---------------------------------------------------------------------------

class Selector(Protocol):
    def __call__(
        self,
        goal: str,
        state: Any,
        memory: str,
        candidates: list[dict[str, Any]],
    ) -> dict[str, Any]: ...


SELECTORS: dict[str, Selector] = {
    # The GPT selector reports a probability per candidate so its confidence can
    # be set beside JEV's. That changes its prompt (it must write the
    # probabilities, not just name a candidate), so results from before this
    # change are not comparable with results after it. core.select_with_gpt55 is
    # the earlier choice-only selector and is kept for reference.
    "gpt": gpt_confidence.select_with_gpt55_probabilities,
    "jev": core.select_with_jev,
}

# Where each selector's timer sits. Recorded in every run summary because the
# guide (section 4.4) forbids comparing latencies across different boundaries
# without saying so.
SELECTOR_BOUNDARY = {
    "gpt": "subprocess(codex) - process spawn + CLI startup + auth + network + inference",
    "jev": "in-process SDK call - network + inference",
}


# ---------------------------------------------------------------------------
# Single-call planner
# ---------------------------------------------------------------------------

# The role paragraph is the ONLY difference between this arm's system prompt and
# the shared planner prompt. Everything else - safety rules, allowed action
# types, target grounding - stays byte-identical, so no arm is advantaged by
# prompt drift (guide section 4.2).
_PLANNER_ROLE = """Generate a small bounded set of useful NEXT actions from the current browser
observation. Do not decide which candidate will be executed."""

_DECIDER_ROLE = """Decide the single best NEXT action from the current browser observation and
return only that action. Nothing selects after you, so commit to one action."""

if _PLANNER_ROLE not in core.CANDIDATE_SYSTEM:
    raise RuntimeError(
        "core.CANDIDATE_SYSTEM no longer contains the planner role paragraph "
        "that arms.py rewrites. Re-sync _PLANNER_ROLE before running, or the "
        "single-call arm silently stops being prompt-comparable."
    )

SINGLE_CALL_SYSTEM = core.CANDIDATE_SYSTEM.replace(_PLANNER_ROLE, _DECIDER_ROLE, 1)

# Guide section 10.6: arms that emit several candidates survive one candidate
# failing validation; an arm that emits one does not. A single retry carrying the
# rejection back is the equivalent recovery, not an extra advantage. Its cost is
# charged to the same step and the attempt count is logged.
CORRECTION_TEMPLATE = """YOUR PREVIOUS ACTION FOR THIS STEP WAS REJECTED BEFORE EXECUTION:
{rejected}

It was rejected because the target was not uniquely resolvable in the snapshot,
was not an actionable control, or did not appear in the snapshot at all. A label
shown both as an image link and as a text link counts as appearing more than once.

Return one corrected action. If the label appears more than once, add a `context`
string copied from a nearby snapshot line so the target resolves to exactly one
control. Otherwise target a different visible control that reaches the same place.

"""


def _history_tail(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "step": x.get("step"),
            "selected_id": x.get("selected_id"),
            "action": x.get("selected_action"),
            "action_error": core.summarize_action_error(x.get("action_error")),
            "url_before": x.get("url_before"),
            "url_after": x.get("url_after"),
            "active_tab_before": x.get("active_tab_before"),
            "active_tab_after": x.get("active_tab_after"),
            "new_tabs_opened": x.get("new_tabs_opened"),
        }
        for x in history[-core.FAILED_ACTION_WINDOW :]
    ]


def _attempt_single_call(
    goal: str,
    state: Any,
    memory: str,
    history: list[dict[str, Any]],
    correction: str = "",
) -> tuple[dict[str, Any], dict[str, Any]]:
    prompt = f"""BENCHMARK TASK:
{goal}

CURRENT URL:
{state.url}

CURRENT TITLE:
{state.title}

TRACKED BENCHMARK TABS:
{json.dumps(core.tracked_tabs_for_prompt(), ensure_ascii=False, indent=2)}

WORKING MEMORY FROM PRIOR OBSERVATIONS:
{memory or "(empty)"}

RECENT ACTION HISTORY:
{json.dumps(_history_tail(history), ensure_ascii=False, indent=2)}

CURRENT INTERACTIVE SNAPSHOT:
<<<SNAPSHOT
{state.snapshot}
SNAPSHOT

{correction}Decide the next action yourself and return exactly this JSON shape:
{{
  "memory_update": "Only factual information newly visible in this snapshot that may help answer the task. Empty string if none.",
  "action_rationale": "One sentence on why this action is the best next step",
  "action": {{
    "type": "search",
    "context": "optional nearby card or section text",
    "frame": "optional iframe label",
    "target": "combobox \\"search\\"",
    "query": "example query"
  }}
}}

Return exactly one action. Do not return a list of options.
Targets for click/fill/press/search must be actionable controls. Never target an
`iframe`, heading, text node, or other container; target a visible control inside
it, or choose a different action when no such control is exposed.
If a target appears multiple times, context is required and must make it unique.
Use frame when the target is a child control inside a named iframe subtree.
Do not repeat an action that RECENT ACTION HISTORY shows failed on the current URL.
Choose a materially different action such as another visible control, back, or wait.
If the task appears complete, return:
{{
  "type": "done",
  "answer": "concise final answer grounded only in observed facts and working memory"
}}
as the action.

Use only the allowed action types.
Never produce a goto action or a guessed URL. Search using visible browser UI.
Never put session-local refs (e1, e2, ...) in the action; use semantic `target`.
The tracked tab whose `active` field is true produced CURRENT INTERACTIVE SNAPSHOT.
When a click opens a new tab, that new tab becomes active automatically. Inspect it
before moving on. Use `switch_tab` with `tab_index` to return to a search/results tab
or to revisit an already-open resource.
If no visible control can be used safely from this state, return action
{{"type": "abstain"}}.
"""

    wrapper, meta = core.codex_json(prompt, SINGLE_CALL_SYSTEM)
    data = json.loads(core.strip_json_fence(wrapper["result"]))

    raw_action = data.get("action")
    raw_candidates = (
        [
            {
                "id": "a1",
                "description": str(data.get("action_rationale") or "").strip()
                or "Model-selected next action.",
                "action": raw_action,
            }
        ]
        if isinstance(raw_action, dict)
        else []
    )

    validated = core.validate_candidates(
        raw_candidates,
        state.snapshot,
        tab_count=len(core.BROWSER_STATE["tabs"]),
        current_tab_index=core.active_tab_index(),
        alive_tab_indexes={
            index
            for index, tab in enumerate(core.BROWSER_STATE["tabs"].values())
            if tab.get("alive", False)
        },
    )
    context_inferred = sum(1 for c in validated if c.get("context_inferred"))
    filtered, failed_filtered = core.filter_recently_failed_actions(
        validated, history, state.url
    )
    filtered, completed_filtered = (
        core.filter_recently_completed_navigation_actions(filtered, history, state.url)
    )

    candidates: list[dict[str, Any]] = []
    if len(filtered) == 1 and filtered[0]["action"].get("type") != "abstain":
        candidate = filtered[0]
        candidate["id"] = "a1"
        candidates.append(candidate)
    candidates.append(
        {
            "id": "abstain",
            "description": "The single-call planner did not yield one safe executable action.",
            "action": {"type": "abstain"},
        }
    )

    # Guide section 10.5: keep the raw value, so a dropped action can be told
    # apart from a validator false positive without re-running.
    meta.update(
        {
            "raw_action": raw_action,
            "raw_action_validated": len(validated) > 0,
            "validation_dropped_only_action": bool(raw_candidates) and not filtered,
        }
    )
    return (
        {
            "memory_update": str(data.get("memory_update") or "").strip(),
            "candidates": candidates,
            "failed_action_candidates_filtered": failed_filtered,
            "completed_navigation_candidates_filtered": completed_filtered,
            "context_inferred_candidates": context_inferred,
        },
        meta,
    )


def plan_single_call(
    goal: str,
    state: Any,
    memory: str,
    history: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Decide the next action, retrying once if validation drops the only one."""
    planned, meta = _attempt_single_call(goal, state, memory, history)
    if not meta.get("validation_dropped_only_action"):
        meta["attempts"] = 1
        return planned, meta

    rejected = json.dumps(meta.get("raw_action"), ensure_ascii=False)
    retry_planned, retry_meta = _attempt_single_call(
        goal,
        state,
        memory,
        history,
        correction=CORRECTION_TEMPLATE.format(rejected=rejected),
    )
    # The retry is a second LLM call; charge both to this step so the arm's
    # timing reflects what it actually spent.
    retry_meta["e2e_latency_ms"] = float(meta.get("e2e_latency_ms") or 0.0) + float(
        retry_meta.get("e2e_latency_ms") or 0.0
    )
    retry_meta["attempts"] = 2
    retry_meta["first_attempt_rejected_action"] = meta.get("raw_action")
    return retry_planned, retry_meta


# ---------------------------------------------------------------------------
# Arm registry
# ---------------------------------------------------------------------------

Planner = Callable[
    [str, Any, str, list[dict[str, Any]]], tuple[dict[str, Any], dict[str, Any]]
]

# The driver value that means "the planner already committed; execute its only
# candidate and make no selector call".
PLANNER_DRIVES = "planner"


@dataclass(frozen=True)
class Arm:
    name: str
    planner: Planner
    selectors: tuple[str, ...]
    driver: str
    llm_calls_per_step: str
    role: str
    comparable_total_runtime: bool

    @property
    def needs_jev(self) -> bool:
        return "jev" in self.selectors


ARMS: dict[str, Arm] = {
    "single-call": Arm(
        name="single-call",
        planner=plan_single_call,
        selectors=(),
        driver=PLANNER_DRIVES,
        llm_calls_per_step="LLM 1",
        role="reference baseline (guide 3.2) - the agent someone would build",
        comparable_total_runtime=True,
    ),
    "gen+gpt-select": Arm(
        name="gen+gpt-select",
        planner=core.generate_candidates,
        selectors=("gpt",),
        driver="gpt",
        llm_calls_per_step="LLM 2",
        role="main comparison, GPT side (guide 3.1)",
        comparable_total_runtime=True,
    ),
    "gen+jev-select": Arm(
        name="gen+jev-select",
        planner=core.generate_candidates,
        selectors=("jev",),
        driver="jev",
        llm_calls_per_step="LLM 1 + JEV 1",
        role="main comparison, JEV side (guide 3.1)",
        comparable_total_runtime=True,
    ),
    "shadow": Arm(
        name="shadow",
        planner=core.generate_candidates,
        # Both selectors see the same candidate set from one shared generation
        # call. This is how the main comparison is actually run (guide 6).
        selectors=("gpt", "jev"),
        driver="gpt",
        llm_calls_per_step="LLM 2 + JEV 1",
        role="main comparison, run as one paired trajectory (guide 6)",
        # Guide section 6 "the cost": an extra selection call per step means this
        # run's wall clock matches no real configuration. Refuse to report it.
        comparable_total_runtime=False,
    ),
}

# `jev-only` is intentionally absent. Guide section 3.3: its candidates come from
# deterministic code rather than an LLM, so it is a system-replacement question,
# not a selector question, and it does not belong in the same table as the main
# comparison. Add it as its own arm with its own planner when that experiment is
# actually wanted.

MAIN_COMPARISON = ("gen+gpt-select", "gen+jev-select")


def get(name: str) -> Arm:
    try:
        return ARMS[name]
    except KeyError:
        raise SystemExit(
            f"ERROR: unknown arm {name!r}. Known arms: {', '.join(sorted(ARMS))}"
        ) from None
