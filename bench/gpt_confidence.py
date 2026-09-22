"""GPT-5.5 selector that reports a probability for every candidate.

Why this exists: the JEV selector returns `confidence` and `probabilities`, the
plain GPT-5.5 selector returned only a choice, so the two could not be compared
on confidence. The published `system-one-adapter` shows how to ask an ordinary
LLM for the same thing, but it needs an API key. The benchmark reaches GPT-5.5
through Codex OAuth, which that adapter cannot use (its provider takes no
custom headers and the OAuth token is not an API key).

So this module keeps the transport the benchmark already uses (`core.codex_json`)
and takes only two things from the adapter:

  1. the instruction that asks for a probability per label, and
  2. the confidence formula, `choice_confidence`.

The formula is read out of the adapter's own source rather than retyped, and the
instruction is checked against that same file, so a reworded prompt upstream
stops the run instead of silently making the two selectors incomparable. The
adapter is a declared dependency, so `uv sync` supplies it; a clone beside this
repo is used only as a fallback.

What the number is, and is not. The probabilities are written out by the model
as text; they are self-reported, not token log-probabilities. JEV's come from
the model itself. The same formula is applied to both (verified: it reproduces
the confidence JEV recorded on all 90 steps of the earlier batch, max error
0.017), but the distributions have different origins, so the two confidences
are comparable in scale and not equivalent in meaning. Every result carries
`confidence_kind` so a figure cannot forget that.
"""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path
from typing import Any

from . import core


# The adapter is a declared dependency (`system-one-adapter` on PyPI), so the
# formula comes from the installed package. A local clone beside this repo is
# used only if the package is missing, which keeps a checkout of the adapter's
# source working without making this repo depend on one being there.
ADAPTER_DIR_FALLBACK = (
    Path(__file__).resolve().parent.parent
    / "system-one-adapter-python"
    / "src"
    / "system_one_adapter"
)

# Two parts, kept apart on purpose. The lead is ours: the adapter's sentence is
# "For Choice and Score questions, return an object mapping ...", which does not
# read naturally in this prompt. The tail is verbatim from the adapter's
# _PROBABILITY_SYSTEM_PROMPT, and only the tail is checked byte for byte.
PROBABILITY_LEAD = "Return an object mapping every allowed label to its probability."
PROBABILITY_TAIL = (
    "Preserve genuine uncertainty. Include every allowed label, do not add "
    "labels, keep each probability between 0 and 1, and make the probabilities "
    "sum to 1."
)
PROBABILITY_INSTRUCTION = f"{PROBABILITY_LEAD} {PROBABILITY_TAIL}"

CONFIDENCE_KIND = "self-reported"


def _squash(text: str) -> str:
    return re.sub(r"\s+", " ", text)


def _adapter_dir() -> Path:
    """Directory of the adapter package, installed for preference."""
    spec = importlib.util.find_spec("system_one_adapter")
    if spec is not None and spec.origin:
        return Path(spec.origin).parent
    if (ADAPTER_DIR_FALLBACK / "_client.py").exists():
        return ADAPTER_DIR_FALLBACK
    raise RuntimeError(
        "system-one-adapter is not available. It is a declared dependency, so "
        "`uv sync` should provide it; otherwise clone "
        "https://github.com/typesafe-ai/system-one-adapter-python beside this repo."
    )


def _load_adapter_formula() -> Any:
    directory = _adapter_dir()
    client = directory / "_client.py"
    if not client.exists():
        raise RuntimeError(f"adapter found at {directory} but _client.py is missing")

    # The instruction below is quoted from the adapter, so confirm the adapter
    # still says it. If upstream rewords the prompt, the two selectors stop
    # being asked the same thing and the run should stop rather than quietly
    # produce confidences that are no longer comparable.
    adapter_text = _squash(client.read_text(encoding="utf-8"))
    if (
        _squash(PROBABILITY_TAIL) not in adapter_text
        or "mapping every allowed label to its probability" not in adapter_text
    ):
        raise RuntimeError(
            "The adapter's probability instruction no longer matches the copy in "
            "gpt_confidence.py. Re-read _client.py and update PROBABILITY_TAIL "
            "before running, or the GPT and JEV confidences stop being comparable."
        )

    # Loaded from the file that was just checked, so the formula and the
    # instruction always come from the same copy of the adapter - installed or
    # cloned. Importing the package instead could pick up a different one.
    metrics = directory / "_utils" / "confidence_metrics.py"
    spec = importlib.util.spec_from_file_location("_adapter_confidence_metrics", metrics)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load the adapter's confidence metrics from {metrics}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.choice_confidence


choice_confidence = _load_adapter_formula()


def _parse_probabilities(raw: str, ids: list[str]) -> dict[str, float]:
    data = json.loads(core.strip_json_fence(raw))
    probs = data.get("probabilities") if isinstance(data, dict) else None
    if not isinstance(probs, dict):
        raise ValueError('expected an object with a "probabilities" mapping')
    missing = [i for i in ids if i not in probs]
    extra = [k for k in probs if k not in ids]
    if missing or extra:
        raise ValueError(f"labels must match exactly; missing={missing} unexpected={extra}")
    out: dict[str, float] = {}
    for key in ids:
        value = probs[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"probability for {key!r} is not a number: {value!r}")
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"probability for {key!r} is outside [0, 1]: {value}")
        out[key] = float(value)
    if sum(out.values()) <= 0.0:
        raise ValueError("probabilities sum to zero")
    return out


def select_with_gpt55_probabilities(
    goal: str,
    state: Any,
    memory: str,
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    table = core.candidate_table(candidates)
    ids = list(table)
    shape = json.dumps(
        {"probabilities": {cid: "<probability>" for cid in ids}}, ensure_ascii=False
    )

    base_prompt = f"""BENCHMARK TASK:
{goal}

CURRENT URL:
{state.url}

TRACKED BENCHMARK TABS:
{json.dumps(core.tracked_tabs_for_prompt(), ensure_ascii=False, indent=2)}

WORKING MEMORY:
{memory or "(empty)"}

CURRENT SNAPSHOT:
<<<SNAPSHOT
{state.snapshot}
SNAPSHOT

CANDIDATES:
{json.dumps(table, ensure_ascii=False, indent=2)}

Estimate how likely each candidate is to be the single best NEXT action.
{PROBABILITY_INSTRUCTION}

Return exactly this JSON shape and nothing else:
{shape}
"""

    # One corrective retry, as the adapter does for malformed output. Both calls
    # are charged to the step so the latency is what the selector really cost.
    total_ms = 0.0
    attempts = 0
    prompt = base_prompt
    last_error = ""
    meta: dict[str, Any] = {}
    for attempts in (1, 2):
        wrapper, meta = core.codex_json(prompt, core.SELECTOR_SYSTEM)
        total_ms += float(meta.get("e2e_latency_ms") or 0.0)
        try:
            probs = _parse_probabilities(wrapper["result"], ids)
            break
        except (ValueError, json.JSONDecodeError) as exc:
            last_error = str(exc)
            prompt = (
                base_prompt
                + f"\nYOUR PREVIOUS ANSWER WAS REJECTED: {last_error}\n"
                "Return the corrected JSON only.\n"
            )
    else:
        raise RuntimeError(f"GPT-5.5 returned invalid probabilities twice: {last_error}")

    raw_sum = sum(probs.values())
    normalised = {k: v / raw_sum for k, v in probs.items()}
    ranked = sorted(normalised.values(), reverse=True)
    choice = max(normalised, key=normalised.__getitem__)

    return {
        **meta,
        "selector": "gpt-5.5",
        "choice": choice,
        "confidence": choice_confidence(list(normalised.values())),
        "probabilities": normalised,
        "confidence_kind": CONFIDENCE_KIND,
        "probabilities_raw_sum": raw_sum,
        "top_two_tied": len(ranked) > 1 and ranked[0] == ranked[1],
        "attempts": attempts,
        "e2e_latency_ms": round(total_ms, 2),
    }
