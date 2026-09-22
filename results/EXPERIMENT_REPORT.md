# JEV next-action experiment — 2026-09-21

## Protocol and evidence

The three tasks in `data/pilot_tasks.json` were run three times each under the paired `shadow` arm, the `single-call` reference arm, and the real `gen+jev-select` arm. The shadow arm gave GPT and JEV the same generated candidates and executed GPT's choice. GPT-5.5 was called through Codex OAuth; JEV used the local credential supplied in `.env`. A 20-call GPT CLI floor measurement preceded the batches.

All 27 runs ended with `done`, had zero action errors, and visited the required product detail pages in order. The final answers contain the required product URLs and UPCs and the task totals. The per-run audit is in `experiment_audit.json`; raw step records remain in `../runs/<run_id>/steps.jsonl`.

| Batch | Runs | Steps | Result files |
|---|---:|---:|---|
| Paired `shadow` | 9 | 90 | `20260921-174919_shadow/` |
| `single-call` reference | 9 | 90 | `20260921-183136_single-call/` |
| JEV driven `gen+jev-select` | 9 | 90 | `20260921-185403_gen+jev-select/` |

## Scope

The JEV integration measured here is given candidate actions produced by GPT-5.5
at high reasoning effort. The reference implementation published by browser-use
(`jev-ultrafast`) does not work this way: it extracts an indexed control table
from the DOM deterministically and issues one TypeSafe request that predicts
operation and target together, calling a small LLM only to write text.

Everything below therefore describes one integration of JEV, not the
architecture JEV is designed for. The end-to-end comparison shows that putting
an expensive generator in front of JEV costs more than JEV's faster selection
saves. It says nothing about the deterministic-extraction design, which was not
run here. The closest available data point is a `jev-only` arm in the earlier
repository - deterministic candidates, JEV selection, zero GPT calls - which
completed the easy task in 7.0 s and exhausted its step budget on the other two.

Published figures from other harnesses are not compared against these. Those
runs use a different site, task, browser transport and text model, so a direct
time comparison would repeat the measurement error this protocol exists to
avoid.

## Selector comparison

The paired selectors agreed on 75 of 90 steps (83.3%). Their raw mean selection-call times were 7.55 s for GPT through the Codex CLI and 0.69 s for JEV through the in-process SDK. These timers have different boundaries. The 20-call GPT CLI floor was 4.07 s minimum, with a 3.63 s range on the identical prompt. These figures are reported together without subtracting a correction. They show a faster measured JEV selection path in this harness, not an intrinsic model speed comparison.

JEV confidence averaged 0.852 on agreeing steps and 0.556 on disagreeing steps. The 15 disagreements are reviewed in `20260921-174919_shadow/disagreement_review.md`: 14 choices were browser back versus a category link with the same intended listing; one was Home versus the Philosophy listing as a route toward Poetry. Only GPT's choice was executed in the paired arm, so the alternate outcomes were not observed. The disagreements do not establish an accuracy advantage for either selector.

## End-to-end reference

Mean completed run time across three repeats at each task level:

| Task | `single-call` | `gen+jev-select` | Observed JEV driven difference |
|---|---:|---:|---:|
| Easy | 26.18 s | 31.03 s | +4.85 s (+18.5%) |
| Medium | 96.36 s | 130.86 s | +34.50 s (+35.8%) |
| Hard | 292.22 s | 429.24 s | +137.02 s (+46.9%) |

Both actual configurations completed all tasks in 2, 7, and 21 steps respectively. The JEV driven two-stage architecture was slower end to end than the one-call reference in these runs. This comparison changes architecture as well as selector and therefore does not isolate a JEV accuracy or speed effect. The arms ran in separate blocks, so session drift may also affect the observed timing gaps. The `shadow` arm includes an extra selector call and its total wall time is not used in this table.

The paired result supports a narrower finding: JEV usually selected the same next action as GPT on this task set, and its measured selector call path was shorter. A task set with decisions that lead to materially different outcomes is needed to test relative selection accuracy.

## Artifacts

- `call_floor.json` — 20-call GPT CLI floor distribution.
- `20260921-174919_shadow/analysis.json` — paired agreement, latency, confidence, controls, and registered verdict.
- `20260921-174919_shadow/disagreement_review.md` — all 15 disputed steps.
- `20260921-183136_single-call/analysis.json` — reference controls.
- `20260921-185403_gen+jev-select/analysis.json` — JEV driven controls and call times.
- `experiment_audit.json` — 27-run completion and task-content audit.
