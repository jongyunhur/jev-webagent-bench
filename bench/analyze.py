"""Apply the guide's metrics and decision rules to a finished batch.

This exists so the conclusion is read off the pre-registered rules in guide
section 8 rather than chosen after looking at the numbers. It reports:

    agreement rate          guide 7.1
    per-selector latency    guide 7.2, printed with the measured floor beside it
    confidence split        guide 7.3, JEV confidence on agree vs disagree steps
    disagreement list       guide 7.4, the steps a human still has to judge
    controls                guide 7.5
    verdict                 guide 8, chosen by rule, not by taste

It refuses to analyse a batch containing an unfinished run (guide 10.4).

Usage:

    python -m bench.analyze results/20260921-101500_shadow
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FLOOR = REPO_ROOT / "results" / "call_floor.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Report guide metrics and the pre-registered verdict."
    )
    parser.add_argument("batch_dir", help="a results/<batch_id>_<arm> directory")
    parser.add_argument("--runs-dir", default=str(REPO_ROOT / "runs"))
    parser.add_argument("--floor", default=str(DEFAULT_FLOOR))
    parser.add_argument("--out", default=None, help="write the report as JSON here")
    return parser.parse_args()


def load_steps(runs_dir: Path, run_ids: list[str]) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    for run_id in run_ids:
        path = runs_dir / run_id / "steps.jsonl"
        if not path.exists():
            raise SystemExit(f"ERROR: step log missing for {run_id}: {path}")
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                steps.append(json.loads(line))
    return steps


def pct(part: int, whole: int) -> float | None:
    return None if whole == 0 else part / whole * 100.0


def main() -> int:
    args = parse_args()
    batch_dir = Path(args.batch_dir)
    manifest = json.loads((batch_dir / "manifest.json").read_text(encoding="utf-8"))
    runs = json.loads((batch_dir / "runs.json").read_text(encoding="utf-8"))

    incomplete = [r for r in runs if r.get("stop_reason") != "done"]
    if incomplete:
        print("ERROR: this batch contains unfinished runs; refusing to analyse.")
        for row in incomplete:
            print(f"  {row.get('level')}/r{row.get('repeat')}: {row.get('stop_reason')}")
        print("\nGuide 10.4: an unfinished run ends early, so its totals read as a")
        print("speed advantage it did not earn. Re-run them first.")
        return 1

    selectors: list[str] = list(manifest.get("selectors") or [])
    steps = load_steps(Path(args.runs_dir), [r["run_id"] for r in runs])

    report: dict[str, Any] = {
        "batch_id": manifest["batch_id"],
        "arm": manifest["arm"],
        "driver": manifest["driver"],
        "repeats": manifest["repeats"],
        "runs": len(runs),
        "steps": len(steps),
    }

    print("=" * 74)
    print(f"BATCH {manifest['batch_id']}  |  arm {manifest['arm']}  |  driver {manifest['driver']}")
    print(f"{len(runs)} runs, {len(steps)} steps, {manifest['repeats']} repeat(s)")
    print("=" * 74)

    # --- guide 7.1 agreement ------------------------------------------------
    comparable = [s for s in steps if s.get("agreed") is not None]
    agreed = sum(1 for s in comparable if s["agreed"])
    agreement_rate = pct(agreed, len(comparable))
    report["agreement"] = {
        "comparable_steps": len(comparable),
        "agreed_steps": agreed,
        "rate_pct": agreement_rate,
    }

    print("\n[7.1] AGREEMENT")
    if not comparable:
        print("  not applicable - this arm consults one selector, so there is")
        print("  nothing to agree with. Run --arm shadow for the main comparison.")
    else:
        print(f"  {agreed}/{len(comparable)} steps  ({agreement_rate:.1f}%)")

    # --- guide 7.2 latency, raw, with the floor beside it -------------------
    print("\n[7.2] PER-CALL LATENCY")
    latency: dict[str, Any] = {}
    for name in selectors:
        per_call = [
            float(s["selections"][name]["e2e_latency_ms"])
            for s in steps
            if s.get("selections", {}).get(name)
        ]
        if not per_call:
            continue
        latency[name] = {
            "calls": len(per_call),
            "mean_ms": statistics.fmean(per_call),
            "median_ms": statistics.median(per_call),
            "boundary": manifest["selector_boundaries"].get(name),
        }
        print(
            f"  {name:4} {statistics.fmean(per_call) / 1000:6.2f}s/call"
            f"  (median {statistics.median(per_call) / 1000:.2f}s, n={len(per_call)})"
        )
        print(f"       boundary: {latency[name]['boundary']}")

    floor_path = Path(args.floor)
    floor_note = None
    if floor_path.exists():
        floor = json.loads(floor_path.read_text(encoding="utf-8"))
        floor_ms = floor["stats_ms"]["min_ms"]
        spread_ms = floor["spread_ms"]
        report["floor"] = floor
        print(f"\n  measured call floor: {floor_ms / 1000:.2f}s  (any CLI call)")
        print(f"  floor spread       : {spread_ms / 1000:.2f}s on an identical prompt")
        if not floor.get("trustworthy"):
            print("  WARNING: floor came from fewer than 20 samples; re-measure.")
        floor_note = (
            "Report both per-call numbers and this floor in the same sentence. "
            "Do not subtract (guide 5.2)."
        )
        print(f"  {floor_note}")
    else:
        print(f"\n  NO FLOOR MEASURED. Run `python -m bench.floor` first (guide 5.1);")
        print("  without it no latency claim is reportable.")
    report["latency"] = latency

    # --- guide 7.3 confidence ----------------------------------------------
    # Every selector that recorded a confidence gets its own split. The two are
    # not the same kind of number: JEV's distribution comes from the model, the
    # GPT one is written out by the model as text. The step log says which, and
    # it is printed here so the comparison cannot be read as like for like.
    report["confidence"] = {}
    if comparable:
        print("\n[7.3] CONFIDENCE, split by whether the two selectors agreed")
    for name in selectors:
        if not comparable:
            break

        def confidences(want_agreement: bool, who: str = name) -> list[float]:
            out = []
            for step in comparable:
                if step["agreed"] is not want_agreement:
                    continue
                value = step["selections"].get(who, {}).get("confidence")
                if isinstance(value, (int, float)):
                    out.append(float(value))
            return out

        on_agree, on_disagree = confidences(True), confidences(False)
        if not on_agree and not on_disagree:
            print(f"  {name.upper():4} : no confidence recorded")
            continue
        kinds = {
            s["selections"][name].get("confidence_kind", "native")
            for s in comparable
            if name in s["selections"]
        }
        report["confidence"][name] = {
            "kind": sorted(kinds),
            "agree_mean": statistics.fmean(on_agree) if on_agree else None,
            "disagree_mean": statistics.fmean(on_disagree) if on_disagree else None,
            "agree_n": len(on_agree),
            "disagree_n": len(on_disagree),
        }
        print(f"  {name.upper():4} ({', '.join(sorted(kinds))})")
        if on_agree:
            print(f"       on agreement   : {statistics.fmean(on_agree):.3f}  (n={len(on_agree)})")
        if on_disagree:
            print(f"       on disagreement: {statistics.fmean(on_disagree):.3f}  (n={len(on_disagree)})")
        else:
            print("       on disagreement: no disagreements to measure")

    # --- guide 7.4 disagreements to judge by hand ---------------------------
    disagreements = [s for s in comparable if not s["agreed"]]
    report["disagreements"] = [
        {
            "run_id": s["run_id"],
            "level": s["level"],
            "step": s["step"],
            "url": s["url_before"],
            "choices": {n: s["selections"][n].get("choice") for n in selectors},
            "confidence": {
                n: s["selections"][n].get("confidence") for n in selectors
            },
        }
        for s in disagreements
    ]
    if disagreements:
        print(f"\n[7.4] DISAGREEMENTS - {len(disagreements)} step(s) to judge by hand")
        for item in report["disagreements"]:
            picks = "  ".join(f"{n}={item['choices'][n]}" for n in selectors)
            confs = "  ".join(
                f"{n}={item['confidence'][n]:.2f}"
                for n in selectors
                if isinstance(item["confidence"][n], (int, float))
            )
            print(f"  {item['level']:6} step {item['step']:2}  {picks}   conf: {confs}")
            print(f"         {item['url'][:88]}")

    # --- guide 7.5 controls -------------------------------------------------
    print("\n[7.5] CONTROLS")
    by_level: dict[str, list[dict[str, Any]]] = {}
    for row in runs:
        by_level.setdefault(row["level"], []).append(row)
    controls: dict[str, Any] = {}
    for level, level_runs in by_level.items():
        step_counts = sorted({r["steps_logged"] for r in level_runs})
        urls = sorted({r["final_url"] for r in level_runs})
        errors = sum(r["action_error_count"] for r in level_runs)
        controls[level] = {
            "step_counts": step_counts,
            "distinct_final_urls": len(urls),
            "action_errors": errors,
        }
        flag = "" if len(step_counts) == 1 and len(urls) == 1 else "   <- varies"
        print(f"  {level:6} steps={step_counts} final_urls={len(urls)} errors={errors}{flag}")
    report["controls"] = controls

    # --- guide 8 verdict, by rule -------------------------------------------
    print("\n" + "=" * 74)
    print("[8] VERDICT (pre-registered rule, not chosen after the fact)")
    print("=" * 74)

    if not comparable:
        verdict = "not_applicable"
        print("  This arm consults one selector. No JEV claim follows from it.")
    elif agreement_rate == 100.0:
        verdict = "no_discriminating_power"
        print("  Agreement is 100%. THESE TASKS CANNOT TELL THE SELECTORS APART.")
        print("  Report that finding. Do NOT report a latency advantage as a JEV")
        print("  result. The next variable to change is task difficulty.")
    else:
        verdict = "disagreements_found_adjudicate"
        print(f"  Agreement is {agreement_rate:.1f}%. {len(disagreements)} disagreement(s)")
        print("  must be judged by hand before any accuracy claim. The sample size")
        print(f"  for that claim is {len(disagreements)}, not {len(steps)}.")

    if manifest["repeats"] < 3:
        print(f"\n  CAUTION: {manifest['repeats']} repeat(s). Guide 10.3 forbids")
        print("  claiming any percentage from this batch.")
    if not manifest.get("comparable_total_runtime", True):
        print("\n  CAUTION: this arm's wall clock matches no real configuration")
        print("  (guide 6). Do not compare its total against another arm's.")

    report["verdict"] = verdict

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nWritten: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
