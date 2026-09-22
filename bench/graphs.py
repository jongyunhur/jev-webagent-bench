#!/usr/bin/env python3
"""Render the experiment's figures from the retained batch results.

Drawn in the conference-paper idiom: hatched fills with thin dark outlines, a
boxed legend above the panel, rotated group labels down the left edge, a dashed
rule between groups, a framed plot area, and a caption underneath.

The hatch is not decoration. Series identity is carried by pattern and outline
as well as by fill, so the panels survive greyscale printing and full-severity
colour-vision deficiency without relying on hue at all.

  figure1_runtime.png      end-to-end wall-clock time, per task and arm
  figure2_agreement.png    what the two selectors chose, over all paired steps
  figure3_confidence.png   JEV confidence on every paired step

Everything is computed from results/ and runs/, so the figures follow the data
rather than being retyped from a report. Nothing here reads the shadow arm's
wall clock: that arm adds a selection call per step and its total matches no
real configuration (guide section 6).

Usage:
    python -m bench.graphs
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle


REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS = REPO_ROOT / "results"
RUNS = REPO_ROOT / "runs"
ASSETS = REPO_ROOT / "assets"

LEVELS = ("easy", "medium", "hard")
LEVEL_TITLES = {"easy": "Easy", "medium": "Medium", "hard": "Hard"}
ARMS = ("single-call", "gen+jev-select")
ARM_LABELS = {
    "single-call": "GPT decides",
    "gen+jev-select": "GPT lists,\nJEV picks",
}

INK = "#00000d"
RULE = "#00000d"

# (fill, edge, hatch) per series. Fills stay pale so the hatch reads on top of
# them; the outline and the hatch angle are what separate the series in print.
PHASES = (
    ("generation_ms", "GPT-5.5 call", "#eef3f7", "#215281", ""),
    ("selector_ms", "JEV selection", "#cedde5", "#215281", "///"),
    ("browser_ms", "Browser actions", "#e6e6e2", "#5a5a54", "\\\\\\"),
)

SAME_CHOICE = ("#e6e6e2", "#5a5a54", "")
SAME_DEST = ("#cedde5", "#215281", "///")
DIFF_DEST = ("#8fb4c9", "#215281", "xxx")

BAR_EDGE_W = 0.9
FRAME_W = 1.1
DPI = 300


def use_paper_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "hatch.linewidth": 0.6,
        "axes.linewidth": FRAME_W,
        "axes.edgecolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "text.color": INK,
        "axes.labelcolor": INK,
    })


def frame(ax: Any) -> None:
    """Full box around the plot area, ticks outward, no vertical grid clutter."""
    ax.set_facecolor("#ffffff")
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(FRAME_W)
        ax.spines[side].set_color(INK)
    ax.tick_params(axis="x", direction="out", length=3.2, width=0.9,
                   labelsize=10.5, pad=3)
    ax.set_yticks([])


def group_label(ax: Any, y: float, text: str) -> None:
    """Rotated group name down the left edge, outside the frame."""
    ax.text(-0.225, y, text, transform=ax.get_yaxis_transform(),
            rotation=90, ha="center", va="center", fontsize=11.5, color=INK)


def separator(ax: Any, y: float) -> None:
    ax.axhline(y, color=RULE, linestyle=(0, (5, 4)), linewidth=1.0, zorder=5)


def boxed_legend(ax: Any, entries: Iterable[tuple[str, str, str, str]], ncol: int,
                 fontsize: float = 11.0, handlelength: float = 2.4) -> None:
    """Boxed key above the panel. Long entry sets need a smaller size, or the
    box grows past the frame it is supposed to sit over."""
    handles = [
        Patch(facecolor=fill, edgecolor=edge, hatch=hatch, linewidth=BAR_EDGE_W, label=name)
        for name, fill, edge, hatch in entries
    ]
    legend = ax.legend(
        handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.015),
        ncol=ncol, frameon=True, fontsize=fontsize, handlelength=handlelength,
        handleheight=1.25, borderpad=0.42, columnspacing=1.2, handletextpad=0.5,
    )
    legend.get_frame().set_edgecolor(INK)
    legend.get_frame().set_linewidth(FRAME_W)
    legend.get_frame().set_boxstyle("Square", pad=0.34)


def caption(fig: Any, x: float, y: float, text: str) -> None:
    fig.text(x, y, text, ha="center", va="center", fontsize=12.5, color=INK)


def bar(ax: Any, x: float, y: float, w: float, h: float,
        fill: str, edge: str, hatch: str) -> None:
    ax.add_patch(Rectangle((x, y), w, h, facecolor=fill, edgecolor=edge,
                           hatch=hatch, linewidth=BAR_EDGE_W, zorder=3))


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def newest_batch(arm: str) -> Path:
    """Most recent batch for an arm in which every run finished.

    An unfinished run ends early, so its bar would read as a speed advantage it
    did not earn (guide 10.4); such a batch is skipped, not plotted.
    """
    complete = []
    for path in sorted(RESULTS.glob(f"*_{arm}")):
        runs_file = path / "runs.json"
        if not runs_file.exists():
            continue
        runs = json.loads(runs_file.read_text(encoding="utf-8"))
        if runs and all(r.get("stop_reason") == "done" for r in runs) \
                and {r["level"] for r in runs} == set(LEVELS):
            complete.append(path)
    if not complete:
        raise SystemExit(f"ERROR: no complete batch for arm {arm!r} in {RESULTS}")
    return complete[-1]


def load_runs(batch: Path) -> list[dict[str, Any]]:
    return json.loads((batch / "runs.json").read_text(encoding="utf-8"))


def load_steps(runs: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    steps = []
    for run in runs:
        for line in (RUNS / run["run_id"] / "steps.jsonl").read_text(encoding="utf-8").splitlines():
            if line.strip():
                steps.append(json.loads(line))
    return steps


def phase_mean(runs: list[dict[str, Any]], level: str) -> tuple[float, ...]:
    rows = [r for r in runs if r["level"] == level]
    out = []
    for key, *_ in PHASES:
        values = []
        for row in rows:
            raw = row.get(key, 0.0)
            values.append(sum(raw.values()) if isinstance(raw, dict) else float(raw))
        out.append(statistics.fmean(values) / 1000.0)
    return tuple(out)


def classify(steps: list[dict[str, Any]]) -> dict[str, int]:
    buckets: dict[str, int] = defaultdict(int)
    for step in steps:
        if step.get("agreed") is not False:
            continue
        actions = {c["id"]: c["action"] for c in step["candidates"]}
        kinds = {actions.get(step["selections"][n]["choice"], {}).get("type")
                 for n in ("gpt", "jev")}
        buckets["route" if kinds == {"back", "click"} else "destination"] += 1
    return dict(buckets)


# ---------------------------------------------------------------------------
# Figure 1 - runtime
# ---------------------------------------------------------------------------

def figure_runtime(data: dict[str, list[dict[str, Any]]], out: Path, dpi: int) -> None:
    bar_h, arm_gap, group_gap = 0.62, 0.10, 0.44
    step = len(ARMS) * bar_h + (len(ARMS) - 1) * arm_gap + group_gap
    span = len(LEVELS) * step - group_gap

    fig = plt.figure(figsize=(7.8, 4.0), dpi=dpi, facecolor="#ffffff")
    ax = fig.add_axes((0.255, 0.215, 0.725, 0.595))

    peak = max(sum(phase_mean(data[a], l)) for a in ARMS for l in LEVELS)
    frame(ax)
    ax.set_xlim(0, peak * 1.16)
    ax.set_ylim(span, 0)
    ax.set_xlabel("wall-clock time (s)", fontsize=11, labelpad=2)

    for gi, level in enumerate(LEVELS):
        top = gi * step
        if gi:
            separator(ax, top - group_gap / 2)
        group_label(ax, top + (len(ARMS) * bar_h + arm_gap) / 2, LEVEL_TITLES[level])
        for ai, arm in enumerate(ARMS):
            y = top + ai * (bar_h + arm_gap)
            values = phase_mean(data[arm], level)
            total = sum(values)
            ax.text(-0.012, y + bar_h / 2, ARM_LABELS[arm],
                    transform=ax.get_yaxis_transform(), ha="right", va="center",
                    fontsize=10.5, color=INK, linespacing=1.3)
            cursor = 0.0
            for pi, value in enumerate(values):
                _, _, fill, edge, hatch = PHASES[pi]
                if value > 0:
                    bar(ax, cursor, y, value, bar_h, fill, edge, hatch)
                cursor += value
            ax.text(total + peak * 0.012, y + bar_h / 2, f"{total:.0f} s",
                    ha="left", va="center", fontsize=10.5, color=INK)

    boxed_legend(ax, [(n, f, e, h) for _, n, f, e, h in PHASES], ncol=3, fontsize=10.5)
    caption(fig, 0.62, 0.045,
            "End-to-end runtime of a completed task, mean of 3 repeats")
    fig.savefig(out, dpi=dpi, facecolor="#ffffff")
    plt.close(fig)
    print(f"  {out}")


# ---------------------------------------------------------------------------
# Figure 2 - agreement
# ---------------------------------------------------------------------------

def figure_agreement(steps: list[dict[str, Any]], out: Path, dpi: int) -> None:
    comparable = [s for s in steps if s.get("agreed") is not None]
    agreed = sum(1 for s in comparable if s["agreed"])
    buckets = classify(steps)
    route, destination = buckets.get("route", 0), buckets.get("destination", 0)

    fig = plt.figure(figsize=(7.4, 2.6), dpi=dpi, facecolor="#ffffff")
    ax = fig.add_axes((0.155, 0.335, 0.815, 0.325))

    # One bar, three outcomes. Two rows would have to reuse the same legend
    # entries for two different meanings, and the finding is precisely that
    # "differed" and "went somewhere else" are not the same thing here.
    frame(ax)
    ax.set_xlim(0, 100)
    ax.set_ylim(1, 0)
    ax.set_xticks(range(0, 101, 10))
    ax.set_xticklabels([f"{v}%" for v in range(0, 101, 10)])

    total = len(comparable) or 1
    parts = (
        (agreed, "Same choice", SAME_CHOICE),
        (route, "Differed, same destination", SAME_DEST),
        (destination, "Differed, other destination", DIFF_DEST),
    )
    cursor = 0.0
    for count, _, (fill, edge, hatch) in parts:
        width = count / total * 100
        if width <= 0:
            continue
        bar(ax, cursor, 0, width, 1, fill, edge, hatch)
        # Only label a segment wide enough to hold the text; a number crushed
        # against two edges is worse than the axis carrying it.
        if width >= 7:
            ax.text(cursor + width / 2, 0.5, str(count), ha="center", va="center",
                    fontsize=10.5, color=INK, zorder=4)
        cursor += width
    ax.text(-0.012, 0.5, f"{total} steps", transform=ax.get_yaxis_transform(),
            ha="right", va="center", fontsize=11, color=INK)
    boxed_legend(ax, [(name, *style) for _, name, style in parts], ncol=3,
                 fontsize=9.5, handlelength=1.9)

    caption(fig, 0.56, 0.085, "What the two selectors chose, over all paired steps")
    fig.savefig(out, dpi=dpi, facecolor="#ffffff")
    plt.close(fig)
    print(f"  {out}")


# ---------------------------------------------------------------------------
# Figure 3 - confidence
# ---------------------------------------------------------------------------

def figure_confidence(steps: list[dict[str, Any]], out: Path, dpi: int) -> None:
    comparable = [s for s in steps if s.get("agreed") is not None]

    # Two rows per selector, both selectors in the one figure, so the reader
    # never has to hold "which row was JEV again?" across two files. Marker
    # SHAPE carries agreement status (same as figure 2: circle = same choice,
    # square = differed); the rotated block label on the left carries selector
    # identity, the same device figure 1 uses for the task name.
    outcomes = (
        (True, "Same choice", "o", "#ffffff", "#5a5a54"),
        (False, "Differed", "s", "#cedde5", "#215281"),
    )
    selectors = (("jev", "JEV", "native"), ("gpt", "GPT", "self-reported"))

    row_h, row_gap, block_gap = 1.0, 0.14, 0.55
    rows: list[dict[str, list[float]]] = []
    for key, _, _ in selectors:
        values: dict[str, list[float]] = {True: [], False: []}
        for step in comparable:
            value = step["selections"].get(key, {}).get("confidence")
            if isinstance(value, (int, float)):
                values[step["agreed"]].append(float(value))
        rows.append(values)

    # Four row centres, spaced (row_h + row_gap) within a block and
    # (row_h + block_gap) across the block boundary, so both blocks occupy the
    # same vertical span (2 * row_h + row_gap each) and the separator - placed
    # at the true midpoint of the two block-adjacent rows, not by formula - cuts
    # exactly through the middle of that shared gap. An earlier version placed
    # the separator at a fixed offset from the second block's first row, which
    # left 1.275 units above it and 0.275 below: the JEV block looked ~4.6x
    # taller than GPT's for no reason in the data.
    row_ys = [0.0]
    for gap in (row_gap, block_gap, row_gap):
        row_ys.append(row_ys[-1] + row_h + gap)
    sep_y = (row_ys[1] + row_ys[2]) / 2.0
    margin = row_h / 2.0 + 0.12

    fig = plt.figure(figsize=(7.4, 4.7), dpi=dpi, facecolor="#ffffff")
    ax = fig.add_axes((0.235, 0.180, 0.735, 0.650))
    frame(ax)

    # Headroom past 1.0: a marker centred on 1.0 is otherwise cut in half by
    # the frame. The ticks still stop at 1.0, which is the real end of the
    # scale.
    ax.set_xlim(0.0, 1.03)
    ax.set_xticks([i / 10 for i in range(11)])
    ax.set_xticklabels([f"{i / 10:.1f}" for i in range(11)])
    ax.set_xlabel("confidence in its own choice", fontsize=11, labelpad=2)

    # Symmetric top/bottom margin, so the JEV block isn't given extra headroom
    # the GPT block doesn't get.
    ax.set_ylim(row_ys[-1] + margin, row_ys[0] - margin)
    separator(ax, sep_y)

    for block_index, (key, sel_name, kind) in enumerate(selectors):
        values = rows[block_index]
        block_rows = row_ys[block_index * 2 : block_index * 2 + 2]
        block_mid = sum(block_rows) / 2.0

        group_label(ax, block_mid, sel_name)
        ax.text(-0.245, block_mid, f"({kind})",
                transform=ax.get_yaxis_transform(), rotation=90, ha="center",
                va="center", fontsize=8.5, color="#7a7a72")

        for oi, (flag, name, marker, face, edge) in enumerate(outcomes):
            y = block_rows[oi]
            points = values[flag]
            ax.text(-0.012, y, f"{name}\n(n={len(points)})",
                    transform=ax.get_yaxis_transform(), ha="right", va="center",
                    fontsize=10, color=INK, linespacing=1.3)
            if not points:
                continue
            jitter = [(i % 5 - 2) * 0.10 for i in range(len(points))]
            ax.scatter(points, [y + j for j in jitter], marker=marker, s=34,
                       facecolor=face, edgecolor=edge, linewidth=0.9, zorder=3)
            mean = statistics.fmean(points)
            ax.plot([mean, mean], [y - 0.30, y + 0.30], color=INK,
                    linewidth=1.6, zorder=4)
            ax.text(mean, y - 0.42, f"{mean:.2f}", ha="center", va="center",
                    fontsize=9.5, color=INK, zorder=5)

    handles = [
        Line2D([], [], marker=marker, linestyle="none", markersize=7,
               markerfacecolor=face, markeredgecolor=edge, markeredgewidth=0.9,
               label=name)
        for _, name, marker, face, edge in outcomes
    ]
    handles.append(Line2D([], [], color=INK, linewidth=1.6, label="Group mean"))
    legend = ax.legend(
        handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.015),
        ncol=3, frameon=True, fontsize=10.0, handlelength=1.5, borderpad=0.42,
        columnspacing=1.2, handletextpad=0.45,
    )
    legend.get_frame().set_edgecolor(INK)
    legend.get_frame().set_linewidth(FRAME_W)
    legend.get_frame().set_boxstyle("Square", pad=0.34)

    # The two selectors' numbers are not the same kind of measurement (see
    # gpt_confidence.py): JEV's distribution comes from the model, GPT's is
    # written out by the model as text. Both use the same formula, but the
    # caption keeps the reader from reading them as equivalent.
    caption(fig, 0.60, 0.04,
            "Selector confidence, split by agreement — JEV: native, GPT: self-reported")
    fig.savefig(out, dpi=dpi, facecolor="#ffffff")
    plt.close(fig)
    print(f"  {out}")


# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Render the experiment figures.")
    parser.add_argument("--out-dir", default=str(ASSETS))
    parser.add_argument("--dpi", type=int, default=DPI)
    args = parser.parse_args()

    use_paper_style()
    data = {arm: load_runs(newest_batch(arm)) for arm in ARMS}
    shadow_steps = load_steps(load_runs(newest_batch("shadow")))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print("Figures written:")
    figure_runtime(data, out_dir / "figure1_runtime.png", args.dpi)
    figure_agreement(shadow_steps, out_dir / "figure2_agreement.png", args.dpi)
    figure_confidence(shadow_steps, out_dir / "figure3_confidence.png", args.dpi)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
