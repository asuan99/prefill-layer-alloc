#!/usr/bin/env python3
"""Render the prefill-layer-alloc figure set.

    python reports/figures/make_figures.py            # PDF (light) + PNG light & dark
    python reports/figures/make_figures.py --only 6   # one figure

Static text is English on purpose: no CJK font is installed in the venv, and the
figures are meant to be submission-ready. The Korean narrative lives in the HTML
dashboard, which renders with system fonts.

Grades travel with the numbers. A figure never shows a value whose evidence level
is not printed beside it.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

import canon as K
import style as S

HERE = Path(__file__).resolve().parent
RAW = json.loads((HERE / "raw_extracted.json").read_text())
FIGS: dict[int, tuple[str, callable]] = {}


def figure(n: int, slug: str):
    def deco(fn):
        FIGS[n] = (slug, fn)
        return fn
    return deco


def _wrap(text: str, width: int) -> str:
    import textwrap
    return "\n".join(textwrap.wrap(text, width))


def note_block(fig, text: str, c: dict, y=0.055, width=150, size=7.4):
    fig.text(0.008, y, _wrap(text, width), fontsize=size, color=c["secondary"],
             ha="left", va="bottom", linespacing=1.5)


# ══════════════════════════════════════════════════════════════════════════════
@figure(1, "arc_killchain")
def fig_arc(c):
    """How the founding hypothesis died: three axes, twelve stages.

    Twelve stages each carrying a title, a decisive number and a reason is text,
    not magnitude -- so this is a structured list, not a chart. The only encoded
    channels are the verdict marker and which of three axis columns it sits in.
    """
    fig, ax = plt.subplots(figsize=(13.4, 8.2))
    n = len(K.ARC)
    # three spine columns, one per research axis
    axis_x = {"layer-type": 0.072, "time": 0.103, "space": 0.134}
    axis_head = {"layer-type": "layer\ntype", "time": "time", "space": "space"}
    top = n - 0.4

    for name, x in axis_x.items():
        ax.plot([x, x], [-0.6, top - 0.15], color=c["grid"], lw=1.2, zorder=0)
        ax.text(x, top, axis_head[name], fontsize=7.2, color=c["muted"],
                ha="center", va="bottom", linespacing=1.3)

    for i, s in enumerate(K.ARC):
        y = n - 1 - i
        col = c[S.VERDICT_COLOR[s["verdict"]]]
        ax.plot(axis_x[s["axis"]], y, S.VERDICT_MARKER[s["verdict"]], color=col,
                markersize=11, zorder=3, markeredgecolor=c["surface"],
                markeredgewidth=1.7)
        ax.text(0.0, y, s["stage"], fontsize=9, color=c["ink"], weight="bold",
                va="center")
        ax.text(0.170, y + 0.20, s["title"], fontsize=8.8, color=c["ink"], va="center")
        ax.text(0.170, y - 0.13, s["number"], fontsize=7.8, color=col, va="center",
                style="italic", weight="bold")
        ax.text(0.170, y - 0.38, s["why"], fontsize=7.0, color=c["muted"], va="center")
        ax.text(1.0, y, s["grade"], fontsize=7.2, color=c["secondary"], ha="right",
                va="center")
        if i:
            ax.axhline(y + 0.52, color=c["grid"], lw=0.6, zorder=0, alpha=0.55)

    ax.set_xlim(-0.012, 1.02)
    ax.set_ylim(-0.85, top + 0.85)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)

    handles = [Line2D([], [], marker=S.VERDICT_MARKER[v], color="none",
                      markerfacecolor=c[S.VERDICT_COLOR[v]], markersize=9, label=lbl)
               for v, lbl in [("dead", "refuted / dead"), ("reopened", "reopened"),
                              ("reversed", "reversal"), ("open", "open frontier")]]
    ax.legend(handles=handles, loc="upper right", ncol=4, handletextpad=0.4,
              columnspacing=1.5, bbox_to_anchor=(1.0, 1.075))

    S.title(ax, "How the founding hypothesis died — three axes, twelve stages",
            "Founding claim: 'splitting SM by layer type beats a type-agnostic split.' "
            "Alive in simulation; refuted in every serving form.", c, pad=26)
    note_block(fig, "Every verdict on the layer-type and time axes is a real-engine "
                    "serving measurement, not a micro-benchmark — that distinction is what "
                    "the arc taught, after micro-measurements overturned a conclusion four "
                    "separate times. The space axis is the live frontier: C2 established "
                    "that a decode-SM lever exists at the operating point, S0-ax found the "
                    "measurement layer beneath it was self-contradictory, and S2s resolved "
                    "that contradiction as label dilution. No performance verdict on the "
                    "space axis yet.",
               c, y=0.050, width=178)
    S.footer(fig, f"source: {K.SRC['arc']} · {K.SRC['consensus']} · {K.SRC['status']}"
                  f"   |   substrate: {K.SUBSTRATE}", c, width=230)
    fig.subplots_adjust(left=0.012, right=0.992, top=0.885, bottom=0.145)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
@figure(2, "pd_separation")
def fig_pd(c):
    """P1: prefill/decode separation beats fused -- at the NON-operating point."""
    ser = RAW["p1_4model_goodput"]["series"]
    models = [m for m in ("Nemotron-H-8B", "Granite-4.0-h-micro", "Falcon-H1-3B")
              if "fused" in ser.get(m, {}) and "agnostic" in ser.get(m, {})]
    fig, axes = plt.subplots(1, len(models), figsize=(11.6, 4.3), sharey=True)
    for ax, model in zip(np.atleast_1d(axes), models):
        for k, (pol, lbl) in enumerate([("fused", "fused"), ("agnostic", "PD-mux agnostic")]):
            rows = ser[model][pol]["rows"]
            x = [r["rate"] for r in rows]
            y = [r["goodput"] for r in rows]
            ax.plot(x, y, "-o", color=c["series"][k], label=lbl, zorder=3,
                    markeredgecolor=c["surface"], markeredgewidth=1.2)
            ax.annotate(f"{y[-1]:.2f}", (x[-1], y[-1]), xytext=(5, 0),
                        textcoords="offset points", fontsize=8,
                        color=c["series"][k], va="center", weight="bold")
        job_a = ser[model]["agnostic"]["job"]
        job_f = ser[model]["fused"]["job"]
        ax.set_title(model, color=c["ink"], loc="left", fontsize=10, pad=16)
        ax.annotate(f"jobs {job_f} / {job_a}", xy=(0, 1), xycoords="axes fraction",
                    xytext=(0, 4), textcoords="offset points", fontsize=7.2,
                    color=c["muted"], ha="left", va="bottom")
        ax.set_xlabel("request rate (req/s)")
        ax.set_xticks([1, 2, 3, 4, 6])
        ax.set_xlim(0.6, 7.3)
    np.atleast_1d(axes)[0].set_ylabel("goodput @ SLO  (req/s)")
    np.atleast_1d(axes)[0].legend(loc="upper left")

    fig.suptitle("P1  —  prefill/decode separation beats fused execution",
                 x=0.008, ha="left", color=c["ink"], fontsize=12.5, weight="semibold")
    fig.text(0.008, 0.915,
             "SCOPE REDUCED BY AUDIT: every cell is --disable-cuda-graph, i.e. the "
             "NON-operating point, n=1 per cell, and rate 1 is a tie (arrival ceiling).",
             fontsize=8.2, color=c["critical"], ha="left", va="top")
    note_block(fig,
               "Counter-evidence on record: fused dies of TPOT crossing the 60 ms SLO "
               "(Granite rate 4: 61.21 ms) — and cudagraph removes exactly that wall "
               "(plain TPOT 62.70 -> 13.51 ms; rate 4: 82.41 -> 54.04 ms, which PASSES the "
               "60 ms SLO). The cudagraph-ON control has never been run for any model, so "
               "this result may shrink or vanish at the operating point. It is the front "
               "half of the project's top-line sentence, and its top verification priority. "
               "Zamba2's fused arm is absent from this clean-async harness.",
               c, y=0.035, width=168)
    S.footer(fig, "workload: synthetic random-ids in2000/out96, range-ratio 1.0, 120 prompts "
                  "· goodput @ (TTFT<=3s AND TPOT<=60ms) · raw: workspace/engine-port/triage/",
             c)
    fig.subplots_adjust(left=0.062, right=0.975, top=0.80, bottom=0.30, wspace=0.10)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
@figure(3, "layer_aware_death")
def fig_la(c):
    """The founding hypothesis, killed twice: TPOT, then goodput collapse."""
    fig = plt.figure(figsize=(12.6, 4.9))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.55], wspace=0.20)

    # -- A: the decisive TPOT numbers
    ax = fig.add_subplot(gs[0, 0])
    d = K.LAYER_AWARE_TPOT
    labels = [b[0] for b in d["bars"]]
    vals = [b[1] for b in d["bars"]]
    kinds = ["good", "warning", "critical", "serious"]
    bars = ax.bar(range(len(vals)), vals, width=0.62,
                  color=[c[k] for k in kinds], zorder=3,
                  edgecolor=c["surface"], linewidth=2.0)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 3, f"{v:.0f}", ha="center",
                va="bottom", fontsize=10, color=c["ink"], weight="bold")
    ax.axhline(42, color=c["axis"], lw=1.0, ls=(0, (4, 3)), zorder=2)
    ax.annotate("agnostic baseline", xy=(3.42, 42), fontsize=7.4, color=c["muted"],
                va="bottom", ha="right")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=7.2, linespacing=1.4)
    ax.set_ylabel(d["unit"])
    ax.set_ylim(0, 150)
    S.title(ax, "A. Coordination was not the problem",
            "E2 · Zamba2-2.7B · all four cells at decode = 54 SM · serving, "
            "no-cudagraph substrate", c, subwidth=52)

    # -- B: goodput collapse under load, per model
    ax2 = fig.add_subplot(gs[0, 1])
    ser = RAW["p1_4model_goodput"]["series"]
    pairs = [("Zamba2-2.7B", 0), ("Granite-4.0-h-micro", 1), ("Nemotron-H-8B", 2)]
    ends = []
    for model, k in pairs:
        if "layer_aware" not in ser.get(model, {}):
            continue
        agn = {r["rate"]: r["goodput"] for r in ser[model]["agnostic"]["rows"]}
        la = {r["rate"]: r["goodput"] for r in ser[model]["layer_aware"]["rows"]}
        rates = sorted(set(agn) & set(la))
        rel = [(la[r] / agn[r] if agn[r] > 0 else np.nan) for r in rates]
        ax2.plot(rates, rel, "-o", color=c["series"][k], zorder=3,
                 markeredgecolor=c["surface"], markeredgewidth=1.2)
        ends.append((rel[-1], rates[-1], model.split("-4.0")[0].split("-8B")[0], k))
    # de-collide the direct labels: two arms both land exactly on 0.0
    ends.sort(key=lambda e: e[0])
    placed = -1e9
    for yv, xv, name, k in ends:
        placed = max(yv, placed + 0.095)
        ax2.annotate(name, (xv, placed), xytext=(7, 0), textcoords="offset points",
                     fontsize=8, color=c["series"][k], va="center", weight="bold")
    ax2.axhline(1.0, color=c["axis"], lw=1.2, zorder=2)
    ax2.annotate("parity with agnostic", xy=(1.05, 1.0), xytext=(0, 5),
                 textcoords="offset points", fontsize=7.4, color=c["muted"], va="bottom")
    ax2.set_xlabel("request rate (req/s)")
    ax2.set_ylabel("layer-aware goodput / agnostic goodput")
    ax2.set_xticks([1, 2, 3, 4, 6])
    ax2.set_xlim(0.8, 7.6)
    ax2.set_ylim(-0.14, 1.26)
    S.title(ax2, "B. Equal at low load, collapsing under it",
            "Zamba2 hits goodput 0 from rate 3 — the pre-registered prediction "
            "was the exact opposite", c, subwidth=74)

    note_block(fig,
               "S1 argued the first layer-aware result was a strawman: decode sat in an "
               "independent green context, uncoordinated with pdmux prefill, and the same "
               "54 SM cost 121 ms uncoordinated but 42 ms coordinated. That reopened the "
               "hypothesis — what had been refuted was the measurement, not the idea. "
               "S2 then built the coordinated per-type version for real (event_loop_pdmux_coord, "
               "correctness gate passed) and it landed at 124 ms. COORD_OPT recovers 47% "
               "(124 -> 85 ms), so the magnitude was half substrate; the SIGN is robust, and "
               "the residual — monolithic prefill overlapping in a single window — is structural.",
               c, y=0.035, width=176)
    S.footer(fig, f"source: {K.SRC['consensus']} §1-3/§1-15 · {K.SRC['arc']} S1/S2 "
                  f"· raw goodput: workspace/engine-port/triage/", c)
    fig.subplots_adjust(left=0.055, right=0.965, top=0.76, bottom=0.30)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
@figure(4, "he0_dynamic_loses")
def fig_he0(c):
    """HE0: dynamic control never beats the best static -- at either SLO."""
    fig = plt.figure(figsize=(12.6, 4.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.45, 1.0], wspace=0.30)

    ax = fig.add_subplot(gs[0, 0])
    d = K.HE0_LENIENT
    rows = d["rows"][::-1]
    ys = np.arange(len(rows))
    for y, (lbl, mean, sd, n, kind) in zip(ys, rows):
        col = c["series"][0] if kind == "static" else c["series"][1]
        if sd is not None:
            ax.errorbar(mean, y, xerr=sd, fmt="none", ecolor=col, elinewidth=2.0,
                        capsize=4, capthick=1.6, zorder=3)
        ax.plot(mean, y, "o", color=col, markersize=9, zorder=4,
                markeredgecolor=c["surface"], markeredgewidth=1.6)
        ax.annotate(f"{mean:.3f}" + (f" ± {sd:.3f}" if sd else "") + f"   n={n}",
                    (mean + (sd or 0), y), xytext=(11, 0), textcoords="offset points",
                    fontsize=8, color=c["secondary"], va="center")
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=8.5)
    ax.set_xlabel(d["unit"])
    ax.set_xlim(2.72, 3.42)
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    best = rows[-1]
    ax.axvline(best[1], color=c["series"][0], lw=1.0, ls=(0, (4, 3)), zorder=2)
    S.title(ax, "A. Lenient SLO (TTFT ≤ 3 s) — the best static wins",
            f"varying trace, rate 3→12, 3 rounds, durations summed · cudagraph ON "
            f"· {d['sigma']}", c)
    ax.legend(handles=[Line2D([], [], marker="o", color="none", markersize=9,
                              markerfacecolor=c["series"][0], label="static split"),
                       Line2D([], [], marker="o", color="none", markersize=9,
                              markerfacecolor=c["series"][1], label="dynamic controller")],
              loc="upper left", bbox_to_anchor=(0.0, 0.62))

    ax2 = fig.add_subplot(gs[0, 1])
    t = K.HE0_TIGHT
    labels = [r[0] for r in t["rows"]]
    vals = [r[1] for r in t["rows"]]
    cols = [c["series"][0] if r[2] == "static" else c["series"][1] for r in t["rows"]]
    bars = ax2.bar(range(len(vals)), vals, width=0.62, color=cols, zorder=3,
                   edgecolor=c["surface"], linewidth=2.0)
    for b, v in zip(bars, vals):
        ax2.text(b.get_x() + b.get_width() / 2, v + 1.4, f"{v:.1f}%", ha="center",
                 va="bottom", fontsize=9.5, color=c["ink"], weight="bold")
    ax2.set_xticks(range(len(labels)))
    ax2.set_xticklabels(labels, fontsize=8.5)
    ax2.set_ylabel(t["unit"])
    ax2.set_ylim(0, 88)
    ax2.annotate("28.9 pp", xy=(1.5, 79), ha="center", fontsize=9,
                 color=c["critical"], weight="bold")
    ax2.annotate("", xy=(0, 76), xytext=(2, 76),
                 arrowprops=dict(arrowstyle="<->", color=c["critical"], lw=1.3))
    S.title(ax2, "B. Tight SLO — the gap widens",
            "chat SLO 300/50 ms, rate 8 · controller RETUNED, not re-scored", c, subwidth=52)

    note_block(fig,
               "§1-16 once read a tight SLO as reviving dynamic control. §1-17 refuted it: "
               "re-scoring post-hoc grades wherever a 3s-tuned controller happened to settle. "
               "Retuning for the tight SLO makes dynamic WORSE, because the controller reacts "
               "to tight TTFT by moving prefill-ward, which starves decode and springs the "
               "entanglement trap. => 'decode-heavy static dominates' is independent of SLO "
               "strictness. A canon discrepancy travels with panel A: research_arc.md lists "
               "d16 = 2.817 where the two synthesis documents list 2.846.",
               c, y=0.035, width=176)
    S.footer(fig, f"source: {K.SRC['neg']} §2.1 · {K.SRC['consensus']} §0/§1-17", c)
    fig.subplots_adjust(left=0.155, right=0.975, top=0.76, bottom=0.30)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
@figure(5, "why_dynamic_loses")
def fig_regime(c):
    """The structural reason: the two regimes' optima never conflict."""
    fig, ax = plt.subplots(figsize=(9.6, 4.9))
    d = K.REGIME_SPLIT
    xs = [0, 1]
    for k, (lbl, lo, lo_sd, hi, hi_sd) in enumerate(d["rows"]):
        col = c["series"][k]
        ax.plot(xs, [lo, hi], "-o", color=col, lw=2.4, markersize=10, zorder=3,
                markeredgecolor=c["surface"], markeredgewidth=1.8)
        ax.annotate(lbl, (1, hi), xytext=(12, 0), textcoords="offset points",
                    fontsize=9, color=col, va="center", weight="bold")
        for x, v, sd in ((0, lo, lo_sd), (1, hi, hi_sd)):
            txt = f"{v:.3f}" + (f" ± {sd:.3f}" if sd else "")
            # keep the value clear of the x tick labels at the HI end
            dy = 12 if (k == 0 or x == 1) else -17
            ax.annotate(txt, (x, v), xytext=(0, dy), textcoords="offset points",
                        fontsize=8.4, color=c["secondary"], ha="center")
    # the spread lives under the axis, where it cannot collide with the marks
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{ph}\npolicy spread {sp:.3f}  ({pc:.1f}%)"
                        for ph, (_, sp, pc) in zip(d["phases"], d["spread"])],
                       fontsize=9.5, linespacing=1.7)
    for tick, (_, _, pc) in zip(ax.get_xticklabels(), d["spread"]):
        tick.set_color(c["critical"] if pc > 10 else c["secondary"])
    ax.set_xlim(-0.32, 1.42)
    ax.set_ylim(2.62, 4.10)
    ax.set_ylabel(d["unit"])
    ax.grid(axis="y")
    S.title(ax, "Why dynamic control has nothing to chase",
            "~95% of the discrimination between policies comes from the overload phase; "
            "the low-load phase is indifferent to the split", c)
    note_block(fig,
               "At low load the two extreme splits are indistinguishable (d16 2.861 vs d44 "
               "2.858). At high load the spread is 43%, and the winner is decode-heavy. "
               "Because over-supplying decode is nearly free at low load, HI's optimum is "
               "also acceptable at LO — so 'always run the HI optimum', i.e. a decode-heavy "
               "STATIC, is best by construction, and a dynamic controller can only pay the "
               "transit. Dynamic control needs the regimes' optima to CONFLICT; in this "
               "workload no such region exists. Caveat added later: the LO metric is "
               "CEILING-CENSORED by the arrival rate (199-200 of 200 served), so LO "
               "indifference is a silent metric, not a proven absence of lever.",
               c, y=0.052, width=140)
    S.footer(fig, f"source: {K.SRC['neg']} §2.3 · {K.SRC['arc']} S10 · "
                  f"ShareGPT varying trace, Zamba2-2.7B, cudagraph ON", c)
    fig.subplots_adjust(left=0.085, right=0.80, top=0.76, bottom=0.38)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
@figure(6, "c2_decode_sm_lever")
def fig_c2(c):
    """C2 -- the strongest surviving positive, with its mandatory riders."""
    fig = plt.figure(figsize=(12.4, 5.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.5, 1.0], wspace=0.26)
    d = K.C2

    ax = fig.add_subplot(gs[0, 0])
    for k, (arm, ys) in enumerate(d["arms"].items()):
        xs = [x for x, y in zip(d["sm"], ys) if y is not None]
        vs = [y for y in ys if y is not None]
        ax.plot(xs, vs, "-o", color=c["series"][k], zorder=3,
                markeredgecolor=c["surface"], markeredgewidth=1.3)
        ax.annotate(arm.split()[0], (xs[-1], vs[-1]), xytext=(7, 0),
                    textcoords="offset points", fontsize=8.6,
                    color=c["series"][k], va="center", weight="bold")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks(d["sm"]); ax.set_xticklabels([str(s) for s in d["sm"]])
    ax.set_yticks([10, 15, 20, 30, 50]); ax.set_yticklabels(["10", "15", "20", "30", "50"])
    ax.set_xlabel("decode SM  (prefill pinned at 16 SM)")
    ax.set_ylabel(d["unit"])
    ax.set_xlim(14, 155)
    ax.grid(axis="both")
    S.no_minor_labels(ax)
    ax.axvspan(44, 155, color=c["band"], zorder=0)
    ax.annotate("local ε 0.09–0.35\nDO NOT apply the\nendpoint ratio here",
                xy=(78, 34), fontsize=7.6, color=c["critical"], ha="center",
                va="center", weight="bold", linespacing=1.4)
    ax.annotate("local ε\n0.77–0.88", xy=(19.6, 12.2), fontsize=7.6,
                color=c["secondary"], ha="center", va="center", linespacing=1.4)
    S.title(ax, "C2 — the decode-SM lever is real at the operating point",
            "partition- and batch-matched per-token intervals, batch=1 · cudagraph ON "
            "· 4 arms alike", c, pad=22, subwidth=74)
    ax.annotate("CONFIRMED (scoped)", xy=(1.0, 1.0), xycoords="axes fraction",
                xytext=(0, 9), textcoords="offset points", ha="right", va="bottom",
                fontsize=8.5, color=c["good"], weight="bold")

    ax2 = fig.add_subplot(gs[0, 1])
    arms = list(d["headline"])
    lo = [d["headline"][a][1] for a in arms]
    hi = [d["headline"][a][0] for a in arms]
    ratio = [d["headline"][a][2] for a in arms]
    ys = np.arange(len(arms))
    for y, a, l, h, r in zip(ys, arms, lo, hi, ratio):
        ax2.plot([l, h], [y, y], "-", color=c["axis"], lw=2.0, zorder=2)
        ax2.plot(l, y, "o", color=c["series"][2], markersize=9, zorder=3,
                 markeredgecolor=c["surface"], markeredgewidth=1.5)
        ax2.plot(h, y, "o", color=c["series"][3], markersize=9, zorder=3,
                 markeredgecolor=c["surface"], markeredgewidth=1.5)
        ax2.annotate(f"{r:.2f}×", ((l + h) / 2, y), xytext=(0, 8),
                     textcoords="offset points", ha="center", fontsize=9,
                     color=c["ink"], weight="bold")
    ax2.set_yticks(ys); ax2.set_yticklabels(arms, fontsize=9)
    ax2.set_xlabel("decode ITL p50 (ms)")
    ax2.set_xlim(0, 100)
    ax2.grid(axis="x"); ax2.grid(axis="y", visible=False)
    ax2.legend(handles=[Line2D([], [], marker="o", color="none", markersize=9,
                               markerfacecolor=c["series"][3], label="decode 16 SM"),
                        Line2D([], [], marker="o", color="none", markersize=9,
                               markerfacecolor=c["series"][2], label="decode 92 SM")],
               loc="lower right")
    S.title(ax2, "Endpoint ratio, clean prefill-fixed slice",
            "SM16 → SM92 · high-n batch slice · family changes the ABSOLUTE cost, "
            "not the sensitivity", c, pad=22, subwidth=52)

    note_block(fig, "CITATION RULE — " + d["citation_rule"], c, y=0.062, width=176)
    S.footer(fig, f"source: {K.SRC['c2']} · scope (must travel with the number): {d['scope']}",
             c, width=190)
    fig.subplots_adjust(left=0.075, right=0.965, top=0.75, bottom=0.31)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
@figure(7, "prefill_axis_mirror")
def fig_prefill(c):
    """The mirror campaign, and the inference it does NOT support."""
    fig = plt.figure(figsize=(12.4, 5.0))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1.0], wspace=0.26)
    d = K.PREFILL_AXIS

    ax = fig.add_subplot(gs[0, 0])
    for k, (arm, ys) in enumerate(d["arms"].items()):
        ax.plot(d["sm"], ys, "-o", color=c["series"][k], zorder=3,
                markeredgecolor=c["surface"], markeredgewidth=1.3)
        ax.annotate(arm.split()[0], (d["sm"][-1], ys[-1]), xytext=(7, 0),
                    textcoords="offset points", fontsize=8.6,
                    color=c["series"][k], va="center", weight="bold")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks(d["sm"]); ax.set_xticklabels([str(s) for s in d["sm"]])
    ax.set_yticks([100, 200, 400, 800]); ax.set_yticklabels(["100", "200", "400", "800"])
    ax.set_xlabel("prefill SM  (decode pinned at 16 SM)")
    ax.set_ylabel(d["unit"])
    ax.set_xlim(14, 145)
    ax.grid(axis="both")
    S.no_minor_labels(ax)
    S.title(ax, "Prefill axis — the mirror of C2",
            "TTFT~L regression slope per cell, n=4 reps · slope ratio 4.74–5.16×, "
            "ε 0.89–0.94, arms alike", c, pad=22, subwidth=70)
    ax.annotate("UNAUDITED — canon citation banned", xy=(1.0, 1.0),
                xycoords="axes fraction", xytext=(0, 9), textcoords="offset points",
                ha="right", va="bottom", fontsize=8.5, color=c["critical"],
                weight="bold")

    ax2 = fig.add_subplot(gs[0, 1])
    groups = ["prefill axis\n(this campaign)", "decode axis\n(C2)"]
    lo_hi = [(0.71, 1.05), (0.09, 0.88)]
    for i, (lo, hi) in enumerate(lo_hi):
        col = c["series"][0] if i == 0 else c["series"][1]
        ax2.barh(i, hi - lo, left=lo, height=0.36, color=col, zorder=3,
                 edgecolor=c["surface"], linewidth=2.0)
        ax2.annotate(f"{lo:.2f} – {hi:.2f}", (hi, i), xytext=(8, 0),
                     textcoords="offset points", fontsize=9, va="center",
                     color=c["ink"], weight="bold")
        ax2.annotate(f"{hi/lo:.1f}× spread", ((lo + hi) / 2, i), xytext=(0, -19),
                     textcoords="offset points", fontsize=8, ha="center",
                     color=c["secondary"])
    ax2.set_yticks([0, 1]); ax2.set_yticklabels(groups, fontsize=9)
    ax2.set_xlabel("local elasticity ε across the SM range")
    ax2.set_xlim(0, 1.35)
    ax2.set_ylim(-0.55, 1.55)
    ax2.grid(axis="x"); ax2.grid(axis="y", visible=False)
    S.title(ax2, "The asymmetry that matters",
            "prefill ε is near-constant; decode ε varies 9× and saturates above D≈44",
            c, pad=22)

    note_block(fig, "REFUTED INFERENCE (kept because it is worth not repeating) — " + d["refuted"],
               c, y=0.062, width=176)
    S.footer(fig, f"source: {K.SRC['pf']} · jobs 865973 / 865974 · "
                  f"gates: M8 4/4 PASS, other arms partial · grade: {d['grade']}", c,
             width=190)
    fig.subplots_adjust(left=0.075, right=0.955, top=0.75, bottom=0.33)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
@figure(8, "diffb_retraction")
def fig_diffb(c):
    """The retraction that rewrote the mechanism story.

    The three series are NOT three measurements of the same quantity: they differ
    along two orthogonal axes applied in order (clean the SAMPLE, then change the
    UNIT). Panels B and C exist because conflating those two is exactly what made
    'the lever opens at short L' look confirmed.
    """
    fig = plt.figure(figsize=(13.4, 9.0))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.5, 1.0],
                          height_ratios=[1.0, 0.86], wspace=0.26, hspace=0.52)
    d = K.DIFF_B

    ax = fig.add_subplot(gs[0, 0])
    styles = [("as reported (kernel unit)", 3, (0, (5, 3)), 9),
              ("steady, defects removed", 0, "-", -13),
              ("policy unit (R_policy)", 2, "-", 9)]
    for name, slot, ls, dy in styles:
        ys = d["series"][name]
        xs = [x for x, y in zip(d["L"], ys) if y is not None]
        vs = [y for y in ys if y is not None]
        ax.plot(xs, vs, linestyle=ls, color=c["series"][slot], marker="o", zorder=3,
                markeredgecolor=c["surface"], markeredgewidth=1.3, label=name)
        # direct-label at the RIGHT end, where the three series are far apart in y
        ax.annotate(name.split(" (")[0].split(", ")[0], (xs[-1], vs[-1]),
                    xytext=(6, dy), textcoords="offset points",
                    fontsize=8.2, color=c["series"][slot], ha="left", weight="bold")
    ax.axhline(1.0, color=c["axis"], lw=1.2, zorder=2)
    ax.annotate("Diff B = 1  →  nothing to reallocate", xy=(9000, 0.925),
                fontsize=7.6, color=c["muted"], ha="center")
    ax.set_xscale("log")
    ax.set_xticks(d["L"]); ax.set_xticklabels([str(x) for x in d["L"]], fontsize=8)
    S.no_minor_labels(ax, "x")
    ax.set_xlabel("prefill sequence length L (tokens)")
    ax.set_ylabel(d["unit"])
    ax.set_ylim(0.88, 1.50)
    ax.set_xlim(200, 200000)
    ax.grid(axis="both")
    ax.axvspan(215, 640, color=c["band"], zorder=0)
    ax.annotate("the claimed\nshort-L lever", xy=(370, 1.44), fontsize=7.8,
                color=c["critical"], ha="center", va="center", weight="bold",
                linespacing=1.4)
    S.title(ax, "A. 'The lever opens at short L' — RETRACTED",
            "micro, no-cudagraph, n_indep = 1 per cell · Zamba2-2.7B prefill "
            "instrumentation", c, subwidth=78)

    # ── B. the two transformations, spelled out at the cell that carried the claim
    axb = fig.add_subplot(gs[1, 0])
    steps = K.DIFF_B_STEPS
    vals = [("as reported\n(kernel)", d["series"]["as reported (kernel unit)"][0], 3),
            ("steady\n(kernel)", d["series"]["steady, defects removed"][0], 0),
            ("policy unit\n(R_policy)", d["series"]["policy unit (R_policy)"][0], 2)]
    ys = [2, 1, 0]
    lo, hi = d["band_256"]
    axb.barh(1, hi - lo, left=lo, height=0.62, color=c["band"], zorder=1)
    for y, (lbl, v, slot) in zip(ys, vals):
        axb.barh(y, v - 1.0, left=1.0, height=0.44, color=c["series"][slot], zorder=3,
                 edgecolor=c["surface"], linewidth=2.0)
        axb.annotate(f"{v:.3f}", (v, y), xytext=(8, 0), textcoords="offset points",
                     va="center", fontsize=9.5, color=c["ink"], weight="bold")
        axb.annotate(lbl, (1.0, y), xytext=(-10, 0), textcoords="offset points",
                     va="center", ha="right", fontsize=8.4, color=c["secondary"],
                     linespacing=1.35)
    axb.annotate(f"steady band\n[{lo:.2f}, {hi:.2f}]  — cite the band, never a point",
                 (hi, 1), xytext=(8, -20), textcoords="offset points", va="top",
                 fontsize=7.4, color=c["muted"], linespacing=1.35)
    for y0, y1, txt, delta in ((2, 1, "① SAMPLE cleaned\n     drop the cold-start block", -0.084),
                               (1, 0, "② UNIT changed\n     kernel → policy window", -0.304)):
        axb.annotate("", xy=(1.0, y1 + 0.30), xytext=(1.0, y0 - 0.24),
                     arrowprops=dict(arrowstyle="->", color=c["critical"], lw=1.4))
        axb.annotate(f"{txt}   ({delta:+.3f})", (1.012, (y0 + y1) / 2),
                     fontsize=7.8, color=c["critical"], va="center",
                     weight="bold", linespacing=1.5)
    axb.axvline(1.0, color=c["axis"], lw=1.3, zorder=2)
    axb.set_yticks([]); axb.set_xlim(0.98, 1.50); axb.set_ylim(-0.55, 2.55)
    axb.set_xlabel("Diff B at L = 256")
    axb.grid(axis="x"); axb.grid(axis="y", visible=False)
    S.title(axb, "B. The three series differ along TWO axes, not one",
            "① is a defect correction — same quantity, cleaner sample.  "
            "② is a unit change — same sample, the aggregation a policy can act on.",
            c, subwidth=92)

    # ── C. the mechanism behind ①: the cold block does not inflate both types equally
    axc = fig.add_subplot(gs[1, 1])
    fb = K.FIRST_BLOCK_INFLATION
    idx = np.arange(len(fb["L"]))
    axc.bar(idx - 0.19, fb["attn"], width=0.36, color=c["series"][0], zorder=3,
            edgecolor=c["surface"], linewidth=2.0, label="attention (numerator)")
    axc.bar(idx + 0.19, fb["mamba"], width=0.36, color=c["series"][1], zorder=3,
            edgecolor=c["surface"], linewidth=2.0, label="mamba (denominator)")
    for i, v in zip(idx - 0.19, fb["attn"]):
        axc.text(i, v + 0.10, f"{v:.2f}", ha="center", fontsize=7.6, color=c["secondary"])
    for i, v in zip(idx + 0.19, fb["mamba"]):
        axc.text(i, v + 0.10, f"{v:.2f}", ha="center", fontsize=7.6, color=c["ink"],
                 weight="bold")
    axc.axhline(1.0, color=c["axis"], lw=1.2, zorder=2)
    axc.set_xticks(idx); axc.set_xticklabels([f"L={x}" for x in fb["L"]], fontsize=8.5)
    axc.set_ylabel("first block ÷ steady value   (1.0 = no inflation)")
    axc.set_ylim(0, 6.4)
    axc.legend(loc="upper right", fontsize=7.6)
    S.title(axc, "C. Why ① moves the ratio at all",
            "the cold block inflates BOTH types — but mamba far more, and mamba is "
            "the denominator", c, subwidth=54)

    ax2 = fig.add_subplot(gs[0, 1])
    a = K.DIFF_A
    xs = np.array([0, 1])
    ax2.bar(xs - 0.19, [a["old"]["attn"], a["old"]["mamba"]], width=0.36,
            color=c["muted"], zorder=3, edgecolor=c["surface"], linewidth=2.0,
            label=a["old"]["label"])
    ax2.bar(xs + 0.19, [a["new"]["attn"], a["new"]["mamba"]], width=0.36,
            yerr=[a["new"]["attn_se"], a["new"]["mamba_se"]],
            error_kw=dict(ecolor=c["ink"], elinewidth=1.2, capsize=3),
            color=c["series"][0], zorder=3, edgecolor=c["surface"], linewidth=2.0,
            label=a["new"]["label"])
    for x, v in zip(xs - 0.19, [a["old"]["attn"], a["old"]["mamba"]]):
        ax2.text(x, v + 0.04, f"{v:.2f}", ha="center", fontsize=8.6, color=c["secondary"])
    for x, v in zip(xs + 0.19, [a["new"]["attn"], a["new"]["mamba"]]):
        ax2.text(x, v + 0.09, f"{v:.3f}", ha="center", fontsize=8.6,
                 color=c["ink"], weight="bold")
    ax2.set_xticks(xs); ax2.set_xticklabels(["attention", "mamba / SSD"], fontsize=9.5)
    ax2.set_ylabel("prefill cost scaling exponent  (time ~ L^k)")
    ax2.set_ylim(0, 2.85)
    ax2.legend(loc="upper center", fontsize=7.6, bbox_to_anchor=(0.5, 1.005))
    S.title(ax2, "D. Diff A — exponents corrected, verdict unchanged",
            "the cost ratio is real and spans 21× — but it was never the quantity "
            "the hypothesis needed", c, pad=22, subwidth=52)

    note_block(fig,
               "① SAMPLE — " + K.DIFF_B_STEPS[0]["why"] + " " +
               K.DIFF_B_STEPS[0]["asymmetry"] + " NON-CIRCULAR PROOF: " +
               K.DIFF_B_STEPS[0]["proof"] + "  ||  ② UNIT — " +
               K.DIFF_B_STEPS[1]["why"] + " " + K.DIFF_B_STEPS[1]["proof"] +
               "  ||  WHICH ONE ANSWERS THE HYPOTHESIS: the policy unit. The founding "
               "claim asked for a lever a RUNTIME POLICY could pull, and a policy "
               "reallocates SM for a whole window, not for one kernel.",
               c, y=0.030, width=210, size=7.0)
    S.footer(fig, f"source: {K.SRC['pos']} §1.2/§1.3 · {K.SRC['neg']} §1.4 · "
                  f"the serving-grade verdict does not rest on this figure", c,
             width=215)
    fig.subplots_adjust(left=0.085, right=0.972, top=0.905, bottom=0.225)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
@figure(9, "metric_cliff")
def fig_cliff(c):
    """Why the benchmark itself had to be retired."""
    fig = plt.figure(figsize=(12.2, 4.9))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.25], wspace=0.28)
    d = K.METRIC_CLIFF

    ax = fig.add_subplot(gs[0, 0])
    names = [p[0] for p in d["perturbation"]]
    pct = [abs(p[2] - p[1]) / p[1] * 100 for p in d["perturbation"]]
    cols = [c["series"][0], c["series"][0], c["critical"]]
    bars = ax.barh(range(len(names))[::-1], pct, height=0.5, color=cols, zorder=3,
                   edgecolor=c["surface"], linewidth=2.0)
    for b, p, (nm, a, bv, lbl) in zip(bars, pct, d["perturbation"]):
        ax.annotate(f"{a:g} → {bv:g}    ({lbl})", (p, b.get_y() + b.get_height() / 2),
                    xytext=(8, 0), textcoords="offset points", fontsize=8.6,
                    va="center", color=c["ink"], weight="bold")
    ax.set_yticks(range(len(names))[::-1]); ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("change between two runs of the SAME config (%)")
    ax.set_xlim(0, 78)
    ax.grid(axis="x"); ax.grid(axis="y", visible=False)
    S.title(ax, "A. A 3% perturbation becomes a 2× signal",
            "identical config, identical prompt fingerprint (e0b77fd12329)", c, subwidth=56)

    ax2 = fig.add_subplot(gs[0, 1])
    for k, (rate, counts) in enumerate(d["stability"].items()):
        total = d["totals"][rate]
        frac = [v / total * 100 for v in counts]
        jitter = np.linspace(-0.13, 0.13, len(frac))
        stable = (max(frac) - min(frac)) < 5
        col = c["good"] if stable else c["critical"]
        ax2.plot([k] * len(frac) + jitter, frac, "o", color=col, markersize=10,
                 zorder=3, markeredgecolor=c["surface"], markeredgewidth=1.5)
        ax2.annotate("robust" if stable else "UNSTABLE", (k, 113), ha="center",
                     fontsize=8.4, color=col, weight="bold")
        # collapse values that would overprint into one range label
        groups: list[list[int]] = []
        for v in sorted(set(counts)):
            if groups and (v - groups[-1][-1]) / total * 100 < 3:
                groups[-1].append(v)
            else:
                groups.append([v])
        for g in groups:
            lbl = f"{g[0]}/{total}" if len(g) == 1 else f"{g[0]}–{g[-1]}/{total}"
            ax2.annotate(lbl, (k + 0.22, np.mean(g) / total * 100), fontsize=7.6,
                         color=c["secondary"], va="center")
    ax2.set_xticks(range(3))
    ax2.set_xticklabels([f"{r}\n(n={len(v)} reps)" for r, v in d["stability"].items()],
                        fontsize=9)
    ax2.set_ylabel("requests meeting the SLO (% of offered)")
    ax2.set_ylim(22, 124)
    ax2.set_xlim(-0.5, 2.75)
    S.title(ax2, "B. Only the boundary regime is unstable",
            "rate 8 sits exactly at TTFT ≈ SLO · offered 8/s vs capacity 6.3/s", c, subwidth=64)

    note_block(fig,
               "The overload queue's TTFT plateau moves 1.5 s → 3.7 s under a 3% throughput "
               "deficit and crosses the 3 s threshold: 400/400 collapses to 206/400. Goodput "
               "was being evaluated at the steepest point of the CDF. Consequences: the GPU "
               "clock-throttling hypothesis was dropped as unnecessary; resource isolation "
               "was unnecessary; and " + d["static_scatter"] + " — so stationary "
               "ShareGPT r8 was RETIRED as a policy benchmark. Policy comparison now uses a "
               "varying trace only. This is methodology gates #2, #3 and #6.",
               c, y=0.035, width=176)
    S.footer(fig, f"source: {K.SRC['arc']} S10 · {K.SRC['consensus']} §2 · "
                  f"reports/bench_noise_root_cause.md", c)
    fig.subplots_adjust(left=0.115, right=0.97, top=0.76, bottom=0.31)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
@figure(10, "sticky_partition")
def fig_sticky(c):
    """2026-08-04: the label was never the hardware state."""
    fig = plt.figure(figsize=(12.4, 4.9))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 0.8, 1.2], wspace=0.34)
    d = K.STICKY

    ax = fig.add_subplot(gs[0, 0])
    for i, (run, res) in enumerate(d["residency"]):
        left = 0.0
        for k, key in enumerate(("D16", "D108")):
            v = res[key]
            ax.barh(i, v, left=left, height=0.46,
                    color=c["series"][0] if key == "D16" else c["muted"], zorder=3,
                    edgecolor=c["surface"], linewidth=2.0)
            if v > 8:
                ax.annotate(f"{key}\n{v:.1f}%", (left + v / 2, i), ha="center",
                            va="center", fontsize=8.4, color=c["surface"],
                            weight="bold", linespacing=1.3)
            left += v
    ax.annotate("D16  3.8%", (3.8, 0), xytext=(4, -22), textcoords="offset points",
                fontsize=8, color=c["series"][0], weight="bold")
    ax.set_yticks([0, 1]); ax.set_yticklabels([r[0] for r in d["residency"]], fontsize=8.8)
    ax.set_xlabel("share of decode-busy time (%)")
    ax.set_xlim(0, 100)
    ax.grid(axis="x"); ax.grid(axis="y", visible=False)
    S.title(ax, "A. Where the 'd16' cell actually ran",
            "time-weighted realized partition", c, subwidth=40)

    ax2 = fig.add_subplot(gs[0, 1])
    labels = [r[0] for r in d["ratio"]]
    vals = [r[1] for r in d["ratio"]]
    errs = [r[2] for r in d["ratio"]]
    bars = ax2.bar(range(2), vals, yerr=errs, width=0.55,
                   error_kw=dict(ecolor=c["ink"], elinewidth=1.4, capsize=4),
                   color=[c["muted"], c["series"][0]], zorder=3,
                   edgecolor=c["surface"], linewidth=2.0)
    for b, v, e in zip(bars, vals, errs):
        ax2.text(b.get_x() + b.get_width() / 2, v + 0.10, f"{v:.3f}\n±{e:.3f}",
                 ha="center", va="bottom", fontsize=8.8, color=c["ink"],
                 weight="bold", linespacing=1.3)
    ax2.axhline(1.0, color=c["axis"], lw=1.2, zorder=2)
    ax2.annotate("1.000 = the split label had\nno per-token effect at all",
                 xy=(0.5, 1.72), fontsize=7.4, color=c["muted"], ha="center",
                 va="center", linespacing=1.4)
    ax2.set_xticks([0, 1]); ax2.set_xticklabels(["sticky OFF", "sticky ON"], fontsize=9)
    ax2.set_ylabel(d["ratio_label"], fontsize=8)
    ax2.set_ylim(0, 3.15)
    S.title(ax2, "B. Label → effect", "block-paired, n=8", c)

    ax3 = fig.add_subplot(gs[0, 2])
    for i, (cell, p50, lo, hi, plo, phi, hit) in enumerate(d["itl"]):
        ax3.add_patch(Rectangle((plo, i - 0.20), phi - plo, 0.40,
                                color=c["band"], zorder=1))
        ax3.annotate("pre-registered", ((plo + phi) / 2, i + 0.24), ha="center",
                     fontsize=7.2, color=c["muted"])
        col = c["good"] if hit else c["critical"]
        ax3.errorbar(p50, i, xerr=[[p50 - lo], [hi - p50]], fmt="none", ecolor=col,
                     elinewidth=2.4, capsize=4, capthick=1.6, zorder=3)
        ax3.plot(p50, i, "o", color=col, markersize=10, zorder=4,
                 markeredgecolor=c["surface"], markeredgewidth=1.6)
        ax3.annotate(f"{p50:.2f} ms  " + ("HIT" if hit else "MISS — below"),
                     (p50, i), xytext=(0, -22), textcoords="offset points",
                     ha="center", fontsize=8.4, color=col, weight="bold")
    ax3.axvline(d["c2_anchor"], color=c["series"][1], lw=1.4, ls=(0, (4, 3)), zorder=2)
    ax3.annotate(f"C2 anchor {d['c2_anchor']} ms", (d["c2_anchor"], 1.34),
                 xytext=(-6, 0), textcoords="offset points", fontsize=7.8,
                 color=c["series"][1], ha="right", weight="bold")
    ax3.set_yticks([0, 1])
    ax3.set_yticklabels([r[0].split("  ")[0] for r in d["itl"]], fontsize=9)
    for i, (cell, *_rest) in enumerate(d["itl"]):
        ax3.annotate(cell.split("  ")[1], xy=(0, i), xycoords=("axes fraction", "data"),
                     xytext=(-4, -11), textcoords="offset points", ha="right",
                     va="center", fontsize=7.4, color=c["muted"])
    ax3.set_xlabel("per-token ITL p50 of the SPLIT population (ms)")
    ax3.set_xlim(8, 38)
    ax3.set_ylim(-0.55, 1.55)
    ax3.grid(axis="x"); ax3.grid(axis="y", visible=False)
    S.title(ax3, "C. The pre-registered readout",
            "job 873015 · n=8 blocks · realization 0.9956–1.0000", c, subwidth=50)

    note_block(fig, "VERDICT — " + d["verdict"] + "  ||  SCOPE — " + d["scope"],
               c, y=0.055, width=176)
    S.footer(fig, f"source: {K.SRC['s2']} · pre-registration: "
                  f"results/s8_frontier/PREREG_S2_STICKY_ITL_2026-08-03.md · "
                  f"grade: {d['grade']}", c, width=200)
    fig.subplots_adjust(left=0.105, right=0.975, top=0.77, bottom=0.34)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
@figure(11, "evidence_ledger")
def fig_ledger(c):
    """What is actually established, and what was taken back."""
    fig = plt.figure(figsize=(13.4, 8.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.12], wspace=0.06)

    ax = fig.add_subplot(gs[0, 0])
    items = ([(p["id"], p["text"], p["grade"], p["rider"]) for p in K.POSITIVES]
             + [("—", "", "", "")]
             + [(c_["id"], c_["text"], c_["grade"], c_["rider"]) for c_ in K.CLAIMS])
    # rows are 2.15 units tall so a two-line rider cannot reach either neighbour
    ys = np.array([2.15 * (len(items) - 1 - i) for i in range(len(items))])
    for y, (pid, text, grade, rider) in zip(ys, items):
        if pid == "—":
            ax.axhline(y + 0.35, color=c["grid"], lw=1.0)
            ax.text(0.0, y + 0.52, "PAPER CLAIMS", fontsize=7.6, color=c["muted"],
                    weight="bold")
            continue
        col = c[S.GRADE_COLOR.get(grade, "warning")]
        ax.text(0.0, y + 0.34, pid, fontsize=9, color=c["ink"], weight="bold",
                va="center")
        ax.text(0.052, y + 0.34, text, fontsize=8.2, color=c["ink"], va="center")
        ax.text(0.052, y - 0.42, _wrap(rider, 112), fontsize=6.7, color=c["muted"],
                va="top", linespacing=1.4)
        ax.text(1.0, y + 0.34, grade.upper(), fontsize=7.6, color=col, weight="bold",
                ha="right", va="center")
    ax.set_xlim(-0.02, 1.06); ax.set_ylim(-1.2, ys.max() + 0.9)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    S.title(ax, "What is established — and at what grade",
            "every rider is load-bearing", c)

    ax2 = fig.add_subplot(gs[0, 1])
    ys = np.array([1.55 * (len(K.RETRACTIONS) - 1 - i)
                   for i in range(len(K.RETRACTIONS))])
    kindmap = {"self-retracted": "warning", "retracted": "critical",
               "refuted": "critical", "refuted by real trace": "critical",
               "refuted by §1-17": "critical", "benchmark retired": "serious",
               "unnecessary": "warning", "retired": "critical",
               "abandoned": "serious", "citation-barred": "critical",
               "downgraded": "serious"}
    # entries stay in the order they happened; the ordering is the chronology, so
    # the date column itself is redundant
    for y, (_date, what, kind, why) in zip(ys, K.RETRACTIONS):
        col = c[kindmap.get(kind, "warning")]
        ax2.plot(0.012, y, "o", color=col, markersize=7.5, zorder=3,
                 markeredgecolor=c["surface"], markeredgewidth=1.4)
        ax2.text(0.042, y + 0.20, what, fontsize=7.9, color=c["ink"], va="center")
        ax2.text(0.042, y - 0.14, _wrap(why, 112), fontsize=6.5, color=c["muted"],
                 va="top", linespacing=1.4)
        ax2.text(1.0, y + 0.20, kind, fontsize=6.9, color=col, ha="right",
                 va="center", weight="bold")
    ax2.set_xlim(-0.01, 1.02)
    ax2.set_ylim(-1.2, ys.max() + 0.9)
    ax2.set_xticks([]); ax2.set_yticks([]); ax2.grid(False)
    for sp in ax2.spines.values():
        sp.set_visible(False)
    S.title(ax2, f"The retraction ledger — {len(K.RETRACTIONS)} entries",
            "results that were published and then taken back, with the reason", c)

    note_block(fig,
               "The ledger is the point, not an embarrassment: nine of these were caught by "
               "the project's own adversarial audit rather than by a reviewer, and each one "
               "became a standing methodology gate. Recurring failure modes: using an "
               "IDENTITY as evidence (the pin gate, kv_mamba_occupancy, g); treating a "
               "MICRO measurement as a serving prediction (four separate reversals); "
               "choosing ENDPOINTS after the fact (three recurrences); and comparing two "
               "wrong hypotheses without testing their shared premise.",
               c, y=0.028, width=196)
    S.footer(fig, f"source: {K.SRC['status']} 'retracted hypotheses' · "
                  f"{K.SRC['consensus']} §2/§3 · {K.SRC['neg']} §3", c)
    fig.subplots_adjust(left=0.012, right=0.995, top=0.87, bottom=0.155)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
@figure(12, "controller_anatomy")
def fig_ctrl(c):
    """What the three 'dynamic' arms actually are, and where each one sat."""
    fig = plt.figure(figsize=(13.4, 8.6))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.5, 1.0], hspace=0.36)

    # ── A. the three decision rules, side by side ─────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    xs = [0.0, 0.345, 0.690]
    W = 0.295
    TOP, FLOOR = 0.945, 0.035

    def column(x, ctl, unit: float, draw: bool) -> float:
        """Lay the blocks out sequentially, so a longer RULE can never collide with
        WHAT BROKE. Advances are in LINE UNITS; pass 1 measures with unit=1 and pass 2
        draws with the unit scaled so the tallest column exactly fills the panel."""
        y = TOP
        blocks = [
            (ctl["id"], 11, c["ink"], "bold", None, 0.35),
            (ctl["full"], 7.6, c["secondary"], "normal", None, 0.12),
            (f'{ctl["fn"]}()  ·  line {ctl["line"]}', 7.0, c["muted"], "normal",
             "monospace", 0.12),
            (ctl["env"], 7.0, c["series"][0], "normal", "monospace", 0.80),
            ("SIGNAL", 6.6, c["muted"], "bold", None, 0.12),
            (_wrap(ctl["signal"], 60), 7.2, c["ink"], "normal", None, 0.80),
            ("RULE", 6.6, c["muted"], "bold", None, 0.12),
            (_wrap(ctl["rule"], 60), 7.2, c["ink"], "normal", None, 0.80),
            ("WHAT BROKE", 6.6, c["critical"], "bold", None, 0.12),
            (_wrap(ctl["broke"], 62), 7.0, c["secondary"], "normal", None, 0.0),
        ]
        for text, size, col, weight, family, gap in blocks:
            if draw:
                ax.text(x + 0.014, y, text, transform=ax.transAxes, fontsize=size,
                        color=col, va="top", weight=weight, linespacing=1.45,
                        **({"family": family} if family else {}))
            lines = text.count("\n") + 1
            y -= (lines * size / 7.2 + gap) * unit
        return y

    used = max(TOP - column(x, ctl, 1.0, False) for x, ctl in zip(xs, K.CONTROLLERS))
    unit = (TOP - FLOOR) / used
    for x, ctl in zip(xs, K.CONTROLLERS):
        ax.add_patch(Rectangle((x, FLOOR - 0.03), W, 1.0 - (FLOOR - 0.03),
                               transform=ax.transAxes, facecolor=c["band"],
                               edgecolor="none", zorder=0))
        column(x, ctl, unit, True)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    S.title(ax, "A. All three are one mechanism — only the decision rule differs",
            "per prefill-layer-span, move the green-context split index by ±1 · "
            "min dwell 3 steps after a switch (anti-oscillation)", c, subwidth=130)

    # ── B. where each controller actually sat on the split axis ───────────────
    ax2 = fig.add_subplot(gs[1, 0])
    p = K.POSITIONING
    sx = [s for s, _ in p["static"]]
    sy = [g for _, g in p["static"]]
    ax2.plot(sx, sy, "-o", color=c["series"][0], zorder=3, markersize=8,
             markeredgecolor=c["surface"], markeredgewidth=1.5,
             label="static split (measured)")
    for x, y in p["static"]:
        ax2.annotate(f"d{x}\n{y:.3f}", (x, y), xytext=(0, 9),
                     textcoords="offset points", ha="center", fontsize=7.8,
                     color=c["secondary"], linespacing=1.35)
    ax2.plot(p["optimum"], dict(p["static"])[p["optimum"]], "o",
             color=c["good"], markersize=15, zorder=2, alpha=0.30)
    ax2.annotate("optimum", (p["optimum"], dict(p["static"])[p["optimum"]]),
                 xytext=(16, -4), textcoords="offset points", fontsize=8.4,
                 color=c["good"], weight="bold", va="center")

    for lbl, band, xrep, g, sd, note in p["realized"]:
        if band:
            ax2.plot(list(band), [g, g], "-", color=c["series"][1], lw=8, alpha=0.28,
                     zorder=2, solid_capstyle="butt")
            ax2.annotate(f"dwell {band[0]}–{band[1]}", ((band[0] + band[1]) / 2, g),
                         xytext=(0, -21), textcoords="offset points", ha="center",
                         fontsize=7.4, color=c["series"][1])
        ax2.errorbar(xrep, g, yerr=sd, fmt="none", ecolor=c["series"][1],
                     elinewidth=2.0, capsize=4, capthick=1.5, zorder=3)
        ax2.plot(xrep, g, "D", color=c["series"][1], markersize=9, zorder=4,
                 markeredgecolor=c["surface"], markeredgewidth=1.5)
        dy = 13 if band else -30
        va = "bottom" if band else "top"
        ax2.annotate(f"{lbl}\n{g:.3f} ± {sd:.3f}", (xrep, g), xytext=(-13, dy),
                     textcoords="offset points", ha="right", va=va, fontsize=8,
                     color=c["series"][1], weight="bold", linespacing=1.35)
    # the settling cost: the dynamic arm scores BELOW the static it froze at
    ax2.annotate("", xy=(34, 3.171), xytext=(34, 3.132),
                 arrowprops=dict(arrowstyle="<->", color=c["critical"], lw=1.3))
    ax2.annotate("−0.039\nsettling cost", (34.7, 3.128), fontsize=7.6,
                 color=c["critical"], va="top", ha="left", weight="bold",
                 linespacing=1.35)

    ax2.set_xlabel("decode SM  (prefill = 108 − decode)")
    ax2.set_ylabel(p["unit"])
    ax2.set_xticks([16, 24, 34, 44, 54])
    ax2.set_xlim(12, 52)
    ax2.set_ylim(2.60, 3.38)
    ax2.legend(handles=[
        Line2D([], [], marker="o", color=c["series"][0], markersize=8,
               markerfacecolor=c["series"][0], label="static split (measured)"),
        Line2D([], [], marker="D", color="none", markersize=8,
               markerfacecolor=c["series"][1], label="dynamic — where it actually sat")],
        loc="lower right")
    S.title(ax2, "B. The loss is the seat, not the switching",
            "same varying trace as the HE0 ranking, so the two are directly comparable "
            "· n≥4", c, subwidth=130)

    note_block(fig,
               "Two candidate causes were eliminated by direct measurement, not argument. "
               "SWITCH OVERHEAD: a rep matched static with only 2 switches, and slo "
               "(5 switches) scores below bind (21 switches). CONTROLLER CPU: mean 32–36 µs "
               "per call = 0.9% of one decode step, 0.014% of wall clock. What remains is "
               "POSITIONING — the controller is decode-STARVED, and risk/reward is 18:1 "
               "(a prefill-ward move gains ≤2.3% at low load and costs ≤41.5% at high load). "
               + p["ceiling"],
               c, y=0.030, width=196)
    S.footer(fig, f"source: {K.CTRL_SRC} · {K.SRC['consensus']} §1-8/§1-10/§1-12/§1-17 · "
                  f"{K.SRC['neg']} §2.2/§2.5", c, width=210)
    fig.subplots_adjust(left=0.050, right=0.990, top=0.925, bottom=0.190)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", type=int, default=None)
    ap.add_argument("--outdir", default=str(HERE))
    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    wanted = [args.only] if args.only else sorted(FIGS)
    for n in wanted:
        slug, fn = FIGS[n]
        for mode in ("light", "dark"):
            colors = S.apply(mode)
            fig = fn(colors)
            png = outdir / f"fig{n:02d}_{slug}.{mode}.png"
            fig.savefig(png, dpi=200)
            if mode == "light":
                pdf = outdir / f"fig{n:02d}_{slug}.pdf"
                fig.savefig(pdf)
                print(f"  fig{n:02d} {slug:24s} -> {pdf.name} + both PNGs")
            plt.close(fig)


if __name__ == "__main__":
    main()
