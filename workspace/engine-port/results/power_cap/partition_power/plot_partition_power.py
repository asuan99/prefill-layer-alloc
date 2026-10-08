#!/usr/bin/env python3
"""Figures for the partition-power probe (CPU only).   python3 plot_partition_power.py <job_dir>

figA_power_increments.png : per split, board power above idle for P-only / D-only / P+D, with the
                            sum P-only + D-only marked (additivity check).  Whole-GPU refs at left.
figB_toggle_timeseries.png: per split, 20 ms buffered board power while decode runs continuously and
                            prefill toggles 200 ms ON / OFF (gray = prefill ON).
figC_throughput.png       : per split, work rate of each partition alone vs together (interference).
"""
import bisect, json, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.facecolor": SURF, "figure.facecolor": SURF,
                     "axes.spines.top": False, "axes.spines.right": False})
d = Path(sys.argv[1]); out = Path(__file__).resolve().parent / "figures"; out.mkdir(exist_ok=True)
S = json.load(open(d / "summary.json")); W = json.load(open(d / "work.json")); G = json.load(open(d / "green.json"))
tags = list(S["splits"]); idle = S["idle_W"]; ph = S["phases"]


def lab(tag):
    g = G.get(tag, {}); p = (g.get("prefill") or {}).get("smCount"); q = (g.get("decode") or {}).get("smCount")
    return f"{tag}\n(realized {p}/{q} SM)" if p and q else tag


# ---- A: increments
fig, ax = plt.subplots(figsize=(10, 4.4))
groups = ["whole GPU"] + tags; w = 0.24
for gi, g in enumerate(groups):
    if g == "whole GPU":
        vals = [ph["whole_P"]["ctr_W"] - idle, ph["whole_D"]["ctr_W"] - idle, None]
    else:
        s = S["splits"][g]; vals = [s["dW_P_only"], s["dW_D_only"], s["dW_PD"]]
    for k, (v, c, name) in enumerate(zip(vals, (BLUE, ORANGE, AQUA), ("prefill-like only", "decode-like only", "both together"))):
        if v is None:
            continue
        ax.bar(gi + (k - 1) * (w + 0.02), v, width=w, color=c, label=name if gi == 1 else None)
    if g != "whole GPU":
        tot = vals[0] + vals[1]
        ax.plot([gi + 0.26 - w / 2, gi + 0.26 + w / 2], [tot, tot], color=INK, lw=1.5, ls="--", label="sum of the two alone" if gi == 1 else None)
        ax.annotate(f"{s['additivity_ratio']:.2f}× of sum", (gi + 0.26, max(tot, vals[2])), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=9, color=INK)
ax.set_xticks(range(len(groups))); ax.set_xticklabels(["whole GPU\n(108 SM, default)"] + [lab(t) for t in tags])
ax.set_ylabel(f"board power above idle (W)\nidle = {idle:.0f} W"); ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.13), fontsize=9)
fig.tight_layout(); fig.savefig(out / "figA_power_increments.png", dpi=150); plt.close(fig)

# ---- B: toggle time series
B = [ln.strip().split(",") for ln in open(d / "buffered.csv") if ln[0].isdigit()]
bt = [int(r[0]) / 1e6 for r in B]; bp = [int(r[1]) / 1000 for r in B]
fig, axes = plt.subplots(len(tags), 1, figsize=(10, 2.4 * len(tags) + 0.6), sharey=True)
for ax, tag in zip(axes, tags):
    ed = W[f"{tag}_toggle"]["toggle_edges"]; a0 = ed[2][1]; a1 = a0 + 2.0
    for (k, x0), (_, x1) in zip(ed, ed[1:]):
        if k == "p_on" and x1 > a0 and x0 < a1:
            ax.axvspan(max(x0, a0) - a0, min(x1, a1) - a0, color="#d9d8d4", lw=0)
    i, j = bisect.bisect_left(bt, a0), bisect.bisect_right(bt, a1)
    ax.plot([x - a0 for x in bt[i:j]], bp[i:j], color=AQUA, lw=1.5, marker="o", ms=2.5)
    s = S["splits"][tag]
    ax.set_title(f"{lab(tag).replace(chr(10), ' ')} — decode continuous, prefill 200 ms ON/OFF (gray = prefill ON);"
                 f" ON {s['toggle_prefill_on_W']:.0f} W vs OFF {s['toggle_prefill_off_W']:.0f} W", loc="left", fontsize=9, color=INK)
    ax.set_xlim(0, 2.0); ax.grid(axis="y", color=GRID, lw=0.8); ax.set_ylabel("W")
axes[-1].set_xlabel("time (s) — NVML buffered power samples (~20 ms)")
fig.tight_layout(); fig.savefig(out / "figB_toggle_timeseries.png", dpi=150); plt.close(fig)

# ---- C: throughput
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
for ax, (kind, name) in zip(axes, (("P", "prefill-like GEMM units/s"), ("D", "decode-like GEMV units/s"))):
    alone = [S["splits"][t][f"{kind}_rate_alone"] for t in tags]; tog = [S["splits"][t][f"{kind}_rate_with_{'D' if kind == 'P' else 'P'}"] for t in tags]
    xs = range(len(tags))
    ax.bar([x - 0.2 for x in xs], alone, width=0.38, color=BLUE if kind == "P" else ORANGE, label="alone")
    ax.bar([x + 0.2 for x in xs], tog, width=0.38, color=AQUA, label="with the other partition busy")
    for x, a, b in zip(xs, alone, tog):
        ax.annotate(f"{b / a:.2f}×", (x + 0.2, b), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9, color=INK)
    ax.set_xticks(list(xs)); ax.set_xticklabels(tags); ax.set_title(name, loc="left", fontsize=10, color=INK)
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True); ax.legend(frameon=False, fontsize=9)
fig.tight_layout(); fig.savefig(out / "figC_throughput.png", dpi=150); plt.close(fig)
print("wrote", *(out / n for n in ("figA_power_increments.png", "figB_toggle_timeseries.png", "figC_throughput.png")))
