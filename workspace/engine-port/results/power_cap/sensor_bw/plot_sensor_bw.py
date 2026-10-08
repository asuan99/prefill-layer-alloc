#!/usr/bin/env python3
"""Figures for the sensor-bandwidth probes (CPU only).

    python3 plot_sensor_bw.py <job_dir> [<job_dir_v2>]   -> figures/*.png

fig1_timeseries.png : square-wave load (ground truth) vs NVML power snapshot, energy-counter power and,
                      if present, NVML buffered power samples, for P = 400 / 200 / 50 ms (1.2 s window each).
fig2_block_means.png: 5 s block mean power per period, counter vs snapshot (and buffered), one axis (W).
Palette: reference categorical slots 1-3 (blue, orange, aqua); ground truth in neutral gray.
"""
import bisect, csv, statistics as st, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.facecolor": SURF, "figure.facecolor": SURF,
                     "axes.spines.top": False, "axes.spines.right": False})

jobs = [Path(a) for a in sys.argv[1:]]
job = jobs[-1]                                  # prefer the v2 job (has buffered samples) for the time series
out = Path(__file__).resolve().parent / "figures"; out.mkdir(exist_ok=True)


def load(d):
    S = list(csv.DictReader(open(d / "samples.csv")))
    E = list(csv.DictReader(open(d / "edges.csv")))
    t = [float(r["t"]) for r in S]; pw = [float(r["usage_mW"]) / 1000 for r in S]; en = [float(r["energy_mJ"]) for r in S]
    ct, cp, li = [], [], 0
    for i in range(1, len(en)):
        if en[i] != en[li]:
            if t[i] > t[li]:
                ct += [t[li], t[i]]; v = (en[i] - en[li]) / 1000 / (t[i] - t[li]); cp += [v, v]
            li = i
    B = None
    if (d / "buffered.csv").exists():
        rows = [ln.strip().split(",") for ln in open(d / "buffered.csv") if ln[0].isdigit()]
        B = ([int(x[0]) / 1e6 for x in rows], [int(x[1]) / 1000 for x in rows])
    edges = {}
    for r in E:
        edges.setdefault(int(r["P_ms"]), []).append((r["edge"], float(r["t"])))
    return t, pw, en, ct, cp, B, edges


t, pw, en, ct, cp, B, edges = load(job)

# ---- fig 1: time series
Ps = [400, 200, 50]
fig, axes = plt.subplots(len(Ps), 1, figsize=(10, 7.5), sharey=True)
for ax, P in zip(axes, Ps):
    ed = edges[P]; mid = ed[len(ed) // 2][1]; a, b = mid, mid + 1.2
    gx, gy = [], []
    for (k, x0), (_, x1) in zip(ed, ed[1:]):
        if x1 < a or x0 > b:
            continue
        lvl = 400 if k == "on" else 90
        gx += [max(x0, a), min(x1, b)]; gy += [lvl, lvl]
    ax.fill_between([x - a for x in gx], 0, gy, step=None, color="#d9d8d4", lw=0, label="load ON (ground truth)")
    i, j = bisect.bisect_left(t, a), bisect.bisect_right(t, b)
    ax.step([x - a for x in t[i:j]], pw[i:j], where="post", color=BLUE, lw=2, label="NVML power (polled, = power.draw)")
    ci, cj = bisect.bisect_left(ct, a), bisect.bisect_right(ct, b)
    ax.plot([x - a for x in ct[ci:cj]], cp[ci:cj], color=ORANGE, lw=2, label="energy counter (power per update)")
    if B:
        bi, bj = bisect.bisect_left(B[0], a), bisect.bisect_right(B[0], b)
        ax.plot([x - a for x in B[0][bi:bj]], B[1][bi:bj], color=AQUA, lw=1.5, marker="o", ms=3,
                label="NVML buffered power samples")
    ax.set_title(f"square-wave period {P} ms (ON {P // 2} ms / OFF {P // 2} ms)", loc="left", color=INK, fontsize=10)
    ax.set_ylim(0, 450); ax.set_xlim(0, 1.2); ax.grid(axis="y", color=GRID, lw=0.8); ax.set_ylabel("W")
axes[-1].set_xlabel("time within window (s)")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 0.965), ncol=2, frameon=False, fontsize=9)
fig.suptitle(f"VESSL A100 power sensors vs a known load ({job.name})", y=0.995, color=INK, fontsize=11, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.90)); fig.savefig(out / "fig1_timeseries.png", dpi=150); plt.close(fig)

# ---- fig 2: block means
rows = []
for P in sorted(edges):
    ts = [x for _, x in edges[P]]; a, b = ts[0] + 0.5, (ts[1] if P == 0 else ts[-1]) - 0.2
    def ctr(x):
        k = bisect.bisect_left(t, x); k = min(max(k, 1), len(t) - 1)
        return en[k - 1] + (en[k] - en[k - 1]) * (x - t[k - 1]) / (t[k] - t[k - 1])
    cm = (ctr(b) - ctr(a)) / 1000 / (b - a)
    i, j = bisect.bisect_left(t, a), bisect.bisect_right(t, b); sm = st.fmean(pw[i:j])
    bm = None
    if B:
        bi, bj = bisect.bisect_left(B[0], a), bisect.bisect_right(B[0], b)
        bm = st.fmean(B[1][bi:bj]) if bj > bi else None
    rows.append(("continuous" if P == 0 else f"{P} ms", cm, sm, bm))
fig, ax = plt.subplots(figsize=(9, 4))
n = len(rows); w = 0.26 if B else 0.38; xs = range(n)
series = [("energy counter", 1, ORANGE), ("polled power (snapshot mean)", 2, BLUE)] + ([("buffered samples mean", 3, AQUA)] if B else [])
for k, (lab, idx, c) in enumerate(series):
    off = (k - (len(series) - 1) / 2) * (w + 0.02)
    vals = [r[idx] if r[idx] is not None else 0 for r in rows]
    ax.bar([x + off for x in xs], vals, width=w, color=c, label=lab)
    for x, v in zip(xs, vals):
        if lab.startswith("polled") and rows[x][1] and abs(v / rows[x][1] - 1) > 0.1:
            ax.annotate(f"{v / rows[x][1] - 1:+.0%}", (x + off, v), xytext=(0, 3), textcoords="offset points",
                        ha="center", fontsize=9, color=INK)
ax.set_xticks(list(xs)); ax.set_xticklabels([r[0] for r in rows]); ax.set_ylabel("mean power over 5 s block (W)")
ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True); ax.set_xlabel("load pattern (square-wave period)")
ax.legend(frameon=False, ncol=len(series), loc="upper center", bbox_to_anchor=(0.5, 1.12), fontsize=9)
fig.tight_layout(); fig.savefig(out / "fig2_block_means.png", dpi=150); plt.close(fig)
print("wrote", out / "fig1_timeseries.png", out / "fig2_block_means.png")
