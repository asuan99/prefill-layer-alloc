"""N2(3-a) Job-PROFILE pdmux-n2-static-grid-a5de2d6-20261012021243 — figures of the runner's mechanical output
(N2_PROFILE_POINTS.csv / N2_PROFILE_RESULT.json).  RESULT AUDIT PENDING: these are not judgments.
Reads only alone-window (W) values; no co-location statistics (blind rule §5-4 (ii)(나))."""
import csv
import json
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
JOB = "pdmux-n2-static-grid-a5de2d6-20261012021243"
SRC = os.path.join(HERE, "..", f"n2p_{JOB}")
DS = [16, 24, 34, 44, 108]
BS = [1, 8, 32, 48]
CS = [512, 2048, 8192]
RAMP = {16: "#86b6ef", 24: "#5598e7", 34: "#2a78d6", 44: "#1c5cab", 108: "#0d366b"}   # ordinal blue 250..700
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
MISS = "#d03b3b"
NOTE = (f"Job {JOB} · A100 SXM4 80GB · Nemotron-Nano-9B-v2 · cudagraph ON · cap 48 · "
        "alone decode (no prefill) · runner output, RESULT AUDIT PENDING — not a judgment")

plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.facecolor": SURF, "figure.facecolor": SURF,
                     "axes.spines.top": False, "axes.spines.right": False})


def load():
    pts = {}
    with open(os.path.join(SRC, "N2_PROFILE_POINTS.csv")) as fh:
        for r in csv.DictReader(fh):
            pts[(int(r["decode_sms"]), int(r["batch_size"]), int(r["context_tokens"]))] = r
    res = json.load(open(os.path.join(SRC, "N2_PROFILE_RESULT.json")))
    return pts, res


def fig_step_time(pts):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), sharey=True)
    for ax, c in zip(axes, CS):
        for d in DS:
            xs, ys = [], []
            for b in BS:
                r = pts.get((d, b, c))
                if r:
                    xs.append(b); ys.append(float(r["itl_p95_ms"]))
            ax.plot(xs, ys, "-o", color=RAMP[d], lw=2, ms=6, label=f"D{d}" if d != 108 else "108 (no split)",
                    markeredgecolor=SURF, markeredgewidth=1.5)
        for bb in BS:
            miss = [d for d in DS if (d, bb, c) not in pts]
            if miss:
                ax.plot([bb], [0.5], marker="x", color=MISS, ms=8, mew=2, clip_on=False)
                ax.annotate(" · ".join(f"D{d}" for d in miss) + "\nmissing (WINDOW_SHORT 4/4)", (bb, 0.5),
                            textcoords="offset points", xytext=(0, 8), ha="center", fontsize=7.5, color=MISS)
        ax.set_title(f"context {c} tokens", color=INK, fontsize=11)
        ax.set_xticks(BS); ax.set_xlabel("decode batch size b")
        ax.grid(axis="y", color=GRID, lw=0.8); ax.set_ylim(0, 110)
    axes[0].set_ylabel("decode step time, p95 (ms)\nmedian over 4 replicates")
    axes[-1].legend(title="decode SMs", frameon=False, loc="upper left", fontsize=9)
    fig.suptitle("Alone decode step time vs batch — fewer decode SMs, slower steps", color=INK, fontsize=13, x=0.01,
                 ha="left")
    fig.text(0.01, 0.005, NOTE, fontsize=7.5, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    out = os.path.join(HERE, "fig1_step_time_vs_batch.png"); fig.savefig(out, dpi=150); plt.close(fig)
    return out


def fig_slowdown(pts):
    rows = [(b, c) for b in BS for c in CS]
    fig, ax = plt.subplots(figsize=(8.6, 6.4))
    data = []
    for (b, c) in rows:
        base = float(pts[(108, b, c)]["itl_p95_ms"])
        data.append([float(pts[(d, b, c)]["itl_p95_ms"]) / base if (d, b, c) in pts else math.nan for d in DS[:-1]])
    seq = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("blue", seq); cmap.set_bad("#f0efec")
    import numpy as np
    arr = np.array(data)
    im = ax.imshow(arr, cmap=cmap, vmin=1.0, vmax=3.0, aspect="auto")
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            v = arr[i, j]
            if math.isnan(v):
                ax.text(j, i, "missing\n(WINDOW_SHORT)", ha="center", va="center", fontsize=7.5, color=MISS)
            else:
                ax.text(j, i, f"{v:.2f}×", ha="center", va="center", fontsize=9,
                        color=SURF if v > 2.0 else INK)
    ax.set_xticks(range(4)); ax.set_xticklabels([f"D{d}" for d in DS[:-1]])
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([f"b={b}, c={c}" for b, c in rows], fontsize=8.5)
    ax.set_xlabel("decode SMs (sticky split)")
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02); cb.set_label("step time p95 ÷ unsplit (108 SM)")
    cb.outline.set_visible(False)
    ax.set_title("How much slower is decode on a partition than on the whole GPU?", color=INK, loc="left",
                 fontsize=12)
    fig.text(0.01, 0.005, NOTE, fontsize=6.5, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    out = os.path.join(HERE, "fig2_slowdown_vs_unsplit.png"); fig.savefig(out, dpi=150); plt.close(fig)
    return out


def fig_replicates(res):
    reps = res["replicates"]
    units = sorted({r["unit"] for r in reps}, key=lambda u: (u.split("_")[1], u))
    cells = [(b, c) for b in BS for c in CS]
    fig, ax = plt.subplots(figsize=(11, 7.4))
    col = {"VALID": "#1baf7a", "WINDOW_SHORT": MISS}
    for i, u in enumerate(units):
        for r in reps:
            if r["unit"] != u:
                continue
            j = cells.index((r["b"], r["c"]))
            reason = r["status"]
            ax.add_patch(plt.Rectangle((j - 0.45, i - 0.4), 0.9, 0.8, color=col.get(reason, "#eda100")))
    ax.set_xlim(-0.6, len(cells) - 0.4); ax.set_ylim(len(units) - 0.5, -0.5)
    ax.set_xticks(range(len(cells))); ax.set_xticklabels([f"b{b}\nc{c}" for b, c in cells], fontsize=7.5)
    ax.set_yticks(range(len(units))); ax.set_yticklabels(units, fontsize=7.5)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#1baf7a", label="VALID"), Patch(color=MISS, label="WINDOW_SHORT (window not formed)")],
              frameon=False, loc="upper left",
              bbox_to_anchor=(1.0, 1.0), fontsize=9)
    cc = res.get("replicate_class_counts", {})
    ax.set_title(f"Every replicate by boot × cell — valid {cc.get('valid')}, predicate failure "
                 f"{cc.get('predicate_failure')}, measurement failure {cc.get('measurement_failure')}",
                 color=INK, loc="left", fontsize=11)
    fig.text(0.01, 0.005, NOTE, fontsize=6.5, color=INK2)
    fig.text(0.01, 0.022, "blank = cell not in that boot's plan (OBS44 boots carry 2 observer cells; "
             "OFF16/OFF44 boots supply the unsplit 108-SM points)", fontsize=7, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    out = os.path.join(HERE, "fig3_replicate_status.png"); fig.savefig(out, dpi=150); plt.close(fig)
    return out


if __name__ == "__main__":
    pts, res = load()
    for f in (fig_step_time(pts), fig_slowdown(pts), fig_replicates(res)):
        print(f)
    sys.exit(0)
