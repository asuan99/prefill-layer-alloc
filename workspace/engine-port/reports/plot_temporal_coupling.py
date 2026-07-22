#!/usr/bin/env python
"""Does between-step (TEMPORAL) intervention escape the 116>108 coupling? — user's objection.

The user is right that I over-stated it: "92 prefill + 24 decode = 116 > 108" is a SPATIAL
(within-step, concurrent) statement. Between steps you CAN intervene — give prefill more SM some
steps, decode more other steps. That is exactly what the dynamic controllers I tested do (adjust
the split per window). So the intervention room is real and IS used.

But temporal muxing does NOT decouple TTFT and ITL — it MOVES the same tradeoff from the SM axis
to the TIME axis:
  - a decode request's ITL is the wall-clock GAP between its tokens. Interleave a prefill chunk
    between decode steps -> that request's ITL for that step includes the prefill duration -> ITL up.
  - a new request's TTFT is the wait until its prompt is processed. Spend the GPU on decode now
    -> the new request waits -> TTFT up.
Both latencies are ticking for CONCURRENTLY-active requests, and on ONE GPU, time spent serving
one is time NOT serving the other. So the decoupled oracle (d16-TTFT AND d24-ITL at once) is out of
reach in TIME just as in SPACE. Chunked prefill BOUNDS the ITL inflation (partial mitigation) but
cannot remove it. Only DISAGGREGATION gives two independent time-lines -> genuine decoupling.
"""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

C_P, C_D, C_GAP = "#c1121f", "#0077b6", "#f4a261"
fig, ax = plt.subplots(3, 1, figsize=(14.5, 9.2))
fig.suptitle("Between-step (temporal) intervention IS possible — but on one GPU it re-couples TTFT and ITL in the TIME axis\n"
             "prefill-time and decode-time are the SAME time; serving one starves the other's latency. Only disaggregation gives two independent time-lines.",
             fontsize=11.5, fontweight="bold")

# ── (1) SPATIAL mux: concurrent, shares 108 SM ──
a = ax[0]; a.set_xlim(0, 20); a.set_ylim(0, 3); a.axis("off")
a.set_title("① SPATIAL mux (SM-split, concurrent) — the 116>108 conflict I described", fontsize=10.5, fontweight="bold", loc="left")
for t in range(0, 20, 2):
    a.add_patch(plt.Rectangle((t, 1.6), 1.9, 1.0, fc=C_P, ec="white", alpha=.85))
    a.add_patch(plt.Rectangle((t, 0.4), 1.9, 1.0, fc=C_D, ec="white", alpha=.85))
a.text(-0.3, 2.1, "PREFILL\n(wants 92 SM)", ha="right", va="center", fontsize=8.5, color=C_P, fontweight="bold")
a.text(-0.3, 0.9, "DECODE\n(wants 24 SM)", ha="right", va="center", fontsize=8.5, color=C_D, fontweight="bold")
a.text(10, 2.85, "both run every instant → 92+24 = 116 SM needed, only 108 exist → prefill must drop to 84 (d24)", ha="center", fontsize=9, color="#a00", fontweight="bold")

# ── (2) TEMPORAL mux: time-slice, full SM each — but latency inflates ──
a = ax[1]; a.set_xlim(0, 20); a.set_ylim(0, 3); a.axis("off")
a.set_title("② TEMPORAL mux (time-slice, full 108 SM each window) — user's intervention room", fontsize=10.5, fontweight="bold", loc="left")
# alternating windows
wins = [(0,4,"P"),(4,8,"D"),(8,12,"P"),(12,16,"D"),(16,20,"P")]
for s,e,k in wins:
    a.add_patch(plt.Rectangle((s, 1.0), e-s-0.2, 1.2, fc=C_P if k=="P" else C_D, ec="white", alpha=.85))
    a.text((s+e)/2-0.1, 1.6, "PREFILL win\n(108 SM)" if k=="P" else "DECODE win\n(108 SM)", ha="center", va="center", fontsize=8, color="white", fontweight="bold")
# ITL gap annotation during prefill window
a.annotate("", xy=(4,0.7), xytext=(8,0.7), arrowprops=dict(arrowstyle="<->", color=C_GAP, lw=2))
a.text(2, 0.45, "in-flight decode PAUSED during P-win\n→ its ITL gap = whole P-window → ITL ↑", fontsize=8, color="#a15c00", fontweight="bold")
a.text(12, 0.45, "new request arriving in D-win WAITS for a P-win\n→ its TTFT ↑", fontsize=8, color="#a15c00", fontweight="bold")
a.text(10, 2.7, "each window gets full SM, BUT pausing decode inflates ITL / pausing prefill inflates TTFT → SAME tradeoff, now in TIME", ha="center", fontsize=9, color="#a00", fontweight="bold")

# ── (3) DISAGGREGATION: two independent time-lines ──
a = ax[2]; a.set_xlim(0, 20); a.set_ylim(0, 3); a.axis("off")
a.set_title("③ DISAGGREGATION (two GPUs) — genuinely independent → decoupled (+16% reachable)", fontsize=10.5, fontweight="bold", loc="left")
for t in range(0, 20, 2):
    a.add_patch(plt.Rectangle((t, 1.6), 1.9, 1.0, fc=C_P, ec="white", alpha=.85))
for t in range(0, 20, 1):
    a.add_patch(plt.Rectangle((t, 0.4), 0.9, 1.0, fc=C_D, ec="white", alpha=.85))
a.text(-0.3, 2.1, "GPU-P\nprefill 108 SM\ncontinuous", ha="right", va="center", fontsize=8.5, color=C_P, fontweight="bold")
a.text(-0.3, 0.9, "GPU-D\ndecode 108 SM\ncontinuous", ha="right", va="center", fontsize=8.5, color=C_D, fontweight="bold")
a.text(10, 2.85, "prefill runs full-tilt (best TTFT) AND decode runs full-tilt (best ITL) — never pausing each other", ha="center", fontsize=9, color="#1a7a3a", fontweight="bold")

fig.tight_layout(rect=[0, 0.04, 1, 0.92])
fig.text(0.5, 0.012,
         "So the intervention room between steps is REAL and is what the dynamic controllers use (and I tested — HE0). It PARTIALLY mitigates (chunked prefill bounds the ITL inflation) but cannot DECOUPLE, "
         "because on one GPU prefill-time and decode-time are the same time. The +16% decoupled oracle needs two capacities. The 116>108 was a spatial shorthand for a resource-sharing truth that holds in time too.",
         ha="center", fontsize=8.5, style="italic")
fig.legend(handles=[Patch(fc=C_P, label="prefill work"), Patch(fc=C_D, label="decode work"), Patch(fc=C_GAP, label="latency gap (inflation)")],
           loc="lower center", ncol=3, fontsize=9, bbox_to_anchor=(0.5, -0.005))
fig.savefig("temporal_coupling.png", dpi=150)
print("wrote temporal_coupling.png")
