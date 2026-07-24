#!/usr/bin/env python
"""CORRECTED (user was right): the coupling is THROUGHPUT-SHARING, not serialization.

My earlier framing drew "prefill window / decode window" alternation — as if all decode PAUSES
while prefill runs. That falsely implies requests are serialized. They are NOT: continuous batching
keeps ~48 requests concurrent, and prefill (of new reqs) + decode (of in-flight reqs) run TOGETHER,
either co-batched in one forward (mixed batch) or concurrently on split SMs (PD-mux). No pause.

The coupling survives without any serialization, because ONE GPU has ONE finite throughput budget,
and both TTFT-work (prefill) and ITL-work (decode) draw from it:
  · MIXED-BATCH mode: put more prefill tokens in a step -> the step takes LONGER -> the decode tokens
    co-batched in that same step wait longer -> ITL rises. (not a pause; the shared step just lengthens)
  · PD-MUX (SM-split) mode: give prefill more SMs -> fewer SMs left for the concurrent decode -> decode
    slower -> ITL rises. (the 116>108 shorthand)
Both are ways of DIVIDING one finite throughput between TTFT-work and ITL-work; the division trades
them off. Concurrency does NOT escape it — concurrent requests still share the one budget.
The decoupled oracle (d16-TTFT AND d24-ITL) needs MORE throughput than one GPU has. Disaggregation =
TWO budgets (two GPUs) -> prefill full-tilt AND decode full-tilt -> decoupled.
"""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

C_P, C_D = "#c1121f", "#0077b6"
fig, ax = plt.subplots(2, 2, figsize=(15, 8.6))
fig.suptitle("The coupling is THROUGHPUT-SHARING, not serialization (user's correction) — requests run CONCURRENTLY, they just share one finite budget",
             fontsize=12, fontweight="bold")

# ── (1) WRONG picture I drew ──
a = ax[0][0]; a.set_xlim(0, 12); a.set_ylim(0, 4); a.axis("off")
a.set_title("✗ what I wrongly implied: serialized windows", fontsize=10.5, fontweight="bold", color="#999")
for s,e,k in [(0,4,"P"),(4,8,"D"),(8,12,"P")]:
    a.add_patch(plt.Rectangle((s,1.4),e-s-0.15,1.2,fc=C_P if k=="P" else C_D,alpha=.4))
    a.text((s+e)/2,2.0,"all PREFILL\n(decode paused)" if k=="P" else "all DECODE\n(prefill paused)",ha="center",va="center",fontsize=8.5,color="#777",fontweight="bold")
a.text(6,0.6,"implies requests wait for each other — FALSE",ha="center",fontsize=9,color="#999",style="italic")

# ── (2) RIGHT: continuous batching, concurrent ──
a = ax[0][1]; a.set_xlim(0, 12); a.set_ylim(0, 6); a.axis("off")
a.set_title("✓ reality: ~48 requests CONCURRENT, each step is a MIXED batch", fontsize=10.5, fontweight="bold", color="#1a7a3a")
# each step = a stack of concurrent requests: mostly decode tokens + a few prefill tokens
np.random.seed(1)
for step in range(6):
    x0 = 0.3 + step*1.9
    # decode tokens (many, thin)
    for r in range(8):
        a.add_patch(plt.Rectangle((x0, 0.5+r*0.42), 1.5, 0.36, fc=C_D, ec="white", alpha=.8))
    # prefill chunk (a few, on top)
    a.add_patch(plt.Rectangle((x0, 0.5+8*0.42), 1.5, 0.7, fc=C_P, ec="white", alpha=.85))
    a.text(x0+0.75, 0.5+8*0.42+0.35, "P", ha="center", va="center", fontsize=8, color="white", fontweight="bold")
a.text(6, 5.5, "one forward = 48 decode tokens (in-flight reqs) + a prefill chunk (new req), TOGETHER", ha="center", fontsize=9, color="#333", fontweight="bold")
a.text(0.3, 4.1, "prefill\nchunk", fontsize=8, color=C_P, fontweight="bold")
a.text(0.3, 2.0, "48 concurrent\ndecodes", fontsize=8, color=C_D, fontweight="bold")

# ── (3) the real coupling: step time inflates with prefill work ──
a = ax[1][0]
pf_tokens = [0, 512, 1024, 2048, 4096]
step_ms = [12, 20, 30, 48, 82]  # illustrative: more prefill in the batch -> longer step
itl = step_ms  # the co-batched decodes see this as their ITL
a.plot(pf_tokens, itl, "o-", color="#8000a0", lw=2.5, ms=9)
a.axhline(60, color="k", ls="--", lw=1.4); a.text(50, 62, "ITL SLO 60ms", fontsize=8.5)
a.fill_between(pf_tokens, 60, itl, where=[v>60 for v in itl], color="#c1121f", alpha=.15)
a.annotate("more prefill per step (better TTFT)\n→ longer shared step\n→ higher ITL for the co-batched decodes",
           (2048, 48), textcoords="offset points", xytext=(-150, -40), fontsize=8.6, color="#8000a0", fontweight="bold",
           arrowprops=dict(arrowstyle="->", color="#8000a0"))
a.set_xlabel("prefill tokens co-batched into the step"); a.set_ylabel("step time = ITL of in-flight decodes (ms)")
a.set_title("③ the REAL coupling: no pause — the shared step LENGTHENS\n(TTFT-work and ITL-work share one step's time)", fontsize=10, fontweight="bold")
a.grid(alpha=.3)

# ── (4) one budget vs two budgets ──
a = ax[1][1]; a.axis("off"); a.set_xlim(0,10); a.set_ylim(0,10)
a.set_title("④ one GPU = one throughput budget (shared) · disagg = two", fontsize=10, fontweight="bold")
a.text(0.3, 9.2, "SINGLE GPU — TTFT-work + ITL-work ≤ ONE budget", fontsize=9.5, color="#a00", fontweight="bold")
a.add_patch(plt.Rectangle((0.5, 7.2), 6.0, 1.3, fc=C_P, alpha=.7)); a.text(3.5, 7.85, "prefill (TTFT)", ha="center", color="white", fontsize=9, fontweight="bold")
a.add_patch(plt.Rectangle((6.5, 7.2), 2.6, 1.3, fc=C_D, alpha=.7)); a.text(7.8, 7.85, "decode (ITL)", ha="center", color="white", fontsize=8.5, fontweight="bold")
a.plot([9.1,9.1],[7.0,8.7], "k--", lw=2); a.text(9.2, 7.85, "budget\nedge", fontsize=7.5, va="center")
a.text(5, 6.5, "improving one latency draws budget from the other → tradeoff (space OR time OR batch-mix)", ha="center", fontsize=8.3, color="#a00", fontweight="bold")
a.plot([0.3,9.7],[5.6,5.6], color="#bbb", ls="--")
a.text(0.3, 5.0, "DISAGGREGATION — TWO budgets, independent", fontsize=9.5, color="#1a7a3a", fontweight="bold")
a.add_patch(plt.Rectangle((0.5, 3.0), 4.0, 1.3, fc=C_P, alpha=.7)); a.text(2.5, 3.65, "GPU-P: prefill\nfull budget", ha="center", color="white", fontsize=8.5, fontweight="bold")
a.add_patch(plt.Rectangle((5.2, 3.0), 4.0, 1.3, fc=C_D, alpha=.7)); a.text(7.2, 3.65, "GPU-D: decode\nfull budget", ha="center", color="white", fontsize=8.5, fontweight="bold")
a.text(5, 2.2, "prefill AND decode both full-tilt → best TTFT AND best ITL → decoupled (+16%)", ha="center", fontsize=8.5, color="#1a7a3a", fontweight="bold")

fig.tight_layout(rect=[0, 0.03, 1, 0.93])
fig.text(0.5, 0.01,
         "So requests are NOT serialized — they run concurrently. The tradeoff is real anyway because one GPU delivers a fixed amount of compute per unit time, and prefill (→TTFT) and decode (→ITL) both consume it. "
         "Whether you divide it by SM (space), by step (time), or by co-batch ratio (mix), the sum is capped. The +16% decoupled oracle needs two budgets — that is the case for disaggregation / dual-worker.",
         ha="center", fontsize=8.4, style="italic")
fig.legend(handles=[Patch(fc=C_P, label="prefill work → TTFT"), Patch(fc=C_D, label="decode work → ITL")],
           loc="lower center", ncol=2, fontsize=9, bbox_to_anchor=(0.5, -0.005))
fig.savefig("throughput_coupling.png", dpi=150)
print("wrote throughput_coupling.png")
