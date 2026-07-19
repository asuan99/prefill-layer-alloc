#!/usr/bin/env python
"""The entanglement (얽힘) mechanism — concretely, and why TP scale-out inherits it.

The green-context SM split isolates COMPUTE (disjoint SM partitions) but NOT the scheduler-level
shared resources: one running-batch pool (max_running_requests=48), one KV pool, one admission
control. So prefill and decode stay COUPLED through the batch, and prefill's TTFT is governed by
DECODE's SM allocation, not by prefill's own SM.

Smoking gun (ShareGPT r8, committed): d16 gives prefill 92 SM (max) yet TTFT 7.24s; d24 gives
prefill only 84 SM yet TTFT 1.21s. MORE prefill compute -> WORSE prefill latency -> the bottleneck
is admission (coupling), not compute.

Chain: decode few SM -> ITL up -> each req holds its batch slot longer -> Little's law demanded
concurrency N = lambda x out_len x ITL rises above the cap 48 -> batch full -> scheduler blocks new
prefill admission -> prefills queue -> TTFT explodes.

Scale-out: TP keeps ONE scheduler / ONE batch pool / ONE KV -> inherits the SAME entanglement.
Only DISAGGREGATION (separate prefill & decode clusters, each with its own batch+admission) breaks it.
"""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

def box(a, x, y, w, h, text, fc, ec="k", fs=9, tc="k", bold=True):
    a.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.02",
                               fc=fc, ec=ec, lw=1.6))
    a.text(x+w/2, y+h/2, text, ha="center", va="center", fontsize=fs, color=tc,
           fontweight="bold" if bold else "normal")

def arrow(a, x1, y1, x2, y2, c="k", lw=2.0, style="-|>"):
    a.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=16,
                                color=c, lw=lw, shrinkA=2, shrinkB=2))

fig, ax = plt.subplots(1, 2, figsize=(16, 7.2))
fig.suptitle("The entanglement — the SM split isolates COMPUTE but not the SHARED SCHEDULER STATE\n"
             "prefill's TTFT is governed by DECODE's SM allocation, through the shared running batch (measured: ShareGPT r8)",
             fontsize=12.5, fontweight="bold")

# ══ Panel 1: the causal chain on a single GPU ══
a = ax[0]; a.set_xlim(0, 10); a.set_ylim(0, 10); a.axis("off")
a.set_title("① single GPU — the coupling path (d16 example: decode 16 SM)", fontsize=11, fontweight="bold")

# the GPU with two SM partitions (isolated compute)
a.add_patch(plt.Rectangle((0.3, 7.6), 9.4, 1.9, fc="#eef2f6", ec="#888", lw=1.3))
a.text(0.5, 9.25, "GPU (108 SM), green-context split — COMPUTE is isolated:", fontsize=9, style="italic")
box(a, 0.7, 7.9, 5.3, 1.0, "PREFILL partition\n92 SM (d16: the MAX)", "#f6d9d5", ec="#c1121f", fs=9)
box(a, 6.3, 7.9, 3.1, 1.0, "DECODE partition\n16 SM (starved)", "#d5e3f0", ec="#0077b6", fs=9)

# the shared scheduler layer (the coupling)
a.add_patch(plt.Rectangle((0.3, 5.7), 9.4, 1.3, fc="#fff3d6", ec="#e0a000", lw=2.0))
a.text(5.0, 6.72, "★ SHARED SCHEDULER STATE — NOT isolated by the split", fontsize=9.5, ha="center", fontweight="bold", color="#a06a00")
a.text(5.0, 6.12, "one running-batch pool (max_running_requests = 48)   ·   one KV pool   ·   one admission control",
       fontsize=8.7, ha="center", color="#7a5200")

# causal chain (vertical)
yy = [5.0, 4.0, 3.0, 2.0, 1.0]
steps = [
    ("decode has 16 SM  →  decode is SLOW  →  ITL ↑ = 61.9ms (> 60ms SLO)", "#d5e3f0", "#0077b6"),
    ("each request HOLDS its batch slot longer  (slot time = out_len × ITL)", "#e8e8e8", "#666"),
    ("Little's law:  N = λ × out_len × ITL  →  demanded concurrency 57  >  cap 48", "#fff3d6", "#e0a000"),
    ("running batch FULL  →  scheduler BLOCKS new prefill admission", "#f6d9d5", "#c1121f"),
    ("prefills QUEUE (compute-idle 92 SM)  →  TTFT ↑ = 7.24s   ← prefill dies", "#f6c0b8", "#c1121f"),
]
for (txt, fc, ec), y in zip(steps, yy):
    box(a, 0.7, y-0.38, 8.7, 0.76, txt, fc, ec=ec, fs=8.6)
    if y > 1.0:
        arrow(a, 5.0, y-0.42, 5.0, y-0.62, c="#555", lw=1.8)

# ══ Panel 2: scale-out — TP inherits, disaggregation breaks ══
a = ax[1]; a.set_xlim(0, 10); a.set_ylim(0, 10); a.axis("off")
a.set_title("② scale-out: does the coupling persist?", fontsize=11, fontweight="bold")

# TP (top)
a.text(0.3, 9.5, "TENSOR-PARALLEL (same logical model on N GPUs)", fontsize=10, fontweight="bold", color="#c1121f")
box(a, 0.6, 8.2, 2.5, 0.9, "GPU0\nprefill|decode SM", "#eef2f6", fs=8)
box(a, 3.4, 8.2, 2.5, 0.9, "GPU1\nprefill|decode SM", "#eef2f6", fs=8)
box(a, 6.2, 8.2, 2.5, 0.9, "GPU2 …\nprefill|decode SM", "#eef2f6", fs=8)
box(a, 2.0, 6.9, 6.0, 0.9, "ONE scheduler · ONE running batch (48) · ONE KV · ONE admission", "#fff3d6", ec="#e0a000", fs=8.5)
for x in (1.85, 4.65, 7.45):
    arrow(a, x, 8.15, 5.0, 7.85, c="#aaa", lw=1.2)
a.text(5.0, 6.35, "→ decode congestion still fills the shared batch → blocks prefill admission", fontsize=8.7, ha="center", color="#c1121f", fontweight="bold")
a.text(5.0, 5.95, "SAME entanglement, now distributed — scale-out does NOT decouple it", fontsize=9, ha="center", color="#c1121f", fontweight="bold")

a.plot([0.3, 9.7], [5.4, 5.4], color="#bbb", ls="--", lw=1)

# Disaggregation (bottom)
a.text(0.3, 4.9, "DISAGGREGATION (separate prefill & decode clusters)", fontsize=10, fontweight="bold", color="#1a7a3a")
box(a, 0.6, 3.4, 3.9, 1.1, "PREFILL cluster\nown scheduler · own batch\n· own admission", "#f6d9d5", ec="#c1121f", fs=8.3)
box(a, 5.5, 3.4, 3.9, 1.1, "DECODE cluster\nown scheduler · own batch\n· own KV", "#d5e3f0", ec="#0077b6", fs=8.3)
arrow(a, 4.5, 3.95, 5.5, 3.95, c="#1a7a3a", lw=2.2)
a.text(5.0, 4.25, "KV handoff\n(once)", fontsize=7.6, ha="center", color="#1a7a3a")
a.text(5.0, 2.75, "decode congestion holds DECODE-pool slots only —", fontsize=9, ha="center", color="#1a7a3a", fontweight="bold")
a.text(5.0, 2.35, "prefill admission governed by PREFILL pool's own batch → COUPLING BROKEN", fontsize=9, ha="center", color="#1a7a3a", fontweight="bold")
a.text(5.0, 1.6, "⇒ this is the only scale-out that removes the entanglement\n(new cost: KV migration, whole-GPU granularity)",
       fontsize=8.8, ha="center", color="#444", style="italic",
       bbox=dict(boxstyle="round", fc="#eef7ee", ec="#1a7a3a"))

# smoking gun box
a.add_patch(FancyBboxPatch((0.5, 0.15), 9.0, 0.85, boxstyle="round,pad=0.02", fc="#fff8f6", ec="#c1121f", lw=1.5))
a.text(5.0, 0.57, "SMOKING GUN:  d16 → prefill 92 SM, TTFT 7.24s   vs   d24 → prefill 84 SM, TTFT 1.21s   |   MORE prefill compute → 6× WORSE TTFT",
       fontsize=8.7, ha="center", fontweight="bold", color="#c1121f")

fig.tight_layout(rect=[0, 0.01, 1, 0.93])
fig.savefig("entanglement.png", dpi=150)
print("wrote entanglement.png")
