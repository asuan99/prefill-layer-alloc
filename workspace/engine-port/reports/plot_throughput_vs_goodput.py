#!/usr/bin/env python
"""Policy comparison re-sorted by THROUGHPUT (SLO-agnostic) vs GOODPUT (SLO-constrained).

User's point: the goodput ranking is largely an artifact of the SLO constraint. Re-ranking by
raw throughput (completed req/s, ignoring the TTFT/ITL SLO) tests that.

Finding: on throughput the policies span only 3.4% (they all saturate the GPU about equally);
on goodput they span 13.1% (4x). The ranking is NOT reversed -- d44 leads both -- but nearly the
entire differentiation is SLO ATTAINMENT (which requests hit the TTFT wall), not raw capacity.
Entanglement still leaves a small real throughput footprint (d16 lowest on both: starving decode
congests the batch and blocks admission -> fewer completions), ~1/4 of its goodput deficit.

Data (SLO-agnostic req/s from the same varying-trace jsonl used for goodput): results/slo_sched.
"""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# (policy, n, goodput, gp_std, req/s, rq_std) -- apples-to-apples, same fileset
D = [
    ("d44",          4, 3.220, 0.013, 3.776, 0.005),
    ("d34",          4, 3.171, 0.025, 3.759, 0.009),
    ("bind+GATE",    9, 3.132, 0.019, 3.745, 0.013),
    ("d24",          4, 3.039, 0.130, 3.677, 0.118),
    ("slo",          4, 2.964, 0.025, 3.696, 0.002),
    ("bind no-gate", 4, 2.934, 0.306, 3.700, 0.075),
    ("d16",          4, 2.846, 0.055, 3.654, 0.024),
]
names = [d[0] for d in D]
gp = np.array([d[2] for d in D]); gps = np.array([d[3] for d in D])
rq = np.array([d[4] for d in D]); rqs = np.array([d[5] for d in D])
attain = gp/rq*100

fig, ax = plt.subplots(1, 3, figsize=(16.5, 5.6))
fig.suptitle("Policy comparison re-sorted by THROUGHPUT vs GOODPUT — the ranking is an SLO-attainment effect, not a capacity effect\n"
             "variance trace (rate 3↔12), Zamba2-2.7B, A100-80GB · SLO-agnostic req/s from the same runs as goodput",
             fontsize=12, fontweight="bold")

# ── panel 1: throughput ranking (nearly flat) ──
a = ax[0]
order = np.argsort(-rq)
y = np.arange(len(D))
a.barh(y, rq[order], xerr=rqs[order], color="#4a7ba6", alpha=.9, capsize=3)
a.set_yticks(y); a.set_yticklabels([names[i] for i in order]); a.invert_yaxis()
for i, yi in enumerate(y):
    a.text(rq[order][i]-0.02, yi, f"{rq[order][i]:.3f}", va="center", ha="right", fontsize=9, color="white", fontweight="bold")
a.set_xlim(3.55, 3.82)
a.axvline(rq.max(), color="gray", ls=":", lw=1)
a.set_xlabel("throughput (completed req/s, SLO-agnostic)")
a.set_title(f"① ranked by THROUGHPUT\nspread only {(rq.max()-rq.min())/rq.min()*100:.1f}% — all ≈ saturate GPU", fontsize=10.5, fontweight="bold")
a.grid(alpha=.3, axis="x")

# ── panel 2: goodput ranking (fans out) ──
a = ax[1]
order = np.argsort(-gp)
a.barh(y, gp[order], xerr=gps[order], color="#c1121f", alpha=.85, capsize=3)
a.set_yticks(y); a.set_yticklabels([names[i] for i in order]); a.invert_yaxis()
for i, yi in enumerate(y):
    a.text(gp[order][i]-0.02, yi, f"{gp[order][i]:.3f}", va="center", ha="right", fontsize=9, color="white", fontweight="bold")
a.set_xlim(2.7, 3.28)
a.set_xlabel("goodput (req/s meeting TTFT≤3s ∧ ITL SLO)")
a.set_title(f"② ranked by GOODPUT (SLO-constrained)\nspread {(gp.max()-gp.min())/gp.min()*100:.1f}% — 4× wider than throughput", fontsize=10.5, fontweight="bold")
a.grid(alpha=.3, axis="x")

# ── panel 3: scatter throughput vs goodput, attainment as the differentiator ──
a = ax[2]
a.errorbar(rq, gp, xerr=rqs, yerr=gps, fmt="none", ecolor="gray", alpha=.5, zorder=1)
sc = a.scatter(rq, gp, c=attain, cmap="RdYlGn", s=140, edgecolor="k", lw=.7, zorder=3, vmin=77, vmax=86)
for d, x, yv in zip(D, rq, gp):
    a.annotate(d[0], (x, yv), textcoords="offset points", xytext=(7, 4), fontsize=8.5, fontweight="bold")
# iso-attainment reference lines
for pct in (0.80, 0.85):
    xs = np.array([3.60, 3.82]); a.plot(xs, pct*xs, "--", color="gray", lw=1, alpha=.6)
    a.text(3.81, pct*3.81, f"{int(pct*100)}% attain", fontsize=8, color="gray", ha="right", va="bottom")
a.set_xlabel("throughput (req/s)"); a.set_ylabel("goodput (req/s)")
a.set_title("③ the gap = SLO ATTAINMENT\nvertical spread (goodput) ≫ horizontal (throughput)", fontsize=10.5, fontweight="bold")
a.grid(alpha=.3)
fig.colorbar(sc, ax=a, label="SLO attainment (goodput/throughput %)")

fig.tight_layout(rect=[0, 0.03, 1, 0.9])
fig.text(0.5, 0.008,
         "Throughput is nearly flat (3.4%): every split completes about the same raw req/s. The split decides WHICH requests miss the TTFT wall (attainment 77.9–85.3%), not how many finish. "
         "d16 lowest on BOTH → entanglement leaves a small (~1/4) real throughput footprint; the other ~3/4 of its goodput deficit is pure SLO attainment.",
         ha="center", fontsize=8.8, style="italic")
fig.savefig("throughput_vs_goodput.png", dpi=150)
print("wrote throughput_vs_goodput.png")
