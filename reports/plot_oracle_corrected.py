#!/usr/bin/env python
"""The CORRECTED oracle (user's point): build it from TTFT-pass ⊗ ITL-pass, not from per-static goodput.

My earlier oracle (best-static-per-phase) gave only +2.1% headroom -- it was still restricted to
COUPLED tradeoff points (each single split forces a TTFT/ITL tradeoff). The user is right: decompose
goodput into its TTFT-pass and ITL-pass marginals, and a DECOUPLED oracle -- best-TTFT config AND
best-ITL config at once -- shows much larger headroom.

Phase A (prefill-heavy, BOTH SLOs bind at different splits), from data (job 860497-518):
  split  TTFT-pass  ITL-pass  BOTH(=real gp)
  d16      57.8%      54.2%      36.7%     <- best TTFT, but ITL fails (decode starved)
  d24      49.7%     100.0%      49.7%     <- best single-split compromise
  d44      28.1%     100.0%      28.1%     <- best ITL, but TTFT fails (prefill starved)
  DECOUPLED oracle = d16's TTFT (57.8%) with d24's ITL (100%) = 57.8%  ->  +16% over best static.

BUT that decoupled oracle DEMANDS 92 prefill SM (for d16-TTFT) AND 24 decode SM (for 100% ITL)
= 116 SM > 108 available. The 8-SM deficit is the COUPLING TAX. It is INFEASIBLE on one GPU
(and entanglement couples them further through the shared batch). The +16% is exactly the
DISAGGREGATION headroom -- capturable only by separate prefill/decode device pools, NOT by any
single-GPU split, dynamic or static.

So both are right, they bound different things:
  - single-GPU dynamic ceiling  = +2.1%  (coupled: my earlier oracle) -> dynamic can't win on 1 GPU
  - decoupled/disaggregation ceiling = +16% (this) -> the real headroom lives across devices
"""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

splits = ["d16", "d24", "d34", "d44"]; x = [16, 24, 34, 44]
tp = [57.8, 49.7, 37.5, 28.1]   # TTFT-pass %
ip = [54.2, 100, 100, 100]      # ITL-pass %
both = [36.7, 49.7, 37.5, 28.1] # real goodput %
decoupled = 57.8                # d16 TTFT ⊗ d24 ITL
best_static = 49.7
C_T, C_I, C_B = "#c1121f", "#0077b6", "#1a7a3a"

fig, ax = plt.subplots(1, 3, figsize=(16.5, 5.6))
fig.suptitle("The corrected Oracle — build it from TTFT-pass ⊗ ITL-pass (user's point): headroom is +16%, not +2%\n"
             "…and that +16% is the DISAGGREGATION headroom — it demands 116 SM on a 108-SM GPU, unreachable by any single-GPU split (phase A, job 860497–518)",
             fontsize=11, fontweight="bold")

# ── panel 1: TTFT-pass / ITL-pass / BOTH, and the decoupled oracle ──
a = ax[0]
a.plot(x, tp, "o-", color=C_T, lw=2.4, ms=8, label="TTFT-pass % (prefill-limited)")
a.plot(x, ip, "s-", color=C_I, lw=2.4, ms=8, label="ITL-pass % (decode-limited)")
a.plot(x, both, "D--", color=C_B, lw=2.2, ms=7, label="BOTH = real goodput")
a.axhline(decoupled, color="#8000a0", lw=2.2, ls=":")
a.text(17, decoupled+1.5, f"DECOUPLED oracle = {decoupled:.0f}%  (d16 TTFT ⊗ d24 ITL)", color="#8000a0", fontsize=9, fontweight="bold")
a.annotate("", xy=(24, decoupled), xytext=(24, best_static), arrowprops=dict(arrowstyle="<->", color="#8000a0", lw=2))
a.text(25, 53.5, "+16%\nheadroom", color="#8000a0", fontsize=9, fontweight="bold")
a.plot(24, best_static, "*", color=C_B, ms=20, mec="k", mew=.5, zorder=5)
a.text(30, 46, "best single static d24\n(the coupled compromise)", fontsize=8.5, color=C_B, fontweight="bold")
a.set_xticks(x); a.set_xticklabels(splits); a.set_xlabel("split (decode SM)"); a.set_ylabel("pass rate (%)")
a.set_ylim(20, 108)
a.set_title("① decompose goodput → TTFT-pass ⊗ ITL-pass\nd16 best TTFT, d24+ best ITL; no single split gets both", fontsize=10, fontweight="bold")
a.grid(alpha=.3); a.legend(fontsize=8.3, loc="center right")

# ── panel 2: the SM budget coupling tax ──
a = ax[1]
a.barh([2], [92], color=C_T, alpha=.9, label="prefill SM (for d16-TTFT)")
a.barh([2], [24], left=[92], color=C_I, alpha=.9, label="decode SM (for 100% ITL)")
a.barh([1], [84], color=C_T, alpha=.5)
a.barh([1], [24], left=[84], color=C_I, alpha=.5)
a.axvline(108, color="k", ls="--", lw=2)
a.text(108, 2.9, "GPU budget = 108 SM", ha="center", fontsize=9, fontweight="bold")
a.text(116, 2, "116 SM\n(8 over!)", va="center", fontsize=9, color="#a00", fontweight="bold")
a.text(108, 1, "108 (fits)", va="center", ha="left", fontsize=8.5, color="#555")
a.set_yticks([1, 2]); a.set_yticklabels(["single-GPU\ncompromise\n(d24: 84+24)", "DECOUPLED\noracle demand\n(92+24)"])
a.set_xlim(0, 130); a.set_xlabel("SM demanded")
a.set_title("② WHY single-GPU can't reach it: the coupling TAX\ndecoupled needs 92+24=116 > 108 → deficit 8 SM", fontsize=10, fontweight="bold")
a.legend(fontsize=8.3, loc="lower right"); a.grid(alpha=.3, axis="x")

# ── panel 3: the two ceilings ──
a = ax[2]
labels = ["best single\nstatic (d24)", "single-GPU\nDYNAMIC ceiling\n(coupled)", "DECOUPLED /\nDISAGG ceiling\n(user's oracle)"]
vals = [49.7, 49.7*1.021, 57.8]
cols = [C_B, "#e9a13b", "#8000a0"]
bars = a.bar(labels, vals, color=cols, alpha=.9, width=.6)
for b, v in zip(bars, vals):
    a.text(b.get_x()+b.get_width()/2, v+0.6, f"{v:.1f}%", ha="center", fontsize=10, fontweight="bold")
a.annotate("+2.1%\n(can't win\non 1 GPU)", (1, 51.5), fontsize=8.4, color="#a15c00", fontweight="bold", ha="center")
a.annotate("+16%\n(needs SEPARATE\ndevices)", (2, 59.2), fontsize=8.6, color="#8000a0", fontweight="bold", ha="center")
a.set_ylabel("phase-A goodput (%)"); a.set_ylim(44, 62)
a.set_title("③ two DIFFERENT ceilings\ndynamic (coupled) +2% · decoupling (disagg) +16%", fontsize=10, fontweight="bold")
a.grid(alpha=.3, axis="y")

fig.tight_layout(rect=[0, 0.03, 1, 0.9])
fig.text(0.5, 0.008,
         "The user is right: the TTFT⊗ITL oracle exposes +16% headroom my per-static oracle missed. But that headroom = achieving d16's TTFT AND d24's ITL at once = 116 SM on a 108-SM GPU = the coupling tax. "
         "Single-GPU dynamic is bounded by the COUPLED ceiling (+2%); the +16% is only reachable by DISAGGREGATION (separate prefill/decode pools, each with a full budget). Same conclusion, sharper: the headroom lives across devices, not in a single-GPU split.",
         ha="center", fontsize=8.3, style="italic")
fig.savefig("oracle_corrected.png", dpi=150)
print("wrote oracle_corrected.png")
