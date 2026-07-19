#!/usr/bin/env python
"""MIX-swing trace — does a workload that alternates prefill-heavy <-> decode-heavy let dynamic win?

User's hypothesis: optimum should swing with the prefill/decode MIX (not just load); output length
is unpredictable; so dynamic (which observes decode load as it materializes) should beat static.

Test: phase A = prefill-heavy (in2048/o32, in/out 64), phase B = decode-heavy (in256/o512, in/out 0.5),
alternating; full static sweep d16..d44 + dynamic bind. job 860452-860470, TRUE goodput from jsonl.

Findings:
  1. The optimum SWINGS, but only within a NARROW middle band: phase A -> d24, phase B -> d34.
     NOT d16<->d44. And d16 (most prefill SM) does NOT win the prefill-heavy phase.
  2. WHY: goodput = TTFT-SLO AND ITL-SLO, and the two pull OPPOSITE ways. d16 gives the best TTFT
     (1.47s) but the worst ITL (56ms, brushing the 60ms wall); d44 the reverse (ITL 22ms but TTFT
     3.64s FAILS the 3s wall). A middle split (d24) satisfies BOTH -> wins.
  3. A single middle static (d24) is near-optimal in BOTH phases (A 5.006=best, B 2.854 vs best
     2.870 = 0.6% off). Dynamic bind 3.419 << d24 3.930 -> static still wins.
Caveat: phase B is saturated (thru ~2.9 < offered 5), n=2. A more extreme mix where NO single static
        satisfies both SLOs in both phases is untested -- that is the remaining door for dynamic.
"""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

splits = ["d16", "d24", "d34", "d44"]
x = [16, 24, 34, 44]  # decode SM
# phase A (prefill-heavy): TTFT p50 (s), ITL p50 (ms), goodput
A_ttft = [1.47, 1.82, 2.51, 3.64]
A_itl  = [56.0, 36.1, 27.3, 22.3]
A_gp   = [3.016, 5.006, 3.644, 1.564]
B_gp   = [2.690, 2.854, 2.870, 2.864]
comb   = [2.853, 3.930, 3.257, 2.214]
bind_comb = 3.419
C_T, C_I, C_G = "#c1121f", "#0077b6", "#1a7a3a"

fig, ax = plt.subplots(1, 3, figsize=(16.5, 5.4))
fig.suptitle("MIX-swing trace (prefill-heavy ⇄ decode-heavy) — the optimum swings only in a NARROW middle band, and a single middle static wins\n"
             "phase A = in2048/o32 (prefill-heavy) · phase B = in256/o512 (decode-heavy) · Zamba2-2.7B · job 860452–470 · TRUE goodput",
             fontsize=11.5, fontweight="bold")

# ── panel 1: WHY the prefill-heavy optimum is a MIDDLE split (two SLOs pull opposite) ──
a = ax[0]
a.plot(x, A_ttft, "o-", color=C_T, lw=2.4, ms=8, label="TTFT p50 (wants prefill-heavy →d16)")
a.axhline(3.0, color=C_T, ls=":", lw=1.5); a.text(16.5, 3.05, "TTFT SLO 3s", color=C_T, fontsize=8)
a.set_ylabel("TTFT p50 (s)", color=C_T); a.tick_params(axis="y", labelcolor=C_T)
a.set_ylim(0, 4.3)
a2 = a.twinx()
a2.plot(x, A_itl, "s--", color=C_I, lw=2.4, ms=8, label="ITL p50 (wants decode-heavy →d44)")
a2.axhline(60, color=C_I, ls=":", lw=1.5); a2.text(38, 61, "ITL SLO 60ms", color=C_I, fontsize=8)
a2.set_ylabel("ITL p50 (ms)", color=C_I); a2.tick_params(axis="y", labelcolor=C_I); a2.set_ylim(0, 66)
a.axvspan(21, 27, color=C_G, alpha=.13)
a.text(24, 0.4, "d24 satisfies\nBOTH SLOs\n→ goodput max", color=C_G, fontsize=9, fontweight="bold", ha="center")
a.set_xticks(x); a.set_xticklabels(splits)
a.set_xlabel("split (decode SM)")
a.set_title("① prefill-heavy phase: the two SLOs pull OPPOSITE\nd16 best TTFT/worst ITL · d44 reverse · d24 satisfies both", fontsize=10, fontweight="bold")
a.grid(alpha=.3)

# ── panel 2: per-phase goodput — the optimum swings d24 <-> d34 (narrow) ──
a = ax[1]
a.plot(x, A_gp, "o-", color="#c1121f", lw=2.4, ms=9, label="phase A (prefill-heavy)")
a.plot(x, B_gp, "s-", color="#6a0dad", lw=2.4, ms=9, label="phase B (decode-heavy)")
a.plot(x[np.argmax(A_gp)], max(A_gp), "*", color="#c1121f", ms=22, mec="k", mew=.6, zorder=5)
a.plot(x[np.argmax(B_gp)], max(B_gp), "*", color="#6a0dad", ms=22, mec="k", mew=.6, zorder=5)
a.annotate("A opt = d24", (24, 5.006), textcoords="offset points", xytext=(6, -4), fontsize=9, color="#c1121f", fontweight="bold")
a.annotate("B opt = d34\n(nearly flat, saturated)", (34, 2.870), textcoords="offset points", xytext=(-30, 18), fontsize=9, color="#6a0dad", fontweight="bold")
a.axvspan(24, 34, color="gray", alpha=.12)
a.text(29, 1.6, "optimum swings\nONLY d24↔d34\n(not d16↔d44)", ha="center", fontsize=9, fontweight="bold", color="#444")
a.set_xticks(x); a.set_xticklabels(splits); a.set_xlabel("split (decode SM)")
a.set_ylabel("TRUE goodput (req/s)")
a.set_title("② the optimum swings, but NARROWLY\nd16 does NOT win prefill-heavy; d44 does NOT win decode-heavy", fontsize=10, fontweight="bold")
a.grid(alpha=.3); a.legend(fontsize=9)

# ── panel 3: combined — single middle static beats dynamic ──
a = ax[2]
bars = a.bar(splits + ["bind\n(dynamic)"], comb + [bind_comb],
             color=["#9ab0c0", "#1a7a3a", "#9ab0c0", "#9ab0c0", "#e9a13b"], alpha=.9, width=.62)
for b, v in zip(bars, comb + [bind_comb]):
    a.text(b.get_x()+b.get_width()/2, v+0.03, f"{v:.3f}", ha="center", fontsize=9.5, fontweight="bold")
a.axhline(max(comb), color="#1a7a3a", ls="--", lw=1.3)
a.annotate("d24 (single middle static)\nnear-optimal in BOTH phases\n→ beats dynamic by 13%",
           (1, 3.5), fontsize=9, color="#1a7a3a", fontweight="bold", ha="center")
a.set_ylabel("combined goodput (req/s)"); a.set_ylim(0, 4.4)
a.set_title("③ single middle static (d24) > dynamic (bind)\neven with the mix swinging — static still wins", fontsize=10, fontweight="bold")
a.grid(alpha=.3, axis="y")

fig.tight_layout(rect=[0, 0.03, 1, 0.9])
fig.text(0.5, 0.008,
         "Answer to 'shouldn't dynamic win when the mix swings?': the optimum swings only d24↔d34 because goodput = TTFT-SLO ∧ ITL-SLO and the two pull opposite ways, so a MIDDLE split satisfies both in every regime. "
         "A single static (d24) nails both phases; dynamic has nothing to chase. Caveat: phase B saturated, n=2; a mix extreme enough that no static satisfies both SLOs in both phases is the untested door.",
         ha="center", fontsize=8.4, style="italic")
fig.savefig("mixswing.png", dpi=150)
print("wrote mixswing.png")
