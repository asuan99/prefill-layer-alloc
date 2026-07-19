#!/usr/bin/env python
"""EXTREME mix-swing — the disjoint-feasibility region the user predicted DOES exist, yet dynamic still loses.

User's argument: there MUST be a workload where no single static satisfies both SLOs in both phases
(disjoint feasible sets), and there dynamic should win. I engineered it (phase B given LONG context
so decode-attn makes ITL SLO-binding, unlike the earlier flat phase B).

Result:
  1. Median-feasibility IS disjoint: feasible-A(prefill-heavy)={d16}, feasible-B(decode-heavy)={d24,d34},
     intersection empty -> no static passes both phases on the median. User's logic CONFIRMED.
  2. Yet on graded GOODPUT the per-phase optima are ADJACENT (A->d24, B->d34), and d24 is near-optimal
     in BOTH -> even an ORACLE dynamic beats the best static by only +2.1%; the reactive controller
     LOSES by -20.6%.
  3. WHY: the conjunctive SLO (TTFT AND ITL) -- the very tension that motivates dynamic (d16 best TTFT,
     d44 best ITL) -- also PULLS each phase's optimum to a middle compromise (d16 can't win phase A
     because its ITL hits 60ms; d44 can't win phase B because its TTFT hits 3s). Same mechanism that
     seems to create dynamic's territory squeezes it shut.
  4. REALISM: this region only appears in OVERLOAD (all policies goodput 0.99-1.87, most requests fail).
     Below capacity everything passes (static trivially works); above it everything fails (static loses
     least). No operating regime gives dynamic a useful win.

Source: he2_bench EXTREME (A=in2048/o32@8, B=in2048/o512@5), job 860497-860518, TRUE goodput.
"""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

splits = ["d16", "d24", "d34", "d44"]; x = [16, 24, 34, 44]
A_ttft = [2.83, 3.02, 3.75, 4.67]; A_itl = [60, 36, 27, 22]
B_ttft = [3.21, 2.08, 2.36, 3.23]; B_itl = [55, 50, 45, 45]
A_gp = [2.03, 2.73, 1.86, 1.23]; B_gp = [0.28, 1.01, 1.09, 0.74]
comb = [(a+b)/2 for a, b in zip(A_gp, B_gp)]
bind_comb = 1.485
oracle = (max(A_gp)+max(B_gp))/2

fig, ax = plt.subplots(1, 3, figsize=(16.5, 5.5))
fig.suptitle("EXTREME mix-swing — the disjoint-feasibility region EXISTS (user's logic confirmed), yet dynamic STILL loses\n"
             "A=in2048/o32@8 (prefill-heavy) · B=in2048/o512@5 (decode-heavy, long ctx → ITL-binding) · job 860497–518 · TRUE goodput",
             fontsize=11.5, fontweight="bold")

# ── panel 1: disjoint median-feasibility ──
a = ax[0]
# show feasibility as pass/fail grid
passA = [(t <= 3 and i <= 60) for t, i in zip(A_ttft, A_itl)]
passB = [(t <= 3 and i <= 60) for t, i in zip(B_ttft, B_itl)]
a.axis("off")
a.text(0.5, 0.99, "median-SLO feasibility (TTFT≤3s ∧ ITL≤60ms)", ha="center", fontsize=9.5, style="italic", transform=a.transAxes)
for j, s in enumerate(splits):
    a.text(0.29+j*0.16, 0.88, s, ha="center", fontsize=10, fontweight="bold")
a.text(0.11, 0.68, "phase A\n(prefill-heavy)", ha="center", fontsize=9.5, fontweight="bold", color="#c1121f")
a.text(0.11, 0.40, "phase B\n(decode-heavy)", ha="center", fontsize=9.5, fontweight="bold", color="#6a0dad")
for j, (pa, pb) in enumerate(zip(passA, passB)):
    a.add_patch(plt.Rectangle((0.22+j*0.16, 0.60), 0.14, 0.16, facecolor="#8fd19e" if pa else "#e88", ec="white", lw=2))
    a.text(0.29+j*0.16, 0.68, "PASS" if pa else "fail", ha="center", va="center", fontsize=8.5, fontweight="bold")
    a.add_patch(plt.Rectangle((0.22+j*0.16, 0.32), 0.14, 0.16, facecolor="#8fd19e" if pb else "#e88", ec="white", lw=2))
    a.text(0.29+j*0.16, 0.40, "PASS" if pb else "fail", ha="center", va="center", fontsize=8.5, fontweight="bold")
a.text(0.5, 0.12, "feasible-A = {d16}   ∩   feasible-B = {d24, d34}   =   ∅\n★ DISJOINT — no static passes BOTH phases",
       ha="center", fontsize=10.5, fontweight="bold", color="#6a0dad", transform=a.transAxes,
       bbox=dict(boxstyle="round", fc="#f0e8f8", ec="#6a0dad"))
a.set_title("① the disjoint region EXISTS (user's logic ✓)", fontsize=10.5, fontweight="bold", pad=20)

# ── panel 2: yet graded goodput optima are ADJACENT, single static near-optimal in both ──
a = ax[1]
a.plot(x, A_gp, "o-", color="#c1121f", lw=2.4, ms=9, label="phase A goodput")
a.plot(x, B_gp, "s-", color="#6a0dad", lw=2.4, ms=9, label="phase B goodput")
a.plot(24, 2.73, "*", color="#c1121f", ms=20, mec="k", mew=.5, zorder=5)
a.plot(34, 1.09, "*", color="#6a0dad", ms=20, mec="k", mew=.5, zorder=5)
a.axvspan(24, 34, color="gray", alpha=.12)
a.annotate("A opt d24", (24, 2.73), textcoords="offset points", xytext=(4, 6), fontsize=8.5, color="#c1121f", fontweight="bold")
a.annotate("B opt d34", (34, 1.09), textcoords="offset points", xytext=(4, 8), fontsize=8.5, color="#6a0dad", fontweight="bold")
a.axvline(24, color="#1a7a3a", ls="--", lw=1.3)
a.text(24.5, 2.3, "d24 near-optimal\nin BOTH phases", fontsize=8.8, color="#1a7a3a", fontweight="bold")
a.set_xticks(x); a.set_xticklabels(splits); a.set_xlabel("split (decode SM)")
a.set_ylabel("TRUE goodput (req/s)")
a.set_title("② graded goodput: optima ADJACENT (d24↔d34)\ndisjoint median ≠ separated goodput optima", fontsize=10.3, fontweight="bold")
a.grid(alpha=.3); a.legend(fontsize=8.5)

# ── panel 3: even oracle dynamic barely wins; reactive loses ──
a = ax[2]
labels = ["best static\n(d24)", "ORACLE dyn\n(perfect\nper-phase)", "reactive dyn\n(bind)"]
vals = [comb[1], oracle, bind_comb]
cols = ["#1a7a3a", "#e9a13b", "#c1121f"]
bars = a.bar(labels, vals, color=cols, alpha=.9, width=.6)
for b, v in zip(bars, vals):
    a.text(b.get_x()+b.get_width()/2, v+0.02, f"{v:.3f}", ha="center", fontsize=10, fontweight="bold")
a.axhline(comb[1], color="#1a7a3a", ls="--", lw=1.2)
a.annotate("oracle beats static\nby only +2.1%\n(optima adjacent)", (1, oracle+0.08), fontsize=8.6, color="#a15c00", fontweight="bold", ha="center")
a.annotate("reactive LOSES\n−20.6%\n(mispositions,\nfails both)", (2, bind_comb-0.42), fontsize=8.6, color="#c1121f", fontweight="bold", ha="center")
a.set_ylabel("combined goodput (req/s)"); a.set_ylim(0, 2.3)
a.set_title("③ even ORACLE dynamic gains ~2% — reactive loses 21%\n(all policies fail: overload regime, gp 0.99–1.87)", fontsize=10.3, fontweight="bold")
a.grid(alpha=.3, axis="y")

fig.tight_layout(rect=[0, 0.04, 1, 0.9])
fig.text(0.5, 0.01,
         "The conjunctive SLO (TTFT∧ITL) that MOTIVATES dynamic (d16 best TTFT, d44 best ITL) also DEFEATS it: it pulls each phase's optimum to a middle compromise (d16 can't win A — ITL wall; d44 can't win B — TTFT wall), "
         "so different phases' optima stay adjacent (d24↔d34) and one middle static serves both. And this region only exists in OVERLOAD, where everything fails and static loses least. No regime gives dynamic a useful win.",
         ha="center", fontsize=8.3, style="italic")
fig.savefig("extreme_disjoint.png", dpi=150)
print("wrote extreme_disjoint.png")
