#!/usr/bin/env python
"""per-layer-type SM allocation — the kill chain, visualized.

A per-layer-type policy needs BOTH:
  (C1) a lever: Diff B = attn-SM-sensitivity / mamba-SM-sensitivity  != 1
  (C2) exploitable: the (D) granularity execution cost < the lever's value
Each panel shows WHERE one of these breaks. The regimes where C1 holds are exactly the
regimes where C2 fails; their intersection (a usable policy) is empty.

Data (all committed):
  prefill Diff B vs L : results/prefill_knee/knee2d_wide_table.csv (jobs 857371/857477)
  decode per-type     : results/r0c/deckneectx_result_C16384* (job 858811)
  (D) exploitation cost: results/a_substrate/A_substrate_isolation_results.md (jobs 837520/1)
"""
import csv, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
C_ATT, C_MAM, C_LEV, C_NO, C_COST = "#c1121f", "#0077b6", "#2a9d8f", "#f4a261", "#6a0dad"

# --- data ---
rows = {(int(r["L"]), int(r["B_req"]), int(r["SM"])): r
        for r in csv.DictReader(open(os.path.join(RES, "prefill_knee", "knee2d_wide_table.csv")))}
Ls = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
pf_diffB = []
for L in Ls:
    hi, lo = rows[(L, 1, 108)], rows[(L, 1, 8)]
    a = float(lo["per_attn_ms"])/float(hi["per_attn_ms"]); m = float(lo["per_mamba_ms"])/float(hi["per_mamba_ms"])
    pf_diffB.append(a/m)
# decode per-type sensitivity (ctx16384): SM -> (attn_ms, mamba_ms)
DEC = {108:(6.7584,0.2959),44:(16.5087,0.2092),24:(29.1736,0.2913),16:(43.8411,0.3717),8:(85.7418,0.6549)}
SMs = [8,16,24,44,108]

fig, ax = plt.subplots(2, 2, figsize=(15, 10.4))
fig.suptitle("Where per-layer-type SM allocation BREAKS — the two conditions and their failure points\n"
             "needs (C1) a lever: Diff B ≠ 1   AND   (C2) exploitable: (D) cost < lever value.   The regimes where C1 holds are exactly where C2 fails.",
             fontsize=12.5, fontweight="bold")

# ── Panel 1: (C1) prefill lever vanishes with L ──────────────────────────────
a = ax[0][0]
a.plot(Ls, pf_diffB, "o-", color=C_LEV, lw=2.6, ms=9, zorder=5)
a.axhspan(0.9, 1.1, color=C_NO, alpha=.18)
a.axhline(1.0, color="k", ls="--", lw=1.2)
a.axvspan(Ls[0]*0.85, 640, color=C_LEV, alpha=.10)
for L, v in zip(Ls, pf_diffB):
    a.annotate(f"{v:.2f}", (L, v), textcoords="offset points", xytext=(0, 10), fontsize=8.5, ha="center",
               fontweight="bold" if v > 1.15 else "normal")
a.annotate("★ BREAK POINT (C1):\nlever collapses into the\n'no-lever' band at L≈1024",
           (1024, 0.91), fontsize=9.5, color="#a15c00", fontweight="bold",
           textcoords="offset points", xytext=(30, -6),
           arrowprops=dict(arrowstyle="->", color="#a15c00", lw=1.8))
a.text(300, 1.34, "lever\nexists", fontsize=9, color=C_LEV, fontweight="bold", ha="center")
a.text(6000, 1.02, "NO lever (both compute-bound, ~13× each)", fontsize=9, color="#7a5a2a", fontweight="bold")
a.axvline(352, color="#8a5a00", ls="-.", lw=1.6)
a.annotate("real workload\nmean 352 tok", (352, 1.28), fontsize=8, color="#8a5a00",
           fontweight="bold", textcoords="offset points", xytext=(5, 0))
a.set_xscale("log"); a.set_xticks(Ls); a.set_xticklabels(Ls, rotation=45, fontsize=8)
a.get_xaxis().set_minor_formatter(plt.NullFormatter())
a.set_xlabel("prefill sequence length L"); a.set_ylabel("Diff B (attn : mamba SM-sensitivity)")
a.set_title("① (C1) PREFILL — lever exists only at L ≤ 512\n→ C1 BREAKS for L ≥ 1024 (attempt D, E)", fontsize=10.5, fontweight="bold")
a.grid(alpha=.3, which="major")

# ── Panel 2: (C1) decode lever exists (mamba SM-free) ────────────────────────
a = ax[0][1]
sa = [DEC[s][0]/DEC[108][0] for s in SMs]; sm = [DEC[s][1]/DEC[108][1] for s in SMs]
# plot slowdown vs SM=108 (so higher = more SM-sensitive); invert to speedup-from-8 style
a.plot(SMs, [DEC[s][0] for s in SMs], "o-", color=C_ATT, lw=2.5, ms=8, label="attention (SM-hungry, 12.7×)")
a.plot(SMs, [DEC[s][1] for s in SMs], "s-", color=C_MAM, lw=2.5, ms=8, label="mamba/SSD (≈SM-free, 2.2×)")
a.set_yscale("log")
a.annotate("mamba nearly FLAT\n→ Diff B ≈ 5.7\n(C1 HOLDS in decode)", (44, 0.24), fontsize=9.5,
           color=C_MAM, fontweight="bold", textcoords="offset points", xytext=(6, 30))
a.text(60, 30, "attn steep\n(scans KV)", fontsize=9, color=C_ATT, fontweight="bold")
a.set_xlabel("SM allocated to decode"); a.set_ylabel("decode per-layer time (ms), ctx=16384")
a.set_title("② (C1) DECODE — lever DOES exist (Diff B ≈ 5.7)\n→ C1 holds; so the break must be C2 (attempt A, C, F, H)", fontsize=10.5, fontweight="bold")
a.grid(alpha=.3, which="both"); a.legend(fontsize=9)

# ── Panel 3: (C2) exploitation cost — even cheap switching can't reach agnostic ──
a = ax[1][0]
labels = ["coordinated\n(host sync ×19)", "OPT\n(cheap switch,\nGPU order)", "agnostic\n(uniform,\nno split)"]
tpot = [124, 85, 43]
cols = [C_COST, "#b07fd0", C_LEV]
bars = a.bar(labels, tpot, color=cols, alpha=.9, width=.6)
for b, v in zip(bars, tpot):
    a.text(b.get_x()+b.get_width()/2, v+2, f"{v}ms", ha="center", fontsize=11, fontweight="bold")
a.axhline(43, color=C_LEV, ls="--", lw=1.4)
# arrows showing the decomposition
a.annotate("", xy=(1, 85), xytext=(0, 124), arrowprops=dict(arrowstyle="->", color="green", lw=2.2))
a.text(0.5, 108, "cheap switch\nrecovers ~half\n(−39ms)", fontsize=9, color="green", fontweight="bold", ha="center")
a.annotate("", xy=(2, 43), xytext=(1, 85), arrowprops=dict(arrowstyle="->", color="red", lw=2.2))
a.text(1.5, 66, "★ BREAK (C2):\nresidual −42ms\nWON'T close\n(overlap loss +\ncudagraph ✗)", fontsize=8.8, color="red", fontweight="bold", ha="center")
a.set_ylabel("decode TPOT p50 (ms), in3600/o32 rate 2")
a.set_title("③ (C2) EXPLOITATION COST — the (D) wall\ncheap switching halves it but can't reach uniform (attempt C, H)", fontsize=10.5, fontweight="bold")
a.set_ylim(0, 140); a.grid(alpha=.3, axis="y")

# ── Panel 4: kill map — C1 ∩ C2 = ∅ ──────────────────────────────────────────
a = ax[1][1]
a.axis("off")
regimes = ["decode\n(any ctx)", "prefill\nL ≥ 1024", "prefill\nL ≤ 512"]
c1 = ["✓  Diff B ≈ 4–6", "✗  Diff B ≈ 1.0", "✓  Diff B 1.2–1.4"]
c2 = ["✗  (D) 42→124ms\n(cheap switch: 85 > 43)", "—  (no lever to\nexploit)", "✗  stakes sub-ms/layer\n≪ (D) 42→124ms"]
verdict = ["DEAD\n(A·C·F·G·H)", "DEAD\n(D)", "DEAD\n(E)"]
ncol = 4
xw = [0.16, 0.30, 0.34, 0.20]; xs = [sum(xw[:i]) for i in range(ncol)]
yr = [0.66, 0.42, 0.18]; rh = 0.20
head = ["regime", "(C1) lever?", "(C2) exploitable?", "verdict"]
for j, h in enumerate(head):
    a.text(xs[j]+xw[j]/2, 0.90, h, ha="center", va="center", fontsize=10.5, fontweight="bold")
a.plot([0, 1], [0.84, 0.84], color="k", lw=1)
def cell_color(txt):
    if txt.startswith("✓"): return (0.85, 0.95, 0.87)
    if txt.startswith("✗"): return (0.98, 0.86, 0.83)
    return (0.92, 0.92, 0.92)
for i in range(3):
    a.text(xs[0]+xw[0]/2, yr[i]+rh/2, regimes[i], ha="center", va="center", fontsize=9.5, fontweight="bold")
    for j, txt in [(1, c1[i]), (2, c2[i])]:
        a.add_patch(plt.Rectangle((xs[j], yr[i]), xw[j], rh, facecolor=cell_color(txt), edgecolor="white", lw=2))
        a.text(xs[j]+xw[j]/2, yr[i]+rh/2, txt, ha="center", va="center", fontsize=8.3, fontweight="bold")
    a.add_patch(plt.Rectangle((xs[3], yr[i]), xw[3], rh, facecolor=(0.80, 0.24, 0.24), edgecolor="white", lw=2))
    a.text(xs[3]+xw[3]/2, yr[i]+rh/2, verdict[i], ha="center", va="center", fontsize=9, fontweight="bold", color="white")
a.text(0.5, 0.045, "★ C1 holds ⟺ C2 fails, in every regime.  Their intersection (lever AND exploitable) is EMPTY ⇒ per-layer-type is dead in all forms.",
       ha="center", va="center", fontsize=9.5, fontweight="bold", color="#6a0dad")
a.set_xlim(0, 1); a.set_ylim(0, 1)
a.set_title("④ THE KILL MAP — where each attempt dies (C1 ∩ C2 = ∅)", fontsize=10.5, fontweight="bold")

fig.tight_layout(rect=[0, 0.02, 1, 0.93])
fig.text(0.5, 0.008,
         "Surviving degenerate form: use layer-type composition only to size the WHOLE-phase SM floor offline (single partition, no within-step split → no (D)); e.g. decode floor = f(ctx).",
         ha="center", fontsize=9, style="italic")
fig.savefig(os.path.join(HERE, "killchain.png"), dpi=150)
print("wrote killchain.png")
