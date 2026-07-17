#!/usr/bin/env python
"""Diff A (cost ratio) vs Diff B (SM-sensitivity ratio) from the (B,L) 2D knee sweep.

Why this plot exists: the founding layer-aware hypothesis was motivated by the observation
that the attn/mamba COST ratio swings enormously with sequence length (Diff A: 0.47x -> 10.1x).
That observation is real and is reproduced in panels 1-2. But a *reallocation* policy needs the
two layer types to respond to SM DIFFERENTLY (Diff B), and panels 3-4 show they do not --
for L >= 8000, where both speed up ~13x from SM 8->108 (Diff B = 0.96-1.04).

TWO HONEST CAVEATS this plot makes visible (they correct an earlier over-claim of
"Diff B ~= 1.0 across the whole grid"):
  1. At the grid's low edge L=2000, Diff B is NOT 1.0 -- it is ~1.35 in 2 of 3 batch cells
     (B=1: 1.38, B=48: 1.34) with B=4 dissenting (0.97). mamba nears a floor at short L and
     stops absorbing SM, while attn keeps scaling.
  2. The grid starts at L=2000, but 98% of the real ShareGPT workload is BELOW that
     (mean 352, p50 204, p95 1042 tok). Diff B was never measured in the regime that
     actually serves -- and the trend is opening, not closing, as L shrinks.
The serving-level refutation of layer-aware (4-model sweep; coordinated per-type impl)
stands on its own measurements; what this plot cannot support is the *mechanistic* claim
"no lever exists" at short L.

Source: knee2d_table.csv (jobs 847690/847711/847897), Zamba2-2.7B, A100-80GB, sglang v0.5.10.
"""
import csv, collections, math
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullLocator, FuncFormatter

def _fmt(v):
    return f"{v/1000:g}k" if v >= 1000 and v % 1000 == 0 else f"{v:g}"

def tick_all(ax, vals, rot=0):
    """Tick every measured anchor (log axes otherwise drop most of them)."""
    ax.xaxis.set_major_locator(FixedLocator(list(vals)))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, p: _fmt(v)))
    if rot:
        plt.setp(ax.get_xticklabels(), rotation=rot, ha="right")

rows = list(csv.DictReader(open("knee2d_table.csv")))
for r in rows:
    for k in ("L", "B", "SM", "per_attn_ms", "per_mamba_ms", "cost_ratio_attn_over_mamba"):
        r[k] = float(r[k])

cell = collections.defaultdict(dict)          # (L,B) -> SM -> (attn_ms, mamba_ms)
for r in rows:
    cell[(r["L"], r["B"])][r["SM"]] = (r["per_attn_ms"], r["per_mamba_ms"])
Ls = sorted({r["L"] for r in rows})
Bs = sorted({r["B"] for r in rows})
C_ATT, C_MAM, C_A, C_B, C_WARN = "#c1121f", "#0077b6", "#c1121f", "#2a9d8f", "#f4a261"
SGPT_MEAN, SGPT_P50, SGPT_P95 = 352, 204, 1042

def diffB(L, B):
    d = cell[(L, B)]
    if 8.0 not in d or 108.0 not in d: return None
    return (d[8.0][0]/d[108.0][0]) / (d[8.0][1]/d[108.0][1])

fig, ax = plt.subplots(2, 2, figsize=(14.4, 10.2))
fig.suptitle("Zamba2-2.7B prefill — the attn/mamba COST ratio swings 21× with L, but their SM SENSITIVITY does not (for L ≥ 8k)\n"
             "(B,L) 2D knee sweep · A100-80GB · sglang v0.5.10 · jobs 847690/847711/847897",
             fontsize=12.5, fontweight="bold")

# ---- panel 1: per-layer time vs L --------------------------------------------
a = ax[0][0]
at = [cell[(L, 1.0)][108.0][0] for L in Ls]
mm = [cell[(L, 1.0)][108.0][1] for L in Ls]
ka = math.log(at[-1]/at[0]) / math.log(Ls[-1]/Ls[0])
km = math.log(mm[-1]/mm[0]) / math.log(Ls[-1]/Ls[0])
a.loglog(Ls, at, "o-", color=C_ATT, lw=2.4, ms=8, label=f"attention  ~ $L^{{{ka:.2f}}}$  (superlinear)")
a.loglog(Ls, mm, "s-", color=C_MAM, lw=2.4, ms=8, label=f"mamba/SSD  ~ $L^{{{km:.2f}}}$  (sublinear)")
for L, v in zip(Ls, at): a.annotate(f"{v:.1f}", (L, v), textcoords="offset points", xytext=(5, 6), fontsize=8, color=C_ATT)
for L, v in zip(Ls, mm): a.annotate(f"{v:.1f}", (L, v), textcoords="offset points", xytext=(5, -12), fontsize=8, color=C_MAM)
a.axvline(3000, color=C_MAM, ls=":", lw=1.6)
a.annotate("crossover ≈ 3k tok\n(below: mamba costs more)", (3000, 1.5), fontsize=8.5, color="#444",
           textcoords="offset points", xytext=(6, 0))
a.set_xlabel("sequence length L (tokens)"); a.set_ylabel("per-layer prefill time (ms)")
a.set_title("① WHY the cost ratio moves: the two types scale differently in L\n(SM=108, B=1)", fontsize=10.5)
a.grid(alpha=.3, which="both"); a.legend(fontsize=9, loc="upper left")
tick_all(a, sorted(set(Ls) | {3000}))

# ---- panel 2: Diff A vs L (the recalled result) ------------------------------
a = ax[0][1]
for B in Bs:
    xs = [L for L in Ls if (L, B) in cell and 108.0 in cell[(L, B)]]
    ys = [cell[(L, B)][108.0][0] / cell[(L, B)][108.0][1] for L in xs]
    if xs: a.plot(xs, ys, "o-", lw=1.9, ms=6, alpha=.85, label=f"B={int(B)}")
a.axhline(1.0, color="k", ls="--", lw=1.2); a.text(2150, 1.06, "equal cost", fontsize=8.5)
a.annotate("mamba ~2× MORE\nexpensive than attn", (2000, 0.5), textcoords="offset points", xytext=(14, -4),
           fontsize=9, color=C_MAM, fontweight="bold")
a.annotate("attn ~10× MORE\nexpensive than mamba", (32000, 10.1), textcoords="offset points", xytext=(-112, -34),
           fontsize=9, color=C_ATT, fontweight="bold")
a.set_xscale("log"); a.set_yscale("log")
a.set_xlabel("sequence length L (tokens)"); a.set_ylabel("Diff A  =  attn / mamba  (per-layer cost)")
a.set_title("② Diff A — the COST ratio: 0.47× → 10.1×  (21× swing)\nREAL — this was the hypothesis's motivation", fontsize=10.5)
a.grid(alpha=.3, which="both"); a.legend(fontsize=9, title="batch", loc="upper left")
tick_all(a, Ls)

# ---- panel 3: SM speedup curves ---------------------------------------------
a = ax[1][0]
for L, mk in zip(Ls, ["o", "s", "^", "D"]):
    d = cell[(L, 1.0)]; sms = sorted(d)
    a.plot(sms, [d[8.0][0]/d[s][0] for s in sms], mk+"-", color=C_ATT, lw=1.8, ms=5.5, alpha=.85,
           label="attention" if L == Ls[0] else None)
    a.plot(sms, [d[8.0][1]/d[s][1] for s in sms], mk+"--", color=C_MAM, lw=1.8, ms=5.5, alpha=.85,
           label="mamba/SSD" if L == Ls[0] else None)
a.plot([8, 108], [1, 13.5], ":", color="gray", lw=1.4, label="ideal linear")
a.annotate("only L=2000 separates:\nmamba stalls at 8.3× while\nattn reaches 11.5× (Diff B=1.38)",
           (108, 8.3), textcoords="offset points", xytext=(-158, 6), fontsize=8.5, color=C_WARN, fontweight="bold",
           arrowprops=dict(arrowstyle="->", color=C_WARN, lw=1.4))
a.set_xlabel("SM count allocated to prefill"); a.set_ylabel("speedup vs SM=8")
a.set_title("③ SM sensitivity: curves coincide for L ≥ 8k\n(B=1; markers = L)", fontsize=10.5)
a.grid(alpha=.3); a.legend(fontsize=9, loc="upper left")
tick_all(a, sorted({s for d in cell.values() for s in d}))

# ---- panel 4: THE CRUX + the honest caveats ---------------------------------
a = a4 = ax[1][1]
xs = Ls
a.plot(xs, [sum(cell[(L, B)][108.0][0]/cell[(L, B)][108.0][1] for B in Bs if (L, B) in cell) /
            sum(1 for B in Bs if (L, B) in cell) for L in xs],
       "o-", color=C_A, lw=2.8, ms=9, label="Diff A — cost ratio  (what was OBSERVED)")
dbm = [sum(v for B in Bs if (L, B) in cell and (v := diffB(L, B))) /
       sum(1 for B in Bs if (L, B) in cell and diffB(L, B)) for L in xs]
a.plot(xs, dbm, "s-", color=C_B, lw=2.8, ms=9, label="Diff B — SM-sensitivity ratio  (what a POLICY NEEDS)")
for L in xs:
    for B in Bs:
        if (L, B) in cell and (v := diffB(L, B)): a.plot(L, v, ".", color=C_B, ms=6, alpha=.55)
a.axhline(1.0, color="k", ls="--", lw=1.2)
a.axhspan(0.9, 1.1, color=C_B, alpha=.10)
a.text(7000, 1.13, "Diff B = 0.96–1.04  →  NO lever", fontsize=9.5, color=C_B, fontweight="bold")
# caveat 1: L=2000 Diff B is not 1
a.annotate("⚠ L=2000: Diff B ≈ 1.35\n(B=1:1.38, B=48:1.34; B=4:0.97)\nNOT 1.0 — lever OPENS as L↓",
           (2000, 1.23), textcoords="offset points", xytext=(24, 44), fontsize=8.8, color=C_WARN, fontweight="bold",
           arrowprops=dict(arrowstyle="->", color=C_WARN, lw=1.5))
# caveat 2: the grid never covers the real workload
a.axvspan(150, 2000, color=C_WARN, alpha=.16)
a.axvline(SGPT_MEAN, color="#8a5a00", ls="-.", lw=1.8)
a.annotate(f"real ShareGPT workload\nmean {SGPT_MEAN} · p50 {SGPT_P50} · p95 {SGPT_P95} tok\n★ 98% of requests live HERE —\noff the grid, never measured",
           (SGPT_MEAN, 4.0), textcoords="offset points", xytext=(6, 6), fontsize=8.8, color="#8a5a00", fontweight="bold")
a.set_xscale("log"); a.set_yscale("log"); a.set_xlim(150, 45000); a.set_ylim(0.35, 16)
a.set_xlabel("sequence length L (tokens)"); a.set_ylabel("ratio (attn : mamba)")
a.set_title("④ THE CRUX — cost differs, sensitivity does not (for L ≥ 8k)\n…but the grid stops right where the real workload starts",
            fontsize=10.5, fontweight="bold")
a.grid(alpha=.3, which="both"); a.legend(fontsize=9, loc="lower right")
tick_all(a, sorted(set(Ls) | {SGPT_P50, SGPT_MEAN, SGPT_P95}), rot=45)

fig.tight_layout(rect=[0, 0.055, 1, 0.935])
fig.text(0.5, 0.030,
         "Both types are compute-bound at long L and absorb SM identically (~13×, 8→108) ⇒ Diff B ≈ 1 ⇒ nothing to reallocate.",
         ha="center", fontsize=9.4, style="italic")
fig.text(0.5, 0.008,
         "⚠ Caveat: at L=2000 Diff B ≈ 1.35, and 98% of real traffic is shorter still — the “no lever” mechanism is established for LONG context, EXTRAPOLATED for short.",
         ha="center", fontsize=9.4, style="italic", color="#8a5a00", fontweight="bold")
fig.savefig("diffA_vs_diffB.png", dpi=150)
print("wrote diffA_vs_diffB.png")

# ---------- companion table ----------
out = ["| L | B | attn ms @108 | mamba ms @108 | **Diff A** (cost) | attn 8→108 | mamba 8→108 | **Diff B** (lever) |",
       "|---|---|---|---|---|---|---|---|"]
for L in Ls:
    for B in Bs:
        if (L, B) not in cell: continue
        d = cell[(L, B)]
        if 8.0 not in d or 108.0 not in d: continue
        a108, m108 = d[108.0]; sa = d[8.0][0]/a108; sm = d[8.0][1]/m108
        flag = " ⚠" if abs(sa/sm - 1) > 0.15 else ""
        out.append(f"| {int(L)} | {int(B)} | {a108:.2f} | {m108:.2f} | **{a108/m108:.2f}×** | {sa:.1f}× | {sm:.1f}× | **{sa/sm:.2f}**{flag} |")
open("diffA_vs_diffB_table.md", "w").write("\n".join(out) + "\n")
print("wrote diffA_vs_diffB_table.md")
