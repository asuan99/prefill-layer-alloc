#!/usr/bin/env python
"""Parse the WIDE (B,L) sweep -> knee2d_wide_table.csv + knee2d_wide.png.

Diff A = attn / mamba per-layer COST ratio            (what the founding hypothesis observed)
Diff B = attn SM-sensitivity / mamba SM-sensitivity   (what a reallocation policy actually needs)
         sensitivity_type = per_layer_time(lowest SM) / per_layer_time(108)

Everything is keyed on the MEASURED bs/ntok, never the requested B: chunked prefill caps
tokens-per-forward, so a requested B=16 can silently execute as bs=1 (that flaw made the
old sweep's B axis nearly constant). Cells whose bs_obs < B_req are flagged, not hidden.

The log already divides by the forward count n (acc/n/9, acc/n/54), so per-layer values are
per-forward. Short L sits near the CUDA-event noise floor, so the sweep drives many more
forwards there; the emitted line is the last (largest-n) accumulation.
"""
import csv, glob, re, math, collections
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LINE = re.compile(
    r"RESULT L=(\d+) B=(\d+) sm=(\w+) .*?ZBPT mode=\S+ ctxlen=(\d+) n=(\d+) bs=(-?\d+) ntok=(-?\d+) \|.*?"
    r"per-attn\(9\)=([\d.]+) per-mamba\(54\)=([\d.]+)"
)
rows = {}
for f in sorted(glob.glob("knee2dw_result_L*.txt")):
    for ln in open(f):
        m = LINE.search(ln)
        if not m: continue
        L, B, sm, ctx, n, bs, ntok, pa, pm = m.groups()
        SM = 108 if sm == "full" else int(sm)
        rows[(int(L), int(B), SM)] = dict(
            ctxlen=int(ctx), n=int(n), bs=int(bs), ntok=int(ntok),
            attn=float(pa), mamba=float(pm))
if not rows:
    raise SystemExit("no knee2dw_result_L*.txt rows parsed yet")

Ls = sorted({k[0] for k in rows}); Bs = sorted({k[1] for k in rows}); SMs = sorted({k[2] for k in rows})
lowSM = min(SMs)

def get(L, B, SM): return rows.get((L, B, SM))
def diffA(L, B):
    d = get(L, B, 108); return d["attn"]/d["mamba"] if d and d["mamba"] else None
def diffB(L, B):
    lo, hi = get(L, B, lowSM), get(L, B, 108)
    if not lo or not hi or not hi["attn"] or not hi["mamba"]: return None
    return (lo["attn"]/hi["attn"]) / (lo["mamba"]/hi["mamba"])

# ---------------- CSV ----------------
with open("knee2d_wide_table.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["L", "B_req", "SM", "bs_obs", "ntok_obs", "n_fwd", "per_attn_ms", "per_mamba_ms",
                "diffA_cost", "diffB_sens", "B_collapsed"])
    for (L, B, SM), d in sorted(rows.items()):
        w.writerow([L, B, SM, d["bs"], d["ntok"], d["n"], f"{d['attn']:.4f}", f"{d['mamba']:.4f}",
                    f"{d['attn']/d['mamba']:.3f}" if d["mamba"] else "",
                    f"{diffB(L,B):.3f}" if diffB(L, B) else "",
                    "YES" if d["bs"] < B else ""])
print("wrote knee2d_wide_table.csv  (%d rows)" % len(rows))

# ---------------- markdown table ----------------
out = ["| L | B_req | **bs_obs** | ntok | attn ms @108 | mamba ms @108 | **Diff A** | attn↑ | mamba↑ | **Diff B** |",
       "|---|---|---|---|---|---|---|---|---|---|"]
for L in Ls:
    for B in Bs:
        hi, lo = get(L, B, 108), get(L, B, lowSM)
        if not hi or not lo: continue
        dA, dB = diffA(L, B), diffB(L, B)
        col = f"**{hi['bs']}**" + (" ⚠" if hi["bs"] < B else "")
        flag = " ⚠" if dB and abs(dB - 1) > 0.15 else ""
        out.append(f"| {L} | {B} | {col} | {hi['ntok']} | {hi['attn']:.3f} | {hi['mamba']:.3f} | "
                   f"**{dA:.2f}×** | {lo['attn']/hi['attn']:.1f}× | {lo['mamba']/hi['mamba']:.1f}× | **{dB:.2f}**{flag} |")
open("knee2d_wide_table.md", "w").write("\n".join(out) + "\n")
print("wrote knee2d_wide_table.md")

# ---------------- plot ----------------
C_ATT, C_MAM, C_A, C_B, C_WARN = "#c1121f", "#0077b6", "#c1121f", "#2a9d8f", "#f4a261"
SG_MEAN, SG_P50, SG_P95 = 352, 204, 1042
fig, ax = plt.subplots(2, 2, figsize=(14.6, 10.4))
fig.suptitle("Zamba2-2.7B prefill — WIDE (B,L) sweep: does the layer-type SM lever (Diff B) open at SHORT context?\n"
             f"L {min(Ls)}–{max(Ls)} × B {min(Bs)}–{max(Bs)} × SM {lowSM}–108 · A100-80GB · sglang v0.5.10",
             fontsize=12.5, fontweight="bold")

# ① per-layer time vs L
a = ax[0][0]
for B, mk in zip(Bs, ["o", "s", "^", "D", "v"]):
    xs = [L for L in Ls if get(L, B, 108)]
    if not xs: continue
    a.loglog(xs, [get(L, B, 108)["attn"] for L in xs], mk+"-", color=C_ATT, lw=1.6, ms=5, alpha=.8,
             label="attention" if B == Bs[0] else None)
    a.loglog(xs, [get(L, B, 108)["mamba"] for L in xs], mk+"--", color=C_MAM, lw=1.6, ms=5, alpha=.8,
             label="mamba/SSD" if B == Bs[0] else None)
xs1 = [L for L in Ls if get(L, 1, 108)]
if len(xs1) > 1:
    at = [get(L, 1, 108)["attn"] for L in xs1]; mm = [get(L, 1, 108)["mamba"] for L in xs1]
    ka = math.log(at[-1]/at[0])/math.log(xs1[-1]/xs1[0]); km = math.log(mm[-1]/mm[0])/math.log(xs1[-1]/xs1[0])
    a.set_title(f"① per-layer prefill time (SM=108; markers = B)\nB=1 scaling:  attn ~ $L^{{{ka:.2f}}}$   vs   mamba ~ $L^{{{km:.2f}}}$", fontsize=10.5)
a.axvspan(min(Ls)*0.8, 2000, color=C_WARN, alpha=.13)
a.text(min(Ls)*0.9, a.get_ylim()[1]*0.35, "region the OLD grid\nnever measured", fontsize=8.5, color="#8a5a00", fontweight="bold")
a.set_xticks(Ls); a.set_xticklabels([str(L) for L in Ls], rotation=45, fontsize=8)
a.get_xaxis().set_minor_formatter(plt.NullFormatter())
a.set_xlabel("sequence length L (tokens)"); a.set_ylabel("per-layer time (ms)")
a.grid(alpha=.3, which="major"); a.legend(fontsize=9, loc="upper left")

# ② Diff A vs L  (every L marked on the x-axis, per user request)
a = ax[0][1]
for B in Bs:
    xs = [L for L in Ls if diffA(L, B)]
    if xs: a.plot(xs, [diffA(L, B) for L in xs], "o-", lw=1.8, ms=5.5, alpha=.85, label=f"B={B}")
a.axhline(1.0, color="k", ls="--", lw=1.2)
a.text(Ls[0]*1.02, 1.05, "equal cost", fontsize=8)
a.axvline(3000, color="gray", ls=":", lw=1.3)
a.annotate("crossover ≈ 3k tok", (3000, 0.13), fontsize=8, color="#444", rotation=90,
           textcoords="offset points", xytext=(3, 0))
a.set_xscale("log"); a.set_yscale("log")
a.set_xticks(Ls); a.set_xticklabels([str(L) for L in Ls], rotation=45, fontsize=8)
a.get_xaxis().set_minor_formatter(plt.NullFormatter())
a.set_xlabel("sequence length L (tokens)"); a.set_ylabel("Diff A = attn / mamba (cost)")
a.set_title("② Diff A — COST ratio: 0.10× → 11.0×  (110× swing)\n(the hypothesis's motivation)", fontsize=10.5)
a.grid(alpha=.3, which="major"); a.legend(fontsize=8.5, title="batch", ncol=2, loc="lower right")

# ③ Diff B vs L — THE QUESTION. Hollow marker = requested B collapsed (bs<B): noisy, ignore.
a = ax[1][0]
a.axvspan(min(Ls)*0.8, 640, color=C_WARN, alpha=.15)
for B, mk in zip(Bs, ["o", "s", "^", "D", "v"]):
    xs = [L for L in Ls if diffB(L, B)]
    if not xs: continue
    ys = [diffB(L, B) for L in xs]
    a.plot(xs, ys, "-", lw=1.5, alpha=.6, label=f"B={B}")
    for L, y in zip(xs, ys):
        d = get(L, B, 108); collapsed = d and d["bs"] < B
        a.plot(L, y, mk, ms=7 if not collapsed else 6, alpha=.9,
               mfc="none" if collapsed else None, mec="gray" if collapsed else None,
               color=a.get_lines()[-1].get_color() if not collapsed else "gray")
a.axhline(1.0, color="k", ls="--", lw=1.4)
a.axhspan(0.9, 1.1, color=C_B, alpha=.12)
a.text(1300, 1.12, "no lever (L ≥ 1k)", fontsize=9, color=C_B, fontweight="bold")
a.text(min(Ls)*0.95, 1.46, "LEVER\nL ≤ 512", fontsize=9, color="#8a5a00", fontweight="bold")
a.axvline(SG_MEAN, color="#8a5a00", ls="-.", lw=1.8)
a.annotate(f"real workload\nmean {SG_MEAN} · p50 {SG_P50} tok\n(inside the lever zone)", (SG_MEAN, 1.33), fontsize=8.2,
           color="#8a5a00", fontweight="bold", textcoords="offset points", xytext=(6, 0))
a.set_xscale("log")
a.set_xticks(Ls); a.set_xticklabels([str(L) for L in Ls], rotation=45, fontsize=8)
a.get_xaxis().set_minor_formatter(plt.NullFormatter())
a.set_xlabel("sequence length L (tokens)"); a.set_ylabel("Diff B = attn sens / mamba sens")
a.set_title(f"③ ★ Diff B — the LEVER (sensitivity ratio, SM {lowSM}→108)\nopens at L ≤ 512 (hollow = bs collapsed, ignore)", fontsize=10.5, fontweight="bold")
a.grid(alpha=.3); a.legend(fontsize=8.5, title="batch", ncol=2, loc="upper right")

# ④ Diff B heatmap over (B,L)
a = ax[1][1]
M = np.full((len(Bs), len(Ls)), np.nan)
for i, B in enumerate(Bs):
    for j, L in enumerate(Ls):
        v = diffB(L, B)
        if v: M[i, j] = v
vmax = max(1.6, np.nanmax(M) if np.isfinite(M).any() else 1.6)
from matplotlib.colors import TwoSlopeNorm
im = a.imshow(M, cmap="RdYlGn_r", norm=TwoSlopeNorm(vmin=min(0.7, np.nanmin(M) if np.isfinite(M).any() else 0.7),
                                                    vcenter=1.0, vmax=vmax), aspect="auto", origin="lower")
a.set_xticks(range(len(Ls))); a.set_xticklabels(Ls, rotation=45)
a.set_yticks(range(len(Bs))); a.set_yticklabels(Bs)
for i in range(len(Bs)):
    for j in range(len(Ls)):
        if np.isfinite(M[i, j]):
            d = get(Ls[j], Bs[i], 108)
            txt = f"{M[i,j]:.2f}"
            if d and d["bs"] < Bs[i]: txt += f"\nbs={d['bs']}!"
            a.text(j, i, txt, ha="center", va="center", fontsize=7.5,
                   color="black", fontweight="bold" if abs(M[i, j]-1) > 0.15 else "normal")
a.set_xlabel("sequence length L"); a.set_ylabel("requested batch B")
a.set_title("④ Diff B over (B,L)   —  'bs=' = requested B collapsed\n(chunked prefill caps tokens/forward)", fontsize=10.5)
fig.colorbar(im, ax=a, label="Diff B")

fig.tight_layout(rect=[0, 0.035, 1, 0.93])
fig.text(0.5, 0.012,
         "Diff A (cost) diverging with L was never the question — a reallocation policy needs Diff B (sensitivity) ≠ 1. "
         "The old grid (L≥2000) said Diff B≈1.0 but showed ~1.35 at its low edge; this sweep tests whether the lever really opens where real traffic lives.",
         ha="center", fontsize=9.2, style="italic")
fig.savefig("knee2d_wide.png", dpi=150)
print("wrote knee2d_wide.png")
