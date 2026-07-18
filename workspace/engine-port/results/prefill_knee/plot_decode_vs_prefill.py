#!/usr/bin/env python
"""Decode vs Prefill SM-sensitivity by layer type — they are OPPOSITE.

Answers "does decode SM-sensitivity behave like prefill?": NO.
  PREFILL: every layer processes L tokens in parallel. Both attn (O(L^2)) and mamba/SSD
           (O(L)) do real compute -> both compute-bound -> both ~equally SM-hungry
           -> Diff B ~= 1.0 (no layer-type lever).  [wide sweep, this dir]
  DECODE : every layer processes 1 token/seq. mamba/SSD decode is a tiny recurrent state
           update -> memory-bound, ~SM-free (saturates in a few SM). attn decode must still
           attend over the whole KV cache -> SM-hungry -> Diff B ~= 4x (a real lever).
           [R0c decode knee, ctx3600]

So the founding layer-aware hypothesis was BORN here (decode mamba is SM-free!) and the
lever is genuinely real -- but it DIED anyway, because a per-layer-type SM split inside a
single fused decode cudagraph step is exactly the (D) granularity problem (sub-step green-ctx
repartition fragments the step; cudagraph is incompatible). Coordinated per-type impl: TPOT 42->124ms.

Sources: DECODE = results/r0c/knee_result_835571.txt (ctx3600).
         PREFILL = results/prefill_knee/knee2d_wide_table.csv (B=1).
"""
import csv
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

# DECODE knee (ctx3600, single decode step): SM -> (attn_ms, mamba_ms) per layer
DEC = {108: (3.1389, 0.3605), 44: (6.2678, 0.2654), 24: (11.7668, 0.3976),
       16: (17.5125, 0.5309), 8: (34.0082, 0.9682)}

# PREFILL (wide sweep, B=1) at a representative long L where prefill is fully compute-bound
rows = {(int(r["L"]), int(r["B_req"]), int(r["SM"])): r
        for r in csv.DictReader(open("knee2d_wide_table.csv"))}
def pf(L, SM):
    r = rows.get((L, 1, SM)); return (float(r["per_attn_ms"]), float(r["per_mamba_ms"])) if r else None
L_PF = 8192
SMs = [8, 16, 24, 44, 108]
C_ATT, C_MAM = "#c1121f", "#0077b6"

fig, ax = plt.subplots(1, 3, figsize=(16, 5.2))
fig.suptitle("SM-sensitivity by layer type — DECODE and PREFILL are OPPOSITE  (Zamba2-2.7B, A100-80GB)",
             fontsize=13, fontweight="bold")

def sens(d, SM): return d[SM] / d[108]

# panel 1: decode — normalized sensitivity (per-layer time / time@108)
a = ax[0]
da = {sm: DEC[sm][0] for sm in SMs}; dm = {sm: DEC[sm][1] for sm in SMs}
a.plot(SMs, [sens(da, s) for s in SMs], "o-", color=C_ATT, lw=2.4, ms=8, label="attention  (SM-hungry)")
a.plot(SMs, [sens(dm, s) for s in SMs], "s-", color=C_MAM, lw=2.4, ms=8, label="mamba/SSD  (≈ SM-free)")
a.axhline(1, color="k", ls=":", lw=1)
a.annotate("mamba nearly FLAT:\n0.97→0.36ms over 8→108\n(2.7× vs attn 10.8×)", (44, 2.7),
           fontsize=9, color=C_MAM, fontweight="bold", textcoords="offset points", xytext=(10, 20))
a.set_xlabel("SM allocated"); a.set_ylabel("slowdown vs SM=108")
a.set_title("① DECODE (1 tok/seq)\nDiff B = 4.0×  →  lever EXISTS", fontsize=11, fontweight="bold")
a.grid(alpha=.3); a.legend(fontsize=9.5)

# panel 2: prefill — same axes
a = ax[1]
pa = {sm: pf(L_PF, sm)[0] for sm in SMs}; pm = {sm: pf(L_PF, sm)[1] for sm in SMs}
a.plot(SMs, [sens(pa, s) for s in SMs], "o-", color=C_ATT, lw=2.4, ms=8, label="attention")
a.plot(SMs, [sens(pm, s) for s in SMs], "s-", color=C_MAM, lw=2.4, ms=8, label="mamba/SSD")
a.axhline(1, color="k", ls=":", lw=1)
a.annotate("curves COINCIDE:\nboth ~13× over 8→108\n(both compute-bound)", (24, 6),
           fontsize=9, color="#444", fontweight="bold", textcoords="offset points", xytext=(12, 6))
a.set_xlabel("SM allocated"); a.set_ylabel("slowdown vs SM=108")
a.set_title(f"② PREFILL (L={L_PF} tok, {SMs[0]}→108)\nDiff B = 1.0×  →  NO lever", fontsize=11, fontweight="bold")
a.grid(alpha=.3); a.legend(fontsize=9.5)

# panel 3: the punchline — Diff B bar
a = ax[2]
dec_dB = sens(da, 8) / sens(dm, 8)
pf_dB = {L: (pf(L, 8)[0]/pf(L, 108)[0]) / (pf(L, 8)[1]/pf(L, 108)[1]) for L in [256, 1024, 8192, 32768]}
labels = ["DECODE\n(any ctx)"] + [f"PREFILL\nL={L}" for L in pf_dB]
vals = [dec_dB] + list(pf_dB.values())
cols = ["#6a0dad"] + ["#2a9d8f"]*len(pf_dB)
bars = a.bar(labels, vals, color=cols, alpha=.85)
a.axhline(1.0, color="k", ls="--", lw=1.3)
a.axhspan(0.9, 1.1, color="#2a9d8f", alpha=.12)
for b, v in zip(bars, vals):
    a.text(b.get_x()+b.get_width()/2, v+0.08, f"{v:.1f}×" if v >= 2 else f"{v:.2f}", ha="center", fontsize=10, fontweight="bold")
a.text(0, dec_dB*0.5, "mamba\nSM-free\n= lever", ha="center", fontsize=9, color="white", fontweight="bold")
a.set_ylabel("Diff B = attn sens / mamba sens")
a.set_title("③ THE LEVER lives in DECODE, not prefill\n(but a fused cudagraph step can't exploit it → (D))", fontsize=11, fontweight="bold")
a.grid(alpha=.3, axis="y")

fig.tight_layout(rect=[0, 0.03, 1, 0.93])
fig.text(0.5, 0.01,
         "Mechanism: decode does 1 token/layer so mamba's recurrent update is memory-bound & SM-free while attn still scans the KV cache; "
         "prefill does L tokens/layer so BOTH types are compute-bound. The layer-type lever is real in decode — and unusable, because per-type SM split inside one fused decode step is the (D) granularity cost (TPOT 42→124ms).",
         ha="center", fontsize=9, style="italic")
fig.savefig("decode_vs_prefill_sensitivity.png", dpi=150)
print("wrote decode_vs_prefill_sensitivity.png")
print(f"DECODE Diff B = {dec_dB:.2f}   PREFILL Diff B = {pf_dB}")
