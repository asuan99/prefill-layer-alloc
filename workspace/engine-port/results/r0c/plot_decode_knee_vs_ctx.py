#!/usr/bin/env python
"""Decode SM-sensitivity vs context length — the whole-decode SM knee shifts with ctx.

User's question: within a decode step, whether attn or mamba dominates depends on ctx, so the
number of SM decode can benefit from should shift with ctx. This confirms it directly.

Mechanism: decode-attn scans the KV cache (time ~ ctx, SM-hungry); decode-mamba is a fixed
recurrent update (time ~ const, SM-free). Model = 9 attn + 54 mamba layers. So a decode step is
mamba-dominated (SM-free) at short ctx and attn-dominated (SM-hungry) at long ctx -> the whole
step's SM-sensitivity, and thus the optimal decode SM, grows with ctx.

This is WHOLE-DECODE SM sizing keyed on ctx (an offline predictor input for the PD split), NOT a
per-layer-type split inside the step (that stays dead by (D) granularity).

Source: deckneectx_result_C*.txt (job 858811), Zamba2-2.7B decode-only, A100-80GB.
NB: decode-mamba per-layer is ~0.2-0.5ms and near the noise floor; the ctx256 mamba@full=1.22
    point is an outlier (launch-overhead noise). The attn trend and the step-composition
    conclusion are robust to it.
"""
import re, glob
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

N_ATT, N_MAM = 9, 54
LINE = re.compile(r"RESULT CTX=(\d+) sm=(\w+) .*?ctxlen=(\d+) n=(\d+) .*?per-attn\(9\)=([\d.]+) per-mamba\(54\)=([\d.]+)")
data = {}   # ctx -> SM -> (attn_ms, mamba_ms)
for f in glob.glob("deckneectx_result_C*.txt"):
    for ln in open(f):
        m = LINE.search(ln)
        if not m: continue
        ctx, sm, cl, n, pa, pm = m.groups()
        SM = 108 if sm == "full" else int(sm)
        data.setdefault(int(ctx), {})[SM] = (float(pa), float(pm))
ctxs = sorted(data); SMs = sorted({s for d in data.values() for s in d}, reverse=True)

def step_total(ctx, SM):
    a, mm = data[ctx][SM]; return N_ATT*a + N_MAM*mm
def attn_frac(ctx, SM):
    a, mm = data[ctx][SM]; return N_ATT*a / (N_ATT*a + N_MAM*mm)

print(f"{'ctx':>6} {'attn@108':>9} {'mam@108':>8} {'step@108':>9} {'attn_frac':>10} {'step 8→108 sens':>16} {'opt SM (knee)':>13}")
knee = {}
for ctx in ctxs:
    s108 = step_total(ctx, 108); s8 = step_total(ctx, 8)
    # knee = smallest SM whose step time is within 10% of full-SM step time
    k = 108
    for SM in sorted(data[ctx]):
        if step_total(ctx, SM) <= s108 * 1.10: k = SM; break
    knee[ctx] = k
    a, mm = data[ctx][108]
    print(f"{ctx:>6} {a:>9.3f} {mm:>8.3f} {s108:>9.1f} {attn_frac(ctx,108)*100:>9.0f}% {s8/s108:>15.1f}x {k:>13}")

C_ATT, C_MAM = "#c1121f", "#0077b6"
cmap = plt.cm.viridis
fig, ax = plt.subplots(1, 3, figsize=(16.2, 5.4))
fig.suptitle("Decode SM-sensitivity vs context length — the whole-decode SM knee SHIFTS with ctx\n"
             "Zamba2-2.7B (9 attn + 54 mamba layers), decode-only, A100-80GB · job 858811",
             fontsize=12.5, fontweight="bold")

# ① step total vs SM, one curve per ctx (knee deepens with ctx)
a = ax[0]
for i, ctx in enumerate(ctxs):
    col = cmap(i/max(1, len(ctxs)-1))
    sms = sorted(data[ctx])
    a.plot(sms, [step_total(ctx, s) for s in sms], "o-", color=col, lw=2.2, ms=7, label=f"ctx={ctx}")
    a.plot(knee[ctx], step_total(ctx, knee[ctx]), "*", color=col, ms=18, mec="k", mew=.6, zorder=5)
a.set_yscale("log")
a.set_xlabel("SM allocated to decode"); a.set_ylabel("decode step time (ms)  [9·attn + 54·mamba]")
a.set_title("① decode step vs SM  (★ = knee, within 10% of full)\nshort ctx: flat (SM-free) · long ctx: steep (SM-hungry)", fontsize=10.3)
a.grid(alpha=.3, which="both"); a.legend(fontsize=9, title="context")

# ② composition: attn fraction of the step vs ctx
a = ax[1]
af = [attn_frac(ctx, 108)*100 for ctx in ctxs]
a.plot(ctxs, af, "o-", color=C_ATT, lw=2.6, ms=9)
a.axhline(50, color="k", ls="--", lw=1.1)
a.fill_between(ctxs, af, 100, color=C_MAM, alpha=.12)
a.fill_between(ctxs, 0, af, color=C_ATT, alpha=.12)
a.text(ctxs[0]*1.1, 12, "mamba-dominated\n(SM-free)", fontsize=9, color=C_MAM, fontweight="bold")
a.text(ctxs[-1]*0.28, 86, "attn-dominated\n(SM-hungry)", fontsize=9, color=C_ATT, fontweight="bold", ha="right")
for ctx, y in zip(ctxs, af): a.annotate(f"{y:.0f}%", (ctx, y), textcoords="offset points", xytext=(4, 8), fontsize=8.5)
a.axvline(352, color="#8a5a00", ls="-.", lw=1.6)
a.annotate("real workload\nmean 352 tok", (352, 60), fontsize=8.2, color="#8a5a00", fontweight="bold",
           textcoords="offset points", xytext=(6, 0))
a.set_xscale("log"); a.set_xticks(ctxs); a.set_xticklabels([str(c) for c in ctxs], rotation=45, fontsize=8)
a.get_xaxis().set_minor_formatter(plt.NullFormatter())
a.set_xlabel("context length (tokens)"); a.set_ylabel("attn share of decode step (%)")
a.set_title("② WHY the knee moves: attn share grows with ctx\n(decode-attn ∝ ctx, decode-mamba ≈ const)", fontsize=10.3)
a.grid(alpha=.3, which="major"); a.set_ylim(0, 100)

# ③ the money plot: whole-decode SM-sensitivity vs ctx
a = ax[2]
sens = [step_total(ctx, 8)/step_total(ctx, 108) for ctx in ctxs]
a.plot(ctxs, sens, "o-", color="#6a0dad", lw=2.8, ms=10)
for ctx, y in zip(ctxs, sens): a.annotate(f"{y:.1f}×", (ctx, y), textcoords="offset points", xytext=(5, 8), fontsize=9, fontweight="bold")
a.axhline(1.0, color="k", ls="--", lw=1.1)
a.text(ctxs[0]*1.1, 1.5, "SM-free\n(give decode FEW SM)", fontsize=9, color=C_MAM, fontweight="bold")
a.text(ctxs[-1]*0.9, sens[-1]*0.72, "SM-hungry\n(give decode MANY SM)", fontsize=9, color=C_ATT, fontweight="bold", ha="right")
a.set_xscale("log"); a.set_yscale("log")
a.set_xticks(ctxs); a.set_xticklabels([str(c) for c in ctxs], rotation=45, fontsize=8)
a.get_xaxis().set_minor_formatter(plt.NullFormatter())
a.set_xlabel("context length (tokens)"); a.set_ylabel("decode step speedup, SM 8 → 108")
a.set_title("③ ★ optimal decode SM grows with ctx\n1.1× (ctx256, flat) → 10.5× (ctx16k, steep)", fontsize=10.3, fontweight="bold")
a.grid(alpha=.3, which="both")

fig.tight_layout(rect=[0, 0.04, 1, 0.92])
fig.text(0.5, 0.012,
         "This is WHOLE-DECODE SM sizing keyed on ctx — realizable as an offline predictor for the PD split (the 'decode floor' in optimal D_sm = max(floor, load) is itself ctx-dependent). "
         "It does NOT resurrect per-layer-type SM split inside the step, which stays dead by (D) granularity.",
         ha="center", fontsize=9, style="italic")
fig.savefig("decode_knee_vs_ctx.png", dpi=150)
print("\nwrote decode_knee_vs_ctx.png")
