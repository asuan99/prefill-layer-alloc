#!/usr/bin/env python
"""Is decode-attn really 'SM-hungry', or is it memory-bound (so SM shouldn't help)?

User's objection: decode-attn reads the whole KV cache -> memory-bandwidth-bound -> adding SM
shouldn't speed it up, so calling it 'SM-hungry' at long ctx is wrong.

The data settles it EMPIRICALLY: marginal SM efficiency = per-step speedup / SM-ratio.
  = 1.0  -> perfect linear scaling with SM (NOT bandwidth-saturated; more SM still helps)
  -> 0   -> saturated (memory-bound in the roofline sense; more SM wasted)

Finding: at LONG ctx decode-attn scales ~LINEARLY to 108 SM (efficiency ~1.0 even 44->108),
i.e. this triton no-cudagraph kernel is MLP/occupancy-limited, NOT HBM-bandwidth-saturated ->
more SM genuinely helps. At SHORT ctx it saturates early (overhead-bound, tiny work).

CAVEAT (the user's point still lands at the operating point): this is the TRITON, no-cudagraph
microbench. A bandwidth-optimal kernel (FlashDecoding) or the cudagraph operating point may
saturate at far fewer SMs -- and HE2 already found cudagraph decode 'robust / non-binding'.
So decode-attn's SM-scaling here is real but its OPERATING-POINT magnitude is the open question.

Source: deckneectx_result_C*.txt (job 858811).
"""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

# decode-attn per-layer ms vs SM, per ctx
ATT = {256:{108:0.3528,44:0.6220,24:1.0236,16:1.5156,8:2.9512},
       1024:{108:1.0470,44:2.1000,24:3.7856,16:5.5919,8:10.8776},
       4096:{108:3.9699,44:7.9865,24:15.0235,16:22.3627,8:43.3730},
       16384:{108:6.7584,44:16.5087,24:29.1736,16:43.8411,8:85.7418}}
ctxs = sorted(ATT); SMs = [8,16,24,44,108]
cmap = plt.cm.viridis

fig, ax = plt.subplots(1, 2, figsize=(13.6, 5.4))
fig.suptitle("Is decode-attn 'SM-hungry' or memory-bound?  —  in the triton microbench it scales ~LINEARLY with SM at long ctx\n"
             "Zamba2-2.7B decode-attn, no-cudagraph/triton, A100-80GB · job 858811",
             fontsize=12, fontweight="bold")

# panel 1: attn per-layer ms vs SM (log-log) with ideal-linear reference
a = ax[0]
for i, ctx in enumerate(ctxs):
    col = cmap(i/(len(ctxs)-1))
    a.loglog(SMs, [ATT[ctx][s] for s in SMs], "o-", color=col, lw=2.2, ms=7, label=f"ctx={ctx}")
# ideal 1/SM reference anchored at ctx16384, SM=8
ref = ATT[16384][8]
a.loglog(SMs, [ref*8/s for s in SMs], "k--", lw=1.3, alpha=.6, label="ideal 1/SM (linear)")
a.set_xlabel("SM allocated to decode"); a.set_ylabel("decode-attn per-layer time (ms)")
a.set_title("① decode-attn vs SM\nlong-ctx curves parallel the 1/SM ideal → still scaling", fontsize=10.5)
a.grid(alpha=.3, which="both"); a.legend(fontsize=9)

# panel 2: marginal efficiency (speedup / SM-ratio) per SM step
a = ax[1]
labels = [f"{SMs[i-1]}→{SMs[i]}" for i in range(1, len(SMs))]
x = range(len(labels))
for i, ctx in enumerate(ctxs):
    col = cmap(i/(len(ctxs)-1))
    eff = []
    for j in range(1, len(SMs)):
        lo, hi = SMs[j-1], SMs[j]
        eff.append((ATT[ctx][lo]/ATT[ctx][hi])/(hi/lo))
    a.plot(x, eff, "o-", color=col, lw=2.2, ms=8, label=f"ctx={ctx}")
a.axhline(1.0, color="k", ls="--", lw=1.3)
a.axhspan(0.9, 1.05, color="#2a9d8f", alpha=.12)
a.text(0.05, 1.02, "= 1.0 : perfect linear scaling (NOT saturated, more SM helps)", fontsize=8.8, color="#1a7a5a", fontweight="bold")
a.text(2.0, 0.66, "< 1.0 : diminishing\n(short ctx = overhead-bound)", fontsize=8.8, color="#8a5a00", fontweight="bold")
a.set_xticks(list(x)); a.set_xticklabels(labels)
a.set_ylim(0.55, 1.12)
a.set_xlabel("SM step"); a.set_ylabel("marginal efficiency = speedup / SM-ratio")
a.set_title("② ★ efficiency stays ≈1.0 to 108 SM at long ctx\n⇒ MLP/occupancy-limited, NOT bandwidth-saturated", fontsize=10.5, fontweight="bold")
a.grid(alpha=.3); a.legend(fontsize=9)

fig.tight_layout(rect=[0, 0.05, 1, 0.9])
fig.text(0.5, 0.015,
         "So 'SM-hungry' is empirically right for THIS kernel (giving SM raises memory-level parallelism, not yet at the HBM ceiling). "
         "But the user's roofline point lands at the OPERATING POINT: a bandwidth-optimal / cudagraph decode may saturate at far fewer SMs — HE2 found cudagraph decode non-binding. Open question.",
         ha="center", fontsize=8.8, style="italic")
fig.savefig("decode_attn_saturation.png", dpi=150)
print("wrote decode_attn_saturation.png")
