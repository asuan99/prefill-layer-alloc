#!/usr/bin/env python3
"""T1 comparison frame: decode step time T_dec(bs, ctx) -- ideal-roofline lower bound (analytic-v1) vs measurement, when it exists.
No T1 data exists yet (PREREG rev3.5 unsealed).  Run without --data: draws the prediction and the registered grid cells.
Run with --data cells.csv (columns: bs,ctx,t_s[,label]) : overlays the measured S_eng medians and adds the observed/ideal panel.
Prediction is a LOWER BOUND [derived]; the form a0 + b0*bs + c0*bs*ctx is the harness prior (analytic_terms.decode_coefficients)."""
import argparse, csv, pathlib, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
HERE = pathlib.Path(__file__).resolve().parent
exec((HERE / "scaling_batch_seqlen.py").read_text().split("a = T.nano9b_v2(); pre")[0])
ap = argparse.ArgumentParser(); ap.add_argument("--data"); ap.add_argument("--out", default=str(HERE / "t1_compare.png")); args = ap.parse_args()
a = T.nano9b_v2(); _, _, _, dec_model = mk(a); co = T.decode_coefficients(a, BW)
prior = lambda bs, c: co["a0_s"] + co["b0_s_per_req"]*bs + co["c0_s_per_req_token"]*bs*c          # harness form (memory-bound bound)
CELLS = [(bs, c) for bs in (8, 32) for c in (2048, 8192, 32768)]                                     # PREREG §1-1 J-A grid, 108v
data = {}
if args.data:
    for r in csv.DictReader(open(args.data)): data[(int(r["bs"]), int(r["ctx"]))] = float(r["t_s"])
n = 3 if data else 2
fig, ax = plt.subplots(1, n, figsize=(5.8*n, 5.2), constrained_layout=True)
cc = {2048: "#56B4E9", 8192: "#0072B2", 32768: "#003A5C"}; bb = {8: "#E69F00", 32: "#D55E00"}
bs_ = np.logspace(0, np.log10(96), 40); cs_ = np.logspace(np.log10(512), np.log10(40000), 40)
for c, col in cc.items():
    ax[0].plot(bs_, [dec_model(b, c)*1e3 for b in bs_], color=col, lw=2, label=f"ctx {c//1024}k  (ideal)")
    ax[0].plot(bs_, [prior(b, c)*1e3 for b in bs_], color=col, lw=1, ls=":")
for bs, c in CELLS:
    ax[0].scatter(bs, dec_model(bs, c)*1e3, s=45, color=cc[c], zorder=5)
    if (bs, c) in data: ax[0].scatter(bs, data[(bs, c)]*1e3, s=90, marker="X", color=cc[c], edgecolor="#111", zorder=6)
ax[0].set_xscale("log"); ax[0].set_xlabel("decode batch bs"); ax[0].set_ylabel("step time (ms)"); ax[0].grid(alpha=.2)
ax[0].set_title("(a) T_dec vs bs — dots = registered T1 cells (X = measured)", loc="left", fontsize=10); ax[0].legend(frameon=False, fontsize=8.5)
for bs, col in bb.items():
    ax[1].plot(cs_, [dec_model(bs, c)*1e3 for c in cs_], color=col, lw=2, label=f"bs {bs}  (ideal)")
    ax[1].plot(cs_, [prior(bs, c)*1e3 for c in cs_], color=col, lw=1, ls=":")
for bs, c in CELLS:
    ax[1].scatter(c, dec_model(bs, c)*1e3, s=45, color=bb[bs], zorder=5)
    if (bs, c) in data: ax[1].scatter(c, data[(bs, c)]*1e3, s=90, marker="X", color=bb[bs], edgecolor="#111", zorder=6)
ax[1].set_xscale("log"); ax[1].set_xlabel("context length c"); ax[1].set_ylabel("step time (ms)"); ax[1].grid(alpha=.2)
ax[1].set_title("(b) T_dec vs ctx (dotted = harness prior a0+b0·bs+c0·bs·c)", loc="left", fontsize=10); ax[1].legend(frameon=False, fontsize=8.5)
if data:
    xs = [k for k in CELLS if k in data]; r = [data[k]/dec_model(*k) for k in xs]
    ax[2].bar(range(len(xs)), r, color=[bb[k[0]] for k in xs]); ax[2].axhline(1, color="#333", lw=1)
    for i, v in enumerate(r): ax[2].text(i, v + .02, f"{v:.2f}×", ha="center", fontsize=9)
    ax[2].set_xticks(range(len(xs))); ax[2].set_xticklabels([f"bs{b}\n{c//1024}k" for b, c in xs]); ax[2].set_ylabel("measured ÷ ideal lower bound")
    ax[2].set_title("(c) Gap factor per cell (1.0 = bandwidth floor reached)", loc="left", fontsize=10); ax[2].grid(alpha=.2, axis="y")
fig.suptitle("T1 decode step: ideal lower bound" + (" vs measurement" if data else " (no measurement yet — prediction only)") + "  | Nano-9B-v2, 108 SM A100, vanilla 108v grid", x=0.01, ha="left", fontsize=11)
fig.savefig(args.out, dpi=135)
print("prior a0/b0/c0:", {"a0_ms": round(co["a0_s"]*1e3, 3), "b0_ms_per_req": round(co["b0_s_per_req"]*1e3, 4), "c0_ns_per_req_token": round(co["c0_s_per_req_token"]*1e9, 2)})
print(f"{'bs':>3} {'ctx':>6} {'ideal ms (per-op max)':>22} {'prior form ms':>14}  bytes shares w/state/KV")
for bs, c in CELLS:
    d = T.decode_step(a, bs, c); print(f"{bs:>3} {c:>6} {dec_model(bs, c)*1e3:>22.2f} {prior(bs, c)*1e3:>14.2f}  {d['bytes_weights']/d['bytes_total']:.2f}/{d['bytes_state_rw']/d['bytes_total']:.2f}/{d['bytes_kv_read']/d['bytes_total']:.2f}")
print(args.out)
