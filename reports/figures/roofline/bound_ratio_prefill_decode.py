#!/usr/bin/env python3
"""Compute-vs-memory 'bound ratio' R = (FLOP/PEAK)/(bytes/BW) = intensity/ridge, Nemotron-Nano-9B-v2 on A100.
R>1 compute-bound, R<1 memory-bound.  analytic-v1 FLOPs, ideal bytes. [derived], not measured."""
import sys, pathlib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "workspace/engine-port/scripts/costmodel"))
import analytic_terms as T
a = T.nano9b_v2(); be, bs_ = a.beta, a.beta_ssm; RIDGE = 312e12/2.039e12
CA, CM = "#0072B2", "#D55E00"
sM = T.ssm_state_bytes_per_layer(a) + T.conv_state_bytes_per_layer(a); kA = 2*a.n_kv*a.d_h*be
qkvN = a.d_h*(a.n_q + 2*a.n_kv); inN = 2*a.d_in + 2*a.G*a.N + a.n_h
def g(M, K, N): return 2*M*K*N, be*(M*K + K*N + M*N)
def pre_ops(L):
    return [("A", "qkv_proj", *g(L, a.d, qkvN)), ("A", "attn core", T.flops_attn_core_prefill(a, L), be*L*a.d_h*(2*a.n_q + 2*a.n_kv)),
            ("A", "o_proj", *g(L, a.n_q*a.d_h, a.d)),
            ("M", "in_proj", *g(L, a.d, inN)), ("M", "conv1d", T.flops_mamba_conv_per_token(a)*L, be*2*L*a.conv_dim),
            ("M", "SSD scan", T.flops_ssd_prefill_per_token(a)*L, be*L*(2*a.d_in + 2*a.G*a.N + a.n_h) + 2*(L/a.Q)*a.n_h*a.p*a.N*bs_),
            ("M", "out_proj", *g(L, a.d_in, a.d))]
def dec_ops(b, c):
    return [("A", "qkv_proj", *g(b, a.d, qkvN)), ("A", "KV attn", b*T.flops_attn_core_decode(a, c), b*c*kA),
            ("A", "o_proj", *g(b, a.n_q*a.d_h, a.d)),
            ("M", "in_proj", *g(b, a.d, inN)), ("M", "SSM update", b*T.flops_ssm_decode_per_token(a), b*2*sM),
            ("M", "out_proj", *g(b, a.d_in, a.d))]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.6), gridspec_kw={"width_ratios": [1.05, 1]}, constrained_layout=True)
def style(ax, xl=(-2.9, 2.6)):
    ax.axvline(0, color="#333", lw=1.6); ax.set_xlim(*xl)
    ticks = [-2, -1, 0, 1, 2]; ax.set_xticks(ticks); ax.set_xticklabels([f"{10.0**t:g}×" for t in ticks])
    ax.axvspan(xl[0], 0, color="#EEF1F5", zorder=0); ax.text(xl[0]+.05, 1.01, "◀ memory-bound", transform=ax.get_xaxis_transform(), fontsize=9.5, color="#444")
    ax.text(xl[1]-.05, 1.01, "compute-bound ▶", transform=ax.get_xaxis_transform(), fontsize=9.5, color="#444", ha="right")
    ax.set_xlabel("R = compute time ÷ memory time  (= intensity ÷ ridge 153)")
# (a) operators
rows = [("PREFILL  L=4k", pre_ops(4096)), ("DECODE  b=32, ctx 8k", dec_ops(32, 8192))]
y = 0; yt, yl = [], []
for title, ops in rows:
    ax1.text(-2.85, y + .15, title, fontsize=10.5, fontweight="bold", color="#222"); y -= 1
    for lay, op, f, by in ops:
        r = (f/by)/RIDGE; col = CA if lay == "A" else CM
        ax1.barh(y, np.log10(r), color=col, height=.62)
        ha, off = ("left", .06) if r >= 1 else ("right", -.06)
        ax1.text(np.log10(r)+off, y, f"{f/by:,.1f}" if f/by < 100 else f"{f/by:,.0f}", va="center", ha=ha, fontsize=8.5, color="#222")
        yt.append(y); yl.append(op); y -= 1
    y -= .6
ax1.set_yticks(yt); ax1.set_yticklabels(yl); ax1.set_ylim(y+.4, .9); style(ax1)
ax1.set_title("(a) Per operator (number beside bar = FLOP/byte)", loc="left", fontsize=11, pad=22)
# (b) layer level across scenarios
sc = [("prefill L=512", "p", 512), ("prefill L=4k", "p", 4096), ("prefill L=64k", "p", 65536),
      ("decode b=1", "d", 1), ("decode b=32", "d", 32), ("decode b=256", "d", 256)]
def lai(ops, k): return sum(o[2] for o in ops if o[0] == k)/sum(o[3] for o in ops if o[0] == k)
yy = 0
for i, (nm, ph, v) in enumerate(sc):
    ops = pre_ops(v) if ph == "p" else dec_ops(v, 8192)
    for j, (k, col) in enumerate((("A", CA), ("M", CM))):
        r = lai(ops, k)/RIDGE; yv = -i*1.0 - j*.36
        ax2.barh(yv, np.log10(r), color=col, height=.32)
        ha, off = ("left", .06) if r >= 1 else ("right", -.06)
        ax2.text(np.log10(r)+off, yv, f"{lai(ops, k):,.1f}" if lai(ops, k) < 100 else f"{lai(ops, k):,.0f}", va="center", ha=ha, fontsize=8.5)
ax2.set_yticks([-i*1.0 - .18 for i in range(len(sc))]); ax2.set_yticklabels([s[0] for s in sc]); ax2.set_ylim(-5.8, .6)
ax2.axhline(-2.6, color="#bbb", lw=1, ls="--"); style(ax2)
ax2.set_title("(b) Whole layer, by scenario (decode at ctx 8k)", loc="left", fontsize=11, pad=22)
from matplotlib.patches import Patch
ax2.legend(handles=[Patch(color=CA, label="Attention"), Patch(color=CM, label="Mamba2")], loc="lower right", frameon=False)
out = pathlib.Path(__file__).with_name("bound_ratio_prefill_decode.png"); fig.savefig(out, dpi=140); print(out)
