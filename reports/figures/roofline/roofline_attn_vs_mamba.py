#!/usr/bin/env python3
"""Per-operator roofline of Attention vs Mamba2 layers (Nemotron-Nano-9B-v2 dims), A100 SXM.
FLOPs come from workspace/engine-port/scripts/costmodel/analytic_terms.py; bytes are ideal (each tensor moved once)
=> intensities are UPPER bounds for the scan/conv/core kernels.  All [derived], not measurements."""
import sys, pathlib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "workspace/engine-port/scripts/costmodel"))
import analytic_terms as T

a = T.nano9b_v2(); b2, bs = a.beta, a.beta_ssm
PEAK, BW = 312e12, 2.039e12; RIDGE = PEAK / BW
CA, CM = "#0072B2", "#D55E00"        # attention / mamba (Okabe-Ito)

def gemm(M, K, N):  return 2*M*K*N, b2*(M*K + K*N + M*N)

def prefill_ops(L):
    """-> list of (layer, op, flops, bytes)"""
    qkvN = a.d_h*(a.n_q + 2*a.n_kv); inN = 2*a.d_in + 2*a.G*a.N + a.n_h
    nchunk = L / a.Q
    ops = []
    ops.append(("A", "qkv_proj", *gemm(L, a.d, qkvN)))
    ops.append(("A", "attn core", T.flops_attn_core_prefill(a, L), b2*L*a.d_h*(2*a.n_q + 2*a.n_kv)))
    ops.append(("A", "o_proj", *gemm(L, a.n_q*a.d_h, a.d)))
    ops.append(("M", "in_proj", *gemm(L, a.d, inN)))
    ops.append(("M", "conv1d", T.flops_mamba_conv_per_token(a)*L, b2*2*L*a.conv_dim))
    ops.append(("M", "SSD scan", T.flops_ssd_prefill_per_token(a)*L,
                b2*L*(2*a.d_in + 2*a.G*a.N + a.n_h) + 2*nchunk*a.n_h*a.p*a.N*bs))
    ops.append(("M", "out_proj", *gemm(L, a.d_in, a.d)))
    return ops

def decode_ops(bt, c):
    qkvN = a.d_h*(a.n_q + 2*a.n_kv); inN = 2*a.d_in + 2*a.G*a.N + a.n_h
    sM = T.ssm_state_bytes_per_layer(a) + T.conv_state_bytes_per_layer(a)
    return [("A", "qkv_proj", *gemm(bt, a.d, qkvN)),
            ("A", "KV attn", bt*T.flops_attn_core_decode(a, c), bt*c*2*a.n_kv*a.d_h*b2),
            ("A", "o_proj", *gemm(bt, a.n_q*a.d_h, a.d)),
            ("M", "in_proj", *gemm(bt, a.d, inN)),
            ("M", "SSM update", bt*T.flops_ssm_decode_per_token(a), bt*2*sM),
            ("M", "out_proj", *gemm(bt, a.d_in, a.d))]

def layer_AI(ops, layer): 
    f = sum(o[2] for o in ops if o[0] == layer); by = sum(o[3] for o in ops if o[0] == layer); return f/by

fig = plt.figure(figsize=(15, 9.5), constrained_layout=True)
gs = fig.add_gridspec(2, 2, width_ratios=[1.35, 1])
axR = fig.add_subplot(gs[:, 0]); axI = fig.add_subplot(gs[0, 1]); axS = fig.add_subplot(gs[1, 1])

# ---- (a) roofline
x = np.logspace(-1, 4, 400); axR.loglog(x, np.minimum(PEAK, BW*x), color="#333", lw=2)
axR.axvline(RIDGE, color="#999", ls=":", lw=1); axR.text(RIDGE*1.05, 3e11, f"ridge ≈ {RIDGE:.0f}", color="#666", fontsize=9)
axR.text(0.13, BW*0.13*1.6, "HBM 2.04 TB/s", rotation=36, color="#666", fontsize=9)
axR.text(1e4, PEAK*1.12, "bf16 tensor 312 TFLOP/s", color="#666", fontsize=9, ha="right")
axR.annotate("prefill GEMMs (qkv/o/in/out)\n+ attention core (all ≈1.5–2k)", (1800, PEAK), xytext=(60, 5.5e13), fontsize=9, color="#222", arrowprops=dict(arrowstyle="-", color="#888"))
axR.annotate("decode GEMMs, b=32\n(AI ≈ b)", (32, BW*32), xytext=(3, 2.2e13), fontsize=9, color="#222", arrowprops=dict(arrowstyle="-", color="#888"))
L0, B0, C0 = 4096, 32, 8192
mk = {"qkv_proj": "s", "o_proj": "s", "in_proj": "s", "out_proj": "s", "attn core": "^", "KV attn": "^",
      "conv1d": "D", "SSD scan": "o", "SSM update": "o"}
for ops, fill in ((prefill_ops(L0), True), (decode_ops(B0, C0), False)):
    for lay, op, f, by in ops:
        col = CA if lay == "A" else CM; ai = f/by
        perf = min(PEAK, BW*ai)
        axR.scatter(ai, perf, s=110 if fill else 90, marker=mk[op], facecolor=col if fill else "white",
                    edgecolor=col, linewidth=2, zorder=5)
        dx, dy = (1.12, 1.0)
        if op in ("SSD scan", "SSM update", "conv1d", "KV attn"):
            axR.annotate(op, (ai, perf), xytext=(8, 6), textcoords="offset points", fontsize=9, color="#222")
axR.set_xlim(0.1, 1e4); axR.set_ylim(1e10, 6e14)
axR.set_xlabel("Arithmetic intensity (FLOP / byte)"); axR.set_ylabel("Attainable FLOP/s (ideal roof)")
axR.set_title(f"(a) Per-operator roofline, A100.  Filled = prefill L={L0};  hollow = decode b={B0}, ctx={C0}", fontsize=11, loc="left")
from matplotlib.lines import Line2D
axR.legend(handles=[Line2D([], [], marker="s", color=CA, ls="", ms=8, label="Attention layer ops"),
                    Line2D([], [], marker="s", color=CM, ls="", ms=8, label="Mamba2 layer ops"),
                    Line2D([], [], marker="s", color="#555", ls="", ms=8, label="GEMM"),
                    Line2D([], [], marker="^", color="#555", ls="", ms=8, label="attention core"),
                    Line2D([], [], marker="o", color="#555", ls="", ms=8, label="SSD scan / SSM update"),
                    Line2D([], [], marker="D", color="#555", ls="", ms=8, label="conv1d")], loc="lower right", fontsize=9, frameon=False)
axR.grid(alpha=.2, which="both")

# ---- (b) layer-level intensity vs L (prefill) 
Ls = np.logspace(np.log10(256), np.log10(131072), 40)
for lay, col, name in (("A", CA, "Attention layer"), ("M", CM, "Mamba2 layer")):
    axI.loglog(Ls, [layer_AI(prefill_ops(int(L)), lay) for L in Ls], color=col, lw=2.2, label=name + " (prefill)")
Bs = Ls
axI.axhline(RIDGE, color="#999", ls=":"); axI.text(260, RIDGE*1.1, "A100 ridge", color="#666", fontsize=9)
axI.set_xlabel("tokens in the prefill chunk  L"); axI.set_ylabel("layer FLOP/byte")
axI.set_title("(b) Layer-level intensity vs chunk length", fontsize=11, loc="left"); axI.legend(frameon=False, fontsize=9); axI.grid(alpha=.2, which="both")

# ---- (c) FLOP share
def shares(L):
    o = prefill_ops(L)
    return {lay: {op: f/sum(x[2] for x in o if x[0] == lay) for l, op, f, _ in o if l == lay} for lay in "AM"}
cols = {"qkv_proj": "#56B4E9", "o_proj": "#0072B2", "attn core": "#CCDDEE",
        "in_proj": "#D55E00", "out_proj": "#E69F00", "SSD scan": "#CC79A7", "conv1d": "#999999"}
labels, rows = [], []
for L in (4096, 65536):
    s = shares(L)
    for lay, nm in (("A", "Attn"), ("M", "Mamba2")):
        labels.append(f"{nm}\nL={L//1024}k"); rows.append(s[lay])
bottom = np.zeros(len(rows)); y = np.arange(len(rows))[::-1]
for op in ["qkv_proj", "o_proj", "attn core", "in_proj", "out_proj", "SSD scan", "conv1d"]:
    v = np.array([r.get(op, 0) for r in rows]); axS.barh(y, v, left=bottom, color=cols[op], edgecolor="white", linewidth=1.5, label=op)
    for yi, vi, bi in zip(y, v, bottom):
        if vi > 0.06: axS.text(bi + vi/2, yi, f"{vi*100:.0f}%", ha="center", va="center", fontsize=8.5, color="#111")
    bottom += v
axS.set_yticks(y); axS.set_yticklabels(labels); axS.set_xlim(0, 1); axS.set_xlabel("share of layer FLOPs")
axS.set_title("(c) Where the FLOPs go (prefill)", fontsize=11, loc="left"); axS.legend(ncol=4, fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.22))
out = pathlib.Path(__file__).with_name("roofline_attn_vs_mamba.png"); fig.savefig(out, dpi=140)
# ---- numbers for the write-up
print(T.VERSION, "ridge", round(RIDGE, 1))
for tag, ops in ((f"prefill L={L0}", prefill_ops(L0)), (f"prefill L=65536", prefill_ops(65536)), (f"decode b={B0} c={C0}", decode_ops(B0, C0))):
    print(tag)
    for lay, op, f, by in ops: print(f"  {lay} {op:11s} AI={f/by:9.2f} FLOP/B   flops={f:.3e}")
    print("  layer-level AI: A=%.1f M=%.1f" % (layer_AI(ops, "A"), layer_AI(ops, "M")))
for L in (4096, 65536):
    s = shares(L); print(L, {l: {k: round(v, 3) for k, v in d.items()} for l, d in s.items()})
print(out)
