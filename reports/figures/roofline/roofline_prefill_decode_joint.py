#!/usr/bin/env python3
"""One roofline: prefill (sweep L) and decode (sweep b) trajectories of Attention and Mamba2 layers, Nemotron-Nano-9B-v2, A100.
FLOPs from analytic_terms.py (analytic-v1); bytes = ideal single-pass (upper-bound intensity). [derived], not measured."""
import sys, pathlib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "workspace/engine-port/scripts/costmodel"))
import analytic_terms as T
a = T.nano9b_v2(); be, bs_ = a.beta, a.beta_ssm; PEAK, BW = 312e12, 2.039e12; RIDGE = PEAK/BW
CA, CM = "#0072B2", "#D55E00"
sM = T.ssm_state_bytes_per_layer(a) + T.conv_state_bytes_per_layer(a); kA = 2*a.n_kv*a.d_h*be
qkvN = a.d_h*(a.n_q + 2*a.n_kv); inN = 2*a.d_in + 2*a.G*a.N + a.n_h
def g(M, K, N): return 2*M*K*N, be*(M*K + K*N + M*N)
def prefill(kind, L):
    if kind == "A":
        p = [g(L, a.d, qkvN), g(L, a.n_q*a.d_h, a.d), (T.flops_attn_core_prefill(a, L), be*L*a.d_h*(2*a.n_q + 2*a.n_kv))]
    else:
        p = [g(L, a.d, inN), g(L, a.d_in, a.d), (T.flops_mamba_conv_per_token(a)*L, be*2*L*a.conv_dim),
             (T.flops_ssd_prefill_per_token(a)*L, be*L*(2*a.d_in + 2*a.G*a.N + a.n_h) + 2*(L/a.Q)*a.n_h*a.p*a.N*bs_)]
    return sum(x[0] for x in p), sum(x[1] for x in p)
def decode(kind, b, c):
    if kind == "A":
        return b*(2*T.params_attn(a) + T.flops_attn_core_decode(a, c)), be*T.params_attn(a) + b*c*kA
    return (b*(2*T.params_mamba_matmul(a) + T.flops_mamba_conv_per_token(a) + T.flops_ssm_decode_per_token(a)),
            be*T.params_mamba_matmul(a) + b*2*sM)
def pt(fb): f, by = fb; ai = f/by; return ai, min(PEAK, BW*ai)

fig, ax = plt.subplots(figsize=(10.5, 7.2), constrained_layout=True)
x = np.logspace(-0.5, 5, 500); ax.loglog(x, np.minimum(PEAK, BW*x), color="#333", lw=2)
ax.axvline(RIDGE, color="#999", ls=":", lw=1); ax.text(RIDGE*1.06, 2.2e11, f"ridge ≈ {RIDGE:.0f}", color="#666", fontsize=9)
ax.text(1e5, PEAK*1.1, "bf16 tensor 312 TFLOP/s", ha="right", color="#666", fontsize=9)
Ls = np.logspace(np.log10(128), np.log10(131072), 60); Bs = np.logspace(0, 3, 60)
for k, col, nm in (("A", CA, "Attention"), ("M", CM, "Mamba2")):
    P = np.array([pt(prefill(k, int(L))) for L in Ls]); ax.loglog(P[:, 0], P[:, 1], color=col, lw=2.4)
    D = np.array([pt(decode(k, b, 8192)) for b in Bs]); ax.loglog(D[:, 0], D[:, 1], color=col, lw=2.4, ls="--")
    for L in (1024, 4096, 16384, 65536):
        ai, pf = pt(prefill(k, L)); ax.scatter(ai, pf, s=55, color=col, zorder=5)
    for b in (1, 8, 32, 256):
        ai, pf = pt(decode(k, b, 8192)); ax.scatter(ai, pf, s=55, facecolor="white", edgecolor=col, linewidth=2, zorder=5)
# attention decode band over context
for c, ls in ((1024, ":"), (65536, ":")):
    D = np.array([pt(decode("A", b, c)) for b in Bs]); ax.loglog(D[:, 0], D[:, 1], color=CA, lw=1, ls=ls, alpha=.8)
ax.text(3.0, 4.3e12, "attention decode band:\nc=1k (upper) … c=64k (lower)", color=CA, fontsize=8.5)
def lab(txt, xy, off, col): ax.annotate(txt, xy, xytext=off, textcoords="offset points", fontsize=9, color=col)
lab("L=1k", pt(prefill("A", 1024)), (-34, 8), CA); lab("L=4k", pt(prefill("A", 4096)), (6, -14), CA); lab("L=64k", pt(prefill("A", 65536)), (-8, -16), CA)
lab("L=1k", pt(prefill("M", 1024)), (-4, -18), CM); lab("L=64k", pt(prefill("M", 65536)), (-10, 10), CM)
lab("b=1", pt(decode("A", 1, 8192)), (-30, 4), CA); lab("b=32", pt(decode("A", 32, 8192)), (-8, 10), CA); lab("b=256", pt(decode("A", 256, 8192)), (8, 2), CA)
lab("b=1", pt(decode("M", 1, 8192)), (6, -12), CM); lab("b=32", pt(decode("M", 32, 8192)), (-4, 10), CM); lab("b=256", pt(decode("M", 256, 8192)), (8, -12), CM)
from matplotlib.lines import Line2D
ax.legend(handles=[Line2D([], [], color=CA, lw=2.4, label="Attention layer"), Line2D([], [], color=CM, lw=2.4, label="Mamba2 layer"),
                   Line2D([], [], color="#555", lw=2.4, label="prefill: sweep L  (filled dots)"),
                   Line2D([], [], color="#555", lw=2.4, ls="--", label="decode: sweep b, ctx 8k  (hollow dots)")], loc="lower right", frameon=False, fontsize=9.5)
ax.set_xlim(0.3, 1e5); ax.set_ylim(1e10, 7e14); ax.grid(alpha=.2, which="both")
ax.set_xlabel("Layer arithmetic intensity (FLOP / byte)"); ax.set_ylabel("Attainable FLOP/s (ideal roof)")
ax.set_title("Prefill vs decode on one roofline, A100  (layer level; analytic-v1, ideal bytes)", loc="left", fontsize=11.5)
out = pathlib.Path(__file__).with_name("roofline_prefill_decode_joint.png"); fig.savefig(out, dpi=140)
for k in "AM":
    print(k, "prefill L=128/1k/64k/128k AI", [round(pt(prefill(k, L))[0]) for L in (128, 1024, 65536, 131072)],
          "| L where AI=ridge", next(int(L) for L in Ls if pt(prefill(k, int(L)))[0] >= RIDGE))
print(out)
