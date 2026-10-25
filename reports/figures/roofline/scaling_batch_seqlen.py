#!/usr/bin/env python3
"""Scaling of ideal-roofline lower-bound time with the two runtime variables: sequence length (prefill L, decode ctx c)
and decode batch b.  time = sum over operators of max(FLOP/PEAK, bytes/BW).  analytic-v1 FLOPs, ideal bytes, A100. [derived]"""
import sys, pathlib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
from matplotlib.colors import LogNorm
ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "workspace/engine-port/scripts/costmodel"))
import analytic_terms as T
PEAK, BW = 312e12, 2.039e12; CA, CM, CF = "#0072B2", "#D55E00", "#777777"
def t_ops(ops): return sum(max(f/PEAK, by/BW) for f, by in ops)
def mk(a):
    be, bs_ = a.beta, a.beta_ssm
    sM = T.ssm_state_bytes_per_layer(a) + T.conv_state_bytes_per_layer(a); kA = 2*a.n_kv*a.d_h*be
    qkvN = a.d_h*(a.n_q + 2*a.n_kv); inN = 2*a.d_in + 2*a.G*a.N + a.n_h
    g = lambda M, K, N: (2*M*K*N, be*(M*K + K*N + M*N))
    def pre(kind, L):
        if kind == "A": return [g(L, a.d, qkvN), g(L, a.n_q*a.d_h, a.d), (T.flops_attn_core_prefill(a, L), be*L*a.d_h*(2*a.n_q + 2*a.n_kv))]
        if kind == "M": return [g(L, a.d, inN), g(L, a.d_in, a.d), (T.flops_mamba_conv_per_token(a)*L, be*2*L*a.conv_dim),
                                (T.flops_ssd_prefill_per_token(a)*L, be*L*(2*a.d_in + 2*a.G*a.N + a.n_h) + 2*(L/a.Q)*a.n_h*a.p*a.N*bs_)]
        return [g(L, a.d, a.d_ff), g(L, a.d_ff, a.d)]
    def dec(kind, b, c):
        if kind == "A": return [g(b, a.d, qkvN), (b*T.flops_attn_core_decode(a, c), b*c*kA), g(b, a.n_q*a.d_h, a.d)]
        if kind == "M": return [g(b, a.d, inN), (b*T.flops_ssm_decode_per_token(a), b*2*sM), g(b, a.d_in, a.d)]
        return [g(b, a.d, a.d_ff), g(b, a.d_ff, a.d)]
    head = lambda n: max(2*a.d*a.V*n/PEAK, a.d*a.V*be/BW)          # LM head: weight read once per forward, FLOPs per sampled token
    def pre_model(L): return a.n_A*t_ops(pre("A", L)) + a.n_M*t_ops(pre("M", L)) + a.n_F*t_ops(pre("F", L)) + head(1)
    def dec_model(b, c): return a.n_A*t_ops(dec("A", b, c)) + a.n_M*t_ops(dec("M", b, c)) + a.n_F*t_ops(dec("F", b, c)) + head(b)
    return pre, dec, pre_model, dec_model
a = T.nano9b_v2(); pre, dec, pre_model, dec_model = mk(a)
Ls = np.logspace(np.log10(256), np.log10(131072), 50); Bs = np.logspace(0, np.log10(512), 50); Cs = np.logspace(2, np.log10(131072), 50)
def sty(ax, xl, yl, ttl): ax.set_xlabel(xl); ax.set_ylabel(yl); ax.set_title(ttl, loc="left", fontsize=10.5); ax.grid(alpha=.2, which="both")
fig, axs = plt.subplots(2, 3, figsize=(17, 9.4), constrained_layout=True)
# ---- prefill row
ax = axs[0, 0]
for k, col, nm in (("A", CA, "Attention"), ("M", CM, "Mamba2"), ("F", CF, "MLP")):
    ax.loglog(Ls, [t_ops(pre(k, L))*1e3 for L in Ls], color=col, lw=2.2, label=nm)
ax.loglog(Ls, [t_ops(pre("A", 256))*1e3*(L/256) for L in Ls], color="#bbb", lw=1, ls=":"); ax.text(300, 6, "linear in L (dotted)", fontsize=8.5, color="#888")
ax.legend(frameon=False); sty(ax, "prefill length L (tokens, one request)", "layer time (ms)", "(a) PREFILL: layer time vs L")
ax = axs[0, 1]
for k, col, nm in (("A", CA, "Attention"), ("M", CM, "Mamba2"), ("F", CF, "MLP")):
    ax.semilogx(Ls, [t_ops(pre(k, L))/L*1e6 for L in Ls], color=col, lw=2.2, label=nm)
ax.legend(frameon=False); sty(ax, "prefill length L", "layer time per token (µs)", "(b) PREFILL: per-token cost — flat = linear, rising = L² term")
ax = axs[0, 2]
for (nA, nM), col in (((4, 27), "#222222"), ((13, 18), "#CC79A7"), ((22, 9), "#E69F00"), ((31, 0), CA)):
    _, _, pm, _ = mk(T.nano9b_v2(nA, nM))
    axs[0, 2].semilogx(Ls, [pm(L)/L*1e6 for L in Ls], color=col, lw=2.2, label=f"{nA}A / {nM}M" + ("  (Nano)" if nA == 4 else ""))
ax.legend(frameon=False, title="composition (25 MLP layers)"); sty(ax, "prefill length L", "whole-model time per token (µs)", "(c) PREFILL: whole model, by attention share")
# ---- decode row
ax = axs[1, 0]
for k, col, nm in (("A", CA, "Attention"), ("M", CM, "Mamba2"), ("F", CF, "MLP")):
    ax.loglog(Bs, [t_ops(dec(k, b, 8192))*1e6 for b in Bs], color=col, lw=2.2, label=nm)
ax.legend(frameon=False); sty(ax, "decode batch b  (ctx 8k)", "layer time per step (µs)", "(d) DECODE: layer step time vs b — flat = weight-read floor")
ax = axs[1, 1]
for k, col, nm in (("A", CA, "Attention"), ("M", CM, "Mamba2"), ("F", CF, "MLP")):
    ax.loglog(Cs, [t_ops(dec(k, 32, c))*1e6 for c in Cs], color=col, lw=2.2, label=nm)
ax.legend(frameon=False); sty(ax, "context length c  (b = 32)", "layer time per step (µs)", "(e) DECODE: layer step time vs context")
ax = axs[1, 2]
Z = np.array([[dec_model(b, c)*1e3 for c in Cs] for b in Bs])
im = ax.pcolormesh(Cs, Bs, Z, norm=LogNorm(), cmap="viridis", shading="auto"); ax.set_xscale("log"); ax.set_yscale("log")
cb = fig.colorbar(im, ax=ax); cb.set_label("whole-model step time (ms)")
d0 = T.decode_step(a, 1, 1000); w = d0["bytes_weights"]; s2 = d0["bytes_state_rw"]; k = T.kv_bytes_per_token(a)
cc = np.array([(w/b - s2)/k for b in Bs]); m = cc > Cs[0]
ax.plot(cc[m], Bs[m], color="white", lw=2, ls="--"); ax.text(300, 110, "weights = state + KV bytes\n(left/below: weight-read dominated)", color="white", fontsize=8.5)
ax.axvline(T.L_star(a)*2, color="#ff9", lw=1.2, ls=":"); ax.text(T.L_star(a)*2*1.06, 1.3, "c* = 17.5k\n(state = KV)", color="#ff9", fontsize=8.5)
ax.set_xlim(Cs[0], Cs[-1]); ax.set_ylim(Bs[0], Bs[-1])
sty(ax, "context length c", "decode batch b", "(f) DECODE: whole-model step time over (b, c)")
fig.suptitle("Scaling with the runtime variables (sequence length, batch) — ideal-roofline lower bound, Nemotron-Nano-9B-v2, A100  [derived]", fontsize=12, x=0.01, ha="left")
out = pathlib.Path(__file__).with_name("scaling_batch_seqlen.png"); fig.savefig(out, dpi=135)
print("prefill model/token µs L=1k,16k,128k (Nano):", [round(pre_model(L)/L*1e6, 1) for L in (1024, 16384, 131072)])
print("a31:", [round(mk(T.nano9b_v2(31, 0))[2](L)/L*1e6, 1) for L in (1024, 16384, 131072)])
print("decode model ms (b,c): ", {(b, c): round(dec_model(b, c)*1e3, 2) for b, c in ((1, 1024), (32, 8192), (256, 8192), (32, 131072), (256, 131072))})
print("layer dec us b=1/512 c=8k:", {k: [round(t_ops(dec(k, b, 8192))*1e6, 1) for b in (1, 512)] for k in "AMF"})
print(out)
