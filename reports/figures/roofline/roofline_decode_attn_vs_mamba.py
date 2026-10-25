#!/usr/bin/env python3
"""Decode-step behaviour of Attention vs Mamba2 layers (Nemotron-Nano-9B-v2, A100). FLOPs/bytes from analytic_terms.py
(analytic-v1); layer bytes = weights (read once per step) + KV read / state read+write.  [derived], not measured."""
import sys, pathlib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "workspace/engine-port/scripts/costmodel"))
import analytic_terms as T
a = T.nano9b_v2(); be = a.beta; BW = 2.039e12; RIDGE = 312e12/BW
CA, CM = "#0072B2", "#D55E00"
sM = T.ssm_state_bytes_per_layer(a) + T.conv_state_bytes_per_layer(a); kA = 2*a.n_kv*a.d_h*be
def lay(kind, b, c):
    if kind == "A":
        f = b*(2*T.params_attn(a) + T.flops_attn_core_decode(a, c)); w = be*T.params_attn(a); st = b*c*kA
    else:
        f = b*(2*T.params_mamba_matmul(a) + T.flops_mamba_conv_per_token(a) + T.flops_ssm_decode_per_token(a)); w = be*T.params_mamba_matmul(a); st = b*2*sM
    return f, w, st
fig, ax = plt.subplots(1, 3, figsize=(16, 5.2), constrained_layout=True)
bs = np.logspace(0, 3, 60)
for k, col, nm in (("A", CA, "Attention layer"), ("M", CM, "Mamba2 layer")):
    ax[0].loglog(bs, [(lambda f, w, s: f/(w+s))(*lay(k, b, 8192)) for b in bs], color=col, lw=2.2, label=nm)
ax[0].axhline(RIDGE, color="#999", ls=":"); ax[0].text(1.1, RIDGE*1.08, "A100 ridge ≈153", color="#666", fontsize=9)
ax[0].set_xlabel("decode batch b (ctx c = 8k)"); ax[0].set_ylabel("layer FLOP/byte"); ax[0].legend(frameon=False)
ax[0].set_title("(a) Batching raises intensity, then saturates far below ridge", fontsize=10.5, loc="left"); ax[0].grid(alpha=.2, which="both")
cs = np.logspace(2, np.log10(131072), 60)
for k, col, nm in (("A", CA, "Attention layer"), ("M", CM, "Mamba2 layer")):
    ax[1].loglog(cs, [(lambda f, w, s: f/(w+s))(*lay(k, 32, c)) for c in cs], color=col, lw=2.2, label=nm)
ax[1].axhline(RIDGE, color="#999", ls=":"); ax[1].set_ylim(0.8, 300)
ax[1].set_xlabel("context length c (batch b = 32)"); ax[1].set_ylabel("layer FLOP/byte"); ax[1].legend(frameon=False)
ax[1].set_title("(b) Attention intensity falls with context; Mamba2's is flat", fontsize=10.5, loc="left"); ax[1].grid(alpha=.2, which="both")
# (c) whole-model bytes per step, b=32
parts = {"weights": [], "SSM/conv state r+w": [], "KV read": []}
for c in cs:
    d = T.decode_step(a, 32, int(c)); parts["weights"].append(d["bytes_weights"]); parts["SSM/conv state r+w"].append(d["bytes_state_rw"]); parts["KV read"].append(d["bytes_kv_read"])
tot = np.array([sum(v) for v in zip(*parts.values())])
bot = np.zeros(len(cs))
for (nm, v), col in zip(parts.items(), ("#999999", CM, CA)):
    v = np.array(v)/tot; ax[2].fill_between(cs, bot, bot+v, color=col, label=nm, lw=0); bot += v
ax[2].set_xscale("log"); ax[2].set_xlim(cs[0], cs[-1]); ax[2].set_ylim(0, 1)
ax[2].axvline(T.decode_step(a, 32, 8192)["c_state_eq_kv"]*1.0, color="white", ls="--", lw=1.5)
ax[2].text(T.decode_step(a, 32, 8192)["c_state_eq_kv"]*1.08, 0.04, "c* ≈ 17.5k\n(state = KV)", color="white", fontsize=9)
ax[2].set_xlabel("context length c (batch b = 32)"); ax[2].set_ylabel("share of HBM bytes / decode step")
ax[2].set_title("(c) Whole model (4A/27M/25F): who the bytes belong to", fontsize=10.5, loc="left"); ax[2].legend(frameon=True, facecolor="white", framealpha=0.95, loc="lower left", fontsize=9)
out = pathlib.Path(__file__).with_name("roofline_decode_attn_vs_mamba.png"); fig.savefig(out, dpi=140)
for b in (1, 32, 256, 1024):
    print("b", b, {k: round((lambda f, w, s: f/(w+s))(*lay(k, b, 8192)), 2) for k in "AM"})
print("b->inf c=8k:", {k: round(lay(k, 1e9, 8192)[0]/lay(k, 1e9, 8192)[2], 2) for k in "AM"})
for c in (1024, 8192, 65536): print("c", c, {k: round((lambda f, w, s: f/(w+s))(*lay(k, 32, c)), 2) for k in "AM"})
for c in (1024, 8192, 32768, 131072):
    d = T.decode_step(a, 32, c); print("c", c, {k: round(d[k]/d["bytes_total"], 3) for k in ("bytes_weights", "bytes_state_rw", "bytes_kv_read")}, "AI", round(d["intensity_flop_per_byte"], 1))
print(out)
