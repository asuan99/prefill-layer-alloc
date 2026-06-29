"""Characterization: attn vs ssm execution time BY PHASE (prefill/decode) and by load (batch,
sequence length). Reflects E3 (decode_sweep: latency x batch x context x SM) + E5 (solo_prefill).
Fully measured. Output: reports/figures/char_phase_layertype.png

(A) decode latency vs decode batch (attn steep memory-bound vs ssm flat O(1)-ish); prefill levels.
(B) decode latency vs sequence length (attn O(L) KV-read vs ssm O(1) state).
(C) phase x layer-type matrix at a serving batch: prefill ~symmetric (compute-bound) vs decode
    asymmetric (attn >> ssm).
"""
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_CHAR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ATT, SSM = "#e8710a", "#1a73e8"
M = "zamba2_2.7b"


def load(f):
    df = pd.read_csv(f, skiprows=1); df.columns = [c.split("__")[0] for c in df.columns]
    return df[df.status == "ok"]


e3 = load(f"{_CHAR}/results_v2/e3/decode_sweep_{M}_a100_sxm4_80gb.csv")
e5 = load(f"{_CHAR}/results_v2/e5_sim_b8_opt/serving_coexec_full_{M}_a100_sxm4_80gb.csv")
SM = sorted(e3.sm_count.unique())[-1]                       # 108 = full GPU
pa = e5[e5.prefill_layer == "attn"].solo_prefill_ms.dropna().median()   # ~0.95 (chunk-fixed)
ps = e5[e5.prefill_layer == "ssm"].solo_prefill_ms.dropna().median()    # ~0.90

fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(16, 4.6))

# (A) decode latency vs decode batch, ctx=4096
g = e3[(e3.sm_count == SM) & (e3.context_len == 4096)]
for lt, c in [("attn", ATT), ("ssm", SSM)]:
    s = g[g.layer_type == lt].sort_values("batch")
    axA.plot(s.batch, s.latency_ms, "-o", color=c, lw=2.2, ms=5, label=f"{lt}-decode")
axA.axhline(pa, color=ATT, ls=":", lw=1.5, alpha=0.7); axA.axhline(ps, color=SSM, ls=":", lw=1.5, alpha=0.7)
axA.text(1.1, pa * 1.05, "attn-prefill", color=ATT, fontsize=8)
axA.text(1.1, ps * 0.78, "ssm-prefill", color=SSM, fontsize=8)
axA.set_xscale("log", base=2); axA.set_yscale("log")
axA.set_xlabel("decode batch"); axA.set_ylabel("per-layer latency (ms)")
axA.set_title("(A) decode vs batch — attn↑(memory-bound, KV×B)\nvs ssm flat; prefill chunk-fixed (점선)",
              fontsize=10, fontweight="bold")
axA.legend(fontsize=9); axA.grid(alpha=0.3, which="both")

# (B) decode latency vs sequence length (context), at batch 8 and 64
for b, mk in [(8, "-o"), (64, "--s")]:
    g = e3[(e3.sm_count == SM) & (e3.batch == b)]
    for lt, c in [("attn", ATT), ("ssm", SSM)]:
        s = g[g.layer_type == lt].sort_values("context_len")
        axB.plot(s.context_len, s.latency_ms, mk, color=c, lw=2, ms=6,
                 label=f"{lt}-decode (b={b})")
axB.set_xscale("log", base=2); axB.set_yscale("log")
axB.set_xlabel("sequence length (KV context)"); axB.set_ylabel("attn-decode latency (ms)")
axB.set_title("(B) decode vs seq-len — attn O(L) (KV read)\nvs ssm O(1) (fixed state)",
              fontsize=10, fontweight="bold")
axB.legend(fontsize=8); axB.grid(alpha=0.3, which="both")
axB.text(0.5, 0.06, "attn ~13× across 1k→16k;  ssm flat", transform=axB.transAxes,
         ha="center", fontsize=8.5, color="#555", bbox=dict(boxstyle="round", fc="#fff8e1", ec="#e0c060"))

# (C) phase × layer-type matrix at decode batch = 64
b = 64
g = e3[(e3.sm_count == SM) & (e3.context_len == 4096) & (e3.batch == b)]
da = g[g.layer_type == "attn"].latency_ms.iloc[0]; ds = g[g.layer_type == "ssm"].latency_ms.iloc[0]
cats = ["attn\nprefill", "attn\ndecode", "ssm\nprefill", "ssm\ndecode"]
vals = [pa, da, ps, ds]; cols = [ATT, ATT, SSM, SSM]
hatch = ["//", "", "//", ""]
bars = axC.bar(range(4), vals, color=cols, edgecolor="k", linewidth=0.6)
for bar, h in zip(bars, hatch): bar.set_hatch(h)
for i, v in enumerate(vals): axC.text(i, v + 0.08, f"{v:.2f}", ha="center", fontsize=9, fontweight="bold")
axC.set_xticks(range(4)); axC.set_xticklabels(cats, fontsize=9)
axC.set_ylabel("per-layer latency (ms)")
axC.set_title(f"(C) phase × layer-type @batch={b}\nprefill ~symmetric (compute) · decode asymmetric (memory)",
              fontsize=10, fontweight="bold")
axC.text(0.5, 0.93, "// = prefill (compute-bound)", transform=axC.transAxes, ha="center", fontsize=8, color="#555")
axC.grid(axis="y", alpha=0.3)

fig.suptitle("attn vs ssm execution time by PHASE and LOAD (zamba2_2.7b, A100, measured E3+E5)",
             fontsize=12.5, fontweight="bold", y=1.04)
out = Path(_CHAR).parents[1] / "reports" / "figures"
p = out / "char_phase_layertype.png"
fig.savefig(p, dpi=130, bbox_inches="tight")
print(f"wrote {p}")
