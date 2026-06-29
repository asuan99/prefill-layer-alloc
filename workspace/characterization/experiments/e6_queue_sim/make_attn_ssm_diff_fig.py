"""attn vs ssm difference from measured data (E3 decode_sweep + E5):
(A) decode SM-sensitivity: latency vs SM count -- attn-decode is SM-hungry (keeps benefiting up
    to ~108), ssm-decode saturates early (~54) => protect attn-decode SMs, release ssm SMs.
(B) one-forward latency composition: attn vs ssm x prefill vs decode share (b8 vs b64).
Output: reports/figures/attn_ssm_diff.png
"""
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

_CHAR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ATT, SSM = "#e8710a", "#1a73e8"
M, NA, NS = "zamba2_2.7b", 9, 45


def load(f):
    df = pd.read_csv(f, skiprows=1); df.columns = [c.split("__")[0] for c in df.columns]
    return df[df.status == "ok"]


e3 = load(f"{_CHAR}/results_v2/e3/decode_sweep_{M}_a100_sxm4_80gb.csv")
e5 = load(f"{_CHAR}/results_v2/e5_sim_b8_opt/serving_coexec_full_{M}_a100_sxm4_80gb.csv")

fig, (axA, axB) = plt.subplots(1, 2, figsize=(14, 5))

# (A) decode latency vs SM count -- SM-sensitivity
for b, ls in [(8, "-"), (64, "--")]:
    g = e3[(e3.context_len == 4096) & (e3.batch == b)]
    for lt, c in [("attn", ATT), ("ssm", SSM)]:
        s = g[g.layer_type == lt].sort_values("sm_count")
        axA.plot(s.sm_count, s.latency_ms, ls, color=c, lw=2.2, ms=6, marker="o",
                 label=f"{lt}-decode (b={b})")
axA.set_yscale("log")
axA.set_xlabel("SMs given to the decode kernel"); axA.set_ylabel("decode latency (ms, log)")
axA.set_title("(A) decode SM-sensitivity per layer-type\n"
              "attn-decode SM-hungry (steep) | ssm-decode saturates early (flat)",
              fontsize=11, fontweight="bold")
axA.axvline(54, color=SSM, ls=":", lw=1.3, alpha=0.7)
# short annotations next to their own curves (upper-left = steep attn; lower-right = flat ssm)
axA.annotate("attn: SM-hungry\n=> RESERVE", xy=(40, 9.3), xytext=(17, 17),
             color=ATT, fontsize=9.5, fontweight="bold", ha="left",
             arrowprops=dict(arrowstyle="->", color=ATT))
axA.annotate("ssm: flat after ~54 SM\n=> RELEASE to prefill", xy=(70, 0.55), xytext=(58, 0.30),
             color=SSM, fontsize=9.5, fontweight="bold", ha="left",
             arrowprops=dict(arrowstyle="->", color=SSM))
axA.legend(fontsize=8.5, loc="upper right", ncol=2); axA.grid(alpha=0.3, which="both")

# (B) one-forward latency composition
pa = e5[e5.prefill_layer == "attn"].solo_prefill_ms.median()
ps = e5[e5.prefill_layer == "ssm"].solo_prefill_ms.median()
dec = lambda lt, b: e5[(e5.decode_layer == lt) & (e5.decode_batch == b)].solo_decode_ms.median()
labels = ["attn-prefill", "ssm-prefill", "attn-decode", "ssm-decode"]
cols = [ATT, SSM, ATT, SSM]
hatch = ["//", "//", "", ""]
bs = [8, 64]
x = np.arange(len(bs)); w = 0.5
bottoms = np.zeros(len(bs))
for k, lab in enumerate(labels):
    if lab == "attn-prefill": vals = [NA * pa] * len(bs)
    elif lab == "ssm-prefill": vals = [NS * ps] * len(bs)
    elif lab == "attn-decode": vals = [NA * dec("attn", b) for b in bs]
    else: vals = [NS * dec("ssm", b) for b in bs]
    bar = axB.bar(x, vals, w, bottom=bottoms, color=cols[k], alpha=0.9, edgecolor="white",
                  linewidth=0.8, hatch=hatch[k])
    for i, v in enumerate(vals):
        if v > 4:
            axB.text(i, bottoms[i] + v / 2, f"{v:.0f}\n{v/( [NA*pa+NS*ps+NA*dec('attn',b)+NS*dec('ssm',b) for b in bs][i])*100:.0f}%",
                     ha="center", va="center", fontsize=8, color="white", fontweight="bold")
    bottoms += vals
axB.set_xticks(x); axB.set_xticklabels([f"decode batch = {b}" for b in bs])
axB.set_ylabel("full-forward latency (ms)")
axB.set_title("(B) one-forward latency composition\nssm dominates; attn-decode grows with batch",
              fontsize=11, fontweight="bold")
leg = [Patch(facecolor=ATT, label="attn"), Patch(facecolor=SSM, label="ssm"),
       Patch(facecolor="gray", hatch="//", label="prefill"), Patch(facecolor="gray", label="decode")]
axB.legend(handles=leg, fontsize=8.5, ncol=2, loc="upper left")
axB.grid(axis="y", alpha=0.3)

fig.suptitle("attn vs ssm -- measured difference (zamba2_2.7b, A100): "
             "SM-sensitivity (left) motivates the reservation policy", fontsize=12.5, fontweight="bold", y=1.02)
out = Path(_CHAR).parents[1] / "reports" / "figures"
p = out / "attn_ssm_diff.png"
fig.savefig(p, dpi=130, bbox_inches="tight")
print(f"wrote {p}")
