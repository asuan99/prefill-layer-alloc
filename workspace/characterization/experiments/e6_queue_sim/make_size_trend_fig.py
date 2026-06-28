"""Size trend of the layer_aware benefit across zamba2 1.2B / 2.7B / 7B.

la/agnostic goodput ratio (SLO-loose = throughput ratio): 1.2b 1.37x, 2.7b 2.0x (PEAK), 7b 1.06x.
=> the lever is an SLM phenomenon, NON-monotone, peaking at 2.7B; gone at 7B.
1.2b/2.7b from real measured LUTs; 7b from the synth LUT (real E5 ssm-decode failed, OSError).
Output: reports/figures/layer_aware_size_trend.png
"""
from __future__ import annotations
import os, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_here = os.path.dirname(os.path.abspath(__file__))
_CHAR = os.path.abspath(os.path.join(_here, "..", ".."))

# (model, params, co, agnostic, layer_aware, source)
DATA = [
    ("zamba2_1.2b", 1.2, 1302, 1376, 1887, "real"),
    ("zamba2_2.7b", 2.7, 752, 629, 1268, "real"),
    ("zamba2_7b", 7.0, 334, 495, 495, "real"),    # clean LUT (job 797630, TRITON_CACHE_DIR fix)
]
names = [d[0].replace("zamba2_", "") for d in DATA]
ratio = [d[4] / d[3] for d in DATA]              # la / agnostic
COLco, COLag, COLla = "#9aa0a6", "#e8710a", "#1a73e8"

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.4))

# --- left: la/agnostic ratio vs size ---
x = np.arange(len(DATA))
bars = ax1.bar(x, ratio, 0.55,
               color=["#1a73e8", "#1a73e8", "#c0c4cc"],
               edgecolor="k", linewidth=0.6)
ax1.axhline(1.0, color="k", lw=1, ls="--", alpha=0.6)
ax1.text(len(DATA) - 0.5, 1.02, "no benefit (1.0x)", ha="right", va="bottom", fontsize=8, color="k")
for i, (r, d) in enumerate(zip(ratio, DATA)):
    tag = f"{r:.2f}x" + ("" if d[5] == "real" else "\n(synth*)")
    ax1.text(i, r + 0.03, tag, ha="center", va="bottom", fontsize=10, fontweight="bold",
             color="#1a73e8" if d[5] == "real" else "#888")
ax1.annotate("PEAK", xy=(1, ratio[1]), xytext=(1, ratio[1] + 0.35),
             ha="center", fontsize=10, fontweight="bold", color="#1a73e8",
             arrowprops=dict(arrowstyle="->", color="#1a73e8"))
ax1.set_xticks(x); ax1.set_xticklabels([f"{n}\n({d[1]}B)" for n, d in zip(names, DATA)])
ax1.set_ylabel("layer_aware / agnostic  goodput ratio")
ax1.set_title("(A) benefit is an SLM phenomenon, peaks at 2.7B", fontweight="bold")
ax1.set_ylim(0, 2.5)
ax1.grid(axis="y", alpha=0.3)

# --- right: absolute goodput, 3 policies x 3 models ---
w = 0.26
for k, (lab, idx, col) in enumerate([("co_schedule", 2, COLco), ("agnostic", 3, COLag), ("layer_aware", 4, COLla)]):
    vals = [d[idx] for d in DATA]
    ax2.bar(x + (k - 1) * w, vals, w, label=lab, color=col, edgecolor="k", linewidth=0.4)
ax2.set_xticks(x); ax2.set_xticklabels([f"{n}\n({d[1]}B)" for n, d in zip(names, DATA)])
ax2.set_ylabel("goodput @ loose SLO (tok/s)")
ax2.set_title("(B) absolute goodput by policy", fontweight="bold")
ax2.legend(fontsize=9)
ax2.grid(axis="y", alpha=0.3)

fig.suptitle("layer_aware benefit vs model size (zamba2, temporal) -- all real measured LUTs",
             fontsize=12, fontweight="bold", y=1.02)
out = Path(_CHAR).parents[1] / "reports" / "figures"
out.mkdir(parents=True, exist_ok=True)
f = out / "layer_aware_size_trend.png"
fig.savefig(f, dpi=130, bbox_inches="tight")
print(f"wrote {f}")
for n, d in zip(names, DATA):
    print(f"  {n}: co={d[2]} ag={d[3]} la={d[4]} -> la/ag={d[4]/d[3]:.2f}x [{d[5]}]")
