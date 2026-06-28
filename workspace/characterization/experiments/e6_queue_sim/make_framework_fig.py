"""Framework comparison: 4-way decode ITL (fused / co_schedule / agnostic / layer_aware)
across zamba2 1.2/2.7/7B, with the vLLM-measured fused ITL marked as the real anchor.
Shows fused (vLLM default) has the highest decode ITL (decode coupled to prefill); the
*_protect policies bound it. Output: reports/figures/framework_comparison.png
"""
import os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_CHAR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
# decode ITL_p99 (ms), sim, matched workload
POL = ["fused", "co_schedule", "agnostic", "layer_aware"]
COL = {"fused": "#c5221f", "co_schedule": "#9aa0a6", "agnostic": "#e8710a", "layer_aware": "#1a73e8"}
DATA = {  # model -> [fused, co, agnostic, layer_aware]
    "1.2B": [111, 46, 26, 30],
    "2.7B": [179, 80, 38, 44],
    "7B":   [403, 164, 62, 71],
}
VLLM_FUSED = {"2.7B": 109}   # measured p99 TPOT (saturated)

fig, ax = plt.subplots(figsize=(10, 5))
models = list(DATA)
x = np.arange(len(models)); w = 0.2
for k, pol in enumerate(POL):
    vals = [DATA[m][k] for m in models]
    ax.bar(x + (k - 1.5) * w, vals, w, label=pol, color=COL[pol], edgecolor="k", linewidth=0.4)
    for i, v in enumerate(vals):
        ax.text(x[i] + (k - 1.5) * w, v + 4, f"{v}", ha="center", va="bottom", fontsize=7.5)
# vLLM anchor
for m, v in VLLM_FUSED.items():
    i = models.index(m)
    ax.plot([i - 2 * w, i + 2 * w], [v, v], "k--", lw=1.6)
    ax.text(i + 2 * w + 0.02, v, f"vLLM fused\nmeasured {v}ms", va="center", fontsize=8, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels([f"zamba2 {m}" for m in models])
ax.set_ylabel("decode ITL p99 (ms)  — SLO-critical")
ax.set_title("Framework comparison: fused (vLLM) has the highest decode ITL;\n"
             "PD-mux reservation (agnostic/layer_aware) bounds it 4–6x lower",
             fontsize=11.5, fontweight="bold")
ax.legend(fontsize=9.5, title="policy")
ax.grid(axis="y", alpha=0.3)
ax.text(0.5, 0.97, "sim absolute ITL overestimates vLLM ~1.6x for zamba2 (no-GQA); "
        "the fused/layer_aware RATIO (4–6x) is calibration-invariant",
        transform=ax.transAxes, ha="center", va="top", fontsize=8, color="#555",
        bbox=dict(boxstyle="round", fc="#fff8e1", ec="#e0c060"))
out = Path(_CHAR).parents[1] / "reports" / "figures"
out.mkdir(parents=True, exist_ok=True)
p = out / "framework_comparison.png"
fig.savefig(p, dpi=130, bbox_inches="tight")
print(f"wrote {p}")
