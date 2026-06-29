"""Size trend of the layer_aware benefit across zamba2 1.2B / 2.7B / 7B, including fused (vLLM).
(A) layer_aware advantage over each baseline (vs agnostic, vs fused).
(B) absolute goodput @ loose SLO for all 4 policies (fused / co_schedule / agnostic / layer_aware).
All from real measured LUTs. Output: reports/figures/layer_aware_size_trend.png
"""
import os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_CHAR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FUS, CO, AG, LA = "#c5221f", "#9aa0a6", "#e8710a", "#1a73e8"

# (model, params_B, fused, co_schedule, agnostic, layer_aware)  -- goodput @ loose SLO (tok/s)
DATA = [
    ("1.2b", 1.2, 1047, 1301, 1374, 1885),
    ("2.7b", 2.7, 644, 752, 629, 1267),
    ("7b",  7.0, 292, 337, 272, 496),
]
x = np.arange(len(DATA))
fus = [d[2] for d in DATA]; co = [d[3] for d in DATA]; ag = [d[4] for d in DATA]; la = [d[5] for d in DATA]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6))

# --- (A) layer_aware advantage vs each baseline ---
r_ag = [la[i] / ag[i] for i in range(len(DATA))]
r_fu = [la[i] / fus[i] for i in range(len(DATA))]
w = 0.34
b1 = ax1.bar(x - w / 2, r_ag, w, label="vs agnostic", color=AG, edgecolor="k", linewidth=0.5)
b2 = ax1.bar(x + w / 2, r_fu, w, label="vs fused (vLLM)", color=FUS, edgecolor="k", linewidth=0.5)
for i in range(len(DATA)):
    ax1.text(i - w / 2, r_ag[i] + 0.03, f"{r_ag[i]:.2f}x", ha="center", va="bottom", fontsize=9, fontweight="bold", color=AG)
    ax1.text(i + w / 2, r_fu[i] + 0.03, f"{r_fu[i]:.2f}x", ha="center", va="bottom", fontsize=9, fontweight="bold", color=FUS)
ax1.axhline(1.0, color="k", ls="--", lw=0.9, alpha=0.6)
ax1.text(len(DATA) - 0.5, 1.03, "no benefit (1.0x)", ha="right", va="bottom", fontsize=8)
ax1.set_xticks(x); ax1.set_xticklabels([f"zamba2 {d[0]}\n({d[1]}B)" for d in DATA])
ax1.set_ylabel("layer_aware goodput / baseline")
ax1.set_title("(A) layer_aware advantage over baselines\n(@ loose SLO; at tight SLO fused fails -> ratio grows)",
              fontsize=10.5, fontweight="bold")
ax1.set_ylim(0, 2.5)
ax1.legend(fontsize=9, loc="upper right")
ax1.grid(axis="y", alpha=0.3)

# --- (B) absolute goodput, 4 policies x 3 models ---
w = 0.2
for k, (lab, vals, col) in enumerate([("fused (vLLM)", fus, FUS), ("co_schedule", co, CO),
                                       ("agnostic", ag, AG), ("layer_aware", la, LA)]):
    ax2.bar(x + (k - 1.5) * w, vals, w, label=lab, color=col, edgecolor="k", linewidth=0.4)
ax2.set_xticks(x); ax2.set_xticklabels([f"zamba2 {d[0]}\n({d[1]}B)" for d in DATA])
ax2.set_ylabel("goodput @ loose SLO (tok/s)")
ax2.set_title("(B) absolute goodput by policy\n(fused=vLLM lowest; layer_aware highest)",
              fontsize=10.5, fontweight="bold")
ax2.legend(fontsize=8.5, ncol=2, loc="upper right")
ax2.grid(axis="y", alpha=0.3)
ax2.set_ylim(0, max(la) * 1.18)

fig.suptitle("layer_aware benefit vs model size (zamba2, temporal) -- all real measured LUTs, 4 policies",
             fontsize=12.5, fontweight="bold", y=1.02)
out = Path(_CHAR).parents[1] / "reports" / "figures"
p = out / "layer_aware_size_trend.png"
fig.savefig(p, dpi=130, bbox_inches="tight")
print(f"wrote {p}")
for d in DATA:
    print(f"  {d[0]}: fused {d[2]} co {d[3]} agnostic {d[4]} layer_aware {d[5]}  "
          f"(la/ag {d[5]/d[4]:.2f}x, la/fused {d[5]/d[2]:.2f}x)")
