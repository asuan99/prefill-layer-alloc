"""Simple mechanism schematic for layer_aware reservation.
On attn-decode layers the SM-hungry decode reserves most SMs; on ssm-decode layers the decode
saturates with few SMs so the rest are released to prefill.
Output: reports/figures/mechanism_schematic.png
"""
import os
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch

_CHAR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ATT, SSM, PRE = "#e8710a", "#1a73e8", "#34a853"
SM = 108

fig, ax = plt.subplots(figsize=(11, 3.2))

# SM allocation per layer-type (decode reserved | prefill)
rows = [   # (label, decode_floor_SM, decode_color)
    ("attn-decode\nlayer", 94, ATT),
    ("ssm-decode\nlayer", 54, SSM),
]
yh = 0.62
for k, (lab, floor, dc) in enumerate(rows):
    y = 1 - k * 1.4
    ax.add_patch(Rectangle((0, y), floor, yh, color=dc, alpha=0.9))                          # decode reserved
    ax.add_patch(Rectangle((floor, y), SM - floor, yh, color=PRE, alpha=0.55, hatch="//"))    # prefill
    ax.text(floor / 2, y + yh / 2, f"decode\n{floor} SM", ha="center", va="center", color="white",
            fontsize=9.5, fontweight="bold")
    ax.text(floor + (SM - floor) / 2, y + yh / 2, f"prefill\n{SM-floor} SM", ha="center", va="center",
            color="#0a5a2a", fontsize=9.5, fontweight="bold")
    ax.text(-3, y + yh / 2, lab, ha="right", va="center", fontsize=10.5, fontweight="bold")
ax.annotate("", xy=(54, 1.78), xytext=(94, 1.78), arrowprops=dict(arrowstyle="<->", color=PRE, lw=1.6))
ax.text(74, 1.9, "~40 SMs reclaimed for prefill\non every ssm layer", ha="center", color="#0a5a2a",
        fontsize=8.8, fontweight="bold")
ax.set_xlim(-26, SM + 12); ax.set_ylim(-0.55, 2.15)
ax.set_xticks([0, 27, 54, 81, 108]); ax.set_yticks([])
ax.set_xlabel("SMs on the GPU (108 total)")
for sp in ("top", "right", "left"):
    ax.spines[sp].set_visible(False)
ax.legend(handles=[Patch(color=ATT, label="attn-decode (reserved)"),
                   Patch(color=SSM, label="ssm-decode (reserved)"),
                   Patch(facecolor=PRE, alpha=0.55, hatch="//", label="prefill (gets the rest)")],
          loc="lower center", ncol=3, fontsize=9, frameon=False, bbox_to_anchor=(0.42, -0.42))

fig.suptitle("layer_aware reservation: protect SM-hungry attn-decode, release SM-light ssm layers to prefill",
             fontsize=12.5, fontweight="bold", y=1.04)
out = Path(_CHAR).parents[1] / "reports" / "figures"
p = out / "mechanism_schematic.png"
fig.savefig(p, dpi=130, bbox_inches="tight")
print(f"wrote {p}")
