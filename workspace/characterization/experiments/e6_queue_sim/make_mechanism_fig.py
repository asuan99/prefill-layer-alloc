"""Simple mechanism schematic for layer_aware reservation.
A forward = few expensive attn-decode layers + many cheap ssm-decode layers. On attn layers the
SM-hungry decode reserves most SMs; on ssm layers the decode saturates with few SMs so the rest
are released to prefill. Output: reports/figures/mechanism_schematic.png
"""
import os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch

_CHAR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ATT, SSM, PRE = "#e8710a", "#1a73e8", "#34a853"
SM = 108

fig, (axtop, ax) = plt.subplots(2, 1, figsize=(11, 5.4), gridspec_kw={"height_ratios": [1, 3.2], "hspace": 0.45})

# --- top: the forward as a layer sequence (9 attn + 45 ssm) ---
NA, NS, L = 9, 45, 54
attn_pos = set(np.linspace(2, L - 2, NA).round().astype(int).tolist())
for i in range(L):
    axtop.add_patch(Rectangle((i, 0), 0.92, 1, color=ATT if i in attn_pos else SSM,
                              alpha=0.9 if i in attn_pos else 0.75))
axtop.set_xlim(-0.5, L + 8); axtop.set_ylim(0, 1)
axtop.axis("off")
axtop.set_title("one forward pass = 9 expensive attn-decode layers (orange) + 45 cheap ssm-decode layers (blue)",
                fontsize=10.5, fontweight="bold")
axtop.annotate("", xy=(L, -0.05), xytext=(0, -0.05), arrowprops=dict(arrowstyle="->", color="k"))
axtop.text(L / 2, -0.55, "depth ->", ha="center", fontsize=8)

# --- bottom: SM allocation per layer-type (decode reserved | prefill) ---
rows = [   # (label, decode_floor_SM, decode_color, note)
    ("attn-decode\nlayer (x9)", 94, ATT, "decode is SM-hungry -> RESERVE ~all SMs; prefill barely runs"),
    ("ssm-decode\nlayer (x45)", 54, SSM, "decode saturates ~54 SM -> RELEASE the other ~54 to prefill"),
]
yh = 0.62
for k, (lab, floor, dc, note) in enumerate(rows):
    y = 1 - k * 1.4
    ax.add_patch(Rectangle((0, y), floor, yh, color=dc, alpha=0.9))                       # decode reserved
    ax.add_patch(Rectangle((floor, y), SM - floor, yh, color=PRE, alpha=0.55, hatch="//"))  # prefill
    ax.text(floor / 2, y + yh / 2, f"decode\n{floor} SM", ha="center", va="center", color="white",
            fontsize=9, fontweight="bold")
    ax.text(floor + (SM - floor) / 2, y + yh / 2, f"prefill\n{SM-floor} SM", ha="center", va="center",
            color="#0a5a2a", fontsize=9, fontweight="bold")
    ax.text(-3, y + yh / 2, lab, ha="right", va="center", fontsize=9.5, fontweight="bold")
    ax.text(SM + 2, y + yh / 2, note, ha="left", va="center", fontsize=8.8)
ax.annotate("", xy=(54, 1.78), xytext=(94, 1.78),
            arrowprops=dict(arrowstyle="<->", color=PRE, lw=1.6))
ax.text(74, 1.9, "~40 SMs reclaimed for prefill\non every ssm layer", ha="center", color="#0a5a2a",
        fontsize=8.5, fontweight="bold")
ax.set_xlim(-26, SM + 40); ax.set_ylim(-0.55, 2.15)
ax.set_xticks([0, 27, 54, 81, 108]); ax.set_yticks([])
ax.set_xlabel("SMs on the GPU (108 total)")
for sp in ("top", "right", "left"): ax.spines[sp].set_visible(False)
ax.legend(handles=[Patch(color=ATT, label="attn-decode (reserved)"),
                   Patch(color=SSM, label="ssm-decode (reserved)"),
                   Patch(facecolor=PRE, alpha=0.55, hatch="//", label="prefill (gets the rest)")],
          loc="lower center", ncol=3, fontsize=8.5, frameon=False, bbox_to_anchor=(0.42, -0.32))

fig.suptitle("layer_aware reservation: protect SM-hungry attn-decode, release SM-light ssm layers to prefill",
             fontsize=12.5, fontweight="bold", y=1.0)
out = Path(_CHAR).parents[1] / "reports" / "figures"
p = out / "mechanism_schematic.png"
fig.savefig(p, dpi=130, bbox_inches="tight")
print(f"wrote {p}")
