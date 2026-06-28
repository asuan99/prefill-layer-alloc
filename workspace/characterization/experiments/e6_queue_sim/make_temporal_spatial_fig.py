"""Temporal (Zamba2) vs Spatial (Falcon-H1) hybrid decode structure.
Shows WHY layer_aware applies to temporal but not spatial:
  temporal = few EXPENSIVE (no-GQA) attn-decode layers + many cheap ssm layers (separable)
  spatial  = every layer = CHEAP (GQA) attn-decode + ssm, in parallel (not separable)
Output: reports/figures/temporal_vs_spatial.png
"""
import os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

_CHAR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# per-layer decode @b8 (ms): (model, L, attn_positions_or_all, decode_attn, decode_ssm, kind)
ATT, SSM = "#e8710a", "#1a73e8"
fig, axes = plt.subplots(1, 2, figsize=(14, 4.2))

# --- LEFT: zamba2_2.7b (temporal) ---
ax = axes[0]
L = 54; na = 9; da, ds = 0.591, 0.516
attn_pos = set(np.linspace(2, L - 3, na).round().astype(int).tolist())
for i in range(L):
    if i in attn_pos:
        ax.bar(i, da, width=0.9, color=ATT)            # tall expensive attn-decode
    else:
        ax.bar(i, ds, width=0.9, color=SSM)            # short cheap ssm-decode
ax.set_title("(A) TEMPORAL hybrid — zamba2_2.7b (54 layers)\n"
             "attn & ssm in SEPARATE layers; attn-decode EXPENSIVE (no-GQA, 0.59ms)",
             fontsize=10, fontweight="bold")
ax.set_xlabel("layer index (depth)")
ax.set_ylabel("per-layer decode @b8 (ms)")
ax.set_ylim(0, 1.05)
ax.text(0.98, 0.95, "→ reserve SMs ONLY on the 9 tall attn layers,\n"
        "   release on the 45 cheap ssm layers = layer_aware (1.4–2.0×)",
        transform=ax.transAxes, ha="right", va="top", fontsize=8.5, color="#0b6")

# --- RIGHT: falcon_h1_3b (spatial) ---
ax = axes[1]
L = 32; da, ds = 0.056, 0.515
for i in range(L):
    ax.bar(i, ds, width=0.9, color=SSM)                 # ssm part
    ax.bar(i, da, width=0.9, bottom=ds, color=ATT)      # tiny attn part on top
ax.set_title("(B) SPATIAL hybrid — falcon_h1_3b (32 layers)\n"
             "attn+ssm PARALLEL in EVERY layer; attn-decode CHEAP (GQA 5×, 0.056ms)",
             fontsize=10, fontweight="bold")
ax.set_xlabel("layer index (depth)")
ax.set_ylabel("per-layer decode @b8 (ms)")
ax.set_ylim(0, 1.05)
ax.text(0.98, 0.95, "→ every layer is attn+ssm (can't separate)\n"
        "   AND attn-decode is only ~10% (GQA) ⇒ nothing to protect\n"
        "   = layer_aware N/A",
        transform=ax.transAxes, ha="right", va="top", fontsize=8.5, color="#b00")

leg = [Patch(color=ATT, label="attn-decode (memory-bound, KV read)"),
       Patch(color=SSM, label="ssm-decode (O(1) state, cheap)")]
fig.legend(handles=leg, loc="lower center", ncol=2, fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.04))
fig.suptitle("Why layer-type-aware reservation works for TEMPORAL but not SPATIAL hybrids",
             fontsize=12.5, fontweight="bold", y=1.04)
out = Path(_CHAR).parents[1] / "reports" / "figures"
out.mkdir(parents=True, exist_ok=True)
p = out / "temporal_vs_spatial.png"
fig.savefig(p, dpi=130, bbox_inches="tight")
print(f"wrote {p}")
