"""Prefill SM-sensitivity curves (job 797524): is 7B prefill SM-insensitive? -> NO.
Solo prefill latency vs SM partition for 1.2b/2.7b/7b, ssm & attn.
Shows the curves overlap (size-invariant) and the 108->54 region is flat (why the old
108-54-only measurement wrongly concluded '7B insensitive'). Output: reports/figures/prefill_sm_sensitivity.png
"""
import os, sys
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_here = os.path.dirname(os.path.abspath(__file__))
_CHAR = os.path.abspath(os.path.join(_here, "..", ".."))
RES = Path(_CHAR) / "results_v2" / "e7_prefill_sm"
MODELS = ["zamba2_1.2b", "zamba2_2.7b", "zamba2_7b"]
COL = {"zamba2_1.2b": "#34a853", "zamba2_2.7b": "#1a73e8", "zamba2_7b": "#ea4335"}

fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
for ax, lt in zip(axes, ["ssm", "attn"]):
    for m in MODELS:
        f = RES / f"prefill_sm_{m}_a100_sxm4_80gb.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f)
        g = df[(df.layer_type == lt) & (df.status == "ok")].sort_values("n_sm_actual")
        if not len(g):
            continue
        ax.plot(g.n_sm_actual, g.ratio_vs_full, "-o", color=COL[m], lw=2, ms=5,
                label=f"{m.replace('zamba2_','')}  (108->14: {g.ratio_vs_full.max():.1f}x)")
    ax.axvspan(54, 108, color="gray", alpha=0.08)
    ax.text(0.72, 0.86, "108->54 region\n(old measurement: flat)",
            transform=ax.transAxes, ha="center", fontsize=8, color="gray")
    ax.axhline(1.0, color="k", ls="--", lw=0.8, alpha=0.5)
    ax.set_xlabel("SM given to prefill (partition size)")
    ax.set_ylabel("prefill latency / latency@108SM  (sensitivity)")
    ax.set_title(f"({'A' if lt=='ssm' else 'B'}) {lt}-prefill SM-sensitivity", fontweight="bold")
    ax.legend(fontsize=9, title="model (sensitivity)")
    ax.grid(alpha=0.3)
    ax.invert_xaxis()   # fewer SM on the right = more starved
fig.suptitle("Prefill SM-sensitivity is ~size-invariant -- 7B is NOT insensitive (refutes old premise(2))\n"
             "solo prefill (batch=8, tokens=256), Green-Context N-SM partition",
             fontsize=12, fontweight="bold", y=1.06)
out = Path(_CHAR).parents[1] / "reports" / "figures"
out.mkdir(parents=True, exist_ok=True)
p = out / "prefill_sm_sensitivity.png"
fig.savefig(p, dpi=130, bbox_inches="tight")
print(f"wrote {p}")
