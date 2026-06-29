"""vLLM calibration of the sim's fused policy, now across ALL zamba2 sizes + falcon.
(A) sim fused ITL vs vLLM measured p99 TPOT (saturated) -> overestimate grows with size (no-GQA).
(B) decode-cost SIZE-SCALING matches (sim decode_total ratio ~= vLLM TPOT ratio).
Output: reports/figures/vllm_calibration.png
"""
import os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_CHAR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
BLUE, GREEN, GRAY = "#1a73e8", "#0b8043", "#9aa0a6"

# model: (sim_fused_ITL, vLLM_p99_TPOT_sat, sim_thru, vLLM_thru, sim_decode_total)
D = {
    "z1.2b": (111, 67, 1121, 1413, 18.8),
    "z2.7b": (179, 109, 692, 870, 28.5),
    "z7b":   (403, 172, 312, 346, 49.3),
}
models = list(D)

fig, (axA, axB) = plt.subplots(1, 2, figsize=(13.5, 4.8))

# (A) sim fused ITL vs vLLM measured
x = np.arange(len(models)); w = 0.36
sim_itl = [D[m][0] for m in models]
vllm_itl = [D[m][1] for m in models]
axA.bar(x - w / 2, sim_itl, w, label="sim fused ITL", color=BLUE, edgecolor="k", linewidth=0.4)
axA.bar(x + w / 2, vllm_itl, w, label="vLLM measured p99 TPOT", color=GREEN, edgecolor="k", linewidth=0.4)
for i, m in enumerate(models):
    axA.text(i, max(sim_itl[i], vllm_itl[i]) + 8, f"{sim_itl[i]/vllm_itl[i]:.2f}x",
             ha="center", fontsize=9, fontweight="bold", color="#c5221f")
axA.set_xticks(x); axA.set_xticklabels(models)
axA.set_ylabel("decode ITL (ms)")
axA.set_title("(A) sim fused OVERESTIMATES vLLM -- grows with size\n"
              "(unfused per-layer sum, worst for no-GQA attn-decode)", fontsize=10.5, fontweight="bold")
axA.legend(fontsize=9)
axA.grid(axis="y", alpha=0.3)

# (B) size-scaling match (normalize to z1.2b)
zs = ["z1.2b", "z2.7b", "z7b"]
sim_dec = np.array([D[m][4] for m in zs]); sim_dec = sim_dec / sim_dec[0]
vllm_t = np.array([D[m][1] for m in zs]); vllm_t = vllm_t / vllm_t[0]
axB.plot(range(3), sim_dec, "-o", color=BLUE, lw=2, ms=8, label="sim decode_total")
axB.plot(range(3), vllm_t, "-s", color=GREEN, lw=2, ms=8, label="vLLM TPOT (sat)")
# label only the endpoint (well-separated) to avoid overlap; lines visibly coincide elsewhere
axB.text(2.05, sim_dec[2], f"{sim_dec[2]:.2f}", ha="left", va="bottom", color=BLUE, fontsize=10, fontweight="bold")
axB.text(2.05, vllm_t[2], f"{vllm_t[2]:.2f}", ha="left", va="top", color=GREEN, fontsize=10, fontweight="bold")
axB.set_xlim(-0.2, 2.5)
axB.set_xticks(range(3)); axB.set_xticklabels(zs)
axB.set_ylabel("decode cost, normalized to 1.2B")
axB.set_title("(B) decode-cost SIZE-SCALING matches\n(relative trend valid even though absolute is off)",
              fontsize=10.5, fontweight="bold")
axB.legend(fontsize=9, loc="upper left")
axB.grid(alpha=0.3)

fig.suptitle("vLLM calibration of the sim (zamba2 1.2/2.7/7B): absolute OFF (1.6-2.3x), "
             "relative scaling VALID", fontsize=12, fontweight="bold", y=1.03)
out = Path(_CHAR).parents[1] / "reports" / "figures"
out.mkdir(parents=True, exist_ok=True)
p = out / "vllm_calibration.png"
fig.savefig(p, dpi=130, bbox_inches="tight")
print(f"wrote {p}")
