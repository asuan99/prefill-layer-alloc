"""F3. decode_sm_sensitivity_asymmetry (decode-side)
decode 단계 attn vs mamba per-layer latency의 SM별 곡선 = ~9× 비대칭의 근거.
SM=108: attn 3.14ms / mamba 0.36ms (8.7×), SM=8까지 벌어짐(35×).
값 전부 measured: results/r0c/knee_result_835571.txt (108/44/24/16/8 전 지점)."""
import matplotlib.pyplot as plt
from _deck import EP, ATT, SSM, read_knee, save

k = read_knee(f"{EP}/r0c/knee_result_835571.txt")
i108 = k["sm"].index(108); i8 = k["sm"].index(8)
r108 = k["attn"][i108] / k["mamba"][i108]
r8 = k["attn"][i8] / k["mamba"][i8]

fig, ax = plt.subplots(figsize=(7.6, 5.0))
ax.plot(k["sm"], k["attn"],  "o-", color=ATT, lw=2.6, ms=8, label=f"attn-decode  (×{k['n_attn']} layers)")
ax.plot(k["sm"], k["mamba"], "s-", color=SSM, lw=2.6, ms=8, label=f"mamba-decode (×{k['n_mamba']} layers)")
# 108 지점 9× 강조
ax.annotate(f"SM=108:  attn {k['attn'][i108]:.2f} / mamba {k['mamba'][i108]:.2f}\n= {r108:.1f}× — attn SM-hungry, mamba SM-insensitive",
            xy=(108, (k["attn"][i108] * k["mamba"][i108]) ** 0.5), xytext=(40, 1.5),
            fontsize=9, fontweight="bold", ha="left",
            arrowprops=dict(arrowstyle="->", color="0.3", lw=1.4),
            bbox=dict(boxstyle="round", fc="white", ec="0.6"))
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xticks(k["sm"]); ax.set_xticklabels([str(s) for s in k["sm"]]); ax.invert_xaxis()
ax.set_xlabel("SMs given to the decode kernel  (108 → 8)", fontsize=10)
ax.set_ylabel("per-layer decode latency (ms, log)", fontsize=10)
ax.set_title(f"F3 · decode SM-sensitivity is asymmetric: attn/mamba {r108:.1f}× → {r8:.0f}×\n"
             "mamba is SM-insensitive (releasable) — the lever the policy tries to pull",
             fontsize=11, fontweight="bold")
ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=10, loc="center right")
fig.tight_layout()
save(fig, "f3_decode_sm_sensitivity_asymmetry")
