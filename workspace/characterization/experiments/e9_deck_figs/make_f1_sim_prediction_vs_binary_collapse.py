"""F1. sim_prediction_vs_binary_collapse (decode-side)
sim은 0-비용 전환을 가정해 layer-aware가 agnostic을 매칭/상회한다고 예측했으나,
실엔진 binary(uncoordinated, sensitive→full / insensitive→floor16)는 TPOT +115% 붕괴.
- [derived] sim 예측: 0-cost 전환 가정 → net gain ≈ 0 → ≈ agnostic baseline
  (p1_4_layer_aware_평가_kr.md §"layer-aware의 순이득 ≈ 0")
- [measured] engine binary: agnostic 33.5 → la 72.0 ms (r6, +115%)
  (triage/notes.md P1.7g, Granite floor16; 결정 control job 834486)."""
import matplotlib.pyplot as plt
from _deck import AGN, LA, FUSED, save

BASELINE = 33.5   # [measured] agnostic TPOT (r6)
PREDICT  = 33.5   # [derived] sim 0-cost → ≈ baseline (net gain ≈ 0)
MEASURED = 72.0   # [measured] engine binary layer-aware (r6)
inc = (MEASURED - BASELINE) / BASELINE * 100

fig, ax = plt.subplots(figsize=(6.6, 5.0))
xs = [0, 1]
bars = ax.bar(xs, [PREDICT, MEASURED], width=0.6,
              color=[AGN, LA], edgecolor="white", zorder=3)
ax.axhline(BASELINE, ls="--", lw=1.6, color="0.35", zorder=2)
ax.text(1.48, BASELINE, f" agnostic baseline {BASELINE:.0f} ms", va="center",
        ha="left", fontsize=9, color="0.35")

ax.set_xticks(xs)
ax.set_xticklabels(["sim prediction\n(0-cost switching)\n[derived]",
                    "engine binary\n(uncoordinated)\n[measured]"], fontsize=10)
for b, v, tag in zip(bars, [PREDICT, MEASURED], ["≈ baseline\n(net gain ≈ 0)", f"+{inc:.0f}%"]):
    ax.text(b.get_x() + b.get_width() / 2, v + 2, f"{v:.0f} ms", ha="center", fontsize=11, fontweight="bold")
    ax.text(b.get_x() + b.get_width() / 2, v / 2, tag, ha="center", va="center",
            fontsize=10, color="white", fontweight="bold")
# +115% collapse arrow
ax.annotate("", xy=(1, MEASURED), xytext=(1, PREDICT),
            arrowprops=dict(arrowstyle="->", color=FUSED, lw=2.2))
ax.set_ylim(0, MEASURED * 1.18)
ax.set_ylabel("decode TPOT (ms)", fontsize=10)
ax.set_title("F1 · sim assumed a free lunch; the engine's binary build collapsed +115%\n"
             "(the case sim called the big win became the worst)", fontsize=11, fontweight="bold")
ax.grid(axis="y", alpha=0.3, zorder=0)
fig.tight_layout()
save(fig, "f1_sim_prediction_vs_binary_collapse")
