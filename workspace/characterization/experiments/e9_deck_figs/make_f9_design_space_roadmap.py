"""F9. design_space_roadmap (개념도 — 수치/job/goodput 없음)
이번 덱이 닫은 범위(decode-side layer-type)와 다루지 않은 범위(prefill-side layer-type,
workload-regime phase-size)를 한 장에 배치. "다음에 뭘 볼 것인가" 지도이지 결과 보고 아님."""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from _deck import AGN, LA, FUSED, TUNED, save

fig, ax = plt.subplots(figsize=(10.5, 5.6))
ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")

def box(x, y, w, h, title, body, ec, fc, closed=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08,rounding_size=0.18",
                                ec=ec, fc=fc, lw=2.4 if closed else 1.6,
                                ls="-" if closed else "--", alpha=0.95, zorder=3))
    ax.text(x + w / 2, y + h - 0.5, title, ha="center", va="top", fontsize=11, fontweight="bold", color=ec)
    ax.text(x + w / 2, y + h - 1.2, body, ha="center", va="top", fontsize=9, color="0.15")

ax.text(5, 9.5, "Design space — what this deck closes vs what remains", ha="center",
        fontsize=13, fontweight="bold")
# 두 축 라벨
ax.text(2.7, 8.3, "split BY  layer-type", ha="center", fontsize=10.5, fontweight="bold", color="0.35")
ax.text(7.5, 8.3, "split BY  phase-size", ha="center", fontsize=10.5, fontweight="bold", color="0.35")

# 좌상: decode-side layer-type — CLOSED (이번 덱)
box(0.4, 4.6, 4.6, 3.3, "decode-side layer-type",
    "THIS DECK — closed.\nsim → binary → graduated → coordinated\nall lose to step-level (agnostic).\n"
    "→ granularity (D) is fundamental.", ec=FUSED, fc="#f7e7e3", closed=True)
# 좌하: prefill-side layer-type — 별도 트랙
box(0.4, 0.7, 4.6, 3.1, "prefill-side layer-type",
    "separate track — under verification.\n(not shown in this deck)", ec=LA, fc="#f7efe0")
# 우: workload-regime phase-size — 별도 트랙
box(5.4, 2.4, 4.2, 5.5, "workload-regime  phase-size",
    "separate track — in progress.\ntuned-uniform · SLO-aware\n(dynamic split by measured latency)\n"
    "(not shown in this deck)", ec=AGN, fc="#e3f1ef")

ax.text(2.7, 4.35, "✓ closed here", ha="center", fontsize=9.5, color=FUSED, fontweight="bold")
ax.text(2.7, 0.45, "→ future", ha="center", fontsize=9.5, color=LA, fontweight="bold")
ax.text(7.5, 2.15, "→ future", ha="center", fontsize=9.5, color=AGN, fontweight="bold")
fig.tight_layout()
save(fig, "f9_design_space_roadmap")
