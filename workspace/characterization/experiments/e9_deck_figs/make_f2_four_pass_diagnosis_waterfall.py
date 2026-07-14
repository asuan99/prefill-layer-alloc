"""F2. four_pass_diagnosis_waterfall (decode-side)
layer-aware 붕괴(agnostic 33.5 → binary la 72.0, +115%)의 원인을 네 pass로 좁힘.
전부 sm_policy_report.html §07 "four passes" 서술의 measured 진술:
 (A) per-span switching overhead ~0        (release-floor 96 → agnostic과 동일)
 (B) compute-starvation of released layers ~8% (in isolation)
 (C) prefill contention  = 실제 붕괴의 대부분 (+115%는 prefill 있을 때만 나타남)
 (D) granularity         = 근본 이유 (per-layer 창 ~0.8ms < concurrent prefill kernel;
     커널은 mid-flight SM 재분배 불가 → C를 조율로 없앨 수 없음)."""
import matplotlib.pyplot as plt
from _deck import AGN, LA, FUSED, TUNED, save

BASE = 33.5; TOTAL = 72.0
A = 0.0
B = round(BASE * 0.08, 1)           # ~8% in isolation
C = round(TOTAL - BASE - A - B, 1)  # 나머지 = prefill contention (표면 붕괴)
steps = [("agnostic\nbaseline", BASE, AGN, "measured"),
         ("(A) switching\noverhead", A, "0.6", "~0 · measured"),
         ("(B) released-layer\nstarvation", B, TUNED, "~8% · measured"),
         ("(C) prefill\ncontention", C, FUSED, "the real collapse · measured"),
         ("engine binary\n(total)", TOTAL, LA, "measured")]

fig, ax = plt.subplots(figsize=(10.5, 5.2))
x = range(len(steps)); W = 0.62; HALF = W / 2
last = len(steps) - 1
# 각 막대의 bottom/top: baseline·total은 0부터, 증분은 누적 위에 floating
bottoms = [0.0, BASE, BASE + A, BASE + A + B, 0.0]
tops    = [BASE, BASE + A, BASE + A + B, BASE + A + B + C, TOTAL]
LABEL_DY = 1.9
for i, (lab, val, col, tag) in enumerate(steps):
    b, t = bottoms[i], tops[i]
    ax.bar(i, t - b, W, bottom=b, color=col, edgecolor="white", zorder=3)
    if abs(t - b) < 1e-9:                    # (A) 0-기여: 캡 마커로 표시
        ax.plot([i - HALF, i + HALF], [t, t], color=col, lw=3.5, solid_capstyle="butt", zorder=4)
    lbl = f"{t:.1f}" if i in (0, last) else f"+{val:.1f}"
    ax.text(i, t + LABEL_DY, lbl, ha="center", fontsize=10, fontweight="bold")
    ax.text(i, -6.5, tag, ha="center", fontsize=7.8, color="0.35")
    if i < last:                             # 다음 막대와 공유 레벨(=이 막대 top)을 잇는 전폭 점선
        ax.plot([i + HALF, (i + 1) - HALF], [t, t], color="0.55", lw=1.1, ls="--", zorder=2)

# (D) 근본 원인 = C 위에 브라켓 주석 (좌상단, C 막대를 가리킴)
ax.annotate("(D) granularity — the fundamental one:\nper-layer window ~0.8 ms < concurrent prefill kernel\n"
            "→ freed SM can't be captured → (C) is un-coordinatable",
            xy=(3, BASE + B + C / 2), xytext=(0.55, TOTAL * 1.34),
            fontsize=8.8, color=FUSED, fontweight="bold", ha="left",
            arrowprops=dict(arrowstyle="->", color=FUSED, lw=1.5))

ax.set_xticks(list(x)); ax.set_xticklabels([s[0] for s in steps], fontsize=9)
ax.set_ylim(-9, TOTAL * 1.5); ax.set_ylabel("decode TPOT (ms)", fontsize=10)
ax.set_title("F2 · four passes to the cause: A(~0) → B(~8%) → C(prefill contention) → D(granularity)\n"
             "each pass overturned the previous guess (all measured, sm_policy_report §07)",
             fontsize=11, fontweight="bold")
ax.grid(axis="y", alpha=0.3, zorder=0)
fig.tight_layout()
save(fig, "f2_four_pass_diagnosis_waterfall")
