"""F6. coordinated_implementations_vs_agnostic (decode-side) — 발표 최중요 그림
coordinated per-type layer-aware 3종 구현(R0d 124 → v4 95 → (a) 85 ms)이 전부
agnostic/tuned-uniform 기준선(42-43ms) 위 → 작동하는 best-case 구현으로 (D) 실증 반증.
기준선 아래로 넘어오는 막대가 하나도 없어야 메시지가 산다.
값 전부 measured @in3600/o32 rate2:
  R0d job 835918 · (a) job 837520 · v4 job 837718 · agnostic r0c job 835303."""
import matplotlib.pyplot as plt
from _deck import EP, AGN, LA, tpot_at, save

RATE = 2
la = [
    ("R0d\ncoordinated per-type",       tpot_at(f"{EP}/r0d_coord_la/_rows_m16a74_in3600o32_rep1_835918.csv", RATE), "835918"),
    ("v4\nfaithful multi-window",        tpot_at(f"{EP}/v4_multiwin/_rows_m16a74_in3600o32_rep1_837718.csv", RATE),   "837718"),
    ("(a)\nsubstrate-isolated (best)",   tpot_at(f"{EP}/a_substrate/_rows_m16a74_in3600o32_rep1_837520.csv", RATE),   "837520"),
]
agn = tpot_at(f"{EP}/r0c/_rows_agn_rep1_835303.csv", RATE)   # 42.51
la.sort(key=lambda t: -t[1])   # 최악 아래, best (a) 위

fig, ax = plt.subplots(figsize=(9.8, 4.2))
ys = range(len(la)); vals = [v for _, v, _ in la]
ax.barh(list(ys), vals, color=LA, edgecolor="white", height=0.6, zorder=3)
ax.set_yticks(list(ys)); ax.set_yticklabels([n for n, _, _ in la], fontsize=9.5)
ax.axvline(agn, color=AGN, lw=3.0, zorder=4)
ax.text(agn + max(vals) * 0.01, len(la) - 1 + 0.55,
        f"agnostic / tuned-uniform floor  {agn:.0f} ms", color=AGN, fontsize=10,
        fontweight="bold", va="bottom")
for i, (n, v, job) in enumerate(la):
    ax.text(v + max(vals) * 0.012, i, f"{v:.0f} ms   ({v/agn:.1f}× floor)",
            va="center", fontsize=10, fontweight="bold")
ax.set_xlim(0, max(vals) * 1.2); ax.set_ylim(-0.6, len(la) - 1 + 0.95)
ax.set_xlabel("decode TPOT p50 (ms)  —  in3600/o32, rate 2  [lower = better]", fontsize=10)
ax.set_title("F6 · all three coordinated builds stay ABOVE the step-level floor\n"
             "(D) proven with a working best-case implementation, not argued — layer-aware refuted",
             fontsize=11.5, fontweight="bold")
ax.grid(axis="x", alpha=0.3, zorder=0)
fig.tight_layout()
save(fig, "f6_coordinated_implementations_vs_agnostic")
