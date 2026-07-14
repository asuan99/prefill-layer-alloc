"""F8. switching_overhead_accumulation (개념도)
v4는 decode 한 스텝(54층) 안에서 prefill과 ~19번 파티션을 주고받고, 전환마다 비용이 붙어
freed-SM 이득을 상쇄. 누적 전환비용이 이득을 넘는 교차점 시각화(conceptual).
서술 근거: session_handoff_2026-07-07.md R0d "decode 19윈도우 파편화→sync 직렬화·핀·overlap감소"."""
import numpy as np
import matplotlib.pyplot as plt
from _deck import LA, AGN, FUSED, save

N = 19                              # v4가 스텝당 주고받는 파티션 전환 수
sw = np.arange(0, N + 1)
benefit = np.full_like(sw, 1.0, dtype=float) * 6.0       # freed-SM 이득 (개념: 전환수 무관 상한)
cost = 0.55 * sw                                          # 전환당 누적 비용 (개념 선형)
cross = benefit[0] / 0.55                                 # 교차점

fig, ax = plt.subplots(figsize=(8.6, 4.9))
ax.bar(sw, cost, width=0.7, color=FUSED, alpha=0.85, edgecolor="white", zorder=3,
       label="cumulative switch cost (per-window drain/sync/pin)")
ax.axhline(benefit[0], color=AGN, lw=2.4, ls="--", zorder=4,
           label="freed-SM benefit (bounded, step-level)")
ax.axvline(cross, color="0.3", lw=1.4, ls=":", zorder=5)
ax.annotate(f"crossover ≈ {cross:.0f} switches\nbeyond here cost > benefit",
            xy=(cross, benefit[0]), xytext=(cross + 1.2, benefit[0] * 1.35),
            fontsize=9.5, fontweight="bold", color="0.2",
            arrowprops=dict(arrowstyle="->", color="0.3", lw=1.4))
ax.annotate(f"v4 pays {N} switches / step\n(54-layer decode fragmented into ~19 windows)",
            xy=(N, cost[-1]), xytext=(1.0, cost[-1] * 0.92),
            fontsize=9.5, fontweight="bold", color=FUSED, ha="left", va="top",
            arrowprops=dict(arrowstyle="->", color=FUSED, lw=1.4))
ax.set_xticks(sw[::2]); ax.set_xlim(-0.6, N + 0.6); ax.set_ylim(0, cost[-1] * 1.12)
ax.set_xlabel("partition switches within one decode step", fontsize=10)
ax.set_ylabel("relative cost / benefit (conceptual)", fontsize=10)
ax.set_title("F8 · per-window switching cost accumulates past the freed-SM benefit\n"
             "(sub-step re-partitioning loses to holding the split for a whole step — conceptual)",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=9, loc="upper left"); ax.grid(axis="y", alpha=0.3, zorder=0)
fig.tight_layout()
save(fig, "f8_switching_overhead_accumulation")
