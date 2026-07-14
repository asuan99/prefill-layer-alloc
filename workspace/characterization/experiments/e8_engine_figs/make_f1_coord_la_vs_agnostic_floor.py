"""F1. coord_la_vs_agnostic_floor
4개 coordinated layer-aware(decode-type) 구현의 TPOT p50가 전부 step-level 기준선
(agnostic/tuned-uniform ~42-46ms) 위에 있다 = "실증적 반증". 기준선 아래로 넘어오는 막대 없음.
regime: in3600/o32(prefill-bound), load: rate2(첫 stressed point, 모두 측정 가능).
모든 값 measured (workspace/engine-port/results/ 실측 rows CSV에서 직접 파싱)."""
import matplotlib.pyplot as plt
from _common import EP, LA, AGN, TUNED, tpot_at, save

RATE = 2  # 첫 부하점: 네 구현 모두 이 지점에서 붕괴가 드러남

# (막대) layer-aware 구현 4종 — 전부 measured tpot_p50 @ rate2, in3600/o32
la = [
    ("inefficient_v1\n(green-ctx, uncoordinated)", tpot_at(f"{EP}/r0d_coord_la/inefficient_v1/_rows_m16a74_in3600o32_rep1_835906.csv", RATE), "835906"),
    ("R0d\n(coordinated per-type)",                 tpot_at(f"{EP}/r0d_coord_la/_rows_m16a74_in3600o32_rep1_835918.csv", RATE),               "835918"),
    ("v4\n(faithful multi-window)",                 tpot_at(f"{EP}/v4_multiwin/_rows_m16a74_in3600o32_rep1_837718.csv", RATE),                 "837718"),
    ("(a)\n(substrate-isolated, best)",             tpot_at(f"{EP}/a_substrate/_rows_m16a74_in3600o32_rep1_837520.csv", RATE),                 "837520"),
]
la.sort(key=lambda t: -t[1])  # 최악을 아래, 그나마 나은 (a)를 맨 위로 (barh는 y=0이 바닥)

# (기준선) step-level: agnostic auto + tuned-uniform(d16)
agn  = tpot_at(f"{EP}/r0c/_rows_agn_rep1_835303.csv", RATE)   # 42.51
tuned = tpot_at(f"{EP}/r0c/_rows_16_rep1_835300.csv", RATE)   # 45.70 (prefill92/decode16)
floor = max(agn, tuned)

fig, ax = plt.subplots(figsize=(10.5, 4.6))
ys = range(len(la))
vals = [v for _, v, _ in la]
bars = ax.barh(list(ys), vals, color=LA, edgecolor="white", height=0.62, zorder=3)
ax.set_yticks(list(ys)); ax.set_yticklabels([n for n, _, _ in la], fontsize=9.5)

# step-level 기준선(band): agnostic..tuned
ax.axvspan(agn, tuned, color=AGN, alpha=0.10, zorder=1)
ax.axvline(agn, color=AGN, lw=2.2, zorder=2)
ax.set_ylim(-0.6, len(la) - 1 + 0.95)  # 상단 여백 = floor 라벨 자리
ax.text(agn * 1.15, len(la) - 1 + 0.55,
        f"step-level floor\nagnostic {agn:.0f} · tuned {tuned:.0f} ms",
        color=AGN, fontsize=9.5, fontweight="bold", va="bottom", ha="left")

for b, (name, v, job) in zip(bars, la):
    ax.text(v + max(vals) * 0.01, b.get_y() + b.get_height() / 2,
            f"{v:.0f} ms   ({v/agn:.1f}× floor)", va="center", fontsize=9.5, fontweight="bold")

ax.set_xlim(0, max(vals) * 1.18)
ax.set_xlabel("decode TPOT p50 (ms)  —  in3600/o32, request rate 2  [lower = better]", fontsize=10)
ax.set_title("F1 · every coordinated layer-aware implementation stays ABOVE the step-level floor\n"
             "(4 builds, all measured — none crosses the agnostic line → layer-aware refuted)",
             fontsize=11.5, fontweight="bold")
ax.grid(axis="x", alpha=0.3, zorder=0)
fig.tight_layout()
save(fig, "f1_coord_la_vs_agnostic_floor")
